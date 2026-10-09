"""The episode-group miner's production entry point (phase 2, plan 2C): the per-window loop and its wiring.

Each poll reads the RL validator's ``/miner-state``: the window, its randomness, the announcement
(``service_policy``: the order and the checkpoint revision), the window's end (``submission_deadline_at``)
and, per environment, the prompt range, the cooldown and whether it accepts groups. It picks
(environment, task) among the configured signed-episode environments (never a task in cooldown, nor one
this miner already took in the window) and runs at most ``groups_in_flight`` (<= 2, the validator's
per-operator cap) ``EpisodeGroupMiner.mine_task`` at once. When the announced checkpoint (revision) or
order (contract sha) changes, no new group starts, the groups in flight are drained (bounded by
``drain_s``, then cancelled), the stack is closed and the process exits with ``EXIT_CHECKPOINT_CHANGED``
(75): the miner runs under a supervisor that restarts it (systemd ``Restart=always``, a docker restart
policy), and the new process loads the new checkpoint and re-checks the order's env package pin. vLLM
and the proof model are never reloaded in-process.

Harness secrets (API keys the episode harness needs) belong in ``--harness-env-file`` (or the
supervisor's environment file), never on the command line, where ``ps`` shows them.

The stack (``build_episode_stack``): one ``ForcedDraws`` shared by the miner and the forced
``GenerateEngine(VllmTurnCore(...))`` served on loopback; the renderer of the policy's tools; the engine
caps checked against every configured environment's episode policy BEFORE anything is served (refusal to
start); ``HttpRlEpisodes`` over the validator's ``/rl`` routes; ``RlSignedEpisodeRunner`` per precommit;
``hf_prover`` over the HF model at the announced revision (same GPU as vLLM: ``gpu_memory_utilization``
is vLLM's share); ``submit_batch_v2`` with the wallet (a new envelope per send) and the verdict poll.

The task prompt is the validator's: the registry environment's task source, ``source.prompt(index)``
(the same loader and code path the validator's episode admission uses), with the installed env package
checked against the order's pin."""
from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import math
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT

logger = logging.getLogger(__name__)

STATE_POLL_S = 2.0
PICK_DRAWS = 64
EXIT_CHECKPOINT_CHANGED = 75
"""The process exit code on a new checkpoint or order: "restart me" (EX_TEMPFAIL) for the supervisor."""
DRAIN_S = 600.0
"""How long the groups in flight may finish after a checkpoint change before they are cancelled."""
OUTCOMES_KEPT = 256
BUILD_BACKOFF_S = 5.0
BUILD_BACKOFF_MAX_S = 300.0
SECONDS_PER_WAVE = 300.0
"""Default time one wave of live sessions (an episode played to its end) is expected to take."""


class CheckpointChanged(Exception):
    """The announced checkpoint or order changed: the process must restart (``EXIT_CHECKPOINT_CHANGED``)."""
    exit_code = EXIT_CHECKPOINT_CHANGED


@dataclass(frozen=True)
class EpisodeWindow:
    """What one ``/miner-state`` says about the open window."""
    window: int
    randomness: str
    announcement: dict
    open_until: float | None
    repo: str
    revision: str
    cooldowns: Mapping[str, frozenset[int]] = field(default_factory=dict)
    ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    accepting: Mapping[str, bool] = field(default_factory=dict)
    contract_sha: str = ""
    opened_at: float | None = None

    @property
    def key(self) -> tuple[str, str]:
        """What the stack is built for: a change of either needs a restart."""
        return self.revision, self.contract_sha


def episode_window(state) -> EpisodeWindow | None:
    """The open window of a ``MinerState``, or None (not open, no randomness or no announcement yet)."""
    from reliquary.protocol.service_contract import ServiceContract
    from reliquary.protocol.submission import WindowState

    if state is None or state.state != WindowState.OPEN or not state.randomness:
        return None
    policy = getattr(state, "service_policy", None)
    if policy is None:
        return None
    announcement = policy.model_dump() if hasattr(policy, "model_dump") else dict(policy)
    checkpoint = announcement["checkpoint"]
    environments = state.environments or {}
    return EpisodeWindow(
        window=int(state.window_n), randomness=state.randomness, announcement=announcement,
        open_until=(None if state.submission_deadline_at is None else float(state.submission_deadline_at)),
        repo=str(checkpoint["repo"]), revision=str(checkpoint["revision"]),
        cooldowns={name: frozenset(env.cooldown_prompts()) for name, env in environments.items()},
        ranges={name: (int(env.prompt_range[0]), int(env.prompt_range[1])) for name, env in environments.items()},
        accepting={name: bool(env.accepting_submissions) for name, env in environments.items()},
        contract_sha=ServiceContract.from_dict(announcement["contract"]).sha256,
        opened_at=(None if state.window_opened_at is None else float(state.window_opened_at)))


def pick_episode_task(view: EpisodeWindow, *, environments: Sequence[str], sizes: Mapping[str, int],
                      taken: Collection[tuple[str, int]], rng: random.Random) -> tuple[str, int] | None:
    """A random (environment, task) of this window: an environment that accepts groups, a task of its
    announced prompt range (and of its dataset), not in its cooldown and not ``taken``; None if none."""
    names = [name for name in environments if view.accepting.get(name) and name in view.ranges]
    rng.shuffle(names)
    for name in names:
        low, high = view.ranges[name]
        high = min(high, int(sizes.get(name, 0)))
        if high <= low:
            continue
        cooled = view.cooldowns.get(name, frozenset())

        def free(index: int) -> bool:
            return index not in cooled and (name, index) not in taken

        for _ in range(PICK_DRAWS):
            index = rng.randrange(low, high)
            if free(index):
                return name, index
        eligible = [index for index in range(low, high) if free(index)]
        if eligible:
            return name, rng.choice(eligible)
    return None


class EpisodeTaskPrompts:
    """``task_prompt(environment, task)``: the registry environment's task source, exactly as the
    validator loads it (``load_environment(name).source.prompt(index)``); never re-rendered."""

    def __init__(self, environments: Sequence[str], *, load: Callable[[str], Any] | None = None) -> None:
        from reliquary.environment.registry import get_environment_spec

        if load is None:
            from reliquary.environment import load_environment as load
        self._environments = {}
        for name in environments:
            if get_environment_spec(name).interaction_mode != "signed_episode":
                raise ValueError(f"{name} is not a signed-episode environment")
            self._environments[name] = load(name)

    def __call__(self, environment: str, task_index: int) -> str:
        return self._environments[environment].source.prompt(int(task_index))

    def size(self, environment: str) -> int:
        return len(self._environments[environment].source)


def parse_episode_environments(value: str) -> tuple[str, ...]:
    """``--episode-envs``: distinct registered signed-episode environments (ValueError otherwise)."""
    from reliquary.environment.registry import get_environment_spec

    names = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    if not names:
        raise ValueError("--episode-envs names no environment")
    for name in names:
        if get_environment_spec(name).interaction_mode != "signed_episode":
            raise ValueError(f"{name} is not a signed-episode environment (the legacy `mine` mines it)")
    return names


def parse_harness_env(values: Sequence[str], *, env_file: str | None = None) -> dict[str, str]:
    """``--harness-env-file`` (KEY=VALUE lines, ``#`` comments; where secrets belong) then
    ``--harness-env KEY=VALUE`` (repeatable; it wins over the file)."""
    parsed = {}
    if env_file:
        with open(env_file, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        for line in lines:
            if not line or line.startswith("#"):
                continue
            key, sep, val = line.removeprefix("export ").partition("=")
            if not sep or not key.strip():
                raise ValueError(f"{env_file}: KEY=VALUE lines only")
            parsed[key.strip()] = val.strip()
    for item in values:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--harness-env takes KEY=VALUE, not {item!r}")
        parsed[key.strip()] = val
    return parsed


def short_window_warning(view: EpisodeWindow, *, environments: Sequence[str], max_live: int | None,
                         groups_in_flight: int, seconds_per_wave: float) -> str | None:
    """Why the announced collection window looks too short for ``groups_in_flight`` groups of 2M seeds at
    ``max_live`` live sessions per group (groups run side by side: ceil(2M / max_live) waves of
    ``seconds_per_wave`` each), or None."""
    from reliquary.protocol.seed_pool import pool_from_service_policy

    if view.open_until is None or view.opened_at is None or seconds_per_wave <= 0:
        return None
    window_s = view.open_until - view.opened_at
    for name in environments:
        pool = pool_from_service_policy(view.announcement, environment=name, prompt_idx=0,
                                        checkpoint_hash=view.revision)
        if pool is None:
            continue
        live = int(max_live or pool.pool_seeds)
        needed = math.ceil(pool.pool_seeds / live) * float(seconds_per_wave)
        if window_s < needed:
            return (f"{name}: the collection window is {window_s:.0f} s, {groups_in_flight} groups x "
                    f"{pool.pool_seeds} seeds at max_live {live} need about {needed:.0f} s "
                    f"({seconds_per_wave:.0f} s per wave): groups will be cut at the window end")
    return None


def env_package_refusal(policy, *, identity_of: Callable[[str], str] | None = None,
                        need_bridge: bool = True) -> str | None:
    """Why this install cannot play the order's episodes, or None: the env package installed (record 0
    carries its identity, and the task prompts come from it) must be the order's pinned one, and the
    verifiers bridge must load."""
    import importlib

    package = policy.env_package.split("==", 1)[0]
    if identity_of is None:
        from reliquary.environment.agentic_swe import installed_env_package as identity_of
    try:
        identity = identity_of(package)
    except Exception as exc:   # an unreadable install cannot be the pinned code
        return f"cannot compute the installed {package} identity ({exc})"
    if identity != policy.env_package:
        return f"{identity} is installed, the order pins {policy.env_package}"
    if need_bridge:
        try:
            importlib.import_module("reliquary_sandbox_verifiers")
        except Exception as exc:
            return f"the verifiers bridge cannot load: {exc}"
    return None


# -- the stack (one checkpoint) ---------------------------------------------------------------------------


def _download(repo: str, revision: str, cache_dir: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, **({"cache_dir": cache_dir} if cache_dir else {}))


def _make_core(checkpoint_dir: str, **kwargs):
    from reliquary.miner.corpus_generate_server import VllmTurnCore

    return VllmTurnCore(checkpoint_dir, forced=True, **kwargs)


def _max_model_len(core, default: int) -> int:
    """vLLM's own context length (what the engine actually serves), else ``default``."""
    llm = getattr(core, "_llm", None)
    config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
    value = getattr(config, "max_model_len", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else int(default)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("emptying the CUDA cache failed")


def _release_core(core) -> None:
    """Best effort: vLLM's in-process engine and its hidden-state capture go, so a new checkpoint fits."""
    capture = getattr(core, "_capture_cm", None)
    if capture is not None:
        with contextlib.suppress(Exception):
            capture.__exit__(None, None, None)
    for name in ("_engine", "_llm"):
        if hasattr(core, name):
            with contextlib.suppress(Exception):
                delattr(core, name)
    _free_gpu()


def _load_proof_model(checkpoint_dir: str):
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_generation_model

    return load_text_generation_model(checkpoint_dir, torch_dtype=torch.bfloat16,
                                      attn_implementation=ATTN_IMPLEMENTATION).to("cuda:0").eval()


def _toploc():
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof

    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)
    if proof is None:
        raise ValueError("the active protocol profile declares no TOPLOC proof")
    return proof


def _sampling():
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

    return ACTIVE_PROTOCOL_PROFILE.sampling


def _serve(app, port: int):
    from reliquary.miner.agentic_miner import serve_loopback

    return serve_loopback(app, port)


def _make_runner(**kwargs):
    from reliquary.miner.rl_episode_client import RlSignedEpisodeRunner

    return RlSignedEpisodeRunner(**kwargs)


def _submit(url: str, request, *, client, wallet, randomness: str):
    from reliquary.miner.submitter import submit_batch_v2

    return submit_batch_v2(url, request, client=client, wallet=wallet, randomness=randomness)


def _sync_http(url: str):
    import httpx

    return httpx.Client(base_url=url, timeout=60.0)


def _async_http():
    import httpx

    return httpx.AsyncClient(timeout=60.0)


def _hf_prover(**kwargs):
    from reliquary.miner.episode_group_miner import hf_prover

    return hf_prover(**kwargs)


@dataclass
class EpisodeStackDeps:
    """The stack's heavy parts (tests replace them with fakes)."""
    download: Callable[[str, str, str | None], str] = _download
    load_renderer: Callable[..., Any] = None              # agentic_swe.load_turn_renderer
    make_core: Callable[..., Any] = _make_core
    max_model_len: Callable[[Any, int], int] = _max_model_len
    release_core: Callable[[Any], None] = _release_core
    load_proof_model: Callable[[str], Any] = _load_proof_model
    release_model: Callable[[Any], None] = lambda model: _free_gpu()
    make_prover: Callable[..., Callable[[dict, str], dict]] = _hf_prover
    toploc: Callable[[], Any] = _toploc
    sampling: Callable[[], Any] = _sampling
    serve: Callable[[Any, int], Awaitable[tuple[Any, Any]]] = _serve
    make_runner: Callable[..., Any] = _make_runner
    submit: Callable[..., Awaitable[Any]] = _submit
    sync_http: Callable[[str], Any] = _sync_http
    async_http: Callable[[], Any] = _async_http
    package_refusal: Callable[[Any], str | None] = env_package_refusal

    def __post_init__(self) -> None:
        if self.load_renderer is None:
            from reliquary.environment.agentic_swe import load_turn_renderer

            self.load_renderer = load_turn_renderer


@dataclass(frozen=True)
class EpisodeMinerConfig:
    environments: tuple[str, ...]
    validator_url: str
    validator_hotkey: str
    generate_port: int = 8012
    checkpoint_dir: str | None = None          # the HF download cache; None = huggingface_hub's own
    max_live: int | None = None                # live sessions per group; None = the pool's 2M seeds
    groups_in_flight: int = GROUPS_IN_FLIGHT
    harness_env: dict | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int = 16
    seconds_per_wave: float = SECONDS_PER_WAVE
    drain_s: float = DRAIN_S

    def __post_init__(self) -> None:
        if not self.environments:
            raise ValueError("at least one signed-episode environment is needed")
        if not 1 <= int(self.groups_in_flight) <= GROUPS_IN_FLIGHT:
            raise ValueError(f"groups in flight must be 1..{GROUPS_IN_FLIGHT} (the validator's per-operator cap)")
        if self.max_live is not None and int(self.max_live) < 1:
            raise ValueError("max_live must be positive")


class EpisodeStack:
    """One checkpoint's engine, endpoint and proof model; ``miner_for(window)`` builds that window's
    ``EpisodeGroupMiner`` (its submit signs with the window's randomness)."""

    def __init__(self, *, revision: str, make_miner: Callable[[EpisodeWindow], Any],
                 closers: list[Callable[[], Awaitable[None] | None]], engine=None, draws=None,
                 engine_caps: Mapping[str, int] | None = None) -> None:
        self.revision = revision
        self.engine, self.draws = engine, draws
        self.engine_caps = dict(engine_caps or {})
        self._make_miner = make_miner
        self._closers = closers

    def miner_for(self, view: EpisodeWindow):
        return self._make_miner(view)

    async def close(self) -> None:
        await _run_closers(self._closers)


async def _run_closers(closers) -> None:
    while closers:
        close = closers.pop()
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("closing the episode stack failed")


def _episode_policies(contract, environments: Sequence[str], package_refusal) -> dict:
    policies = {}
    for name in environments:
        policy = contract.episode_policy(name)
        if policy is None:
            raise ValueError(f"{name} is not a signed-episode environment of the announced order")
        refusal = package_refusal(policy)
        if refusal:
            raise ValueError(f"{name}: {refusal}")
        policies[name] = policy
    return policies


async def build_episode_stack(view: EpisodeWindow, *, config: EpisodeMinerConfig, wallet, hotkey: str,
                              sign_binding: Callable[[bytes], str], task_prompt: Callable[[str, int], str],
                              deps: EpisodeStackDeps | None = None) -> EpisodeStack:
    """The stack on ``view``'s checkpoint. Raises (refusal to start) when a configured environment is not an
    episode env of the order, its env package is not the pinned one, or the engine's caps are not every
    configured environment's episode policy; nothing stays loaded then."""
    from reliquary.miner.corpus_generate_server import GenerateEngine, build_generate_app
    from reliquary.miner.episode_group_miner import EpisodeGroupMiner, http_verdicts
    from reliquary.miner.forced_draw import ForcedDraws
    from reliquary.miner.rl_episode_client import RL_PREFIX, HttpRlEpisodes, check_engine_caps
    from reliquary.protocol.service_contract import ServiceContract

    deps = deps or EpisodeStackDeps()
    contract = ServiceContract.from_dict(view.announcement["contract"])
    policies = _episode_policies(contract, config.environments, deps.package_refusal)
    first = policies[config.environments[0]]
    total, per_turn = int(first.max_episode_tokens), int(first.max_tokens_per_turn)
    # Before anything loads: one engine serves every configured environment, so their caps must agree.
    for policy in policies.values():
        check_engine_caps(policy, max_total_tokens=total, max_tokens_per_turn=per_turn, max_model_len=total)

    closers: list = []
    try:
        checkpoint_dir = await asyncio.to_thread(deps.download, view.repo, view.revision, config.checkpoint_dir)
        renderers: dict[tuple, Any] = {}

        def renderer_for(policy):
            tools = tuple(policy.tools)
            if tools not in renderers:
                renderers[tools] = deps.load_renderer(checkpoint_dir, tools=tools)
            return renderers[tools]

        renderer = renderer_for(first)
        sampling, proof = deps.sampling(), deps.toploc()
        draws = ForcedDraws()
        core = deps.make_core(checkpoint_dir, sampling=sampling, proof=proof,
                              stop_token_ids=sorted(renderer.stop_ids), max_total_tokens=total,
                              max_num_seqs=config.max_num_seqs, gpu_memory_utilization=config.gpu_memory_utilization)
        closers.append(lambda: deps.release_core(core))
        engine_caps = {"max_total_tokens": total, "max_tokens_per_turn": per_turn,
                       "max_model_len": deps.max_model_len(core, total)}
        for policy in policies.values():
            check_engine_caps(policy, **engine_caps)       # vLLM's own context length included
        engine = GenerateEngine(core, max_total_tokens=total, max_tokens_per_turn=per_turn, draws=draws)
        engine.start()
        closers.append(lambda: asyncio.to_thread(engine.stop))       # it joins a thread: off the event loop
        server, serving = await deps.serve(build_generate_app(engine, model_name=view.repo), config.generate_port)

        async def stop_serving() -> None:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await serving

        closers.append(stop_serving)
        model = await asyncio.to_thread(deps.load_proof_model, checkpoint_dir)
        closers.append(lambda: deps.release_model(model))
        prove = deps.make_prover(model=model, wallet=wallet, toploc=proof)
        sync_http = deps.sync_http(config.validator_url)
        closers.append(getattr(sync_http, "close", lambda: None))
        sessions = HttpRlEpisodes(sync_http, validator_hotkey=config.validator_hotkey, prefix=RL_PREFIX)
        client = deps.async_http()
        closers.append(getattr(client, "aclose", lambda: None))
        verdicts = http_verdicts(config.validator_url, hotkey, client=client)
        generate_url = f"http://127.0.0.1:{config.generate_port}"

        def runner_factory(policy, precommit, *, prompt: str, open_until: float | None):
            return deps.make_runner(policy=policy, precommit=precommit, prompt=prompt, hotkey=hotkey,
                                    sign_binding=sign_binding, sessions=sessions, model_name=view.repo,
                                    renderer_model_dir=checkpoint_dir, generate_url=generate_url, sampling=sampling,
                                    harness_env=config.harness_env, open_until=open_until,
                                    max_live=config.max_live)

        def make_miner(window: EpisodeWindow):
            def submit(request):
                return deps.submit(config.validator_url, request, client=client, wallet=wallet,
                                   randomness=window.randomness)

            return EpisodeGroupMiner(hotkey=hotkey, sign_binding=sign_binding, sessions=sessions,
                                     runner_factory=runner_factory, engine=engine, draws=draws, prove=prove,
                                     submit=submit, renderer_for=renderer_for, task_prompt=task_prompt,
                                     engine_caps=engine_caps, verdicts=verdicts)
    except BaseException:
        await _run_closers(closers)
        raise
    return EpisodeStack(revision=view.revision, make_miner=make_miner, closers=closers, engine=engine,
                        draws=draws, engine_caps=engine_caps)


# -- the loop --------------------------------------------------------------------------------------------


class EpisodeMiningLoop:
    """Poll the window; build the stack on the announced checkpoint once; keep up to ``groups_in_flight``
    groups mining. ``read_state()`` -> ``MinerState`` (or None); ``build_stack(window)`` -> ``EpisodeStack``
    (a ValueError is a refusal to start and ends the loop; any other error is logged and the build retried
    at a later poll, with a bounded backoff); ``sizes(environment)`` -> tasks the miner's source holds.
    A new (revision, contract sha) drains the groups in flight (``drain_s``, then cancelled) and raises
    ``CheckpointChanged``: the process exits and its supervisor restarts it on the new checkpoint;
    ``preflight(window)`` (optional) says beforehand why the restarted miner would refuse that order."""

    def __init__(self, *, environments: Sequence[str], read_state: Callable[[], Awaitable[Any]],
                 build_stack: Callable[[EpisodeWindow], Awaitable[EpisodeStack]],
                 sizes: Callable[[str], int], groups_in_flight: int = GROUPS_IN_FLIGHT,
                 rng: random.Random | None = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep, poll_s: float = STATE_POLL_S,
                 drain_s: float = DRAIN_S, preflight: Callable[[EpisodeWindow], str | None] | None = None,
                 max_live: int | None = None, seconds_per_wave: float = SECONDS_PER_WAVE) -> None:
        if not 1 <= int(groups_in_flight) <= GROUPS_IN_FLIGHT:
            raise ValueError(f"groups in flight must be 1..{GROUPS_IN_FLIGHT} (the validator's per-operator cap)")
        self._environments = tuple(environments)
        self._read_state = read_state
        self._build_stack = build_stack
        self._sizes = sizes
        self._cap = int(groups_in_flight)
        self._rng = rng or random.Random()
        self._clock = clock
        self._sleep = sleep
        self._poll_s = float(poll_s)
        self._drain_s = float(drain_s)
        self._preflight = preflight
        self._max_live = max_live
        self._seconds_per_wave = float(seconds_per_wave)
        self.stack: EpisodeStack | None = None
        self.stack_key: tuple[str, str] | None = None
        self._next_build_at = 0.0
        self._build_backoff = BUILD_BACKOFF_S
        self._inflight: dict[asyncio.Task, tuple[int, str, int]] = {}
        self._taken: set[tuple[int, str, int]] = set()
        self.outcomes: deque[tuple[tuple[int, str, int], Any]] = deque(maxlen=OUTCOMES_KEPT)

    @property
    def in_flight(self) -> list[tuple[int, str, int]]:
        return list(self._inflight.values())

    async def step(self) -> EpisodeWindow | None:
        """One poll: the window read (or None). Raises ``CheckpointChanged`` (after the drain) on a new
        checkpoint or order, and a build's ValueError (refusal to start)."""
        try:
            state = await self._read_state()
        except Exception as exc:
            logger.warning("miner state unavailable: %s", exc)
            return None
        view = episode_window(state)
        if view is None:
            return None
        if self.stack is None:
            if not await self._build(view):
                return None
        elif view.key != self.stack_key:
            await self._restart(view)
        self._fill(view)
        return view

    async def _build(self, view: EpisodeWindow) -> bool:
        if self._clock() < self._next_build_at:
            return False
        try:
            stack = await self._build_stack(view)
        except ValueError:
            raise
        except Exception as exc:
            logger.exception("building the episode stack failed (%r); retried in %.0f s", exc, self._build_backoff)
            self._next_build_at = self._clock() + self._build_backoff
            self._build_backoff = min(self._build_backoff * 2, BUILD_BACKOFF_MAX_S)
            return False
        self.stack, self.stack_key = stack, view.key
        self._build_backoff, self._next_build_at = BUILD_BACKOFF_S, 0.0
        try:
            warning = short_window_warning(view, environments=self._environments, max_live=self._max_live,
                                           groups_in_flight=self._cap, seconds_per_wave=self._seconds_per_wave)
        except Exception:
            logger.exception("checking the collection window failed")
            warning = None
        if warning:
            logger.warning("!!! COLLECTION WINDOW TOO SHORT !!! %s", warning)
        return True

    async def _restart(self, view: EpisodeWindow) -> None:
        logger.warning("checkpoint %s / order %s announced (was %s / %s): draining %d groups, then exit %d "
                       "for the supervisor to restart the miner", view.revision[:12], view.contract_sha[:12],
                       *(part[:12] for part in self.stack_key), len(self._inflight), EXIT_CHECKPOINT_CHANGED)
        if self._preflight is not None:
            try:
                refusal = self._preflight(view)
            except Exception as exc:
                refusal = repr(exc)
            if refusal:
                logger.error("the restarted miner will refuse the new order: %s", refusal)
        await self._drain()
        raise CheckpointChanged(f"checkpoint {view.revision} / order {view.contract_sha} announced")

    async def _drain(self) -> None:
        tasks = list(self._inflight)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self._drain_s)
            if pending:
                logger.warning("%d groups still in flight after %.0f s: cancelled", len(pending), self._drain_s)
        await self._cancel_inflight()

    def _sizes_of(self, view: EpisodeWindow) -> dict[str, int]:
        from reliquary.protocol.service_contract import ServiceContract

        contract = ServiceContract.from_dict(view.announcement["contract"])
        sizes = {}
        for name in self._environments:
            if name in contract.environments:
                sizes[name] = min(int(contract.environments[name]["dataset"]["rows"]), int(self._sizes(name)))
        return sizes

    def _fill(self, view: EpisodeWindow) -> None:
        self._taken = {key for key in self._taken if key[0] == view.window}
        if view.open_until is not None and self._clock() >= view.open_until:
            return
        if len(self._inflight) >= self._cap:
            return
        sizes = self._sizes_of(view)
        miner = None
        while len(self._inflight) < self._cap:
            taken = {(env, task) for window, env, task in self._taken if window == view.window}
            pick = pick_episode_task(view, environments=self._environments, sizes=sizes, taken=taken, rng=self._rng)
            if pick is None:
                return
            environment, task_index = pick
            key = (view.window, environment, task_index)
            self._taken.add(key)
            miner = miner or self.stack.miner_for(view)
            task = asyncio.ensure_future(miner.mine_task(
                announcement=view.announcement, randomness=view.randomness, window=view.window,
                environment=environment, task_index=task_index, open_until=view.open_until))
            self._inflight[task] = key
            task.add_done_callback(self._finished)
            logger.info("window %d: mining %s task %d (%d groups in flight)", view.window, environment,
                        task_index, len(self._inflight))

    def _finished(self, task: asyncio.Task) -> None:
        key = self._inflight.pop(task, None)
        if task.cancelled():
            outcome: Any = "cancelled"
        elif task.exception() is not None:
            outcome = repr(task.exception())          # no traceback frames kept alive
            logger.warning("group %s ended: %s", key, outcome)
        else:
            outcome = task.result()
            logger.info("group %s: %s", key, outcome)
        self.outcomes.append((key, outcome))

    async def _cancel_inflight(self) -> None:
        tasks = list(self._inflight)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        """Until ``stop`` is set (or forever). A refusal to start (ValueError) ends the loop with its error,
        a new checkpoint or order with ``CheckpointChanged``."""
        try:
            while stop is None or not stop.is_set():
                await self.step()
                await self._sleep(self._poll_s)
        finally:
            await self._cancel_inflight()
            if self.stack is not None:
                stack, self.stack = self.stack, None
                await stack.close()


def miner_state_reader(url: str, *, client) -> Callable[[], Awaitable[Any]]:
    """``read_state()`` over ``GET /miner-state`` (conditional on its ETag): the latest state, or None
    between windows."""
    from reliquary.miner.submitter import SubmissionError, get_miner_state_v1

    cache: dict[str, Any] = {"state": None, "etag": None}

    async def read():
        try:
            state, etag = await get_miner_state_v1(url, client=client, etag=cache["etag"])
        except SubmissionError as exc:          # 503 between windows
            logger.debug("miner state: %s", exc)
            return None
        if state is not None:
            cache["state"] = state
        cache["etag"] = etag
        return cache["state"]

    return read


async def run_episode_miner(*, config: EpisodeMinerConfig, wallet, stop: asyncio.Event | None = None,
                            deps: EpisodeStackDeps | None = None, task_prompts: EpisodeTaskPrompts | None = None,
                            read_state: Callable[[], Awaitable[Any]] | None = None, **loop_kwargs) -> None:
    """The episode miner in one process: the loop over ``config``'s environments on the RL validator."""
    import httpx

    prompts = task_prompts or EpisodeTaskPrompts(config.environments)
    hotkey = wallet.hotkey.ss58_address

    def sign_binding(binding: bytes) -> str:
        return wallet.hotkey.sign(binding).hex()

    async with contextlib.AsyncExitStack() as stack:
        if read_state is None:
            client = await stack.enter_async_context(httpx.AsyncClient(timeout=30.0))
            read_state = miner_state_reader(config.validator_url, client=client)

        def build(view: EpisodeWindow):
            return build_episode_stack(view, config=config, wallet=wallet, hotkey=hotkey, sign_binding=sign_binding,
                                       task_prompt=prompts, deps=deps)

        package_refusal = deps.package_refusal if deps is not None else env_package_refusal

        def preflight(view: EpisodeWindow) -> str | None:
            # The new order's pins, checked now: the restarted miner would refuse to start on them.
            from reliquary.protocol.service_contract import ServiceContract

            try:
                _episode_policies(ServiceContract.from_dict(view.announcement["contract"]), config.environments,
                                  package_refusal)
            except ValueError as exc:
                return str(exc)
            return None

        loop_kwargs = {"preflight": preflight, "max_live": config.max_live, "drain_s": config.drain_s,
                       "seconds_per_wave": config.seconds_per_wave, **loop_kwargs}
        loop = EpisodeMiningLoop(environments=config.environments, read_state=read_state, build_stack=build,
                                 sizes=prompts.size, groups_in_flight=config.groups_in_flight, **loop_kwargs)
        await loop.run(stop=stop)


__all__ = ["CheckpointChanged", "EXIT_CHECKPOINT_CHANGED", "EpisodeMinerConfig", "EpisodeMiningLoop", "EpisodeStack", "EpisodeStackDeps", "EpisodeTaskPrompts",
           "EpisodeWindow", "GROUPS_IN_FLIGHT", "build_episode_stack", "env_package_refusal", "episode_window",
           "miner_state_reader", "parse_episode_environments", "parse_harness_env", "pick_episode_task",
           "run_episode_miner", "short_window_warning"]
