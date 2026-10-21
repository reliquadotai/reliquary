"""Episodes for the async reference miner (rl-async-design.md §4.10; implementation map §1.15, §9).

``EpisodeLeaseGenerator`` is a ``GroupGenerator`` with ``agenerate``: per lease of a signed-episode env it plays
the lease's ``ROLLOUTS`` episodes at once through the window path's machinery, unchanged but for the binding:

* the session is opened for rollout i of the lease (``rl_episode_client.LeaseEpisodeRunner``: engagement
  ``rl-lease:{lease_id}:{i}``; the validator's sandbox issuer grants at most one per rollout);
* every generate session of the episode is bound to ``forced_draw.LeaseDrawBinding`` (nonce, prompt, k, i): the
  forced engine (``corpus_generate_server.GenerateEngine``) starts each turn's draw at the model tokens generated so
  far, so the draw is keyed per model-token position across turns (``protocol/rl_draw``);
* an ended episode is kept only when the validator would admit it (``episode_commit.episode_admissible``: graded,
  usable, its stop, length, prompt fidelity and turn shape) and becomes an ``rl_async.EpisodeRollout``.

Submission is mandatory and an episode cannot be withheld: when one of the lease's episodes is unusable the whole
lease fails (its graded sessions are withdrawn, the lease expires unsubmitted); the miner never chooses among
episodes. A machine-signed ``aborted`` final is played again once (the issuer frees that engagement once), as in
the window miner.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

ABORT_REPLAYS = 1
# A rollout whose open a full machine refused (signed receipt: its engagement is free again) is opened again after
# this wait, doubled up to the cap, until the lease's deadline: a lease takes all 16 of its episodes.
OPEN_RETRY_S, OPEN_RETRY_MAX_S = 15.0, 120.0


class EpisodeLeaseFailed(RuntimeError):
    """A lease whose episodes cannot all be submitted."""


class EpisodeLeaseGenerator:
    """``policies[env]``: the env's episode policy; ``runner_factory(policy, lease, prompt)``: an async context
    manager running episodes (``LeaseEpisodeRunner``); ``engine``: the forced generate engine (``take_session``,
    ``drop_session``); ``draws``: its ``ForcedDraws``; ``renderer_for(policy)``: the turn renderer the validator
    builds; ``task_prompt(env, prompt_idx)``: the task's prompt text; ``load_weights(state, version)``: how the
    engine takes a new checkpoint (None: refused, the miner restarts on new weights)."""

    def __init__(self, *, policies: Mapping[str, Any], runner_factory: Callable[..., Any], engine, draws,
                 renderer_for: Callable[[Any], Any], task_prompt: Callable[[str, int], str],
                 load_weights: Callable[[Mapping[str, Any], int], None] | None = None,
                 rollouts: int = 16, suites: Mapping[str, Any] | None = None) -> None:
        self.policies, self._factory, self._engine, self._draws = dict(policies), runner_factory, engine, draws
        # Replayed envs (``rl_suite_episodes.SuiteLeasePlayer``): played on the same engine, no sandbox.
        self.suites = dict(suites or {})
        self._renderer_for, self._task_prompt, self._load = renderer_for, task_prompt, load_weights
        self.rollouts = int(rollouts)
        self._renderers: dict[str, Any] = {}
        self._running: dict[str, asyncio.Task] = {}

    @property
    def environments(self) -> frozenset[str]:
        """Every env this generator plays: the signed-episode ones and the replayed ones."""
        return frozenset(self.policies) | frozenset(self.suites)

    @property
    def swaps_running(self) -> bool:
        """New weights go into the episode engine with episodes in flight (``EpisodeEngineHost.push``)."""
        return self._load is not None

    def generate(self, leases) -> list:
        raise RuntimeError("episodes are played on the event loop: call agenerate")

    async def agenerate(self, leases: Sequence[Any]) -> list[Any]:
        """Every lease's ``ROLLOUTS`` episodes (``EpisodeRollout``), leases played side by side. A lease that fails
        comes back as its ``EpisodeLeaseFailed``, the others' episodes are kept: a lease's sessions are single-use,
        so it cannot be played again and the miner gives it up (``AsyncMiner.generate_once``)."""
        ensure = getattr(self._engine, "ensure_serving", None)
        if ensure is not None:
            await ensure()                    # the loopback generate endpoint, served on this loop once
        results = await asyncio.gather(*(self.play_lease(lease) for lease in leases), return_exceptions=True)
        return [result if not isinstance(result, BaseException)
                else result if isinstance(result, EpisodeLeaseFailed)
                else EpisodeLeaseFailed(f"lease {lease.lease_id[:12]}: {result!r}")
                for lease, result in zip(leases, results)]

    async def aadvance(self, leases: Sequence[Any], budget_s: float) -> list[tuple[str, Any]]:
        """Continuous play (``AsyncMiner._stream_once``): start ``leases``, each in a task of its own, wait about
        ``budget_s`` for any running lease to end, return the ended ones as ``(lease id, episodes or its
        EpisodeLeaseFailed)``. A long lease never holds the others back."""
        ensure = getattr(self._engine, "ensure_serving", None)
        if leases and ensure is not None:
            await ensure()

        async def play(lease):
            try:
                return await self.play_lease(lease)
            except EpisodeLeaseFailed as failed:
                return failed
            except Exception as exc:
                return EpisodeLeaseFailed(f"lease {lease.lease_id[:12]}: {exc!r}")

        for lease in leases:
            self._running[lease.lease_id] = asyncio.create_task(play(lease))
        if not self._running:
            return []
        await asyncio.wait(list(self._running.values()), timeout=budget_s, return_when=asyncio.FIRST_COMPLETED)
        done = [(lease_id, task.result()) for lease_id, task in self._running.items() if task.done()]
        for lease_id, _ in done:
            del self._running[lease_id]
        return done

    def abort(self) -> None:
        for task in self._running.values():
            task.cancel()
        self._running.clear()

    def load_weights(self, state: Mapping[str, Any], version: int) -> None:
        if self._load is None:
            raise RuntimeError("this episode engine takes no new weights in place: restart the miner")
        self._load(state, version)

    def _renderer(self, policy, variant: int = 0, spec=None):
        """The validator's renderer of the lease's surface: ``spec`` is named only for a harness mix."""
        key = (policy.environment, None if spec is None else spec.key, int(variant))
        if key not in self._renderers:
            kwargs = {"spec": spec} if spec is not None else {}
            self._renderers[key] = (self._renderer_for(policy, **kwargs) if variant == 0
                                    else self._renderer_for(policy, variant=variant, **kwargs))
        return self._renderers[key]

    async def play_lease(self, lease) -> list[Any]:
        from reliquary.miner.episode_commit import episode_admissible
        from reliquary.miner.episode_group_miner import play_bound
        from reliquary.miner.forced_draw import LeaseDrawBinding
        from reliquary.miner.rl_async import EpisodeRollout

        player = self.suites.get(lease.environment)
        if player is not None:
            return await player.play(lease)
        policy = self.policies.get(lease.environment)
        if policy is None:
            raise EpisodeLeaseFailed(f"{lease.environment} is not an episode environment of this miner")
        from reliquary.environment.agentic_swe import lease_surface, policy_harness_choices

        # The lease's harness (drawn from a mix, else the env's one) and its surface variant (0 for a harness
        # without axes): its 16 episodes all play them.
        spec, variant = lease_surface(policy, lease)
        mixed = len(policy_harness_choices(policy)) > 1
        renderer = self._renderer(policy, variant, spec if mixed else None)
        prompt = self._task_prompt(lease.environment, int(lease.prompt_idx))
        prompt_len = len(renderer.initial_ids(prompt))
        kwargs = {**({"variant": variant} if variant else {}), **({"harness": spec} if mixed else {})}
        runner = self._factory(policy, lease, prompt, **kwargs)
        label = f"lease {lease.lease_id[:12]}"
        if variant or mixed:
            logger.info("%s plays %s surface variant %d of %s", label, spec.key, variant, lease.environment)

        async def one(index: int):
            binding = LeaseDrawBinding(lease.nonce, int(lease.prompt_idx), int(lease.checkpoint), index)
            replays, wait = 0, OPEN_RETRY_S
            while True:
                played = await play_bound(runner, self._engine, self._draws, index, binding,
                                          getattr(runner, "deadline", None), label=label)
                if played is None:
                    return None
                result, session = played
                status = None if result.ok else getattr(result, "final_status", None)
                if status == "open_refused":
                    logger.info("rollout %d of %s: its machine refused the open; opened again in %.0f s",
                                index, label, wait)
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, OPEN_RETRY_MAX_S)
                    continue
                if status == "aborted" and replays < ABORT_REPLAYS:
                    replays += 1
                    logger.info("rollout %d of %s aborted by its machine: played again", index, label)
                    continue
                return SimpleNamespace(seed_index=index, result=result, session=session)

        async with runner:
            outcomes = await asyncio.gather(*(one(i) for i in range(self.rollouts)), return_exceptions=True)
            kept, why = [], None
            for index, outcome in enumerate(outcomes):
                if isinstance(outcome, BaseException) or outcome is None:
                    why = why or f"rollout {index}: no episode ({outcome!r})"
                    kept.append(None)
                    continue
                admitted, refusal = episode_admissible(outcome, policy=policy, renderer=renderer, prompt=prompt,
                                                       lease=True)
                if admitted is None:
                    why = why or f"rollout {index}: {refusal}"
                kept.append(admitted)
            if why is not None:
                logger.warning("%s cannot be submitted: %s", label, why)   # now, not when its batch ends
                await _release([o for o in outcomes if not isinstance(o, BaseException) and o is not None])
                raise EpisodeLeaseFailed(why)
            for outcome in outcomes:
                done = getattr(outcome.result, "submitted", None)
                if done is not None:
                    done()                    # always submitted: the live slot is free, the session kept graded
        return [EpisodeRollout(tuple(int(t) for t in value.tokens[prompt_len:]),
                               tuple((int(s) - prompt_len, int(e) - prompt_len) for s, e in value.spans),
                               str(value.stop), dict(outcome.result.transcript))
                for value, outcome in zip(kept, outcomes)]


class EpisodeEngineHost:
    """The forced generate engine of the async episode miner: ``VllmTurnCore`` (forced, the lease draw read per
    request by the runner-keyed processor) under a ``GenerateEngine`` bound to ``draws``, served on loopback by one
    generate app that outlives the engines (the app holds this host, which hands every call to the current engine).

    The first checkpoint (or one whose in-place push failed with no turn in flight) starts the engine on an HF
    directory (``restart``); later ones are pushed in place by the engine's own thread between two steps, the
    turns in flight preempted and resumed on the new weights (``push``), so a swap never waits for an episode.

    ``make_core(model_dir, **kwargs)``, ``release_core(core)``, ``serve(app, port) -> (server, task)``: the
    window stack's (``episode_mining``), replaceable in tests. ``stop_ids()``: the renderer's turn stops."""

    def __init__(self, *, make_core, release_core, serve, draws, model_name: str, port: int, max_total_tokens: int,
                 max_tokens_per_turn: int, stop_ids: Callable[[], Any], sampling, proof,
                 core_kwargs: Mapping[str, Any] | None = None) -> None:
        self._make_core, self._release_core, self._serve = make_core, release_core, serve
        self.draws, self.model_name, self.port = draws, model_name, int(port)
        self.max_total_tokens, self.max_tokens_per_turn = int(max_total_tokens), int(max_tokens_per_turn)
        self._stop_ids, self._sampling, self._proof = stop_ids, sampling, proof
        self._core_kwargs = dict(core_kwargs or {})
        self.core = self.engine = None
        self.model_dir: str | None = None
        self.restarts = 0
        self._server = self._serving = None
        # ``push`` ships a function to the workers through collective_rpc: allowed before the engine starts them.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    @property
    def generate_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __getattr__(self, name: str):
        # The generate app's calls (``generate``, ...) and the miner's (``take_session``, ``drop_session``) go to the
        # engine of the moment.
        engine = self.__dict__.get("engine")
        if engine is None:
            raise AttributeError(name)
        return getattr(engine, name)

    def restart(self, model_dir: str) -> None:
        """Stop the engine on the old weights (if any) and start one on ``model_dir``. Blocking."""
        from reliquary.miner.corpus_generate_server import GenerateEngine

        self.close_engine()
        core = self._make_core(str(model_dir), sampling=self._sampling, proof=self._proof,
                               stop_token_ids=sorted(self._stop_ids()), max_total_tokens=self.max_total_tokens,
                               **self._core_kwargs)
        engine = GenerateEngine(core, max_total_tokens=self.max_total_tokens,
                                max_tokens_per_turn=self.max_tokens_per_turn, draws=self.draws)
        engine.start()
        self.core, self.engine, self.model_dir = core, engine, str(model_dir)
        self.restarts += 1

    def push(self, state: Mapping[str, Any], scratch: Path) -> None:
        """New weights into the running engine, in place and with turns in flight: between two engine steps (the
        engine thread runs it), every worker loads the shards through the model's ``load_weights``
        (``LLM.collective_rpc``) and the prefix cache is reset with the running turns preempted, so they resume on
        the new weights (their KV recomputed, their draw placed by the runner's own token count). The engine, its
        sessions and their waiters stay: no episode is interrupted, no model load, no second copy on the GPU."""
        from reliquary.miner.rl_async import _load_weights_from_files
        from reliquary.shared.ckpt_delta import write_anchor_shards

        if self.core is None or self.engine is None:
            raise RuntimeError("no engine to push weights into")
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            paths = [str(p) for p in write_anchor_shards(state, scratch)]
            llm = self.core._llm

            def load() -> None:
                llm.collective_rpc(_load_weights_from_files, args=(paths,))
                llm.reset_prefix_cache(reset_running_requests=True)

            self.engine.between_steps(load)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def has_turns_in_flight(self) -> bool:
        core = self.core
        return core is not None and bool(core.has_unfinished())

    def close_engine(self) -> None:
        engine, core = self.engine, self.core
        self.engine = self.core = None
        if engine is not None:
            engine.stop()
        if core is not None:
            self._release_core(core)

    async def ensure_serving(self) -> None:
        if self._server is not None:
            return
        from reliquary.miner.corpus_generate_server import build_generate_app

        if self.engine is None:
            raise RuntimeError("no weights loaded")
        self._server, self._serving = await self._serve(build_generate_app(self, model_name=self.model_name),
                                                        self.port)

    async def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            with contextlib.suppress(Exception):
                await self._serving
            self._server = None
        await asyncio.to_thread(self.close_engine)


class EpisodeWeights:
    """``EpisodeLeaseGenerator``'s ``load_weights``: pushed into the running engine (``EpisodeEngineHost.push``);
    on the first load, or when the push fails, the state as an HF directory under ``work_dir`` (config and
    tokenizer from ``base_model_dir``) and the engine restarted on it (``EpisodeEngineHost.restart``); the
    previous directory is removed."""

    def __init__(self, host: EpisodeEngineHost, *, base_model_dir: str, work_dir: str) -> None:
        self.host, self.base, self.work = host, Path(base_model_dir), Path(work_dir)
        self._dir: Path | None = None

    def __call__(self, state: Mapping[str, Any], version: int) -> None:
        from reliquary.miner.rl_async import write_hf_dir

        if self.host.core is not None:
            try:
                self.host.push(state, Path("/dev/shm" if Path("/dev/shm").is_dir() else self.work)
                               / f"rl-episode-push-{os.getpid()}")
                logger.info("episode engine on checkpoint %d (weights pushed in place)", version)
                return
            except Exception:
                if self.host.has_turns_in_flight():
                    raise                     # a restart would end the episodes in flight: the miner drains first
                logger.exception("in-place weight push failed; restarting the episode engine on checkpoint %d",
                                 version)
        if not (self.base / "config.json").is_file():
            raise RuntimeError(f"no model files (config, tokenizer) in {self.base}")
        new_dir = write_hf_dir(state, self.base, self.work / f"episode-weights-{int(version):08d}")
        self.host.restart(str(new_dir))
        if self._dir is not None and self._dir != new_dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = new_dir
        logger.info("episode engine restarted on checkpoint %d", version)


class MixedGenerator:
    """One ``GroupGenerator`` over the miner's envs: the episode envs' leases go to ``episodes``
    (``EpisodeLeaseGenerator``, on the event loop), the others to ``single`` (``VllmGroupGenerator``, in a worker
    thread); groups come back in the batch's order. ``load_weights`` reaches both (between batches only)."""

    def __init__(self, *, episodes: EpisodeLeaseGenerator | None, single=None) -> None:
        if episodes is None and single is None:
            raise ValueError("no generator")
        self.episodes, self.single = episodes, single

    def _episode_names(self) -> set[str]:
        if self.episodes is None:
            return set()
        return set(getattr(self.episodes, "environments", None) or self.episodes.policies)

    def generate(self, leases) -> list:
        raise RuntimeError("episodes are played on the event loop: call agenerate")

    async def agenerate(self, leases: Sequence[Any]) -> list[list[Any]]:
        names = self._episode_names()
        played = [lease for lease in leases if lease.environment in names]
        drawn = [lease for lease in leases if lease.environment not in names]
        if drawn and self.single is None:
            raise EpisodeLeaseFailed(f"{drawn[0].environment} is not an episode environment of this miner")
        jobs = []
        if played:
            jobs.append(self.episodes.agenerate(played))
        if drawn:
            jobs.append(asyncio.to_thread(self.single.generate, drawn))
        results = await asyncio.gather(*jobs)
        by_id = {}
        for group_leases, groups in zip([x for x in (played, drawn) if x], results):
            by_id.update({lease.lease_id: group for lease, group in zip(group_leases, groups)})
        return [by_id[lease.lease_id] for lease in leases]

    async def aadvance(self, leases: Sequence[Any], budget_s: float) -> list[tuple[str, Any]]:
        """Continuous play of both kinds: the episode leases through ``EpisodeLeaseGenerator.aadvance``, the others
        through the single-turn generator's ``advance`` (in a worker thread), side by side."""
        names = self._episode_names()
        played = [lease for lease in leases if lease.environment in names]
        drawn = [lease for lease in leases if lease.environment not in names]
        done: list[tuple[str, Any]] = []
        if drawn and (self.single is None or not callable(getattr(self.single, "advance", None))):
            done += [(lease.lease_id, EpisodeLeaseFailed(f"{lease.environment} cannot be played continuously here"))
                     for lease in drawn]
            drawn = []
        jobs = []
        if self.episodes is not None:
            jobs.append(self.episodes.aadvance(played, budget_s))
        if self.single is not None and callable(getattr(self.single, "advance", None)):
            jobs.append(asyncio.to_thread(self.single.advance, drawn, budget_s))
        for result in await asyncio.gather(*jobs):
            done += list(result)
        return done

    def abort(self) -> None:
        if self.episodes is not None:
            self.episodes.abort()
        if self.single is not None and callable(getattr(self.single, "abort", None)):
            self.single.abort()

    @property
    def swaps_running(self) -> bool:
        return all(bool(getattr(side, "swaps_running", False)) for side in (self.episodes, self.single)
                   if side is not None)

    def load_weights(self, state: Mapping[str, Any], version: int) -> None:
        if self.single is not None:
            self.single.load_weights(state, version)
        if self.episodes is not None:
            self.episodes.load_weights(state, version)


def build_episode_generator(blocks: Mapping[str, Mapping[str, Any]], *, validator_url: str, validator_hotkey: str,
                            hotkey: str, sign_binding: Callable[[bytes], str], base_model_dir: str, work_dir: str,
                            generate_port: int = 8013, harness_env: Mapping[str, str] | None = None,
                            max_num_seqs: int = 64, gpu_memory_utilization: float | None = None,
                            deps=None, load_environment=None, suites: Sequence[str] = (),
                            load_tokenizer=None, suite_environments=None) -> EpisodeLeaseGenerator:
    """The episode side of ``mine-async`` (the window ``build_episode_stack`` minus precommits and the TOPLOC
    prover): ``blocks``: the validator's ``GET /rl/episodes`` envs (``rl_async_episodes`` blocks, validated here
    again); one forced engine for every env (their caps must be equal: ``check_engine_caps``), started at the
    first ``load_weights``; one ``LeaseEpisodeRunner`` per lease over ``HttpRlEpisodes``; the task prompts are the
    validator's own source (``load_episode_environment``: the catalogue's bridged source, the installed identity
    checked against the block's pin). ``suites``: the miner's replayed envs (``rl_suite_episodes``), played on the
    same engine (their caps must be the engine's too), their worlds built here (``suite_environments``: tests).
    ``deps``: ``episode_mining.EpisodeStackDeps`` (tests: fakes); ``load_tokenizer``: the replayed envs' tokenizer
    (tests)."""
    from reliquary.miner.episode_mining import EpisodeStackDeps
    from reliquary.miner.forced_draw import ForcedDraws
    from reliquary.miner.rl_episode_client import RL_PREFIX, HttpRlEpisodes, LeaseEpisodeRunner, check_engine_caps
    from reliquary.validator.rl_async_episodes import async_episode_policy, validate_episode_block

    if load_environment is None:
        from reliquary.validator.rl_async_episodes import load_episode_environment as load_environment
    from reliquary.validator.rl_suite_episodes import suite_lease_policy

    deps = deps or EpisodeStackDeps()
    blocks = {name: validate_episode_block(name, block) for name, block in blocks.items()}
    suites = tuple(suites or ())
    if not blocks and not suites:
        raise ValueError("no episode environment")
    policies = {name: async_episode_policy(name, block) for name, block in blocks.items()}
    capped = {**policies, **{name: suite_lease_policy(name) for name in suites}}
    first = capped[sorted(capped)[0]]
    total, per_turn = int(first.max_episode_tokens), int(first.max_tokens_per_turn)
    for name, policy in capped.items():
        try:
            check_engine_caps(policy, max_total_tokens=total, max_tokens_per_turn=per_turn, max_model_len=total)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}; one miner engine serves only envs with the same caps: mine {name} "
                             f"with a miner process of its own") from exc
    from reliquary.environment.agentic_swe import (
        HARNESS, policy_default_harness, policy_harness_refusal,
    )

    for name, policy in policies.items():
        refusal = deps.package_refusal(policy) or policy_harness_refusal(policy)
        if refusal:
            raise ValueError(f"{name}: {refusal}")
    sources = {name: load_environment(name, block).source for name, block in blocks.items()}
    codec = None
    if suites:
        from reliquary.corpus.suite_episode import TokenizerCodec
        from reliquary.miner.suite_episode import suite_stop_ids

        if load_tokenizer is None:
            from reliquary.shared.modeling import load_tokenizer
        tokenizer = load_tokenizer(str(base_model_dir))
        codec = TokenizerCodec(tokenizer)
    renderers: dict[tuple, Any] = {}

    def renderer_for(policy, variant: int = 0, spec=None):
        """The validator's turn renderer of ``policy``'s tools in harness ``spec`` (default: its first, the only
        one without a mix) and surface ``variant``."""
        tools, spec = tuple(policy.tools), spec or policy_default_harness(policy)
        key = (tools, spec.key, int(variant))
        if key not in renderers:
            renderers[key] = (deps.load_renderer(str(base_model_dir), tools=tools) if spec is HARNESS and variant == 0
                              else deps.load_renderer(str(base_model_dir), tools=tools, spec=spec,
                                                      variant=int(variant)))
        return renderers[key]

    def stop_ids():
        """The engine's turn stops: the signed envs' renderer's, the replayed envs' (ChatML) otherwise; one engine
        serves both kinds only when they are the same."""
        signed = None if not policies else set(renderer_for(policies[sorted(policies)[0]]).stop_ids)
        replayed = None if codec is None else set(suite_stop_ids(codec.encode))
        if signed is not None and replayed is not None and signed != replayed:
            raise ValueError(f"the signed episodes stop on {sorted(signed)}, the replayed ones on {sorted(replayed)}: "
                             "mine them with a miner process each")
        return signed if signed is not None else replayed

    draws = ForcedDraws()
    sampling = deps.sampling()
    host = EpisodeEngineHost(make_core=deps.make_core, release_core=deps.release_core, serve=deps.serve, draws=draws,
                             model_name=str(base_model_dir), port=generate_port, max_total_tokens=total,
                             max_tokens_per_turn=per_turn, stop_ids=stop_ids,
                             sampling=sampling, proof=None,          # no span proof on the async path
                             core_kwargs={"max_num_seqs": max_num_seqs,
                                          "gpu_memory_utilization": gpu_memory_utilization})
    sessions = HttpRlEpisodes(deps.sync_http(validator_url), validator_hotkey=validator_hotkey, prefix=RL_PREFIX)

    def runner_factory(policy, lease, prompt, variant: int = 0, harness=None):
        return LeaseEpisodeRunner(policy=policy, lease=lease, hotkey=hotkey, prompt=prompt,
                                  sign_binding=sign_binding, sessions=sessions, model_name=host.model_name,
                                  renderer_model_dir=str(base_model_dir), generate_url=host.generate_url,
                                  sampling=sampling, harness_env=dict(harness_env or {}) or None,
                                  harness=harness or policy_default_harness(policy), variant=variant)

    players = {}
    if suites:
        from reliquary.miner.rl_suite_episodes import SuiteLeasePlayer
        from reliquary.environment import load_episode_environment

        for name in suites:
            world = (suite_environments or {}).get(name) or load_episode_environment(name)
            players[name] = SuiteLeasePlayer(policy=suite_lease_policy(name, environment=world), environment=world,
                                             codec=codec, engine=host, draws=draws)
    generator = EpisodeLeaseGenerator(
        policies=policies, runner_factory=runner_factory, engine=host, draws=draws, renderer_for=renderer_for,
        task_prompt=lambda env, idx: sources[env].prompt(int(idx)),
        load_weights=EpisodeWeights(host, base_model_dir=base_model_dir, work_dir=work_dir), suites=players)
    generator.host, generator.sources, generator.blocks = host, sources, blocks
    return generator


async def _release(outcomes) -> None:
    for outcome in outcomes:
        release = getattr(outcome.result, "release", None)
        if release is None:
            continue
        try:
            await release()
        except Exception:
            logger.exception("withdrawing the episode of rollout %s failed", outcome.seed_index)


__all__ = ["ABORT_REPLAYS", "EpisodeEngineHost", "EpisodeLeaseFailed", "EpisodeLeaseGenerator", "EpisodeWeights",
           "MixedGenerator", "build_episode_generator"]
