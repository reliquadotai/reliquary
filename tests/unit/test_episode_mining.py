"""The episode-group miner's production entry point: the per-window loop and its wiring.
CPU only: fake vLLM core, proof model and validator; the generate app, the forced engine, the draws, the
RL precommit client, the verdict poll and the group miner are the real ones."""
import asyncio
import json
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

bt = pytest.importorskip("bittensor")
attest = pytest.importorskip("reliquary_sandbox.attest")
httpx = pytest.importorskip("httpx")

from reliquary.constants import M_ROLLOUTS  # noqa: E402
from reliquary.miner import episode_mining as em  # noqa: E402
from reliquary.miner.agentic_episode import EpisodeResult  # noqa: E402
from reliquary.miner.corpus_generate_server import SESSION_HEADER, Finished  # noqa: E402
from reliquary.miner.vllm_generation import FORCED_SEED_KEY  # noqa: E402
from reliquary.protocol.service_episode import EpisodePrecommit  # noqa: E402
from reliquary.protocol.signatures import verify_episode_precommit_signature  # noqa: E402
from reliquary.protocol.submission import MinerEnvironmentState, MinerState, WindowState, encode_cooldown_bitmap  # noqa: E402
from reliquary.sandbox.rl_routes import episode_precommit_path  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, PROMPT_TEXT, WINDOW_BEACON, episode_runtime, episode_signers, play_episode, register_episode_env,
)
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_episode_group_miner import prove  # noqa: E402
from tests.unit.test_trajectory_parse import FakeRenderer  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
URL = "http://validator.test"
RANDOMNESS = "cd" * 32
REVISION_B = "e" * 40


def miner_state(rt, *, window=1, cooldown=(), prompt_range=(0, 1000), accepting=True, open_until=None,
                state=WindowState.OPEN, randomness=RANDOMNESS, revision=None):
    announcement = rt.announcement(window=1, randomness=WINDOW_BEACON)      # the runtime's one open window
    if revision is not None:
        announcement = {**announcement, "checkpoint": {**announcement["checkpoint"], "revision": revision}}
    bitmap, count = encode_cooldown_bitmap(set(cooldown), prompt_range)
    envs = {EPISODE: MinerEnvironmentState(prompt_range=prompt_range, cooldown_bitmap=bitmap, cooldown_count=count,
                                           accepting_submissions=accepting, admission_remaining=10)}
    return MinerState(state=state, window_n=window, anchor_block=window, window_opened_at=time.time() - 10,
                      submission_deadline_at=open_until if open_until is not None else time.time() + 3600,
                      environments=envs, randomness=randomness, service_policy=announcement)


# -- the window, the pick, the prompt ---------------------------------------------------------------------


def test_the_window_is_read_from_the_miner_state(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    state = miner_state(rt, cooldown={3, 5}, prompt_range=(0, 64), open_until=1234.5)
    view = em.episode_window(state)
    assert (view.window, view.randomness, view.open_until) == (1, RANDOMNESS, 1234.5)
    assert view.announcement == rt.announcement(window=1, randomness=WINDOW_BEACON)
    assert view.revision == view.announcement["checkpoint"]["revision"] and view.repo == "models/test"
    assert view.cooldowns[EPISODE] == {3, 5} and view.ranges[EPISODE] == (0, 64) and view.accepting[EPISODE]
    assert em.episode_window(miner_state(rt, state=WindowState.READY)) is None
    assert em.episode_window(miner_state(rt, randomness="")) is None
    assert em.episode_window(None) is None


def test_a_task_in_cooldown_or_already_taken_is_never_picked(tmp_path):
    view = em.episode_window(miner_state(episode_runtime(tmp_path / "rt"), cooldown=set(range(9)), prompt_range=(0, 10)))
    rng = random.Random(7)
    picks = {em.pick_episode_task(view, environments=[EPISODE], sizes={EPISODE: 1000}, taken=(), rng=rng)
             for _ in range(50)}
    assert picks == {(EPISODE, 9)}
    assert em.pick_episode_task(view, environments=[EPISODE], sizes={EPISODE: 1000}, taken={(EPISODE, 9)},
                                rng=rng) is None
    # The task source's size bounds the range too; an env that takes no group is skipped.
    view = em.episode_window(miner_state(episode_runtime(tmp_path / "rt2"), prompt_range=(0, 10)))
    assert {em.pick_episode_task(view, environments=[EPISODE], sizes={EPISODE: 2}, taken=(), rng=rng)[1]
            for _ in range(40)} == {0, 1}
    closed = em.episode_window(miner_state(episode_runtime(tmp_path / "rt3"), accepting=False))
    assert em.pick_episode_task(closed, environments=[EPISODE], sizes={EPISODE: 1000}, taken=(), rng=rng) is None


class IndexedSource:
    def __len__(self):
        return 50

    def prompt(self, index):
        return f"\n  Tâche n°{index} — écris « {index * 7} » dans /work/answer.txt\r\n\tfin\u200b \n"

    async def resolve(self, index):
        raise NotImplementedError


def make_indexed_env():
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    return SignedEpisodeEnvironment(EPISODE, IndexedSource())


def test_the_miners_task_prompt_is_the_validators_byte_for_byte(monkeypatch):
    import dataclasses
    from types import MappingProxyType

    from reliquary.environment import load_environment, registry
    from reliquary.validator.batcher import _render_environment_prompt
    from tests.unit.episode_v2_fixtures import episode_spec

    spec = dataclasses.replace(episode_spec(), factory_path="tests.unit.test_episode_mining:make_indexed_env")
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", MappingProxyType({**registry.ENVIRONMENT_SPECS, EPISODE: spec}))
    prompts = em.EpisodeTaskPrompts([EPISODE])
    validator_env = load_environment(EPISODE)          # what the validator's episode services load
    for index in (0, 1, 7, 49):
        mine = prompts(EPISODE, index)
        assert mine.encode("utf-8") == validator_env.source.prompt(index).encode("utf-8")
        assert mine.encode("utf-8") == _render_environment_prompt(validator_env, None, index).encode("utf-8")
    assert prompts.size(EPISODE) == 50
    with pytest.raises(ValueError, match="not a signed-episode"):
        em.EpisodeTaskPrompts([MATH])


def test_the_installed_env_package_must_be_the_orders_pin(tmp_path):
    policy = episode_runtime(tmp_path / "rt").contract.episode_policy(EPISODE)
    assert em.env_package_refusal(policy, identity_of=lambda package: policy.env_package, need_bridge=False) is None
    refusal = em.env_package_refusal(policy, identity_of=lambda package: package + "==9.9", need_bridge=False)
    assert "the order pins" in refusal

    def broken(package):
        raise RuntimeError("no such install")

    assert "cannot compute" in em.env_package_refusal(policy, identity_of=broken, need_bridge=False)


# -- the stack and one window end to end --------------------------------------------------------------------


class FakeCore:
    """vLLM's place: each turn's completion is scripted by its prompt; every request must carry its draw."""

    def __init__(self):
        self.script, self.draws, self.queue, self.released = {}, [], [], False

    def add(self, request_id, prompt_ids, max_tokens, draw=None):
        if draw is None:
            raise ValueError("a forced core takes every request with its draw")
        self.draws.append((tuple(prompt_ids), draw[FORCED_SEED_KEY]))
        self.queue.append((request_id, tuple(prompt_ids)))

    def has_unfinished(self):
        return bool(self.queue)

    def step(self):
        done = []
        while self.queue:
            request_id, prompt = self.queue.pop(0)
            completion = tuple(self.script[prompt])
            done.append(Finished(request_id, completion, tuple(-0.5 for _ in completion), "stop", ()))
        return done


class World:
    """The fakes behind ``EpisodeStackDeps`` and the validator's routes."""

    def __init__(self, tmp_path, *, rewards=None, max_model_len=4096, package_refusal=None):
        self.rt = episode_runtime(tmp_path / "rt")
        self.keys = episode_signers(tmp_path / "keys")
        self.rewards = rewards if rewards is not None else {seed: float(seed % 2) for seed in range(2 * M_ROLLOUTS)}
        self.cores, self.apps, self.servers, self.models, self.runners = [], [], [], [], []
        self.released_models, self.precommits, self.requests, self.downloads = [], [], [], []
        self.events = []
        self.max_model_len = max_model_len
        self.deps = em.EpisodeStackDeps(
            download=self.download, load_renderer=lambda directory, tools: FakeRenderer(),
            make_core=self.make_core, max_model_len=lambda core, default: self.max_model_len,
            release_core=self.release_core, load_proof_model=self.load_model, release_model=self.release_model,
            make_prover=lambda **kwargs: prove, toploc=lambda: SimpleNamespace(chunk_tokens=32, topk=128),
            sampling=lambda: SimpleNamespace(temperature=1.0, top_p=1.0, top_k=-1), serve=self.serve,
            make_runner=self.make_runner, submit=self.submit, sync_http=self.sync_http,
            async_http=self.async_http,
            package_refusal=package_refusal or (lambda policy: None))

    def download(self, repo, revision, cache_dir):
        self.downloads.append((repo, revision, cache_dir))
        return f"/ckpt/{revision[:8]}"

    def make_core(self, directory, **kwargs):
        self.events.append("core")
        core = FakeCore()
        core.kwargs, core.directory = kwargs, directory
        self.cores.append(core)
        return core

    def release_core(self, core):
        self.events.append("release_core")
        core.released = True

    def load_model(self, directory):
        self.events.append("model")
        model = SimpleNamespace(directory=directory)
        self.models.append(model)
        return model

    def release_model(self, model):
        self.events.append("release_model")
        self.released_models.append(model)

    async def serve(self, app, port):
        self.events.append("serve")
        self.apps.append(app)
        server = SimpleNamespace(should_exit=False)
        self.servers.append(server)
        serving = asyncio.get_running_loop().create_future()
        serving.set_result(None)
        return server, serving

    def make_runner(self, **kwargs):
        runner = Runner(self, kwargs)
        self.runners.append(runner)
        return runner

    def sync_http(self, url):
        def handle(request):
            body = json.loads(request.content)
            assert request.url.path == episode_precommit_path("/rl")
            assert verify_episode_precommit_signature(
                MINER.ss58_address, body["precommit"], at=body["at"], signature=body["signature"],
                validator_hotkey=VALIDATOR.ss58_address, path=episode_precommit_path("/rl"))
            precommit = EpisodePrecommit.from_dict(body["precommit"])
            self.precommits.append(precommit)
            return httpx.Response(200, json={"precommit_sha256": precommit.sha256, "created": True})

        return httpx.Client(base_url=url, transport=httpx.MockTransport(handle))

    def async_http(self):
        def handle(request):
            _, route, hotkey, window, root = request.url.path.split("/")
            assert route == "miner-verdicts" and hotkey == MINER.ss58_address
            for sent in self.requests:
                if sent.merkle_root == root and sent.window_start == int(window):
                    return httpx.Response(200, json={"status": "found", "verdict": {
                        "merkle_root": root, "accepted": True, "reason": "accepted", "is_final": True, "ts": 1.0}})
            return httpx.Response(404, json={"status": "not_found"})

        return httpx.AsyncClient(transport=httpx.MockTransport(handle))

    async def submit(self, url, request, *, client, wallet, randomness):
        assert url == URL and wallet is WALLET and randomness == RANDOMNESS
        self.requests.append(request)
        return SimpleNamespace(accepted=True, reason="submitted")


class Runner:
    """The harness's place: plays each seed's episode through the REAL generate app (forced engine)."""

    def __init__(self, world, kwargs):
        self.world, self.kwargs = world, kwargs
        self.released, self.submitted, self.entered = [], [], []

    async def __aenter__(self):
        self.entered.append("in")
        return self

    async def __aexit__(self, *exc):
        self.entered.append("out")

    async def run(self, seed, on_session=None):
        world, precommit = self.world, self.kwargs["precommit"]
        trace = f"trace-{seed}"
        on_session(trace)
        validator, machine = world.keys
        tokens, spans, transcript = play_episode(validator=validator, machine=machine, precommit=precommit,
                                                 seed=seed, session_id=f"s-{seed}", reward=world.rewards[seed])
        app = world.apps[-1]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url=self.kwargs["generate_url"]) as client:
            for start, end in spans:
                world.cores[-1].script[tuple(tokens[:start])] = tokens[start:end]
                response = await client.post("/inference/v1/generate", headers={SESSION_HEADER: trace},
                                             json={"token_ids": tokens[:start], "model": self.kwargs["model_name"]})
                assert response.status_code == 200, response.text

        async def release():
            self.released.append(seed)

        return EpisodeResult(trace, "", "agent_completed", True, world.rewards[seed], transcript=transcript,
                             release=release, submitted=lambda: self.submitted.append(seed))


WALLET = SimpleNamespace(hotkey=MINER)


def config(**overrides):
    values = dict(environments=(EPISODE,), validator_url=URL, validator_hotkey=VALIDATOR.ss58_address,
                  generate_port=8123, checkpoint_dir="/cache", groups_in_flight=1, harness_env={"A": "1"})
    values.update(overrides)
    return em.EpisodeMinerConfig(**values)


def test_one_window_end_to_end_precommit_sessions_choice_submit_verdict(tmp_path, monkeypatch):
    register_episode_env(monkeypatch)
    w = World(tmp_path)
    state = miner_state(w.rt, cooldown=set(range(0, 999)), prompt_range=(0, 1000))
    stop = asyncio.Event()

    async def sleep(seconds):
        if w.requests and all(len(r.submitted) + len(r.released) == 2 * M_ROLLOUTS for r in w.runners):
            stop.set()
        await asyncio.sleep(0.01)

    async def read_state():
        return state

    asyncio.run(asyncio.wait_for(em.run_episode_miner(
        config=config(), wallet=WALLET, stop=stop, deps=w.deps, task_prompts=em.EpisodeTaskPrompts([EPISODE]),
        read_state=read_state, sleep=sleep), timeout=60))

    # The checkpoint of the announcement, once; the engine caps of the order's policy.
    assert w.downloads == [("models/test", state.service_policy.checkpoint["revision"], "/cache")]
    (core,) = w.cores
    assert core.kwargs["max_total_tokens"] == 4096 and core.kwargs["stop_token_ids"] == sorted(FakeRenderer.stop_ids)
    # Precommit: the only task out of cooldown, signed for the validator's /rl route.
    (precommit,) = w.precommits
    assert (precommit.environment, precommit.task_index, precommit.window) == (EPISODE, 999, 1)
    (runner,) = w.runners
    kwargs = runner.kwargs
    assert kwargs["prompt"] == PROMPT_TEXT and kwargs["precommit"] == precommit
    assert kwargs["hotkey"] == MINER.ss58_address and kwargs["model_name"] == "models/test"
    assert kwargs["generate_url"] == "http://127.0.0.1:8123" and kwargs["harness_env"] == {"A": "1"}
    assert kwargs["open_until"] == state.submission_deadline_at and kwargs["renderer_model_dir"].startswith("/ckpt/")
    assert kwargs["sessions"].validator_hotkey == VALIDATOR.ss58_address and kwargs["sessions"].prefix == "/rl"
    assert kwargs["sign_binding"](b"x") and MINER.verify(b"x", bytes.fromhex(kwargs["sign_binding"](b"x")))
    # Every turn went through the forced engine with its seed's draw, from the session's model-token offset.
    pool = w.rt.seed_pool(environment=EPISODE, prompt_idx=999, window=1)
    seeds = sorted({draw["seed_index"] for _, draw in core.draws})
    assert seeds == list(range(pool.pool_seeds))
    assert all(draw["public_pool"] == pool.to_dict() for _, draw in core.draws)
    offsets = sorted({draw["base_offset"] for _, draw in core.draws})
    assert offsets[0] == 0 and len(offsets) == 2          # turn 2 starts after turn 1's model tokens
    # One group: the chosen M submitted after the admitted verdict; the others withdrawn.
    (request,) = w.requests
    assert request.prompt_idx == 999 and request.window_start == 1 and len(request.rollouts) == M_ROLLOUTS
    assert all(r.commit["beacon"]["randomness"] == RANDOMNESS for r in request.rollouts)
    assert sorted(runner.submitted + runner.released) == list(range(2 * M_ROLLOUTS))
    assert len(runner.submitted) == M_ROLLOUTS and runner.entered == ["in", "out"]
    # Shut down: endpoint, engine, models released.
    assert w.servers[0].should_exit and core.released and w.released_models == w.models


def test_the_stack_refuses_to_start_when_vllms_context_is_not_the_policys(tmp_path):
    w = World(tmp_path, max_model_len=8192)
    view = em.episode_window(miner_state(w.rt))
    with pytest.raises(ValueError, match="max_model_len 8192 != 4096"):
        asyncio.run(em.build_episode_stack(view, config=config(), wallet=WALLET, hotkey=MINER.ss58_address,
                                           sign_binding=lambda b: "", task_prompt=lambda e, t: "", deps=w.deps))
    # Nothing is served, no proof model loaded, and the engine's GPU memory is given back.
    assert w.events == ["core", "release_core"] and w.apps == []


def test_the_stack_refuses_a_non_episode_env_or_a_wrong_package_before_loading_anything(tmp_path):
    w = World(tmp_path, package_refusal=lambda policy: "x==1 is installed, the order pins y")
    view = em.episode_window(miner_state(w.rt))
    for cfg, message in ((config(), "the order pins"), (config(environments=(MATH,)), "not a signed-episode")):
        with pytest.raises(ValueError, match=message):
            asyncio.run(em.build_episode_stack(view, config=cfg, wallet=WALLET, hotkey=MINER.ss58_address,
                                               sign_binding=lambda b: "", task_prompt=lambda e, t: "", deps=w.deps))
    assert w.downloads == [] and w.events == []


def test_a_stack_that_cannot_start_ends_the_loop_with_its_refusal(tmp_path):
    rt = episode_runtime(tmp_path / "rt")

    async def read_state():
        return miner_state(rt)

    async def build(view):
        raise ValueError("the episode engine's caps are not the contract's episode policy")

    loop = em.EpisodeMiningLoop(environments=[EPISODE], read_state=read_state, build_stack=build,
                                sizes=lambda name: 1000)
    with pytest.raises(ValueError, match="caps"):
        asyncio.run(asyncio.wait_for(loop.run(), timeout=10))


# -- the loop: in-flight cap, cooldown, checkpoint change ---------------------------------------------------


class Blocking:
    """A stack whose groups run until released."""

    def __init__(self, revision):
        self.revision, self.started, self.cancelled, self.closed = revision, [], [], False
        self.gate = asyncio.Event()

    def miner_for(self, view):
        stack = self

        class Miner:
            async def mine_task(self, *, announcement, randomness, window, environment, task_index, open_until):
                stack.started.append((window, environment, task_index, randomness, open_until))
                try:
                    await stack.gate.wait()
                except asyncio.CancelledError:
                    stack.cancelled.append(task_index)
                    raise
                return "done"

        return Miner()

    async def close(self):
        self.closed = True


def loop_world(rt, states, **kwargs):
    built = []

    async def build(view):
        stack = Blocking(view.revision)
        built.append(stack)
        return stack

    async def read_state():
        return states[0]

    loop = em.EpisodeMiningLoop(environments=[EPISODE], read_state=read_state, build_stack=build,
                                sizes=lambda name: 1000, rng=random.Random(3), **kwargs)
    return loop, built


def test_at_most_two_groups_are_in_flight(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    states = [miner_state(rt)]
    loop, built = loop_world(rt, states)

    async def scenario():
        await loop.step()
        await loop.step()
        await loop.step()
        await asyncio.sleep(0.01)
        assert len(loop.in_flight) == 2 and len(built[0].started) == 2
        built[0].gate.set()
        await asyncio.sleep(0.05)
        assert loop.in_flight == []
        built[0].gate = asyncio.Event()
        await loop.step()
        await asyncio.sleep(0.01)
        assert len(loop.in_flight) == 2 and len(built[0].started) == 4
        await loop._cancel_inflight()

    asyncio.run(scenario())
    tasks = [started[2] for started in built[0].started]
    assert len(set(tasks)) == 4                     # a task is mined once per window
    for groups in (0, 3):
        with pytest.raises(ValueError, match="1..2"):
            em.EpisodeMiningLoop(environments=[EPISODE], read_state=None, build_stack=None, sizes=len,
                                 groups_in_flight=groups)
    with pytest.raises(ValueError, match="1..2"):
        config(groups_in_flight=3)
    from reliquary.validator.rl_sandbox_wiring import RL_GROUPS_IN_FLIGHT

    assert em.GROUPS_IN_FLIGHT == RL_GROUPS_IN_FLIGHT


def test_the_loop_never_mines_a_task_in_cooldown_and_retakes_none_in_the_window(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    states = [miner_state(rt, cooldown=set(range(10)) - {2, 6}, prompt_range=(0, 10))]
    loop, built = loop_world(rt, states)

    async def scenario():
        await loop.step()
        built[0].gate.set()
        await asyncio.sleep(0.05)
        await loop.step()                      # both free tasks taken: nothing more in this window
        first = [s[2] for s in built[0].started]
        states[0] = miner_state(rt, window=2, cooldown=set(range(10)) - {2, 6}, prompt_range=(0, 10))
        await loop.step()                      # a new window: the same tasks may be mined again
        await asyncio.sleep(0.05)
        return first

    first = asyncio.run(scenario())
    assert sorted(first) == [2, 6]
    assert sorted(s[2] for s in built[0].started[2:]) == [2, 6]
    assert {s[0] for s in built[0].started[2:]} == {2}
    assert all(s[3] == RANDOMNESS for s in built[0].started)


def test_no_group_starts_past_the_windows_end(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    loop, built = loop_world(rt, [miner_state(rt, open_until=time.time() - 1)])
    asyncio.run(loop.step())
    assert built[0].started == []


def changed_order(state):
    """The same checkpoint, another order (its env package pin moved)."""
    announcement = state.service_policy.model_dump()
    contract = json.loads(json.dumps(announcement["contract"]))
    contract["environments"][EPISODE]["episode"]["env_package"] = "moved-package==9.9.9"
    return state.model_copy(update={"service_policy": {**announcement, "contract": contract}})


def test_a_checkpoint_change_drains_the_groups_then_exits_75_and_builds_no_second_stack(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    states = [miner_state(rt)]
    loop, built = loop_world(rt, states, drain_s=5.0)

    async def scenario():
        await loop.step()
        await loop.step()                     # same checkpoint: nothing rebuilt
        await asyncio.sleep(0.01)
        assert len(built) == 1 and len(loop.in_flight) == 2
        states[0] = miner_state(rt, revision=REVISION_B)
        asyncio.get_running_loop().call_later(0.05, built[0].gate.set)     # the groups finish while draining
        with pytest.raises(em.CheckpointChanged) as raised:
            await asyncio.wait_for(loop.run(), timeout=10)
        return raised.value

    error = asyncio.run(scenario())
    assert error.exit_code == em.EXIT_CHECKPOINT_CHANGED == 75
    first, = built                            # no second stack in-process: the restarted process builds it
    assert first.closed and first.cancelled == [] and len(first.started) == 2
    assert [outcome for _, outcome in loop.outcomes] == ["done", "done"]   # drained, not cancelled


def test_a_drain_that_runs_out_cancels_the_groups(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    states = [miner_state(rt)]
    loop, built = loop_world(rt, states, drain_s=0.05)

    async def scenario():
        await loop.step()
        await asyncio.sleep(0.01)
        states[0] = miner_state(rt, revision=REVISION_B)
        with pytest.raises(em.CheckpointChanged):
            await loop.step()
        # Cancelled and awaited before the exit, not left for the event loop's shutdown.
        assert loop.in_flight == [] and sorted(built[0].cancelled) == sorted(s[2] for s in built[0].started)

    asyncio.run(scenario())
    assert len(built) == 1 and len(built[0].started) == 2


def test_a_new_order_on_the_same_checkpoint_restarts_and_its_package_pin_is_rechecked(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    states = [miner_state(rt)]
    pinned = em.episode_window(states[0])
    checked = []

    def preflight(view):
        checked.append(view.key)
        from reliquary.protocol.service_contract import ServiceContract

        installed = ServiceContract.from_dict(pinned.announcement["contract"]).episode_policy(EPISODE).env_package
        try:
            em._episode_policies(ServiceContract.from_dict(view.announcement["contract"]), [EPISODE],
                                 lambda policy: None if policy.env_package == installed else "pin moved")
        except ValueError as exc:
            return str(exc)
        return None

    loop, built = loop_world(rt, states, drain_s=1.0, preflight=preflight)

    async def scenario():
        await loop.step()
        await loop.step()
        states[0] = changed_order(states[0])
        view = em.episode_window(states[0])
        assert view.revision == pinned.revision and view.contract_sha != pinned.contract_sha
        with pytest.raises(em.CheckpointChanged):
            await loop.step()
        return view

    view = asyncio.run(scenario())
    assert checked == [view.key] and len(built) == 1


def test_a_build_that_fails_otherwise_than_by_refusal_is_retried_with_backoff(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    clock = SimpleNamespace(now=1000.0)
    attempts = []

    async def read_state():
        return miner_state(rt, cooldown=set(range(10)), prompt_range=(0, 10))

    async def build(view):
        attempts.append(clock.now)
        if len(attempts) < 3:
            raise OSError("hub unreachable")
        return Blocking(view.revision)

    loop = em.EpisodeMiningLoop(environments=[EPISODE], read_state=read_state, build_stack=build,
                                sizes=lambda name: 1000, clock=lambda: clock.now)

    async def scenario():
        assert await loop.step() is None                   # failed: logged, not raised
        assert await loop.step() is None                   # backing off: not retried at once
        clock.now += em.BUILD_BACKOFF_S
        assert await loop.step() is None                   # second failure: the backoff doubles
        clock.now += em.BUILD_BACKOFF_S
        assert await loop.step() is None
        clock.now += em.BUILD_BACKOFF_S
        assert await loop.step() is not None and loop.stack is not None

    asyncio.run(scenario())
    assert attempts == [1000.0, 1000.0 + em.BUILD_BACKOFF_S, 1000.0 + 3 * em.BUILD_BACKOFF_S]


def test_the_group_outcomes_kept_are_bounded_and_hold_no_traceback(tmp_path):
    rt = episode_runtime(tmp_path / "rt")
    loop, _ = loop_world(rt, [miner_state(rt)])

    async def boom():
        raise RuntimeError("sandbox gone")

    async def scenario():
        for index in range(em.OUTCOMES_KEPT + 7):
            task = asyncio.ensure_future(boom())
            loop._inflight[task] = (1, EPISODE, index)
            task.add_done_callback(loop._finished)
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)

    asyncio.run(scenario())
    assert len(loop.outcomes) == em.OUTCOMES_KEPT
    assert loop.outcomes[-1] == ((1, EPISODE, em.OUTCOMES_KEPT + 6), "RuntimeError('sandbox gone')")
    assert all(isinstance(outcome, str) for _, outcome in loop.outcomes)


def test_a_collection_window_too_short_for_the_pool_is_warned_about_at_startup(tmp_path, caplog):
    rt = episode_runtime(tmp_path / "rt")
    import dataclasses

    short = miner_state(rt, open_until=time.time() + 590)           # opened ~10 s ago
    view = dataclasses.replace(em.episode_window(short), opened_at=1000.0, open_until=1600.0)   # a 600 s window
    seeds = 2 * M_ROLLOUTS
    kwargs = dict(environments=[EPISODE], groups_in_flight=2, seconds_per_wave=300.0)
    assert em.short_window_warning(view, max_live=None, **kwargs) is None           # one wave of 300 s
    assert em.short_window_warning(view, max_live=seeds // 2, **kwargs) is None     # two waves: 600 s
    warning = em.short_window_warning(view, max_live=seeds // 2 - 1, **kwargs)      # three waves: 900 s
    assert warning and "900" in warning and f"{seeds} seeds" in warning
    loop, _ = loop_world(rt, [short], max_live=1, seconds_per_wave=300.0)
    with caplog.at_level("WARNING", logger=em.__name__):
        asyncio.run(loop.step())
    assert "COLLECTION WINDOW TOO SHORT" in caplog.text and loop.stack is not None   # a warning, not a refusal


def test_the_real_stack_closes_in_order_and_stops_the_engine_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    from reliquary.miner.corpus_generate_server import GenerateEngine

    register_episode_env(monkeypatch)
    w = World(tmp_path)
    stopped_on = []
    stop = GenerateEngine.stop

    def recorded_stop(self):
        stopped_on.append(threading.current_thread() is threading.main_thread())
        stop(self)

    monkeypatch.setattr(GenerateEngine, "stop", recorded_stop)
    view = em.episode_window(miner_state(w.rt))

    async def scenario():
        stack = await em.build_episode_stack(view, config=config(), wallet=WALLET, hotkey=MINER.ss58_address,
                                             sign_binding=lambda b: "", task_prompt=lambda e, t: "", deps=w.deps)
        await stack.close()

    asyncio.run(scenario())
    assert w.events == ["core", "serve", "model", "release_model", "release_core"]
    assert stopped_on == [False] and all(s.should_exit for s in w.servers)


# -- the CLI ------------------------------------------------------------------------------------------------


def test_the_cli_maps_its_flags_and_refuses_a_single_turn_env(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    register_episode_env(monkeypatch)
    seen = {}

    async def fake_run(*, config, wallet):
        seen["config"], seen["wallet"] = config, wallet

    monkeypatch.setattr(em, "run_episode_miner", fake_run)
    monkeypatch.setattr(bt, "Wallet", lambda **kwargs: SimpleNamespace(kwargs=kwargs))
    runner = CliRunner()
    result = runner.invoke(app, ["mine-episodes", "--episode-envs", EPISODE, "--validator-url", URL + "/",
                                 "--validator-hotkey", VALIDATOR.ss58_address, "--generate-port", "8100",
                                 "--checkpoint-dir", "/models", "--max-live", "5", "--groups-in-flight", "1",
                                 "--harness-env", "A=1", "--harness-env", "B=x=y", "--gpu-memory-utilization", "0.5",
                                 "--wallet-name", "w", "--hotkey", "h"])
    assert result.exit_code == 0, result.output
    cfg = seen["config"]
    assert cfg == em.EpisodeMinerConfig(environments=(EPISODE,), validator_url=URL,
                                        validator_hotkey=VALIDATOR.ss58_address, generate_port=8100,
                                        checkpoint_dir="/models", max_live=5, groups_in_flight=1,
                                        harness_env={"A": "1", "B": "x=y"}, gpu_memory_utilization=0.5)
    assert seen["wallet"].kwargs == {"name": "w", "hotkey": "h"}
    refused = runner.invoke(app, ["mine-episodes", "--episode-envs", MATH, "--validator-url", URL,
                                  "--validator-hotkey", VALIDATOR.ss58_address])
    assert refused.exit_code == 2 and "not a signed-episode" in refused.output
    too_many = runner.invoke(app, ["mine-episodes", "--episode-envs", EPISODE, "--validator-url", URL,
                                   "--validator-hotkey", VALIDATOR.ss58_address, "--groups-in-flight", "3"])
    assert too_many.exit_code == 2
    secrets = tmp_path / "harness.env"
    secrets.write_text("# harness secrets\nexport API_KEY=sk-file\nA=from-file\n\n")
    result = runner.invoke(app, ["mine-episodes", "--episode-envs", EPISODE, "--validator-url", URL,
                                 "--validator-hotkey", VALIDATOR.ss58_address, "--harness-env-file", str(secrets),
                                 "--harness-env", "A=1", "--seconds-per-wave", "120"])
    assert result.exit_code == 0, result.output
    assert seen["config"].harness_env == {"API_KEY": "sk-file", "A": "1"}       # the command line wins
    assert seen["config"].seconds_per_wave == 120.0

    async def restart(*, config, wallet):
        raise em.CheckpointChanged("checkpoint e announced")

    monkeypatch.setattr(em, "run_episode_miner", restart)
    restarted = runner.invoke(app, ["mine-episodes", "--episode-envs", EPISODE, "--validator-url", URL,
                                    "--validator-hotkey", VALIDATOR.ss58_address])
    assert restarted.exit_code == 75 and "restart" in restarted.output


def test_the_legacy_miner_paths_do_not_import_the_episode_miner():
    code = ("import sys, reliquary.miner.engine, reliquary.miner.submitter, reliquary.cli.main; "
            "assert 'reliquary.miner.episode_mining' not in sys.modules; "
            "assert 'reliquary.miner.episode_group_miner' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)
