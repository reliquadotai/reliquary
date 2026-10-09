"""The miner's forced draw over an episode's model tokens (plan 2C, Task 13)."""
import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from reliquary.miner.corpus_generate_server import Finished, GenerateEngine, VllmTurnCore
from reliquary.miner.forced_draw import DrawBinding, ForcedDraws, forced_sampling_params
from reliquary.miner.vllm_generation import FORCED_SEED_KEY, ForcedSeedVLLMProcessor
from reliquary.protocol.seed_pool import SeedPool
from reliquary.validator.verifier import policy_token_positions
from tests.unit.episode_v2_fixtures import episode_contract, episode_pool

POOL = episode_pool(episode_contract())
SEED = 5


class Core:
    """Answers each request with ``length`` tokens and records what it was asked."""

    def __init__(self, length=3):
        self.added, self.queue, self.length = [], [], length

    def add(self, request_id, prompt_ids, max_tokens, draw=None):
        self.added.append((request_id, list(prompt_ids), max_tokens, draw))
        self.queue.append(request_id)

    def step(self):
        done = [Finished(r, tuple(range(100, 100 + self.length)), (0.0,) * self.length, "stop")
                for r in self.queue]
        self.queue = []
        return done

    def has_unfinished(self):
        return bool(self.queue)


class LegacyCore(Core):
    def add(self, request_id, prompt_ids, max_tokens):
        super().add(request_id, prompt_ids, max_tokens)


class HeldCore(Core):
    """Holds every request until ``release`` is set (a turn still running)."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def step(self):
        if not self.release.wait(0.01):
            return []
        return super().step()


def run(engine, session, prompt):
    return asyncio.run(engine.generate(session, prompt, None))


def test_a_turns_draw_starts_after_the_sessions_earlier_model_tokens():
    draws = ForcedDraws()
    draws.bind("trace-1", DrawBinding(POOL, SEED))
    core = Core()
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50, draws=draws)
    engine.start()
    try:
        first = run(engine, "trace-1", [1, 2, 3])
        completion = first["choices"][0]["token_ids"]
        run(engine, "trace-1", [1, 2, 3] + completion + [7, 7])        # a tool output: no position
        assert engine.model_tokens("trace-1") == 2 * core.length
    finally:
        engine.stop()
    (_, _, _, turn1), (_, _, _, turn2) = core.added
    assert turn1[FORCED_SEED_KEY]["base_offset"] == 0
    assert turn2[FORCED_SEED_KEY]["base_offset"] == core.length
    assert turn2[FORCED_SEED_KEY]["seed_index"] == SEED
    assert turn2[FORCED_SEED_KEY]["public_pool"] == POOL.to_dict()


def test_a_forced_engine_refuses_a_session_with_no_seed():
    core = Core()
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50, draws=ForcedDraws())
    engine.start()
    try:
        with pytest.raises(ValueError, match="forced draw"):
            run(engine, "unbound", [1, 2, 3])
    finally:
        engine.stop()
    assert core.added == [] and engine.take_session("unbound") is None


def _forced_engine(core, **kw):
    draws = ForcedDraws()
    draws.bind("t", DrawBinding(POOL, SEED))
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50, draws=draws, **kw)
    engine.start()
    return engine, draws


def test_a_forced_engine_refuses_a_second_turn_while_one_runs():
    """Two turns in flight would both start at the same model-token position."""
    core = HeldCore()
    engine, _ = _forced_engine(core)

    async def both():
        first = asyncio.ensure_future(engine.generate("t", [1, 2, 3], None))
        while not core.added:
            await asyncio.sleep(0.01)
        with pytest.raises(ValueError, match="already running"):
            await engine.generate("t", [1, 2, 3, 100, 101, 102, 7], None)
        core.release.set()
        await first

    try:
        asyncio.run(both())
    finally:
        engine.stop()
    assert len(core.added) == 1


def test_a_session_stays_busy_between_the_engine_resolving_its_turn_and_the_log():
    """The gap: the engine has resolved the turn, the handler has not logged it yet. A second request
    arriving there would read the offset without the first turn's tokens."""
    core = HeldCore()
    engine, _ = _forced_engine(core)

    async def gap():
        first = asyncio.ensure_future(engine.generate("t", [1, 2, 3], None))
        while not core.added:
            await asyncio.sleep(0.01)
        engine._resolve(Finished(core.added[0][0], (100, 101, 102), (0.0,) * 3, "stop"))
        second = asyncio.ensure_future(engine.generate("t", [1, 2, 3, 100, 101, 102, 7], None))
        with pytest.raises(ValueError, match="already running"):
            await asyncio.wait_for(second, timeout=5)
        await first

    try:
        asyncio.run(gap())
    finally:
        engine.stop()
    assert len(core.added) == 1 and engine.model_tokens("t") == 3


def test_a_session_that_lost_its_log_after_a_turn_is_refused_not_restarted_at_zero():
    core = Core()
    engine, draws = _forced_engine(core)
    try:
        run(engine, "t", [1, 2, 3])
        with engine._lock:
            del engine._sessions["t"]                                     # the log is gone, the binding is not
        with pytest.raises(ValueError, match="lost its log"):
            run(engine, "t", [1, 2, 3, 100, 101, 102, 7])
    finally:
        engine.stop()
    assert len(core.added) == 1


def test_a_taken_or_dropped_session_leaves_no_binding_behind():
    core = Core()
    engine, draws = _forced_engine(core)
    draws.bind("u", DrawBinding(POOL, SEED))
    try:
        run(engine, "t", [1, 2, 3])
        run(engine, "u", [1, 2, 3])
        assert engine.take_session("t") is not None
        engine.drop_session("u")
        assert draws.get("t") is None and draws.get("u") is None
        with pytest.raises(ValueError, match="forced draw"):
            run(engine, "t", [1, 2, 3])
    finally:
        engine.stop()


def test_a_session_whose_log_expired_loses_its_binding():
    now = [0.0]
    core = Core()
    engine, draws = _forced_engine(core, session_ttl_seconds=10.0, clock=lambda: now[0])
    draws.bind("other", DrawBinding(POOL, SEED))
    try:
        run(engine, "t", [1, 2, 3])
        now[0] = 100.0
        run(engine, "other", [1, 2, 3, 9])                                # sweeps the expired log of "t"
        assert draws.get("t") is None and draws.get("other") is not None
    finally:
        engine.stop()


def test_forced_mode_ignores_the_clients_max_tokens():
    core = Core()
    engine, _ = _forced_engine(core)
    try:
        asyncio.run(engine.generate("t", [1, 2, 3], 2))
    finally:
        engine.stop()
    assert core.added[0][2] == 50                                         # min(per_turn, total - prompt)


def test_a_failed_turn_leaves_the_offset_and_the_session_free():
    class Failing(Core):
        def step(self):
            done = [Finished(r, error="boom") for r in self.queue]
            self.queue = []
            return done

    core = Failing()
    engine, _ = _forced_engine(core)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run(engine, "t", [1, 2, 3])
        assert engine.model_tokens("t") == 0
        with pytest.raises(RuntimeError, match="boom"):                   # not "already running"
            run(engine, "t", [1, 2, 3])
    finally:
        engine.stop()
    assert [a[3][FORCED_SEED_KEY]["base_offset"] for a in core.added] == [0, 0]


def test_a_cancelled_turn_leaves_the_offset_and_the_session_free():
    core = HeldCore()
    engine, _ = _forced_engine(core)

    async def scenario():
        first = asyncio.ensure_future(engine.generate("t", [1, 2, 3], None))
        while not core.added:
            await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert engine.model_tokens("t") == 0
        core.release.set()
        await engine.generate("t", [1, 2, 3], None)

    try:
        asyncio.run(scenario())
    finally:
        engine.stop()
    assert [a[3][FORCED_SEED_KEY]["base_offset"] for a in core.added] == [0, 0]


def test_an_engine_without_draws_drives_a_three_argument_core_as_before():
    core = LegacyCore()
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50)
    engine.start()
    try:
        run(engine, "corpus-trace", [1, 2, 3])
    finally:
        engine.stop()
    assert core.added[0][3] is None


def test_a_binding_refuses_a_seed_outside_the_pool():
    with pytest.raises(ValueError):
        DrawBinding(POOL, POOL.pool_seeds)


def test_a_session_is_bound_once():
    draws = ForcedDraws()
    draws.bind("t", DrawBinding(POOL, 1))
    draws.bind("t", DrawBinding(POOL, 1))                                 # the same binding again
    with pytest.raises(ValueError, match="already bound"):
        draws.bind("t", DrawBinding(POOL, 2))
    draws.drop("t")
    assert draws.get("t") is None


def test_forced_sampling_takes_the_processors_pick_and_keeps_the_jobs_stops():
    base = {"n": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stop_token_ids": [2], "ignore_eos": True,
            "logprobs": 1, "detokenize": False}
    draw = DrawBinding(POOL, SEED).extra_args(7)
    params = forced_sampling_params(base, draw)
    assert params["temperature"] == 0.0 and params["top_p"] == 1.0 and params["top_k"] == -1
    assert params["stop_token_ids"] == [2] and params["logprobs"] == 1 and params["extra_args"] == draw
    assert base["temperature"] == 1.0


def test_the_processor_reads_the_episode_stream_at_model_token_positions(monkeypatch):
    """Two turns through ForcedSeedVLLMProcessor read u(seed, 0..n-1) over the concatenated model tokens,
    exactly the validator's ``seed_uniforms`` over the episode's policy positions."""
    read = []
    real = SeedPool.uniform

    def spy(self, seed_index, position):
        read.append((seed_index, position))
        return real(self, seed_index, position)

    monkeypatch.setattr(SeedPool, "uniform", spy)
    binding = DrawBinding(POOL, SEED)
    processor = ForcedSeedVLLMProcessor()
    logits = torch.zeros(1, 16)
    offset = 0
    reads = []
    for turn_tokens in (3, 2):
        params = SimpleNamespace(extra_args=binding.extra_args(offset))
        processor.update_state(SimpleNamespace(removed=(), added=[(0, params)], moved=()))
        read.clear()
        for _ in range(turn_tokens):
            processor.apply(logits)
        assert read == [(SEED, offset + j) for j in range(turn_tokens)]
        reads += read
        processor.update_state(SimpleNamespace(removed=(0,), added=(), moved=()))
        offset += turn_tokens
    # the validator's stream: u(seed, j) for j over the episode's policy positions (batcher.seed_uniforms)
    tokens = list(range(12))
    meta = {"episode": {"assistant_spans": [[3, 6], [8, 10]]}}
    positions = policy_token_positions(tokens, meta)
    assert reads == [(SEED, j) for j in range(len(positions))]
    assert [real(POOL, s, p) for s, p in reads] == [POOL.uniform(SEED, j) for j in range(len(positions))]


class _FakeVllmEngine:
    def __init__(self):
        self.requests = []

    def add_request(self, request_id, prompt, params):
        self.requests.append((request_id, params))


class _FakeLLM:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs
        self.llm_engine = _FakeVllmEngine()


def _core(forced, monkeypatch):
    """A VllmTurnCore built through ``__init__`` against a fake vLLM."""
    from contextlib import nullcontext

    import reliquary.miner.corpus_miner as corpus_miner
    import reliquary.miner.vllm_hidden_capture as hidden

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = _FakeLLM, dict
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: list(prompt_token_ids)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(corpus_miner, "_has_vision_encoder", lambda _dir: False)
    monkeypatch.setattr(hidden, "capture_hidden_states", lambda: nullcontext(object()))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    sampling = SimpleNamespace(temperature=0.7, top_p=0.9, top_k=0)
    return VllmTurnCore("ckpt", sampling=sampling, proof=SimpleNamespace(chunk_tokens=8, topk=4),
                        stop_token_ids=[2], max_total_tokens=64, forced=forced)


def test_the_vllm_core_sends_the_draw_only_when_built_forced(monkeypatch):
    draw = DrawBinding(POOL, SEED).extra_args(4)
    forced = _core(True, monkeypatch)
    assert _FakeLLM.kwargs["logits_processors"] == [ForcedSeedVLLMProcessor]
    forced.add("r", [1, 2], 8, draw=draw)
    (_, params), = forced._engine.requests
    assert params["extra_args"] == draw and params["temperature"] == 0.0 and params["max_tokens"] == 8
    with pytest.raises(ValueError, match="forced"):
        forced.add("r2", [1, 2], 8)                                       # a forced core never samples freely
    free = _core(False, monkeypatch)
    assert "logits_processors" not in _FakeLLM.kwargs
    with pytest.raises(ValueError, match="forced"):
        free.add("r3", [1, 2], 8, draw=draw)                              # no processor: the draw would be lost
    free.add("r4", [1, 2], 8)
    (_, params), = free._engine.requests
    assert "extra_args" not in params and params["temperature"] == 0.7
