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


def test_a_forced_engine_refuses_a_second_turn_while_one_runs():
    """Two turns in flight would both start at the same model-token position."""
    draws = ForcedDraws()
    draws.bind("t", DrawBinding(POOL, SEED))
    core = HeldCore()
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50, draws=draws)
    engine.start()

    async def both():
        first = asyncio.ensure_future(engine.generate("t", [1, 2, 3], None))
        while not core.added:
            await asyncio.sleep(0.01)
        second = asyncio.ensure_future(engine.generate("t", [1, 2, 3, 100, 101, 102, 7], None))
        await asyncio.wait([second], timeout=2)
        core.release.set()
        await first
        with pytest.raises(ValueError, match="already running"):
            await asyncio.wait_for(second, timeout=5)

    try:
        asyncio.run(both())
    finally:
        engine.stop()
    assert len(core.added) == 1


def test_an_engine_without_draws_drives_a_three_argument_core_as_before():
    core = LegacyCore()
    engine = GenerateEngine(core, max_total_tokens=1000, max_tokens_per_turn=50)
    engine.start()
    try:
        run(engine, "corpus-trace", [1, 2, 3])
    finally:
        engine.stop()
    assert core.added[0][3] is None


def test_the_binding_draws_the_validators_uniforms():
    binding = DrawBinding(POOL, SEED)
    assert binding.uniforms(0, 4) == [POOL.uniform(SEED, j) for j in range(4)]
    assert binding.uniforms(3, 2) == [POOL.uniform(SEED, 3), POOL.uniform(SEED, 4)]
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
    for turn_tokens in (3, 2):
        params = SimpleNamespace(extra_args=binding.extra_args(offset))
        processor.update_state(SimpleNamespace(removed=(), added=[(0, params)], moved=()))
        read.clear()
        for _ in range(turn_tokens):
            processor.apply(logits)
        assert read == [(SEED, offset + j) for j in range(turn_tokens)]
        processor.update_state(SimpleNamespace(removed=(0,), added=(), moved=()))
        offset += turn_tokens


class _FakeVllmEngine:
    def __init__(self):
        self.requests = []

    def add_request(self, request_id, prompt, params):
        self.requests.append((request_id, params))


def _core(forced):
    core = object.__new__(VllmTurnCore)
    core._engine = _FakeVllmEngine()
    core._params = dict
    core._sampling = dict(n=1, temperature=0.7, top_p=0.9, top_k=-1, stop_token_ids=[2], ignore_eos=True,
                          logprobs=1, detokenize=False)
    core._prompt_len = {}
    core._forced = forced
    return core


def test_the_vllm_core_sends_the_draw_only_when_built_forced(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: list(prompt_token_ids)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    draw = DrawBinding(POOL, SEED).extra_args(4)
    forced = _core(True)
    forced.add("r", [1, 2], 8, draw=draw)
    (_, params), = forced._engine.requests
    assert params["extra_args"] == draw and params["temperature"] == 0.0 and params["max_tokens"] == 8
    with pytest.raises(ValueError, match="forced"):
        forced.add("r2", [1, 2], 8)                                       # a forced core never samples freely
    free = _core(False)
    with pytest.raises(ValueError, match="forced"):
        free.add("r3", [1, 2], 8, draw=draw)                              # no processor: the draw would be lost
    free.add("r4", [1, 2], 8)
    (_, params), = free._engine.requests
    assert "extra_args" not in params and params["temperature"] == 0.7
