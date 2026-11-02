"""The vLLM path must draw exactly what the protocol draws.

vLLM addresses requests by their slot in a persistent batch and recycles a slot
the moment a request leaves it, so these tests pin the bookkeeping as much as
the arithmetic: a stale slot would silently draw another rollout's stream.
"""

from types import SimpleNamespace

import pytest
import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp
from reliquary.miner.vllm_generation import (
    ForcedSeedVLLMProcessor,
    VLLMRolloutGenerator,
    forced_seed_extra_args,
)

RANDOMNESS = "ab" * 32
CHECKPOINT = "c" * 40
VOCAB = 512


def _params(rollout_index: int, prompt_idx: int = 7, base_offset: int = 0):
    return SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=prompt_idx, checkpoint_hash=CHECKPOINT,
        rollout_index=rollout_index, base_offset=base_offset,
    ))


def _update(added=(), removed=(), moved=()):
    return SimpleNamespace(batch_size=8, added=list(added),
                           removed=list(removed), moved=list(moved))


def _expected(logits: torch.Tensor, rollout_index: int, step: int,
              prompt_idx: int = 7) -> int:
    probs = warp(logits, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    return pick(probs, u_at(RANDOMNESS, prompt_idx, CHECKPOINT, rollout_index, step))


def test_the_processor_picks_what_the_protocol_picks():
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (row, _params(rollout_index=row * 5), [], []) for row in range(3)
    ]))

    forced = processor.apply(logits.clone())

    assert forced.shape == logits.shape
    for row in range(3):
        chosen = int(forced[row].argmax())
        assert chosen == _expected(logits[row], rollout_index=row * 5, step=0)
        assert forced[row, chosen] == 0.0
        assert torch.isinf(forced[row]).sum() == VOCAB - 1


def test_each_call_advances_that_slot_in_the_draw():
    torch.manual_seed(1)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(0, _params(rollout_index=2), [], [])]))

    for step in range(3):
        logits = torch.randn(1, VOCAB)
        chosen = int(processor.apply(logits.clone())[0].argmax())
        assert chosen == _expected(logits[0], rollout_index=2, step=step)


def test_a_request_resumes_at_the_offset_it_declares():
    torch.manual_seed(2)
    logits = torch.randn(1, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (0, _params(rollout_index=1, base_offset=9), [], []),
    ]))

    chosen = int(processor.apply(logits.clone())[0].argmax())

    assert chosen == _expected(logits[0], rollout_index=1, step=9)


def test_a_freed_slot_stops_drawing_and_a_new_one_takes_over():
    torch.manual_seed(3)
    logits = torch.randn(2, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[
        (0, _params(rollout_index=0), [], []),
        (1, _params(rollout_index=1), [], []),
    ]))
    processor.apply(logits.clone())

    processor.update_state(_update(removed=[0]))
    forced = processor.apply(logits.clone())
    # Slot 0 is nobody's: it must be left alone rather than drawn for.
    assert torch.equal(forced[0], torch.full((VOCAB,), float("-inf")))
    assert int(forced[1].argmax()) == _expected(logits[1], rollout_index=1, step=1)

    processor.update_state(_update(added=[(0, _params(rollout_index=4), [], [])]))
    forced = processor.apply(logits.clone())
    assert int(forced[0].argmax()) == _expected(logits[0], rollout_index=4, step=0)


def test_a_moved_request_keeps_its_place_in_the_draw():
    torch.manual_seed(4)
    logits = torch.randn(3, VOCAB)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(2, _params(rollout_index=6), [], [])]))
    processor.apply(logits.clone())

    processor.update_state(_update(moved=[(2, 0, "UNIDIRECTIONAL")]))
    forced = processor.apply(logits.clone())

    assert int(forced[0].argmax()) == _expected(logits[0], rollout_index=6, step=1)
    assert torch.equal(forced[2], torch.full((VOCAB,), float("-inf")))


def test_an_untracked_batch_is_returned_untouched():
    logits = torch.randn(2, VOCAB)
    assert torch.equal(ForcedSeedVLLMProcessor().apply(logits.clone()), logits)


class _FakeParams(SimpleNamespace):
    """vLLM's SamplingParams, reduced to what the generator sets on it."""


class _FakeEngine:
    """Enough of vLLM's LLM to check what the generator asks for and returns."""

    def __init__(self, completions):
        self.completions = completions
        self.seen = None

    def generate(self, requests, sampling, use_tqdm=False):
        self.seen = (requests, sampling)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=list(tokens), finish_reason="stop", stop_reason=None,
            )])
            for tokens in self.completions
        ]


def test_the_generator_truncates_at_the_first_stop_token():
    engine = _FakeEngine([[11, 12, 99, 13], [21, 22, 23]])
    generator = VLLMRolloutGenerator(
        "model", engine=engine, sampling_params_class=_FakeParams)

    completions = generator.generate(
        [1, 2, 3], randomness=RANDOMNESS, prompt_idx=7, checkpoint_hash=CHECKPOINT,
        rollouts=2, max_new_tokens=64, eos_ids=[99],
    )

    assert completions == [[11, 12, 99], [21, 22, 23]]


def test_every_rollout_asks_for_its_own_stream():
    engine = _FakeEngine([[1], [2], [3]])
    generator = VLLMRolloutGenerator(
        "model", engine=engine, sampling_params_class=_FakeParams)

    generator.generate(
        [5, 6], randomness=RANDOMNESS, prompt_idx=42, checkpoint_hash=CHECKPOINT,
        rollouts=3, max_new_tokens=8, eos_ids=[99],
    )

    requests, sampling = engine.seen
    assert [r["prompt_token_ids"] for r in requests] == [[5, 6]] * 3
    forced = [p.extra_args["forced_seed"] for p in sampling]
    assert [f["rollout_index"] for f in forced] == [0, 1, 2]
    assert {f["prompt_idx"] for f in forced} == {42}
    assert all(p.temperature == 0.0 for p in sampling)
    assert all(p.max_tokens == 8 for p in sampling)


def test_the_engine_is_only_built_when_none_is_supplied():
    with pytest.raises(ImportError):
        VLLMRolloutGenerator("model")


# --------------------------------------------------------------------------
# The runner-keyed draw. A per-slot call counter restarts when vLLM preempts a
# request and re-adds it (measured: 12-15 % of honest rollouts desynced on an
# H100, 30 % on an RTX 5090), so the engine path reads the position from the
# model runner's own input batch, keyed by request id, at sampling time.

import numpy as np  # noqa: E402

from reliquary.miner import vllm_generation  # noqa: E402
from reliquary.miner.vllm_generation import (  # noqa: E402
    RunnerForcedSeedVLLMProcessor,
    install_runner_forced_draw,
    require_runner_forced_draw,
)

PROMPT_LEN = 5


class _Runner:
    """vLLM 0.30's GPUModelRunner, reduced to what the forced draw reads.

    ``_sample`` hands the logits to the processor and takes the argmax, as the
    engine does for a greedy request whose logits the processor one-hot.
    """

    processor = None

    def __init__(self, capacity: int = 8) -> None:
        self.requests: dict[str, SimpleNamespace] = {}
        self.input_batch = SimpleNamespace(
            req_ids=[], num_reqs=0,
            num_tokens_no_spec=np.zeros(capacity, dtype=np.int32),
            num_prompt_tokens=np.zeros(capacity, dtype=np.int32),
        )

    def _sample(self, logits, spec_decode_metadata):
        return self.processor.apply(logits.clone()).argmax(-1)


def _runner_class():
    """A fresh runner class per test: the draw patches the class in place."""
    return type("GPUModelRunner", (_Runner,), {})


class _Engine:
    """Drives a fake runner + the runner-keyed processor the way vLLM does:
    batch updates by slot, input-batch rows keyed by request id, and
    ``num_tokens_no_spec`` bumped once per kept sample (async scheduling bumps
    it with a placeholder before the token is known)."""

    def __init__(self, monkeypatch, capacity: int = 8) -> None:
        cls = _runner_class()
        monkeypatch.setattr(vllm_generation, "_runner_class", lambda: cls)
        self.processor = RunnerForcedSeedVLLMProcessor()
        cls.processor = self.processor
        self.runner = cls(capacity)
        self.batch = self.runner.input_batch

    def add(self, req_id: str, row: int, params, *, generated: int = 0) -> None:
        self.runner.requests[req_id] = SimpleNamespace(sampling_params=params)
        while len(self.batch.req_ids) <= row:
            self.batch.req_ids.append(None)
        self.batch.req_ids[row] = req_id
        self.batch.num_reqs = sum(1 for r in self.batch.req_ids if r is not None)
        self.batch.num_prompt_tokens[row] = PROMPT_LEN
        self.batch.num_tokens_no_spec[row] = PROMPT_LEN + generated
        self.processor.update_state(_update(added=[(row, params, [], [])]))

    def remove(self, row: int) -> None:
        self.batch.req_ids[row] = None
        self.batch.num_reqs -= 1
        self.processor.update_state(_update(removed=[row]))

    def move(self, source: int, destination: int) -> None:
        ids, nt, npt = self.batch.req_ids, self.batch.num_tokens_no_spec, self.batch.num_prompt_tokens
        ids[destination], ids[source] = ids[source], None
        nt[destination], npt[destination] = nt[source], npt[source]
        self.processor.update_state(_update(moved=[(source, destination, "UNIDIRECTIONAL")]))

    def swap(self, a: int, b: int) -> None:
        ids, nt, npt = self.batch.req_ids, self.batch.num_tokens_no_spec, self.batch.num_prompt_tokens
        ids[a], ids[b] = ids[b], ids[a]
        nt[a], nt[b] = nt[b], nt[a]
        npt[a], npt[b] = npt[b], npt[a]
        self.processor.update_state(_update(moved=[(a, b, "SWAP")]))

    def step(self, logits, *, keep=True, spec=None):
        picks = self.runner._sample(logits, spec)
        if keep:  # _bookkeeping_sync: one more token (or async placeholder) per row
            for row, req_id in enumerate(self.batch.req_ids[: logits.shape[0]]):
                if req_id is not None:
                    self.batch.num_tokens_no_spec[row] += 1
        return picks

    def position(self, row: int) -> int:
        return int(self.batch.num_tokens_no_spec[row] - self.batch.num_prompt_tokens[row])


def test_the_runner_draw_picks_what_the_protocol_picks(monkeypatch):
    torch.manual_seed(10)
    engine = _Engine(monkeypatch)
    for row in range(3):
        engine.add(f"r{row}", row, _params(rollout_index=row * 5))

    for step in range(4):
        logits = torch.randn(3, VOCAB)
        picks = engine.step(logits)
        for row in range(3):
            assert int(picks[row]) == _expected(logits[row], rollout_index=row * 5, step=step)


def test_a_preempted_request_resumes_at_the_position_it_reached(monkeypatch):
    """The bug: vLLM's recompute preemption removes a request and re-adds it with
    its output so far; a call counter restarts at base_offset there."""
    torch.manual_seed(11)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=3))
    engine.add("b", 1, _params(rollout_index=4))
    for _ in range(3):
        engine.step(torch.randn(2, VOCAB))

    engine.remove(0)                                     # KV pressure: "a" preempted
    engine.move(1, 0)                                    # condense: "b" fills the hole
    engine.step(torch.randn(1, VOCAB))                   # "b" keeps going alone
    params_a = engine.runner.requests["a"].sampling_params
    engine.add("a", 1, params_a, generated=3)            # re-added in another slot, 3 tokens out

    logits = torch.randn(2, VOCAB)
    picks = engine.step(logits)
    assert int(picks[1]) == _expected(logits[1], rollout_index=3, step=3)
    assert int(picks[0]) == _expected(logits[0], rollout_index=4, step=4)
    assert engine.position(1) == 4


def test_a_discarded_sample_does_not_advance_the_draw(monkeypatch):
    """A partial chunked-prefill step samples a row whose token vLLM throws away:
    it does not count, so the next kept sample is still position 0."""
    torch.manual_seed(12)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=2))

    engine.step(torch.randn(1, VOCAB), keep=False)
    logits = torch.randn(1, VOCAB)
    picks = engine.step(logits)

    assert int(picks[0]) == _expected(logits[0], rollout_index=2, step=0)


def test_reordered_rows_keep_their_own_streams(monkeypatch):
    torch.manual_seed(13)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=1))
    engine.add("b", 1, _params(rollout_index=2))
    engine.step(torch.randn(2, VOCAB))
    engine.step(torch.randn(2, VOCAB))
    engine.add("c", 2, _params(rollout_index=9))
    engine.swap(0, 2)                                    # condense / reorder

    logits = torch.randn(3, VOCAB)
    picks = engine.step(logits)

    assert int(picks[0]) == _expected(logits[0], rollout_index=9, step=0)
    assert int(picks[1]) == _expected(logits[1], rollout_index=2, step=2)
    assert int(picks[2]) == _expected(logits[2], rollout_index=1, step=2)


def test_an_async_placeholder_counts_as_a_drawn_position(monkeypatch):
    """Async scheduling bumps num_tokens_no_spec with a -1 placeholder before the
    previous token is known; the draw reads that count, not the live list."""
    torch.manual_seed(14)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=5), generated=6)   # 6 out, last one a placeholder

    logits = torch.randn(1, VOCAB)
    picks = engine.step(logits)

    assert int(picks[0]) == _expected(logits[0], rollout_index=5, step=6)


def test_the_runner_draw_adds_the_declared_base_offset(monkeypatch):
    torch.manual_seed(15)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=1, base_offset=9), generated=2)

    logits = torch.randn(1, VOCAB)
    picks = engine.step(logits)

    assert int(picks[0]) == _expected(logits[0], rollout_index=1, step=11)


def test_unforced_rows_are_left_to_the_engine(monkeypatch):
    torch.manual_seed(16)
    engine = _Engine(monkeypatch)
    engine.add("free", 0, SimpleNamespace(extra_args=None))
    engine.add("a", 1, _params(rollout_index=0))

    logits = torch.randn(2, VOCAB)
    picks = engine.step(logits)

    assert int(picks[0]) == int(logits[0].argmax())
    assert int(picks[1]) == _expected(logits[1], rollout_index=0, step=0)


def test_the_runner_draw_reads_a_public_pool_by_seed(monkeypatch):
    from tests.unit.test_public_pool_sampling import _pool

    pool = _pool()
    torch.manual_seed(17)
    engine = _Engine(monkeypatch)
    params = SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=pool.randomness, prompt_idx=pool.prompt_idx, checkpoint_hash=pool.checkpoint_hash,
        rollout_index=0, base_offset=1, seed_pool=pool, seed_index=3))
    engine.add("a", 0, params, generated=2)

    logits = torch.randn(1, VOCAB)
    picks = engine.step(logits)

    probs = warp(logits[0], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    assert int(picks[0]) == pick(probs, pool.uniform(3, 3))


def test_speculative_decoding_is_refused_at_sampling(monkeypatch):
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=0))

    with pytest.raises(RuntimeError, match="speculative"):
        engine.step(torch.randn(1, VOCAB), spec=object())


def test_speculative_decoding_is_refused_at_construction(monkeypatch):
    cls = _runner_class()
    monkeypatch.setattr(vllm_generation, "_runner_class", lambda: cls)
    config = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=2))

    with pytest.raises(RuntimeError, match="speculative"):
        RunnerForcedSeedVLLMProcessor(config, "cpu", False)


def test_forced_requests_sampled_outside_the_patched_runner_are_refused(monkeypatch):
    """If the patch is not on the path (another runner, a vLLM upgrade), the
    processor must not hand back raw logits for a greedy argmax."""
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=0))

    with pytest.raises(RuntimeError, match="runner"):
        engine.processor.apply(torch.randn(1, VOCAB))


def test_the_runner_patch_installs_once_and_composes_with_hidden_capture(monkeypatch):
    from reliquary.miner.vllm_hidden_capture import capture_hidden_states

    cls = _runner_class()
    cls.execute_model = lambda self, scheduler_output: self._model_forward()
    cls._model_forward = lambda self: torch.zeros(1, 4)
    install_runner_forced_draw(cls)
    patched = cls._sample
    install_runner_forced_draw(cls)
    assert cls._sample is patched

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(vllm_generation, "_runner_class", lambda: cls)
    torch.manual_seed(18)
    processor = RunnerForcedSeedVLLMProcessor()
    cls.processor = processor
    runner = cls()
    params = _params(rollout_index=4)
    runner.requests["a"] = SimpleNamespace(sampling_params=params)
    runner.input_batch.req_ids, runner.input_batch.num_reqs = ["a"], 1
    runner.input_batch.num_prompt_tokens[0] = runner.input_batch.num_tokens_no_spec[0] = PROMPT_LEN
    processor.update_state(_update(added=[(0, params, [], [])]))
    with capture_hidden_states(cls) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"a": 1}))
        logits = torch.randn(1, VOCAB)
        assert int(runner._sample(logits, None)[0]) == _expected(logits[0], rollout_index=4, step=0)
    assert capture.ids() == ["a"]
    assert cls._sample is patched


def _fake_vllm(monkeypatch, version="0.30.0", sample=None):
    from types import ModuleType
    import sys

    cls = _runner_class()
    if sample is not None:
        cls._sample = sample
    vllm = ModuleType("vllm")
    vllm.__version__ = version
    runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
    runner_module.GPUModelRunner = cls
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", runner_module)
    return vllm


def test_the_pinned_vllm_is_accepted(monkeypatch):
    _fake_vllm(monkeypatch, "0.30.0")
    require_runner_forced_draw()


@pytest.mark.parametrize("version", ["0.29.1", "0.31.0", "1.0.0"])
def test_another_vllm_is_refused(monkeypatch, version):
    _fake_vllm(monkeypatch, version)
    with pytest.raises(RuntimeError, match="vLLM"):
        require_runner_forced_draw()


def test_a_runner_without_the_sampling_hook_is_refused(monkeypatch):
    _fake_vllm(monkeypatch, sample=lambda self, logits: None)
    with pytest.raises(RuntimeError, match="_sample"):
        require_runner_forced_draw()


def test_a_runner_batch_without_token_counts_is_refused(monkeypatch):
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _params(rollout_index=0))
    del engine.batch.num_tokens_no_spec

    with pytest.raises(RuntimeError, match="num_tokens_no_spec"):
        engine.step(torch.randn(1, VOCAB))


def test_the_generator_builds_its_engine_with_the_runner_draw(monkeypatch):
    vllm = _fake_vllm(monkeypatch)
    built = {}

    class _LLM:
        def __init__(self, **kwargs):
            built.update(kwargs)

    vllm.LLM = _LLM
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    VLLMRolloutGenerator("model")

    assert built["logits_processors"] == [RunnerForcedSeedVLLMProcessor]
    import os
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_the_generator_refuses_an_unpinned_vllm_before_loading(monkeypatch):
    vllm = _fake_vllm(monkeypatch, "0.31.0")
    vllm.LLM = lambda **kwargs: pytest.fail("the engine must not be built")

    with pytest.raises(RuntimeError, match="vLLM"):
        VLLMRolloutGenerator("model")


def test_the_generator_refuses_the_v2_runner(monkeypatch):
    vllm = _fake_vllm(monkeypatch)
    vllm.LLM = lambda **kwargs: pytest.fail("the engine must not be built")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    with pytest.raises(RuntimeError, match="V2"):
        VLLMRolloutGenerator("model")
