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
