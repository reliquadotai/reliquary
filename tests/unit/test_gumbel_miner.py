"""The miner draws what the profile prescribes: the Gumbel-max pick on the runner-keyed vLLM path, the
slot-counter fakes, the HF processor and the corpus/episode core; an fp32 LM head when asked."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.miner import vllm_generation
from reliquary.miner.forced_draw import DrawBinding
from reliquary.miner.vllm_generation import (
    FORCED_SEED_KEY,
    ForcedSeedVLLMProcessor,
    VLLMRolloutGenerator,
    forced_seed_extra_args,
)
from tests.unit.episode_v2_fixtures import episode_contract, episode_pool
from tests.unit.test_vllm_forced_seed import (
    CHECKPOINT,
    RANDOMNESS,
    VOCAB,
    _Engine,
    _FakeEngine,
    _FakeParams,
    _fake_vllm,
    _update,
)

GUMBEL = fs.DRAW_GUMBEL_MAX


def _gumbel_params(rollout_index, *, base_offset=0, prompt_idx=7):
    return SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=prompt_idx, checkpoint_hash=CHECKPOINT,
        rollout_index=rollout_index, base_offset=base_offset, draw=GUMBEL,
    ))


def _gumbel_expected(row_logits, rollout_index, position, prompt_idx=7):
    u = fs.u_at(RANDOMNESS, prompt_idx, CHECKPOINT, rollout_index, position)
    return int(fs.pick_rows(row_logits.reshape(1, -1), [u], draw=GUMBEL)[0])


def test_the_inverse_cdf_request_is_unchanged_on_the_wire():
    args = forced_seed_extra_args(randomness="r", prompt_idx=1, checkpoint_hash="c",
                                  rollout_index=2, base_offset=3, draw=fs.DRAW_INVERSE_CDF)
    assert args == {FORCED_SEED_KEY: {"randomness": "r", "prompt_idx": 1, "checkpoint_hash": "c",
                                      "rollout_index": 2, "base_offset": 3}}
    gumbel = forced_seed_extra_args(randomness="r", prompt_idx=1, checkpoint_hash="c",
                                    rollout_index=2, base_offset=3, draw=GUMBEL)
    assert gumbel[FORCED_SEED_KEY] == {**args[FORCED_SEED_KEY], "draw": GUMBEL}


def test_the_request_draw_defaults_to_the_active_profile(monkeypatch):
    from reliquary import constants

    monkeypatch.setattr(constants, "FORCED_DRAW", GUMBEL)
    args = forced_seed_extra_args(randomness="r", prompt_idx=1, checkpoint_hash="c", rollout_index=0)
    assert args[FORCED_SEED_KEY]["draw"] == GUMBEL
    pool = episode_pool(episode_contract())
    assert DrawBinding(pool, 5).extra_args(4)[FORCED_SEED_KEY]["draw"] == GUMBEL
    monkeypatch.setattr(constants, "FORCED_DRAW", fs.DRAW_INVERSE_CDF)
    assert "draw" not in DrawBinding(pool, 5).extra_args(4)[FORCED_SEED_KEY]


def test_an_unknown_draw_is_refused_when_the_request_arrives():
    params = SimpleNamespace(extra_args={FORCED_SEED_KEY: {
        "randomness": "r", "prompt_idx": 1, "checkpoint_hash": "c", "rollout_index": 0,
        "base_offset": 0, "draw": "top1"}})
    with pytest.raises(ValueError, match="draw"):
        ForcedSeedVLLMProcessor().update_state(_update(added=[(0, params, [], [])]))


def test_the_runner_draw_picks_the_gumbel_pick(monkeypatch):
    torch.manual_seed(20)
    engine = _Engine(monkeypatch)
    for row in range(3):
        engine.add(f"r{row}", row, _gumbel_params(rollout_index=row * 3))
    for step in range(5):
        logits = torch.randn(3, VOCAB)
        picks = engine.step(logits)
        for row in range(3):
            assert int(picks[row]) == _gumbel_expected(logits[row], row * 3, step)


def test_a_preempted_gumbel_request_resumes_at_its_position(monkeypatch):
    torch.manual_seed(21)
    engine = _Engine(monkeypatch)
    params = _gumbel_params(rollout_index=4, base_offset=10)
    engine.add("a", 0, params)
    for _ in range(3):
        engine.step(torch.randn(1, VOCAB))
    engine.remove(0)
    engine.add("a", 1, params, generated=3)
    logits = torch.randn(2, VOCAB)
    picks = engine.step(logits)
    assert int(picks[1]) == _gumbel_expected(logits[1], 4, 13)


def test_mixed_draws_in_one_batch_keep_their_own_pick(monkeypatch):
    torch.manual_seed(22)
    engine = _Engine(monkeypatch)
    engine.add("g", 0, _gumbel_params(rollout_index=1))
    cdf = SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=7, checkpoint_hash=CHECKPOINT, rollout_index=1,
        draw=fs.DRAW_INVERSE_CDF))
    engine.add("c", 1, cdf)
    logits = torch.randn(2, VOCAB)
    picks = engine.step(logits)
    assert int(picks[0]) == _gumbel_expected(logits[0], 1, 0)
    u = fs.u_at(RANDOMNESS, 7, CHECKPOINT, 1, 0)
    from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
    assert int(picks[1]) == fs.pick(fs.warp(logits[1], T_PROTO, TOP_K_PROTO, TOP_P_PROTO), u)


def test_the_slot_counter_processor_draws_gumbel_too():
    torch.manual_seed(23)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(0, _gumbel_params(rollout_index=2), [], [])]))
    for step in range(3):
        logits = torch.randn(1, VOCAB)
        assert int(processor.apply(logits.clone()).argmax()) == _gumbel_expected(logits[0], 2, step)


def test_what_the_miner_draws_the_validator_accepts(monkeypatch):
    """Miner (runner-keyed processor) and validator (seed diagnostics) on the same logits."""
    torch.manual_seed(24)
    engine = _Engine(monkeypatch)
    engine.add("a", 0, _gumbel_params(rollout_index=6))
    rows, tokens = [], []
    for _ in range(60):
        logits = torch.randn(1, VOCAB) * 0.7
        rows.append(logits[0])
        tokens.append(int(engine.step(logits)[0]))
    us = [fs.u_at(RANDOMNESS, 7, CHECKPOINT, 6, t) for t in range(60)]
    d = fs.seed_consistency_diagnostics(torch.stack(rows), tokens, us, t=1.0, top_k=0, top_p=1.0,
                                        stochastic_threshold=0.99, boundary_epsilon=0.0, draw=GUMBEL)
    assert d.n_stochastic == 60 and d.n_exact_match == 60 and d.n_hard_mismatch == 0


def test_the_public_pool_stream_feeds_the_gumbel_key(monkeypatch):
    pool = episode_pool(episode_contract())
    torch.manual_seed(25)
    engine = _Engine(monkeypatch)
    binding = DrawBinding(pool, 5)
    monkeypatch.setattr("reliquary.constants.FORCED_DRAW", GUMBEL)
    engine.add("e", 0, SimpleNamespace(extra_args=binding.extra_args(7)))
    logits = torch.randn(1, VOCAB)
    pick = int(engine.step(logits)[0])
    assert pick == int(fs.pick_rows(logits, [pool.uniform(5, 7)], draw=GUMBEL)[0])


# ---------------------------------------------------------------- generator and fp32 head

def test_the_generator_asks_for_its_draw():
    engine = _FakeEngine([[1], [2]])
    generator = VLLMRolloutGenerator("model", engine=engine, sampling_params_class=_FakeParams,
                                     draw=GUMBEL)
    generator.generate([5], randomness=RANDOMNESS, prompt_idx=1, checkpoint_hash=CHECKPOINT,
                       rollouts=2, max_new_tokens=4, eos_ids=[9])
    _, sampling = engine.seen
    assert all(p.extra_args[FORCED_SEED_KEY]["draw"] == GUMBEL for p in sampling)


def _built_engine(monkeypatch, **kw):
    vllm = _fake_vllm(monkeypatch)
    built = {}

    class _LLM:
        def __init__(self, **kwargs):
            built.update(kwargs)

    vllm.LLM = _LLM
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    VLLMRolloutGenerator("model", **kw)
    return built


def test_the_fp32_head_is_asked_of_vllm_only_when_set(monkeypatch):
    assert "hf_overrides" not in _built_engine(monkeypatch, lm_head_fp32=False)
    assert _built_engine(monkeypatch, lm_head_fp32=True)["hf_overrides"] == {"head_dtype": "float32"}


def test_the_fp32_head_defaults_to_the_active_profile(monkeypatch):
    monkeypatch.setattr("reliquary.constants.LM_HEAD_FP32", True)
    assert _built_engine(monkeypatch)["hf_overrides"] == {"head_dtype": "float32"}


def test_the_forced_corpus_core_takes_the_fp32_head(monkeypatch):
    from tests.unit.test_forced_draw import _FakeLLM, _core

    monkeypatch.setattr("reliquary.constants.LM_HEAD_FP32", True)
    _core(True, monkeypatch)
    assert _FakeLLM.kwargs["hf_overrides"] == {"head_dtype": "float32"}
    _core(False, monkeypatch)  # free sampling draws nothing: the model's own head
    assert "hf_overrides" not in _FakeLLM.kwargs


def test_the_hf_processor_draws_gumbel():
    from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor

    torch.manual_seed(26)
    processor = ForcedSeedLogitsProcessor(
        randomness=RANDOMNESS, hotkey="h", prompt_idx=7, checkpoint_hash=CHECKPOINT,
        rollout_indices=[0, 3], base_offsets=[0, 5], start_len=4, temperature=1.0, top_k=0,
        top_p=1.0, draw=GUMBEL)
    scores = torch.randn(2, VOCAB)
    out = processor(torch.zeros(2, 6, dtype=torch.long), scores.clone())
    assert int(out[0].argmax()) == _gumbel_expected(scores[0], 0, 2)
    assert int(out[1].argmax()) == _gumbel_expected(scores[1], 3, 7)
