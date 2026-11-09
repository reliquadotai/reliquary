"""The miner draws what the profile prescribes: the Gumbel-max pick on the runner-keyed vLLM path, the
slot-counter fakes, the HF processor and the corpus/episode core; an fp32 LM head when asked."""

from types import SimpleNamespace

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
    _fake_vllm,
    _update,
)

GUMBEL = fs.DRAW_GUMBEL_MAX


def _gumbel_params(rollout_index, *, base_offset=0, prompt_idx=7):
    return SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=prompt_idx, checkpoint_hash=CHECKPOINT,
        rollout_index=rollout_index, base_offset=base_offset,
    ))


def _gumbel_expected(row_logits, rollout_index, position, prompt_idx=7):
    u = fs.u_at(RANDOMNESS, prompt_idx, CHECKPOINT, rollout_index, position)
    return int(fs.pick_rows(row_logits.reshape(1, -1), [u], draw=GUMBEL)[0])


def test_a_request_is_unchanged_on_the_wire():
    """The draw is the process's profile, never a request field."""
    args = forced_seed_extra_args(randomness="r", prompt_idx=1, checkpoint_hash="c",
                                  rollout_index=2, base_offset=3)
    assert args == {FORCED_SEED_KEY: {"randomness": "r", "prompt_idx": 1, "checkpoint_hash": "c",
                                      "rollout_index": 2, "base_offset": 3}}


def test_a_gumbel_draw_under_a_warped_profile_is_refused(monkeypatch):
    monkeypatch.setattr("reliquary.constants.FORCED_DRAW", GUMBEL)
    monkeypatch.setattr(vllm_generation, "T_PROTO", 0.6)
    processor = ForcedSeedVLLMProcessor()
    processor.update_state(_update(added=[(0, _gumbel_params(rollout_index=0), [], [])]))
    with pytest.raises(ValueError, match="gumbel_max"):
        processor.apply(torch.randn(1, VOCAB))


# ---------------------------------------------------------------- generator and fp32 head

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


