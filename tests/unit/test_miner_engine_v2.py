"""Miner prompt-picking strategy: pull random in-range, skip cooldown."""

import random
from unittest.mock import MagicMock

import pytest

from reliquary.miner.engine import pick_prompt_idx
from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS


class FakeEnv:
    def __len__(self):
        return 100


def test_pick_prompt_in_range():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_prompt_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(0, 95))  # only 5 choices free: 95..99
    for _ in range(20):
        idx = pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)
        assert idx not in cooldown


def test_pick_prompt_all_cooldown_raises():
    env = FakeEnv()
    rng = random.Random(42)
    cooldown = set(range(100))
    with pytest.raises(RuntimeError, match="no eligible prompt"):
        pick_prompt_idx(env, cooldown_prompts=cooldown, rng=rng)


def test_engine_default_max_new_tokens_is_protocol_cap():
    """The env-var override is removed; max_new_tokens is the protocol cap."""
    import inspect
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
    from reliquary.miner.engine import MiningEngine

    sig = inspect.signature(MiningEngine.__init__)
    default = sig.parameters["max_new_tokens"].default
    assert default == MAX_NEW_TOKENS_PROTOCOL_CAP

    # Belt and suspenders: source must not reference the env var anywhere
    # in the engine module — catches a regression that re-introduces the
    # `int(os.environ.get("RELIQUARY_MAX_NEW_TOKENS", ...))` default.
    src = inspect.getsource(MiningEngine.__init__)
    assert "RELIQUARY_MAX_NEW_TOKENS" not in src


@pytest.mark.asyncio
async def test_checkpoint_download_allows_qwen35_shards(monkeypatch):
    from reliquary.miner import engine as engine_mod

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = await engine_mod._hf_download("repo/id", "abc123")

    assert result == "/tmp/snapshot"
    assert captured["allow_patterns"] == MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "model*.safetensors" in captured["allow_patterns"]
    assert "model.safetensors.index.json" in captured["allow_patterns"]


def test_build_rollout_submission_uses_placeholder_for_authoritative_reward_env():
    from reliquary.miner.engine import MiningEngine

    class _PrivateRewardEnv:
        name = "opencodeinstruct"
        validator_authoritative_reward = True
        compute_reward = MagicMock(side_effect=AssertionError("must not score locally"))

    eng = object.__new__(MiningEngine)
    eng.env = _PrivateRewardEnv()
    eng.tokenizer = MagicMock()
    eng.tokenizer.decode.return_value = "```python\ndef add(a, b): return a + b\n```"
    eng._build_grail_commit = MagicMock(
        return_value={
            "tokens": [10, 11, 12, 13],
            "rollout": {"prompt_length": 2, "completion_length": 2},
        }
    )
    generation = {"tokens": [10, 11, 12, 13], "prompt_length": 2}

    rollout = eng._build_rollout_submission(
        generation, {"prompt": "p"}, "randomness", env=eng.env
    )

    assert rollout.reward == 0.0
    assert rollout.env_name == "opencodeinstruct"
    eng.env.compute_reward.assert_not_called()


def test_generate_rollouts_passes_full_eos_set_and_trims_first_eos(monkeypatch):
    # Pins the non-BFT generation path (eos union + first-EOS trim).
    monkeypatch.setattr("reliquary.constants.BFT_ENABLED", False)
    import torch
    from types import SimpleNamespace

    from reliquary.constants import M_ROLLOUTS
    from reliquary.miner.engine import MiningEngine

    class _Tok:
        chat_template = None
        eos_token_id = 248046
        pad_token_id = 248044

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [10, 11]

    class _Model:
        device = "cpu"

        generation_config = SimpleNamespace(eos_token_id=248044)
        config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044))

        def __init__(self):
            self.kwargs = None

        def generate(self, input_tensor, **kwargs):
            self.kwargs = kwargs
            row = [10, 11, 1, 248046, 99]
            return torch.tensor([row] * input_tensor.shape[0])

    eng = object.__new__(MiningEngine)
    eng.tokenizer = _Tok()
    eng.vllm_model = _Model()
    eng.max_new_tokens = 32

    rollouts = eng._generate_m_rollouts({"prompt": "p"}, "00")

    assert eng.vllm_model.kwargs["eos_token_id"] == [248044, 248046]
    assert len(rollouts) == M_ROLLOUTS
    assert all(r["tokens"] == [10, 11, 1, 248046] for r in rollouts)


def test_generate_rollouts_bft_path_keeps_finished_rows():
    # BFT default path: rows that already emitted </think> are kept (not forced),
    # phase-1 max_new_tokens is capped at the thinking budget, no phase-2 call.
    import torch
    from types import SimpleNamespace

    from reliquary.constants import BFT_THINKING_BUDGET, M_ROLLOUTS
    from reliquary.miner.engine import MiningEngine

    class _Tok:
        chat_template = None
        eos_token_id = 248046
        pad_token_id = 248044

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [10, 11]

        def convert_tokens_to_ids(self, t):
            return 777 if t == "</think>" else None

    class _Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=248044)
        config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044))

        def __init__(self):
            self.kwargs = None
            self.calls = 0

        def generate(self, input_tensor, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            row = [10, 11, 777, 55, 248046]  # prompt + </think> + answer + EOS
            return torch.tensor([row] * input_tensor.shape[0])

    eng = object.__new__(MiningEngine)
    eng.tokenizer = _Tok()
    eng.vllm_model = _Model()
    eng.max_new_tokens = 40000

    rollouts = eng._generate_m_rollouts({"prompt": "p"}, "00")

    assert eng.vllm_model.kwargs["max_new_tokens"] == min(40000, BFT_THINKING_BUDGET)
    assert "logits_processor" not in eng.vllm_model.kwargs
    assert eng.vllm_model.calls == 1  # all finished → no phase-2 generation
    assert len(rollouts) == M_ROLLOUTS
    assert all(r["forced"] is False for r in rollouts)
    assert all(r["tokens"] == [10, 11, 777, 55, 248046] for r in rollouts)


def test_rollout_metadata_carries_bft_fields():
    from reliquary.miner.engine import _rollout_metadata

    forced = {"tokens": [1, 2, 3, 4, 5], "prompt_length": 2,
              "forced": True, "force_span": (3, 5)}
    m = _rollout_metadata(forced, [0.0, 0.0, 0.0])
    assert m["forced"] is True
    assert m["force_span"] == [3, 5]
    assert m["completion_length"] == 3
    assert m["token_logprobs"] == [0.0, 0.0, 0.0]

    natural = {"tokens": [1, 2, 3], "prompt_length": 1, "forced": False}
    m2 = _rollout_metadata(natural, [0.0, 0.0])
    assert m2["forced"] is False
    assert m2["force_span"] is None


def test_pick_prompt_respects_explicit_range():
    env = FakeEnv()  # len 100
    rng = random.Random(1)
    for _ in range(50):
        idx = pick_prompt_idx(
            env, cooldown_prompts=set(), rng=rng, prompt_range=(20, 40),
        )
        assert 20 <= idx < 40


def test_pick_prompt_range_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(1)
    cooldown = set(range(20, 38))  # leaves only 38, 39 free in [20, 40)
    for _ in range(20):
        idx = pick_prompt_idx(
            env, cooldown_prompts=cooldown, rng=rng, prompt_range=(20, 40),
        )
        assert idx in (38, 39)


def test_pick_prompt_full_range_unchanged():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_env_and_prompt_confines_to_window():
    from reliquary.miner.engine import pick_env_and_prompt
    from reliquary.shared.prompt_range import window_prompt_range
    from reliquary.constants import PROMPT_RANGE_SIZE

    class BigEnv:
        name = "openmathinstruct"
        def __len__(self):
            return 20_000

    envs = {"openmathinstruct": BigEnv()}
    mix = [("openmathinstruct", 8)]
    cooldown = {"openmathinstruct": set()}
    rng = random.Random(7)
    rand = "deadbeefcafe"
    lo, hi = window_prompt_range(rand, "openmathinstruct", 20_000, PROMPT_RANGE_SIZE)
    for _ in range(50):
        name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng, randomness=rand)
        assert name == "openmathinstruct"
        assert lo <= idx < hi
