"""The trainer's view of a signed episode."""
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary import constants as C
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA
from reliquary.validator.training import _policy_token_positions, _rollout_loss, reset_training_state

ENV = "reliquary_test_episode_v1"
SPANS = ((4, 10), (14, 20))
TOKENS = list(range(10, 30))
MODEL_POSITIONS = [t for start, end in SPANS for t in range(start, end)]
SECRET = "session-token-do-not-ship"


def _episode_rollout(*, logprobs=None, reward=1.0, pi_old=True):
    meta = {"prompt_length": 4, "completion_length": len(TOKENS) - 4,
            "token_logprobs": list(logprobs) if logprobs is not None else [-1.0] * len(MODEL_POSITIONS),
            "episode": {"schema_version": SIGNED_EPISODE_SCHEMA, "precommit_sha256": "e" * 64, "seed_index": 0,
                        "assistant_spans": [list(span) for span in SPANS], "stop": "agent_completed",
                        "transcript": {"token": {"claims": {"secret": SECRET}}, "records": []}}}
    rollout = SimpleNamespace(reward=reward, env_name=ENV, commit={"tokens": list(TOKENS), "rollout": meta})
    rollout._validated_assistant_spans = SPANS
    if pi_old:
        rollout._validated_completion_logprobs = [-0.7] * len(MODEL_POSITIONS)
    return rollout


@pytest.fixture
def episode_protocol(monkeypatch):
    monkeypatch.setattr(C, "PROTOCOL_VERSION", 7)
    monkeypatch.setattr(C, "T_PROTO", 1.0)
    monkeypatch.setattr(C, "PI_OLD_FROM_VERIFY_LOGPROBS", True)
    monkeypatch.setattr(C, "RECOMPUTE_PI_OLD_FROM_VERIFY", True)


def _header(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        return json.loads(bytes(npz["header"]))


def _tiny():
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, eos_token_id=2,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).to(torch.float32).eval()


def test_the_loss_reads_model_tokens_only():
    reset_training_state()
    model = _tiny()
    ref = copy.deepcopy(model).eval()
    for parameter in ref.parameters():
        parameter.requires_grad = False
    device = torch.device("cpu")
    full = [-1.0] * len(TOKENS)
    noisy = [value if t in MODEL_POSITIONS else -50.0 for t, value in enumerate(full)]
    moved = list(full)
    moved[MODEL_POSITIONS[0]] = -0.1
    ppo_a, kl_a, n_a = _rollout_loss(model, ref, _episode_rollout(logprobs=full, pi_old=False), 1.0, device)
    ppo_b, kl_b, n_b = _rollout_loss(model, ref, _episode_rollout(logprobs=noisy, pi_old=False), 1.0, device)
    ppo_c, _kl_c, _n_c = _rollout_loss(model, ref, _episode_rollout(logprobs=moved, pi_old=False), 1.0, device)
    assert n_a == n_b == len(MODEL_POSITIONS)
    assert torch.allclose(ppo_a, ppo_b) and torch.allclose(kl_a, kl_b)      # tool-output claims never read
    assert not torch.allclose(ppo_a, ppo_c)                                  # a model position's claim is


def test_tool_only_tokens_receive_no_embedding_gradient():
    reset_training_state()
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, eos_token_id=2,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    ref = copy.deepcopy(model).eval()
    for parameter in ref.parameters():
        parameter.requires_grad = False
    # Tool output after the last model span: causal attention keeps it out of every model position's loss.
    tail_only_ids = [200, 201, 202, 203]
    rollout = _episode_rollout(pi_old=True)
    rollout.commit["tokens"] = TOKENS + tail_only_ids
    rollout.commit["rollout"]["completion_length"] = len(TOKENS) + len(tail_only_ids) - 4
    model_ids = sorted({TOKENS[t] for t in MODEL_POSITIONS})
    ppo, kl, n = _rollout_loss(model, ref, rollout, 1.0, torch.device("cpu"))
    assert n == len(MODEL_POSITIONS)
    (ppo + kl).backward()
    grad = model.get_input_embeddings().weight.grad
    assert grad[tail_only_ids].abs().sum().item() == 0.0
    assert grad[model_ids].abs().sum().item() > 0.0
