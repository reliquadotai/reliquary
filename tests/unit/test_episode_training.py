"""The trainer's view of a signed episode (plan 2C, Task 11)."""
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
from reliquary.shared.training_payload import decode_training_payload, encode_training_payload
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


def test_the_payload_carries_sequence_spans_rewards_pi_old_and_checkpoint_never_the_transcript(episode_protocol):
    group = SimpleNamespace(rollouts=[_episode_rollout(reward=1.0), _episode_rollout(reward=0.0)], prompt_idx=3)
    blob = encode_training_payload({ENV: [group]}, window_start=30100, checkpoint_revision="rev-ep",
                                   env_order=[ENV], env_targets={ENV: 16},
                                   window_quarantine={"quarantined": False, "reasons": []})
    assert SECRET not in json.dumps(_header(blob))
    assert _header(blob)["rollout_checkpoints"] == ["rev-ep", "rev-ep"]
    first, second = decode_training_payload(blob).batches()[ENV][0].rollouts
    assert first.commit["tokens"] == TOKENS and (first.reward, second.reward) == (1.0, 0.0)
    assert first._validated_assistant_spans == SPANS
    assert _policy_token_positions(first) == MODEL_POSITIONS
    assert first._validated_completion_logprobs == pytest.approx([-0.7] * len(MODEL_POSITIONS))
    assert first.checkpoint_revision == "rev-ep"
    assert "transcript" not in first.commit["rollout"]["episode"]
    assert group.rollouts[0].commit["rollout"]["episode"]["transcript"]["token"]["claims"]["secret"] == SECRET


def test_a_payload_without_signed_episodes_keeps_its_header():
    from tests.unit.test_training_payload_codec import _payload_bytes

    assert "rollout_checkpoints" not in _header(_payload_bytes())


@pytest.mark.parametrize("checkpoints", [
    ["rev-ep", "extra"],          # one entry per rollout
    "rev-ep",                     # a list
    [None, None],                 # the signed episode names its checkpoint
    ["rev-ep", "rev-ep"],         # the unsigned rollout names none
    [7, None], ["", None], [" rev-ep", None],
])
def test_a_malformed_checkpoint_list_is_refused(episode_protocol, checkpoints):
    plain = _episode_rollout()
    plain.commit["rollout"]["episode"] = {"schema_version": "test"}
    group = SimpleNamespace(rollouts=[_episode_rollout(), plain], prompt_idx=3)
    blob = encode_training_payload({ENV: [group]}, window_start=30100, checkpoint_revision="rev-ep",
                                   env_order=[ENV], env_targets={ENV: 16},
                                   window_quarantine={"quarantined": False, "reasons": []})
    from tests.unit.test_training_payload_codec import _replace_payload_header

    assert _header(blob)["rollout_checkpoints"] == ["rev-ep", None]
    decoded = decode_training_payload(blob).batches()[ENV][0].rollouts
    assert decoded[0].checkpoint_revision == "rev-ep" and not hasattr(decoded[1], "checkpoint_revision")
    with pytest.raises(ValueError, match="rollout checkpoints"):
        decode_training_payload(_replace_payload_header(blob, rollout_checkpoints=checkpoints))


def test_a_signed_episode_without_checkpoints_is_refused(episode_protocol):
    group = SimpleNamespace(rollouts=[_episode_rollout()], prompt_idx=3)
    blob = encode_training_payload({ENV: [group]}, window_start=30100, checkpoint_revision="rev-ep",
                                   env_order=[ENV], env_targets={ENV: 16},
                                   window_quarantine={"quarantined": False, "reasons": []})
    from tests.unit.test_training_payload_codec import _replace_payload_header

    with pytest.raises(ValueError, match="rollout checkpoints"):
        decode_training_payload(_replace_payload_header(blob, rollout_checkpoints=None))


def _signed_blob(mutate=None):
    rollout = _episode_rollout()
    if mutate:
        mutate(rollout)
    group = SimpleNamespace(rollouts=[rollout], prompt_idx=3)
    return encode_training_payload({ENV: [group]}, window_start=30100, checkpoint_revision="rev-ep",
                                   env_order=[ENV], env_targets={ENV: 16},
                                   window_quarantine={"quarantined": False, "reasons": []})


def test_spans_without_a_signed_episode_are_refused(episode_protocol):
    from tests.unit.test_training_payload_codec import _replace_payload_header

    blob = _signed_blob()
    header = _header(blob)
    meta = [{k: v for k, v in m.items() if k != "episode"} for m in header["rollout_meta"]]
    with pytest.raises(ValueError, match="assistant spans"):
        decode_training_payload(_replace_payload_header(blob, rollout_meta=meta, rollout_checkpoints=[None]))


def test_a_short_span_list_is_refused(episode_protocol):
    from tests.unit.test_training_payload_codec import _replace_payload_header

    with pytest.raises(ValueError, match="assistant spans"):
        decode_training_payload(_replace_payload_header(_signed_blob(), assistant_spans=[]))


def test_a_signed_episode_with_null_spans_is_refused(episode_protocol):
    from tests.unit.test_training_payload_codec import _replace_payload_header

    with pytest.raises(ValueError, match="assistant spans"):
        decode_training_payload(_replace_payload_header(_signed_blob(), assistant_spans=[None]))


def test_encoding_a_signed_episode_without_validated_spans_raises(episode_protocol):
    with pytest.raises(ValueError, match="assistant spans"):
        _signed_blob(lambda r: setattr(r, "_validated_assistant_spans", None))


@pytest.mark.parametrize("spans", [
    [[4, 10], [14, 21]],          # past the end of the tokens
    [[4, 10], [9, 20]],           # overlapping
    [[14, 20], [4, 10]],          # not increasing
    [[2, 10], [14, 20]],          # inside the prompt
    [[4, 4]],                     # empty
    [[-1, 10]], [[4.0, 10]], [[True, 10]], [[4, 10, 12]], [],
])
def test_malformed_episode_spans_are_refused(episode_protocol, spans):
    from tests.unit.test_training_payload_codec import _replace_payload_header

    with pytest.raises(ValueError, match="assistant spans"):
        decode_training_payload(_replace_payload_header(_signed_blob(), assistant_spans=[spans]))


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
