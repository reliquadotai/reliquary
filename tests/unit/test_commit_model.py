"""Pydantic schema for the inner GRAIL commit dict.

Tests the structural contract that miners must satisfy in
``RolloutSubmission.commit``. Cross-field consistency rules
(commitments length, prompt+completion length, token_logprobs
length) are checked here.
"""

import pytest
from pydantic import ValidationError

from reliquary.protocol.submission import CommitModel


def _valid_commit(seq_len: int = 40, prompt_len: int = 8) -> dict:
    """Build a schema-compliant commit dict for a rollout of ``seq_len`` tokens."""
    completion_len = seq_len - prompt_len
    return {
        "tokens": list(range(seq_len)),
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v7",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_len,
            "completion_length": completion_len,
            "success": True,
            "total_reward": 0.5,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def test_valid_commit_parses():
    CommitModel.model_validate(_valid_commit())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_reward", float("nan")),
        ("advantage", float("inf")),
        ("token_logprobs", [0.0] * 39 + [float("-inf")]),
    ],
)
def test_non_finite_rollout_metadata_rejected(field, value):
    payload = _valid_commit()
    payload["rollout"][field] = value
    with pytest.raises(ValidationError, match="finite"):
        CommitModel.model_validate(payload)


def test_missing_tokens_rejected():
    payload = _valid_commit()
    del payload["tokens"]
    with pytest.raises(ValidationError, match="tokens"):
        CommitModel.model_validate(payload)


def test_proof_version_must_be_v6():
    payload = _valid_commit()
    payload["proof_version"] = "v4"
    with pytest.raises(ValidationError, match="proof_version"):
        CommitModel.model_validate(payload)


def test_commitments_length_mismatch_rejected():
    payload = _valid_commit(seq_len=40)
    payload["commitments"] = payload["commitments"][:-1]  # 39, not 40
    with pytest.raises(ValidationError, match="commitments"):
        CommitModel.model_validate(payload)


def test_prompt_plus_completion_must_equal_tokens_len():
    payload = _valid_commit(seq_len=40, prompt_len=8)
    payload["rollout"]["prompt_length"] = 9   # 9 + 32 = 41 != 40
    with pytest.raises(ValidationError, match="prompt_length"):
        CommitModel.model_validate(payload)


def test_token_logprobs_length_must_match_tokens():
    payload = _valid_commit(seq_len=40)
    payload["rollout"]["token_logprobs"] = [0.0] * 39  # off by one
    with pytest.raises(ValidationError, match="token_logprobs"):
        CommitModel.model_validate(payload)


def test_extra_field_in_commit_rejected():
    payload = _valid_commit()
    payload["sneaky_field"] = "should be rejected"
    with pytest.raises(ValidationError, match="sneaky_field|Extra inputs"):
        CommitModel.model_validate(payload)


def test_tokens_below_challenge_k_rejected():
    payload = _valid_commit(seq_len=20, prompt_len=4)  # 20 < CHALLENGE_K=32
    with pytest.raises(ValidationError, match="tokens"):
        CommitModel.model_validate(payload)


def test_signature_must_be_hex():
    payload = _valid_commit()
    payload["signature"] = "not-hex-zzz"
    with pytest.raises(ValidationError, match="signature"):
        CommitModel.model_validate(payload)


def test_beacon_randomness_must_be_hex():
    payload = _valid_commit()
    payload["beacon"]["randomness"] = "not-hex-zzz"
    with pytest.raises(ValidationError, match="randomness"):
        CommitModel.model_validate(payload)


def test_short_tokens_does_not_cascade_misleading_errors():
    """When ``tokens`` fails its own length check, the cross-field validators
    must NOT report misleading "length 0" errors that mask the real cause.
    """
    payload = _valid_commit(seq_len=20, prompt_len=4)  # 20 < CHALLENGE_K
    with pytest.raises(ValidationError) as exc_info:
        CommitModel.model_validate(payload)
    # The real error mentions tokens length being too short.
    # The misleading cascade would say "tokens length 0" — must not appear.
    error_str = str(exc_info.value)
    assert "tokens length 0" not in error_str
    assert "len(tokens)=0" not in error_str
