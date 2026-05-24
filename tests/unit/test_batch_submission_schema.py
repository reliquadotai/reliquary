"""Pydantic schemas for v2 GRPO market submissions."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
    WindowState,
)


def _valid_rollouts(k: int = 4):
    """k successes, (M - k) failures, all with schema-compliant commits."""
    from reliquary.constants import CHALLENGE_K

    rollouts = []
    seq_len = CHALLENGE_K + 4
    prompt_len = 4
    completion_len = seq_len - prompt_len
    for i in range(M_ROLLOUTS):
        tokens = list(range(seq_len))
        commit = {
            "tokens": tokens,
            "commitments": [{"sketch": 0} for _ in range(seq_len)],
            "proof_version": "v6",
            "model": {"name": "test-model", "layer_index": 6},
            "signature": "ab" * 32,
            "beacon": {"randomness": "cd" * 16},
            "rollout": {
                "prompt_length": prompt_len,
                "completion_length": completion_len,
                "success": i < k,
                "total_reward": 1.0 if i < k else 0.0,
                "advantage": 0.0,
                "token_logprobs": [0.0] * seq_len,
            },
        }
        rollouts.append(
            RolloutSubmission(
                tokens=tokens,
                reward=1.0 if i < k else 0.0,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return rollouts


def test_valid_request_parses():
    req = BatchSubmissionRequest(
        miner_hotkey="hk" * 24,
        prompt_idx=42,
        window_start=1000,
        merkle_root="00" * 32,
        rollouts=_valid_rollouts(k=4),
        checkpoint_hash="sha256:test",
    )
    assert req.prompt_idx == 42
    assert len(req.rollouts) == M_ROLLOUTS


def test_rollout_rejects_mismatched_outer_and_commit_tokens():
    rollouts = _valid_rollouts(k=4)
    bad = rollouts[0]
    with pytest.raises(ValidationError, match="tokens must match commit.tokens"):
        RolloutSubmission(
            tokens=bad.tokens[:-1] + [999],
            reward=bad.reward,
            commit=bad.commit,
        )


def test_wrong_rollout_count_rejected():
    with pytest.raises(ValidationError, match="rollouts"):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=42,
            window_start=1000,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(k=4)[:7],  # 7 instead of M
        )


def test_negative_prompt_idx_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=-1,
            window_start=1000,
            merkle_root="00" * 32,
            rollouts=_valid_rollouts(),
        )


def test_malformed_merkle_root_rejected():
    with pytest.raises(ValidationError):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=0,
            window_start=1000,
            merkle_root="zz",
            rollouts=_valid_rollouts(),
        )


def test_all_reject_reasons_serialisable():
    for reason in RejectReason:
        resp = BatchSubmissionResponse(accepted=False, reason=reason)
        assert resp.model_dump()["reason"] == reason.value


def test_accepted_response():
    resp = BatchSubmissionResponse(accepted=True, reason=RejectReason.ACCEPTED)
    dumped = resp.model_dump()
    assert dumped["accepted"] is True
    assert dumped["reason"] == RejectReason.ACCEPTED.value


def test_grpo_batch_state_exposes_cooldown():
    state = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=100,
        anchor_block=1000,
        cooldown_prompts=[42, 7, 99],
        valid_submissions=12,
        checkpoint_n=0,
    )
    dumped = state.model_dump()
    assert set(dumped["cooldown_prompts"]) == {42, 7, 99}


def test_new_reject_reasons_exist():
    """Schema/Token/Termination validators emit dedicated reject codes."""
    assert RejectReason.BAD_SCHEMA.value == "bad_schema"
    assert RejectReason.BAD_TOKENS.value == "bad_tokens"
    assert RejectReason.TOKENS_MISMATCH.value == "tokens_mismatch"
    assert RejectReason.BAD_TERMINATION.value == "bad_termination"
    assert RejectReason.REWARD_DISTRIBUTION.value == "reward_distribution"
