"""v2.1 schema extensions: state in GrpoBatchState, checkpoint_hash in BatchSubmissionRequest."""

import pytest
from pydantic import ValidationError

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    WindowState,
)


def _rollouts(k=4):
    return [
        RolloutSubmission(
            tokens=[1, 2, 3], reward=1.0 if i < k else 0.0,
            commit={"tokens": [1, 2, 3], "proof_version": "v7"},
            env_name="openmathinstruct",
        )
        for i in range(M_ROLLOUTS)
    ]


def test_grpo_batch_state_has_window_state_fields():
    s = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=42,
        anchor_block=12345,
        cooldown_prompts=[],
        valid_submissions=0,
        checkpoint_n=7,
        checkpoint_repo_id="aivolutionedge/reliquary-sn",
        checkpoint_revision="rev_sha_007",
    )
    assert s.state == WindowState.OPEN
    assert s.window_n == 42
    assert s.checkpoint_n == 7


def test_grpo_batch_state_checkpoint_optional_pre_first_publish():
    s = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=0,
        anchor_block=0,
        cooldown_prompts=[],
        valid_submissions=0,
        checkpoint_n=0,
        checkpoint_repo_id=None,
        checkpoint_revision=None,
    )
    assert s.checkpoint_repo_id is None
    assert s.checkpoint_revision is None


def test_batch_submission_requires_checkpoint_hash():
    with pytest.raises(ValidationError, match="checkpoint_hash"):
        BatchSubmissionRequest(
            miner_hotkey="hk", prompt_idx=0, window_start=0, merkle_root="00" * 32, rollouts=_rollouts(),
        )


def test_batch_submission_with_checkpoint_hash_parses():
    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=0, window_start=0, merkle_root="00" * 32, rollouts=_rollouts(),
        checkpoint_hash="sha256:abc",
    )
    assert req.checkpoint_hash == "sha256:abc"


def test_behavioural_reject_reasons_exist():
    """Logprob + Distribution validator ports add two new reject codes."""
    from reliquary.protocol.submission import RejectReason
    assert RejectReason.LOGPROB_MISMATCH.value == "logprob_mismatch"
    assert RejectReason.DISTRIBUTION_SUSPICIOUS.value == "distribution_suspicious"
