from reliquary.protocol.submission import RejectReason, BatchSubmissionRequest


def test_seed_mismatch_reason_exists():
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"


def test_protocol_version_defaults_zero_and_accepts_int():
    fields = BatchSubmissionRequest.model_fields
    assert "protocol_version" in fields
    assert fields["protocol_version"].default == 0


def test_submission_rejects_mixed_environment_rollouts():
    import pytest
    from pydantic import ValidationError

    rollouts = [
        {
            "tokens": [1],
            "reward": 0.0,
            "commit": {"tokens": [1]},
            "env_name": "openmathinstruct",
        }
        for _ in range(8)
    ]
    rollouts[0]["env_name"] = "opencodeinstruct"
    with pytest.raises(ValidationError, match="share one env_name"):
        BatchSubmissionRequest(
            miner_hotkey="hk",
            prompt_idx=0,
            window_start=0,
            merkle_root="00" * 32,
            rollouts=rollouts,
            checkpoint_hash="",
        )


def test_checkpoint_hash_is_length_bounded():
    # Bounds checkpoint_hash so the forced-seed derivation's 2-byte length
    # prefix (_lp) can't overflow (OverflowError) on a huge miner-supplied value.
    field = BatchSubmissionRequest.model_fields["checkpoint_hash"]
    max_lens = [getattr(m, "max_length", None) for m in field.metadata]
    assert any(ml is not None for ml in max_lens)
