"""Tests for the env_name field on RolloutSubmission (v2 wire schema)."""

import pytest


def test_rollout_submission_carries_env_name():
    from reliquary.protocol.submission import RolloutSubmission
    r = RolloutSubmission(
        tokens=[1, 2, 3], reward=0.5, commit={"rollout": {"prompt_length": 1}},
        env_name="opencodeinstruct",
    )
    assert r.env_name == "opencodeinstruct"


def test_rollout_submission_env_name_required_for_v2():
    """env_name has no default in v2 — omitting it raises ValidationError."""
    from pydantic import ValidationError
    from reliquary.protocol.submission import RolloutSubmission
    with pytest.raises((TypeError, ValidationError)):
        RolloutSubmission(tokens=[1], reward=0.0, commit={})


def test_grail_proof_version_bumped_to_v6():
    from reliquary.constants import GRAIL_PROOF_VERSION
    assert GRAIL_PROOF_VERSION == "v7"
