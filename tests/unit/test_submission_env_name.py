"""Tests for the env_name field on RolloutSubmission (v2 wire schema)."""

import pytest


def test_grail_proof_version_bumped_to_v6():
    from reliquary.constants import GRAIL_PROOF_VERSION
    assert GRAIL_PROOF_VERSION == "v7"
