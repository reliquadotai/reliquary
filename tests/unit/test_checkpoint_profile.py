from __future__ import annotations

import json

import pytest

from reliquary.validator.checkpoint_profile import (
    CHECKPOINT_PROFILE_NAME,
    CheckpointProfileMismatch,
    active_checkpoint_profile,
    validate_checkpoint_profile,
    write_checkpoint_profile,
)


def test_checkpoint_profile_round_trip(tmp_path):
    path = write_checkpoint_profile(tmp_path)

    assert path.name == CHECKPOINT_PROFILE_NAME
    assert validate_checkpoint_profile(tmp_path, required=True) == (
        active_checkpoint_profile()
    )


def test_historical_checkpoint_metadata_may_be_optional(tmp_path):
    assert validate_checkpoint_profile(tmp_path, required=False) is None


def test_required_checkpoint_profile_rejects_missing_metadata(tmp_path):
    with pytest.raises(
        CheckpointProfileMismatch,
        match="no protocol-lineage metadata",
    ):
        validate_checkpoint_profile(tmp_path, required=True)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 99),
        ("profile_id", "other-profile"),
        ("protocol_version", -1),
        ("base_model_id", "other/model"),
        ("base_model_revision", "0" * 40),
    ],
)
def test_checkpoint_profile_rejects_each_lineage_mismatch(
    tmp_path,
    field,
    replacement,
):
    profile = active_checkpoint_profile()
    profile[field] = replacement
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(
        json.dumps(profile),
        encoding="utf-8",
    )

    with pytest.raises(
        CheckpointProfileMismatch,
        match=f"mismatch for {field}",
    ):
        validate_checkpoint_profile(tmp_path, required=True)


def test_checkpoint_profile_rejects_unreadable_metadata(tmp_path):
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text("{", encoding="utf-8")

    with pytest.raises(CheckpointProfileMismatch, match="unreadable"):
        validate_checkpoint_profile(tmp_path, required=True)


def test_checkpoint_profile_rejects_run_state_collision(tmp_path):
    with pytest.raises(ValueError, match="cannot replace lineage fields"):
        write_checkpoint_profile(
            tmp_path,
            extra={"profile_id": "replacement"},
        )


def test_schema_v2_validates_generation_contract_hash(tmp_path):
    expected = active_checkpoint_profile()
    expected.update({
        "schema_version": 2,
        "generation_contract_sha256": "a" * 64,
    })
    checkpoint = dict(expected)
    checkpoint["generation_contract_sha256"] = "b" * 64
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )

    with pytest.raises(
        CheckpointProfileMismatch,
        match="generation_contract_sha256",
    ):
        validate_checkpoint_profile(
            tmp_path,
            required=True,
            expected=expected,
        )
