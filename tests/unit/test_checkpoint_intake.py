"""Checkpoint intake: poll dedup, staged download, degrade-to-staleness."""

import json

import pytest

from reliquary.validator.checkpoint_intake import (
    CANDIDATE_MANIFEST_KEY,
    CheckpointIntake,
)


REV_6 = "6" * 40
REV_7 = "7" * 40


class _R2:
    def __init__(self, manifest=None, files=None):
        self.manifest = manifest
        self.files = files or {}  # key -> bytes

    def get_object(self, Bucket, Key):
        if Key == CANDIDATE_MANIFEST_KEY and self.manifest is not None:
            body = json.dumps(self.manifest).encode()
            return {"Body": type("B", (), {"read": lambda s: body})()}
        raise KeyError(Key)

    def list_objects_v2(self, Bucket, Prefix):
        matching = [k for k in self.files if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in matching]}

    def download_file(self, bucket, key, dest, Config=None):
        assert Config is not None
        with open(dest, "wb") as f:
            f.write(self.files[key])


def _manifest(revision=REV_7, *, checkpoint_n=5, repo_id="org/repo"):
    return {
        "checkpoint_n": checkpoint_n, "repo_id": repo_id, "revision": revision,
        "trained_window_cursor": 30110, "reason": "cadence",
    }


def test_poll_returns_new_manifest_once(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(manifest=_manifest()), bucket="b",
        staging_dir=str(tmp_path), installed_revision=REV_6,
    )
    assert intake.poll()["revision"] == REV_7
    intake.installed_revision = REV_7
    assert intake.poll() is None


def test_poll_none_when_no_manifest(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(), bucket="b", staging_dir=str(tmp_path),
    )
    assert intake.poll() is None


def test_poll_ignores_manifest_from_another_protocol_run(tmp_path):
    manifest = {
        **_manifest(),
        "protocol_profile_id": "v4",
        "protocol_version": 4,
        "training_run_id": "old-run",
        "generation_contract_sha256": "a" * 64,
    }
    intake = CheckpointIntake(
        r2_client=_R2(manifest=manifest),
        bucket="b",
        staging_dir=str(tmp_path),
        expected_identity={
            "protocol_profile_id": "v5",
            "protocol_version": 5,
            "training_run_id": "new-run",
            "generation_contract_sha256": "b" * 64,
        },
    )

    assert intake.poll() is None
    assert "identity mismatch" in (intake.last_error or "")


def test_stage_rechecks_manifest_identity(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(),
        bucket="b",
        staging_dir=str(tmp_path),
        expected_identity={"training_run_id": "expected-run"},
    )

    assert intake.stage(
        {**_manifest(), "training_run_id": "other-run"}
    ) is False
    assert "identity mismatch" in (intake.last_error or "")


def test_stage_downloads_validates_and_flags_ready(tmp_path):
    r2 = _R2(
        manifest=_manifest(),
        files={
            f"reliquary/checkpoints/{REV_7}/model.safetensors": b"weights",
            f"reliquary/checkpoints/{REV_7}/config.json": b"{}",
        },
    )
    validated = []
    intake = CheckpointIntake(
        r2_client=r2, bucket="b", staging_dir=str(tmp_path),
        validate_fn=lambda p: validated.append(p) or {"ok": True},
    )
    assert intake.stage(_manifest()) is True
    assert intake.staged_ready
    assert validated
    manifest, path = intake.take_staged()
    assert manifest["revision"] == REV_7
    assert (path / "model.safetensors").read_bytes() == b"weights"
    assert not intake.staged_ready  # handed off exactly once
    with pytest.raises(RuntimeError):
        intake.take_staged()


def test_stage_failure_degrades_to_staleness(tmp_path):
    r2 = _R2(manifest=_manifest(), files={})  # empty mirror
    intake = CheckpointIntake(
        r2_client=r2, bucket="b", staging_dir=str(tmp_path),
    )
    assert intake.stage(_manifest()) is False
    assert not intake.staged_ready
    assert intake.last_error is not None


def test_validation_failure_clears_staged(tmp_path):
    r2 = _R2(
        manifest=_manifest(),
        files={f"reliquary/checkpoints/{REV_7}/model.safetensors": b"w"},
    )

    def bad_validate(path):
        raise RuntimeError("lineage mismatch")

    intake = CheckpointIntake(
        r2_client=r2, bucket="b", staging_dir=str(tmp_path),
        validate_fn=bad_validate,
    )
    assert intake.stage(_manifest()) is False
    assert not intake.staged_ready
    assert not any((tmp_path / REV_7).glob("*")) or not (
        tmp_path / REV_7
    ).exists()


def test_poll_rejects_mutable_or_noncanonical_revision(tmp_path):
    for revision in ("main", "A" * 40, "../outside"):
        intake = CheckpointIntake(
            r2_client=_R2(manifest=_manifest(revision)),
            bucket="b",
            staging_dir=str(tmp_path),
        )

        assert intake.poll() is None
        assert "identity is invalid" in (intake.last_error or "")


@pytest.mark.parametrize("checkpoint_n", [True, 5.0, "5", -1])
def test_poll_and_stage_reject_noncanonical_checkpoint_number(
    tmp_path,
    checkpoint_n,
):
    manifest = _manifest(checkpoint_n=checkpoint_n)
    intake = CheckpointIntake(
        r2_client=_R2(manifest=manifest),
        bucket="b",
        staging_dir=str(tmp_path),
    )

    assert intake.poll() is None
    assert "identity is invalid" in (intake.last_error or "")
    assert intake.stage(manifest) is False
    assert not (tmp_path / REV_7).exists()


@pytest.mark.parametrize(
    "manifest, error",
    [
        (_manifest(REV_7, checkpoint_n=4), "roll back"),
        (_manifest(REV_7, checkpoint_n=5), "rebind"),
    ],
)
def test_intake_rejects_rollback_and_same_number_rebinding(
    tmp_path,
    manifest,
    error,
):
    intake = CheckpointIntake(
        r2_client=_R2(manifest=manifest),
        bucket="b",
        staging_dir=str(tmp_path),
        installed_checkpoint_n=5,
        installed_repo_id="org/repo",
        installed_revision=REV_6,
    )

    assert intake.poll() is None
    assert intake.stage(manifest) is False
    assert error in (intake.last_error or "")


def test_stage_rejects_revision_path_traversal_before_io(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(),
        bucket="b",
        staging_dir=str(tmp_path),
    )

    assert intake.stage(_manifest("../../outside")) is False
    assert not (tmp_path.parent / "outside").exists()


def test_stage_rejects_existing_revision_symlink_outside_root(tmp_path):
    outside = tmp_path.parent / "outside-checkpoint"
    outside.mkdir()
    (tmp_path / REV_7).symlink_to(outside, target_is_directory=True)
    intake = CheckpointIntake(
        r2_client=_R2(),
        bucket="b",
        staging_dir=str(tmp_path),
    )

    assert intake.stage(_manifest()) is False
    assert list(outside.iterdir()) == []
