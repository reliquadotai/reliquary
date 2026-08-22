"""Checkpoint intake: poll dedup, staged download, degrade-to-staleness."""

import json

import pytest

from reliquary.validator.checkpoint_intake import (
    CANDIDATE_MANIFEST_KEY,
    CheckpointIntake,
)


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


def _manifest(revision="rev-7"):
    return {
        "checkpoint_n": 5, "repo_id": "org/repo", "revision": revision,
        "trained_window_cursor": 30110, "reason": "cadence",
    }


def test_poll_returns_new_manifest_once(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(manifest=_manifest()), bucket="b",
        staging_dir=str(tmp_path), installed_revision="rev-6",
    )
    assert intake.poll()["revision"] == "rev-7"
    intake.installed_revision = "rev-7"
    assert intake.poll() is None


def test_poll_none_when_no_manifest(tmp_path):
    intake = CheckpointIntake(
        r2_client=_R2(), bucket="b", staging_dir=str(tmp_path),
    )
    assert intake.poll() is None


def test_stage_downloads_validates_and_flags_ready(tmp_path):
    r2 = _R2(
        manifest=_manifest(),
        files={
            "reliquary/checkpoints/rev-7/model.safetensors": b"weights",
            "reliquary/checkpoints/rev-7/config.json": b"{}",
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
    assert manifest["revision"] == "rev-7"
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
        files={"reliquary/checkpoints/rev-7/model.safetensors": b"w"},
    )

    def bad_validate(path):
        raise RuntimeError("lineage mismatch")

    intake = CheckpointIntake(
        r2_client=r2, bucket="b", staging_dir=str(tmp_path),
        validate_fn=bad_validate,
    )
    assert intake.stage(_manifest()) is False
    assert not intake.staged_ready
    assert not any((tmp_path / "rev-7").glob("*")) or not (
        tmp_path / "rev-7"
    ).exists()
