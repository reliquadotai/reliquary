from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.publish_base_reset_checkpoint import (
    RECOVERY_MANIFEST_NAME,
    _build_recovery_manifest,
    _repo_checkpoint_state,
    _source_load_kwargs,
)


def test_remote_recovery_source_requires_full_immutable_revision():
    revision = "a" * 40

    assert _source_load_kwargs("owner/model", revision, "token") == {
        "token": "token",
        "revision": revision,
    }

    with pytest.raises(SystemExit, match="40-character commit SHA"):
        _source_load_kwargs("owner/model", "main", "token")


def test_local_recovery_source_rejects_remote_revision(tmp_path):
    with pytest.raises(SystemExit, match="local source path"):
        _source_load_kwargs(str(tmp_path), "a" * 40, "token")


def test_unpinned_source_preserves_base_reset_compatibility():
    assert _source_load_kwargs("Qwen/Qwen3.5-2B", None, "token") == {
        "token": "token"
    }


def test_checkpoint_state_binds_latest_number_to_observed_head():
    commits = [
        SimpleNamespace(title="checkpoint 7", commit_id="b" * 40),
        SimpleNamespace(title="checkpoint 9", commit_id="a" * 40),
        SimpleNamespace(title="unrelated", commit_id="c" * 40),
    ]
    api = SimpleNamespace(list_repo_commits=lambda **_kwargs: commits)

    assert _repo_checkpoint_state(api, "owner/model") == (9, "b" * 40)


def test_recovery_manifest_hashes_snapshot_and_redacts_local_path(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "test"}), encoding="utf-8"
    )
    (tmp_path / RECOVERY_MANIFEST_NAME).write_text(
        "stale", encoding="utf-8"
    )

    manifest = _build_recovery_manifest(
        tmp_path,
        checkpoint_n=34,
        repo_id="owner/target",
        source_model=str(tmp_path),
        source_revision=None,
        parent_commit="d" * 40,
        created_at="2026-07-16T00:00:00+00:00",
    )

    assert manifest["source"] == {
        "kind": "local",
        "repo": None,
        "revision": None,
    }
    assert [row["path"] for row in manifest["artifacts"]] == [
        "config.json",
        "model.safetensors",
    ]
    assert manifest["artifacts"][1]["sha256"] == (
        "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    )
