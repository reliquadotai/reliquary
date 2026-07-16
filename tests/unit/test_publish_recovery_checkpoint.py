from __future__ import annotations

import pytest

from scripts.publish_base_reset_checkpoint import _source_load_kwargs


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
