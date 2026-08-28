"""Safety gates for the manual finished-run archive/finalize command."""

from datetime import datetime, timezone
from types import SimpleNamespace
import sys

import pytest

from scripts import archive_finished_hf_run as archive


def _argv(*extra: str) -> list[str]:
    return [
        "archive_finished_hf_run.py",
        "--repo-id",
        "ReliquaryForge/finished",
        "--checkpoints",
        "10",
        *extra,
    ]


def test_apply_requires_explicit_finished_confirmation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--expected-head", "head", "--apply"),
    )
    monkeypatch.setattr(
        archive,
        "HfApi",
        lambda *args, **kwargs: pytest.fail("HF must not be queried"),
    )

    with pytest.raises(SystemExit, match="--confirm-finished"):
        archive.main()


def test_configured_active_repo_is_rejected_before_any_write(
    monkeypatch,
):
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "ReliquaryForge/finished")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--expected-head",
            "head",
            "--confirm-finished",
            "--apply",
        ),
    )
    monkeypatch.setattr(
        archive,
        "HfApi",
        lambda *args, **kwargs: pytest.fail("HF must not be queried"),
    )

    with pytest.raises(SystemExit, match="active trainer repository"):
        archive.main()


def test_squash_refuses_to_create_a_missing_archive(monkeypatch):
    class _HfApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return SimpleNamespace(
                sha="head",
                used_storage=100,
                siblings=[SimpleNamespace(rfilename="model.bin", size=10)],
            )

        def list_repo_commits(self, repo_id, repo_type):
            return [SimpleNamespace(
                title="checkpoint 10",
                commit_id="head",
                created_at=datetime.now(timezone.utc),
            )]

        def list_repo_refs(self, repo_id, repo_type):
            return SimpleNamespace(branches=[SimpleNamespace(name="main")])

    monkeypatch.delenv("RELIQUARY_HF_REPO_ID", raising=False)
    monkeypatch.setattr(archive, "HfApi", _HfApi)
    monkeypatch.setattr(archive, "_r2_client", object)
    monkeypatch.setattr(archive, "_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        archive,
        "snapshot_download",
        lambda *args, **kwargs: pytest.fail("squash must not backfill R2"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--expected-head",
            "head",
            "--confirm-finished",
            "--apply",
            "--squash",
        ),
    )

    with pytest.raises(SystemExit, match="completed earlier archive phase"):
        archive.main()
