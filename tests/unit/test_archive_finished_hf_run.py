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
        _argv("--expected-head", "a" * 40, "--apply"),
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
            "a" * 40,
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
                sha="a" * 40,
                used_storage=100,
                siblings=[SimpleNamespace(rfilename="model.bin", size=10)],
            )

        def list_repo_commits(self, repo_id, repo_type):
            return [SimpleNamespace(
                title="checkpoint 10",
                commit_id="a" * 40,
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
            "a" * 40,
            "--confirm-finished",
            "--apply",
            "--squash",
        ),
    )

    with pytest.raises(SystemExit, match="completed earlier archive phase"):
        archive.main()


@pytest.mark.parametrize("body", [b"safe", b"evil", b"shorter"])
def test_archive_verification_hashes_body_even_if_metadata_claims_match(body):
    import hashlib
    import io

    digest = hashlib.sha256(b"safe").hexdigest()
    client = SimpleNamespace(get_object=lambda **kwargs: {
        "Body": io.BytesIO(body), "ContentLength": 4,
        "Metadata": {"sha256": digest},
    })
    manifest = {"files": [{"path": "model.bin", "size": 4, "sha256": digest}]}
    if body == b"safe":
        archive._verify_objects(client, "bucket", "prefix", manifest)
    else:
        with pytest.raises(RuntimeError, match="wrong (size|SHA-256)"):
            archive._verify_objects(client, "bucket", "prefix", manifest)


def test_mutable_expected_head_rejected_before_hf_access(monkeypatch):
    monkeypatch.delenv("RELIQUARY_HF_REPO_ID", raising=False)
    monkeypatch.setattr(sys, "argv", _argv("--expected-head", "main", "--confirm-finished", "--apply"))
    monkeypatch.setattr(archive, "HfApi", lambda **kw: pytest.fail("HF must not be queried"))
    with pytest.raises(ValueError, match="40-character commit OID"):
        archive.main()


def test_archive_manifest_rejects_duplicate_identity():
    import io

    client = SimpleNamespace(get_object=lambda **kwargs: {
        "Body": io.BytesIO(b'{"revision":"a","revision":"b"}'),
    })
    with pytest.raises(ValueError, match="duplicate JSON key"):
        archive._manifest(client, "bucket", "key")
