"""Async HF publication evidence stays bound to the saved checkpoint."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reliquary.miner.rl_async import write_hf_dir
from reliquary.trainer.hf_backup import HfRepoPerBackup, write_backup_profile
from reliquary.trainer.publisher import PUBLICATION_RECEIPT, PublicationConflict
from reliquary.validator.checkpoint_profile import CHECKPOINT_PROFILE_NAME


class Api:
    def __init__(self, parent=None, revision="b" * 40, existing=(), extra_after_upload=()):
        self.parent, self.revision = parent, revision
        self.existing, self.extra_after_upload = existing, extra_after_upload
        self.created, self.uploaded = [], []

    def create_repo(self, repo, **kwargs):
        self.created.append((repo, kwargs))

    def model_info(self, repo):
        return SimpleNamespace(sha=self.parent)

    def upload_folder(self, **kwargs):
        self.uploaded.append((kwargs, {p.name: p.read_bytes() for p in Path(kwargs["folder_path"]).iterdir()}))
        return SimpleNamespace(oid=self.revision)

    def list_repo_files(self, repo, *, revision):
        if revision == self.parent:
            return list(self.existing)
        return [*self.uploaded[-1][1], *self.extra_after_upload]


def snapshot(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name, body in {"config.json": '{"model_type":"example"}',
                       "tokenizer_config.json": '{"chat_template":"original-template"}',
                       "initial_metadata.json": '{"preserve":true}',
                       CHECKPOINT_PROFILE_NAME: '{"lr_schedule_step":999}',
                       PUBLICATION_RECEIPT: '{"publication_id":"stale"}'}.items():
        (base / name).write_text(body)
    folder = write_hf_dir({"weight": torch.tensor([1.0, 2.0])}, base, tmp_path / "snapshot")
    write_backup_profile(folder, lr_schedule_step=121, checkpoint_run="run-example", task_id="task-example")
    return folder


@pytest.mark.parametrize("parent", [None, "a" * 40])
def test_async_publication_preserves_metadata_and_persists_exact_commit(tmp_path, parent):
    folder = snapshot(tmp_path)
    api = Api(parent)
    receipts = tmp_path / "receipts"
    repos = HfRepoPerBackup("example", "checkpoint", api, receipt_dir=receipts)
    result = repos.upload(4, folder)
    kwargs, files = api.uploaded[0]
    publication = json.loads(files[PUBLICATION_RECEIPT])
    profile = json.loads(files[CHECKPOINT_PROFILE_NAME])
    assert kwargs["parent_commit"] == publication["parent_revision"] == parent
    assert publication["manifest"] == {
        "publication_kind": "async_hf_backup", "repo_id": "example/checkpoint-step-00000004",
        "checkpoint_n": 4, "training_run_id": profile["training_run_id"],
        "async_checkpoint_run": "run-example", "async_task_id": "task-example"}
    assert profile["lr_schedule_step"] == 121
    assert publication["publication_id"] != "stale"
    assert len(publication["publication_id"]) == 32
    for name, identity in publication["files"].items():
        assert identity["size"] == len(files[name])
        assert identity["sha256"] == hashlib.sha256(files[name]).hexdigest()
    assert PUBLICATION_RECEIPT not in publication["files"]
    assert json.loads(files["initial_metadata.json"]) == {"preserve": True}
    assert json.loads(files["tokenizer_config.json"])["chat_template"] == "original-template"
    assert "model.safetensors.index.json" in files
    assert result["revision"] == "b" * 40
    assert result["publication_source"].endswith("/" + "b" * 40 + "/" + PUBLICATION_RECEIPT)
    assert result["publication_sha256"] == hashlib.sha256(files[PUBLICATION_RECEIPT]).hexdigest()
    assert result["profile_sha256"] == hashlib.sha256(files[CHECKPOINT_PROFILE_NAME]).hexdigest()
    receipt_path = receipts / "step-00000004.private.json"
    assert json.loads(receipt_path.read_text()) == result
    assert receipt_path.stat().st_mode & 0o077 == 0
    assert receipts.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="already has a publication receipt"):
        repos.upload(4, folder)
    assert len(api.uploaded) == 1


@pytest.mark.parametrize("step", [None, True, -1, 1.5])
def test_missing_or_invented_scheduler_position_is_not_published(tmp_path, step):
    with pytest.raises(ValueError, match="optimizer step"):
        write_backup_profile(tmp_path, lr_schedule_step=step, checkpoint_run="run", task_id="task")


@pytest.mark.parametrize("revision", [None, "main", "a" * 40])
def test_invalid_or_unchanged_hf_acknowledgement_has_no_local_receipt(tmp_path, revision):
    folder = snapshot(tmp_path)
    receipts = tmp_path / "receipts"
    repos = HfRepoPerBackup("example", "checkpoint", Api("a" * 40, revision), receipt_dir=receipts)
    with pytest.raises(ValueError):
        repos.upload(4, folder)
    assert not receipts.exists()


def test_failed_hf_upload_has_no_success_receipt(tmp_path):
    folder = snapshot(tmp_path)
    api = Api()
    def fail(**kwargs):
        raise RuntimeError("upload failed")
    api.upload_folder = fail
    receipts = tmp_path / "receipts"
    with pytest.raises(RuntimeError, match="upload failed"):
        HfRepoPerBackup("example", "checkpoint", api, receipt_dir=receipts).upload(4, folder)
    assert not receipts.exists()


@pytest.mark.parametrize("existing", [(PUBLICATION_RECEIPT,), ("model.safetensors",)])
def test_restart_without_a_local_receipt_cannot_replace_a_published_backup(tmp_path, existing):
    folder = snapshot(tmp_path)
    api = Api("a" * 40, existing=existing)
    receipts = tmp_path / "receipts"
    with pytest.raises(PublicationConflict):
        HfRepoPerBackup("example", "checkpoint", api, receipt_dir=receipts).upload(4, folder)
    assert api.uploaded == []
    assert not receipts.exists()


def test_remote_extra_weight_shard_is_not_recorded_as_a_successful_backup(tmp_path):
    folder = snapshot(tmp_path)
    api = Api(extra_after_upload=("model-00002-of-00002.safetensors",))
    receipts = tmp_path / "receipts"
    with pytest.raises(PublicationConflict, match="weight file set"):
        HfRepoPerBackup("example", "checkpoint", api, receipt_dir=receipts).upload(4, folder)
    assert len(api.uploaded) == 1
    assert not receipts.exists()


def test_read_only_backup_lookup_does_not_require_a_receipt_directory():
    api = Api()
    repos = HfRepoPerBackup("example", "checkpoint", api)
    assert repos.repo_id(4) == "example/checkpoint-step-00000004"
    with pytest.raises(ValueError, match="receipt directory"):
        repos.upload(4, Path("unused"))
    assert api.created == []
