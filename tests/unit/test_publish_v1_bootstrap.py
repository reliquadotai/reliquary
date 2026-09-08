"""Metadata bootstrap CAS, exact retry and inherited weight preservation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts import prepare_v1_bootstrap as prepare
from scripts.publish_v1_bootstrap import publish_prepared
from tests.unit.test_prepare_v5_fill_checkpoint import _profile


def test_bootstrap_is_prepare_only_then_recovers_exact_uncertain_commit(tmp_path, monkeypatch):
    source = {**_profile("qwen3-4b-base-dapo-reasoning-v5"),
              "trained_window_cursor": 50, "lr_schedule_step": 800}
    target = {**_profile(prepare.PROFILE), "training_run_id": "reliquary-v1-test"}
    monkeypatch.setattr(prepare, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(prepare, "PROTOCOL_PROFILE_ID", prepare.PROFILE)
    monkeypatch.setattr(prepare, "active_checkpoint_profile", lambda: dict(target))
    import reliquary.constants as constants
    monkeypatch.setattr(constants, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    parent, child = "a" * 40, "b" * 40
    plan = prepare.prepare_bootstrap(source, repo_id="owner/model", revision=parent,
                                    checkpoint_n=100, last_archived_window=50,
                                    source_bucket="old-run", target_bucket="new-run")
    for name, value in plan["files"].items():
        (tmp_path / name).write_bytes(prepare._json_bytes(value))
    private_plan = {**{k: v for k, v in plan.items() if k != "files"},
                    "add_files": {k: hashlib.sha256(prepare._json_bytes(v)).hexdigest()
                                  for k, v in plan["files"].items()}}
    (tmp_path / "commit-plan.json").write_text(json.dumps(private_plan))
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source))
    source_data = {name: b"unchanged" for name in
                   ("model.safetensors", "config.json", "tokenizer_config.json", "tokenizer.json")}
    source_data[prepare.CHECKPOINT_PROFILE_NAME] = source_file.read_bytes()

    class Hub:
        head = parent
        calls = 0
        files = {parent: source_data}

        def list_repo_commits(self, repo, revision=None):
            old = NS(commit_id=parent, title="checkpoint 100")
            return [old] if (revision or self.head) == parent else [
                NS(commit_id=child, title=plan["commit_message"]), old]

        def model_info(self, repo, revision=None, files_metadata=False):
            revision = revision or self.head
            siblings = [NS(rfilename=name, size=len(data), lfs=None,
                           blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data,
                                               usedforsecurity=False).hexdigest())
                        for name, data in self.files[revision].items()]
            return NS(sha=revision, siblings=siblings)

        def create_commit(self, *, repo_id, parent_commit, commit_message, operations):
            assert parent_commit == self.head
            self.calls += 1
            self.files[child] = {**self.files[parent], **{
                op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations}}
            self.head = child
            raise OSError("commit persisted, acknowledgement lost")

    hub = Hub()
    def download(*args, **kwargs):
        return str(source_file)
    kwargs = dict(api=hub, download=download)
    assert publish_prepared(tmp_path, **kwargs)["status"] == "verified_not_published"
    assert hub.calls == 0
    with pytest.raises(ValueError, match="fenced parent"):
        publish_prepared(tmp_path, **kwargs, apply=True)
    with pytest.raises(OSError, match="acknowledgement"):
        publish_prepared(tmp_path, **kwargs, apply=True, fenced_parent=parent)
    receipt = publish_prepared(tmp_path, **kwargs, apply=True, fenced_parent=parent)
    assert receipt["revision"] == child and receipt["checkpoint_n"] == 101
    assert hub.calls == 1
    assert json.loads((tmp_path / "published-receipt.json").read_text()) == receipt
    hub.files[child]["model.safetensors"] = b"unexpected foreign weights"
    with pytest.raises(ValueError, match="changed weights"):
        publish_prepared(tmp_path, **kwargs, apply=True, fenced_parent=parent)
    hub.files[child]["model.safetensors"] = source_data["model.safetensors"]
    hub.files[child][prepare.CHECKPOINT_PROFILE_NAME] = b"tampered"
    with pytest.raises(ValueError, match="remote metadata"):
        publish_prepared(tmp_path, **kwargs)
    (tmp_path / prepare.TRANSITION).write_text("{}")
    with pytest.raises(ValueError, match="file changed"):
        publish_prepared(tmp_path, **kwargs)
