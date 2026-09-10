"""Metadata bootstrap CAS, exact retry and inherited weight preservation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scripts import prepare_v1_bootstrap as prepare
from scripts.publish_v1_bootstrap import publish_prepared
from tests.unit.test_prepare_v5_fill_checkpoint import _profile


@pytest.mark.parametrize("storage_mode", ["distinct-bucket", "reuse-after-fence"])
@pytest.mark.parametrize("reset_to_base", [False, True])
def test_bootstrap_is_prepare_only_then_recovers_exact_uncertain_commit(tmp_path, monkeypatch, storage_mode, reset_to_base):
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
                                    source_bucket="old-run", target_bucket="old-run" if storage_mode == "reuse-after-fence" else "new-run",
                                    storage_mode=storage_mode, reset_to_base=reset_to_base)
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
    source_data["reliquary_recovery_manifest.json"] = b"old V5 manifest"
    base_data = {name: b"original base " + name.encode() for name in
                 ("config.json", "tokenizer_config.json", "tokenizer.json", "LICENSE",
                  "model-00001-of-00001.safetensors")}
    base_data["model.safetensors.index.json"] = json.dumps({
        "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
    }).encode()
    base_repo, base_revision = target["base_model_id"], target["base_model_revision"]

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
            data_files = base_data if repo == base_repo and revision == base_revision else self.files[revision]
            siblings = [NS(rfilename=name, size=len(data), lfs=None,
                           blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data,
                                               usedforsecurity=False).hexdigest())
                        for name, data in data_files.items()]
            return NS(sha=revision, siblings=siblings)

        def create_commit(self, *, repo_id, parent_commit, commit_message, operations):
            from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
            assert parent_commit == self.head
            self.calls += 1
            self.files[child] = dict(self.files[parent])
            for op in operations:
                if isinstance(op, CommitOperationCopy):
                    assert reset_to_base and op.src_repo_id == base_repo
                    assert op.src_revision == base_revision and op.src_repo_type == "model"
                    self.files[child][op.path_in_repo] = base_data[op.src_path_in_repo]
                elif isinstance(op, CommitOperationDelete):
                    del self.files[child][op.path_in_repo]
                else:
                    assert isinstance(op, CommitOperationAdd)
                    self.files[child][op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
            self.head = child
            raise OSError("commit persisted, acknowledgement lost")

    hub = Hub()
    def download(repo, name, *, revision):
        if repo == base_repo:
            assert revision == base_revision
            path = tmp_path / ("base-" + name)
            path.write_bytes(base_data[name])
            return str(path)
        assert repo == "owner/model" and revision == parent and name == prepare.CHECKPOINT_PROFILE_NAME
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
    expected_data = base_data if reset_to_base else source_data
    assert {name: data for name, data in hub.files[child].items() if name not in plan["files"]} == {
        name: data for name, data in expected_data.items() if name not in plan["files"]}
    if reset_to_base:
        assert receipt["base_weights"] == {"repo_id": base_repo, "revision": base_revision}
        assert "model.safetensors" not in hub.files[child]
        assert "reliquary_recovery_manifest.json" not in hub.files[child]
    weight_name = "model-00001-of-00001.safetensors" if reset_to_base else "model.safetensors"
    hub.files[child][weight_name] = b"unexpected foreign weights"
    with pytest.raises(ValueError, match="changed weights"):
        publish_prepared(tmp_path, **kwargs, apply=True, fenced_parent=parent)
    hub.files[child][weight_name] = expected_data[weight_name]
    hub.files[child][prepare.CHECKPOINT_PROFILE_NAME] = b"tampered"
    with pytest.raises(ValueError, match="remote metadata"):
        publish_prepared(tmp_path, **kwargs)
    (tmp_path / prepare.TRANSITION).write_text("{}")
    with pytest.raises(ValueError, match="file changed"):
        publish_prepared(tmp_path, **kwargs)
