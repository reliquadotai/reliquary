"""Crash/restart publication tests with independently modeled HF CAS and R2 CAS."""

import asyncio
import hashlib
import io
import json
from pathlib import Path

from botocore.exceptions import ClientError
import pytest

from reliquary.trainer import publisher as module

BASE = "0" * 40
FOREIGN = "f" * 40


def identity(data):
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "blob_id": hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False
        ).hexdigest(),
    }


class HF:
    def __init__(self):
        self.head = BASE
        self.commits = {}
        self.calls = 0
        self.failure = None

    def get_head(self, repo):
        return self.head

    async def upload(self, *, folder_path, repo_id, commit_message, parent_commit):
        self.calls += 1
        assert parent_commit == self.head, "HF CAS failed"
        if self.failure == "before":
            self.failure = None
            raise RuntimeError("HF unavailable before commit")
        revision = f"{len(self.commits) + 1:040x}"
        self.commits[revision] = {
            "parent": parent_commit,
            "title": commit_message,
            "files": {p.name: p.read_bytes() for p in Path(folder_path).iterdir()},
        }
        self.head = revision
        if self.failure == "after":
            self.failure = None
            raise RuntimeError("HF committed but acknowledgement lost")
        return revision

    def verify(self, *, repo_id, revision, parent_revision, commit_message, files):
        found = self.commits.get(revision)
        if (
            found is None
            or found["parent"] != parent_revision
            or found["title"] != commit_message
        ):
            raise module.PublicationConflict("not our exact commit")
        if {name: identity(raw) for name, raw in found["files"].items()} != files:
            raise module.PublicationConflict("remote contents differ")


class R2:
    def __init__(self):
        self.objects = {}
        self.failure = None
        self.puts = 0
        self.uploads = 0

    @staticmethod
    def etag(data):
        return '"' + hashlib.sha256(data).hexdigest() + '"'

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {
            "Body": io.BytesIO(self.objects[Key]),
            "ETag": self.etag(self.objects[Key]),
        }

    def upload_file(self, path, bucket, key, Config):
        self.uploads += 1
        self.objects[key] = Path(path).read_bytes()
        if self.failure == "mirror":
            self.failure = None
            raise RuntimeError("mirror write interrupted")

    def put_object(self, *, Bucket, Key, Body, **condition):
        if self.failure == "manifest_before":
            self.failure = None
            raise RuntimeError("manifest not committed")
        old = self.objects.get(Key)
        if condition == {"IfNoneMatch": "*"}:
            valid = old is None
        else:
            valid = old is not None and condition == {"IfMatch": self.etag(old)}
        if not valid:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts += 1
        self.objects[Key] = Body
        if self.failure == "manifest_after":
            self.failure = None
            raise RuntimeError("manifest committed but acknowledgement lost")

    def list_objects_v2(self, *, Bucket, Prefix, Delimiter=None):
        matching = [key for key in self.objects if key.startswith(Prefix)]
        if Delimiter:
            return {
                "CommonPrefixes": [
                    {"Prefix": prefix}
                    for prefix in sorted(
                        {key[: key.index("/", len(Prefix)) + 1] for key in matching}
                    )
                ]
            }
        return {"Contents": [{"Key": key} for key in matching]}

    def delete_object(self, *, Bucket, Key):
        del self.objects[Key]


@pytest.fixture
def rig(tmp_path):
    hf, r2, saves = HF(), R2(), []

    def save(model, tokenizer, path):
        saves.append(model)
        (path / "model.safetensors").write_bytes(b"trained-weights")
        (path / "config.json").write_bytes(b'{"model_type":"fixture"}')

    def factory(**kwargs):
        return module.TrainerPublisher(
            repo_id="org/repo",
            staging_dir=str(tmp_path),
            tokenizer=None,
            r2_client=r2,
            bucket="test",
            save_fn=save,
            hf_upload_fn=hf.upload,
            hf_head_fn=hf.get_head,
            hf_verify_fn=hf.verify,
            **kwargs,
        )

    return hf, r2, saves, factory


def publish(factory):
    return asyncio.run(
        factory().publish(
            object(),
            parent_revision=BASE,
            checkpoint_n=5,
            lr_schedule_step=80,
            trained_window_cursor=421,
            reason="cadence",
        )
    )


@pytest.mark.parametrize(
    "failure", ["before", "after", "mirror", "manifest_before", "manifest_after"]
)
def test_recovery_completes_one_exact_checkpoint_without_training_again(
    tmp_path, rig, failure
):
    hf, r2, saves, factory = rig
    if failure in {"before", "after"}:
        hf.failure = failure
    else:
        r2.failure = failure
    with pytest.raises(RuntimeError):
        publish(factory)
    assert (tmp_path / module.PENDING_PUBLICATION).is_file()
    assert (
        tmp_path / "ckpt_5" / "model.safetensors"
    ).read_bytes() == b"trained-weights"
    recovered = asyncio.run(factory().recover_pending())
    assert recovered["checkpoint_n"] == 5 and recovered["trained_window_cursor"] == 421
    assert recovered["revision"] == hf.head
    assert json.loads(r2.objects[module.CANDIDATE_MANIFEST_KEY]) == recovered
    assert len(hf.commits) == 1 and len(saves) == 1 and r2.puts == 1
    assert hf.calls == (2 if failure == "before" else 1)
    assert not (tmp_path / module.PENDING_PUBLICATION).exists()
    assert not (tmp_path / "ckpt_5").exists()
    assert asyncio.run(factory().recover_pending()) is None


@pytest.mark.parametrize(
    "state", ["preparing", "prepared", "uploading", "uploaded", "committed"]
)
def test_crash_after_each_durable_transition(tmp_path, rig, monkeypatch, state):
    hf, r2, saves, factory = rig
    original = module.write_json
    armed = [True]

    def crash(path, value):
        original(path, value)
        if (
            path.name == module.PENDING_PUBLICATION
            and value["state"] == state
            and armed[0]
        ):
            armed[0] = False
            raise RuntimeError("process lost after durable transition")

    monkeypatch.setattr(module, "write_json", crash)
    with pytest.raises(RuntimeError):
        publish(factory)
    recovered = asyncio.run(factory().recover_pending())
    if state == "preparing":
        assert recovered is None and not hf.commits and not r2.objects
        publish(factory)
    else:
        assert recovered["revision"] == hf.head
    assert len(hf.commits) == 1 and r2.puts == 1 and len(saves) == 1


def test_hf_response_before_local_revision_fsync_recovers_by_receipt(
    tmp_path, rig, monkeypatch
):
    hf, r2, saves, factory = rig
    original = module.write_json
    armed = [True]

    def crash(path, value):
        if (
            path.name == module.PENDING_PUBLICATION
            and value["state"] == "uploaded"
            and armed[0]
        ):
            armed[0] = False
            raise RuntimeError("crash before revision was durable")
        original(path, value)

    monkeypatch.setattr(module, "write_json", crash)
    with pytest.raises(RuntimeError):
        publish(factory)
    assert (
        json.loads((tmp_path / module.PENDING_PUBLICATION).read_text())["revision"]
        is None
    )
    asyncio.run(factory().recover_pending())
    assert len(hf.commits) == 1 and hf.calls == 1 and len(saves) == 1 and r2.puts == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_head",
        "unknown_head",
        "local_file",
        "remote_file",
        "run",
        "cursor",
        "candidate",
    ],
)
def test_recovery_fails_closed_on_foreign_or_changed_state(tmp_path, rig, mutation):
    hf, r2, saves, factory = rig
    hf.failure = "after"
    with pytest.raises(RuntimeError):
        publish(factory)
    if mutation == "foreign_head":
        hf.head = FOREIGN
    elif mutation == "unknown_head":
        hf.head = None
    elif mutation == "local_file":
        (tmp_path / "ckpt_5" / "model.safetensors").write_bytes(b"changed")
    elif mutation == "remote_file":
        hf.commits[hf.head]["files"]["model.safetensors"] = b"changed"
    elif mutation in {"run", "cursor"}:
        pending = tmp_path / module.PENDING_PUBLICATION
        transaction = json.loads(pending.read_text())
        if mutation == "run":
            transaction["manifest"]["training_run_id"] = "another-run"
        else:
            transaction["manifest"]["trained_window_cursor"] += 1
        pending.write_text(json.dumps(transaction))
    else:
        r2.objects[module.CANDIDATE_MANIFEST_KEY] = json.dumps(
            {"checkpoint_n": 6, "revision": FOREIGN, "repo_id": "org/repo"}
        ).encode()
    before = dict(r2.objects)
    with pytest.raises((module.PublicationConflict, ValueError)):
        asyncio.run(factory().recover_pending())
    assert r2.objects == before and r2.puts == 0 and hf.calls == 1
    assert (tmp_path / module.PENDING_PUBLICATION).exists()


def test_process_lock_rejects_second_publisher_without_mutation(tmp_path, rig):
    hf, r2, saves, factory = rig
    with factory()._lock():
        with pytest.raises(module.PublicationConflict, match="another publisher"):
            publish(factory)
    assert not saves and not hf.commits and not r2.objects


def test_committed_cleanup_can_resume_without_complete_local_snapshot(
    tmp_path, rig, monkeypatch
):
    hf, r2, saves, factory = rig
    original = module.TrainerPublisher._cleanup
    monkeypatch.setattr(
        module.TrainerPublisher,
        "_cleanup",
        lambda *_: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        publish(factory)
    (tmp_path / "ckpt_5" / "model.safetensors").unlink()
    monkeypatch.setattr(module.TrainerPublisher, "_cleanup", original)
    recovered = asyncio.run(factory().recover_pending())
    assert recovered["revision"] == hf.head and hf.calls == 1 and r2.puts == 1


def test_r2_compare_and_swap_does_not_overwrite_a_racing_candidate(tmp_path, rig):
    hf, r2, saves, factory = rig
    original = r2.put_object
    foreign = json.dumps(
        {"checkpoint_n": 6, "repo_id": "org/repo", "revision": FOREIGN}
    ).encode()

    def race(**kwargs):
        r2.objects[module.CANDIDATE_MANIFEST_KEY] = foreign
        return original(**kwargs)

    r2.put_object = race
    with pytest.raises(ClientError, match="PreconditionFailed"):
        publish(factory)
    assert r2.objects[module.CANDIDATE_MANIFEST_KEY] == foreign
    assert (tmp_path / module.PENDING_PUBLICATION).exists()


def test_existing_candidate_is_replaced_with_its_exact_etag(rig):
    hf, r2, saves, factory = rig
    r2.objects[module.CANDIDATE_MANIFEST_KEY] = json.dumps(
        {"checkpoint_n": 4, "repo_id": "org/repo", "revision": BASE}
    ).encode()
    publish(factory)
    assert (
        r2.puts == 1
        and json.loads(r2.objects[module.CANDIDATE_MANIFEST_KEY])["checkpoint_n"] == 5
    )


def test_old_publisher_pruning_never_deletes_a_later_mirror(rig):
    hf, r2, saves, factory = rig
    stale = "reliquary/checkpoints/" + "a" * 40 + "/model.safetensors"
    future = "reliquary/checkpoints/" + FOREIGN + "/model.safetensors"
    r2.objects[stale] = b"stale"
    original = r2.upload_file

    def concurrent_future(*args, **kwargs):
        original(*args, **kwargs)
        r2.objects[future] = b"a later publisher's bytes"

    r2.upload_file = concurrent_future
    publish(factory)
    assert (
        stale not in r2.objects and r2.objects[future] == b"a later publisher's bytes"
    )


@pytest.mark.parametrize("corruption", [None, "parent", "title", "lfs", "git_blob"])
def test_default_hf_recovery_verifies_parent_title_and_both_file_hash_modes(
    monkeypatch, corruption
):
    from types import SimpleNamespace
    import huggingface_hub

    files = {"model.safetensors": identity(b"weights"), "config.json": identity(b"{}")}
    revision = "1" * 40

    class Api:
        def list_repo_commits(self, repo_id, *, revision):
            return [
                SimpleNamespace(
                    commit_id=revision,
                    title="other" if corruption == "title" else "exact",
                ),
                SimpleNamespace(commit_id=FOREIGN if corruption == "parent" else BASE),
            ]

        def get_paths_info(self, repo_id, names, *, revision):
            assert revision == "1" * 40 and set(names) == set(files)
            return [
                SimpleNamespace(
                    path="model.safetensors",
                    size=len(b"weights"),
                    lfs=SimpleNamespace(
                        sha256="wrong"
                        if corruption == "lfs"
                        else files["model.safetensors"]["sha256"]
                    ),
                ),
                SimpleNamespace(
                    path="config.json",
                    size=2,
                    lfs=None,
                    blob_id="wrong"
                    if corruption == "git_blob"
                    else files["config.json"]["blob_id"],
                ),
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    kwargs = dict(
        repo_id="org/repo",
        revision=revision,
        parent_revision=BASE,
        commit_message="exact",
        files=files,
    )
    if corruption is None:
        module._default_hf_verify(**kwargs)
    else:
        with pytest.raises(module.PublicationConflict):
            module._default_hf_verify(**kwargs)
