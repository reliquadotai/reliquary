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


@pytest.mark.parametrize("stale", [False, True])
def test_default_upload_replaces_weight_layout_atomically(monkeypatch, tmp_path, stale):
    from types import SimpleNamespace
    import huggingface_hub

    (tmp_path / "model.safetensors").write_bytes(b"weights")

    class Api:
        def upload_folder(self, **kwargs):
            assert kwargs["parent_commit"] == BASE
            assert kwargs["delete_patterns"] == list(module.WEIGHT_PATTERNS)
            return SimpleNamespace(oid="1" * 40)

        def list_repo_files(self, repo_id, *, revision):
            assert revision == "1" * 40
            return ["model.safetensors", "config.json"] + (
                ["model-00001-of-00002.safetensors"] if stale else []
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    call = module._default_hf_upload(
        folder_path=str(tmp_path), repo_id="org/repo", parent_commit=BASE,
        commit_message="checkpoint",
    )
    if stale:
        with pytest.raises(module.PublicationConflict):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == "1" * 40


@pytest.mark.parametrize("corruption", [None, "parent", "title", "lfs", "git_blob", "weights"])
def test_default_hf_recovery_verifies_parent_title_and_both_file_hash_modes(
    monkeypatch, corruption
):
    from types import SimpleNamespace
    import huggingface_hub

    files = {"model.safetensors": identity(b"weights"), "config.json": identity(b"{}")}
    revision = "1" * 40

    class Api:
        def list_repo_files(self, repo_id, *, revision):
            return list(files) + (["model.safetensors.index.json"] if corruption == "weights" else [])

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
