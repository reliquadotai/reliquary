"""Trainer publisher: HF upload + R2 mirror + candidate manifest ordering."""

import asyncio
import json

from reliquary.trainer.publisher import (
    CANDIDATE_MANIFEST_KEY,
    TrainerPublisher,
    checkpoint_key,
)


class _R2:
    def __init__(self):
        self.uploads = []      # (key, path) via upload_file
        self.objects = {}      # key -> bytes via put_object
        self.deleted = []      # keys via delete_object

    def upload_file(self, path, bucket, key, Config=None):
        assert Config is not None  # multipart config is mandatory
        self.uploads.append((key, path))
        self.objects[key] = b"<file>"

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
        if Delimiter:
            prefixes = sorted({
                k[: k.index(Delimiter, len(Prefix)) + 1]
                for k in self.objects
                if k.startswith(Prefix) and Delimiter in k[len(Prefix):]
            })
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
        return {"Contents": [
            {"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)
        ]}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        self.deleted.append(Key)


def _publisher(tmp_path, r2, order, *, hf_fails=False):
    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")
        order.append("save")

    async def hf_upload(folder_path, repo_id, commit_message):
        if hf_fails:
            raise RuntimeError("hf down")
        order.append("hf")
        return "rev-123"

    return TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
    )


def test_keys():
    assert checkpoint_key("rev-1", "model.safetensors") == (
        "reliquary/checkpoints/rev-1/model.safetensors"
    )
    assert CANDIDATE_MANIFEST_KEY == (
        "reliquary/training/candidate-manifest.json"
    )


def test_publish_order_manifest_and_cleanup(tmp_path):
    r2, order = _R2(), []
    pub = _publisher(tmp_path, r2, order)
    rev = asyncio.run(pub.publish(
        object(), checkpoint_n=5, lr_schedule_step=80,
        trained_window_cursor=30110, reason="cadence",
    ))
    assert rev == "rev-123"
    assert order == ["save", "hf"]
    uploaded_keys = [k for k, _ in r2.uploads]
    assert checkpoint_key("rev-123", "model.safetensors") in uploaded_keys
    # The profile file travels in the snapshot too.
    assert any(k.endswith("reliquary_checkpoint_profile.json") or
               "profile" in k for k in uploaded_keys) or len(uploaded_keys) >= 2
    manifest = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert manifest == {
        "checkpoint_n": 5, "repo_id": "org/repo", "revision": "rev-123",
        "trained_window_cursor": 30110, "reason": "cadence",
    }
    assert not any(tmp_path.iterdir())  # staging cleaned


def test_profile_extra_written(tmp_path):
    r2, order = _R2(), []
    captured = {}

    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"w")

    async def hf_upload(folder_path, repo_id, commit_message):
        import pathlib
        for p in pathlib.Path(folder_path).iterdir():
            if p.suffix == ".json":
                captured[p.name] = json.loads(p.read_text())
        return "rev-9"

    pub = TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
    )
    asyncio.run(pub.publish(
        object(), checkpoint_n=7, lr_schedule_step=99,
        trained_window_cursor=30200, reason="cadence",
    ))
    profiles = [
        doc for doc in captured.values()
        if doc.get("lr_schedule_step") is not None
    ]
    assert profiles and profiles[0]["lr_schedule_step"] == 99
    assert profiles[0]["trained_window_cursor"] == 30200


def test_mirror_keeps_only_last_two_revisions(tmp_path):
    r2, order = _R2(), []
    # A stale revision from before a restart must be cleaned too.
    r2.objects["reliquary/checkpoints/rev-old/model.safetensors"] = b"x"

    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"w")

    revs = iter(["rev-1", "rev-2", "rev-3"])

    async def hf_upload(folder_path, repo_id, commit_message):
        return next(revs)

    pub = TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
    )
    for n in (1, 2, 3):
        asyncio.run(pub.publish(
            object(), checkpoint_n=n, lr_schedule_step=None,
            trained_window_cursor=30100 + n, reason="cadence",
        ))
    live_revs = {
        k.split("/")[2] for k in r2.objects
        if k.startswith("reliquary/checkpoints/")
    }
    assert live_revs == {"rev-2", "rev-3"}
    assert "reliquary/checkpoints/rev-old/model.safetensors" in r2.deleted


def test_staging_cleaned_on_failure(tmp_path):
    r2, order = _R2(), []
    pub = _publisher(tmp_path, r2, order, hf_fails=True)
    import pytest
    with pytest.raises(RuntimeError, match="hf down"):
        asyncio.run(pub.publish(
            object(), checkpoint_n=5, lr_schedule_step=None,
            trained_window_cursor=30110, reason="cadence",
        ))
    assert not any(tmp_path.iterdir())
    assert CANDIDATE_MANIFEST_KEY not in r2.objects
