"""Trainer publisher: HF upload + R2 mirror + candidate manifest ordering."""

import asyncio
import json
import hashlib
import io

import pytest

from reliquary.trainer.publisher import CANDIDATE_MANIFEST_KEY, checkpoint_key


FAKE_HEAD = {"head": "0" * 40}


@pytest.fixture(autouse=True)
def _fake_hf(monkeypatch):
    FAKE_HEAD["head"] = "0" * 40
    monkeypatch.setattr(
        "reliquary.trainer.publisher._default_hf_head", lambda repo: FAKE_HEAD["head"]
    )


def _committed(revision):
    FAKE_HEAD["head"] = revision
    return revision


def _staging_empty(path):
    return not [p for p in path.iterdir() if p.name != ".publication.lock"]


REV_1 = "1" * 40
REV_2 = "2" * 40
REV_3 = "3" * 40
REV_9 = "9" * 40


class _R2:
    def __init__(self):
        self.uploads = []  # (key, path) via upload_file
        self.objects = {}  # key -> bytes via put_object
        self.deleted = []  # keys via delete_object

    def upload_file(self, path, bucket, key, Config=None):
        assert Config is not None  # multipart config is mandatory
        self.uploads.append((key, path))
        self.objects[key] = b"<file>"

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = self.objects[Key]
        return {
            "Body": io.BytesIO(raw),
            "ETag": '"' + hashlib.sha256(raw).hexdigest() + '"',
        }

    def put_object(self, Bucket, Key, Body, **kw):
        old = self.objects.get(Key)
        if "IfNoneMatch" in kw:
            assert old is None
        if "IfMatch" in kw:
            assert (
                old is not None
                and kw["IfMatch"] == '"' + hashlib.sha256(old).hexdigest() + '"'
            )
        self.objects[Key] = Body

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
        if Delimiter:
            prefixes = sorted(
                {
                    k[: k.index(Delimiter, len(Prefix)) + 1]
                    for k in self.objects
                    if k.startswith(Prefix) and Delimiter in k[len(Prefix) :]
                }
            )
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
        return {
            "Contents": [
                {"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)
            ]
        }

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        self.deleted.append(Key)


def test_keys():
    assert checkpoint_key(REV_1, "model.safetensors") == (
        f"reliquary/checkpoints/{REV_1}/model.safetensors"
    )
    assert CANDIDATE_MANIFEST_KEY == ("reliquary/training/candidate-manifest.json")


def _count_identity_reads(monkeypatch):
    import reliquary.trainer.publisher as publisher_module

    reads = {}
    real = publisher_module._file_identity

    def counting(path):
        reads[path.name] = reads.get(path.name, 0) + 1
        return real(path)

    monkeypatch.setattr(publisher_module, "_file_identity", counting)
    return reads


class _OrderedR2(_R2):
    """Records every mirror write and the manifest put, in order."""

    def __init__(self, fail_on=None):
        super().__init__()
        self.events = []
        self.fail_on = fail_on

    def upload_file(self, path, bucket, key, Config=None):
        if self.fail_on and key.endswith(self.fail_on):
            self.fail_on = None
            raise OSError("mirror interrupted")
        super().upload_file(path, bucket, key, Config=Config)
        self.events.append(key.rsplit("/", 1)[1])

    def put_object(self, Bucket, Key, Body, **kw):
        super().put_object(Bucket, Key, Body, **kw)
        self.events.append(Key.rsplit("/", 1)[1])


def _publish(pub):
    return asyncio.run(
        pub.publish(
            object(),
            parent_revision=FAKE_HEAD["head"],
            checkpoint_n=5,
            lr_schedule_step=80,
            trained_window_cursor=30110,
            reason="cadence",
        )
    )


def test_the_marker_is_required_for_a_scoped_namespace_or_the_flag():
    from types import SimpleNamespace

    from reliquary.trainer.publisher import MIRROR_MARKER_ENV, mirror_marker_required

    legacy, scoped = SimpleNamespace(scoped=False), SimpleNamespace(scoped=True)
    assert mirror_marker_required(legacy, {}) is False
    assert mirror_marker_required(legacy, {MIRROR_MARKER_ENV: "0"}) is False
    assert mirror_marker_required(legacy, {MIRROR_MARKER_ENV: "1"}) is True
    assert mirror_marker_required(scoped, {}) is True
