"""A trainer pod publishes checkpoints through presigned uploads: no bucket key on the pod."""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.shared.ckpt_index import index_key
from reliquary.trainer.async_cli import checkpoint_store
from reliquary.trainer.ckpt_publisher import DeltaPublisher
from reliquary.trainer.presigned_store import PresignedObjectStore
from reliquary.validator.rl_async_service import build_rl_trainer_router
from tests.unit.test_ckpt_publication import chain
from tests.unit.test_rl_async_service import TOKEN, components

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path):
    rl = components(tmp_path)
    rl.presign_put = lambda key: f"https://bucket.invalid/put/{key}"
    rl.presign_get = lambda key: f"https://bucket.invalid/get/{key}"
    deleted = []
    rl.delete_object = deleted.append
    app = FastAPI()
    app.include_router(build_rl_trainer_router(rl, token=TOKEN))
    return TestClient(app), rl, deleted


def test_upload_urls_only_under_the_run_prefix(tmp_path):
    client, rl, deleted = _client(tmp_path)
    run = rl.settings.run
    ok = client.post("/rl/trainer/uploads", headers=AUTH, json={"keys": [f"rl/{run}/a", f"rl/{run}/b/c"]})
    assert ok.status_code == 200
    assert ok.json()["urls"] == {f"rl/{run}/a": f"https://bucket.invalid/put/rl/{run}/a",
                                 f"rl/{run}/b/c": f"https://bucket.invalid/put/rl/{run}/b/c"}
    for bad in (["rl/other/a"], [f"rl/{run}x/a"], [f"rl/{run}/../other/a"], ["a"], [f"rl/{run}/a", "rl/other/a"]):
        for path in ("uploads", "deletes", "reads"):
            assert client.post(f"/rl/trainer/{path}", headers=AUTH, json={"keys": bad}).status_code == 422
    assert client.post("/rl/trainer/uploads", headers=AUTH,
                       json={"keys": [f"rl/{run}/{i}" for i in range(513)]}).status_code == 422
    assert client.post("/rl/trainer/uploads", headers=AUTH,
                       json={"keys": [f"rl/{run}/{i}" for i in range(512)]}).status_code == 200
    assert client.post("/rl/trainer/deletes", headers=AUTH, json={"keys": [f"rl/{run}/a"]}).status_code == 200
    assert deleted == [f"rl/{run}/a"]
    assert client.post("/rl/trainer/uploads", json={"keys": [f"rl/{run}/a"]}).status_code == 401
    rl.close()


def test_a_controller_without_a_bucket_serves_no_urls(tmp_path):
    client, rl, _ = _client(tmp_path)
    rl.presign_put = None
    assert client.post("/rl/trainer/uploads", headers=AUTH, json={"keys": [f"rl/{rl.settings.run}/a"]}).status_code == 404
    rl.close()


class _Link:
    """The controller's side, in process: presigned URLs name the object, as the bucket would."""

    def __init__(self, run):
        self.run = run

    def _urls(self, keys, kind):
        assert all(k.startswith(f"rl/{self.run}/") for k in keys)
        return {k: f"https://bucket.invalid/{kind}/{k}" for k in keys}

    def upload_urls(self, keys):
        return self._urls(keys, "put")

    def read_urls(self, keys):
        return self._urls(keys, "get")

    def delete_keys(self, keys):
        for k in keys:
            self.http.objects.pop(k, None)


class _Response:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        assert self.status_code < 400


class _Http:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put(self, url, content):
        self.objects[url.split("/put/", 1)[1]] = content if isinstance(content, bytes) else content.read()
        return _Response(200)

    def get(self, url):
        key = url.split("/get/", 1)[1]
        return _Response(200, self.objects[key]) if key in self.objects else _Response(404)


def test_the_publisher_writes_through_presigned_urls(tmp_path):
    http, link = _Http(), _Link("run1")
    link.http = http
    store = PresignedObjectStore(link, http=http)
    states = chain(3)
    pub = DeltaPublisher(run="run1", store=store, staging_dir=tmp_path / "staging", anchor_interval=2,
                         prune_grace_s=0.0)
    for version, state in enumerate(states):
        index = pub.publish(version, state)
    stored = json.loads(http.objects[index_key("run1")])
    assert stored["latest"]["version"] == index["latest"]["version"] == 3
    named = {f["key"] for a in index["anchors"] for f in a["files"]} | {d["key"] for d in index["deltas"]}
    assert named and named | {index_key("run1")} == set(http.objects)
    assert store.get_bytes("rl/run1/missing") is None
    assert store.get_bytes(index_key("run1")) == http.objects[index_key("run1")]
    store.delete(index_key("run1"))
    assert index_key("run1") not in http.objects


def test_a_trainer_pod_never_reads_bucket_keys(monkeypatch):
    link = _Link("run1")
    pod = {"RELIQUARY_RL_TRAINER_EXECUTOR_ID": "exec-1", "R2_PUBLIC_BUCKET": "reliquary-public"}
    assert isinstance(checkpoint_store(link, pod), PresignedObjectStore)
    keyed = {**pod, "R2_PUBLIC_ACCESS_KEY_ID": "AKID", "R2_PUBLIC_SECRET_ACCESS_KEY": "s"}
    monkeypatch.setattr("reliquary.infrastructure.storage.public_checkpoint_client", lambda: object())
    assert not isinstance(checkpoint_store(link, keyed), PresignedObjectStore)
    assert not isinstance(checkpoint_store(link, {}), PresignedObjectStore)


# --- fix round 1 -------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from reliquary.trainer.presigned_store import MAX_PUT_BYTES, SHARD_BYTES  # noqa: E402


class _Flaky(_Http):
    def __init__(self, failures):
        super().__init__()
        self.failures, self.urls, self.bodies = failures, [], []

    def put(self, url, content):
        self.urls.append(url)
        self.bodies.append(content if isinstance(content, bytes) else content.read())
        if len(self.urls) <= self.failures:
            return _Response(503)
        return super().put(url, content)


class _CountingLink(_Link):
    def __init__(self, run):
        super().__init__(run)
        self.asked = 0

    def upload_urls(self, keys):
        self.asked += 1
        return {k: f"https://bucket.invalid/put/{k}?n={self.asked}" for k in keys}


def test_a_failed_put_is_retried_with_a_fresh_url_and_the_whole_file(tmp_path):
    path = tmp_path / "shard"
    path.write_bytes(b"x" * 100)
    http, link, sleeps = _Flaky(2), _CountingLink("run1"), []
    PresignedObjectStore(link, http=http, sleep=sleeps.append).put_file("rl/run1/s", path)
    assert link.asked == 3 and sleeps == [2.0, 4.0]
    assert len(set(http.urls)) == 3 and http.bodies == [b"x" * 100] * 3


def test_a_put_that_keeps_failing_raises_after_three_attempts():
    http, link = _Flaky(9), _CountingLink("run1")
    with pytest.raises(Exception):
        PresignedObjectStore(link, http=http, sleep=lambda s: None).put_bytes("rl/run1/a", b"x")
    assert len(http.urls) == 3


def test_a_file_over_the_single_put_limit_is_refused(tmp_path):
    path = tmp_path / "big"
    with path.open("wb") as f:
        f.truncate(MAX_PUT_BYTES + 1)
    http = _Http()
    with pytest.raises(ValueError, match="single PUT"):
        PresignedObjectStore(_Link("run1"), http=http).put_file("rl/run1/big", path)
    assert not http.objects and SHARD_BYTES == 2 * 1024**3


def test_a_key_over_1024_bytes_is_refused(tmp_path):
    client, rl, _ = _client(tmp_path)
    key = f"rl/{rl.settings.run}/" + "a" * 1024
    assert client.post("/rl/trainer/uploads", headers=AUTH, json={"keys": [key]}).status_code == 422
    rl.close()


def test_the_publisher_of_a_pod_cuts_anchor_shards_small(tmp_path, monkeypatch):
    from reliquary.trainer import async_cli

    pod = {"RELIQUARY_RL_TRAINER_EXECUTOR_ID": "exec-1"}
    store = async_cli.checkpoint_store(_Link("run1"), pod)
    assert async_cli.publisher_options(store) == {"anchor_shard_bytes": SHARD_BYTES}
    assert async_cli.publisher_options(object()) == {}
    settings = {"hf_prefix": "p", "hf_every": 5}
    async_cli.drop_hf_backup_on_pod(store, settings)
    assert settings["hf_prefix"] is None and settings["hf_every"] is None
    keep = {"hf_prefix": "p", "hf_every": 5}
    async_cli.drop_hf_backup_on_pod(object(), keep)
    assert keep["hf_prefix"] == "p"
