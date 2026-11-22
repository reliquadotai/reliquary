"""A pod order's final model: copied to the private bucket once its bill is final, read back verified."""
from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import torch
from botocore.exceptions import ClientError

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.rl_orders.checkpoint_delivery import (
    CHUNK_BYTES, CheckpointConflict, CheckpointRangeError, CheckpointUnavailable,
)
from reliquary.rl_orders.pod_delivery import (
    PREFIX, SCHEMA, PodCheckpointStore, PodDeliveries, deliver_order, trainer_tokens_live,
)
from reliquary.shared.ckpt_delta import apply_delta, decode_delta, load_anchor_shards, verify_checkpoint
from reliquary.shared.ckpt_index import index_key, plan_download
from reliquary.trainer.ckpt_publisher import DeltaPublisher
from reliquary.validator.rl_supervisor import Supervisor

TASK = "order-train-example"


def _error(code):
    return ClientError({"Error": {"Code": code}}, "op")


class Bucket:
    def __init__(self):
        self.objects, self.uploads, self.aborted, self.part_sizes = {}, {}, [], []

    def get_object(self, Bucket, Key, Range=None):
        if Key not in self.objects:
            raise _error("NoSuchKey")
        data = self.objects[Key]
        if Range:
            start, end = map(int, Range.removeprefix("bytes=").split("-"))
            data = data[start:end + 1]
        return {"Body": io.BytesIO(data)}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, **_):
        if IfNoneMatch == "*" and Key in self.objects:
            raise _error("PreconditionFailed")
        self.objects[Key] = bytes(Body)

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _error("404")
        return {"ContentLength": len(self.objects[Key])}

    def create_multipart_upload(self, Bucket, Key):
        self.uploads[Key] = {}
        return {"UploadId": Key}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.uploads[Key][PartNumber] = Body
        self.part_sizes.append(len(Body))
        return {"ETag": str(PartNumber)}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        parts = self.uploads.pop(Key)
        self.objects[Key] = b"".join(parts[p["PartNumber"]] for p in MultipartUpload["Parts"])

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.uploads.pop(Key)
        self.aborted.append(Key)


class Source:
    def __init__(self):
        self.objects = {}

    def get_bytes(self, key):
        return self.objects.get(key)

    def put_bytes(self, key, data):
        self.objects[key] = data

    def put_file(self, key, path):
        self.objects[key] = path.read_bytes()

    def delete(self, key):
        self.objects.pop(key, None)

    def read_index(self, run_id):
        return self.objects.get(index_key(run_id))

    def iter_object(self, key, expected_size):
        value = self.objects[key]
        for start in range(0, len(value), 1024):       # shorter reads than a chunk: re-cut on the way
            yield value[start:start + 1024]


def _entry(status="retired", **order):
    return SimpleNamespace(task_id=TASK, status=status,
                           params={"training_order": {"training_id": "example", "order_sha256": "a" * 64, **order}})


@pytest.fixture
def run(tmp_path):
    source = Source()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    (model / "tokenizer.json").write_text('{"version":"1.0"}')
    publisher = DeltaPublisher(run=TASK, store=source, staging_dir=tmp_path / "staging", model_dir=model,
                               anchor_interval=50)
    state = {"w": torch.tensor([1, 2, 3], dtype=torch.bfloat16)}
    publisher.publish(0, state)
    state = {"w": torch.tensor([1, 4, 3], dtype=torch.bfloat16)}
    publisher.publish(1, state)
    state = {"w": torch.tensor([1, 4, 9], dtype=torch.bfloat16)}
    publisher.publish(2, state)
    bucket = Bucket()
    return PodCheckpointStore(bucket, "private", source), bucket, source, state


def test_the_frozen_model_rebuilds_exactly_after_the_publication_is_gone(run, tmp_path):
    store, bucket, source, expected = run
    manifest = store.freeze(_entry(), 2)
    assert manifest["schema"] == SCHEMA and manifest["checkpoint"]["version"] == 2
    assert store.load_manifest(TASK) == manifest
    assert all(key.startswith(f"{PREFIX}{TASK}/") for key in bucket.objects)
    raw = canonical_json_bytes(manifest)
    assert f"rl/{TASK}/".encode() not in raw and b"https://" not in raw
    source.objects.clear()
    by_id = {artifact["artifact_id"]: artifact for artifact in manifest["files"]}
    index_id = next(a["artifact_id"] for a in manifest["files"] if a["kind"] == "index")
    index = json.loads(store.open_range(TASK, index_id).data)
    plan = plan_download(index, None)
    paths = []
    for file in plan.anchor["files"]:
        artifact = by_id[file["key"].removeprefix("objects/")]
        path = tmp_path / artifact["name"]
        path.write_bytes(store.open_range(TASK, artifact["artifact_id"]).data)
        paths.append(path)
    actual = load_anchor_shards(paths)
    for file in plan.deltas:
        apply_delta(actual, decode_delta(store.open_range(TASK, file["key"].removeprefix("objects/")).data))
    verify_checkpoint(actual, plan.target_sha256)
    assert torch.equal(actual["w"], expected["w"])
    assert store.freeze(_entry(), 2) == manifest                  # a repeat needs no publication


def test_a_publication_behind_the_bill_is_refused(run):
    store, bucket, *_ = run
    with pytest.raises(CheckpointUnavailable):
        store.freeze(_entry(), 3)
    assert store.load_manifest(TASK) is None and not bucket.objects


def test_a_version_published_but_never_adopted_does_not_block_the_bills_version(run, tmp_path):
    store, _, source, _ = run
    manifest = store.freeze(_entry(), 1)                          # the index is at 2, the bill closed on 1
    assert manifest["checkpoint"]["version"] == 1
    published = json.loads(source.objects[index_key(TASK)])
    assert manifest["checkpoint"]["sha256"] == next(d["target_sha256"] for d in published["deltas"] if d["to"] == 1)
    by_id = {artifact["artifact_id"]: artifact for artifact in manifest["files"]}
    index_id = next(a["artifact_id"] for a in manifest["files"] if a["kind"] == "index")
    plan = plan_download(json.loads(store.open_range(TASK, index_id).data), None)
    paths = []
    for file in plan.anchor["files"]:
        artifact = by_id[file["key"].removeprefix("objects/")]
        path = tmp_path / artifact["name"]
        path.write_bytes(store.open_range(TASK, artifact["artifact_id"]).data)
        paths.append(path)
    actual = load_anchor_shards(paths)
    for file in plan.deltas:
        apply_delta(actual, decode_delta(store.open_range(TASK, file["key"].removeprefix("objects/")).data))
    verify_checkpoint(actual, plan.target_sha256)
    assert torch.equal(actual["w"], torch.tensor([1, 4, 3], dtype=torch.bfloat16))


def test_a_retry_resumes_after_the_objects_already_stored(run):
    store, bucket, source, _ = run
    original, calls = source.iter_object, []

    def failing(key, size):
        calls.append(key)
        if len(calls) == 2:
            raise CheckpointUnavailable("r2 read failed")
        return original(key, size)

    source.iter_object = failing
    with pytest.raises(CheckpointUnavailable):
        store.freeze(_entry(), 2)
    assert store.load_manifest(TASK) is None
    assert len(bucket.objects) == 1
    # A stored object that changed is copied again, never trusted.
    key = next(iter(bucket.objects))
    kept = bucket.objects[key]
    bucket.objects[key] = bytes([kept[0] ^ 1]) + kept[1:]
    calls.clear()
    source.iter_object = original
    manifest = store.freeze(_entry(), 2)
    assert bucket.objects[key] == kept and manifest["checkpoint"]["version"] == 2
    calls.clear()
    bucket.objects.pop(f"{PREFIX}{TASK}/manifest.json")
    read = []

    def counted(key, size):
        read.append(key)
        yield from original(key, size)

    source.iter_object = counted
    assert store.freeze(_entry(), 2) == manifest
    assert read == []                                             # every object was already whole


def test_a_freezes_deadline_grows_with_the_model(run, monkeypatch):
    from reliquary.rl_orders import pod_delivery as pods

    assert pods.MAX_TOTAL_BYTES // pods.MIN_COPY_BYTES_PER_S > pods.FREEZE_SECONDS * 2


def test_a_changed_publication_during_the_copy_publishes_nothing(run):
    store, _, source, _ = run
    original, reads = source.read_index, []

    def moved(run_id):
        reads.append(run_id)
        raw = original(run_id)
        if len(reads) == 1:
            return raw
        document = json.loads(raw)
        document["anchors"][0]["model_files"] = document["anchors"][0]["model_files"][:1]
        return json.dumps(document).encode()

    source.read_index = moved
    with pytest.raises((CheckpointUnavailable, ValueError)):
        store.freeze(_entry(), 2)
    assert store.load_manifest(TASK) is None


def test_a_tampered_source_object_publishes_nothing(run):
    store, _, source, _ = run
    key = next(k for k in source.objects if k.endswith(".safetensors"))
    source.objects[key] = source.objects[key][:-1] + bytes([source.objects[key][-1] ^ 1])
    with pytest.raises(CheckpointUnavailable):
        store.freeze(_entry(), 2)
    assert store.load_manifest(TASK) is None


def test_a_frozen_order_cannot_be_frozen_again_at_another_version(run):
    store, *_ = run
    store.freeze(_entry(), 2)
    with pytest.raises(CheckpointConflict):
        store.freeze(_entry(), 3)
    with pytest.raises(CheckpointConflict):
        store.freeze(_entry(order_sha256="b" * 64), 2)


def test_a_large_object_uploads_in_equal_parts_and_reads_across_chunks():
    bucket = Bucket()
    store = PodCheckpointStore(bucket, "private")
    data = bytes(range(256)) * (CHUNK_BYTES * 2 // 256) + b"tail"
    artifact = store._copy(TASK, kind="weights", name="model.safetensors", size=len(data),
                           sha256=hashlib.sha256(data).hexdigest(), deadline=float("inf"),
                           source=[data[i:i + 1_000_000] for i in range(0, len(data), 1_000_000)])
    assert bucket.part_sizes == [CHUNK_BYTES, CHUNK_BYTES, 4]
    assert [c["size"] for c in artifact["chunks"]] == [CHUNK_BYTES, CHUNK_BYTES, 4]
    assert bucket.objects[f"{PREFIX}{TASK}/objects/{artifact['sha256']}"] == data


def test_a_model_over_the_local_caches_limit_fits_the_pod_manifest():
    from reliquary.rl_orders import checkpoint_delivery as local
    from reliquary.rl_orders import pod_delivery as pods

    assert pods.MAX_TOTAL_BYTES == 128 * 1024 ** 3 > local.MAX_TOTAL_BYTES
    chunk = {"offset": pods.MAX_TOTAL_BYTES - CHUNK_BYTES, "size": CHUNK_BYTES, "sha256": "a" * 64}
    assert len(canonical_json_bytes([chunk] * pods.MAX_CHUNKS)) < pods.MANIFEST_BYTES - 128 * 400


def test_a_failed_large_copy_aborts_its_upload():
    bucket = Bucket()
    store = PodCheckpointStore(bucket, "private")
    data = b"x" * (CHUNK_BYTES + 5)
    with pytest.raises(CheckpointUnavailable):
        store._copy(TASK, kind="weights", name="model.safetensors", size=len(data), sha256="0" * 64,
                    source=[data], deadline=float("inf"))
    assert len(bucket.aborted) == 1 and not bucket.objects


def test_reads_are_bounded_and_fail_closed_on_a_changed_object(run):
    store, bucket, *_ = run
    manifest = store.freeze(_entry(), 2)
    artifact = next(a for a in manifest["files"] if a["kind"] == "weights")
    read = store.open_range(TASK, artifact["artifact_id"], "bytes=1-3")
    assert read.partial and read.total == artifact["size"] and len(read.data) == 3
    for bad in ("bytes=0-", "bytes=1-2,4-5", f"bytes=0-{artifact['size']}"):
        with pytest.raises(CheckpointRangeError):
            store.open_range(TASK, artifact["artifact_id"], bad)
    with pytest.raises(FileNotFoundError):
        store.open_range(TASK, "f" * 64)
    key = f"{PREFIX}{TASK}/objects/{artifact['sha256']}"
    bucket.objects[key] = bytes([bucket.objects[key][0] ^ 1]) + bucket.objects[key][1:]
    with pytest.raises(CheckpointUnavailable):
        store.open_range(TASK, artifact["artifact_id"], "bytes=1-3")


def _deliver(store, *, final=True, version=2, executors=()):
    return deliver_order(_entry(), store, is_final=lambda entry: final,
                         read_summary=lambda run: {"task_id": TASK, "latest_checkpoint": version},
                         read_executors=lambda: list(executors), clock=lambda: 100.0)


def test_nothing_is_copied_before_the_bill_is_final_or_while_a_pod_holds_a_token(run):
    store, bucket, *_ = run
    live = {"scope": "trainer", "task_id": TASK, "status": "active", "expires_at": 200.0}
    assert _deliver(store, final=False) == "waiting"
    assert _deliver(store, executors=[live]) == "waiting"
    assert not bucket.objects
    assert _deliver(store, executors=[{**live, "status": "revoked"}, {**live, "expires_at": 50.0},
                                      {**live, "task_id": "other"}]) == "delivered"
    assert store.load_manifest(TASK)["checkpoint"]["version"] == 2


@pytest.mark.parametrize("version", [None, 0])
def test_an_order_that_trained_nothing_has_nothing_to_deliver(run, version):
    store, bucket, *_ = run
    assert _deliver(store, version=version) == "no_steps"
    assert not bucket.objects


def test_trainer_tokens_ignore_other_scopes():
    document = {"scope": "eval", "task_id": TASK, "status": "active", "expires_at": 200.0}
    assert not trainer_tokens_live(TASK, [document], now=100.0)


def test_deliveries_retry_a_failure_later_and_stop_once_done():
    now, calls, outcomes = [1000.0], [], [RuntimeError("r2 down"), "waiting", "delivered"]

    def deliver(entry):
        calls.append(entry.task_id)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    deliveries = PodDeliveries(deliver, clock=lambda: now[0])
    assert deliveries.offer(_entry())
    deliveries.join(5)
    assert not deliveries.offer(_entry())                         # backing off
    now[0] += 61
    assert deliveries.offer(_entry())
    deliveries.join(5)
    assert not deliveries.offer(_entry())                         # waiting
    now[0] += 61
    assert deliveries.offer(_entry())
    deliveries.join(5)
    now[0] += 7200
    assert not deliveries.offer(_entry()) and len(calls) == 3


class Offers:
    def __init__(self):
        self.seen = []

    def offer(self, entry):
        self.seen.append(entry.task_id)
        return False


class Proc:
    def __init__(self):
        self.code = None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 0

    def kill(self):
        self.code = -9


def test_the_supervisor_offers_only_stopped_retired_pod_orders(tmp_path):
    offers, final = Offers(), set()
    entries = {
        "pod-running": SimpleNamespace(task_id="pod-running", status="active", params={"training_order": {}}),
        "pod-retired": SimpleNamespace(task_id="pod-retired", status="retired", params={"training_order": {}}),
        "plain": SimpleNamespace(task_id="plain", status="retired", params={}),
    }
    sup = Supervisor(lambda: entries, lambda *a: Proc(), ports=range(9100, 9105), map_path=tmp_path / "map.json",
                     clock=lambda: 1000.0, final_of=lambda e: e.task_id in final, deliveries=offers)
    sup.tick()
    assert offers.seen == []                                      # its controller still runs until the bill is final
    final.add("pod-retired")
    sup.tick()                                                    # stopped on this tick
    sup.tick()
    assert set(offers.seen) == {"pod-retired"}
