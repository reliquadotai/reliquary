"""Final checkpoint bytes remain reconstructable and owner-safe after the live index changes."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path

import pytest
import torch

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.rl_orders.checkpoint_delivery import (
    CHUNK_BYTES, MAX_READ_BYTES, ArtifactRead, CheckpointConflict, CheckpointRangeError,
    CheckpointUnavailable, R2CheckpointSource, TrainingCheckpointStore,
)
from reliquary.rl_orders.supervisor import StoppedProcess, StoppedRuntime
from reliquary.shared.ckpt_delta import apply_delta, decode_delta, load_anchor_shards, verify_checkpoint
from reliquary.shared.ckpt_index import index_key, plan_download
from reliquary.trainer.ckpt_publisher import DeltaPublisher
from reliquary.validator.rl_training_usage import (
    BILLING_BASIS, TOKEN_METRIC, TRAINER_METRIC, USAGE_SCHEMA, stopped_document,
)


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.reads = []

    def get_bytes(self, key):
        return self.objects.get(key)

    def put_bytes(self, key, data):
        self.objects[key] = data

    def put_file(self, key, path):
        self.objects[key] = path.read_bytes()

    def delete(self, key):
        self.objects.pop(key, None)

    def read_index(self, run_id):
        self.reads.append(index_key(run_id))
        return self.objects.get(index_key(run_id))

    def iter_object(self, key, expected_size):
        self.reads.append(key)
        value = self.objects[key]
        for start in range(0, len(value), CHUNK_BYTES):
            yield value[start:start + CHUNK_BYTES]


def envelope(stopped, reference, **overrides):
    receipt = {"schema": USAGE_SCHEMA, "billing_basis": BILLING_BASIS, "token_metric": TOKEN_METRIC,
               "trainer_metric": TRAINER_METRIC, "task_id": stopped.task_id, "training_id": "example",
               "order_sha256": stopped.order_sha256, "quote_id": "quote-1", "quote_sha256": "1" * 64,
               "run_id": stopped.run_id, "runtime_id": stopped.runtime_id, "final": True, "drained": True,
               "matched_tokens": 100, "trainer_active_ms": 10000, "finished_at": 21, "event_cursor": 1,
               "accrual_count": 1, "consumption_sha256": "b" * 64,
               "termination_sha256": canonical_sha256(stopped_document(stopped)), "checkpoint": reference,
               **overrides}
    return {"receipt": receipt, "sha256": canonical_sha256(receipt)}


@pytest.fixture
def context(tmp_path, monkeypatch):
    def absent(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(os, "killpg", absent)
    source = MemoryStore()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    (model / "tokenizer.json").write_text('{"version":"1.0"}')
    publisher = DeltaPublisher(run="example-run", store=source, staging_dir=tmp_path / "staging",
                               model_dir=model, anchor_interval=50)
    state = {"w": torch.tensor([1, 2, 3], dtype=torch.bfloat16)}
    publisher.publish(0, state)
    state = {"w": torch.tensor([1, 4, 3], dtype=torch.bfloat16)}
    publisher.publish(1, state)
    state = {"w": torch.tensor([1, 4, 9], dtype=torch.bfloat16)}
    publisher.publish(2, state)
    stopped = StoppedRuntime(
        "order-train-example", "a" * 64, "example-run", "runtime-1", 9.0, 21.0, 10_000_000_000, 2,
        (StoppedProcess("controller", 900000001, 900000001, "controller-start", 0, 10.0, 20.0),
         StoppedProcess("trainer", 900000002, 900000002, "trainer-start", 0, 10.0, 20.0)),
        cgroup_id="runtime-1", cgroup_empty_at=21.0)
    store = TrainingCheckpointStore(tmp_path / "runtime", source)
    reference = store.checkpoint_reference(stopped)
    return store, source, stopped, envelope(stopped, reference), state


def reconstruction(store, manifest):
    by_id = {artifact["artifact_id"]: artifact for artifact in manifest["files"]}
    index_artifact = next(artifact for artifact in manifest["files"] if artifact["kind"] == "index")
    index = json.loads(store.open_range(manifest["task_id"], index_artifact["artifact_id"]).data)
    return index, by_id


def test_freeze_copies_a_complete_lossless_checkpoint_without_exposing_storage(context, tmp_path):
    store, source, stopped, usage, expected = context
    manifest = store.freeze(stopped, usage)
    assert store.load_manifest(stopped.task_id) == manifest
    assert manifest["usage_sha256"] == usage["sha256"]
    assert manifest["index_sha256"] == usage["receipt"]["checkpoint"]["index_sha256"]
    raw = canonical_json_bytes(manifest)
    assert b"rl/example-run/" not in raw and str(tmp_path).encode() not in raw and b"https://" not in raw
    index, by_id = reconstruction(store, manifest)
    plan = plan_download(index, None)
    paths = []
    for file in plan.anchor["files"]:
        artifact = by_id[file["key"].removeprefix("objects/")]
        path = tmp_path / artifact["name"]
        path.write_bytes(store.open_range(stopped.task_id, artifact["artifact_id"]).data)
        paths.append(path)
    actual = load_anchor_shards(paths)
    for file in plan.deltas:
        artifact_id = file["key"].removeprefix("objects/")
        apply_delta(actual, decode_delta(store.open_range(stopped.task_id, artifact_id).data))
    verify_checkpoint(actual, plan.target_sha256)
    assert torch.equal(actual["w"], expected["w"])
    assert {file["name"] for file in manifest["files"] if file["kind"] == "model"} == {"config.json", "tokenizer.json"}
    # The mutable transport and its pruning no longer control this delivery.
    source.objects.clear()
    source.reads.clear()
    assert store.freeze(stopped, usage) == manifest and source.reads == []
    assert store.open_range(stopped.task_id, manifest["files"][0]["artifact_id"]).data


@pytest.mark.parametrize("field,value", [("task_id", "order-train-other"), ("run_id", "other"),
                                          ("runtime_id", "other"), ("order_sha256", "b" * 64),
                                          ("final", False), ("drained", False), ("checkpoint", None),
                                          ("schema", "other"), ("termination_sha256", "c" * 64)])
def test_unbound_or_nonfinal_usage_cannot_publish_a_checkpoint(context, field, value):
    store, _, stopped, usage, _ = context
    receipt = {**usage["receipt"], field: value}
    with pytest.raises(ValueError):
        store.freeze(stopped, {"receipt": receipt, "sha256": canonical_sha256(receipt)})
    assert store.load_manifest(stopped.task_id) is None


def test_index_change_between_usage_and_materialization_is_refused(context):
    store, source, stopped, usage, _ = context
    index = json.loads(source.objects[index_key(stopped.run_id)])
    index["published_at"] += 1
    source.objects[index_key(stopped.run_id)] = canonical_json_bytes(index)
    with pytest.raises(ValueError, match="index differs"):
        store.freeze(stopped, usage)
    assert store.load_manifest(stopped.task_id) is None


def test_unrelated_model_directory_files_are_not_customer_artifacts(context):
    store, source, stopped, _, _ = context
    index = json.loads(source.objects[index_key(stopped.run_id)])
    extra_key = "rl/example-run/anchors/00000000/model/operator-state.json"
    index["anchors"][0]["model_files"].append({
        "name": "operator-state.json", "key": extra_key, "size": 10, "sha256": "f" * 64,
    })
    source.objects[index_key(stopped.run_id)] = canonical_json_bytes(index)
    manifest = store.freeze(stopped, envelope(stopped, store.checkpoint_reference(stopped)))
    assert extra_key not in source.reads
    assert "operator-state.json" not in {artifact["name"] for artifact in manifest["files"]}
    rebuilt, _ = reconstruction(store, manifest)
    assert "operator-state.json" not in {file["name"] for file in rebuilt["anchors"][0]["model_files"]}


def test_empty_optional_model_file_has_a_valid_empty_full_read(context):
    store, source, stopped, _, _ = context
    index = json.loads(source.objects[index_key(stopped.run_id)])
    key = "rl/example-run/anchors/00000000/model/chat_template.jinja"
    source.objects[key] = b""
    index["anchors"][0]["model_files"].append({
        "name": "chat_template.jinja", "key": key, "size": 0, "sha256": hashlib.sha256(b"").hexdigest(),
    })
    source.objects[index_key(stopped.run_id)] = canonical_json_bytes(index)
    manifest = store.freeze(stopped, envelope(stopped, store.checkpoint_reference(stopped)))
    file = next(file for file in manifest["files"] if file["name"] == "chat_template.jinja")
    assert file["chunks"] == []
    result = store.open_range(stopped.task_id, file["artifact_id"])
    assert result.data == b"" and result.total == 0 and not result.partial


def test_index_change_during_copy_never_publishes_a_manifest(context):
    store, source, stopped, usage, _ = context
    original = source.iter_object
    changed = False
    def changing(key, size):
        nonlocal changed
        yield from original(key, size)
        if not changed:
            changed = True
            index = json.loads(source.objects[index_key(stopped.run_id)])
            index["published_at"] += 1
            source.objects[index_key(stopped.run_id)] = canonical_json_bytes(index)
    source.iter_object = changing
    with pytest.raises(CheckpointUnavailable, match="changed during freeze"):
        store.freeze(stopped, usage)
    assert store.load_manifest(stopped.task_id) is None


def test_freeze_deadline_leaves_no_partial_manifest_or_temporary_files(context):
    store, _, stopped, usage, _ = context
    now = 0
    def clock():
        nonlocal now
        now += 1
        return now
    store.clock = clock
    with pytest.raises(CheckpointUnavailable, match="deadline"):
        store.freeze(stopped, usage, timeout_seconds=1)
    assert store.load_manifest(stopped.task_id) is None
    assert not list((store.root / stopped.task_id / "checkpoint" / "objects").glob(".checkpoint-*"))


@pytest.mark.parametrize("change", ["missing", "tampered", "oversize", "wrong_run", "wrong_version", "traversal", "wrong_key", "no_path", "no_tokenizer"])
def test_publication_must_be_present_bound_reachable_and_verified(context, change):
    store, source, stopped, _, _ = context
    index = json.loads(source.objects[index_key(stopped.run_id)])
    weight = index["anchors"][0]["files"][0]
    if change == "missing":
        del source.objects[weight["key"]]
    elif change == "tampered":
        source.objects[weight["key"]] = b"corrupt"
    elif change == "oversize":
        weight["size"] = 16 * 1024 ** 3 + 1
    elif change == "wrong_run":
        index["run"] = "other"
    elif change == "wrong_version":
        index["latest"]["version"] = 3
    elif change == "traversal":
        weight["key"] = "rl/example-run/anchors/00000000/../weights.safetensors"
    elif change == "wrong_key":
        weight["key"] = "rl/example-run/observations/secret.safetensors"
    elif change == "no_path":
        index["deltas"] = []
    else:
        index["anchors"][0]["model_files"] = []
    source.objects[index_key(stopped.run_id)] = canonical_json_bytes(index)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        reference = store.checkpoint_reference(stopped)
        store.freeze(stopped, envelope(stopped, reference))
    assert store.load_manifest(stopped.task_id) is None


def test_frozen_manifest_conflict_and_cache_tamper_fail_closed(context):
    store, source, stopped, usage, _ = context
    manifest = store.freeze(stopped, usage)
    changed = deepcopy(usage["receipt"])
    changed["matched_tokens"] += 1
    with pytest.raises(CheckpointConflict):
        store.freeze(stopped, {"receipt": changed, "sha256": canonical_sha256(changed)})
    artifact = manifest["files"][0]
    blob = store.root / stopped.task_id / "checkpoint" / "objects" / artifact["sha256"]
    blob.chmod(0o600)
    value = bytearray(blob.read_bytes())
    value[-1] ^= 1
    blob.write_bytes(value)
    with pytest.raises(CheckpointUnavailable, match="checksum"):
        store.open_range(stopped.task_id, artifact["artifact_id"])
    source.objects.clear()
    source.reads.clear()
    with pytest.raises(CheckpointUnavailable, match="cache differs"):
        store.freeze(stopped, usage)
    assert source.reads == []


def test_manifest_tamper_and_symlink_are_rejected(context, tmp_path):
    store, _, stopped, usage, _ = context
    manifest = store.freeze(stopped, usage)
    path = store.root / stopped.task_id / "checkpoint" / "manifest.json"
    path.chmod(0o600)
    path.write_bytes(canonical_json_bytes({**manifest, "runtime_id": "other"}))
    with pytest.raises(ValueError, match="checksum"):
        store.load_manifest(stopped.task_id)
    path.unlink()
    target = tmp_path / "foreign.json"
    target.write_bytes(canonical_json_bytes(manifest))
    path.symlink_to(target)
    with pytest.raises(OSError):
        store.load_manifest(stopped.task_id)


def test_unknown_artifact_and_invalid_ranges_never_address_paths(context):
    store, _, stopped, usage, _ = context
    manifest = store.freeze(stopped, usage)
    artifact = manifest["files"][0]
    for artifact_id in ("../private", "f" * 64):
        with pytest.raises(FileNotFoundError):
            store.open_range(stopped.task_id, artifact_id)
    for range_header in ("bytes=1-0", "bytes=-2", "bytes=1-", "bytes=0-9999999", "bytes=0-1,3-4", "bytes=" + "9" * 5000 + "-1"):
        with pytest.raises(CheckpointRangeError):
            store.open_range(stopped.task_id, artifact["artifact_id"], range_header)
    with pytest.raises(ValueError):
        store.load_manifest("../private")


def test_unknown_task_read_does_not_create_state(context):
    store, _, _, _, _ = context
    assert store.load_manifest("order-train-unknown") is None
    assert not (store.root / "order-train-unknown").exists()


def test_range_reader_verifies_chunks_before_returning_a_large_object(context, tmp_path):
    store, source, stopped, _, _ = context
    # A real safetensors anchor larger than an HTTP response; no model or GPU allocation.
    model = tmp_path / "model"
    source.objects.clear()
    publisher = DeltaPublisher(run=stopped.run_id, store=source, staging_dir=tmp_path / "large",
                               model_dir=model)
    publisher.publish(2, {"w": torch.zeros(9 * 1024 * 1024, dtype=torch.bfloat16)})
    usage = envelope(stopped, store.checkpoint_reference(stopped))
    manifest = store.freeze(stopped, usage)
    artifact = next(file for file in manifest["files"] if file["kind"] == "weights")
    assert artifact["size"] > MAX_READ_BYTES
    with pytest.raises(CheckpointRangeError):
        store.open_range(stopped.task_id, artifact["artifact_id"])
    start, end = CHUNK_BYTES - 10, CHUNK_BYTES + 20
    result = store.open_range(stopped.task_id, artifact["artifact_id"], f"bytes={start}-{end}")
    assert isinstance(result, ArtifactRead) and result.partial and len(result.data) == 31
    assert result.sha256 == hashlib.sha256(result.data).hexdigest()
    blob = store.root / stopped.task_id / "checkpoint" / "objects" / artifact["sha256"]
    blob.chmod(0o600)
    with blob.open("r+b") as handle:
        handle.seek(start - 100)
        handle.write(b"tamper")
    with pytest.raises(CheckpointUnavailable, match="checksum"):
        store.open_range(stopped.task_id, artifact["artifact_id"], f"bytes={start}-{end}")


def test_freeze_lock_and_private_cache_boundaries(context, tmp_path):
    store, _, stopped, usage, _ = context
    folder = store.root / stopped.task_id / "checkpoint"
    folder.parent.mkdir(mode=0o700, exist_ok=True)
    folder.mkdir(mode=0o700, exist_ok=True)
    with (folder / ".freeze.lock").open("wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CheckpointUnavailable, match="already running"):
            store.freeze(stopped, usage)
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(CheckpointUnavailable, match="private"):
        TrainingCheckpointStore(public)
    linked = tmp_path / "linked"
    linked.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(CheckpointUnavailable, match="private"):
        TrainingCheckpointStore(linked)


def test_r2_source_is_read_only_bounded_and_closes_bodies():
    responses = []
    class Client:
        def get_object(self, **kwargs):
            body = io.BytesIO(b"abc")
            responses.append((kwargs, body))
            return {"Body": body, "ContentLength": 3}
    source = R2CheckpointSource(Client(), "private-checkpoints")
    assert source.read_index("run-1") == b"abc"
    assert list(source.iter_object("rl/run-1/anchors/00000000/model.safetensors", 3)) == [b"abc"]
    assert all(body.closed for _, body in responses)
    assert responses[0][0]["Key"] == "rl/run-1/index.json"
    with pytest.raises(CheckpointUnavailable, match="size"):
        list(source.iter_object("rl/run-1/anchors/00000000/model.safetensors", 2))
    assert responses[-1][1].closed
