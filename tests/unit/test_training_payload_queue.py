"""Durable local queue feeding the R2 ``reliquary/training/`` prefix."""

import asyncio

from reliquary.infrastructure.training_payload_queue import (
    TrainingPayloadQueue,
    payload_key,
    tombstone_key,
)


def test_keys():
    assert payload_key(30100) == "reliquary/training/window-30100.npz"
    assert tombstone_key(30100) == (
        "reliquary/training/window-30100.tombstone.json"
    )


def test_enqueue_writes_atomically(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    q.enqueue_tombstone(30101, b"{}")
    assert (tmp_path / "window-30100.npz").read_bytes() == b"abc"
    assert (tmp_path / "window-30101.tombstone.json").read_bytes() == b"{}"
    assert not list(tmp_path.glob("*.tmp"))


def test_worker_uploads_and_deletes(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    q.enqueue_tombstone(30101, b"{}")
    uploaded = {}

    def fake_upload(key, data):
        uploaded[key] = data

    asyncio.run(q.drain_once(upload_fn=fake_upload))
    assert uploaded == {
        "reliquary/training/window-30100.npz": b"abc",
        "reliquary/training/window-30101.tombstone.json": b"{}",
    }
    assert not list(tmp_path.glob("window-*"))
    assert q.snapshot()["uploads_succeeded_total"] == 2


def test_failed_upload_keeps_file_and_backs_off(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")

    def bad_upload(key, data):
        raise RuntimeError("r2 down")

    asyncio.run(q.drain_once(upload_fn=bad_upload))
    assert (tmp_path / "window-30100.npz").exists()
    snap = q.snapshot()
    assert snap["upload_failures_total"] == 1
    assert snap["depth"] == 1


def test_restart_rescan_picks_up_pending(tmp_path):
    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q1.enqueue_payload(30100, b"abc")
    # Fresh instance over the same dir (process restart).
    q2 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    uploaded = {}
    asyncio.run(q2.drain_once(upload_fn=lambda k, d: uploaded.update({k: d})))
    assert "reliquary/training/window-30100.npz" in uploaded
