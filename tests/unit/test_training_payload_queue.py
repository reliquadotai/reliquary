"""Durable local queue feeding the R2 ``reliquary/training/`` prefix."""

import asyncio
import json

import pytest

from reliquary.infrastructure.training_payload_queue import (
    TrainingPayloadQueue,
    encoded_window_journal_key,
    payload_key,
    step_cursor_key,
    tombstone_key,
)


def test_keys():
    assert payload_key(30100) == "reliquary/training/window-30100.npz"
    assert tombstone_key(30100) == (
        "reliquary/training/window-30100.tombstone.json"
    )


def test_encoded_journal_key_is_the_raw_window_with_the_gate_off(
    monkeypatch,
):
    """v4/v5 must stay byte-for-byte unchanged: one payload per window,
    keyed by the window number itself, regardless of ``batch_index``."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", False)

    assert encoded_window_journal_key(30100) == 30100
    assert encoded_window_journal_key(30100, batch_index=0) == 30100
    # Even a nonzero batch_index is ignored with the gate off -- v4/v5
    # code never passes one, but the encoding must not apply regardless.
    assert encoded_window_journal_key(30100, batch_index=3) == 30100


def test_encoded_journal_key_under_fill_closed(monkeypatch):
    """R11: window_start * FILL_CLOSED_EMISSIONS_PER_WINDOW + batch_index,
    gated on FILL_CLOSED_ENABLED. Pins ``batch_index=3`` at
    ``window_start=42`` per the brief; ``FILL_CLOSED_EMISSIONS_PER_WINDOW``
    itself is monkeypatched to a fixed 16 so this test doesn't depend on
    the runtime PROTOCOL_VERSION (it derives from
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS, which is profile-dependent --
    see the separate derivation test below)."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)

    assert encoded_window_journal_key(42, batch_index=3) == 42 * 16 + 3
    assert encoded_window_journal_key(42, batch_index=0) == 42 * 16
    assert encoded_window_journal_key(43, batch_index=0) == 43 * 16
    # Consecutive integers within one window's range, exactly what the
    # trainer's cursor (stride=1) already walks -- no trainer change.
    keys = [
        encoded_window_journal_key(42, batch_index=i) for i in range(16)
    ]
    assert keys == list(range(42 * 16, 42 * 16 + 16))


def test_fill_closed_emissions_per_window_derives_from_publish_interval():
    """16 for the same reason FILL_CLOSED_TARGET_GROUPS_PER_ENV is 256:
    one B_BATCH-per-environment emission per optimizer step, and the
    target is CHECKPOINT_PUBLISH_INTERVAL_WINDOWS steps worth of groups."""
    from reliquary.constants import (
        CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
        FILL_CLOSED_EMISSIONS_PER_WINDOW,
    )

    assert (
        FILL_CLOSED_EMISSIONS_PER_WINDOW == CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
    )


def test_encoded_journal_key_rejects_out_of_range_batch_index(monkeypatch):
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    with pytest.raises(ValueError):
        encoded_window_journal_key(42, batch_index=16)
    with pytest.raises(ValueError):
        encoded_window_journal_key(42, batch_index=-1)


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


# ---------------- v6.1: trainer-paced picks step cursor ----------------


def test_step_cursor_absent_reads_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    assert q.read_step_cursor() is None


def test_step_cursor_round_trips(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    assert q.read_step_cursor() == 30142


def test_step_cursor_is_overwritten_not_appended(tmp_path):
    """Never a growing log: each write replaces the single object."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    q.write_step_cursor(30143)
    assert q.read_step_cursor() == 30143
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("step-cursor.json*"))) == 1


def test_step_cursor_corrupt_json_reads_as_stale_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    (tmp_path / "step-cursor.json").write_bytes(b"{not valid json")
    assert q.read_step_cursor() is None


def test_step_cursor_missing_field_reads_as_stale_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    (tmp_path / "step-cursor.json").write_bytes(b"{}")
    assert q.read_step_cursor() is None


def test_step_cursor_key_naming():
    assert step_cursor_key() == "reliquary/training/step-cursor.json"


def test_step_cursor_uploads_via_same_transport_and_stays_local(tmp_path):
    """The cursor rides the same drain/upload transport as payloads, but
    (unlike a payload) it is never deleted after upload -- it is a single
    overwritten object, not a consumed-once queue entry."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    uploaded = {}
    asyncio.run(q.drain_once(upload_fn=lambda k, d: uploaded.update({k: d})))
    assert set(uploaded) == {"reliquary/training/step-cursor.json"}
    assert json.loads(uploaded[step_cursor_key()])["journal_key"] == 30142
    assert (tmp_path / "step-cursor.json").exists()
