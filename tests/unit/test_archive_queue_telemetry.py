from __future__ import annotations

import asyncio
import gzip
import json
from unittest.mock import patch

import pytest

from reliquary.infrastructure.archive_queue import ArchiveQueue


def test_archive_queue_snapshot_tracks_pending_files(tmp_path):
    queue = ArchiveQueue(str(tmp_path))
    first = queue.enqueue(41, {"window_start": 41})
    queue.enqueue(42, {"window_start": 42})
    now = first.stat().st_mtime + 12.5

    snapshot = queue.snapshot(now=now)

    assert snapshot["depth"] == 2
    assert snapshot["oldest_window"] == 41
    assert snapshot["oldest_age_seconds"] == 12.5
    assert snapshot["uploads_succeeded_total"] == 0
    assert snapshot["upload_failures_total"] == 0
    assert snapshot["last_enqueued_window"] == 42
    assert snapshot["archives_enqueued_total"] == 2
    assert snapshot["enqueue_gaps_total"] == 0


def test_archive_queue_reports_enqueue_continuity_gap(tmp_path):
    queue = ArchiveQueue(str(tmp_path))
    queue.enqueue(50, {"window_start": 50})
    queue.enqueue(52, {"window_start": 52})

    snapshot = queue.snapshot()

    assert snapshot["enqueue_gaps_total"] == 1
    assert snapshot["last_enqueue_gap"] == {"after": 50, "before": 52}


def test_archive_queue_replays_pending_bodies_by_exact_window(tmp_path):
    queue = ArchiveQueue(str(tmp_path))
    queue.enqueue(50, {"window_start": 50, "window_status": "aborted"})
    queue.enqueue(51, {"window_start": 51, "batch": []})

    assert queue.pending_window_numbers() == (50, 51)
    assert queue.pending_archives(start_window=51, end_window=51) == {
        51: {"window_start": 51, "batch": []}
    }


def test_archive_queue_rejects_body_key_identity_mismatch(tmp_path):
    queue = ArchiveQueue(str(tmp_path))

    with pytest.raises(ValueError, match="must match the queue identity"):
        queue.enqueue(50, {"window_start": 49})

    assert not (tmp_path / "window-50.json.gz").exists()
    assert not (tmp_path / "window-50.json.gz.tmp").exists()


@pytest.mark.parametrize("window_start", [True, -1, 1.0, "1"])
def test_archive_queue_rejects_noncanonical_queue_identity_before_write(
    tmp_path,
    window_start,
):
    queue = ArchiveQueue(str(tmp_path))

    with pytest.raises(ValueError, match="non-negative integer"):
        queue.enqueue(window_start, {"window_start": window_start})

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [None, [], "archive"])
def test_archive_queue_rejects_non_dictionary_body_before_write(
    tmp_path,
    payload,
):
    queue = ArchiveQueue(str(tmp_path))

    with pytest.raises(TypeError, match="must be a dictionary"):
        queue.enqueue(1, payload)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("body_window", [True, -1, 1.0, "1"])
def test_archive_queue_rejects_noncanonical_body_identity_before_write(
    tmp_path,
    body_window,
):
    queue = ArchiveQueue(str(tmp_path))

    with pytest.raises(ValueError, match="non-negative integer"):
        queue.enqueue(1, {"window_start": body_window})

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "filename",
    [
        "window-True.json.gz",
        "window--1.json.gz",
        "window-01.json.gz",
        "window-1.foo.json.gz",
    ],
)
def test_archive_queue_rejects_noncanonical_pending_filename(
    tmp_path,
    filename,
):
    queue = ArchiveQueue(str(tmp_path))
    body = gzip.compress(json.dumps({"window_start": 1}).encode())
    (tmp_path / filename).write_bytes(body)

    with pytest.raises(RuntimeError, match="invalid pending archive path"):
        queue.pending_window_numbers()
    with pytest.raises(RuntimeError, match="invalid pending archive path"):
        queue.pending_archives(start_window=0, end_window=10)


@pytest.mark.parametrize("body_window", [True, -1, 1.0, "1"])
def test_archive_queue_rejects_noncanonical_pending_body_identity(
    tmp_path,
    body_window,
):
    queue = ArchiveQueue(str(tmp_path))
    body = gzip.compress(json.dumps({"window_start": body_window}).encode())
    (tmp_path / "window-1.json.gz").write_bytes(body)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        queue.pending_window_numbers()


def test_archive_queue_failed_replace_leaves_no_visible_commit(tmp_path):
    queue = ArchiveQueue(str(tmp_path))

    with patch("os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            queue.enqueue(50, {"window_start": 50})

    assert not (tmp_path / "window-50.json.gz").exists()
    assert not (tmp_path / "window-50.json.gz.tmp").exists()


def test_archive_queue_snapshot_tracks_success(tmp_path, monkeypatch):
    queue = ArchiveQueue(str(tmp_path))
    path = queue.enqueue(43, {"window_start": 43})
    monkeypatch.setattr(
        "reliquary.infrastructure.storage._sync_boto3_put",
        lambda *args, **kwargs: None,
    )

    assert asyncio.run(queue._try_upload(path)) is True
    snapshot = queue.snapshot()

    assert snapshot["depth"] == 0
    assert snapshot["uploads_succeeded_total"] == 1
    assert snapshot["upload_failures_total"] == 0
    assert snapshot["last_uploaded_window"] == 43
    assert snapshot["last_upload_success_ts"] is not None


def test_archive_queue_snapshot_tracks_failure_without_dropping(tmp_path, monkeypatch):
    queue = ArchiveQueue(str(tmp_path))
    path = queue.enqueue(44, {"window_start": 44})

    def fail(*args, **kwargs):
        raise TimeoutError("r2 unavailable")

    monkeypatch.setattr(
        "reliquary.infrastructure.storage._sync_boto3_put",
        fail,
    )

    assert asyncio.run(queue._try_upload(path)) is False
    snapshot = queue.snapshot()

    assert snapshot["depth"] == 1
    assert snapshot["uploads_succeeded_total"] == 0
    assert snapshot["upload_failures_total"] == 1
    assert snapshot["last_failed_window"] == 44
    assert snapshot["last_upload_failure_ts"] is not None
