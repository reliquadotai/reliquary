from __future__ import annotations

import asyncio

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
