"""Hugging Face backups: rotation policy and a background uploader that never
blocks the trainer."""

from __future__ import annotations

import threading
import time

import pytest

from reliquary.trainer.hf_backup import BackgroundHfBackup, plan_backup_rotation


def test_rotation_keeps_the_rolling_window_and_milestones():
    steps = [100, 200, 300, 400, 500, 600, 700]
    kept, deleted = plan_backup_rotation(steps, keep_last=2, milestone_every=500,
                                         milestones=[200])
    assert kept == [200, 500, 600, 700]
    assert deleted == [100, 300, 400]


def test_rotation_is_order_insensitive_and_never_deletes_the_newest():
    kept, deleted = plan_backup_rotation([700, 100, 700, 300], keep_last=1)
    assert kept == [700] and deleted == [100, 300]
    assert plan_backup_rotation([], keep_last=3) == ([], [])
    with pytest.raises(ValueError):
        plan_backup_rotation([1], keep_last=0)


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_submit_never_blocks_and_only_the_newest_waits(tmp_path):
    release, busy = threading.Event(), threading.Event()
    on_hub, deleted = set(), []

    def upload(step, folder):
        assert (folder / "weights").read_text() == str(step)
        busy.set()
        release.wait(5)
        on_hub.add(step)

    def delete(step):
        on_hub.discard(step)
        deleted.append(step)

    backup = BackgroundHfBackup(upload=upload, list_backups=lambda: list(on_hub),
                                delete_backup=delete, keep_last=2,
                                milestone_every=10, staging_dir=tmp_path)

    def save_for(step):
        return lambda folder: (folder / "weights").write_text(str(step))

    started = time.monotonic()
    backup.submit(1, save_for(1))
    busy.wait(5)
    for step in (2, 3, 4):
        backup.submit(step, save_for(step))
    assert time.monotonic() - started < 0.5  # upload 1 is stuck, submit is not
    release.set()
    _wait(lambda: backup.completed == [1, 4])
    assert backup.dropped == [2, 3]
    for step in (5, 10, 11):
        backup.submit(step, save_for(step))
        _wait(lambda: step in backup.completed)
    backup.close(timeout=5)
    assert deleted == [1, 4, 5]  # rolling window of 2; 10 is a milestone
    assert on_hub == {10, 11}
    assert not list(tmp_path.iterdir())  # staging cleaned


def test_a_failed_backup_leaves_the_worker_running(tmp_path):
    calls = []

    def upload(step, folder):
        calls.append(step)
        if step == 1:
            raise RuntimeError("hf down")

    backup = BackgroundHfBackup(upload=upload, list_backups=lambda: [2],
                                delete_backup=lambda s: None, keep_last=1,
                                staging_dir=tmp_path)
    backup.submit(1, lambda folder: None)
    _wait(lambda: calls == [1])
    backup.submit(2, lambda folder: None)
    _wait(lambda: backup.completed == [2])
    backup.close(timeout=5)


def test_a_missing_staging_dir_is_created_before_the_first_backup(tmp_path):
    # Production 2026-10-22: <state>/async/hf-staging did not exist on a fresh volume, so every
    # backup failed in mkdtemp before writing anything.
    staging = tmp_path / "async" / "hf-staging"
    uploaded = []

    def upload(step, folder):
        assert folder.parent == staging
        uploaded.append((step, (folder / "weights").read_text()))

    backup = BackgroundHfBackup(upload=upload, list_backups=lambda: [1],
                                delete_backup=lambda s: None, keep_last=1, staging_dir=staging)
    backup.submit(1, lambda folder: (folder / "weights").write_text("1"))
    _wait(lambda: backup.completed == [1])
    backup.close(timeout=5)
    assert uploaded == [(1, "1")]
    assert staging.is_dir() and not list(staging.iterdir())
    assert staging.stat().st_mode & 0o777 == 0o700


def test_a_staging_dir_removed_between_backups_is_recreated(tmp_path):
    import shutil

    staging = tmp_path / "hf-staging"
    backup = BackgroundHfBackup(upload=lambda step, folder: None, list_backups=lambda: [],
                                delete_backup=lambda s: None, keep_last=1, staging_dir=staging)
    backup.submit(1, lambda folder: None)
    _wait(lambda: backup.completed == [1])
    shutil.rmtree(staging)
    backup.submit(2, lambda folder: None)
    _wait(lambda: backup.completed == [1, 2])
    backup.close(timeout=5)
