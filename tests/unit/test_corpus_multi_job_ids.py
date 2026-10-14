"""Several corpus task ids in one process: parsing, and archives per task."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from reliquary.infrastructure import storage
from reliquary.shared.task_id import parse_task_ids


def test_one_id_parses_to_itself():
    assert parse_task_ids("corpus-math") == ("corpus-math",)


def test_unset_or_empty_is_the_legacy_task_alone():
    assert parse_task_ids(None) == ("default",)
    assert parse_task_ids("") == ("default",)


def test_a_comma_list_keeps_its_order_and_strips_spaces():
    assert parse_task_ids(" corpus-math , corpus-code ") == ("corpus-math", "corpus-code")


@pytest.mark.parametrize("value", ["corpus-math,corpus-math", "corpus-math,,corpus-code",
                                   "corpus-math,UPPER", "corpus-math,"])
def test_a_duplicate_empty_or_malformed_member_is_refused(value):
    with pytest.raises(ValueError):
        parse_task_ids(value)


def test_constants_import_with_several_ids():
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env["RELIQUARY_TASK_ID"] = "corpus-math,corpus-code"
    completed = subprocess.run(
        [sys.executable, "-c",
         "from reliquary import constants as c; print(c.TASK_IDS, c.TASK_ID)"],
        capture_output=True, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "('corpus-math', 'corpus-code') corpus-math" in completed.stdout


def _capture_puts(monkeypatch) -> list[str]:
    keys: list[str] = []
    monkeypatch.setattr(storage, "_sync_boto3_put",
                        lambda bucket, key, body, *rest: keys.append(key))
    return keys


def test_upload_window_dataset_keys_by_an_explicit_task(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    keys = _capture_puts(monkeypatch)

    asyncio.run(storage.upload_window_dataset(7, {}, task_id="corpus-code"))

    assert keys == ["reliquary/tasks/corpus-code/dataset/window-7.json.gz"]


def test_upload_window_dataset_default_is_unchanged(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math")
    keys = _capture_puts(monkeypatch)

    asyncio.run(storage.upload_window_dataset(7, {}))

    assert keys == ["reliquary/tasks/corpus-math/dataset/window-7.json.gz"]


class _Period:
    """``R2PeriodArchives`` behind a validator's guard, over an in-memory bucket:
    the period archives a corpus settler writes."""

    def __init__(self, monkeypatch, guard):
        from reliquary.infrastructure import corpus_period_store
        from reliquary.infrastructure.corpus_period_store import R2PeriodArchives

        self.keys: list[str] = []

        async def read(task_id, work, entry):
            return None

        async def write(task_id, work, entry, document):
            self.keys.append(corpus_period_store.period_archive_key(task_id, work, entry))

        monkeypatch.setattr(corpus_period_store, "read_period_archive", read)
        monkeypatch.setattr(corpus_period_store, "write_period_archive", write)
        self.archives = R2PeriodArchives(guard=guard)

    def write(self, task_id, work, entry=None):
        asyncio.run(self.archives.write(task_id, work, work if entry is None else entry, {}))


def test_the_settler_archives_each_served_task_under_its_own_prefix(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    bucket = _Period(monkeypatch, R2Archives())
    bucket.write("corpus-code", 3)
    bucket.write("corpus-math", 4)
    assert bucket.keys == ["reliquary/corpus-periods/corpus-code/0000000003-0000000003.json.gz",
                           "reliquary/corpus-periods/corpus-math/0000000004-0000000004.json.gz"]


def test_the_settler_refuses_a_task_this_process_does_not_serve(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    bucket = _Period(monkeypatch, R2Archives())
    with pytest.raises(RuntimeError, match="refusing to archive"):
        bucket.write("corpus-other", 3)
    assert bucket.keys == []


def test_the_settler_still_refuses_another_task_with_one_id(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math")
    bucket = _Period(monkeypatch, R2Archives())
    with pytest.raises(RuntimeError, match="refusing to archive"):
        bucket.write("corpus-code", 3)
    bucket.write("corpus-math", 3)
    assert bucket.keys == ["reliquary/corpus-periods/corpus-math/0000000003-0000000003.json.gz"]


def test_explicit_archive_owner_tracks_actual_wired_tasks_without_inheriting_legacy_ids(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-legacy")
    served = set()
    bucket = _Period(monkeypatch, R2Archives(served=lambda: served, served_only=True))
    for task in ("corpus-legacy", "corpus-fresh"):
        with pytest.raises(RuntimeError, match="refusing to archive"):
            bucket.write(task, 3)
    monkeypatch.delenv("RELIQUARY_TASK_ID")
    served.add("corpus-fresh")
    bucket.write("corpus-fresh", 3)
    served.clear()
    with pytest.raises(RuntimeError, match="refusing to archive"):
        bucket.write("corpus-fresh", 4)
    assert bucket.keys == ["reliquary/corpus-periods/corpus-fresh/0000000003-0000000003.json.gz"]


@pytest.mark.parametrize("value", ["corpus-math,", "corpus-math,,corpus-code", "corpus-math,corpus-math"])
def test_a_malformed_task_id_list_exits_four_without_a_traceback(value):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env["RELIQUARY_TASK_ID"] = value
    completed = subprocess.run(
        [sys.executable, "-c", "import reliquary.constants"],
        capture_output=True, text=True, env=env,
    )
    assert completed.returncode == 4, completed.stderr
    assert "RELIQUARY_TASK_ID" in completed.stderr
    assert "Traceback" not in completed.stderr
