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
    # Patched in the globals of every loaded `upload_window_dataset`, not only on
    # this file's module object: another test may have reimported storage.
    keys: list[str] = []
    fake = lambda bucket, key, body, *rest: keys.append(key)  # noqa: E731
    for module in list(sys.modules.values()):
        upload = getattr(module, "upload_window_dataset", None)
        if upload is not None and hasattr(upload, "__globals__"):
            monkeypatch.setitem(upload.__globals__, "_sync_boto3_put", fake)
    monkeypatch.setattr(storage, "_sync_boto3_put", fake)
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


def test_the_settler_archives_each_served_task_under_its_own_prefix(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    keys = _capture_puts(monkeypatch)

    asyncio.run(R2Archives().write("corpus-code", 3, {}))
    asyncio.run(R2Archives().write("corpus-math", 4, {}))

    assert keys == ["reliquary/tasks/corpus-code/dataset/window-3.json.gz",
                    "reliquary/tasks/corpus-math/dataset/window-4.json.gz"]


def test_the_settler_refuses_a_task_this_process_does_not_serve(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    keys = _capture_puts(monkeypatch)

    with pytest.raises(RuntimeError, match="refusing to archive"):
        asyncio.run(R2Archives().write("corpus-other", 3, {}))
    assert keys == []


def test_the_settler_still_refuses_another_task_with_one_id(monkeypatch):
    from reliquary.validator.corpus_settlement import R2Archives

    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math")
    keys = _capture_puts(monkeypatch)

    with pytest.raises(RuntimeError, match="refusing to archive"):
        asyncio.run(R2Archives().write("corpus-code", 3, {}))
    asyncio.run(R2Archives().write("corpus-math", 3, {}))
    assert keys == ["reliquary/tasks/corpus-math/dataset/window-3.json.gz"]


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
