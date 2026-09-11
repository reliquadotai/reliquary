"""Two tasks must never write the same archive key.

The window number is a per-process counter, so without a namespace a second
task would overwrite the first task's window 42 and both would land in the
same weight replay.
"""

from __future__ import annotations

import pytest

from reliquary.infrastructure import storage


def test_the_legacy_task_keeps_its_flat_path(monkeypatch):
    monkeypatch.delenv("RELIQUARY_TASK_ID", raising=False)

    assert storage.dataset_prefix() == "reliquary/dataset/window-"
    assert storage.dataset_object_key(42) == "reliquary/dataset/window-42.json.gz"


def test_default_is_spelled_the_same_as_unset(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "default")

    assert storage.dataset_object_key(42) == "reliquary/dataset/window-42.json.gz"


def test_a_named_task_writes_under_its_own_prefix(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert storage.dataset_prefix() == "reliquary/tasks/logic-probe/dataset/window-"
    assert storage.dataset_object_key(42) == "reliquary/tasks/logic-probe/dataset/window-42.json.gz"


def test_an_explicit_task_beats_the_environment(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert storage.dataset_object_key(7, "other") == "reliquary/tasks/other/dataset/window-7.json.gz"


@pytest.mark.parametrize("bad", ["../escape", "UPPER", "with space", "a" * 64, "-lead"])
def test_an_unusable_task_id_is_refused(monkeypatch, bad):
    monkeypatch.setenv("RELIQUARY_TASK_ID", bad)

    with pytest.raises(ValueError):
        storage.dataset_prefix()


def test_the_queue_uploads_to_the_same_key(monkeypatch):
    from reliquary.infrastructure import archive_queue

    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert archive_queue.upload_key(42) == storage.dataset_object_key(42)


@pytest.mark.asyncio
async def test_the_archive_says_which_task_wrote_it(monkeypatch):
    from unittest.mock import MagicMock, patch

    from tests.unit.test_archive_window_content import (
        _FakeEnv,
        _FakeWallet,
        _valid_submission,
    )
    from reliquary.validator.service import ValidationService

    monkeypatch.setattr("reliquary.validator.service.TASK_ID", "logic-probe")
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    service = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=tokenizer,
        env=_FakeEnv(), netuid=99,
    )
    submission = _valid_submission(prompt_idx=7)
    captured: dict = {}

    class _StubQueue:
        def enqueue(self, window, archive):
            captured["archive"] = archive

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []

        def valid_submissions(self):
            return [submission]

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await service._archive_window(_FakeBatcher(), [submission])

    assert captured["archive"]["task_id"] == "logic-probe"
