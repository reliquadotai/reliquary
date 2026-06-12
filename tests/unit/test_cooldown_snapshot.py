"""Run-keyed cooldown snapshot: restore + gap-replay, reset on a fresh run,
and the snapshot write shape. Storage is mocked — no R2."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class _FakeEnv:
    @property
    def name(self):
        return "fake"

    def __len__(self):
        return 100

    def get_problem(self, i):
        return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}

    def compute_reward(self, p, c):
        return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"

        @staticmethod
        def sign(d):
            return b"sig"

    hotkey = _Hk()


def _service(window_n: int):
    from reliquary.validator.service import ValidationService

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    svc._window_n = window_n
    return svc


@pytest.mark.asyncio
async def test_restore_from_snapshot_run_match():
    svc = _service(40)
    snap = {"run_id": "default", "snapshot_window": 40, "envs": {"fake": {"7": 30}}}
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=snap),
    ):
        await svc._rebuild_cooldown_from_history()
    assert svc._cooldown_per_env["fake"].is_in_cooldown(7, 40) is True


@pytest.mark.asyncio
async def test_restore_replays_gap_since_snapshot():
    svc = _service(45)
    snap = {"run_id": "default", "snapshot_window": 40, "envs": {"fake": {"7": 38}}}
    gap = [{"window_start": 43, "environment": "fake", "batch": [{"prompt_idx": 99}]}]
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=snap),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=gap),
    ):
        await svc._rebuild_cooldown_from_history()
    cd = svc._cooldown_per_env["fake"]
    assert cd.is_in_cooldown(7, 45) is True    # from snapshot
    assert cd.is_in_cooldown(99, 45) is True   # from gap-replay


@pytest.mark.asyncio
async def test_fresh_run_id_without_snapshot_resets_to_empty():
    svc = _service(40)
    list_mock = AsyncMock(return_value=[])
    with patch("reliquary.validator.service.TRAINING_RUN_ID", "run5"), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets", new=list_mock,
    ):
        await svc._rebuild_cooldown_from_history()
    assert len(svc._cooldown_per_env["fake"]) == 0  # reset to zero
    list_mock.assert_not_called()  # a fresh run must not rebuild from old archives


@pytest.mark.asyncio
async def test_default_run_without_snapshot_falls_back_to_archive():
    svc = _service(40)
    archives = [{"window_start": 38, "environment": "fake", "batch": [{"prompt_idx": 5}]}]
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=archives),
    ):
        await svc._rebuild_cooldown_from_history()
    assert svc._cooldown_per_env["fake"].is_in_cooldown(5, 40) is True


@pytest.mark.asyncio
async def test_snapshot_cooldown_writes_run_keyed_state():
    svc = _service(77)
    svc._cooldown_per_env["fake"].record_batched(7, 70)
    captured = {}

    async def fake_upload(key, data):
        captured["key"] = key
        captured["data"] = data
        return True

    with patch("reliquary.infrastructure.storage.upload_json", new=fake_upload):
        await svc._snapshot_cooldown()
    assert captured["key"] == "cooldown_snapshots/default.json"
    assert captured["data"]["run_id"] == "default"
    assert captured["data"]["snapshot_window"] == 77
    assert captured["data"]["envs"]["fake"] == {7: 70}


@pytest.mark.asyncio
async def test_corrupt_snapshot_does_not_crash_and_falls_back():
    """B2: a malformed snapshot (bad envs payload) must not crash startup — it
    is discarded and we fall back (empty for a fresh run)."""
    svc = _service(40)
    corrupt = {"run_id": "run5", "snapshot_window": "not-a-number", "envs": {"fake": [1, 2, 3]}}
    with patch("reliquary.validator.service.TRAINING_RUN_ID", "run5"), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=corrupt),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=[]),
    ):
        await svc._rebuild_cooldown_from_history()  # must not raise
    assert len(svc._cooldown_per_env["fake"]) == 0  # partial restore discarded
