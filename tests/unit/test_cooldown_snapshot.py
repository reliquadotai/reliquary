"""Run-keyed cooldown snapshot: restore + gap-replay, reset on a fresh run,
and the snapshot write shape. Storage is mocked — no R2."""

import os
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


def _gap_archives(start: int, stop: int, *, selected_window: int | None = None):
    archives = []
    for window_start in range(start, stop + 1):
        archive = {
            "window_start": window_start,
            "window_status": "aborted",
            "environment": "fake",
            "batch": [],
        }
        if window_start == selected_window:
            archive["window_status"] = "completed"
            archive["batch"] = [{"prompt_idx": 99}]
        archives.append(archive)
    return archives


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
    gap = _gap_archives(41, 45, selected_window=43)
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
async def test_snapshot_lookup_failure_cannot_reset_a_training_run():
    svc = _service(40)
    with patch("reliquary.validator.service.TRAINING_RUN_ID", "run5"), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(side_effect=OSError("object store unavailable")),
    ):
        with pytest.raises(RuntimeError, match="no valid durable copy"):
            await svc._rebuild_cooldown_from_history()


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
async def test_prompt_restore_prefers_newer_local_snapshot(tmp_path):
    from reliquary.validator.service import (
        _cooldown_local_path,
        _write_gzip_json_atomic,
    )

    remote = {
        "run_id": "default",
        "snapshot_window": 30,
        "envs": {"fake": {"1": 20}},
    }
    local = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": 40,
        "complete": True,
        "envs": {"fake": {"1": 20, "2": 40}},
    }
    with patch.dict("os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}):
        _write_gzip_json_atomic(_cooldown_local_path("default"), local)
        svc = _service(45)
        with patch(
            "reliquary.infrastructure.storage.download_json",
            new=AsyncMock(return_value=remote),
        ), patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=_gap_archives(41, 45)),
        ):
            await svc._rebuild_cooldown_from_history()

    assert svc._cooldown_per_env["fake"].export_state() == {1: 20, 2: 40}


@pytest.mark.asyncio
async def test_gap_replay_failure_refuses_to_open_with_stale_cooldown(tmp_path):
    from reliquary.validator.service import (
        _cooldown_local_path,
        _write_gzip_json_atomic,
    )

    snapshot = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": 40,
        "complete": True,
        "envs": {"fake": {"1": 20}},
    }
    with patch.dict("os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}):
        _write_gzip_json_atomic(_cooldown_local_path("default"), snapshot)
        svc = _service(45)
        with patch(
            "reliquary.infrastructure.storage.download_json",
            new=AsyncMock(return_value=None),
        ), patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(side_effect=OSError("archive replay unavailable")),
        ):
            with pytest.raises(RuntimeError, match="gap replay failed"):
                await svc._rebuild_cooldown_from_history()


@pytest.mark.asyncio
async def test_gap_replay_rejects_a_missing_window_record():
    svc = _service(45)
    snapshot = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": 40,
        "complete": True,
        "envs": {"fake": {"1": 20}},
    }
    incomplete = [
        archive
        for archive in _gap_archives(41, 45)
        if archive["window_start"] != 43
    ]
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=snapshot),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=incomplete),
    ):
        with pytest.raises(RuntimeError, match="gap replay failed"):
            await svc._rebuild_cooldown_from_history()


@pytest.mark.asyncio
async def test_gap_replay_merges_a_locally_pending_archive(tmp_path):
    from reliquary.infrastructure.archive_queue import ArchiveQueue

    svc = _service(45)
    snapshot = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": 40,
        "complete": True,
        "envs": {"fake": {"1": 20}},
    }
    queue = ArchiveQueue(str(tmp_path / "pending"))
    svc._archive_queue = queue
    queue.enqueue(
        43,
        {
            "window_start": 43,
            "window_status": "completed",
            "environment": "fake",
            "batch": [{"prompt_idx": 99}],
        },
    )
    remote = [
        archive
        for archive in _gap_archives(41, 45)
        if archive["window_start"] != 43
    ]
    with patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=snapshot),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=remote),
    ):
        await svc._rebuild_cooldown_from_history()

    assert svc._cooldown_per_env["fake"].is_in_cooldown(99, 45) is True


@pytest.mark.asyncio
async def test_prompt_snapshot_is_durable_locally_when_r2_fails(tmp_path):
    from reliquary.validator.service import (
        _cooldown_local_path,
        _read_gzip_json,
    )

    svc = _service(77)
    svc._cooldown_per_env["fake"].record_batched(7, 70)
    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(side_effect=OSError("R2 unavailable")),
    ):
        assert await svc._snapshot_cooldown() is True
        snapshot = _read_gzip_json(_cooldown_local_path("default"))

    assert snapshot is not None
    assert snapshot["snapshot_window"] == 77
    assert snapshot["envs"]["fake"] == {"7": 70}


def test_prompt_snapshot_schema_requires_complete_v2_records():
    from reliquary.validator.service import ValidationService

    legacy = {
        "run_id": "default",
        "snapshot_window": 10,
        "envs": {"fake": {"1": 9}},
    }
    assert ValidationService._validate_cooldown_snapshot(
        legacy,
        {"fake"},
        10,
    ) == 10

    with pytest.raises(ValueError, match="incomplete"):
        ValidationService._validate_cooldown_snapshot(
            {**legacy, "schema_version": 2, "complete": False},
            {"fake"},
            10,
        )
    with pytest.raises(ValueError, match="unsupported"):
        ValidationService._validate_cooldown_snapshot(
            {**legacy, "schema_version": 3, "complete": True},
            {"fake"},
            10,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snapshot_window", True),
        ("snapshot_window", 10.0),
        ("snapshot_window", "10"),
        ("selected_window", False),
        ("selected_window", 9.0),
        ("selected_window", "9"),
    ],
)
def test_prompt_snapshot_rejects_numeric_coercion(field, value):
    from reliquary.validator.service import ValidationService

    snapshot = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": value if field == "snapshot_window" else 10,
        "complete": True,
        "envs": {
            "fake": {"1": value if field == "selected_window" else 9}
        },
    }
    with pytest.raises(ValueError):
        ValidationService._validate_cooldown_snapshot(
            snapshot, {"fake"}, 10
        )


@pytest.mark.parametrize("key", ["01", "+1", " 1", "1 "])
def test_prompt_snapshot_requires_canonical_prompt_keys(key):
    from reliquary.validator.service import ValidationService

    snapshot = {
        "schema_version": 2,
        "run_id": "default",
        "snapshot_window": 10,
        "complete": True,
        "envs": {"fake": {key: 9}},
    }
    with pytest.raises(ValueError, match="canonical decimal"):
        ValidationService._validate_cooldown_snapshot(
            snapshot, {"fake"}, 10
        )


@pytest.mark.asyncio
async def test_snapshot_pair_advances_only_when_both_identities_are_durable():
    svc = _service(77)
    svc._snapshot_cooldown = AsyncMock(return_value=True)
    svc._snapshot_content_cooldown = AsyncMock(return_value=False)

    assert await svc._snapshot_all_cooldowns() is False

    svc._snapshot_content_cooldown.return_value = True
    assert await svc._snapshot_all_cooldowns() is True


@pytest.mark.asyncio
async def test_pipelined_snapshot_uses_last_durable_window(tmp_path):
    from reliquary.validator.service import (
        _content_cooldown_local_path,
        _cooldown_local_path,
        _read_gzip_json,
    )

    svc = _service(11)
    svc._cooldown_durable_window = 10
    svc._cooldown_per_env["fake"].record_batched(1, 10)
    svc._cooldown_per_env["fake"].record_batched(2, 11)
    svc._content_cooldown_per_env["fake"].record_selected("a" * 64, 10)
    svc._content_cooldown_per_env["fake"].record_selected("b" * 64, 11)

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(return_value=True),
    ):
        assert await svc._snapshot_committed_cooldowns() is True
        prompt = _read_gzip_json(_cooldown_local_path("default"))
        content = _read_gzip_json(_content_cooldown_local_path("default"))

    assert prompt["snapshot_window"] == 10
    assert prompt["envs"]["fake"] == {"1": 10}
    assert content["snapshot_window"] == 10
    assert content["envs"]["fake"] == {"a" * 64: 10}


@pytest.mark.asyncio
async def test_pipelined_restart_replays_window_after_snapshot_watermark(
    tmp_path,
):
    from reliquary.infrastructure.archive_queue import ArchiveQueue

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(return_value=True),
    ):
        before = _service(11)
        before._cooldown_per_env["fake"].record_batched(1, 10)
        assert await before._snapshot_cooldown(snapshot_window=10) is True

        queue = ArchiveQueue(str(tmp_path / "pending_archives"))
        queue.enqueue(
            11,
            {
                "window_start": 11,
                "window_status": "completed",
                "environment": "fake",
                "batch": [{"prompt_idx": 99}],
            },
        )

        restarted = _service(11)
        restarted._archive_queue = queue
        with patch(
            "reliquary.infrastructure.storage.download_json",
            new=AsyncMock(return_value=None),
        ), patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ):
            await restarted._rebuild_cooldown_from_history()

    assert restarted._cooldown_per_env["fake"].is_in_cooldown(99, 11)


def test_atomic_snapshot_fsyncs_file_and_parent_directory(tmp_path):
    from reliquary.validator.service import _write_gzip_json_atomic

    real_fsync = os.fsync
    with patch(
        "reliquary.validator.service.os.fsync", wraps=real_fsync
    ) as fsync:
        _write_gzip_json_atomic(tmp_path / "snapshot.json.gz", {"ok": True})

    assert fsync.call_count == 2


@pytest.mark.asyncio
async def test_corrupt_snapshot_fails_closed_instead_of_resetting_cooldown():
    """A present but invalid durable record cannot mean a fresh training run."""
    svc = _service(40)
    corrupt = {"run_id": "run5", "snapshot_window": "not-a-number", "envs": {"fake": [1, 2, 3]}}
    with patch("reliquary.validator.service.TRAINING_RUN_ID", "run5"), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=corrupt),
    ), patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="no valid durable copy"):
            await svc._rebuild_cooldown_from_history()
    assert len(svc._cooldown_per_env["fake"]) == 0


@pytest.mark.parametrize(
    ("snapshot_window", "digest", "selected_window"),
    [
        (True, "a" * 64, 9),
        (10.0, "a" * 64, 9),
        ("10", "a" * 64, 9),
        (10, "A" * 64, 9),
        (10, "a" * 64, False),
        (10, "a" * 64, 9.0),
        (10, "a" * 64, "9"),
    ],
)
def test_content_snapshot_requires_canonical_durable_values(
    snapshot_window,
    digest,
    selected_window,
):
    from reliquary.validator.service import ValidationService

    snapshot = {
        "schema_version": 1,
        "run_id": "default",
        "snapshot_window": snapshot_window,
        "complete": True,
        "envs": {"fake": {digest: selected_window}},
    }
    with pytest.raises(ValueError):
        ValidationService._validate_content_snapshot(
            snapshot, {"fake"}, 10
        )


@pytest.mark.asyncio
async def test_content_cooldown_first_restore_backfills_prompt_state(tmp_path):
    svc = _service(77)
    svc._cooldown_per_env["fake"].record_batched(7, 70)
    uploads = []

    async def fake_upload(key, data):
        uploads.append((key, data))
        return True

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.infrastructure.storage.upload_json", new=fake_upload,
    ):
        await svc._restore_content_cooldown()

    content = svc._content_cooldown_per_env["fake"]
    assert len(content) == 1
    assert svc._content_cooldown_health["complete"] is True
    assert uploads[0][0] == "content_cooldown_snapshots/default.json.gz"
    assert uploads[0][1]["complete"] is True
    assert (tmp_path / "content_cooldown" / "default.json.gz").exists()


@pytest.mark.asyncio
async def test_content_snapshot_restores_and_resolves_only_new_prompt_state(
    tmp_path,
):
    from reliquary.validator.prompt_content import prompt_content_sha256

    svc = _service(80)
    old_digest = prompt_content_sha256("fake", "old")
    snapshot = {
        "schema_version": 1,
        "run_id": "default",
        "snapshot_window": 70,
        "complete": True,
        "envs": {"fake": {old_digest: 60}},
    }
    svc._cooldown_per_env["fake"].record_batched(7, 75)

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=snapshot),
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(return_value=True),
    ):
        await svc._restore_content_cooldown()

    restored = svc._content_cooldown_per_env["fake"].export_state()
    assert restored[old_digest] == 60
    assert len(restored) == 2
    assert max(restored.values()) == 75


@pytest.mark.asyncio
async def test_content_restore_prefers_newer_local_snapshot(tmp_path):
    from reliquary.validator.prompt_content import prompt_content_sha256
    from reliquary.validator.service import (
        _content_cooldown_local_path,
        _write_gzip_json_atomic,
    )

    old_digest = prompt_content_sha256("fake", "old")
    new_digest = prompt_content_sha256("fake", "new")
    remote = {
        "schema_version": 1,
        "run_id": "default",
        "snapshot_window": 50,
        "complete": True,
        "envs": {"fake": {old_digest: 40}},
    }
    local = {
        "schema_version": 1,
        "run_id": "default",
        "snapshot_window": 70,
        "complete": True,
        "envs": {"fake": {old_digest: 40, new_digest: 70}},
    }
    with patch.dict("os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}):
        _write_gzip_json_atomic(
            _content_cooldown_local_path("default"),
            local,
        )
        svc = _service(80)
        with patch(
            "reliquary.infrastructure.storage.download_json",
            new=AsyncMock(return_value=remote),
        ), patch(
            "reliquary.infrastructure.storage.upload_json",
            new=AsyncMock(return_value=True),
        ):
            await svc._restore_content_cooldown()

    assert svc._content_cooldown_per_env["fake"].export_state() == {
        old_digest: 40,
        new_digest: 70,
    }


@pytest.mark.asyncio
async def test_content_restore_allows_r2_outage_after_local_persist(tmp_path):
    svc = _service(77)
    svc._cooldown_per_env["fake"].record_batched(7, 70)

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(side_effect=OSError("R2 unavailable")),
    ):
        await svc._restore_content_cooldown()

    assert svc._content_cooldown_health["complete"] is True
    assert svc._content_cooldown_health["source"] == "local"
    assert svc._content_cooldown_health["last_error_type"] == "OSError"
    assert (tmp_path / "content_cooldown" / "default.json.gz").exists()


@pytest.mark.asyncio
async def test_content_restore_refuses_memory_only_bootstrap(tmp_path):
    svc = _service(77)
    svc._cooldown_per_env["fake"].record_batched(7, 70)

    with patch.dict(
        "os.environ", {"RELIQUARY_STATE_DIR": str(tmp_path)}
    ), patch(
        "reliquary.infrastructure.storage.download_json",
        new=AsyncMock(return_value=None),
    ), patch(
        "reliquary.validator.service._write_gzip_json_atomic",
        side_effect=OSError("disk unavailable"),
    ), patch(
        "reliquary.infrastructure.storage.upload_json",
        new=AsyncMock(return_value=True),
    ) as upload:
        with pytest.raises(RuntimeError, match="restore incomplete"):
            await svc._restore_content_cooldown()

    upload.assert_not_awaited()
    assert svc._content_cooldown_health["complete"] is False
    assert svc._content_cooldown_health["last_error_type"] == "RuntimeError"
