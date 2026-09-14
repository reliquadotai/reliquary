"""Warm recovery must retain the cold recovery result and failure boundaries."""
import asyncio
import gzip
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("fields,limit", [(None, 4), (("window_start",), 1)])
async def test_archive_downloads_overlap_but_remain_bounded_and_ordered(monkeypatch, fields, limit):
    from reliquary.infrastructure import storage

    active = peak = 0

    async def get_object(*, Bucket, Key):
        nonlocal active, peak
        window = int(Key.split("window-")[1].split(".")[0])
        active += 1
        peak = max(peak, active)

        async def read():
            nonlocal active
            await asyncio.sleep(.005 * (4 - window % 4))
            active -= 1
            return gzip.compress(json.dumps({"window_start": window}).encode())

        return {"Body": SimpleNamespace(read=read, close=lambda: None)}

    context = AsyncMock()
    context.__aenter__.return_value.get_object = get_object
    monkeypatch.setattr(storage, "get_s3_client", lambda **kw: context)
    result = await storage.list_recent_datasets(9, 8, strict=True, fields=fields)
    assert [row["window_start"] for row in result] == list(range(1, 9))
    assert peak == limit and active == 0


@pytest.mark.asyncio
async def test_archive_failure_cancels_other_downloads_before_client_close(monkeypatch):
    from reliquary.infrastructure import storage

    active = 0

    async def get_object(*, Bucket, Key):
        nonlocal active
        active += 1
        try:
            if "window-1." in Key:
                await asyncio.sleep(.01)
                raise ValueError("broken archive")
            await asyncio.Event().wait()
        finally:
            active -= 1

    context = AsyncMock()
    context.__aenter__.return_value.get_object = get_object

    async def closed(*args):
        assert active == 0

    context.__aexit__.side_effect = closed
    monkeypatch.setattr(storage, "get_s3_client", lambda **kw: context)
    with pytest.raises(ValueError, match="broken archive"):
        await asyncio.wait_for(storage.list_recent_datasets(5, 4, strict=True), 1)


def test_trainer_cache_reuses_bytes_and_recovers_corruption_or_interruption(tmp_path):
    from reliquary.trainer.cli import _download_checkpoint

    objects = {"config.json": b"{}", "model.safetensors": b"model-bytes"}
    calls = []
    revision = "a" * 40

    def download(bucket, key, filename, **kwargs):
        name = key.rsplit("/", 1)[1]
        calls.append(name)
        from pathlib import Path
        Path(filename).write_bytes(objects[name])

    client = SimpleNamespace(
        list_objects_v2=lambda **kw: {"Contents": [
            {"Key": kw["Prefix"] + name, "Size": len(value), "ETag": name}
            for name, value in objects.items()
        ]}, download_file=download,
    )
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    assert len(calls) == 2
    (tmp_path / "model.safetensors").write_bytes(b"wrong-bytes")
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    assert len(calls) == 4
    (tmp_path / ".download-complete.json").unlink()
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    assert len(calls) == 6

    # A failed refresh removes the completion marker before any replacement.
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    client.download_file = MagicMock(side_effect=OSError("interrupted"))
    with pytest.raises(OSError, match="interrupted"):
        _download_checkpoint(client, "bucket", revision, tmp_path)
    assert not (tmp_path / ".download-complete.json").exists()
    client.download_file = download
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    (tmp_path / "model.safetensors").unlink()
    assert _download_checkpoint(client, "bucket", revision, tmp_path)
    assert (tmp_path / "model.safetensors").read_bytes() == objects["model.safetensors"]


def test_initial_proof_weights_are_reused_only_for_exact_adoption(monkeypatch):
    import reliquary.validator.proof_worker as worker

    revision = "a" * 40
    install = MagicMock()
    monkeypatch.setattr(worker, "_install_from_hub", install)
    context = {"model": object(), "revision": None,
               "_initial_source": ("test/model", revision)}
    worker.reload_proof_context(context, None, revision, "test/model")
    install.assert_not_called()
    assert context["revision"] == revision
    for repo, target in [("other/model", revision), ("test/model", "b" * 40)]:
        context = {"model": object(), "revision": None,
                   "_initial_source": ("test/model", revision)}
        worker.reload_proof_context(context, None, target, repo)
        install.assert_called_with(context, repo, target)


def service():
    from reliquary.validator.service import ValidationService
    from reliquary.validator.dedup import RolloutHashSet
    from reliquary.constants import HASH_DEDUP_RETENTION_WINDOWS

    value = ValidationService.__new__(ValidationService)
    value.hf_repo_id = "test/model"
    value._hash_set = RolloutHashSet(HASH_DEDUP_RETENTION_WINDOWS)
    return value


@pytest.mark.asyncio
async def test_hash_snapshot_replays_only_gap_and_matches_full_rebuild():
    from reliquary.validator.dedup import RolloutHashSet
    from reliquary.constants import HASH_DEDUP_RETENTION_WINDOWS

    archives = [{"window_start": n, "batch": [{"rollouts": [{"tokens": [n, 2]}]}]}
                for n in range(1, 5)]
    first = service()
    first._window_n = 2
    first._load_archive_range = AsyncMock(return_value=archives[:2])
    await first._rebuild_hashes_from_history()
    # Unarchived live entries must never enter the recovery cache.
    first._hash_set.add(b"x" * 32, 3)
    first._cache_archived_hashes(archives[2])
    first._write_hash_snapshot()
    restored = service()
    restored._window_n = 4
    restored._load_archive_range = AsyncMock(return_value=archives[3:])
    await restored._rebuild_hashes_from_history()
    restored._load_archive_range.assert_awaited_once_with(start_window=4, end_window=4, require_all=True)
    reference = RolloutHashSet(HASH_DEDUP_RETENTION_WINDOWS)
    reference.rebuild_from_history(archives, current_window=4)
    assert restored._hash_set.export_state() == reference.export_state()
    warm = service()
    warm._window_n = 4
    warm._load_archive_range = AsyncMock(side_effect=AssertionError("remote read"))
    await warm._rebuild_hashes_from_history()
    assert warm._hash_set.export_state() == reference.export_state()


@pytest.mark.asyncio
async def test_invalid_hash_snapshot_falls_back_and_missing_gap_stays_closed():
    from reliquary.validator.service import _write_gzip_json_atomic

    value = service()
    value._window_n = 2
    _write_gzip_json_atomic(value._hash_snapshot_path(), {
        "identity": value._hash_snapshot_identity(), "through_window": 99, "entries": {},
    })
    value._load_archive_range = AsyncMock(side_effect=RuntimeError("missing history"))
    with pytest.raises(RuntimeError, match="hash history rebuild failed"):
        await value._rebuild_hashes_from_history()
    value._load_archive_range.assert_awaited_once_with(start_window=1, end_window=2, require_all=True)


@pytest.mark.asyncio
async def test_hash_cache_cannot_advance_past_unrecorded_window():
    value = service()
    value._window_n = 0
    await value._rebuild_hashes_from_history()
    value._cache_archived_hashes({"window_start": 2, "batch": []})
    assert value._hash_cache_window == 0
    value._cache_archived_hashes({"window_start": 1, "batch": []})
    assert value._hash_cache_window == 1


@pytest.mark.asyncio
async def test_content_snapshot_startup_does_not_wait_for_remote_mirror(monkeypatch):
    from reliquary.validator.service import ValidationService, _read_gzip_json, _content_cooldown_local_path
    from reliquary.constants import TRAINING_RUN_ID
    from reliquary.infrastructure import storage

    value = ValidationService.__new__(ValidationService)
    value._window_n = 5
    value._content_cooldown_per_env = {}
    value._content_cooldown_health = {}
    upload = AsyncMock(side_effect=AssertionError("startup mirror"))
    monkeypatch.setattr(storage, "upload_json", upload)
    assert await value._snapshot_content_cooldown(mirror=False)
    assert _read_gzip_json(_content_cooldown_local_path(TRAINING_RUN_ID))["snapshot_window"] == 5
    upload.assert_not_awaited()
