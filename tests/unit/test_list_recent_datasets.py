"""list_recent_datasets: downloads last N window archives from R2 for cooldown rebuild."""

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.infrastructure.storage import list_recent_datasets


def _archive_bytes(window_start: int, prompt_ids: list[int]) -> bytes:
    payload = {
        "window_start": window_start,
        "batch": [{"prompt_idx": p, "hotkey": "hk", "k": 4}
                  for p in prompt_ids],
    }
    return gzip.compress(json.dumps(payload).encode())


@pytest.mark.asyncio
async def test_downloads_last_n_windows():
    archives_on_r2 = {
        "reliquary/dataset/window-100.json.gz": _archive_bytes(100, [1, 2]),
        "reliquary/dataset/window-101.json.gz": _archive_bytes(101, [3]),
        "reliquary/dataset/window-102.json.gz": _archive_bytes(102, [4]),
    }

    async def fake_get_object(Bucket, Key):
        class _Body:
            async def read(self_inner):
                return archives_on_r2[Key]

            def close(self_inner):
                pass
        return {"Body": _Body()}

    mock_client = AsyncMock()
    mock_client.get_object = fake_get_object

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        result = await list_recent_datasets(
            current_window=103, n=3,
        )
    assert len(result) == 3
    assert [a["window_start"] for a in result] == [100, 101, 102]
    assert result[0]["batch"][0]["prompt_idx"] == 1

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        projected = await list_recent_datasets(
            current_window=103,
            n=3,
            fields=("window_start",),
        )
    assert projected == [
        {"window_start": 100},
        {"window_start": 101},
        {"window_start": 102},
    ]


@pytest.mark.asyncio
async def test_skips_missing_windows():
    """A window that doesn't exist (NoSuchKey) is skipped with a warning."""
    from botocore.exceptions import ClientError

    archives_on_r2 = {
        "reliquary/dataset/window-100.json.gz": _archive_bytes(100, [1]),
        # window-101 is missing — simulate NoSuchKey
        "reliquary/dataset/window-102.json.gz": _archive_bytes(102, [2]),
    }

    async def fake_get_object(Bucket, Key):
        if Key not in archives_on_r2:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "no"}}, "GetObject"
            )
        class _Body:
            async def read(self_inner):
                return archives_on_r2[Key]

            def close(self_inner):
                pass
        return {"Body": _Body()}

    mock_client = AsyncMock()
    mock_client.get_object = fake_get_object
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        result = await list_recent_datasets(
            current_window=103, n=3,
        )
    assert [a["window_start"] for a in result] == [100, 102]


@pytest.mark.asyncio
async def test_zero_n_returns_empty():
    result = await list_recent_datasets(current_window=100, n=0)
    assert result == []


@pytest.mark.asyncio
async def test_current_window_smaller_than_n_caps_at_zero():
    """If current_window=5, n=50, we should only try windows [0, 5), not negative."""

    async def fake_get_object(Bucket, Key):
        from botocore.exceptions import ClientError
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "no"}}, "GetObject"
        )

    mock_client = AsyncMock()
    mock_client.get_object = fake_get_object
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        result = await list_recent_datasets(current_window=5, n=50)
    assert result == []
