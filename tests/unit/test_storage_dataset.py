"""Tests for the GRPO dataset upload helper in storage.py."""

from __future__ import annotations

import gzip
import json
from unittest.mock import patch

import pytest

from reliquary.infrastructure.storage import download_json, list_recent_datasets, upload_json


class _AsyncBody:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.closed = False

    async def read(self) -> bytes:
        return self.value

    def close(self) -> None:
        self.closed = True


class _JsonClient:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    async def put_object(self, **kwargs) -> None:
        self.captured.update(kwargs)

    async def get_object(self, **_kwargs):
        return {"Body": _AsyncBody(self.captured["Body"])}


class _ClientContext:
    def __init__(self, client) -> None:
        self.client = client

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *_args) -> None:
        return None


@pytest.mark.asyncio
async def test_upload_and_download_json_gzip_by_key_suffix() -> None:
    captured: dict[str, object] = {}
    client = _JsonClient(captured)

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        side_effect=lambda **_kwargs: _ClientContext(client),
    ):
        assert await upload_json(
            "content_cooldown_snapshots/run.json.gz",
            {"complete": True, "items": [1, 2, 3]},
            bucket_name="testbucket",
        )
        restored = await download_json(
            "content_cooldown_snapshots/run.json.gz",
            bucket_name="testbucket",
        )

    assert captured["Bucket"] == "testbucket"
    assert gzip.decompress(captured["Body"]).startswith(b'{"complete":true')
    assert restored == {"complete": True, "items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_strict_download_does_not_hide_storage_failure() -> None:
    class _UnavailableClient:
        async def get_object(self, **_kwargs):
            raise OSError("storage unavailable")

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        return_value=_ClientContext(_UnavailableClient()),
    ):
        assert await download_json("state.json") is None
        with pytest.raises(OSError, match="storage unavailable"):
            await download_json("state.json", strict=True)


@pytest.mark.asyncio
async def test_download_json_rejects_duplicate_keys() -> None:
    captured: dict[str, object] = {
        "Body": b'{"complete":true,"complete":false}'
    }
    client = _JsonClient(captured)

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        return_value=_ClientContext(client),
    ):
        with pytest.raises(ValueError, match="duplicate JSON key: complete"):
            await download_json("state.json", strict=True)
        assert await download_json("state.json") is None


@pytest.mark.asyncio
async def test_strict_archive_read_binds_body_to_object_window() -> None:
    payload = gzip.compress(json.dumps({"window_start": 99}).encode())
    body = _AsyncBody(payload)

    class _MismatchedClient:
        async def get_object(self, **_kwargs):
            return {"Body": body}

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        return_value=_ClientContext(_MismatchedClient()),
    ):
        with pytest.raises(ValueError, match="does not bind window 1"):
            await list_recent_datasets(
                current_window=2,
                n=1,
                strict=True,
            )
    assert body.closed


@pytest.mark.asyncio
async def test_strict_archive_read_rejects_duplicate_keys() -> None:
    payload = gzip.compress(b'{"window_start":1,"window_start":1}')

    class _DuplicateClient:
        async def get_object(self, **_kwargs):
            return {"Body": _AsyncBody(payload)}

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        return_value=_ClientContext(_DuplicateClient()),
    ):
        with pytest.raises(ValueError, match="duplicate JSON key: window_start"):
            await list_recent_datasets(
                current_window=2,
                n=1,
                strict=True,
            )
