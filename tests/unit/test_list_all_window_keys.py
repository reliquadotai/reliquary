"""list_all_window_keys: paginate the flat R2 prefix, return sorted window_n list."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_empty_bucket_returns_empty_list():
    mock_client = AsyncMock()
    paginator = MagicMock()

    async def empty_paginate(*args, **kwargs):
        return
        yield  # empty async generator

    paginator.paginate = lambda *a, **kw: empty_paginate()
    mock_client.get_paginator = MagicMock(return_value=paginator)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        from reliquary.infrastructure.storage import list_all_window_keys
        result = await list_all_window_keys()
    assert result == []


@pytest.mark.asyncio
async def test_parses_and_sorts_window_numbers():
    pages = [{
        "Contents": [
            {"Key": "reliquary/dataset/window-5.json.gz"},
            {"Key": "reliquary/dataset/window-1.json.gz"},
            {"Key": "reliquary/dataset/window-10.json.gz"},
            {"Key": "reliquary/dataset/not-a-window.json.gz"},
        ]
    }]
    mock_client = AsyncMock()
    paginator = MagicMock()

    async def paginate(*args, **kwargs):
        for page in pages:
            yield page

    paginator.paginate = lambda *a, **kw: paginate()
    mock_client.get_paginator = MagicMock(return_value=paginator)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        from reliquary.infrastructure.storage import list_all_window_keys
        result = await list_all_window_keys()
    assert result == [1, 5, 10]


@pytest.mark.asyncio
async def test_multiple_pages_merged_and_sorted():
    """Results from multiple paginator pages are merged and sorted."""
    pages = [
        {"Contents": [
            {"Key": "reliquary/dataset/window-3.json.gz"},
            {"Key": "reliquary/dataset/window-7.json.gz"},
        ]},
        {"Contents": [
            {"Key": "reliquary/dataset/window-1.json.gz"},
            {"Key": "reliquary/dataset/window-100.json.gz"},
        ]},
    ]
    mock_client = AsyncMock()
    paginator = MagicMock()

    async def paginate(*args, **kwargs):
        for page in pages:
            yield page

    paginator.paginate = lambda *a, **kw: paginate()
    mock_client.get_paginator = MagicMock(return_value=paginator)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        from reliquary.infrastructure.storage import list_all_window_keys
        result = await list_all_window_keys()
    assert result == [1, 3, 7, 100]


@pytest.mark.asyncio
async def test_client_error_returns_empty_list():
    """A ClientError during pagination returns empty list rather than raising."""
    from botocore.exceptions import ClientError

    mock_client = AsyncMock()
    paginator = MagicMock()

    async def failing_paginate(*args, **kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "ListObjectsV2")
        yield  # make it a generator

    paginator.paginate = lambda *a, **kw: failing_paginate()
    mock_client.get_paginator = MagicMock(return_value=paginator)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch("reliquary.infrastructure.storage.get_s3_client", return_value=mock_ctx):
        from reliquary.infrastructure.storage import list_all_window_keys
        result = await list_all_window_keys()
    assert result == []


@pytest.mark.asyncio
async def test_strict_client_error_is_not_reinterpreted_as_empty_history():
    from botocore.exceptions import ClientError
    from reliquary.infrastructure.storage import list_all_window_keys

    mock_client = AsyncMock()
    paginator = MagicMock()

    async def failing_paginate(*args, **kwargs):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}},
            "ListObjectsV2",
        )
        yield

    paginator.paginate = lambda *args, **kwargs: failing_paginate()
    mock_client.get_paginator = MagicMock(return_value=paginator)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = None

    with patch(
        "reliquary.infrastructure.storage.get_s3_client",
        return_value=mock_ctx,
    ):
        with pytest.raises(ClientError):
            await list_all_window_keys(strict=True)
