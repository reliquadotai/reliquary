"""The checkpoint publisher's object store for a rented trainer pod: no bucket key on the pod.

The controller (``rl_async_service`` trainer link) hands out presigned PUT/GET URLs under this run's prefix and
deletes on request; the pod only moves bytes over plain HTTPS. Same methods as ``ckpt_publisher.R2ObjectStore``.
A presigned PUT is one request (no multipart), so anchor shards are cut small and a file over the limit is refused.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPLOAD_TIMEOUT_S = 3600.0
SHARD_BYTES = 2 * 1024**3                   # anchor shard size on this path (the operator path uses 5 GiB)
MAX_PUT_BYTES = int(4.5 * 1024**3)          # under the 5 GiB single-PUT limit of S3/R2
PUT_ATTEMPTS = 3
PUT_BACKOFF_S = (2.0, 4.0)


class PresignedObjectStore:
    anchor_shard_bytes = SHARD_BYTES

    def __init__(self, link: Any, *, http: Any = None, sleep: Callable[[float], None] = time.sleep) -> None:
        import httpx

        self._link = link
        self._http = http or httpx.Client(timeout=httpx.Timeout(UPLOAD_TIMEOUT_S, connect=30.0))
        self._sleep = sleep

    def _put(self, key: str, body: Callable[[], Any], size: int) -> None:
        """One attempt = a fresh URL and a freshly opened body; up to ``PUT_ATTEMPTS``."""
        if size > MAX_PUT_BYTES:
            raise ValueError(f"{key} is {size} bytes: over the {MAX_PUT_BYTES}-byte single PUT limit")
        for attempt in range(PUT_ATTEMPTS):
            try:
                url = self._link.upload_urls([key])[key]
                content = body()
                try:
                    self._http.put(url, content=content).raise_for_status()
                finally:
                    close = getattr(content, "close", None)
                    if close is not None:
                        close()
                return
            except Exception as exc:
                if attempt == PUT_ATTEMPTS - 1:
                    raise
                delay = PUT_BACKOFF_S[attempt]
                logger.warning("upload of %s failed (%s); retrying in %.0fs", key, type(exc).__name__, delay)
                self._sleep(delay)

    def put_bytes(self, key: str, data: bytes) -> None:
        data = bytes(data)
        self._put(key, lambda: data, len(data))

    def put_file(self, key: str, path: Path) -> None:
        path = Path(path)
        self._put(key, lambda: path.open("rb"), path.stat().st_size)

    def get_bytes(self, key: str) -> bytes | None:
        response = self._http.get(self._link.read_urls([key])[key])
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    def delete(self, key: str) -> None:
        self._link.delete_keys([key])
