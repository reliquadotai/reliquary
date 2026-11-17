"""The checkpoint publisher's object store for a rented trainer pod: no bucket key on the pod.

The controller (``rl_async_service`` trainer link) hands out presigned PUT/GET URLs under this run's prefix and
deletes on request; the pod only moves bytes over plain HTTPS. Same methods as ``ckpt_publisher.R2ObjectStore``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

UPLOAD_TIMEOUT_S = 3600.0


class PresignedObjectStore:
    def __init__(self, link: Any, *, http: Any = None) -> None:
        import httpx

        self._link = link
        self._http = http or httpx.Client(timeout=httpx.Timeout(UPLOAD_TIMEOUT_S, connect=30.0))

    def _put(self, key: str, content: Any) -> None:
        url = self._link.upload_urls([key])[key]
        self._http.put(url, content=content).raise_for_status()

    def put_bytes(self, key: str, data: bytes) -> None:
        self._put(key, bytes(data))

    def put_file(self, key: str, path: Path) -> None:
        with Path(path).open("rb") as handle:
            self._put(key, handle)

    def get_bytes(self, key: str) -> bytes | None:
        response = self._http.get(self._link.read_urls([key])[key])
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    def delete(self, key: str) -> None:
        self._link.delete_keys([key])
