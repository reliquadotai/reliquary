"""The pod's client for the platform's internal evaluation API.

Every path, header and field name of that contract is in this module, so a
drift on the platform side is fixed here and nowhere else. Authentication is the
per-pod token; after a claim every call also carries the task lease.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_ENV = "RELIQUARY_EXECUTOR_TOKEN"
LEASE_HEADER = "X-Task-Lease"
PART_SHA_HEADER = "X-Part-Sha256"
BASE = "/api/internal/evaluations"
# Upload parts are numbered from this.
PART_BASE = 1
REQUEST_TIMEOUT_SECONDS = 120.0
RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
# The platform no longer recognises this pod or its lease: stop, never retry.
LOST_STATUSES = frozenset({401, 403, 409, 410})
# On one upload's own routes a 409/410 is about that upload (complete,
# expired), not the lease: only the token's refusal stops the pod there.
UPLOAD_LOST_STATUSES = frozenset({401, 403})


class PlatformError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LeaseLost(PlatformError):
    """The token or the lease was refused: the work stops."""


class UploadConflict(PlatformError):
    """An upload is already complete, or gone: never a reason to stop."""


def sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class PlatformClient:
    def __init__(self, base_url: str, token: str, *, executor_id: str, http=None,
                 sleep: Callable[[float], None] = time.sleep,
                 retries: int = RETRIES) -> None:
        import httpx

        if not token:
            raise ValueError(f"{TOKEN_ENV} is empty")
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"),
                                          timeout=REQUEST_TIMEOUT_SECONDS,
                                          follow_redirects=False)
        self._token = token
        self.executor_id = executor_id
        self.task_id: str | None = None
        self._lease: str | None = None
        self._sleep = sleep
        self._retries = retries

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"}
        if self._lease is not None:
            headers[LEASE_HEADER] = self._lease
        return {**headers, **(extra or {})}

    def _task_path(self, suffix: str) -> str:
        if self.task_id is None:
            raise PlatformError("no task claimed")
        return f"{BASE}/{self.task_id}/{suffix}"

    def _request(self, method: str, path: str, *, json_body: Any = None,
                 content: bytes | None = None, headers: dict | None = None,
                 ok: tuple[int, ...] = (200, 201, 202, 204),
                 lost: frozenset[int] = LOST_STATUSES):
        """One call, retried on transport errors and 5xx; a refused token or
        lease raises ``LeaseLost`` at once."""
        import httpx

        last: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(RETRY_BACKOFF_SECONDS * attempt)
            try:
                response = self._http.request(method, path, json=json_body, content=content,
                                              headers=self._headers(headers))
            except httpx.TransportError as exc:
                last = PlatformError(f"{method} {path}: {exc!r}")
                continue
            if response.status_code in ok:
                return response
            detail = response.text[:300]
            if response.status_code in lost:
                raise LeaseLost(f"{method} {path}: {response.status_code} {detail}",
                                response.status_code)
            if response.status_code in LOST_STATUSES:
                raise UploadConflict(f"{method} {path}: {response.status_code} {detail}",
                                     response.status_code)
            last = PlatformError(f"{method} {path}: {response.status_code} {detail}",
                                 response.status_code)
            if response.status_code < 500 and response.status_code != 429:
                raise last
        raise last  # type: ignore[misc]

    # ---- the task ----------------------------------------------------------

    def claim(self) -> dict | None:
        """The claimed task, or None when there is none (204)."""
        response = self._request("POST", f"{BASE}/claim",
                                 json_body={"executor_id": self.executor_id})
        if response.status_code == 204:
            return None
        task = response.json()
        self.task_id, self._lease = str(task["task_id"]), str(task["lease"])
        return task

    def prompts(self) -> Iterator[dict]:
        """This order's problems, one JSON object per line."""
        response = self._request("GET", self._task_path("prompts"))
        for line in response.text.splitlines():
            if line.strip():
                yield json.loads(line)

    def heartbeat(self) -> None:
        self._request("POST", self._task_path("heartbeat"), json_body={})

    def event(self, kind: str, detail: dict) -> None:
        self._request("POST", self._task_path("events"), json_body={"kind": kind,
                                                                    "detail": detail})

    def result(self, *, completion_keys: list[str], vllm_version: str, gpu: str,
               model_sha: str, rows: int, seconds: float) -> None:
        self._request("POST", self._task_path("result"), json_body={
            "completion_keys": completion_keys, "vllm_version": vllm_version, "gpu": gpu,
            "model_sha": model_sha, "rows": rows, "seconds": seconds,
        })

    # ---- uploads -----------------------------------------------------------

    def create_upload(self, *, name: str, size: int, sha256: str) -> dict:
        """``{upload_id, part_size}``."""
        return self._request("POST", self._task_path("uploads"), json_body={
            "name": name, "size": size, "sha256": sha256}).json()

    def upload_status(self, upload_id: str) -> dict:
        """``{parts: set[int], key: str | None}``; ``key`` once it is complete."""
        answer = self._request("GET", self._task_path(f"uploads/{upload_id}"),
                               lost=UPLOAD_LOST_STATUSES).json()
        return {"parts": {int(n) for n in answer.get("parts", ())},
                "key": answer.get("key")}

    def put_part(self, upload_id: str, part: int, body: bytes) -> None:
        self._request("PUT", self._task_path(f"uploads/{upload_id}/{part}"), content=body,
                      headers={PART_SHA_HEADER: sha256_hex(body),
                               "Content-Type": "application/octet-stream"},
                      lost=UPLOAD_LOST_STATUSES)

    def complete_upload(self, upload_id: str) -> str:
        return str(self._request("POST", self._task_path(f"uploads/{upload_id}/complete"),
                                 json_body={}, lost=UPLOAD_LOST_STATUSES).json()["key"])

    def upload_file(self, path: Path, *, name: str, resume: dict | None = None,
                    on_created: Callable[[dict], None] | None = None) -> str:
        """Upload ``path`` in parts and return its key. ``resume`` is a
        ``{upload_id, part_size}`` this file was already started under: only
        the parts the platform lacks are sent, and an upload already complete
        answers its key. An upload that is gone or conflicts is started again,
        once; only a refused token or lease stops."""
        body = Path(path).read_bytes()
        if resume is not None:
            try:
                return self._send(body, resume)
            except UploadConflict as exc:
                logger.warning("upload %s of %s: %s; starting it again",
                               resume.get("upload_id"), name, exc)
            except PlatformError as exc:
                if exc.status != 404:
                    raise
        created = self.create_upload(name=name, size=len(body), sha256=sha256_hex(body))
        upload = {"upload_id": str(created["upload_id"]), "part_size": int(created["part_size"])}
        if on_created is not None:
            on_created(upload)
        return self._send(body, upload, fresh=True)

    def _send(self, body: bytes, upload: dict, *, fresh: bool = False) -> str:
        done: set[int] = set()
        if not fresh:
            status = self.upload_status(upload["upload_id"])
            if status["key"]:
                return str(status["key"])
            done = status["parts"]
        size = int(upload["part_size"])
        if size <= 0:
            raise PlatformError(f"part size {size} is not positive")
        parts = max(1, -(-len(body) // size))
        for index in range(parts):
            number = PART_BASE + index
            if number not in done:
                self.put_part(upload["upload_id"], number, body[index * size:(index + 1) * size])
        return self.complete_upload(upload["upload_id"])

    def close(self) -> None:
        self._http.close()


__all__ = [
    "BASE",
    "LEASE_HEADER",
    "LeaseLost",
    "PART_BASE",
    "PART_SHA_HEADER",
    "PlatformClient",
    "PlatformError",
    "TOKEN_ENV",
    "UploadConflict",
    "sha256_hex",
]
