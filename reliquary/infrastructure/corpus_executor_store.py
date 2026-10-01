"""The audit executor registry: ``reliquary/corpus/executors/{id}.json``.

The admin service registers and revokes executors; the control reads them to
check tokens, and writes heartbeats and quarantines into the same objects.
Every change is a read-modify-write under the object's ETag, so a heartbeat
can never undo a revocation that landed between its read and its write.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.infrastructure.storage import get_s3_client

EXECUTOR_SCHEMA = "reliquary/corpus-executor/v1"
EXECUTOR_PREFIX = "reliquary/corpus/executors/"
EXECUTOR_STATUSES = frozenset({"active", "revoked", "quarantined"})
# Which control an executor serves; a registration without one is the corpus control's.
EXECUTOR_SCOPES = frozenset({"corpus", "eval"})


def scope_of(document: Mapping) -> str:
    return str(document.get("scope") or "corpus")
WRITE_ATTEMPTS = 5
READ_CONCURRENCY = 16

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ABSENT = {"NoSuchKey", "404", "NotFound"}
_CONFLICT = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class ExecutorConflict(RuntimeError):
    """An executor id already registered with other credentials, or a write that
    kept losing its race."""


def validated_executor_id(executor_id: Any) -> str:
    if not isinstance(executor_id, str) or not _ID_RE.fullmatch(executor_id):
        raise ValueError(f"executor id {executor_id!r} is not a name")
    return executor_id


def _key(executor_id: str) -> str:
    return f"{EXECUTOR_PREFIX}{validated_executor_id(executor_id)}.json"


def _bucket(client_kwargs: dict[str, Any]) -> str:
    return client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")


def _code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


async def _get(key: str, **client_kwargs) -> tuple[dict | None, str | None]:
    from botocore.exceptions import ClientError

    bucket = _bucket(client_kwargs)
    async with get_s3_client(**client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if _code(exc) in _ABSENT:
                return None, None
            raise
        return json.loads(await response["Body"].read()), response.get("ETag")


async def _put(key: str, document: Mapping, etag: str | None, **client_kwargs) -> bool:
    """Conditional put; False when it lost the race."""
    from botocore.exceptions import ClientError

    bucket = _bucket(client_kwargs)
    condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
    body = json.dumps(dict(document), sort_keys=True).encode()
    async with get_s3_client(**client_kwargs) as client:
        try:
            await client.put_object(Bucket=bucket, Key=key, Body=body, **condition)
        except ClientError as exc:
            if _code(exc) in _CONFLICT:
                return False
            raise
    return True


async def read_executor(executor_id: str, **client_kwargs) -> dict | None:
    document, _ = await _get(_key(executor_id), **client_kwargs)
    return document


async def register_executor(*, executor_id: str, token_sha256: str, model_id: str,
                            model_revision: str, expires_at: float, now: float,
                            provider_id: str | None = None, host: str | None = None,
                            scope: str = "corpus", **client_kwargs) -> tuple[dict, bool]:
    """Create-only. The same registration again returns the stored one; the
    same id with other credentials or another model is a conflict.
    ``provider_id``/``host`` say where it runs (the eval control pairs
    executors only across both); written only when given."""
    key = _key(executor_id)
    if not isinstance(token_sha256, str) or not _HEX64.fullmatch(token_sha256):
        raise ValueError("token_sha256 must be 64 lowercase hex characters")
    for field, value in (("model_id", model_id), ("model_revision", model_revision)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)) \
            or not math.isfinite(expires_at):
        raise ValueError("expires_at must be a finite unix time")
    document = {
        "schema": EXECUTOR_SCHEMA, "executor_id": executor_id, "token_sha256": token_sha256,
        "model_id": model_id, "model_revision": model_revision,
        "expires_at": float(expires_at), "status": "active", "registered_at": float(now),
        "last_heartbeat": None,
    }
    if scope not in EXECUTOR_SCOPES:
        raise ValueError(f"scope must be one of {sorted(EXECUTOR_SCOPES)}")
    if scope == "eval" and not (provider_id and host):
        raise ValueError("an eval executor needs its provider_id and host")
    if scope != "corpus":
        # Written only off the default, so a corpus registration is unchanged.
        document["scope"] = scope
    for field, value in (("provider_id", provider_id), ("host", host)):
        if value is not None:
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError(f"{field} must be a non-empty string")
            document[field] = value
    for _ in range(WRITE_ATTEMPTS):
        if await _put(key, document, None, **dict(client_kwargs)):
            return document, True
        stored, _ = await _get(key, **dict(client_kwargs))
        if stored is None:
            continue
        same = all(stored.get(f) == document.get(f)
                   for f in ("token_sha256", "model_id", "model_revision", "expires_at",
                             "provider_id", "host", "scope"))
        if not same:
            raise ExecutorConflict(f"executor {executor_id!r} is already registered differently")
        return stored, False
    raise ExecutorConflict(f"executor {executor_id!r} kept changing during registration")


async def _update(executor_id: str, change: Callable[[dict], dict], **client_kwargs) -> dict | None:
    key = _key(executor_id)
    for _ in range(WRITE_ATTEMPTS):
        stored, etag = await _get(key, **dict(client_kwargs))
        if stored is None:
            return None
        updated = change(dict(stored))
        if await _put(key, updated, etag, **dict(client_kwargs)):
            return updated
    raise ExecutorConflict(f"executor {executor_id!r} kept changing under us")


async def set_executor_status(executor_id: str, status: str, *, reason: str | None = None,
                              scope: str | None = None, **client_kwargs) -> dict | None:
    """Revoke or quarantine; never reactivates (a new id is registered instead).
    With ``scope``, an executor of another scope is left untouched (None): the
    eval control can never quarantine a corpus executor."""
    if status not in EXECUTOR_STATUSES - {"active"}:
        raise ValueError(f"status {status!r} is not revoked or quarantined")
    if scope is not None:
        stored = await read_executor(executor_id, **dict(client_kwargs))
        if stored is None or scope_of(stored) != scope:
            return None

    def change(document: dict) -> dict:
        document["status"] = status
        if reason is not None:
            document["status_reason"] = reason
        return document

    return await _update(executor_id, change, **client_kwargs)


async def record_heartbeat(executor_id: str, *, at: float, detail: Mapping | None = None,
                           **client_kwargs) -> dict | None:
    """The control's view of the executor's last contact; status untouched."""

    def change(document: dict) -> dict:
        document["last_heartbeat"] = float(at)
        if detail is not None:
            document["heartbeat"] = dict(detail)
        return document

    return await _update(executor_id, change, **client_kwargs)


async def list_executors(**client_kwargs) -> list[dict]:
    bucket = _bucket(dict(client_kwargs))
    keys: list[str] = []
    async with get_s3_client(**{k: v for k, v in client_kwargs.items()
                                if k != "bucket_name"}) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=EXECUTOR_PREFIX):
            keys.extend(item["Key"] for item in page.get("Contents", ()))
    gate = asyncio.Semaphore(READ_CONCURRENCY)

    async def one(key: str):
        async with gate:
            document, _ = await _get(key, **dict(client_kwargs))
            return document

    found = await asyncio.gather(*(one(k) for k in keys if k.endswith(".json")))
    return [document for document in found if document is not None]


__all__ = [
    "EXECUTOR_PREFIX",
    "EXECUTOR_SCOPES",
    "EXECUTOR_SCHEMA",
    "EXECUTOR_STATUSES",
    "ExecutorConflict",
    "list_executors",
    "read_executor",
    "record_heartbeat",
    "register_executor",
    "scope_of",
    "set_executor_status",
    "validated_executor_id",
]
