"""Signed-episode sandboxes in R2.

Machines: `reliquary/sandbox/machines/{machine_id}.json`, the directory the validator
places sessions on and verifies signatures with. The admin registers machines and
adds, ends or revokes keys; the validator writes heartbeat summaries. Every change is
a read-modify-write under the object's ETag (the executor registry's pattern), so a
summary never undoes an ended key or a status written between its read and its write.
A key's end only moves earlier: on compromise it is set to the earliest time the
compromise is suspected, possibly in the past (transcripts opened after it then fail
`unknown_key`). Revoking a machine stops new sessions; its keys stay valid for the
transcripts already signed in their windows.

The CAS helpers below are the same as `corpus_executor_store`'s.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.infrastructure.storage import get_s3_client

MACHINE_SCHEMA = "reliquary/sandbox-machine/v1"
MACHINE_PREFIX = "reliquary/sandbox/machines/"
MACHINE_STATUSES = frozenset({"active", "draining", "revoked"})
WRITE_ATTEMPTS = 5
READ_CONCURRENCY = 16

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ADDRESS_RE = re.compile(r"^https?://[A-Za-z0-9.\-\[\]:]+$")
_ABSENT = {"NoSuchKey", "404", "NotFound"}
_CONFLICT = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class MachineConflict(RuntimeError):
    """A machine registered differently, a key id reused, an end moved later, or a
    write that kept losing its race."""


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


async def _list_keys(prefix: str, **client_kwargs) -> list[str]:
    bucket = _bucket(dict(client_kwargs))
    keys: list[str] = []
    async with get_s3_client(**{k: v for k, v in client_kwargs.items()
                                if k != "bucket_name"}) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", ()))
    return [key for key in keys if key.endswith(".json")]


async def _read_all(keys, **client_kwargs) -> list[dict]:
    gate = asyncio.Semaphore(READ_CONCURRENCY)

    async def one(key: str):
        async with gate:
            document, _ = await _get(key, **dict(client_kwargs))
            return document

    return [d for d in await asyncio.gather(*(one(k) for k in keys)) if d is not None]


def validated_machine_id(machine_id: Any) -> str:
    if not isinstance(machine_id, str) or not _ID_RE.fullmatch(machine_id):
        raise ValueError(f"machine id {machine_id!r} is not a name")
    return machine_id


def _key(machine_id: str) -> str:
    return f"{MACHINE_PREFIX}{validated_machine_id(machine_id)}.json"


def _check_public_key(text: Any) -> str:
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (AttributeError, binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError("public_key_b64 must be base64") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != text:
        raise ValueError("public_key_b64 must be the canonical base64 of 32 bytes (Ed25519)")
    return text


def _key_entry(key_id: Any, public_key_b64: Any, valid_from: Any) -> dict:
    validated_machine_id(key_id)                     # same alphabet as ids
    if isinstance(valid_from, bool) or not isinstance(valid_from, int) or valid_from < 0:
        raise ValueError("valid_from must be a unix time in whole seconds")
    return {"key_id": key_id, "public_key_b64": _check_public_key(public_key_b64),
            "valid_from": valid_from, "valid_until": None}


async def register_machine(*, machine_id: str, address: str, provider: str, capacity: int,
                           key_id: str, public_key_b64: str, valid_from: int, now: float,
                           **client_kwargs) -> tuple[dict, bool]:
    """Create-only. The same registration again returns the stored one; the same id
    with another address, provider or first key is a conflict."""
    key = _key(machine_id)
    if not isinstance(address, str) or not _ADDRESS_RE.fullmatch(address):
        raise ValueError("address must be scheme://host[:port], with no path or trailing slash")
    if not isinstance(provider, str) or not provider.strip() or len(provider) > 64:
        raise ValueError("provider must be a non-empty name")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive number of episodes")
    document = {"schema": MACHINE_SCHEMA, "machine_id": machine_id, "address": address,
                "provider": provider.strip().lower(), "capacity": capacity, "status": "active",
                "keys": [_key_entry(key_id, public_key_b64, valid_from)],
                "registered_at": float(now), "last_heartbeat": None}
    for _ in range(WRITE_ATTEMPTS):
        if await _put(key, document, None, **dict(client_kwargs)):
            return document, True
        stored, _ = await _get(key, **dict(client_kwargs))
        if stored is None:
            continue
        same = (all(stored.get(f) == document[f] for f in ("address", "provider", "capacity"))
                and stored.get("keys", [None])[0] == document["keys"][0])
        if not same:
            raise MachineConflict(f"machine {machine_id!r} is already registered differently")
        return stored, False
    raise MachineConflict(f"machine {machine_id!r} kept changing during registration")


async def _update(machine_id: str, change: Callable[[dict], dict], **client_kwargs) -> dict | None:
    key = _key(machine_id)
    for _ in range(WRITE_ATTEMPTS):
        stored, etag = await _get(key, **dict(client_kwargs))
        if stored is None:
            return None
        updated = change(json.loads(json.dumps(stored)))
        if await _put(key, updated, etag, **dict(client_kwargs)):
            return updated
    raise MachineConflict(f"machine {machine_id!r} kept changing under us")


async def add_machine_key(machine_id: str, *, key_id: str, public_key_b64: str, valid_from: int,
                          **client_kwargs) -> dict | None:
    entry = _key_entry(key_id, public_key_b64, valid_from)

    def change(document: dict) -> dict:
        if any(k["key_id"] == key_id for k in document["keys"]):
            raise MachineConflict(f"key id {key_id!r} is already used by {machine_id!r}")
        document["keys"].append(entry)
        return document

    return await _update(machine_id, change, **client_kwargs)


async def end_machine_key(machine_id: str, *, key_id: str, valid_until: int,
                          **client_kwargs) -> dict | None:
    """End a key at `valid_until` (unix seconds). Rotation: when the new key starts.
    Compromise: the earliest time the compromise is suspected, past times included.
    An end only ever moves earlier."""
    if isinstance(valid_until, bool) or not isinstance(valid_until, int) or valid_until < 0:
        raise ValueError("valid_until must be a unix time in whole seconds")

    def change(document: dict) -> dict:
        for entry in document["keys"]:
            if entry["key_id"] == key_id:
                current = entry.get("valid_until")
                if current is not None and valid_until > current:
                    raise MachineConflict("a key's end only ever moves earlier")
                entry["valid_until"] = valid_until
                return document
        raise MachineConflict(f"machine {machine_id!r} has no key {key_id!r}")

    return await _update(machine_id, change, **client_kwargs)


async def set_machine_status(machine_id: str, status: str, *, reason: str | None = None,
                             **client_kwargs) -> dict | None:
    if status not in MACHINE_STATUSES:
        raise ValueError(f"status must be one of {sorted(MACHINE_STATUSES)}")

    def change(document: dict) -> dict:
        document["status"] = status
        if reason is not None:
            document["status_reason"] = reason
        return document

    return await _update(machine_id, change, **client_kwargs)


async def record_machine_heartbeat(machine_id: str, *, at: float, summary: Mapping,
                                   **client_kwargs) -> dict | None:
    def change(document: dict) -> dict:
        document["last_heartbeat"] = float(at)
        document["heartbeat"] = dict(summary)
        return document

    return await _update(machine_id, change, **client_kwargs)


async def read_machine(machine_id: str, **client_kwargs) -> dict | None:
    document, _ = await _get(_key(machine_id), **client_kwargs)
    return document


async def list_machines(**client_kwargs) -> list[dict]:
    return await _read_all(await _list_keys(MACHINE_PREFIX, **client_kwargs), **client_kwargs)
