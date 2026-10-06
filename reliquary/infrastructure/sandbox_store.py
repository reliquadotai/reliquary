"""Signed-episode sandboxes in R2.

Machines: `reliquary/sandbox/machines/{machine_id}.json`, the directory the validator
places sessions on and verifies signatures with. The admin registers machines and
adds, ends or revokes keys; the validator writes heartbeat summaries. Every change is
a read-modify-write under the object's ETag (the executor registry's pattern), with
bounded, jittered retries, so a summary never undoes an ended key or a status written
between its read and its write.

Keys. A key's end only moves earlier, and is never reopened. On compromise it is set to
the suspected compromise time minus `CLOCK_SKEW_S` (`compromise_valid_until`, the
`end-key --compromise` flag): the verifier accepts an open up to `CLOCK_SKEW_S` before
its token's issuance, so an end at the compromise time itself would still admit a
forged open stamped just before it. Transcripts opened after the end fail
`unknown_key`; rotation (an end at the new key's start) never voids honest work. A
public key is used once: never under a second key id (a compromised key cannot come
back renamed) and never by a second machine.

Sessions: `reliquary/sandbox/sessions/{yyyymmdd of expires_at}/{session_id}.json`, one
document per issued session token (claims, machine, state), never the token's signature.

Status. `active`, `draining`, `revoked`: status decides placement only, keys decide
signatures. Revoking a machine stops new sessions while its keys keep verifying the
transcripts already signed in their windows; going from revoked back to active is
therefore allowed. Machine documents are never deleted: past transcripts need their keys.

The CAS helpers below are the same as `corpus_executor_store`'s.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.infrastructure.storage import get_s3_client

logger = logging.getLogger(__name__)

MACHINE_SCHEMA = "reliquary/sandbox-machine/v1"
MACHINE_PREFIX = "reliquary/sandbox/machines/"
MACHINE_STATUSES = frozenset({"active", "draining", "revoked"})
WRITE_ATTEMPTS = 5
RETRY_BASE_S = 0.05
READ_CONCURRENCY = 16

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# scheme://host[:port]: a DNS name or IPv4 address, or a bracketed IPv6 literal; no
# path, no trailing slash, no empty host or port.
_ADDRESS_RE = re.compile(
    r"^https?://(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|\[[0-9A-Fa-f:.]{2,45}\])"
    r"(?::(?P<port>[0-9]{1,5}))?$")
_ABSENT = {"NoSuchKey", "404", "NotFound"}
_CONFLICT = {"PreconditionFailed", "412", "ConditionalRequestConflict"}
_sleep = asyncio.sleep


class MachineConflict(RuntimeError):
    """A machine registered differently, a key id or public key reused, an end moved
    later, or a write that kept losing its race."""


def _bucket(client_kwargs: dict[str, Any]) -> str:
    return client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")


def _code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


async def _get(key: str, **client_kwargs) -> tuple[Any, str | None]:
    """The decoded JSON at `key` (any JSON value) and its ETag; (None, None) when absent.
    Raises ValueError when the object is not JSON."""
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


async def _backoff(attempt: int) -> None:
    """Full jitter: concurrent losers spread out instead of colliding again."""
    await _sleep(random.uniform(0.0, RETRY_BASE_S * 2 ** attempt))


async def _list_keys(prefix: str, **client_kwargs) -> list[str]:
    bucket = _bucket(dict(client_kwargs))
    keys: list[str] = []
    async with get_s3_client(**{k: v for k, v in client_kwargs.items()
                                if k != "bucket_name"}) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", ()))
    return [key for key in keys if key.endswith(".json")]


async def _read_all(keys, **client_kwargs) -> list[tuple[str, Any]]:
    """(key, decoded JSON) for every key that still exists. An object that is not JSON
    is logged and left out: one bad object never hides the others."""
    gate = asyncio.Semaphore(READ_CONCURRENCY)

    async def one(key: str):
        async with gate:
            try:
                document, etag = await _get(key, **dict(client_kwargs))
            except ValueError:                       # JSONDecodeError, UnicodeDecodeError
                logger.error("sandbox object %s is not JSON; skipped", key)
                return key, None, False
            return key, document, etag is not None   # a JSON null still exists

    return [(k, d) for k, d, exists in await asyncio.gather(*(one(k) for k in keys)) if exists]


def validated_machine_id(machine_id: Any) -> str:
    if not isinstance(machine_id, str) or not _ID_RE.fullmatch(machine_id):
        raise ValueError(f"machine id {machine_id!r} is not a name")
    return machine_id


def validated_address(address: Any) -> str:
    match = _ADDRESS_RE.fullmatch(address) if isinstance(address, str) else None
    if match is None or (match["port"] is not None and not 0 < int(match["port"]) < 65536):
        raise ValueError("address must be scheme://host[:port], with no path or trailing slash")
    return address


def validated_public_key(text: Any) -> str:
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (AttributeError, binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError("public_key_b64 must be base64") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != text:
        raise ValueError("public_key_b64 must be the canonical base64 of 32 bytes (Ed25519)")
    return text


def unix_seconds(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a unix time in whole seconds")
    return value


def _key(machine_id: str) -> str:
    return f"{MACHINE_PREFIX}{validated_machine_id(machine_id)}.json"


def _key_entry(key_id: Any, public_key_b64: Any, valid_from: Any) -> dict:
    validated_machine_id(key_id)                     # same alphabet as ids
    return {"key_id": key_id, "public_key_b64": validated_public_key(public_key_b64),
            "valid_from": unix_seconds(valid_from, "valid_from"), "valid_until": None}


def compromise_valid_until(suspected_at: int) -> int:
    """The end to give a compromised key: the suspected compromise time minus the
    verifier's `CLOCK_SKEW_S`, since an open may precede its token by that much."""
    from reliquary_sandbox.attest import CLOCK_SKEW_S

    return max(0, unix_seconds(suspected_at, "the compromise time") - CLOCK_SKEW_S)


async def _refuse_key_of_another_machine(machine_id: str, public_key_b64: str,
                                         **client_kwargs) -> None:
    """Refuse a public key another machine lists, ended or not. Admin-only and not
    atomic across objects: two admins registering the same key on two machines at the
    same instant could both pass. The directory has one operator."""
    for other in await list_machines(**client_kwargs):
        if other["machine_id"] == machine_id:
            continue
        if any(isinstance(k, Mapping) and k.get("public_key_b64") == public_key_b64
               for k in other.get("keys") or ()):
            raise MachineConflict(f"that public key is already used by machine "
                                  f"{other['machine_id']!r}")


async def register_machine(*, machine_id: str, address: str, provider: str, capacity: int,
                           key_id: str, public_key_b64: str, valid_from: int, now: float,
                           **client_kwargs) -> tuple[dict, bool]:
    """Create-only. The same registration again returns the stored one; the same id
    with another address, provider or first key is a conflict, and so is a public key
    another machine already lists (checked by listing: admin-only, not atomic across
    objects, see `_refuse_key_of_another_machine`)."""
    key = _key(machine_id)
    validated_address(address)
    if not isinstance(provider, str) or not provider.strip() or len(provider) > 64:
        raise ValueError("provider must be a non-empty name")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive number of episodes")
    document = {"schema": MACHINE_SCHEMA, "machine_id": machine_id, "address": address,
                "provider": provider.strip().lower(), "capacity": capacity, "status": "active",
                "keys": [_key_entry(key_id, public_key_b64, valid_from)],
                "registered_at": float(now), "last_heartbeat": None}
    await _refuse_key_of_another_machine(machine_id, public_key_b64, **client_kwargs)
    for attempt in range(WRITE_ATTEMPTS):
        if attempt:
            await _backoff(attempt)
        if await _put(key, document, None, **dict(client_kwargs)):
            return document, True
        stored, _ = await _get(key, **dict(client_kwargs))
        if stored is None:
            continue
        same = (isinstance(stored, Mapping)
                and all(stored.get(f) == document[f] for f in ("address", "provider", "capacity"))
                and (stored.get("keys") or [None])[0] == document["keys"][0])
        if not same:
            raise MachineConflict(f"machine {machine_id!r} is already registered differently")
        return stored, False
    raise MachineConflict(f"machine {machine_id!r} kept changing during registration")


async def _update(machine_id: str, change: Callable[[dict], dict], **client_kwargs) -> dict | None:
    key = _key(machine_id)
    for attempt in range(WRITE_ATTEMPTS):
        if attempt:
            await _backoff(attempt)
        stored, etag = await _get(key, **dict(client_kwargs))
        if stored is None:
            return None
        if not isinstance(stored, Mapping):
            raise ValueError(f"machine {machine_id!r}: the stored object is not a document")
        updated = change(json.loads(json.dumps(stored)))
        if await _put(key, updated, etag, **dict(client_kwargs)):
            return updated
    raise MachineConflict(f"machine {machine_id!r} kept changing under us")


async def add_machine_key(machine_id: str, *, key_id: str, public_key_b64: str, valid_from: int,
                          **client_kwargs) -> dict | None:
    """Add a key. Its id and its public key must be new to this machine (ended keys
    included), and the public key must not be listed by any other machine (checked by
    listing: admin-only, not atomic across objects)."""
    entry = _key_entry(key_id, public_key_b64, valid_from)
    await _refuse_key_of_another_machine(machine_id, public_key_b64, **client_kwargs)

    def change(document: dict) -> dict:
        if any(k["key_id"] == key_id for k in document["keys"]):
            raise MachineConflict(f"key id {key_id!r} is already used by {machine_id!r}")
        if any(k["public_key_b64"] == public_key_b64 for k in document["keys"]):
            raise MachineConflict(f"that public key is already listed by {machine_id!r}, "
                                  "ended or not: a key is never reused")
        document["keys"].append(entry)
        return document

    return await _update(machine_id, change, **client_kwargs)


async def end_machine_key(machine_id: str, *, key_id: str, valid_until: int,
                          **client_kwargs) -> dict | None:
    """End a key at `valid_until` (unix seconds). Rotation: when the new key starts.
    Compromise: `compromise_valid_until(suspected time)`, i.e. the suspected time minus
    `CLOCK_SKEW_S`, past times included. An end only ever moves earlier."""
    unix_seconds(valid_until, "valid_until")

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
    """Any status to any status, revoked to active included: status decides placement,
    keys decide signatures."""
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
    """Every machine document filed under its own id. An object that is not a JSON
    object, or whose `machine_id` is not the one its key names, is logged and left out."""
    found = []
    for key, document in await _read_all(await _list_keys(MACHINE_PREFIX, **client_kwargs),
                                          **client_kwargs):
        filed_as = key[len(MACHINE_PREFIX):-len(".json")]
        if not isinstance(document, Mapping):
            logger.error("sandbox object %s is not a machine document; skipped", key)
        elif document.get("machine_id") != filed_as:
            logger.error("sandbox object %s names machine %r, not %r; skipped",
                         key, str(document.get("machine_id")), filed_as)
        else:
            found.append(dict(document))
    return found


# -- session documents ------------------------------------------------------------

SESSION_PREFIX = "reliquary/sandbox/sessions/"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DAY = 86400


# Session states and the only moves between them, shared by the in-memory book
# (`reliquary.sandbox.sessions.SessionBook.settle`) and the store's CAS, so a stale
# write (a lapse computed before a submission landed) never overwrites a later state.
# Only `live` and `closed_graded` sessions are ever paid (`submitted`); every other
# state is terminal: `closed` (an `open_failed` or unpaid close freed its slot, so a
# later submission could take someone else's), `aborted`, `voided` (ruling 3 as amended
# 2026-10-06: the miner keeps its open refund and reopens) and `lapsed`.
SESSION_LIVE, SESSION_CLOSED_GRADED = "live", "closed_graded"
SESSION_SUBMITTED, SESSION_CLOSED, SESSION_ABORTED = "submitted", "closed", "aborted"
SESSION_VOIDED, SESSION_LAPSED = "voided", "lapsed"
SESSION_TRANSITIONS: dict[str, frozenset[str]] = {
    SESSION_LIVE: frozenset({SESSION_CLOSED_GRADED, SESSION_SUBMITTED, SESSION_CLOSED,
                             SESSION_ABORTED, SESSION_VOIDED, SESSION_LAPSED}),
    SESSION_CLOSED_GRADED: frozenset({SESSION_SUBMITTED, SESSION_LAPSED}),
    SESSION_CLOSED: frozenset(),
    SESSION_VOIDED: frozenset(),
    # Only through an on-time claim (`SessionIssuer.claim(received=...)`): a session that
    # lapsed while its submission, received by the deadline, was being checked.
    SESSION_LAPSED: frozenset({SESSION_SUBMITTED}),
    SESSION_ABORTED: frozenset(),
    SESSION_SUBMITTED: frozenset(),
}


def session_transition_allowed(current: Any, new: Any) -> bool:
    return isinstance(current, str) and new in SESSION_TRANSITIONS.get(current, ())


class SessionStoreConflict(RuntimeError):
    """A session document that already exists, is missing, kept changing, or whose
    stored state may not move to the new one."""


def _check_replace(stored: Any, document: Mapping) -> bool:
    """True to write, False when the stored document is already this one; raises when
    the stored state may not move to the new state."""
    if stored == document:
        return False
    current = stored.get("state") if isinstance(stored, Mapping) else None
    if not session_transition_allowed(current, document.get("state")):
        raise SessionStoreConflict(f"session {document['session_id']}: stored state "
                                   f"{current!r} may not become {document.get('state')!r}")
    return True


def _day(at: int) -> str:
    return time.strftime("%Y%m%d", time.gmtime(int(at)))


def session_key(session_id: str, expires_at: int) -> str:
    """Bucketed by the day the token expires, so a restart lists a few days, not all."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"session id {session_id!r} is not a name")
    return f"{SESSION_PREFIX}{_day(expires_at)}/{session_id}.json"


def _without_secrets(document: Mapping) -> dict:
    """A session document never carries a token (nor its signature)."""
    if "token" in document or "signature" in document:
        raise ValueError("a session document never carries a token or its signature")
    return dict(document)


class R2SessionStore:
    """One document per issued token: claims, machine, state; never the signature."""

    def __init__(self, **client_kwargs) -> None:
        self._kw = client_kwargs

    async def create(self, document: Mapping) -> None:
        document = _without_secrets(document)
        key = session_key(document["session_id"], document["expires_at"])
        if not await _put(key, document, None, **dict(self._kw)):
            raise SessionStoreConflict(f"session {document['session_id']} already exists")

    async def update(self, document: Mapping) -> None:
        document = _without_secrets(document)
        key = session_key(document["session_id"], document["expires_at"])
        for attempt in range(WRITE_ATTEMPTS):
            if attempt:
                await _backoff(attempt)
            stored, etag = await _get(key, **dict(self._kw))
            if stored is None:
                raise SessionStoreConflict(f"session {document['session_id']} is missing")
            if not _check_replace(stored, document):
                return
            if await _put(key, document, etag, **dict(self._kw)):
                return
        raise SessionStoreConflict(f"session {document['session_id']} kept changing")

    async def list_recent(self, now: int) -> list[dict]:
        """Sessions whose token expires from two days ago to tomorrow: every one still
        live, and every one inside the per-hotkey windows (24 h). An object that is not
        a JSON object is logged and left out."""
        keys: list[str] = []
        for offset in (-2, -1, 0, 1):
            keys += await _list_keys(f"{SESSION_PREFIX}{_day(now + offset * _DAY)}/",
                                     **dict(self._kw))
        found = []
        for key, document in await _read_all(keys, **self._kw):
            if isinstance(document, Mapping):
                found.append(dict(document))
            else:
                logger.error("sandbox object %s is not a session document; skipped", key)
        return found


class MemorySessionStore:
    """For tests and for a validator run without R2 persistence."""

    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.fail = False

    async def create(self, document: Mapping) -> None:
        if self.fail:
            raise OSError("bucket down")
        document = _without_secrets(document)
        if document["session_id"] in self.documents:
            raise SessionStoreConflict(document["session_id"])
        self.documents[document["session_id"]] = document

    async def update(self, document: Mapping) -> None:
        if self.fail:
            raise OSError("bucket down")
        document = _without_secrets(document)
        stored = self.documents.get(document["session_id"])
        if stored is None:
            raise SessionStoreConflict(f"session {document['session_id']} is missing")
        if _check_replace(stored, document):
            self.documents[document["session_id"]] = document

    async def list_recent(self, now: int) -> list[dict]:
        return [dict(d) for d in self.documents.values()]
