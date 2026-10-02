"""The corpus job manifest and its ledgers, persisted with the same
compare-and-swap discipline as ``task_registry_store``: two validators racing
to admit against the same slot/cursor ledgers must not both win.

Modeled on ``task_registry_store.py``: the same absent/conflict error-code
sets, the same ETag compare-and-swap shape, the same "an absent object is
not an error" rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import weakref
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.corpus.job import JOB_ID_RE, JobError, JobSpec, parse_job
from reliquary.infrastructure.storage import get_s3_client

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "reliquary/corpus/jobs/"
# A long-lived client is rebuilt this often even when healthy, so a pool that
# degrades without ever raising (the reason storage.py builds one per call)
# cannot outlive a few minutes.
CLIENT_MAX_AGE_SECONDS = 600.0

# A sealed segment of the seen-digest set; see ``write_seen_segment``.
SEEN_SEGMENT_SCHEMA = "reliquary/corpus-seen-segment/v1"
_SEGMENT_ID_RE = re.compile(r"^[0-9a-f]{64}$")
# A create-only segment PUT can meet R2's 409 (another conditional write to the
# key in flight): it is retried this many times, backing off from this delay.
SEGMENT_WRITE_ATTEMPTS = 4
SEGMENT_RETRY_DELAY_SECONDS = 0.2

_ABSENT_CODES = {"NoSuchKey", "404", "NotFound"}
_CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class CorpusStoreConflict(Exception):
    """A concurrent writer landed first; the caller must re-read and retry."""


class CorpusSegmentCorrupt(Exception):
    """A seen segment a ledger names is absent or is not the bytes its name
    hashes: reading it as empty would admit every duplicate it held."""


def _error_code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def _validated_job_id(job_id: Any) -> str:
    # The id is interpolated into a bucket key right after this call: a
    # traversal or a wildcard value must never reach that interpolation.
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ValueError(f"unusable job id {job_id!r}")
    return job_id


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}.json"


def _ledgers_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}/ledgers.json"


def _ledgers_backup_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}/ledgers.v1-backup.json"


def _validated_segment_id(segment_id: Any) -> str:
    if not isinstance(segment_id, str) or not _SEGMENT_ID_RE.match(segment_id):
        raise ValueError(f"unusable seen segment id {segment_id!r}")
    return segment_id


def _seen_segment_key(job_id: str, segment_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}/seen/{segment_id}.json"


def _bucket(client_kwargs: dict[str, Any]) -> str:
    return client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")


def _encode(document: Any) -> bytes:
    # Sorted, compact bytes: two writers of the same content produce the same
    # object, and no bare NaN/Infinity literal reaches a reader that rejects them.
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class _Slot:
    __slots__ = ("context", "client", "born", "users", "retired")

    def __init__(self, context: Any, client: Any, born: float) -> None:
        self.context, self.client, self.born = context, client, born
        self.users = 0
        self.retired = False


class _ClientPool:
    """One long-lived S3 client per event loop, instead of one per call.

    Building an aiobotocore client re-reads and parses botocore's service
    model, which held the corpus validator's loop at a full core. A client is
    retired after a connection-level error (a server answer such as 404 or 412
    is not one) or past ``max_age_seconds``, and closed once its last in-flight
    call returns.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        max_age_seconds: float = CLIENT_MAX_AGE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._max_age = max_age_seconds
        self._clock = clock
        # Keyed weakly by loop: a client's connections belong to the loop that
        # opened them, and a finished loop must not pin its client.
        self._slots: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    async def _acquire(self) -> _Slot:
        loop = asyncio.get_running_loop()
        lock = self._locks.setdefault(loop, asyncio.Lock())
        async with lock:
            slot = self._slots.get(loop)
            if slot is not None and self._clock() - slot.born > self._max_age:
                await self._retire(loop, slot)
                slot = None
            if slot is None:
                context = self._factory()
                client = await context.__aenter__()
                slot = self._slots[loop] = _Slot(context, client, self._clock())
            slot.users += 1
            return slot

    async def _retire(self, loop, slot: _Slot) -> None:
        slot.retired = True
        if self._slots.get(loop) is slot:
            del self._slots[loop]
        if slot.users == 0:
            await _close(slot)

    @contextlib.asynccontextmanager
    async def client(self):
        slot = await self._acquire()
        try:
            yield slot.client
        except Exception as exc:
            if not _is_server_answer(exc):
                logger.warning("retiring S3 client after %r", exc)
                slot.retired = True
                loop = asyncio.get_running_loop()
                if self._slots.get(loop) is slot:
                    del self._slots[loop]
            raise
        finally:
            slot.users -= 1
            if slot.retired and slot.users == 0:
                await _close(slot)


async def _close(slot: _Slot) -> None:
    try:
        await slot.context.__aexit__(None, None, None)
    except Exception:
        # Closing a client whose connection already broke may itself fail.
        logger.debug("closing a retired S3 client failed", exc_info=True)


def _is_server_answer(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    return isinstance(exc, (ClientError, CorpusStoreConflict))


def _client(pool: _ClientPool | None, client_kwargs: dict[str, Any]):
    # Unbound callers keep the historic fresh client per call.
    return pool.client() if pool is not None else get_s3_client(**client_kwargs)


def _decode(body: bytes) -> Any:
    return json.loads(body)


async def _get(
    key: str, *, pool: _ClientPool | None = None, **client_kwargs
) -> tuple[bytes | None, str | None]:
    from botocore.exceptions import ClientError

    bucket = _bucket(client_kwargs)
    async with _client(pool, client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _ABSENT_CODES:
                return None, None
            raise
        try:
            body = await response["Body"].read()
        finally:
            # Released even when the read fails, so no connection stays held.
            close = getattr(response["Body"], "close", None)
            if close is not None:
                close()
        return body, response.get("ETag")


async def _put(
    key: str, body: bytes, etag: str | None, *, pool: _ClientPool | None = None,
    **client_kwargs,
) -> str | None:
    from botocore.exceptions import ClientError

    bucket = _bucket(client_kwargs)
    condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
    async with _client(pool, client_kwargs) as client:
        try:
            response = await client.put_object(Bucket=bucket, Key=key, Body=body, **condition)
        except ClientError as exc:
            if _error_code(exc) in _CONFLICT_CODES:
                raise CorpusStoreConflict(f"{key} changed under us") from exc
            raise
    return response.get("ETag")


async def read_job(job_id: str, **client_kwargs) -> tuple[JobSpec | None, str | None]:
    """The job and the ETag to write it back against. Absent reads as (None, None).

    Parses through ``parse_job`` here, so a hand-edited or corrupt manifest
    raises ``JobError`` — naming the field — at read time, rather than being
    handed to ``admit()`` where it would misjudge the first submission.
    """
    validated = _validated_job_id(job_id)
    body, etag = await _get(_job_key(validated), **client_kwargs)
    if body is None:
        return None, None
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise JobError(f"stored job {job_id!r} is not JSON: {exc}") from exc
    parsed = parse_job(raw)
    if parsed.job_id != validated:
        # The key IS the job's identity: the ledgers hang off it and the
        # registry entry names it. A manifest declaring another id would be
        # served here while judging submissions as that other job.
        raise JobError(
            f"job stored at {validated!r} declares itself {parsed.job_id!r}"
        )
    return parsed, etag


async def write_job(job: Mapping[str, Any], etag: str | None, **client_kwargs) -> str | None:
    """Conditional put of a raw manifest mapping (not a ``JobSpec``).

    Validated with ``parse_job`` before the write, so a manifest ``read_job``
    could never parse back is refused here instead of reaching the bucket.
    Raises ``CorpusStoreConflict`` if a concurrent writer won the race.
    """
    job_id = _validated_job_id(job.get("job_id") if isinstance(job, Mapping) else None)
    parsed = parse_job(job)
    body = _encode(parsed.to_contract())
    return await _put(_job_key(job_id), body, etag, **client_kwargs)


async def delete_job(job_id: str, **client_kwargs) -> None:
    """Remove a manifest. Unconditional, and safe only for the one caller that
    needs it: ``write_job`` with no ETag CREATES (``IfNoneMatch: "*"``), so a
    declaration rolling itself back is deleting an object it just made."""
    validated = _validated_job_id(job_id)
    bucket = _bucket(client_kwargs)
    async with get_s3_client(**client_kwargs) as client:
        await client.delete_object(Bucket=bucket, Key=_job_key(validated))


async def list_jobs(**client_kwargs) -> list[str]:
    """Every job with a manifest, declared or not — an orphan must be visible."""
    bucket = _bucket(client_kwargs)
    job_ids: list[str] = []
    async with get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=JOB_KEY_PREFIX):
            for obj in page.get("Contents", []) or []:
                name = obj["Key"][len(JOB_KEY_PREFIX):]
                if not name.endswith(".json"):
                    continue
                # The ledgers object lives at `{job_id}/ledgers.json`, whose
                # stem carries a slash and so cannot match a job id.
                stem = name[: -len(".json")]
                if JOB_ID_RE.match(stem):
                    job_ids.append(stem)
    return sorted(job_ids)


async def read_ledgers(job_id: str, **client_kwargs) -> tuple[dict, str | None]:
    """The slot/cursor ledger snapshot and its ETag. Absent reads as ({}, None)."""
    validated = _validated_job_id(job_id)
    body, etag = await _get(_ledgers_key(validated), **client_kwargs)
    if body is None:
        return {}, None
    # Megabytes once a job is busy: parsed off the loop the route shares.
    return await asyncio.to_thread(_decode, body), etag


async def write_ledgers(
    job_id: str, snapshot: Mapping[str, Any], etag: str | None, **client_kwargs
) -> str | None:
    """Conditional put of the ledger snapshot. Raises ``CorpusStoreConflict``
    if a concurrent writer won the race."""
    validated = _validated_job_id(job_id)
    body = await asyncio.to_thread(_encode, dict(snapshot))
    return await _put(_ledgers_key(validated), body, etag, **client_kwargs)


def _is_strictly_sorted(digests) -> bool:
    return all(a < b for a, b in zip(digests, digests[1:]))


def _segment_body(digests) -> bytes:
    digests = list(digests)
    if not digests or any(not isinstance(d, str) for d in digests):
        raise ValueError("a seen segment holds at least one digest string")
    if not _is_strictly_sorted(digests):
        # Sorted and unique is what makes the name a function of the set.
        raise ValueError("a seen segment's digests must be sorted and unique")
    return _encode({"schema": SEEN_SEGMENT_SCHEMA, "digests": digests})


async def write_seen_segment(job_id: str, digests, **client_kwargs) -> str:
    """Create-only put of a sealed segment; returns its id, the sha256 of its
    bytes. A refused create counts as written only once the segment reads back
    under its hash: R2's 409 means a write is in flight, not that one landed."""
    validated = _validated_job_id(job_id)
    body = await asyncio.to_thread(_segment_body, digests)
    segment_id = hashlib.sha256(body).hexdigest()
    key = _seen_segment_key(validated, segment_id)
    for attempt in range(SEGMENT_WRITE_ATTEMPTS):
        try:
            await _put(key, body, None, **client_kwargs)
            return segment_id
        except CorpusStoreConflict:
            pass
        stored, _ = await _get(key, **client_kwargs)
        if stored is not None:
            # Present but not these bytes raises CorpusSegmentCorrupt.
            await asyncio.to_thread(_decode_segment, segment_id, stored)
            return segment_id
        if attempt + 1 < SEGMENT_WRITE_ATTEMPTS:
            await asyncio.sleep(SEGMENT_RETRY_DELAY_SECONDS * 2 ** attempt)
    raise CorpusStoreConflict(f"{key} stayed contended and absent")


def _decode_segment(segment_id: str, body: bytes) -> tuple[str, ...]:
    if hashlib.sha256(body).hexdigest() != segment_id:
        raise CorpusSegmentCorrupt(f"seen segment {segment_id} does not hash to its name")
    try:
        document = json.loads(body)
    except ValueError as exc:
        raise CorpusSegmentCorrupt(f"seen segment {segment_id} is not JSON") from exc
    digests = document.get("digests") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "digests"}
        or document["schema"] != SEEN_SEGMENT_SCHEMA
        or not isinstance(digests, list)
        or not digests
        or any(not isinstance(d, str) for d in digests)
        or not _is_strictly_sorted(digests)
    ):
        raise CorpusSegmentCorrupt(f"seen segment {segment_id} is not a sorted digest list")
    return tuple(digests)


async def read_seen_segment(job_id: str, segment_id: str, **client_kwargs) -> tuple[str, ...]:
    """The segment's digests, checked against its name. Absent is corrupt."""
    validated = _validated_job_id(job_id)
    segment_id = _validated_segment_id(segment_id)
    body, _ = await _get(_seen_segment_key(validated, segment_id), **client_kwargs)
    if body is None:
        raise CorpusSegmentCorrupt(f"seen segment {segment_id} of job {job_id!r} is absent")
    return await asyncio.to_thread(_decode_segment, segment_id, body)


async def write_ledgers_backup(
    job_id: str, snapshot: Mapping[str, Any], **client_kwargs
) -> bool:
    """Create-only copy of the v1 ledgers taken before migration. False when a
    backup already exists; the first one is kept."""
    validated = _validated_job_id(job_id)
    body = await asyncio.to_thread(_encode, dict(snapshot))
    try:
        await _put(_ledgers_backup_key(validated), body, None, **client_kwargs)
    except CorpusStoreConflict:
        return False
    return True


async def read_ledgers_backup(job_id: str, **client_kwargs) -> dict | None:
    validated = _validated_job_id(job_id)
    body, _ = await _get(_ledgers_backup_key(validated), **client_kwargs)
    if body is None:
        return None
    return await asyncio.to_thread(_decode, body)


class BucketJobStore:
    """The three calls the submission endpoint makes, bound to one bucket.

    The endpoint takes an object rather than this module so that a test can
    hand it a fake; binding the client kwargs once, here, is what keeps
    storage configuration out of the request path.
    """

    __slots__ = ("_client_kwargs", "_pool", "_jobs", "_ledgers", "_generation")

    def __init__(self, **client_kwargs: Any) -> None:
        self._client_kwargs = client_kwargs
        credentials = {k: v for k, v in client_kwargs.items() if k != "bucket_name"}
        # Resolved at build time, not bound here, so a patched `get_s3_client` applies.
        self._pool = _ClientPool(lambda: get_s3_client(**credentials))
        # A manifest never changes under its id, so a found one is kept.
        self._jobs: dict[str, tuple[JobSpec, str | None]] = {}
        # The last ledger snapshot this store read or wrote, with its ETag. Its
        # only use is the next compare-and-swap, which R2 still judges: a stale
        # entry costs a conflict, which forgets it, never a lost write.
        self._ledgers: dict[str, tuple[dict, str]] = {}
        # Bumped by every write before its PUT: a read or write that started
        # under an older generation must not fill the memory with what it saw.
        self._generation: dict[str, int] = {}

    async def read_job(self, job_id: str) -> tuple[JobSpec | None, str | None]:
        cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        job, etag = await read_job(job_id, pool=self._pool, **self._client_kwargs)
        if job is not None:
            self._jobs[job_id] = (job, etag)
        return job, etag

    async def read_ledgers(self, job_id: str) -> tuple[dict, str | None]:
        """The snapshot is shared with the memory: callers must not mutate it."""
        cached = self._ledgers.get(job_id)
        if cached is not None:
            return cached
        generation = self._generation.get(job_id, 0)
        snapshot, etag = await read_ledgers(job_id, pool=self._pool, **self._client_kwargs)
        if (etag is not None and self._generation.get(job_id, 0) == generation
                and job_id not in self._ledgers):
            self._ledgers[job_id] = (snapshot, etag)
        return snapshot, etag

    async def write_ledgers(
        self, job_id: str, snapshot: Mapping[str, Any], etag: str | None
    ) -> str | None:
        # Forgotten before the write: a conflict, or a failure that may or may
        # not have landed, must send the next read to the bucket.
        self._ledgers.pop(job_id, None)
        generation = self._generation[job_id] = self._generation.get(job_id, 0) + 1
        new_etag = await write_ledgers(
            job_id, snapshot, etag, pool=self._pool, **self._client_kwargs
        )
        if new_etag is not None and self._generation[job_id] == generation:
            self._ledgers[job_id] = (dict(snapshot), new_etag)
        return new_etag

    async def write_seen_segment(self, job_id: str, digests) -> str:
        return await write_seen_segment(job_id, digests, pool=self._pool, **self._client_kwargs)

    async def read_seen_segment(self, job_id: str, segment_id: str) -> tuple[str, ...]:
        return await read_seen_segment(
            job_id, segment_id, pool=self._pool, **self._client_kwargs
        )

    async def write_ledgers_backup(self, job_id: str, snapshot: Mapping[str, Any]) -> bool:
        return await write_ledgers_backup(
            job_id, snapshot, pool=self._pool, **self._client_kwargs
        )

    async def read_ledgers_backup(self, job_id: str) -> dict | None:
        return await read_ledgers_backup(job_id, pool=self._pool, **self._client_kwargs)
