"""What a corpus job produced: accepted submissions, their audit verdicts, and
the settlement state. Beside the job store, under the same job prefix.

Submissions and verdicts are create-only, so a retry or a second auditor can
never overwrite one; the settlement state is compare-and-swap, because it is
what decides which verdicts have already been paid.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import re
from typing import Any

from reliquary.infrastructure.corpus_job_store import (
    JOB_KEY_PREFIX,
    CorpusStoreConflict,
    _ClientPool,
    _bucket,
    _decode,
    _encode,
    _get,
    _put,
    _validated_job_id,
)
from reliquary.infrastructure.storage import get_s3_client, off_loop

_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _validated_id(submission_id: Any) -> str:
    if not isinstance(submission_id, str) or not _ID_RE.match(submission_id):
        raise ValueError(f"unusable submission id {submission_id!r}")
    return submission_id


def _prefix(job_id: str, kind: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/{kind}/"


def _key(job_id: str, kind: str, submission_id: str) -> str:
    return f"{_prefix(job_id, kind)}{_validated_id(submission_id)}.json"


async def _off(executor, func, *args):
    """``func`` on ``executor``'s threads, or the loop's default ones when None."""
    if executor is None:
        return await asyncio.to_thread(func, *args)
    return await asyncio.get_running_loop().run_in_executor(executor, func, *args)


async def _create(key: str, document: Mapping, *, executor=None, **client_kwargs) -> bool:
    try:
        # Encoded and decoded off the loop, which the route and auditor share.
        body = await _off(executor, _encode, dict(document))
        await _put(key, body, None, **client_kwargs)
    except CorpusStoreConflict:
        return False
    return True


async def _read(key: str, *, executor=None, **client_kwargs) -> dict | None:
    body, _ = await _get(key, **client_kwargs)
    return None if body is None else await _off(executor, _decode, body)


async def _list_ids(prefix: str, *, pool: _ClientPool | None = None, **client_kwargs) -> list[str]:
    # A job's listing is 100k+ keys of XML: paged and parsed off the serving loop.
    return await off_loop(_paginate_ids(prefix, pool=pool, **client_kwargs))


async def _paginate_ids(prefix: str, *, pool: _ClientPool | None = None, **client_kwargs) -> list[str]:
    bucket = _bucket(client_kwargs)
    ids: list[str] = []
    async with (pool.client() if pool is not None else get_s3_client(**client_kwargs)) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                name = obj["Key"][len(prefix):]
                stem = name[: -len(".json")] if name.endswith(".json") else ""
                if _ID_RE.match(stem):
                    ids.append(stem)
    return sorted(ids)


async def write_submission(job_id, submission_id, record, **client_kwargs) -> bool:
    return await _create(_key(job_id, "submissions", submission_id), record, **client_kwargs)


async def read_submission(job_id, submission_id, **client_kwargs) -> dict | None:
    return await _read(_key(job_id, "submissions", submission_id), **client_kwargs)


async def list_submission_ids(job_id, **client_kwargs) -> list[str]:
    return await _list_ids(_prefix(job_id, "submissions"), **client_kwargs)


async def write_verdict(job_id, submission_id, verdict, **client_kwargs) -> bool:
    return await _create(_key(job_id, "verdicts", submission_id), verdict, **client_kwargs)


async def read_verdict(job_id, submission_id, **client_kwargs) -> dict | None:
    return await _read(_key(job_id, "verdicts", submission_id), **client_kwargs)


async def list_verdict_ids(job_id, **client_kwargs) -> list[str]:
    return await _list_ids(_prefix(job_id, "verdicts"), **client_kwargs)


async def write_voided(job_id, submission_id, document, **client_kwargs) -> bool:
    return await _create(_key(job_id, "voided", submission_id), document, **client_kwargs)


async def list_voided_ids(job_id, **client_kwargs) -> list[str]:
    return await _list_ids(_prefix(job_id, "voided"), **client_kwargs)


def _settlement_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/settlement.json"


async def read_settlement(job_id, *, executor=None, **client_kwargs) -> tuple[dict, str | None]:
    body, etag = await _get(_settlement_key(job_id), **client_kwargs)
    return ({}, None) if body is None else (await _off(executor, _decode, body), etag)


async def write_settlement(job_id, state, etag, *, executor=None, **client_kwargs) -> str | None:
    body = await _off(executor, _encode, dict(state))
    return await _put(_settlement_key(job_id), body, etag, **client_kwargs)


def _final_status_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/final-status.json"


async def read_final_status(job_id, **client_kwargs) -> dict | None:
    return await _read(_final_status_key(job_id), **client_kwargs)


async def write_final_status(job_id, status, **client_kwargs) -> bool:
    """A drained job's last status, beside its settlement: create-only (a
    drained job never changes); False when one is already written."""
    return await _create(_final_status_key(job_id), status, **client_kwargs)


def _miners_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/miners.json"


async def read_miners(job_id, *, executor=None, **client_kwargs) -> tuple[dict, str | None]:
    """Every hotkey's audit state for this job, whole-document. Absent reads
    as ({}, None): a hotkey with no entry is handled by the caller (§5,
    "unknown is probation"), not by this store."""
    body, etag = await _get(_miners_key(job_id), **client_kwargs)
    return ({}, None) if body is None else (await _off(executor, _decode, body), etag)


async def write_miners(job_id, state, etag, *, executor=None, **client_kwargs) -> str | None:
    """Compare-and-swap of the whole miners document, like the settlement
    state: two auditors racing on different hotkeys must not let one
    overwrite the other's write."""
    body = await _off(executor, _encode, dict(state))
    return await _put(_miners_key(job_id), body, etag, **client_kwargs)


class BucketRecordStore:
    """The record calls bound to one bucket, so tests can hand the services a fake."""

    __slots__ = ("_kw", "_max_reads", "_reads")

    def __init__(self, *, max_pool_connections: int | None = None, executor=None,
                 max_reads: int | None = None, **client_kwargs: Any) -> None:
        credentials = {k: v for k, v in client_kwargs.items() if k != "bucket_name"}
        if max_pool_connections:
            # Its own connection pool, larger than botocore's 10.
            credentials["max_pool_connections"] = max_pool_connections
        # One long-lived client for every call (see `_ClientPool`); resolved
        # at build time so a patched `get_s3_client` applies.
        pool = _ClientPool(lambda: get_s3_client(**credentials))
        # `executor`: whose threads encode and decode (the loop's default when
        # None); the judges pass their own so the route's never wait for them.
        self._kw = {**client_kwargs, "pool": pool, "executor": executor}
        # Record reads in flight at once (records run to megabytes); None: no bound.
        self._max_reads = max_reads
        self._reads: asyncio.Semaphore | None = None

    async def write_submission(self, job_id, submission_id, record):
        return await write_submission(job_id, submission_id, record, **self._kw)

    async def read_submission(self, job_id, submission_id):
        if self._max_reads is None:
            return await read_submission(job_id, submission_id, **self._kw)
        if self._reads is None:
            self._reads = asyncio.Semaphore(self._max_reads)
        async with self._reads:
            return await read_submission(job_id, submission_id, **self._kw)

    async def list_submission_ids(self, job_id):
        return await list_submission_ids(job_id, **self._kw)

    async def write_verdict(self, job_id, submission_id, verdict):
        return await write_verdict(job_id, submission_id, verdict, **self._kw)

    async def read_verdict(self, job_id, submission_id):
        return await read_verdict(job_id, submission_id, **self._kw)

    async def list_verdict_ids(self, job_id):
        return await list_verdict_ids(job_id, **self._kw)

    async def write_voided(self, job_id, submission_id, document):
        return await write_voided(job_id, submission_id, document, **self._kw)

    async def list_voided_ids(self, job_id):
        return await list_voided_ids(job_id, **self._kw)

    async def read_settlement(self, job_id):
        return await read_settlement(job_id, **self._kw)

    async def write_settlement(self, job_id, state, etag):
        return await write_settlement(job_id, state, etag, **self._kw)

    async def read_final_status(self, job_id):
        return await read_final_status(job_id, **self._kw)

    async def write_final_status(self, job_id, status):
        return await write_final_status(job_id, status, **self._kw)

    async def read_miners(self, job_id):
        return await read_miners(job_id, **self._kw)

    async def write_miners(self, job_id, state, etag):
        return await write_miners(job_id, state, etag, **self._kw)
