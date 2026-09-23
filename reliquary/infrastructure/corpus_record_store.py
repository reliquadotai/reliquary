"""What a corpus job produced: accepted submissions, their audit verdicts, and
the settlement state. Beside the job store, under the same job prefix.

Submissions and verdicts are create-only, so a retry or a second auditor can
never overwrite one; the settlement state is compare-and-swap, because it is
what decides which verdicts have already been paid.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from reliquary.infrastructure.corpus_job_store import (
    JOB_KEY_PREFIX,
    CorpusStoreConflict,
    _bucket,
    _encode,
    _get,
    _put,
    _validated_job_id,
)
from reliquary.infrastructure.storage import get_s3_client

_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _validated_id(submission_id: Any) -> str:
    if not isinstance(submission_id, str) or not _ID_RE.match(submission_id):
        raise ValueError(f"unusable submission id {submission_id!r}")
    return submission_id


def _prefix(job_id: str, kind: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/{kind}/"


def _key(job_id: str, kind: str, submission_id: str) -> str:
    return f"{_prefix(job_id, kind)}{_validated_id(submission_id)}.json"


async def _create(key: str, document: Mapping, **client_kwargs) -> bool:
    try:
        await _put(key, _encode(dict(document)), None, **client_kwargs)
    except CorpusStoreConflict:
        return False
    return True


async def _read(key: str, **client_kwargs) -> dict | None:
    body, _ = await _get(key, **client_kwargs)
    return None if body is None else json.loads(body)


async def _list_ids(prefix: str, **client_kwargs) -> list[str]:
    bucket = _bucket(client_kwargs)
    ids: list[str] = []
    async with get_s3_client(**client_kwargs) as client:
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


def _settlement_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{_validated_job_id(job_id)}/settlement.json"


async def read_settlement(job_id, **client_kwargs) -> tuple[dict, str | None]:
    body, etag = await _get(_settlement_key(job_id), **client_kwargs)
    return ({}, None) if body is None else (json.loads(body), etag)


async def write_settlement(job_id, state, etag, **client_kwargs) -> str | None:
    return await _put(_settlement_key(job_id), _encode(dict(state)), etag, **client_kwargs)


class BucketRecordStore:
    """The record calls bound to one bucket, so tests can hand the services a fake."""

    __slots__ = ("_kw",)

    def __init__(self, **client_kwargs: Any) -> None:
        self._kw = client_kwargs

    async def write_submission(self, job_id, submission_id, record):
        return await write_submission(job_id, submission_id, record, **self._kw)

    async def read_submission(self, job_id, submission_id):
        return await read_submission(job_id, submission_id, **self._kw)

    async def list_submission_ids(self, job_id):
        return await list_submission_ids(job_id, **self._kw)

    async def write_verdict(self, job_id, submission_id, verdict):
        return await write_verdict(job_id, submission_id, verdict, **self._kw)

    async def read_verdict(self, job_id, submission_id):
        return await read_verdict(job_id, submission_id, **self._kw)

    async def list_verdict_ids(self, job_id):
        return await list_verdict_ids(job_id, **self._kw)

    async def read_settlement(self, job_id):
        return await read_settlement(job_id, **self._kw)

    async def write_settlement(self, job_id, state, etag):
        return await write_settlement(job_id, state, etag, **self._kw)
