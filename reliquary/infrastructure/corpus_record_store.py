"""What a corpus job produced: accepted submissions, their audit verdicts, and
the settlement state. Beside the job store, under the same job prefix.

Submissions and verdicts are create-only, so a retry or a second auditor can
never overwrite one; the settlement state is compare-and-swap, because it is
what decides which verdicts have already been paid.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import json
import math
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
    read_job,
    read_ledgers,
)
from reliquary.infrastructure.storage import get_s3_client, off_loop

_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")
# An episode job's record: one trajectory under `completions`, assistant-span
# `token_count`, and the validator-derived `prompt_tokens` beside it.
RECORD_SCHEMA_V2 = "reliquary/corpus-submission-record/v2"
RECORD_SCHEMA_V1 = "reliquary/corpus-submission-record/v1"
# Above the largest record allowed by the current completion/trajectory wire
# bounds, including JSON escaping; unrelated objects must not be decoded here.
MAX_STAGED_RECORD_BYTES = 1024 * 1024 * 1024
_RECORD_FIELDS = frozenset({
    "schema", "submission_id", "job_id", "hotkey", "cursor", "prompt_index",
    "rendered_prompt", "received_at", "token_count", "completions",
})


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


def _record_identity(job_id, submission_id, record) -> None:
    """The immutable receipt's shape; intake still owns proof validation."""
    from reliquary.protocol.corpus_submission import (
        MAX_COMPLETIONS_PER_SUBMISSION,
        MAX_RENDERED_PROMPT_CHARS,
        CorpusCompletion,
        CorpusTrajectory,
    )

    if (not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS
            or record["schema"] not in (RECORD_SCHEMA_V1, RECORD_SCHEMA_V2)
            or record["job_id"] != _validated_job_id(job_id)
            or record["submission_id"] != _validated_id(submission_id)
            or not isinstance(record["hotkey"], str) or not record["hotkey"]
            or not isinstance(record["rendered_prompt"], str)
            or len(record["rendered_prompt"]) > MAX_RENDERED_PROMPT_CHARS
            or any(type(record[k]) is not int or record[k] < 0
                   for k in ("cursor", "prompt_index", "token_count"))
            or type(record["received_at"]) not in (int, float)
            or not math.isfinite(record["received_at"]) or record["received_at"] < 0
            or not isinstance(record["completions"], list)
            or not record["completions"]
            or len(record["completions"]) > MAX_COMPLETIONS_PER_SUBMISSION
            or any(not isinstance(c, Mapping) for c in record["completions"])):
        raise ValueError("Staged submission has an invalid native receipt identity or shape")
    if record["schema"] == RECORD_SCHEMA_V1:
        completions = [CorpusCompletion.model_validate(c, strict=True)
                       for c in record["completions"]]
        count = sum(len(c.tokens) for c in completions)
        if [c.model_dump() for c in completions] != record["completions"]:
            raise ValueError("Staged completion differs from its native wire shape")
    else:
        if len(record["completions"]) != 1:
            raise ValueError("Staged episode must contain one trajectory")
        raw = dict(record["completions"][0])
        prompt_tokens = raw.pop("prompt_tokens", None)
        if (not isinstance(prompt_tokens, list) or not prompt_tokens
                or any(type(t) is not int or not 0 <= t <= 2**32 - 1 for t in prompt_tokens)):
            raise ValueError("Staged episode has invalid validator prompt tokens")
        trajectory = CorpusTrajectory.model_validate(raw, strict=True)
        count = sum(t.end - t.start for t in trajectory.turns)
        native = trajectory.model_dump()
        if native.get("transcript") is None:
            # A replay record carries no transcript field at all (signed-sandbox
            # records alone do), so its bytes stay what they were before.
            native.pop("transcript", None)
        if native != raw:
            raise ValueError("Staged trajectory differs from its native wire shape")
    if count != record["token_count"]:
        raise ValueError("Staged submission token count differs from its native body")


def _record_ref(ref) -> None:
    if (not isinstance(ref, Mapping)
            or set(ref) != {"submission_id", "sha256", "received_at"}
            or type(ref["received_at"]) not in (int, float)
            or not math.isfinite(ref["received_at"]) or ref["received_at"] < 0):
        raise ValueError("Invalid staged submission reference")
    _validated_id(ref["submission_id"])
    _validated_id(ref["sha256"])


async def _record_bytes(key, **client_kwargs) -> bytes | None:
    body, _ = await _get(key, byte_range=f"bytes=0-{MAX_STAGED_RECORD_BYTES}",
                         **client_kwargs)
    if body is not None and len(body) > MAX_STAGED_RECORD_BYTES:
        raise ValueError("Staged submission exceeds the stored-record byte bound")
    return body


async def _create_verified(key, body, **client_kwargs) -> None:
    error = None
    try:
        await _put(key, body, None, **client_kwargs)
    except Exception as exc:
        # A lost PUT acknowledgment or a conditional-write conflict is not
        # proof of absence or success. Only the exact stored bytes decide.
        error = exc
    stored = await _record_bytes(key, **client_kwargs)
    if stored is None:
        if error is not None:
            raise error
        raise CorpusStoreConflict("Submission absent after create-only write")
    if stored != body:
        raise ValueError("Stored submission differs from its committed body")


async def stage_submission(job_id, submission_id, record, *, executor=None,
                           **client_kwargs) -> dict:
    """Keep validated work outside the auditor's listing, before admission CAS.

    The returned reference is not admission authority. Only its inclusion in
    the slot ledger makes it eligible for canonical publication.
    """
    await _off(executor, _record_identity, job_id, submission_id, record)
    body = await _off(executor, _encode, dict(record))
    if len(body) > MAX_STAGED_RECORD_BYTES:
        raise ValueError("Staged submission exceeds the stored-record byte bound")
    sha256 = (await _off(executor, hashlib.sha256, body)).hexdigest()
    # Staged bodies are retained; prune unreferenced ones if their storage
    # footprint warrants it, after proving no committed ledger names them.
    await _create_verified(_key(job_id, "staged-submissions", sha256), body, **client_kwargs)
    return {"submission_id": submission_id, "sha256": sha256,
            "received_at": record["received_at"]}


async def read_staged_submission(job_id, ref, *, executor=None, **client_kwargs) -> dict:
    _record_ref(ref)
    body = await _record_bytes(_key(job_id, "staged-submissions", ref["sha256"]),
                               **client_kwargs)
    if body is None or (await _off(executor, hashlib.sha256, body)).hexdigest() != ref["sha256"]:
        raise ValueError("Committed staged submission is absent or has the wrong hash")
    record = await _off(executor, _decode, body)
    await _off(executor, _record_identity, job_id, ref["submission_id"], record)
    if (record["received_at"] != ref["received_at"]
            or await _off(executor, _encode, record) != body):
        raise ValueError("Staged submission differs from its frozen receipt")
    return record


async def promote_submission(job_id, ref, *, executor=None, **client_kwargs) -> dict:
    """Publish a reference the caller has verified in the committed slot ledger.

    Existing canonical bytes must match exactly; an absent or different body
    never permits the caller to clear the durable reference or acknowledge it.
    """
    record = await read_staged_submission(job_id, ref, executor=executor, **client_kwargs)
    body = await _off(executor, _encode, record)
    await _create_verified(_key(job_id, "submissions", ref["submission_id"]),
                           body, **client_kwargs)
    return record


# A stored record is sorted JSON: its completions come first, then cursor,
# hotkey, job_id, prompt_index, received_at, rendered_prompt, schema,
# submission_id and token_count. Their last bytes carry what scheduling needs.
SUBMISSION_TAIL_BYTES = 8192
_META_KEYS = {
    "hotkey": re.compile(rb'"hotkey":"((?:[^"\\]|\\.)*)"'),
    "received_at": re.compile(rb'"received_at":(-?[0-9][0-9.eE+-]*|null)'),
    "token_count": re.compile(rb'"token_count":([0-9]+)'),
}


def submission_meta(tail: bytes) -> dict | None:
    """``hotkey``, ``received_at`` (None when absent) and ``token_count`` from
    the end of a stored record, or None when ``tail`` does not reach back to
    them. Inside a JSON string every quote is escaped, so a completion text or
    a prompt spelling ``"hotkey":"...`` never matches."""
    if b'"cursor":' not in tail:
        return None
    found = {}
    for name, pattern in _META_KEYS.items():
        matches = pattern.findall(tail)
        found[name] = matches[-1] if matches else None
    if found["hotkey"] is None or found["token_count"] is None:
        return None
    received = found["received_at"]
    return {"hotkey": json.loads(b'"' + found["hotkey"] + b'"'),
            "received_at": None if received in (None, b"null") else float(received),
            "token_count": int(found["token_count"])}


async def read_submission_meta(job_id, submission_id, *,
                               tail_bytes: int = SUBMISSION_TAIL_BYTES,
                               **client_kwargs) -> dict | None:
    """What scheduling a pending record needs, from its last bytes (one ranged
    GET); the whole record only when its prompt is longer than the tail."""
    key = _key(job_id, "submissions", submission_id)
    executor = client_kwargs.pop("executor", None)
    tail, _ = await _get(key, byte_range=f"bytes=-{int(tail_bytes)}", **client_kwargs)
    if tail is None:
        return None
    meta = submission_meta(tail)
    if meta is not None:
        return meta
    record = await _read(key, executor=executor, **client_kwargs)
    if record is None:
        return None
    return {"hotkey": record["hotkey"], "received_at": record.get("received_at"),
            "token_count": int(record["token_count"])}


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


async def read_voided(job_id, submission_id, **client_kwargs) -> dict | None:
    return await _read(_key(job_id, "voided", submission_id), **client_kwargs)


async def list_voided_ids(job_id, **client_kwargs) -> list[str]:
    return await _list_ids(_prefix(job_id, "voided"), **client_kwargs)


async def write_grade(job_id, submission_id, document, **client_kwargs) -> bool:
    return await _create(_key(job_id, "grades", submission_id), document, **client_kwargs)


async def read_grade(job_id, submission_id, **client_kwargs) -> dict | None:
    return await _read(_key(job_id, "grades", submission_id), **client_kwargs)


async def list_grade_ids(job_id, **client_kwargs) -> list[str]:
    return await _list_ids(_prefix(job_id, "grades"), **client_kwargs)


# Regrades of one submission a quarantine can force, at most (corpus_grading).
MAX_REGRADE_GENERATIONS = 3


def _regrade_key(job_id: str, submission_id: str, generation: int) -> str:
    if isinstance(generation, bool) or not isinstance(generation, int) \
            or not 1 <= generation <= MAX_REGRADE_GENERATIONS:
        raise ValueError(f"regrade generation must be in [1, {MAX_REGRADE_GENERATIONS}]")
    if generation == 1:
        return _key(job_id, "regrades", submission_id)
    return f"{_prefix(job_id, 'regrades')}{_validated_id(submission_id)}.g{generation}.json"


async def write_regrade(job_id, submission_id, document, generation: int = 1,
                        **client_kwargs) -> bool:
    """A grade redone after its only executor was quarantined; it supersedes the
    grade, and each later generation the one before (create-only, each)."""
    return await _create(_regrade_key(job_id, submission_id, generation), document,
                         **client_kwargs)


async def read_regrade(job_id, submission_id, **client_kwargs) -> dict | None:
    """The latest regrade of a submission, or None."""
    latest = None
    for generation in range(1, MAX_REGRADE_GENERATIONS + 1):
        document = await _read(_regrade_key(job_id, submission_id, generation), **client_kwargs)
        if document is None:
            break
        latest = document
    return latest


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

    async def read_job(self, job_id):
        return await read_job(job_id, **self._kw)

    async def read_ledgers(self, job_id):
        return await read_ledgers(job_id, **self._kw)

    async def write_submission(self, job_id, submission_id, record):
        return await write_submission(job_id, submission_id, record, **self._kw)

    async def stage_submission(self, job_id, submission_id, record):
        return await stage_submission(job_id, submission_id, record, **self._kw)

    async def read_staged_submission(self, job_id, ref):
        return await read_staged_submission(job_id, ref, **self._kw)

    async def promote_submission(self, job_id, ref):
        return await promote_submission(job_id, ref, **self._kw)

    async def read_submission(self, job_id, submission_id):
        if self._max_reads is None:
            return await read_submission(job_id, submission_id, **self._kw)
        if self._reads is None:
            self._reads = asyncio.Semaphore(self._max_reads)
        async with self._reads:
            return await read_submission(job_id, submission_id, **self._kw)

    async def read_submission_meta(self, job_id, submission_id):
        return await read_submission_meta(job_id, submission_id, **self._kw)

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

    async def read_voided(self, job_id, submission_id):
        return await read_voided(job_id, submission_id, **self._kw)

    async def list_voided_ids(self, job_id):
        return await list_voided_ids(job_id, **self._kw)

    async def write_grade(self, job_id, submission_id, document):
        return await write_grade(job_id, submission_id, document, **self._kw)

    async def read_grade(self, job_id, submission_id):
        return await read_grade(job_id, submission_id, **self._kw)

    async def list_grade_ids(self, job_id):
        return await list_grade_ids(job_id, **self._kw)

    async def write_regrade(self, job_id, submission_id, document, generation: int = 1):
        return await write_regrade(job_id, submission_id, document, generation, **self._kw)

    async def read_regrade(self, job_id, submission_id):
        return await read_regrade(job_id, submission_id, **self._kw)

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
