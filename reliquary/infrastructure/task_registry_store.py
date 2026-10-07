"""The registry object, and the compare-and-swap that makes its sum a rule.

``trainer/publisher.py`` already writes R2 conditionally in production; the
difference here is that losing the race is expected, and the loser must
recompute the sum against the winner before it retries.
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import replace
from collections.abc import Mapping

from reliquary.infrastructure.storage import get_s3_client
from reliquary.shared.task_registry import (
    TaskEntry,
    MECHANISM_CORPUS_GENERATION,
    RegistryError,
    add_task,
    parse_registry,
    render_registry,
    require_default_declared_first,
    retire_task,
    set_cap,
    set_admission,
    validate_registry,
)

logger = logging.getLogger(__name__)

REGISTRY_KEY = "reliquary/tasks/registry.json"
ADMISSION_KEY = "reliquary/tasks/admission.json"
ADMISSION_SCHEMA = "reliquary/task-admission/v1"
ADMISSION_MAX_BYTES = 1 << 20
ADMISSION_MAX_TASKS = 4096

_ABSENT_CODES = {"NoSuchKey", "404", "NotFound"}
_CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class RegistryConflict(RuntimeError):
    """Too many writers kept winning the race ahead of us."""


def _error_code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


async def _read_admissions(**client_kwargs) -> tuple[dict, str | None]:
    from botocore.exceptions import ClientError
    from reliquary.shared.task_id import normalise_task_id

    bucket = client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")
    async with get_s3_client(**client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=ADMISSION_KEY)
        except ClientError as exc:
            if _error_code(exc) in _ABSENT_CODES:
                return {}, None
            raise
        raw = await response["Body"].read(ADMISSION_MAX_BYTES + 1)
    if len(raw) > ADMISSION_MAX_BYTES:
        raise RegistryError("task admission object exceeds its bound")
    document = json.loads(raw)
    if (not isinstance(document, dict) or set(document) != {"schema", "tasks"}
            or document["schema"] != ADMISSION_SCHEMA or not isinstance(document["tasks"], dict)
            or len(document["tasks"]) > ADMISSION_MAX_TASKS):
        raise RegistryError("task admission object has an unsupported shape")
    for task_id, value in document["tasks"].items():
        if (not isinstance(task_id, str) or normalise_task_id(task_id) != task_id
                or not isinstance(value, dict) or set(value) != {"job_id", "profile_sha256", "admission"}
                or not isinstance(value["job_id"], str) or not value["job_id"]
                or not isinstance(value["profile_sha256"], str) or len(value["profile_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in value["profile_sha256"])
                or not isinstance(value["admission"], str) or value["admission"] not in {"open", "paused"}):
            raise RegistryError("task admission object has an invalid task binding")
    return document["tasks"], response.get("ETag")


def _admission_binding(entry: TaskEntry, admission: str) -> dict:
    return {"job_id": entry.job_id, "profile_sha256": entry.profile_sha256, "admission": admission}


def _overlay_admissions(entries: Mapping[str, TaskEntry], tasks: dict) -> dict[str, TaskEntry]:
    result = dict(entries)
    for task_id, entry in entries.items():
        value = tasks.get(task_id)
        if (entry.mechanism == MECHANISM_CORPUS_GENERATION and value is not None
                and value == _admission_binding(entry, value["admission"])):
            result[task_id] = replace(entry, admission=value["admission"])
    return result


async def _write_admissions(tasks: dict, etag: str | None, **client_kwargs) -> None:
    if len(tasks) > ADMISSION_MAX_TASKS:
        raise RegistryError("task admission object exceeds its task bound")
    body = json.dumps({"schema": ADMISSION_SCHEMA, "tasks": tasks}, sort_keys=True,
                      separators=(",", ":")).encode()
    if len(body) > ADMISSION_MAX_BYTES:
        raise RegistryError("task admission object exceeds its byte bound")
    bucket = client_kwargs.pop("bucket_name", None) or os.getenv("R2_BUCKET_ID", "reliquary")
    async with get_s3_client(**client_kwargs) as client:
        await client.put_object(Bucket=bucket, Key=ADMISSION_KEY, Body=body,
                                **({"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}))


async def read_registry(
    *, strict: bool = True, **client_kwargs
) -> tuple[dict[str, TaskEntry], str | None]:
    """The registry and the ETag to write it back against. Absent reads empty.

    ``strict=False`` skips the sum-of-caps invariant so an oversubscribed
    registry can still be read back (e.g. by ``tasks list``, whose whole job
    is letting an operator see a broken registry in order to repair it).
    Every other caller keeps the strict default.
    """
    from botocore.exceptions import ClientError

    bucket = client_kwargs.pop("bucket_name", None) or os.getenv(
        "R2_BUCKET_ID", "reliquary"
    )
    async with get_s3_client(**client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=REGISTRY_KEY)
        except ClientError as exc:
            if _error_code(exc) in _ABSENT_CODES:
                return {}, None
            raise
        body = await response["Body"].read()
        entries = parse_registry(body, strict=strict)
        etag = response.get("ETag")
    if any(entry.mechanism == MECHANISM_CORPUS_GENERATION for entry in entries.values()):
        tasks, _ = await _read_admissions(bucket_name=bucket, **client_kwargs)
        entries = _overlay_admissions(entries, tasks)
    return entries, etag


async def write_registry(
    entries: Mapping[str, TaskEntry], etag: str | None, **client_kwargs
) -> str | None:
    """Conditional put. Raises ClientError with a conflict code if we lost."""
    validate_registry(entries)
    bucket = client_kwargs.pop("bucket_name", None) or os.getenv(
        "R2_BUCKET_ID", "reliquary"
    )
    condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
    async with get_s3_client(**client_kwargs) as client:
        response = await client.put_object(
            Bucket=bucket,
            Key=REGISTRY_KEY,
            Body=render_registry(entries),
            **condition,
        )
    return response.get("ETag")


async def _mutate(change, *, attempts: int, **client_kwargs) -> None:
    """Read, apply, write conditionally; on a lost race read again and REAPPLY.

    Re-applying is what enforces the invariant: the change runs against the
    winner's registry, so a task that no longer fits is refused rather than
    written over someone else's budget.
    """
    from botocore.exceptions import ClientError

    for attempt in range(1, attempts + 1):
        entries, etag = await read_registry(**client_kwargs)
        updated = change(entries)
        try:
            await write_registry(updated, etag, **client_kwargs)
            return
        except ClientError as exc:
            if _error_code(exc) not in _CONFLICT_CODES:
                raise
            logger.info(
                "task registry changed under us (attempt %d/%d); re-reading",
                attempt, attempts,
            )
    raise RegistryConflict(
        f"task registry kept changing under us after {attempts} attempts"
    )


def _create(entries: Mapping[str, TaskEntry], entry: TaskEntry, guard=None):
    # Checked inside the change function, not before the read: `_mutate`
    # re-applies it against the winner of a lost race, so two operators
    # racing cannot slip a non-`default` first entry past each other.
    require_default_declared_first(entries, entry)
    return _guarded(entries, add_task(entries, entry), guard)


def _guarded(before, updated, guard):
    """``guard(before, updated)`` sees the registry read and the one a change
    would write, and raises to refuse it; re-applied on every retry."""
    if guard is not None:
        guard(before, updated)
    return updated


async def create_task(entry: TaskEntry, *, attempts: int = 5, guard=None,
                      **client_kwargs) -> None:
    await _mutate(lambda e: _create(e, entry, guard), attempts=attempts, **client_kwargs)


async def retire_task_entry(
    task_id: str, retired_at: int, *, attempts: int = 5, **client_kwargs
) -> None:
    await _mutate(
        lambda e: retire_task(e, task_id, retired_at),
        attempts=attempts,
        **client_kwargs,
    )


async def set_task_admission(task_id: str, admission: str, *, attempts: int = 5,
                             **client_kwargs) -> None:
    """Admission has its own CAS object; legacy economic writers cannot drop it."""
    from botocore.exceptions import ClientError

    for _ in range(attempts):
        entries, _ = await read_registry(**client_kwargs)
        entry = set_admission(entries, task_id, admission)[task_id]
        tasks, etag = await _read_admissions(**client_kwargs)
        desired = _admission_binding(entry, admission)
        if tasks.get(task_id) != desired:
            try:
                await _write_admissions({**tasks, task_id: desired}, etag, **client_kwargs)
            except ClientError as exc:
                if _error_code(exc) in _CONFLICT_CODES:
                    continue
                raise
        current, _ = await read_registry(**client_kwargs)
        observed = current.get(task_id)
        if observed is None or observed.status != "active":
            raise RegistryError("task is no longer active; admission cannot reopen")
        if _admission_binding(observed, admission) != desired:
            raise RegistryConflict("task binding changed during its admission update")
        if observed.admission != admission:
            continue
        return
    raise RegistryConflict("task admission kept changing under concurrent updates")


async def set_task_cap(
    task_id: str,
    cap: float,
    *,
    floor: float | None = None,
    min_incentive_share: float | None = None,
    audit_q: float | None = None,
    audit_probation_submissions: int | None = None,
    audit_hold_seconds: float | None = None,
    audit_suspect_seconds: float | None = None,
    audit_ban_after_failures: int | None = None,
    audit_ban_window_seconds: float | None = None,
    audit_ban_seconds: float | None = None,
    attempts: int = 5,
    guard=None,
    **client_kwargs,
) -> None:
    """Re-applied against the winner of a lost race, so the new cap is checked
    against the registry that is actually there, not the one first read."""
    await _mutate(
        lambda e: _guarded(e, set_cap(
            e, task_id, cap, floor, min_incentive_share, audit_q,
            audit_probation_submissions, audit_hold_seconds, audit_suspect_seconds,
            audit_ban_after_failures, audit_ban_window_seconds, audit_ban_seconds,
        ), guard),
        attempts=attempts,
        **client_kwargs,
    )
