"""The registry object, and the compare-and-swap that makes its sum a rule.

``trainer/publisher.py`` already writes R2 conditionally in production; the
difference here is that losing the race is expected, and the loser must
recompute the sum against the winner before it retries.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from reliquary.infrastructure.storage import get_s3_client
from reliquary.shared.task_registry import (
    TaskEntry,
    add_task,
    parse_registry,
    render_registry,
    require_default_declared_first,
    retire_task,
    validate_registry,
)

logger = logging.getLogger(__name__)

REGISTRY_KEY = "reliquary/tasks/registry.json"

_ABSENT_CODES = {"NoSuchKey", "404", "NotFound"}
_CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class RegistryConflict(RuntimeError):
    """Too many writers kept winning the race ahead of us."""


def _error_code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


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
        return parse_registry(body, strict=strict), response.get("ETag")


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


def _create(entries: Mapping[str, TaskEntry], entry: TaskEntry):
    # Checked inside the change function, not before the read: `_mutate`
    # re-applies it against the winner of a lost race, so two operators
    # racing cannot slip a non-`default` first entry past each other.
    require_default_declared_first(entries, entry)
    return add_task(entries, entry)


async def create_task(entry: TaskEntry, *, attempts: int = 5, **client_kwargs) -> None:
    await _mutate(lambda e: _create(e, entry), attempts=attempts, **client_kwargs)


async def retire_task_entry(
    task_id: str, retired_at: int, *, attempts: int = 5, **client_kwargs
) -> None:
    await _mutate(
        lambda e: retire_task(e, task_id, retired_at),
        attempts=attempts,
        **client_kwargs,
    )
