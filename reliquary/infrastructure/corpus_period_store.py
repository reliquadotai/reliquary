"""Where period-settled corpus tasks keep their pay: one archive per settled
period of work, under a prefix no window listing reads.

``reliquary/corpus-periods/<task_id>/<work period>-<entry period>.json.gz``,
both periods zero-padded so a listing sorts them. The entry period is in the
name so the weight setter can skip what it will not replay without reading it.
"""

from __future__ import annotations

import logging
import os
import re

from reliquary.shared.task_id import normalise_task_id

logger = logging.getLogger(__name__)

PERIODS_PREFIX = "reliquary/corpus-periods/"
_NAME_RE = re.compile(r"(\d{10})-(\d{10})\.json\.gz$")


def task_prefix(task_id: str) -> str:
    return f"{PERIODS_PREFIX}{normalise_task_id(task_id)}/"


def period_archive_key(task_id: str, work_period: int, entry_period: int) -> str:
    if work_period < 0 or entry_period < work_period:
        raise ValueError(f"periods out of order: work {work_period}, entry {entry_period}")
    return f"{task_prefix(task_id)}{int(work_period):010d}-{int(entry_period):010d}.json.gz"


def parse_key(key: str) -> tuple[int, int] | None:
    match = _NAME_RE.search(key)
    return (int(match.group(1)), int(match.group(2))) if match else None


async def write_period_archive(task_id: str, work_period: int, entry_period: int,
                               document: dict) -> None:
    from reliquary.infrastructure import storage

    await storage.upload_json(period_archive_key(task_id, work_period, entry_period), document)


async def read_period_archive(task_id: str, work_period: int, entry_period: int) -> dict | None:
    from reliquary.infrastructure import storage

    return await storage.download_json(
        period_archive_key(task_id, work_period, entry_period), strict=True)


async def list_period_archives(task_id: str, **client_kwargs) -> list[tuple[int, int]]:
    """Every ``(work period, entry period)`` archived for the task, sorted."""
    from reliquary.infrastructure import storage

    prefix = task_prefix(task_id)
    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    found: list[tuple[int, int]] = []
    async with storage.get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                parsed = parse_key(obj["Key"])
                if parsed is not None:
                    found.append(parsed)
    return sorted(found)


class R2PeriodArchives:
    """The settler's and the weight setter's view of the bucket. ``guard``
    (the settler's window archives) refuses a task this process does not serve."""

    def __init__(self, guard=None) -> None:
        self._guard = guard

    async def write(self, task_id, work_period, entry_period, document) -> None:
        if self._guard is not None:
            self._guard.refuse_unserved(task_id)
        await write_period_archive(task_id, work_period, entry_period, document)

    async def list(self, task_id) -> list[tuple[int, int]]:
        return await list_period_archives(task_id)

    async def read(self, task_id, work_period, entry_period) -> dict | None:
        return await read_period_archive(task_id, work_period, entry_period)


__all__ = [
    "PERIODS_PREFIX",
    "R2PeriodArchives",
    "list_period_archives",
    "parse_key",
    "period_archive_key",
    "read_period_archive",
    "task_prefix",
    "write_period_archive",
]
