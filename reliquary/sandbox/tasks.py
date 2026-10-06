"""What a machine will resolve for a session: the task's image and declared limits,
read from the same env entry point (`reliquary_swe.sandbox.sandbox_task`) so the
token names the image the machine runs and budgets the gateway accepts (422 below the
task's limits)."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field

_LIMIT_FIELDS = ("memory_bytes", "disk_bytes", "pids", "wall_s", "per_call_timeout_s", "max_calls")


@dataclass(frozen=True)
class ResolvedTask:
    image: str
    limits: dict[str, int] = field(default_factory=dict)


class SweTaskResolver:
    def __init__(self, split: str, sandbox_task=None) -> None:
        self._split = split
        self._sandbox_task = sandbox_task
        self._cached = functools.lru_cache(maxsize=8192)(self._resolve)

    def _resolve(self, index: int) -> ResolvedTask:
        factory = self._sandbox_task
        if factory is None:
            from reliquary_swe.sandbox import sandbox_task as factory
        task = factory(self._split, index)
        limits = getattr(task, "limits", None)
        values = {} if limits is None else {
            name: int(getattr(limits, name)) for name in _LIMIT_FIELDS
            if getattr(limits, name, None) is not None}
        return ResolvedTask(task.image, values)

    async def resolve(self, index: int) -> ResolvedTask:
        return await asyncio.to_thread(self._cached, int(index))
