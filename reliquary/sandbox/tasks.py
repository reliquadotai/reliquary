"""What a machine will resolve for a session: the task's image and declared limits,
read from the same bridged env a gateway serves (`bridged_swe.sandbox_task_of`) so the
token names the image the machine runs and budgets the gateway accepts (422 below the
task's limits). The resolved task also carries the digest of the options the env was
built with, which a machine must publish for it to be placed there."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_LIMIT_FIELDS = ("memory_bytes", "disk_bytes", "pids", "wall_s", "per_call_timeout_s", "max_calls")

_HUGE_CPU_MILLIS = 1 << 62        # a task declaring infinite CPUs: no box is ever that big


def task_cpu_millis(cpus: Any) -> int | None:
    """A task's declared CPUs (``TaskLimits.cpus``) in thousandths of a CPU, ROUNDED UP: the one rule shared with the
    gateway's box-size check. The gateway refuses a task (422) when ``cpus x 10^9`` nano-CPUs exceed its box, and
    reports that box floored to millis (``max_box_cpu_millis``); a rounded-up figure above that floor is exactly
    such a task, so this validator never places one the gateway refuses (it may, for a box that is not a whole
    number of millis, decline a task that would just fit: the safe side). The reservation it implies is at most
    one millicpu above the gateway's (which floors), again the safe side. None: the machine's box default."""
    if cpus is None or isinstance(cpus, bool) or not isinstance(cpus, int | float):
        return None
    if math.isnan(cpus):
        return None
    if math.isinf(cpus):
        return _HUGE_CPU_MILLIS if cpus > 0 else 1
    nano = math.ceil(cpus * 1_000_000_000)                  # the gateway's own product, compared exactly
    return max(1, -(-nano // 1_000_000))


@dataclass(frozen=True)
class ResolvedTask:
    image: str
    limits: dict[str, int] = field(default_factory=dict)
    env_options_sha256: str | None = None
    """The digest of the env options the image and limits were computed with
    (`bridged_swe.options_sha256()`); None: no options check at placement."""
    prepare_timeout_s: float | None = None
    """The task's prepare timeout, as its gateway reads it: a session token must outlive it. None: the
    sandbox's default."""
    cpu_millis: int | None = None
    """The task's declared CPUs x 1000 (``TaskLimits.cpus``, ``task_cpu_millis``); None: the machine's box default.
    An RL session reserves it on its machine, and is never placed on a smaller box."""


class SweTaskResolver:
    """Resolves (and remembers, least recently used first out, at most `cache_size`)
    each index's task. The cache is the instance's own dict: no `lru_cache` on a bound
    method, which would hold the resolver alive from a shared cache."""

    def __init__(self, split: str, sandbox_task=None, *, cache_size: int = 8192,
                 options_sha256: str | None = None) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self._split = split
        self._sandbox_task = sandbox_task
        self._options_sha256 = options_sha256
        self._cache_size = cache_size
        self._cache: OrderedDict[int, ResolvedTask] = OrderedDict()
        self._lock = threading.Lock()

    def _resolve(self, index: int) -> ResolvedTask:
        factory, digest = self._sandbox_task, self._options_sha256
        if factory is None:
            from reliquary.environment.bridged_swe import options_sha256
            from reliquary.environment.bridged_swe import sandbox_task_of as factory

            digest = digest or options_sha256()
        task = factory(self._split, index)
        limits = getattr(task, "limits", None)
        values = {} if limits is None else {
            name: int(getattr(limits, name)) for name in _LIMIT_FIELDS
            if getattr(limits, name, None) is not None}
        prepare = getattr(task, "prepare_timeout_s", None)
        prepare = float(prepare) if type(prepare) in (int, float) and prepare > 0 else None
        cpu_millis = task_cpu_millis(getattr(limits, "cpus", None)) if limits is not None else None
        return ResolvedTask(task.image, values, digest, prepare, cpu_millis)

    def _cached(self, index: int) -> ResolvedTask:
        with self._lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
        task = self._resolve(index)
        with self._lock:
            self._cache[index] = task
            self._cache.move_to_end(index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return task

    async def resolve(self, index: int) -> ResolvedTask:
        return await asyncio.to_thread(self._cached, int(index))
