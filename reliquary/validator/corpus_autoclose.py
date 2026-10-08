"""A finished corpus job closes its task's pay by itself.

A job is finished once it is full (every prompt's slots taken) and drained
(every submission has a verdict, every verdict is settled, no archive pending,
nothing still graded or replayed, nothing still being admitted). The corpus
validator then sets its task's cap to 0 in the registry, the write
``reliquary tasks set-cap`` makes (compare-and-swap), once:

- the cap governs only periods still to be worked (``corpus_periods.
  pay_ceiling``): the tail the job earned keeps being paid, in full, by the
  weight setter, up to the cap each period was settled under;
- a 0 cap frees the task's share of the pool (``total_cap``);
- a cap already 0 is left alone, so a restart writes nothing again;
- a paused job (admission paused by an operator) is never closed.

The task is not retired: a retired task named in a corpus validator's
RELIQUARY_TASK_ID still stops that validator from starting, and a retirement
frees nothing a 0 cap has not already freed. ``reliquary tasks close`` retires
it when the operator takes it out of the configuration.

On by default; ``RELIQUARY_CORPUS_AUTOCLOSE=0`` turns it off.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

AUTOCLOSE_ENV = "RELIQUARY_CORPUS_AUTOCLOSE"
# How often finished jobs are looked for: a drain check lists the job's
# submissions and verdicts, so not at every registry refresh.
AUTOCLOSE_EVERY_SECONDS = 600.0


def autoclose_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(AUTOCLOSE_ENV, "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


class AlreadyClosed(RuntimeError):
    """The registry no longer holds the entry this close was decided on."""


def close_guard(task_id: str, job_id: str):
    """The registry write's guard (``set_task_cap(guard=...)``), re-applied
    against the winner of a lost race: the entry must still be this job's,
    active, and paying."""

    def guard(before, updated) -> None:
        entry = before.get(task_id)
        if (entry is None or entry.status != "active" or str(entry.job_id) != job_id
                or float(entry.params.get("cap", 0.0)) <= 0.0):
            raise AlreadyClosed(f"task {task_id} is no longer job {job_id}'s paying entry")

    return guard


async def write_cap_zero(task_id: str, job_id: str) -> None:
    from reliquary.infrastructure.task_registry_store import set_task_cap

    await set_task_cap(task_id, 0.0, guard=close_guard(task_id, job_id))


class AutoClose:
    """Looks for finished jobs among those a ``CorpusJobSet`` serves, at most
    every ``every_seconds``, and sets each one's task cap to 0, once."""

    def __init__(self, *, read_entries: Callable[[], Awaitable[Mapping[str, Any]]],
                 write_cap_zero: Callable[[str, str], Awaitable[None]] = write_cap_zero,
                 every_seconds: float = AUTOCLOSE_EVERY_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._read_entries = read_entries
        self._write = write_cap_zero
        self._every = float(every_seconds)
        self._clock = clock
        self._ran_at: float | None = None
        # Jobs this process closed: never written again, whatever it reads.
        self.closed: set[str] = set()

    async def maybe_run(self, job_set) -> list[str]:
        now = self._clock()
        if self._ran_at is not None and 0 <= now - self._ran_at < self._every:
            return []
        self._ran_at = now
        return await self.run_once(job_set)

    async def run_once(self, job_set) -> list[str]:
        """The task ids closed by this pass."""
        from reliquary.shared.task_registry import RegistryError
        from reliquary.validator.corpus_periods import is_period_task

        entries = await self._read_entries()
        closed = []
        for job_id, wiring in list(job_set.served.items()):
            task_id = str(wiring.entry.task_id)
            entry = entries.get(task_id)
            if (job_id in self.closed or entry is None or entry.status != "active"
                    or str(entry.job_id) != job_id or not is_period_task(entry)
                    or float(entry.params.get("cap", 0.0)) <= 0.0
                    or getattr(entry, "admission", "open") != "open"):
                continue
            try:
                status = await job_set.job_finished(job_id)
            except Exception:
                logger.exception("corpus job %s: finished check failed; retrying later", job_id)
                continue
            if status is None:
                continue
            cap = float(entry.params["cap"])
            try:
                await self._write(task_id, job_id)
            except (AlreadyClosed, RegistryError) as exc:
                logger.info("corpus task %s: not closed automatically: %s", task_id, exc)
                self.closed.add(job_id)
                continue
            except Exception:
                logger.exception("corpus task %s: closing it (cap 0) failed; retrying later",
                                 task_id)
                continue
            self.closed.add(job_id)
            closed.append(task_id)
            logger.info(
                "corpus task %s closed automatically: job %s is full (%s/%s prompts) and "
                "drained (%s submissions, %s audited, %s passed, %s verified tokens, %s "
                "settled); cap %s -> 0, its earned tail keeps being paid",
                task_id, job_id, status.get("prompts_full"), status.get("prompts_total"),
                status.get("submissions_accepted"), status.get("audited"), status.get("passed"),
                status.get("verified_tokens"), status.get("settled"), cap)
        return closed


__all__ = [
    "AUTOCLOSE_ENV",
    "AUTOCLOSE_EVERY_SECONDS",
    "AlreadyClosed",
    "AutoClose",
    "autoclose_enabled",
    "close_guard",
    "write_cap_zero",
]
