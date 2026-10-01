"""What a corpus job's public status is computed from, kept in memory.

The auditor reports every verdict it writes or finds, the route every
submission it accepts; one listing seeds the verdicts written before this
process started. Nothing here names a hotkey.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

STATUS_CACHE_SECONDS = 30.0
ACCEPTED_WINDOW_SECONDS = 3600.0
# Verdict reads in flight at once while seeding, as the auditor reads records.
SEED_CONCURRENCY = 16


class JobStats:
    """Verdict counts and recent acceptances of one job.

    ``accepted_last_hour`` counts from this process's start: acceptances
    before a restart are not re-read, so it can only undercount for an hour.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._judged: set[str] = set()
        self.passed = 0
        self.verified_tokens = 0
        self._accepted: collections.deque[float] = collections.deque()
        self.seeded = False

    @property
    def judged(self) -> int:
        return len(self._judged)

    def observe(self, submission_id: str, verdict: dict | None) -> None:
        if verdict is None or submission_id in self._judged:
            return
        self._judged.add(submission_id)
        if verdict.get("passed"):
            self.passed += 1
            self.verified_tokens += int(verdict.get("token_count") or 0)

    def accepted(self) -> None:
        self._accepted.append(self._clock())

    def accepted_last_hour(self) -> int:
        horizon = self._clock() - ACCEPTED_WINDOW_SECONDS
        while self._accepted and self._accepted[0] < horizon:
            self._accepted.popleft()
        return len(self._accepted)

    async def seed(self, records: Any, job_id: str, *,
                   concurrency: int = SEED_CONCURRENCY) -> None:
        """Read once the verdicts this process has not seen, a bounded number at a time."""
        ids = [sid for sid in await records.list_verdict_ids(job_id) if sid not in self._judged]
        gate = asyncio.Semaphore(concurrency)

        async def one(sid: str) -> None:
            async with gate:
                try:
                    self.observe(sid, await records.read_verdict(job_id, sid))
                except Exception:
                    logger.warning("corpus status seed: verdict %s of %s unreadable",
                                   sid[:12], job_id)

        await asyncio.gather(*(one(sid) for sid in ids))
        self.seeded = True
        logger.info("corpus status of %s seeded from %d stored verdicts", job_id, len(ids))


def job_status(*, job_id: str, job: Any, slots: Any, stats: JobStats, settled: int | None,
               retired: bool, drained: bool = False) -> dict[str, Any]:
    """The public status document: counts only, never a hotkey."""
    full = sum(1 for taken in slots.snapshot().values() if taken >= job.slots_per_prompt)
    if drained:
        state = "drained"
    elif retired:
        state = "retired"
    elif full >= job.prompt_count:
        state = "full"
    else:
        state = "open"
    return {
        "job_id": job_id,
        "state": state,
        "prompts_total": int(job.prompt_count),
        "prompts_full": full,
        "submissions_accepted": int(slots.filled),
        "audited": stats.judged,
        "passed": stats.passed,
        "verified_tokens": stats.verified_tokens,
        "settled": int(settled or 0),
        "accepted_last_hour": stats.accepted_last_hour(),
    }


async def stored_job_counts(records: Any, job_id: str) -> dict[str, Any]:
    """What the bucket says of a job's drain, by listing it: the counts
    `jobs status` prints and the admin service proxies."""
    submissions = set(await records.list_submission_ids(job_id))
    verdicts = set(await records.list_verdict_ids(job_id))
    state, _ = await records.read_settlement(job_id)
    state = state or {}
    settled = verdicts & set(state.get("settled") or ())
    pending = state.get("pending")
    unaudited = len(submissions - verdicts)
    unsettled = len(verdicts - settled)
    return {
        "submissions": len(submissions), "verdicts": len(verdicts), "unaudited": unaudited,
        "settled": len(settled), "unsettled": unsettled,
        "pending_window": pending["window"] if pending else None,
        "last_window": state.get("last_window"),
        "drained": unaudited == 0 and unsettled == 0 and pending is None,
    }


__all__ = [
    "ACCEPTED_WINDOW_SECONDS",
    "JobStats",
    "SEED_CONCURRENCY",
    "STATUS_CACHE_SECONDS",
    "job_status",
    "stored_job_counts",
]
