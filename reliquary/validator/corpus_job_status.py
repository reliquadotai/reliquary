"""What a corpus job's public status is computed from, kept in memory.

Settled verdicts are counted by the settler into its own settlement object
(``totals``); verdicts written since and not yet settled are held here, as the
auditor reports them, and dropped once settled. Nothing is read to build it
beyond what the settler and the ledger reads already do. Nothing names a hotkey.
"""

from __future__ import annotations

import collections
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

STATUS_CACHE_SECONDS = 30.0
ACCEPTED_WINDOW_SECONDS = 3600.0


class JobStats:
    """Unsettled verdicts and recent acceptances of one job.

    After a restart, verdicts written before it count once settled (the
    settler's totals), not before; ``accepted_last_hour`` counts from the start.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        # Unsettled verdicts: id -> (passed, token_count).
        self._unsettled: dict[str, tuple[bool, int]] = {}
        self._accepted: collections.deque[float] = collections.deque()

    def observe(self, submission_id: str, verdict: dict | None) -> None:
        if verdict is None:
            return
        passed = bool(verdict.get("passed"))
        self._unsettled[submission_id] = (passed, int(verdict.get("token_count") or 0))

    def settled(self, submission_ids: Iterable[str]) -> None:
        for sid in submission_ids:
            self._unsettled.pop(sid, None)

    def unsettled(self) -> tuple[int, int, int]:
        """(verdicts, passed, verified tokens) not yet settled."""
        passed = [tokens for ok, tokens in self._unsettled.values() if ok]
        return len(self._unsettled), len(passed), sum(passed)

    def accepted(self) -> None:
        self._accepted.append(self._clock())

    def accepted_last_hour(self) -> int:
        horizon = self._clock() - ACCEPTED_WINDOW_SECONDS
        while self._accepted and self._accepted[0] < horizon:
            self._accepted.popleft()
        return len(self._accepted)


def job_status(*, job_id: str, job: Any, slots: Any, stats: JobStats, settled: int | None,
               totals: dict | None, retired: bool, drained: bool = False) -> dict[str, Any]:
    """The public status document: counts only, never a hotkey.

    ``counts_complete`` is False until the settler has read its totals, and
    for a job settled before totals were kept.
    """
    full = sum(1 for index in slots.snapshot() if slots.remaining(index) <= 0)
    if drained:
        state = "drained"
    elif retired:
        state = "retired"
    elif full >= job.prompt_count:
        state = "full"
    else:
        state = "open"
    verdicts, passed, tokens = stats.unsettled()
    base = totals or {}
    return {
        "job_id": job_id,
        "state": state,
        "prompts_total": int(job.prompt_count),
        "prompts_full": full,
        "submissions_accepted": int(slots.filled),
        "audited": int(base.get("verdicts", 0)) + verdicts,
        "passed": int(base.get("passed", 0)) + passed,
        "verified_tokens": int(base.get("verified_tokens", 0)) + tokens,
        "settled": int(settled or 0),
        "accepted_last_hour": stats.accepted_last_hour(),
        "counts_complete": bool(totals is not None and totals.get("complete", True)),
        **_eval_prompt_counts(job, slots),
    }


def _eval_prompt_counts(job: Any, slots: Any) -> dict[str, Any]:
    """An eval job's completeness: ``prompts_complete`` (V slots of work not
    failed), ``prompts_exhausted`` (every attempt used) and ``complete`` (each
    prompt one or the other). Absent for every other job."""
    from reliquary.eval.prompt_source import is_eval_source

    if not is_eval_source(getattr(job, "prompt_source", "")):
        return {}
    counts = slots.prompt_counts()
    return {"prompts_complete": counts["complete"], "prompts_exhausted": counts["exhausted"],
            "complete": counts["open"] == 0}


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
    "STATUS_CACHE_SECONDS",
    "job_status",
    "stored_job_counts",
]
