"""Settle a period-settled corpus task (``settlement: period-ema-v1``).

Pays each closed period of work its own cap, split by the verified tokens of
submissions received in it, in one archive per period that enters the weights
the period after it is written. No RL window is read: the clock is drand time.

Settlement stays two-phase (the pending archive is recorded in the settlement
state before it is written), so a crash can delay a payment, never repeat it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Mapping

from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_settlement import (
    VERDICT_READ_CONCURRENCY,
    _union,
    rewards_for,
)

logger = logging.getLogger(__name__)

PERIOD_SETTLEMENT_SCHEMA = "reliquary/corpus-period-settlement/v1"
PERIOD_ARCHIVE_SCHEMA = "reliquary/corpus-period/v1"
# A submission received at a period's very end reaches the store within the
# auditor's accept slack (corpus_auditor: 420 s).
ADMISSION_SLACK_SECONDS = 420.0
# Periods an undecided submission may hold the close before it is reported.
STUCK_PERIODS = 6


def _drand_genesis() -> float:
    return cp.PERIOD_EPOCH


class CorpusPeriodSettler:
    """Same surface as ``CorpusSettler`` (observe, settle_once, set_cap,
    settled_count, totals, on_settled, on_window) on the period clock.

    ``oldest_pending`` is the auditor's ``oldest_pending_received_at``: it
    returns when the oldest undecided submission was received (None if none) and
    raises ``LookupError`` while it cannot know."""

    def __init__(self, *, task_id, job_id, cap, records, archives,
                 oldest_pending: Callable[[], float | None],
                 genesis: Callable[[], float] = _drand_genesis, clock=time.time,
                 slack_seconds: float = ADMISSION_SLACK_SECONDS, on_settled=None,
                 full_list_every_seconds: float | None = None, executor=None) -> None:
        self._task_id = task_id
        self._job_id = job_id
        self._cap = float(cap)
        self._records = records
        self._archives = archives
        self._oldest_pending = oldest_pending
        self._genesis = genesis
        self._clock = clock
        self._slack = float(slack_seconds)
        self._executor = executor
        self._full_list_every = full_list_every_seconds
        self._listed_at: float | None = None
        self._unsettled: set[str] = set()
        self._fed: dict[str, Mapping] = {}
        self.settled_count: int | None = None
        self.totals: dict | None = None
        self.on_settled = on_settled
        self.on_window = None

    # -- the feed, as CorpusSettler ------------------------------------------

    def observe(self, submission_id: str, verdict=None) -> None:
        self._unsettled.add(submission_id)
        if isinstance(verdict, Mapping):
            self._fed[submission_id] = verdict

    def set_cap(self, cap: float) -> None:
        self._cap = float(cap)

    async def _off(self, func, *args):
        from reliquary.validator.corpus_judge_threads import run_in

        return await run_in(self._executor, func, *args)

    async def _verdict_ids(self, settled: set) -> list[str]:
        now = self._clock()
        due = (self._full_list_every is None or self._listed_at is None
               or not 0 <= now - self._listed_at < self._full_list_every)
        if due:
            self._unsettled.update(await self._records.list_verdict_ids(self._job_id))
            self._listed_at = now
        self._unsettled -= settled
        if self._fed:
            self._fed = {sid: v for sid, v in self._fed.items() if sid not in settled}
        return sorted(self._unsettled)

    async def _verdicts(self, ids: list[str]) -> dict[str, Mapping | None]:
        gate = asyncio.Semaphore(VERDICT_READ_CONCURRENCY)

        async def one(sid):
            async with gate:
                return sid, await self._records.read_verdict(self._job_id, sid)

        pairs = await asyncio.gather(*(one(sid) for sid in ids if sid not in self._fed),
                                     return_exceptions=True)
        for pair in pairs:
            if isinstance(pair, BaseException):
                raise pair
        read = dict(pairs)
        return {sid: self._fed[sid] if sid in self._fed else read[sid] for sid in ids}

    # -- settlement -----------------------------------------------------------

    def _work_period(self, verdict: Mapping, genesis: float) -> int:
        at = verdict.get("received_at")
        if at is None:
            # A verdict written before arrivals were recorded: its audit time.
            at = verdict.get("audited_at")
        return cp.period_of(float(at), genesis)

    def _report(self, state: dict, ids) -> None:
        self.settled_count = len(state["settled"] or ())
        self.totals = state.get("totals")
        if self.on_settled is not None and ids:
            self.on_settled(list(ids))

    @staticmethod
    def _add(totals: dict, verdicts) -> dict:
        passed = [v for v in verdicts if v and v.get("passed")]
        return {**totals, "verdicts": totals["verdicts"] + len(verdicts),
                "passed": totals["passed"] + len(passed),
                "verified_tokens": totals["verified_tokens"]
                + sum(int(v["token_count"]) for v in passed)}

    def _archive(self, pending: dict) -> dict:
        return {"schema": PERIOD_ARCHIVE_SCHEMA, "task_id": self._task_id,
                "job_id": self._job_id, "mechanism": "corpus-generation",
                "work_period": pending["work_period"], "entry_period": pending["entry_period"],
                "rewards_by_hotkey": dict(pending["rewards"]),
                "tokens": pending["tokens"], "verdicts": len(pending["ids"])}

    async def _finish(self, state: dict, etag) -> int:
        pending = state["pending"]
        written = await self._archives.read(self._task_id, pending["work_period"],
                                            pending["entry_period"])
        if written is None:
            # Not written yet. An entry period already under way, or past, would
            # be missed by the weight-sets that ran before the archive lands:
            # entered again, after them, so no share of it is ever skipped.
            due = cp.period_of(self._clock(), self._genesis()) + 1
            if pending["entry_period"] < due:
                pending = {**pending, "entry_period": due}
                state = {**state, "pending": pending, "last_entry": due}
                etag = await self._records.write_settlement(self._job_id, state, etag)
            await self._archives.write(self._task_id, pending["work_period"],
                                       pending["entry_period"], self._archive(pending))
        period_tokens = {**(state.get("period_tokens") or {}),
                         str(pending["work_period"]): pending["period_tokens"]}
        final = {**state, "settled": await self._off(_union, state.get("settled"), pending["ids"]),
                 "pending": None, "totals": pending["totals"], "period_tokens": period_tokens}
        await self._records.write_settlement(self._job_id, final, etag)
        self._report(final, pending["ids"])
        if self.on_window is not None:
            try:
                self.on_window(pending["work_period"], pending["rewards"])
            except Exception:
                logger.exception("corpus period report for %s failed", self._task_id)
        return pending["work_period"]

    async def settle_once(self) -> int | None:
        """Settle every closed period with unsettled verdicts, oldest first; the
        last work period archived, or None."""
        state, etag = await self._records.read_settlement(self._job_id)
        if state and (state.get("last_window") is not None
                      or "window" in (state.get("pending") or {})):
            # Settled by RL window until now (declared before this binary, or its
            # settlement changed): moving it to periods mid-job is not exact.
            logger.error("corpus task %s: job %s has a window settlement state; it is "
                         "not settled by period", self._task_id, self._job_id)
            return None
        state = {"schema": PERIOD_SETTLEMENT_SCHEMA, "settled": [], "pending": None,
                 "period_tokens": {}, "last_entry": None, **(state or {})}
        if state.get("totals") is None:
            state["totals"] = {"verdicts": 0, "passed": 0, "verified_tokens": 0}
        self._report(state, ())
        if state["pending"]:
            return await self._finish(state, etag)

        settled = await self._off(set, state["settled"])
        new_ids = await self._verdict_ids(settled)
        if not new_ids:
            return None
        try:
            oldest = self._oldest_pending()
        except LookupError:
            return None  # the auditor cannot tell yet what is still undecided
        genesis = self._genesis()
        now = self._clock()
        closed = cp.closed_through(now=now, oldest_pending=oldest, genesis=genesis,
                                   slack=self._slack)
        if oldest is not None and cp.period_of(now, genesis) - cp.period_of(
                max(oldest, now - 10**9), genesis) > STUCK_PERIODS:
            logger.warning("corpus task %s: an undecided submission from %.1f h ago holds every "
                           "later period's pay", self._task_id, (now - oldest) / 3600)
        verdicts = await self._verdicts(new_ids)
        lister = getattr(self._records, "list_voided_ids", None)
        voided = set(await lister(self._job_id)) if lister is not None else set()
        by_period: dict[int, list[str]] = defaultdict(list)
        for sid in new_ids:
            verdict = verdicts.get(sid)
            if verdict is None:
                continue  # listed before it was readable: next call
            period = self._work_period(verdict, genesis)
            if period <= closed:
                by_period[period].append(sid)
        last = None
        for period in sorted(by_period):
            ids = by_period[period]
            batch = [verdicts[sid] if sid not in voided else {**verdicts[sid], "passed": False}
                     for sid in ids]
            new_tokens = sum(int(v["token_count"]) for v in batch if v.get("passed"))
            earlier = int((state.get("period_tokens") or {}).get(str(period), 0))
            totals = self._add(state["totals"], batch)
            if new_tokens <= 0:
                # Nothing payable (spec: that period's cap burns); never reconsidered.
                state = {**state, "settled": await self._off(_union, state["settled"], ids),
                         "totals": totals}
                etag = await self._records.write_settlement(self._job_id, state, etag)
                self._report(state, ids)
                continue
            if earlier:
                # A verdict for a period already paid: its close missed it (it
                # should not). Paid against the period's whole token count, so the
                # period pays slightly over its cap; loud, never silent.
                logger.error("corpus task %s: %d late verdict(s) for settled period %d",
                             self._task_id, len(ids), period)
            share = rewards_for(batch, self._cap * new_tokens / (earlier + new_tokens))
            # One entry period per archive, strictly increasing: an entry never
            # carries more than one period's cap, so a catch-up after a backlog is
            # paid in full, later, instead of clamped at the task's cap.
            last_entry = state.get("last_entry")
            entry = cp.period_of(now, genesis) + 1
            if last_entry is not None:
                entry = max(entry, int(last_entry) + 1)
            state = {**state, "last_entry": entry, "pending": {
                "work_period": period, "entry_period": entry,
                "ids": ids, "rewards": share, "tokens": new_tokens, "totals": totals,
                "period_tokens": earlier + new_tokens}}
            etag = await self._records.write_settlement(self._job_id, state, etag)
            last = await self._finish(state, etag)
            state, etag = await self._records.read_settlement(self._job_id)
        return last


__all__ = [
    "ADMISSION_SLACK_SECONDS",
    "CorpusPeriodSettler",
    "PERIOD_ARCHIVE_SCHEMA",
    "PERIOD_SETTLEMENT_SCHEMA",
]
