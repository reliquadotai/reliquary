"""One miner's view of a corpus job: audit state, verdict counts, recent
failures and pay, answered by ``GET /corpus/jobs/{job_id}/miners/{hotkey}``.

Everything is kept in memory as the auditor and the settler report it. What
was written before this process started is read once, in the background, on
the first request for the job (``backfill``); until then ``counts_complete``
is False. A request never lists the store. Nothing here names another hotkey.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MINER_STATUS_CACHE_SECONDS = 30.0
RECENT_FAILURES = 20
# Settled windows the pay share is measured over (about 6 h of RL windows).
SHARE_WINDOWS = 24
# Store reads in flight at once while backfilling: far below the auditor's.
BACKFILL_CONCURRENCY = 4
BACKFILL_RETRY_SECONDS = 300.0
# An SS58 address: base58, never longer than this.
_HOTKEY = re.compile(r"[1-9A-HJ-NP-Za-km-z]{1,64}")
_MEASURES = ("worst_exp", "worst_mant_mean", "worst_mant_median")


def valid_hotkey(hotkey: str) -> bool:
    return bool(_HOTKEY.fullmatch(hotkey))


def _key(submission_id: str):
    # 32 bytes instead of a 64-character string: these sets hold every verdict.
    try:
        return bytes.fromhex(submission_id) if len(submission_id) == 64 else submission_id
    except ValueError:
        return submission_id


def _guarded(method):
    # Called from the auditor and the settler after their writes: a bug here
    # must never reach them.
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            logger.exception("corpus miner book of %s: %s failed", self._job_id, method.__name__)
    return wrapper


@dataclass
class _Counts:
    verdicts: int = 0
    audited: int = 0
    passed: int = 0
    passed_unaudited: int = 0
    failed: int = 0
    voided: int = 0
    tokens_settled: int = 0
    # Newest first, at most RECENT_FAILURES.
    failures: list = field(default_factory=list)


class MinerBook:
    """Per hotkey, the verdicts of one job and what settling them paid."""

    def __init__(self, *, job_id: str, task_id: str, records,
                 read_windows: Callable[[str, int, int], Awaitable[list]] | None = None,
                 thresholds: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._job_id = job_id
        self._task_id = task_id
        self._records = records
        self._read_windows = read_windows
        self.thresholds = dict(thresholds) if thresholds is not None else None
        self._clock = clock
        self._counts: dict[str, _Counts] = {}
        self._seen: set = set()
        self._settled: set = set()
        self._voided: set = set()
        # Passing verdicts not settled yet: id -> (hotkey, tokens).
        self._unsettled: dict = {}
        # Window -> rewards by hotkey, the last SHARE_WINDOWS settled.
        self._windows: dict[int, dict[str, float]] = {}
        self.complete = False
        self._backfill: asyncio.Task | None = None
        self._backfill_failed_at: float | None = None

    def _of(self, hotkey: str) -> _Counts:
        counts = self._counts.get(hotkey)
        if counts is None:
            counts = self._counts[hotkey] = _Counts()
        return counts

    def _fail(self, counts: _Counts, submission_id: str, doc: Mapping, at) -> None:
        entry = {"submission_id": submission_id, "audited_at": at, "reason": doc.get("reason"),
                 "token_count": doc.get("token_count"),
                 **{k: doc.get(k) for k in _MEASURES}}
        failures = [entry, *counts.failures]
        failures.sort(key=lambda f: -float(f["audited_at"] or 0.0))
        counts.failures = failures[:RECENT_FAILURES]

    @_guarded
    def observe(self, submission_id: str, verdict: Mapping | None) -> None:
        """A verdict that stands (the auditor's ``on_verdict``)."""
        if verdict is None:
            return
        key = _key(submission_id)
        if key in self._seen:
            return
        self._seen.add(key)
        counts = self._of(str(verdict.get("hotkey")))
        counts.verdicts += 1
        passed = bool(verdict.get("passed"))
        if verdict.get("reason") == "banned" and not passed:
            counts.voided += 1
            return
        # A verdict from before sampling carries no "audited": every one was.
        audited = bool(verdict.get("audited", True))
        counts.audited += audited
        if key in self._voided:
            counts.voided += 1
            return
        if not passed:
            counts.failed += 1
            self._fail(counts, submission_id, verdict, verdict.get("audited_at"))
            return
        counts.passed += 1
        counts.passed_unaudited += not audited
        tokens = int(verdict.get("token_count") or 0)
        if key in self._settled:
            counts.tokens_settled += tokens
        else:
            self._unsettled[key] = (str(verdict.get("hotkey")), tokens)

    @_guarded
    def voided(self, submission_id: str, document: Mapping) -> None:
        """A pass withdrawn after its executor's quarantine (``on_voided``)."""
        key = _key(submission_id)
        if key in self._voided:
            return
        self._voided.add(key)
        if key not in self._seen:
            return  # counted as voided when its verdict is seen
        counts = self._of(str(document.get("hotkey")))
        # Unsettled, it is never paid; already settled, it was.
        self._unsettled.pop(key, None)
        counts.passed -= 1
        counts.voided += 1
        self._fail(counts, submission_id, document, document.get("voided_at"))

    @_guarded
    def settled(self, submission_ids: Iterable[str]) -> None:
        """Ids a settlement moved (the settler's ``on_settled``)."""
        for sid in submission_ids:
            key = _key(sid)
            self._settled.add(key)
            paid = self._unsettled.pop(key, None)
            if paid is not None:
                self._of(paid[0]).tokens_settled += paid[1]

    @_guarded
    def window(self, window: int, rewards: Mapping[str, float]) -> None:
        """A window this task paid (the settler's ``on_window``)."""
        self._windows[int(window)] = {str(k): float(v) for k, v in rewards.items()}
        for old in sorted(self._windows)[:-SHARE_WINDOWS]:
            del self._windows[old]

    def counts(self, hotkey: str) -> dict:
        counts = self._counts.get(hotkey) or _Counts()
        return {
            "verdicts": counts.verdicts, "audited": counts.audited, "passed": counts.passed,
            "passed_unaudited": counts.passed_unaudited, "failed": counts.failed,
            "voided": counts.voided, "recent_failures": [dict(f) for f in counts.failures],
            "verified_tokens_settled": counts.tokens_settled, "counts_complete": self.complete,
        }

    def share(self, hotkey: str) -> dict:
        windows = sorted(self._windows)
        total = sum(sum(r.values()) for r in self._windows.values())
        mine = sum(r.get(hotkey, 0.0) for r in self._windows.values())
        return {"windows": len(windows), "first_window": windows[0] if windows else None,
                "last_window": windows[-1] if windows else None, "reward": mine,
                "share": round(mine / total, 6) if total > 0 else 0.0}

    # -- what was written before this process -----------------------------

    def start_backfill(self) -> None:
        """Begin the backfill unless it ran, runs, or failed a moment ago."""
        if self.complete or (self._backfill is not None and not self._backfill.done()):
            return
        if (self._backfill_failed_at is not None
                and self._clock() - self._backfill_failed_at < BACKFILL_RETRY_SECONDS):
            return
        self._backfill = asyncio.ensure_future(self.backfill())

    async def wait_backfill(self) -> None:
        if self._backfill is not None:
            await asyncio.shield(self._backfill)

    def close(self) -> None:
        if self._backfill is not None:
            self._backfill.cancel()

    async def backfill(self) -> None:
        """Read once what this process did not see: the settled ids, the
        voided ids, every verdict not reported live, and the last windows paid.
        Never raises: a failure is retried on a later request."""
        try:
            state, _ = await self._records.read_settlement(self._job_id)
            state = state or {}
            self.settled(state.get("settled") or ())
            lister = getattr(self._records, "list_voided_ids", None)
            if lister is not None:
                for sid in await lister(self._job_id):
                    if _key(sid) not in self._seen:
                        self._voided.add(_key(sid))
            todo = [sid for sid in await self._records.list_verdict_ids(self._job_id)
                    if _key(sid) not in self._seen]
            queue = iter(todo)

            async def reader():
                # A few readers over one iterator: never a coroutine per verdict.
                for sid in queue:
                    self.observe(sid, await self._records.read_verdict(self._job_id, sid))

            await asyncio.gather(*(reader() for _ in range(BACKFILL_CONCURRENCY)))
            last = state.get("last_window")
            if self._read_windows is not None and last is not None:
                for archive in await self._read_windows(self._task_id, int(last), SHARE_WINDOWS):
                    self.window(archive["window_start"], archive.get("rewards_by_hotkey") or {})
            self.complete = True
            logger.info("corpus job %s: miner counts backfilled from %d verdict(s)",
                        self._job_id, len(todo))
        except Exception:
            self._backfill_failed_at = self._clock()
            logger.warning("corpus job %s: miner counts backfill failed; retrying later",
                           self._job_id, exc_info=True)


async def read_recent_windows(task_id: str, last_window: int, k: int) -> list[dict]:
    """The archives of ``task_id`` among the ``k`` windows up to ``last_window``."""
    from reliquary.infrastructure import storage

    return await storage.list_recent_datasets(last_window + 1, k, task_id=task_id)


def feed(stats, book: MinerBook):
    """The auditor's ``on_verdict`` and the settler's ``on_settled`` for a job's
    public status and its miner book: the status first, as before."""

    def on_verdict(submission_id: str, verdict) -> None:
        stats.observe(submission_id, verdict)
        book.observe(submission_id, verdict)

    def on_settled(submission_ids) -> None:
        ids = list(submission_ids)
        stats.settled(ids)
        book.settled(ids)

    return on_verdict, on_settled


def proof_thresholds(proof) -> dict | None:
    """The toploc thresholds a verdict's measures are judged against."""
    if proof is None:
        return None
    fields = {"chunk_tokens": "chunk_tokens", "topk": "topk",
              "exp_mismatch": "exp_mismatch_threshold", "mant_mean": "mant_mean_threshold",
              "mant_median": "mant_median_threshold",
              "min_allowed_failures": "min_allowed_failures",
              "ratio_allowed_failures": "ratio_allowed_failures"}
    return {name: getattr(proof, attr, None) for name, attr in fields.items()}


def miner_status(*, job_id: str, hotkey: str, book: MinerBook, pending: int, state, params,
                 cap: float, now: float) -> dict:
    """The public status of one hotkey on one job: its own data only."""
    from reliquary.corpus.audit_policy import effective_state

    audit_state = effective_state(state, now, params)
    if audit_state == "probation":
        # A ban that ended restarts probation from nothing (§7.3).
        done = 0 if state.banned_until is not None else state.audited_passed
        remaining = max(0, params.probation_submissions - done)
    else:
        remaining = None
    counts = book.counts(hotkey)
    return {
        "job_id": job_id, "hotkey": hotkey, "as_of": now,
        "audit_state": audit_state, "probation_remaining": remaining,
        "suspect_until": state.suspect_until if audit_state == "suspect" else None,
        "banned_until": state.banned_until if audit_state == "banned" else None,
        "submissions_accepted": counts["verdicts"] + pending,
        "audited": counts["audited"], "passed": counts["passed"],
        "passed_unaudited": counts["passed_unaudited"], "failed": counts["failed"],
        "pending_audit": pending, "voided": counts["voided"],
        "counts_complete": counts["counts_complete"],
        "recent_failures": counts["recent_failures"],
        "toploc_thresholds": book.thresholds,
        "verified_tokens_settled": counts["verified_tokens_settled"],
        "share_last_windows": book.share(hotkey),
        "cap": float(cap),
    }


__all__ = [
    "BACKFILL_CONCURRENCY",
    "MINER_STATUS_CACHE_SECONDS",
    "MinerBook",
    "RECENT_FAILURES",
    "SHARE_WINDOWS",
    "feed",
    "miner_status",
    "proof_thresholds",
    "read_recent_windows",
    "valid_hotkey",
]
