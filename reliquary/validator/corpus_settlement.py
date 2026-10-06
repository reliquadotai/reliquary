"""Pay the corpus task's cap by verified tokens, in ordinary per-task archives.

The weight-only replay pays these archives with no change. The one coupling
with other tasks is the replay horizon (the highest index across tasks), so
the index rules here keep the corpus from ever moving it while another task is
alive. Settlement is two-phase so a crash can delay a payment, never repeat it.
"""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable, Iterable, Mapping
import logging
import os
import time

from reliquary.shared.async_tasks import gather_owned

logger = logging.getLogger(__name__)

SETTLEMENT_SCHEMA = "reliquary/corpus-settlement/v1"

# Wall time of one RL window: the V1 cycle measured on 2026-09-13 (proof
# ~11.3 min + rotation ~4.6 min). Alone, the corpus advances at most this often,
# so the shared replay horizon never moves faster than RL itself moved it.
RL_WINDOW_SECONDS = 16 * 60

# A fed settler learns new verdicts from the auditor; the store is listed at
# boot and this often, as the net for any verdict the feed did not report.
SETTLE_FULL_LIST_SECONDS = float(os.environ.get("RELIQUARY_CORPUS_SETTLE_FULL_LIST_SECONDS", "1800"))
# How long the other tasks' highest window is reused before it is listed again.
OTHER_MAX_TTL_SECONDS = 300.0
# Verdict reads in flight at once for the verdicts the auditor did not feed.
VERDICT_READ_CONCURRENCY = 16


def rewards_for(verdicts: Iterable[Mapping], cap: float) -> dict[str, float]:
    tokens: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.get("passed"):
            tokens[verdict["hotkey"]] = tokens.get(verdict["hotkey"], 0) + int(verdict["token_count"])
    total = sum(tokens.values())
    if total <= 0:
        return {}
    return {hotkey: cap * count / total for hotkey, count in tokens.items()}


def _union(settled, ids) -> list[str]:
    return sorted(set(settled or []) | set(ids))


def _stalled(other_max_seen_at, now, stall_seconds) -> bool:
    return other_max_seen_at is not None and now - other_max_seen_at > stall_seconds


def choose_window(*, last_window, other_max, other_max_seen_at, now, stall_seconds,
                  last_advanced_at=None, advance_every_seconds=0.0):
    if other_max is not None and (last_window is None or other_max > last_window):
        return other_max
    if other_max is None and last_window is None:
        return 0
    if other_max is not None and not _stalled(other_max_seen_at, now, stall_seconds):
        return None
    # Every other task is idle (or none exists): advancing alone decays it the
    # way a retired task already decays, but no faster than RL itself would.
    if last_advanced_at is not None and now - last_advanced_at < advance_every_seconds:
        return None
    return last_window + 1


class CorpusSettler:
    def __init__(self, *, task_id, job_id, cap, records, archives,
                 stall_seconds: float = 3 * RL_WINDOW_SECONDS,
                 advance_every_seconds: float = RL_WINDOW_SECONDS, clock=time.time,
                 on_settled=None, full_list_every_seconds: float | None = None,
                 executor=None,
                 ready: Callable[[list[str]], Awaitable[set[str]]] | None = None) -> None:
        # ``ready(ids)`` answers which of ``ids`` may be paid now; an episode
        # job's grader answers those already graded, so a replay that voids a
        # submission lands before its payment. None pays every verdict as before.
        self._ready = ready
        # Whose threads build the settled sets (the judges', so never the route's).
        self._executor = executor
        self._task_id = task_id
        self._job_id = job_id
        self._cap = float(cap)
        self._records = records
        self._archives = archives
        self._stall = stall_seconds
        self._advance_every = advance_every_seconds
        self._clock = clock
        # How many verdicts stand settled, as of the last settlement read.
        self.settled_count: int | None = None
        # Verdict totals of everything settled, persisted in the settlement
        # object (the status route's counts, with no listing).
        self.totals: dict | None = None
        # Told which ids each settlement moved, once it is written.
        self.on_settled = on_settled
        # Told each window paid and its rewards, once its archive is written.
        self.on_window = None
        # None lists every verdict id on every call. Otherwise `observe` feeds
        # new ids and the store is listed at boot, then at most this often.
        self._full_list_every = full_list_every_seconds
        self._listed_at: float | None = None
        # Ids with a verdict, not yet seen settled: fed or listed.
        self._unsettled: set[str] = set()
        # The fed verdicts themselves, until settled: verdicts are create-only,
        # so the one the auditor reports is the one a read would return.
        self._fed: dict[str, Mapping] = {}

    async def _off(self, func, *args):
        from reliquary.validator.corpus_judge_threads import run_in

        return await run_in(self._executor, func, *args)

    def observe(self, submission_id: str, verdict=None) -> None:
        """A verdict stands for ``submission_id`` (the auditor's ``on_verdict``)."""
        self._unsettled.add(submission_id)
        if isinstance(verdict, Mapping):
            self._fed[submission_id] = verdict

    async def _verdict_ids(self, settled: set) -> list[str]:
        """The verdict ids not yet settled, sorted as the store lists them."""
        now = self._clock()
        due = (self._full_list_every is None or self._listed_at is None
               or not 0 <= now - self._listed_at < self._full_list_every)
        if due:
            listed = await self._records.list_verdict_ids(self._job_id)
            self._unsettled.update(listed)
            self._listed_at = now
        self._unsettled -= settled
        if self._fed:
            self._fed = {sid: v for sid, v in self._fed.items() if sid not in settled}
        return sorted(self._unsettled)

    async def _verdicts(self, ids: list[str]) -> list:
        """Each id's verdict: as fed, else read (one read at a time took
        ~2 minutes per RL window at 7k verdicts an hour)."""
        gate = asyncio.Semaphore(VERDICT_READ_CONCURRENCY)

        async def one(sid):
            async with gate:
                return sid, await self._records.read_verdict(self._job_id, sid)

        # Every read finishes before an error is raised: none outlives the call.
        pairs = await gather_owned((one(sid) for sid in ids if sid not in self._fed),
                                     return_exceptions=True)
        for pair in pairs:
            if isinstance(pair, BaseException):
                raise pair
        read = dict(pairs)
        return [self._fed[sid] if sid in self._fed else read[sid] for sid in ids]

    def set_cap(self, cap: float) -> None:
        """A cap changed in the registry: the next settlement pays under it."""
        self._cap = float(cap)

    def _archive(self, window: int, rewards: Mapping[str, float]) -> dict:
        return {
            "window_start": int(window),
            "window_status": "completed",
            "rewards_by_hotkey": dict(rewards),
            "task_id": self._task_id,
            "mechanism": "corpus-generation",
            "job_id": self._job_id,
        }

    async def _finish(self, state: dict, etag, now: float) -> int:
        pending = state["pending"]
        # Idempotent: the same window and the same rewards, however often a
        # crash makes this run again.
        await self._archives.write(self._task_id, pending["window"], self._archive(pending["window"], pending["rewards"]))
        final = {
            **state,
            "last_window": pending["window"],
            "settled": await self._off(_union, state.get("settled"), pending["ids"]),
            "pending": None,
            # Carried in the pending step, so a repeated finish adds nothing twice.
            "totals": pending.get("totals", state.get("totals")),
        }
        if pending.get("alone"):
            # The finish time, not the choice time: a finish delayed by a crash
            # or a hold must still be one RL window from the next lone advance.
            final["advanced_at"] = now
        await self._records.write_settlement(self._job_id, final, etag)
        self._settled(final, pending["ids"])
        if self.on_window is not None:
            try:
                self.on_window(pending["window"], pending["rewards"])
            except Exception:
                logger.exception("corpus window report for %s failed", self._task_id)
        return pending["window"]

    def _settled(self, state: dict, ids) -> None:
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

    async def settle_once(self) -> int | None:
        state, etag = await self._records.read_settlement(self._job_id)
        state = {"schema": SETTLEMENT_SCHEMA, "last_window": None, "settled": [],
                 "other_max_seen": None, "other_max_seen_at": None, "advanced_at": None,
                 "pending": None, "totals": None, **state}
        if state["totals"] is None:
            # A job settled before totals were kept: counted from here on.
            state["totals"] = {"verdicts": 0, "passed": 0, "verified_tokens": 0,
                               "complete": not state["settled"]}
        self._settled(state, ())

        now = self._clock()
        other_max = await self._archives.other_max(self._task_id)
        clock_changed = other_max != state["other_max_seen"]
        if clock_changed:
            # Another task sealed: a new stall, if one comes, starts its own
            # RL-cadence spacing from scratch.
            state["other_max_seen"], state["other_max_seen_at"] = other_max, now
            state["advanced_at"] = None

        if state["pending"]:
            window = state["pending"]["window"]
            live = other_max is not None and not _stalled(state["other_max_seen_at"], now, self._stall)
            if live and window > other_max:
                # Chosen alone during a stall and interrupted; the other task
                # has since revived. Re-targeting the ids to another index could
                # pay them twice if the archive already landed, so hold them
                # until the live task reaches this window (the next corpus index
                # would have waited for exactly that anyway).
                if clock_changed:
                    await self._records.write_settlement(self._job_id, state, etag)
                return None
            return await self._finish(state, etag, now)

        # 300k+ ids on a long job: built off the serving loop.
        settled = await self._off(set, state["settled"])
        new_ids = await self._verdict_ids(settled)
        if self._ready is not None and new_ids:
            payable = await self._ready(new_ids)
            new_ids = [sid for sid in new_ids if sid in payable]
        window = choose_window(last_window=state["last_window"], other_max=other_max,
                               other_max_seen_at=state["other_max_seen_at"], now=now,
                               stall_seconds=self._stall, last_advanced_at=state["advanced_at"],
                               advance_every_seconds=self._advance_every)

        if new_ids and window is not None:
            verdicts = await self._verdicts(new_ids)
            lister = getattr(self._records, "list_voided_ids", None)
            if lister is not None:
                # Withdrawn after a quarantined executor's re-audit: settled, never paid.
                voided = set(await lister(self._job_id))
                verdicts = [v if sid not in voided else {**(v or {}), "passed": False}
                            for sid, v in zip(new_ids, verdicts)]
            rewards = rewards_for(verdicts, self._cap)
            totals = self._add(state["totals"], verdicts)
            if rewards:
                alone = other_max is None or window > other_max
                state["pending"] = {"window": window, "ids": new_ids, "rewards": rewards,
                                    "alone": alone, "at": now, "totals": totals}
                etag = await self._records.write_settlement(self._job_id, state, etag)
                return await self._finish(state, etag, now)
            # Every verdict this period failed (spec §7): no archive, the
            # index does not move, but these ids must not be reconsidered
            # forever, so mark them settled in this same CAS write.
            state["settled"] = await self._off(_union, settled, new_ids)
            state["totals"] = totals
            await self._records.write_settlement(self._job_id, state, etag)
            self._settled(state, new_ids)
            return None

        if clock_changed:
            # Nothing settles this call, but other_max genuinely moved: CAS
            # it in now. Otherwise the next call finds the persisted
            # other_max_seen still stale, "changes" again, and keeps
            # resetting the stall clock to "now" forever — the corpus is
            # never paid again once the other task goes idle (§7b rule 3).
            await self._records.write_settlement(self._job_id, state, etag)
        return None


def settler_fed(settler: CorpusSettler, on_verdict=None):
    """The auditor's ``on_verdict``: the settler hears of the verdict first,
    so a failing status hook can never keep it from being paid on time."""

    def report(submission_id: str, verdict) -> None:
        settler.observe(submission_id, verdict)
        if on_verdict is not None:
            on_verdict(submission_id, verdict)

    return report


class R2Archives:
    """The two archive calls the settler makes, against the real bucket.

    ``served`` names the tasks this process wired after boot, which
    ``RELIQUARY_TASK_ID`` cannot list.
    """

    def __init__(self, *, served=None, ttl_seconds: float = OTHER_MAX_TTL_SECONDS,
                 clock=time.monotonic) -> None:
        self._served = served
        self._ttl = ttl_seconds
        self._clock = clock
        # Per task, its highest window (None: no archive), as of `_listed_at`;
        # shared by every job's settler, raised by this process's own writes.
        self._max: dict[str, int | None] | None = None
        self._listed_at: float | None = None
        self._refresh: asyncio.Lock | None = None

    @staticmethod
    async def _list_maxes() -> dict[str, int | None]:
        from reliquary.infrastructure import storage

        maxes: dict[str, int | None] = {}
        for task in await storage.list_task_ids(strict=True):
            windows = await storage.list_all_window_keys(task_id=task, strict=True)
            maxes[task] = max(windows) if windows else None
        return maxes

    async def other_max(self, task_id: str) -> int | None:
        from reliquary.infrastructure import storage

        if self._refresh is None:
            self._refresh = asyncio.Lock()
        async with self._refresh:
            now = self._clock()
            if self._max is None or not 0 <= now - self._listed_at < self._ttl:
                # Another task's fresh window shows within the TTL; the stall
                # rule (minutes of silence, not seconds) is unchanged by it.
                self._max = await storage.off_loop(self._list_maxes())
                self._listed_at = now
        best = None
        for other, window in self._max.items():
            if other != task_id and window is not None:
                best = max(best or 0, window)
        return best

    def refuse_unserved(self, task_id: str) -> None:
        """The corpus validator runs under its own task id(s), so it refuses to
        write under any task RELIQUARY_TASK_ID (or its hot set) does not name.
        Unset is refused as before, never read as the legacy task."""
        import os

        from reliquary.shared.task_id import parse_task_ids

        served = os.getenv("RELIQUARY_TASK_ID")
        hot = set(self._served()) if self._served is not None else set()
        if not served or (task_id not in parse_task_ids(served) and task_id not in hot):
            raise RuntimeError(f"RELIQUARY_TASK_ID does not name {task_id!r}; refusing to archive")

    async def write(self, task_id: str, window: int, data: dict) -> None:
        from reliquary.infrastructure import storage

        self.refuse_unserved(task_id)
        await storage.upload_window_dataset(window, data, task_id=task_id)
        if self._max is not None:
            # Seen at once by the other jobs' settlers, without a listing.
            known = self._max.get(task_id)
            self._max[task_id] = window if known is None else max(known, int(window))
