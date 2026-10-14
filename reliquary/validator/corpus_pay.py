"""`GET /corpus/pay`: what each active period-settled corpus job is paying per
verified token, for miners choosing where to put their GPUs. Read-only: it
changes nothing in settlement or in the weights.

Everything comes from what the process already holds, or reads behind the
requests on its own clock:

- the current period's verified tokens, dated by arrival as settlement dates
  them: the settler's count of its unsettled verdicts (``open_periods``),
  refreshed every settlement pass; in a split validator, from the judge's
  stats, as the job status reads them;
- the settled periods: the task's period archives (one prefix listing every
  ``HISTORY_REFRESH_SECONDS``, each archive read once: they never change);
- the alpha price and the miners' emission: the chain, every
  ``CHAIN_REFRESH_SECONDS``, on a thread of its own;
- the whole answer is recomputed at most once per ``PAY_CACHE_SECONDS``,
  however many requests ask for it, and a failed recompute serves the last one.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from reliquary.validator import corpus_periods as cp

logger = logging.getLogger(__name__)

PAY_CACHE_SECONDS = 60.0
# While the chain or an archive history is still loading for the first time,
# the answer is kept only this long, so the full one follows within seconds.
LOADING_CACHE_SECONDS = 5.0
HISTORY_PERIODS = 12
HISTORY_REFRESH_SECONDS = 600.0
HISTORY_RETRY_SECONDS = 120.0
CHAIN_REFRESH_SECONDS = 600.0
CHAIN_RETRY_SECONDS = 60.0
# A chain read older than this is not served: null rather than stale.
CHAIN_STALE_SECONDS = 3600.0
# A whole chain read (connect, two queries), on its thread.
CHAIN_READ_BUDGET_SECONDS = 200.0
BLOCK_SECONDS = 12.0

NOTE = (
    "An estimate, not a promise. A period pays min(cap, max price x its verified tokens) "
    "in shares of the miners' emission, split by verified tokens, dated by when each "
    "submission was received; the current period's projection counts only what is verified "
    "so far and falls as more tokens arrive once the cap binds. A settled period's pay enters "
    "the weights through an EMA over about 6 periods and is paid through Yuma consensus, so "
    "what a hotkey receives also depends on the other validators' weights and on when the "
    "period settles (settlement_lag_seconds). Alpha amounts use miner_alpha_per_period; no "
    "fiat conversion is made."
)


def tokens_to_saturate(cap: float, price: float | None) -> float | None:
    """Verified tokens in a period from which its cap, not the price, binds."""
    if price is None or price <= 0:
        return None
    return float(cap) / float(price)


def projection(*, cap: float, price: float | None, tokens: int | None,
               emission: float | None) -> dict[str, Any]:
    """What a period of ``tokens`` verified tokens pays, in all and per million
    tokens; alpha amounts only with the miners' ``emission`` per period."""
    if tokens is None:
        return {"verified_tokens": None, "budget_pool_share": None, "budget_alpha": None,
                "projected_pool_share_per_mtoken": None, "projected_alpha_per_mtoken": None,
                "saturated": None}
    tokens = int(tokens)
    budget = cp.period_budget(cap, price, tokens) if tokens > 0 else 0.0
    per = budget / tokens * 1e6 if tokens > 0 else None
    return {
        "verified_tokens": tokens,
        "budget_pool_share": budget,
        "budget_alpha": None if emission is None else budget * emission,
        "projected_pool_share_per_mtoken": per,
        "projected_alpha_per_mtoken": None if per is None or emission is None else per * emission,
        "saturated": tokens > 0 and (price is None or float(price) * tokens >= float(cap)),
    }


def history_rows(docs: Iterable[Mapping], *, emission: float | None,
                 limit: int = HISTORY_PERIODS) -> list[dict]:
    """The last ``limit`` settled periods, newest first, from their archives
    (a period settled in several passes has several)."""
    periods: dict[int, dict] = {}
    for doc in docs:
        work = int(doc["work_period"])
        row = periods.setdefault(work, {"tokens": 0, "paid": 0.0, "budget": None})
        row["tokens"] += int(doc.get("tokens") or 0)
        row["paid"] += math.fsum(float(v) for v in (doc.get("rewards_by_hotkey") or {}).values())
        bound = doc.get("budget", doc.get("cap"))
        if bound is not None:
            row["budget"] = max(float(bound), row["budget"] or 0.0)
    rows = []
    for work in sorted(periods, reverse=True)[:limit]:
        row = periods[work]
        per = row["paid"] / row["tokens"] * 1e6 if row["tokens"] > 0 else None
        rows.append({"work_period": work, "verified_tokens": row["tokens"],
                     "paid_pool_share": row["paid"], "budget": row["budget"],
                     "paid_pool_share_per_mtoken": per,
                     "paid_alpha_per_mtoken": (None if per is None or emission is None
                                               else per * emission)})
    return rows


class _Behind:
    """Reads run behind the requests that ask for them, one per key at a time."""

    def __init__(self) -> None:
        self._tasks: dict[Any, asyncio.Task] = {}

    def running(self, key) -> bool:
        task = self._tasks.get(key)
        return task is not None and not task.done()

    def kick(self, key, start: Callable[[], Awaitable[None]]) -> None:
        if self.running(key):
            return
        self._tasks[key] = asyncio.get_running_loop().create_task(start())

    async def settle(self) -> None:
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()


class ChainView:
    """The last chain read (``read()``: ``alpha_price_tao``,
    ``miner_alpha_per_block``), refreshed behind. ``current()`` never raises
    and never waits: None until a read succeeds, or once the last success is
    older than ``CHAIN_STALE_SECONDS``."""

    def __init__(self, read: Callable[[], Awaitable[Mapping]], *,
                 clock: Callable[[], float] = time.time) -> None:
        self._read, self._clock = read, clock
        self._value: dict | None = None
        self._value_at: float | None = None
        self._tried_at: float | None = None
        self._failed = False
        self._behind = _Behind()

    @property
    def pending(self) -> bool:
        """Never read yet, and a read under way."""
        return self._value is None and self._behind.running("chain")

    def current(self) -> dict | None:
        now = self._clock()
        wait = CHAIN_RETRY_SECONDS if self._failed else CHAIN_REFRESH_SECONDS
        if self._tried_at is None or now - self._tried_at >= wait:
            self._behind.kick("chain", self._refresh)
        if self._value is None or now - self._value_at > CHAIN_STALE_SECONDS:
            return None
        return self._value

    async def _refresh(self) -> None:
        self._tried_at = self._clock()
        try:
            value = dict(await self._read())
        except Exception as exc:  # noqa: BLE001 - a chain fault only nulls the fields
            self._failed = True
            logger.warning("corpus pay: chain read failed: %r", exc)
            return
        self._failed = False
        self._value, self._value_at = value, self._clock()

    async def settle(self) -> None:
        await self._behind.settle()

    async def close(self) -> None:
        await self._behind.close()


class PeriodHistory:
    """Each task's last ``limit`` settled periods' archives, refreshed behind:
    one listing of the task's own prefix per refresh, each archive read once."""

    def __init__(self, archives, *, clock: Callable[[], float] = time.time,
                 limit: int = HISTORY_PERIODS) -> None:
        self._archives, self._clock, self._limit = archives, clock, limit
        self._docs: dict[str, dict[tuple[int, int], dict]] = {}
        self._loaded: set[str] = set()
        self._tried_at: dict[str, float] = {}
        self._failed: set[str] = set()
        self._behind = _Behind()

    def pending(self, task_id: str) -> bool:
        return task_id not in self._loaded and self._behind.running(task_id)

    def docs(self, task_id: str) -> list[dict] | None:
        """The archives held, oldest first; None before the first load."""
        tried = self._tried_at.get(task_id)
        wait = HISTORY_RETRY_SECONDS if task_id in self._failed else HISTORY_REFRESH_SECONDS
        if tried is None or self._clock() - tried >= wait:
            self._tried_at[task_id] = self._clock()
            self._behind.kick(task_id, lambda: self._load(task_id))
        if task_id not in self._loaded:
            return None
        held = self._docs.get(task_id, {})
        return [held[key] for key in sorted(held)]

    async def _load(self, task_id: str) -> None:
        try:
            listed = sorted(await self._archives.list(task_id))
            works = set(sorted({work for work, _ in listed})[-self._limit:])
            known = self._docs.get(task_id, {})
            fresh: dict[tuple[int, int], dict] = {}
            for key in (key for key in listed if key[0] in works):
                doc = known.get(key)
                if doc is None:
                    doc = await self._archives.read(task_id, *key)
                if doc is not None:
                    fresh[key] = doc
        except Exception as exc:  # noqa: BLE001 - the last history is kept
            self._failed.add(task_id)
            logger.warning("corpus pay: period archives of %s unreadable: %r", task_id, exc)
            return
        self._failed.discard(task_id)
        self._docs[task_id] = fresh
        self._loaded.add(task_id)

    async def settle(self) -> None:
        await self._behind.settle()

    async def close(self) -> None:
        await self._behind.close()


class OffLoopArchives:
    """The period archives, listed and read on the storage listing loop,
    never the serving one."""

    def __init__(self) -> None:
        from reliquary.infrastructure.corpus_period_store import R2PeriodArchives

        self._r2 = R2PeriodArchives()

    async def list(self, task_id):
        from reliquary.infrastructure.storage import off_loop

        return await off_loop(self._r2.list(task_id))

    async def read(self, task_id, work_period, entry_period):
        from reliquary.infrastructure.storage import off_loop

        return await off_loop(self._r2.read(task_id, work_period, entry_period))


async def read_subnet_pool() -> dict:
    """One chain read on a thread with its own event loop: importing
    bittensor and opening its websocket never hold the serving loop."""

    def run() -> dict:
        async def read() -> dict:
            from reliquary.infrastructure.chain import close_subtensor, get_subtensor, subnet_pool

            subtensor = await get_subtensor()
            try:
                return await subnet_pool(subtensor)
            finally:
                await close_subtensor(subtensor)

        return asyncio.run(read())

    return await asyncio.wait_for(asyncio.to_thread(run), timeout=CHAIN_READ_BUDGET_SECONDS)


def _price_view(wiring) -> dict | None:
    if hasattr(wiring, "price"):
        return wiring.price
    return cp.price_view(getattr(getattr(wiring, "entry", None), "params", None))


class PayBoard:
    """The `/corpus/pay` answer over a ``CorpusJobSet``."""

    def __init__(self, job_set, *, archives=None, read_chain=None,
                 clock: Callable[[], float] = time.time, genesis: float = cp.PERIOD_EPOCH,
                 cache_seconds: float = PAY_CACHE_SECONDS) -> None:
        self._job_set = job_set
        self._clock = clock
        self._genesis = float(genesis)
        self._cache_seconds = float(cache_seconds)
        self._history = PeriodHistory(archives if archives is not None else OffLoopArchives(),
                                      clock=clock)
        self._chain = ChainView(read_chain or read_subnet_pool, clock=clock)
        # (computed at, kept for, answer).
        self._cached: tuple[float, float, dict] | None = None
        self._lock = asyncio.Lock()

    @property
    def job_set(self):
        return self._job_set

    async def pay(self) -> dict:
        cached = self._cached
        if cached is not None and self._clock() - cached[0] < cached[1]:
            return cached[2]
        async with self._lock:
            # One recompute, however many requests wait on it.
            cached = self._cached
            now = self._clock()
            if cached is not None and now - cached[0] < cached[1]:
                return cached[2]
            try:
                doc, loading = await self._compute()
            except Exception:
                if cached is None:
                    raise
                logger.warning("corpus pay: recompute failed; serving the last one",
                               exc_info=True)
                self._cached = (now, LOADING_CACHE_SECONDS, cached[2])
                return cached[2]
            self._cached = (now, LOADING_CACHE_SECONDS if loading else self._cache_seconds, doc)
            return doc

    async def settle(self) -> None:
        """Wait for the reads under way (tests)."""
        await asyncio.gather(self._chain.settle(), self._history.settle())

    async def close(self) -> None:
        await asyncio.gather(self._chain.close(), self._history.close())

    def _served(self) -> list[tuple[str, Any]]:
        job_set = self._job_set
        return [(job_id, w) for job_id, w in sorted(job_set.served.items())
                if not job_set.is_retired(job_id)
                and cp.is_period_task(getattr(w, "entry", None))]

    async def _compute(self) -> tuple[dict, bool]:
        now = self._clock()
        current = cp.period_of(now, self._genesis)
        served = self._served()
        chain = self._chain.current()
        emission, source = None, None
        if chain is not None and chain.get("miner_alpha_per_block") is not None:
            emission = float(chain["miner_alpha_per_block"]) * cp.PERIOD_SECONDS / BLOCK_SECONDS
            source = "chain"
        else:
            for _, w in served:
                frozen = (_price_view(w) or {}).get("miner_alpha_per_period")
                if frozen is not None:
                    emission, source = float(frozen), "price_ceiling"
                    break
        jobs = [await self._job(job_id, w, now=now, current=current, emission=emission)
                for job_id, w in served]
        loading = self._chain.pending or any(
            self._history.pending(str(w.entry.task_id)) for _, w in served)
        doc = {
            "as_of": now,
            "period_seconds": cp.PERIOD_SECONDS,
            "current_period": current,
            "current_period_ends_at": cp.period_end(current, self._genesis),
            "miner_alpha_per_period": emission,
            "miner_alpha_per_period_source": source,
            "alpha_price_tao": None if chain is None else chain.get("alpha_price_tao"),
            "note": NOTE,
            "jobs": jobs,
        }
        return doc, loading

    async def _settler_view(self, job_id: str, wiring) -> tuple[dict | None, dict | None,
                                                                 float | None]:
        """(open periods, undecided, counted at), from the settler here or the
        judge's stats there; Nones for what cannot be told."""
        link = getattr(wiring, "judge_link", None)
        if link is not None:
            try:
                judged = await link.stats(job_id)
            except Exception as exc:  # noqa: BLE001 - one job's judge, never the route
                logger.warning("corpus pay: judge stats of %s unavailable: %r", job_id, exc)
                return None, None, None
            judged = judged or {}
            return (judged.get("open_periods"), judged.get("undecided"),
                    judged.get("open_periods_at"))
        settler = getattr(wiring, "settler", None)
        undecided = getattr(settler, "undecided_since", None)
        return (getattr(settler, "open_periods", None),
                undecided() if callable(undecided) else None,
                getattr(settler, "open_periods_at", None))

    async def _job(self, job_id: str, wiring, *, now: float, current: int,
                   emission: float | None) -> dict:
        view = _price_view(wiring)
        price = None if view is None else view.get("max_price_pool_share_per_token")
        cap = float(wiring.cap)
        try:
            status = await self._job_set.status(job_id)
        except Exception as exc:  # noqa: BLE001 - the counts are optional here
            logger.warning("corpus pay: status of %s unavailable: %r", job_id, exc)
            status = None
        open_periods, undecided, counted_at = await self._settler_view(job_id, wiring)
        tokens = None
        unsettled = None
        if open_periods is not None:
            tokens = int((open_periods.get(str(current)) or {}).get("verified_tokens", 0))
            unsettled = [{"work_period": int(p), "verified_tokens": int(row["verified_tokens"])}
                         for p, row in sorted(open_periods.items(), key=lambda kv: int(kv[0]))
                         if int(p) < current]
        if undecided is None or not undecided.get("known"):
            lag = None
        elif undecided.get("at") is None:
            lag = 0.0
        else:
            lag = max(0.0, now - float(undecided["at"]))
        docs = self._history.docs(str(wiring.entry.task_id))
        prompts_total = prompts_open = None
        if status is not None and "prompts_total" in status:
            prompts_total = int(status["prompts_total"])
            prompts_open = prompts_total - int(status.get("prompts_full", 0))
        return {
            "job_id": job_id,
            "task_id": str(wiring.entry.task_id),
            "state": None if status is None else status.get("state"),
            "admission": None if status is None else status.get("admission"),
            "cap": cap,
            "max_price_alpha_per_mtoken": None if view is None else view.get(
                "max_price_alpha_per_mtoken"),
            "max_price_pool_share_per_token": price,
            "tokens_to_saturate": tokens_to_saturate(cap, price),
            "current_period": {
                "work_period": current,
                "started_at": cp.period_end(current - 1, self._genesis),
                "ends_at": cp.period_end(current, self._genesis),
                "counted_at": counted_at,
                **projection(cap=cap, price=price, tokens=tokens, emission=emission),
            },
            "unsettled_periods": unsettled,
            "history": (None if docs is None
                        else history_rows(docs, emission=emission, limit=HISTORY_PERIODS)),
            "settlement_lag_seconds": lag,
            "prompts_open": prompts_open,
            "prompts_total": prompts_total,
        }


__all__ = [
    "CHAIN_REFRESH_SECONDS",
    "ChainView",
    "HISTORY_PERIODS",
    "HISTORY_REFRESH_SECONDS",
    "NOTE",
    "PAY_CACHE_SECONDS",
    "PayBoard",
    "PeriodHistory",
    "history_rows",
    "projection",
    "read_subnet_pool",
    "tokens_to_saturate",
]
