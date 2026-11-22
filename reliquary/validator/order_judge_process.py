"""The order control's judge process: each non-episode order's auditor, settler
and miner book, as ``eval_control.wire_order_judge`` wires them in one process,
on this process's own store connections and threads. Design: ``order_judge``."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import secrets
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import Request

from reliquary.validator.order_judge import (
    DISPATCH_SOCKET,
    JUDGE_SOCKET,
    LOCK_FILE,
    DispatchClient,
    OrderJudgeConfig,
)

logger = logging.getLogger(__name__)

# A job whose wiring failed is tried again this much later.
WIRE_RETRY_SECONDS = 30.0
LOCK_WAIT_SECONDS = 10.0
# The first settlement waits this long for who scored what to be read back and re-audited. Past it, it settles
# anyway and flags the order: a restore that keeps failing never stops pay, and a pass settled inside the hold can
# still be re-audited and voided.
RESTORE_WAIT_SECONDS = 2700.0


class CountedStates:
    """A job's ``MinerStates``, counting its writes: the front re-reads the document when the count moves."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.writes = 0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def update_many(self, changes, attempts: int = 6):
        result = await self._inner.update_many(changes, attempts)
        self.writes += 1
        return result

    async def update(self, hotkey, change, attempts: int = 6):
        return (await self.update_many({hotkey: change}, attempts))[hotkey]


class CountedJobStore:
    """The job store, counting each order's ledger writes (tried, as one that failed may have landed):
    the front forgets its memory of a ledger when the count moves."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.ledger_writes: dict[str, int] = {}

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def read_ledgers(self, job_id):
        # The front writes this ledger all the time: never a compare-and-swap from a remembered version.
        forget = getattr(self._inner, "forget_ledgers", None)
        if forget is not None:
            forget(job_id)
        return await self._inner.read_ledgers(job_id)

    async def write_ledgers(self, job_id, snapshot, etag):
        try:
            return await self._inner.write_ledgers(job_id, snapshot, etag)
        finally:
            self.ledger_writes[str(job_id)] = self.ledger_writes.get(str(job_id), 0) + 1


def entry_from_wire(document: Mapping):
    from reliquary.shared.task_registry import TaskEntry

    return TaskEntry(**document)


class OrderJudgeHost:
    """The orders judged here, kept equal to what the front's sync asks for."""

    def __init__(self, *, records, store, client: DispatchClient, feed, threads=None, tokenizer_for=None,
                 settle_seconds: float = 60.0, auditor_kwargs: dict | None = None,
                 clock=time.monotonic, restore_wait_seconds: float = RESTORE_WAIT_SECONDS) -> None:
        from reliquary.validator.eval_control import load_cpu_tokenizer

        self.records, self.store, self.client, self.feed = records, CountedJobStore(store), client, feed
        self.threads = threads
        self._tokenizer_for = tokenizer_for or load_cpu_tokenizer
        self._settle_seconds = settle_seconds
        self._restore_wait = restore_wait_seconds
        self._auditor_kwargs = auditor_kwargs or {}
        self._clock = clock
        self.boot = secrets.token_hex(8)
        self.served: dict[str, Any] = {}
        self.entries: dict[str, tuple[str, Any]] = {}
        self._tokenizers: dict[tuple[str, str], tuple[Any, int]] = {}
        self._wiring: dict[str, asyncio.Task] = {}
        self._job_tasks: dict[str, list[asyncio.Task]] = {}
        self._stopping: dict[str, asyncio.Task] = {}
        self._retry_at: dict[str, float] = {}
        self._attention: dict[str, list[str]] = {}
        self._coldkeys: dict[str, str] | None = None
        self._operators_version: str | None = None

    # -- what the front asks for ---------------------------------------------

    def apply(self, document: Mapping) -> None:
        operators = document.get("operators") or {}
        if "coldkeys" in operators:
            self._coldkeys = None if operators["coldkeys"] is None else dict(operators["coldkeys"])
            self._operators_version = operators.get("version")
        wanted = document.get("jobs") or {}
        for job_id, spec in wanted.items():
            if "entry" in spec:
                self.entries[job_id] = (spec["sha"], entry_from_wire(spec["entry"]))
            elif job_id not in self.entries or self.entries[job_id][0] != spec.get("sha"):
                continue                            # its entry comes with the next sync
            w = self.served.get(job_id)
            if w is not None:
                self._update(w, spec)
            elif (self._operators_version is not None and job_id not in self._wiring
                  and job_id not in self._stopping and self._clock() >= self._retry_at.get(job_id, 0.0)):
                self._wiring[job_id] = asyncio.ensure_future(self._wire(job_id, float(spec["cap"])))
        for job_id in [j for j in (*self.served, *self._wiring) if j not in wanted]:
            if job_id in self._wiring:
                self._wiring.pop(job_id).cancel()
            if job_id in self.served and job_id not in self._stopping:
                self._stopping[job_id] = asyncio.ensure_future(self._stop(job_id))
        for job_id in [j for j in self.entries if j not in wanted]:
            del self.entries[job_id]

    def _update(self, w, spec: Mapping) -> None:
        """A cap and price change, applied as the job set's refresh applies them."""
        from reliquary.validator.corpus_period_settlement import apply_max_price

        job_id = str(w.job.job_id)
        entry = self.entries[job_id][1]
        if entry.status != "active" or str(entry.job_id) != job_id:
            return
        cap = float(spec["cap"])
        if cap != float(w.cap):
            logger.info("order task %s cap %s -> %s", w.entry.task_id, w.cap, cap)
            w.cap = cap
            w.settler.set_cap(cap)
        apply_max_price(w, entry)

    # -- wiring --------------------------------------------------------------

    def _served_tasks(self) -> set[str]:
        return {str(w.entry.task_id) for w in self.served.values()}

    def _operator_of(self, hotkey: str) -> str | None:
        return None if self._coldkeys is None else self._coldkeys.get(hotkey)

    async def _wire(self, job_id: str, cap: float) -> None:
        from reliquary.protocol.profiles import profile_from_contract, toploc_proof
        from reliquary.validator.corpus_job_status import JobStats
        from reliquary.validator.corpus_validator import build_corpus_audit_wiring, warm_drand_chain
        from reliquary.validator.eval_control import ORDER_TRUST_ID, OrderArchives, wire_order_judge

        try:
            entry = self.entries[job_id][1]
            job, _ = await self.store.read_job(job_id)
            if job is None:
                raise RuntimeError(f"order {job_id} has no manifest")
            if job.episode is not None:
                raise RuntimeError(f"episode order {job_id} in the judge process")
            key = (job.checkpoint_repo, job.checkpoint_revision)
            if key not in self._tokenizers:
                self._tokenizers[key] = await asyncio.to_thread(self._tokenizer_for, *key)
            tokenizer, vocab_size = self._tokenizers[key]
            proof = toploc_proof(profile_from_contract(entry.contract))
            params, miner_states, _, beacon, round_at = build_corpus_audit_wiring(
                entry=entry, job=job, records=self.records, trust_id=ORDER_TRUST_ID)
            await warm_drand_chain(getattr(self.threads, "beacon", None))
            counted = CountedStates(miner_states)
            w = SimpleNamespace(entry=entry, cap=cap, job=job, tokenizer=tokenizer, vocab_size=vocab_size,
                                proof=proof, stats=JobStats(), audit_params=params, miner_states=counted)
            view = self.client.view(job_id)
            wire_order_judge(
                w, records=self.records, job_store=self.store, params=params, miner_states=counted,
                beacon=beacon, round_at=round_at, remote=view,
                settle_archives=OrderArchives(served=self._served_tasks),
                operator_of=self._operator_of if self._coldkeys is not None else None,
                arrivals_covered=self.feed.covered, threads=self.threads,
                settle_executor=getattr(self.threads, "codec", None), retry_reopens=True,
                restores_scored=True)
            for name, value in self._auditor_kwargs.items():
                setattr(w.auditor, name, value)
            # Before its listing: what the front accepts from now on reaches it by the feed.
            self.feed.auditors[job_id] = w.auditor
            self.served[job_id] = w
            self._job_tasks[job_id] = [asyncio.ensure_future(self._guarded(job_id, w.auditor.run())),
                                       asyncio.ensure_future(self._guarded(job_id, self._settle_forever(w)))]
            self._retry_at.pop(job_id, None)
            logger.info("order judge: %s wired", job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("order judge: wiring %s failed; retrying", job_id)
            self._flag(job_id, f"judge wiring failed: {type(exc).__name__}: {exc}"[:300])
            self._retry_at[job_id] = self._clock() + WIRE_RETRY_SECONDS
        finally:
            self._wiring.pop(job_id, None)

    def _flag(self, job_id: str, reason: str) -> None:
        reasons = self._attention.setdefault(job_id, [])
        if reason not in reasons:
            reasons.append(reason)
            del reasons[:-10]

    async def _guarded(self, job_id: str, coroutine) -> None:
        """One job's background work: a failure flags the job, never the process."""
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("order judge: %s: its background work failed", job_id)
            self._flag(job_id, f"background work stopped: {type(exc).__name__}: {exc}"[:300])

    async def _settle_forever(self, w) -> None:
        # After the auditor's start: its listing and reads come first after a restart.
        await w.auditor.seeded.wait()
        try:
            await asyncio.wait_for(w.auditor.scores_restored.wait(), self._restore_wait)
        except asyncio.TimeoutError:
            logger.error("order judge: %s: executor scores not restored after %.0f s; settling without them",
                         w.job.job_id, self._restore_wait)
            self._flag(str(w.job.job_id), "settling before its executor scores were restored")
        while True:
            try:
                await w.settler.settle_once()
            except Exception:
                logger.exception("eval settlement of %s failed; retrying", w.job.job_id)
            await asyncio.sleep(self._settle_seconds)

    async def _stop(self, job_id: str) -> None:
        w = self.served.get(job_id)
        try:
            tasks = self._job_tasks.pop(job_id, [])
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if w is not None:
                w.miners.close()
                await asyncio.gather(w.miners.wait_backfill(), return_exceptions=True)
                w.auditor._remote.close()
            self.feed.auditors.pop(job_id, None)
            self.served.pop(job_id, None)
            self._attention.pop(job_id, None)
            if w is not None:
                key = (w.job.checkpoint_repo, w.job.checkpoint_revision)
                if not any((o.job.checkpoint_repo, o.job.checkpoint_revision) == key
                           for o in self.served.values()):
                    self._tokenizers.pop(key, None)
            logger.info("order judge: %s unwired", job_id)
        finally:
            self._stopping.pop(job_id, None)

    # -- the snapshot --------------------------------------------------------

    def _job_view(self, job_id: str, w) -> dict:
        auditor, settler = w.auditor, w.settler
        attention = list(self._attention.get(job_id, ()))
        if not auditor.seeded.is_set():
            return {"seeded": False, "attention": attention}
        pending = auditor.audits_pending()
        return {"seeded": True, "audit_backlog": auditor.audit_backlog(),
                "audit_share": auditor.audit_share(), "audit_decisions": auditor.audit_decisions(),
                "audits_pending": pending, "expected_audits": auditor.expected_audits(pending),
                "audits_idle_since": auditor.audits_idle_since,
                "audit_reasons": dict(auditor.audit_reasons), "ahead_wasted": int(auditor.ahead_wasted),
                "recent_paid": list(settler.recent_paid), "open_value": settler.open_value(),
                "state_writes": w.miner_states.writes, "attention": attention}

    def snapshot(self) -> dict:
        jobs: dict[str, dict] = {}
        for job_id, w in list(self.served.items()):
            try:
                jobs[job_id] = self._job_view(job_id, w)
            except Exception:
                # Unknown for now: the front holds this order.
                logger.exception("order judge: the snapshot of %s failed", job_id)
                jobs[job_id] = {"seeded": False, "attention": list(self._attention.get(job_id, ()))}
        for job_id in self._wiring:
            jobs.setdefault(job_id, {"seeded": False, "attention": list(self._attention.get(job_id, ()))})
        for job_id, reasons in self._attention.items():
            jobs.setdefault(job_id, {"seeded": False, "attention": list(reasons)})
        return {"boot": self.boot, "at": time.time(), "jobs": jobs,
                "entries": {job_id: sha for job_id, (sha, _) in self.entries.items()},
                "operators_version": self._operators_version, "feed": self.feed.state(),
                "ledger_writes": dict(self.store.ledger_writes)}

    async def aclose(self) -> None:
        for task in list(self._wiring.values()):
            task.cancel()
        tasks = [t for group in self._job_tasks.values() for t in group]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, *self._wiring.values(), *self._stopping.values(), return_exceptions=True)
        books = [w.miners for w in self.served.values()]
        for book in books:
            book.close()
        await asyncio.gather(*(book.wait_backfill() for book in books), return_exceptions=True)
        await self.feed.aclose()


def build_order_judge_app(host: OrderJudgeHost):
    from reliquary.validator.corpus_judge_process import build_judge_app

    app = build_judge_app(host.served, host.feed, backfill_after_seed=True)

    @app.post("/sync")
    async def sync(request: Request) -> dict:
        host.apply(await request.json())
        return host.snapshot()

    return app


def _exclusive(run_dir: Path) -> int:
    """One judge at a time: the kernel frees this lock when its holder dies."""
    fd = os.open(run_dir / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise RuntimeError(f"another order judge holds {run_dir / LOCK_FILE}") from None
            time.sleep(0.2)


async def run_order_judge(config: OrderJudgeConfig) -> None:
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_feed import ArrivalFeed
    from reliquary.validator.corpus_gpu import serve_unix
    from reliquary.validator.corpus_judge_threads import JudgeThreads
    from reliquary.validator.eval_control import ORDER_POOL_CONNECTIONS

    run_dir = Path(config.run_dir)
    lock = _exclusive(run_dir)
    threads = JudgeThreads()
    client = DispatchClient(run_dir / DISPATCH_SOCKET)
    host = OrderJudgeHost(
        records=BucketRecordStore(max_pool_connections=ORDER_POOL_CONNECTIONS, executor=threads.codec),
        store=BucketJobStore(), client=client, feed=ArrivalFeed(), threads=threads,
        settle_seconds=config.settle_seconds, auditor_kwargs=config.auditor_kwargs)
    socket_path = run_dir / JUDGE_SOCKET
    try:
        await serve_unix(build_order_judge_app(host), socket_path, services=[client.run()])
    finally:
        await host.aclose()
        await asyncio.to_thread(threads.shutdown, wait=True)
        socket_path.unlink(missing_ok=True)
        os.close(lock)


def judge_main(config: OrderJudgeConfig) -> None:
    """The child's entry point (spawned: a fresh interpreter)."""
    from reliquary.validator.corpus_split import _call, _run_child, _set_parent_death_signal

    _set_parent_death_signal()
    if os.getppid() != config.parent_pid:
        os._exit(1)                                 # the front died before the signal was armed
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | order-judge | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    from reliquary.shared.log_levels import unmute_reliquary_loggers

    unmute_reliquary_loggers()
    if config.child_init:
        _call(config.child_init)
    asyncio.run(_run_child(run_order_judge(config)))


__all__ = ["CountedStates", "OrderJudgeHost", "build_order_judge_app", "judge_main", "run_order_judge"]
