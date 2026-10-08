"""A judge process of the split corpus validator.

It judges and pays a group of jobs with exactly the code the single process
runs (``wire_job_judge``: the scheduled ``CorpusAuditor`` and the fed
``CorpusPeriodSettler``), with its forward on the GPU process and its arrivals from
the front's feed. Its unix socket takes the feed and answers the front's
status reads; it serves nothing to miners.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)


def build_judge_app(served: Mapping[str, Any], feed) -> FastAPI:
    """``served`` maps job id to its wiring (``auditor``, ``settler``,
    ``stats``, ``miners``)."""
    app = FastAPI()

    def wiring(job_id: str):
        w = served.get(job_id)
        if w is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_judged_here")
        return w

    @app.post("/feed")
    async def take_feed(request: Request) -> dict:
        feed.receive(await request.json())
        return {"ok": True}

    @app.get("/jobs/{job_id}/stats")
    async def stats(job_id: str) -> dict:
        w = wiring(job_id)
        return {"unsettled": list(w.stats.unsettled()),
                "settled_count": w.settler.settled_count, "totals": w.settler.totals}

    @app.get("/jobs/{job_id}/miners/{hotkey}")
    async def miner(job_id: str, hotkey: str) -> dict:
        w = wiring(job_id)
        w.miners.start_backfill()
        return {"counts": w.miners.counts(hotkey), "share": w.miners.share(hotkey),
                "thresholds": w.miners.thresholds, "pending": w.auditor.pending_count(hotkey)}

    @app.get("/health")
    async def health() -> dict:
        return {"jobs": sorted(served), "feed": feed.state()}

    return app


def registry_reader() -> Callable[[], Awaitable[Mapping[str, Any]]]:
    from reliquary.infrastructure.task_registry_store import read_registry

    async def entries():
        found, _ = await read_registry()
        return found

    return entries


async def refresh_caps(served: Mapping[str, Any], read_entries, every_seconds: float) -> None:
    """The registry's cap of each job judged here, applied to its settler as
    the single process's job set does (the front's copy is for its status)."""
    while True:
        await asyncio.sleep(every_seconds)
        try:
            entries = await read_entries()
        except Exception:
            logger.exception("corpus judge: the task registry could not be read; caps unchanged")
            continue
        for w in served.values():
            entry = entries.get(str(w.entry.task_id))
            if entry is None or entry.status != "active" or str(entry.job_id) != str(w.job.job_id):
                continue
            cap = float(entry.params["cap"])
            if cap != float(w.cap):
                logger.info("corpus task %s cap %s -> %s", w.entry.task_id, w.cap, cap)
                w.cap = cap
                w.settler.set_cap(cap)


async def run_corpus_judges(*, served, directory: str, run_dir: str, proof, socket_path: str,
                            settle_every_seconds: float = 60.0, hot: bool = False,
                            auditor_kwargs: dict | None = None) -> None:
    """Judge and settle the jobs of ``served`` (``(entry, cap)`` pairs) until
    one of them stops (an auditor halted on validator-side errors): then this
    process exits and its supervisor starts it again."""
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_feed import ArrivalFeed
    from reliquary.validator.corpus_gpu import GPU_SOCKET, GpuScorer, read_info, serve_unix
    from reliquary.validator.corpus_hot_jobs import JOB_REFRESH_SECONDS
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_judge_threads import JudgeThreads, judge_record_store
    from reliquary.validator.corpus_settlement import R2Archives
    from reliquary.validator.corpus_validator import (
        JUDGE_POOL_CONNECTIONS, settle_forever, wire_job_judge,
    )

    from reliquary.validator.corpus_validator import period_served

    store = BucketJobStore()
    jobs = []
    served, _ = period_served(served)
    for entry, cap in served:
        job, _ = await store.read_job(str(entry.job_id))
        if job is None:
            raise RuntimeError(f"task {entry.task_id!r} declares job {entry.job_id!r} "
                               "but it has no manifest")
        if getattr(job, "episode", None) is not None:
            # The supervisor's plan refuses it first; never paid ungraded.
            raise RuntimeError(f"episode job {job.job_id!r} in a judge process: judge processes "
                               "host no grader; leave it out of RELIQUARY_CORPUS_SPLIT_JUDGES")
        jobs.append((entry, cap, job))
    tokenizer = load_tokenizer(str(directory))
    vocab_size = (await read_info(run_dir))["vocab_size"]
    records = BucketRecordStore()
    judge_threads = JudgeThreads()
    judge_records = judge_record_store(judge_threads, max_pool_connections=JUDGE_POOL_CONNECTIONS)
    # The settlers of this process share one view of the other tasks' windows.
    archives = R2Archives()
    scorer = GpuScorer(Path(run_dir) / GPU_SOCKET, chunk_tokens=proof.chunk_tokens,
                       topk=proof.topk, executor=judge_threads.codec)
    feed = ArrivalFeed()
    # One job of this process prepares and scores at a time, as in-process.
    gpu_lock = asyncio.Lock()
    wiring: dict[str, Any] = {}
    for entry, cap, job in jobs:
        w = SimpleNamespace(entry=entry, cap=cap, job=job, stats=JobStats())
        wire_job_judge(w, records=records, judge_records=judge_records,
                       judge_threads=judge_threads, archives=archives, proof=proof, model=None,
                       tokenizer=tokenizer, gpu_lock=gpu_lock, scorer=scorer,
                       vocab_size=vocab_size, arrivals_covered=feed.covered,
                       auditor_kwargs=auditor_kwargs)
        wiring[str(job.job_id)] = w
    feed.auditors = {job_id: w.auditor for job_id, w in wiring.items()}
    app = build_judge_app(wiring, feed)
    logger.info("corpus judge serving %s on %s", sorted(wiring), socket_path)
    tasks = []
    for w in wiring.values():
        tasks += [w.auditor.run(), settle_forever(w.entry.task_id, w.settler, settle_every_seconds)]
    if hot:
        tasks.append(refresh_caps(wiring, registry_reader(), JOB_REFRESH_SECONDS))
    try:
        await serve_unix(app, socket_path, services=tasks)
    finally:
        await feed.aclose()
        books = [w.miners for w in wiring.values()]
        for book in books:
            book.close()
        await asyncio.gather(*(book.wait_backfill() for book in books), return_exceptions=True)
        await asyncio.to_thread(judge_threads.shutdown, wait=True)


__all__ = ["build_judge_app", "refresh_caps", "registry_reader", "run_corpus_judges"]
