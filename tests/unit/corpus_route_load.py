"""Real time, real threads: four judges catching up a backlog in the same
process as a route, as on 2026-10-02 12:00-12:25. The store is the real
``BucketRecordStore`` over an in-memory bucket with R2-like latency; the drand
beacon blocks its thread like the relay race does; the GPU forward blocks its
thread for tokens / 4860 s. The route's probe measures how long its own
thread hops and store calls take meanwhile. Not a test module itself."""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from reliquary.corpus.audit_policy import AuditParams, MinerState
from reliquary.infrastructure import corpus_record_store as record_store
from reliquary.infrastructure.corpus_record_store import BucketRecordStore
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

# 20 vCPU in prod: Python's default executor is min(32, cpu + 4).
PROD_DEFAULT_WORKERS = 24
# Set when a run ends: blocked fake threads return at once, so nothing outlives it.
STOP = threading.Event()
HOTKEYS = [f"5Hk{i:03d}" for i in range(24)]


class LatentR2(_FakeMultiObjectR2):
    """Every call waits like R2 (reads 50-100 ms, puts 300-500 ms); bodies close."""

    def __init__(self, seed=0):
        super().__init__()
        self._rng = random.Random(seed)
        self.unclosed = 0

    async def get_object(self, Bucket, Key):
        await asyncio.sleep(self._rng.uniform(0.05, 0.10))
        response = await super().get_object(Bucket=Bucket, Key=Key)
        body, store = response["Body"], self

        class _Body:
            async def read(self):
                return await body.read()

            def close(self):
                store.unclosed -= 1

        self.unclosed += 1
        return {**response, "Body": _Body()}

    async def put_object(self, Bucket, Key, Body, **condition):
        await asyncio.sleep(self._rng.uniform(0.3, 0.5))
        return await super().put_object(Bucket=Bucket, Key=Key, Body=Body, **condition)


def blocking_beacon(seconds=0.4):
    """The relay race holds its calling thread until the first relay answers."""

    def beacon(round_number):
        STOP.wait(seconds)
        return "%064x" % (round_number * 2654435761 % (1 << 255))

    return beacon


def round_at(t):
    return int(t // 3) + 1


class Gpu:
    def __init__(self, tokens_per_second=4860.0):
        self.rate = tokens_per_second

    def __call__(self, records):
        STOP.wait(sum(int(r["token_count"]) for r in records) / self.rate)
        return [{"passed": True, "reason": None, "worst_exp": 0, "worst_mant_mean": 0.01,
                 "worst_mant_median": 0.01} for _ in records]


def _record(rng, hotkey, received_at, completion_tokens):
    tokens = [rng.randrange(150000) for _ in range(completion_tokens)]
    return {"schema": "reliquary/corpus-record/v1", "hotkey": hotkey,
            "rendered_prompt": "p" * 2000, "received_at": received_at,
            "token_count": completion_tokens,
            "completions": [{"tokens": tokens, "text": "w " * completion_tokens,
                             "proofs": ["A" * 344] * (completion_tokens // 32)}]}


def seed_backlog(bucket, job_id, n, *, completion_tokens, seed, spacing=3.1):
    """``n`` records past their hold, ``spacing`` s apart (a drand round is 3 s)."""
    rng = random.Random(seed)
    now = time.time()
    template = {hk: _record(rng, hk, 0.0, completion_tokens) for hk in HOTKEYS}
    for i in range(n):
        hotkey = HOTKEYS[i % len(HOTKEYS)]
        sid = "%064x" % rng.getrandbits(256)
        body = json.dumps({**template[hotkey], "received_at": now - 6000 - spacing * i}).encode()
        bucket.objects[record_store._key(job_id, "submissions", sid)] = (body, '"s"')
    miners = {hk: MinerState(audited_passed=1000).to_dict() for hk in HOTKEYS}
    bucket.objects[record_store._miners_key(job_id)] = (json.dumps(miners).encode(), '"m"')


async def _probe(route_store, seconds, waits, writes):
    """The route: a trivial thread hop (its admit) and a record write, every 50 ms."""
    rng = random.Random(99)
    record = _record(rng, "5Route", time.time(), 2000)
    end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < end:
        start = time.monotonic()
        await asyncio.to_thread(lambda: None)
        waits.append(time.monotonic() - start)
        start = time.monotonic()
        n += 1
        await route_store.write_submission("route-job", "%064x" % n, record)
        writes.append(time.monotonic() - start)
        await asyncio.sleep(0.05)


def run(build_judges, *, jobs=4, backlog=400, completion_tokens=8000, seconds=12.0,
        default_workers=PROD_DEFAULT_WORKERS, spacing=3.1):
    """``build_judges(bucket_factory, job_ids)`` returns (auditors, cleanup).
    Returns the route's thread-hop waits and record-write times, and how many
    records the judges wrote verdicts for."""
    bucket = LatentR2()

    def factory(**_kw):
        return bucket

    saved = record_store.get_s3_client
    record_store.get_s3_client = factory
    job_ids = [f"job{j}" for j in range(jobs)]
    for j, job_id in enumerate(job_ids):
        seed_backlog(bucket, job_id, backlog, completion_tokens=completion_tokens, seed=j,
                     spacing=spacing)

    STOP.clear()
    errors: list = []

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=default_workers))
        auditors, cleanup = build_judges(job_ids)
        runners = [asyncio.ensure_future(a.run()) for a in auditors]
        waits, writes = [], []
        try:
            await _probe(BucketRecordStore(), seconds, waits, writes)
        finally:
            STOP.set()
            for r in runners:
                r.cancel()
            ended = await asyncio.gather(*runners, return_exceptions=True)
            cleanup()
        errors.extend(e for e in ended if not isinstance(e, asyncio.CancelledError))
        return waits, writes

    try:
        waits, writes = asyncio.run(scenario())
    finally:
        record_store.get_s3_client = saved
    verdicts = sum(1 for key in bucket.objects if "/verdicts/" in key)
    return {"waits": waits, "writes": writes, "verdicts": verdicts,
            "unclosed": bucket.unclosed, "errors": errors}


def summary(result):
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

    w, r = result["waits"], result["writes"]
    return (f"probes={len(w)} hop p50={statistics.median(w):.3f}s p99={q(w, .99):.3f}s "
            f"max={max(w):.3f}s | record write p50={statistics.median(r):.3f}s "
            f"p99={q(r, .99):.3f}s max={max(r):.3f}s | verdicts={result['verdicts']}")


PARAMS = AuditParams(q=0.15, probation_submissions=100, hold_seconds=4320.0,
                     ban_after_failures=1000)
