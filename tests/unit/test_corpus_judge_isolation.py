"""The judges never take the route's threads: on 2026-10-02 (12:00-12:25)
four judges catching up a backlog held the event loop's default executor with
drand relay races and forwards, and a trivial admit waited 6 s for a thread,
a store call 10-35 s. Real threads, real time, an in-memory bucket."""

from __future__ import annotations

import asyncio
import threading

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_record_store import BucketRecordStore
from reliquary.validator.corpus_auditor import CorpusAuditor
from reliquary.validator.corpus_judge_threads import JudgeThreads, judge_record_store
from reliquary.validator.corpus_miner_states import MinerStates
from tests.unit import corpus_route_load as load


def _judges(threads):
    def build(job_ids):
        store = (judge_record_store(threads) if threads is not None
                 else BucketRecordStore(max_pool_connections=64))
        lock = asyncio.Lock()
        auditors = []
        for job in job_ids:
            auditor = CorpusAuditor(
                job_id=job, records=store, model=None, tokenizer=None, proof=None,
                params=load.PARAMS, miner_states=MinerStates(store, job),
                beacon=load.blocking_beacon(1.5), round_at=load.round_at, gpu_lock=lock,
                threads=threads)
            auditor._judge_many = load.Gpu(tokens_per_second=2e6)
            auditors.append(auditor)
        return auditors, (threads.shutdown if threads is not None else (lambda: None))

    return build


def test_the_route_keeps_its_threads_while_four_judges_catch_up():
    result = load.run(_judges(JudgeThreads()), backlog=200, completion_tokens=4000,
                      seconds=15.0, spacing=0.3)
    waits, writes = sorted(result["waits"]), sorted(result["writes"])
    p99 = lambda xs: xs[min(len(xs) - 1, int(0.99 * len(xs)))]  # noqa: E731
    # Without the judges' own threads: hop p99 2.6 s, record write p99 3.5 s.
    assert p99(waits) < 0.2, (load.summary(result), result["errors"])
    assert p99(writes) < 1.2, (load.summary(result), result["errors"])
    # About one probe per 0.5 s (a record write and 50 ms); 14 in 40 s without.
    assert len(waits) > 20
    assert result["verdicts"] > 0 and result["unclosed"] == 0, (load.summary(result), result["errors"])


def test_without_their_own_threads_the_judges_starve_the_route():
    """The failure itself, reproduced: judges on the default executor."""
    result = load.run(_judges(None), backlog=200, completion_tokens=4000, seconds=15.0,
                      spacing=0.3)
    waits = sorted(result["waits"])
    assert waits[-1] > 1.0, load.summary(result)


def test_judge_work_runs_on_the_judges_threads():
    threads = JudgeThreads()
    seen = {}

    def beacon(round_number):
        seen["beacon"] = threading.current_thread().name
        return "ab" * 32

    def forward(records):
        seen["gpu"] = threading.current_thread().name
        return []

    auditor = CorpusAuditor(job_id="math-v1", records=None, model=None, tokenizer=None,
                            proof=None, beacon=beacon, threads=threads)
    auditor._judge_many = forward

    async def scenario():
        await auditor._randomness_for(5)
        await auditor._forward([])

    try:
        asyncio.run(scenario())
    finally:
        threads.shutdown()
    assert seen["beacon"].startswith("corpus-judge-drand")
    assert seen["gpu"].startswith("corpus-judge-gpu")


def test_a_judge_store_decodes_on_its_own_threads_and_bounds_its_reads(monkeypatch):
    threads = JudgeThreads()
    bucket = load.LatentR2()
    monkeypatch.setattr("reliquary.infrastructure.corpus_record_store.get_s3_client",
                        lambda **kw: bucket)
    load.seed_backlog(bucket, "math-v1", 40, completion_tokens=200, seed=1)
    store = judge_record_store(threads)
    store._max_reads = 4
    names, peak, live = set(), [0], [0]
    real_decode = job_store._decode

    def decode(body):
        names.add(threading.current_thread().name)
        return real_decode(body)

    monkeypatch.setattr("reliquary.infrastructure.corpus_record_store._decode", decode)
    real_get = bucket.get_object

    async def get_object(**kw):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        try:
            return await real_get(**kw)
        finally:
            live[0] -= 1

    bucket.get_object = get_object

    async def scenario():
        ids = await store.list_submission_ids("math-v1")
        return await asyncio.gather(*(store.read_submission("math-v1", sid) for sid in ids))

    try:
        records = asyncio.run(scenario())
    finally:
        threads.shutdown()
    assert len(records) == 40 and all(r["hotkey"] for r in records)
    assert names and all(n.startswith("corpus-judge-codec") for n in names)
    assert peak[0] <= 4
    assert bucket.unclosed == 0


def test_a_body_whose_read_fails_is_still_closed():
    closed = []

    class _Body:
        async def read(self):
            raise ConnectionError("reset mid-body")

        def close(self):
            closed.append(True)

    class _Client:
        async def get_object(self, **kw):
            return {"Body": _Body(), "ETag": '"e"'}

    class _Pool:
        def client(self):
            client = _Client()

            class _Ctx:
                async def __aenter__(self):
                    return client

                async def __aexit__(self, *exc):
                    return False

            return _Ctx()

    with pytest.raises(ConnectionError):
        asyncio.run(job_store._get("k", pool=_Pool(), bucket_name="b"))
    assert closed == [True]
