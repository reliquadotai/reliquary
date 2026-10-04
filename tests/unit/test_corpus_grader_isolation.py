"""The grader in a process that also serves submit routes (the split front):
its trajectory parses and drand calls run on its own bounded threads, never on
the loop's default executor the routes' ledger turns use, and on a renderer of
its own, never the intake's."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from reliquary.corpus.audit_policy import AuditParams
from reliquary.environment.agentic_swe import SweSource
from reliquary.validator import corpus_grading
from reliquary.validator.corpus_grading import CorpusGrader
from tests.unit.test_corpus_grading import CERTIFIED, PASSED, _Dispatcher, _job, _record
from tests.unit.test_trajectory_parse import R

N = 48
PARSE_SECONDS = 0.4


class _ManyRecords:
    def __init__(self, n):
        self.submissions = {"%064x" % (k + 1): _record() for k in range(n)}
        self.grades = {}

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def list_submission_ids(self, job_id):
        return sorted(self.submissions)

    async def read_verdict(self, job_id, sid):
        return None

    async def write_grade(self, job_id, sid, document):
        self.grades[sid] = document
        return True

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def list_grade_ids(self, job_id):
        return sorted(self.grades)

    async def read_regrade(self, job_id, sid):
        return None

    async def read_voided(self, job_id, sid):
        return None


def _slow_parse(monkeypatch):
    """A parse that holds its thread (as a 60k-token one does), recording
    where it ran and how many ran at once."""
    seen = SimpleNamespace(threads=set(), now=0, peak=0, lock=threading.Lock())
    real = corpus_grading.parse_trajectory

    def parse(*args, **kwargs):
        with seen.lock:
            seen.now += 1
            seen.peak = max(seen.peak, seen.now)
            seen.threads.add(threading.current_thread().name)
        try:
            time.sleep(PARSE_SECONDS)
            return real(*args, **kwargs)
        finally:
            with seen.lock:
                seen.now -= 1

    monkeypatch.setattr(corpus_grading, "parse_trajectory", parse)
    return seen


def test_a_grading_burst_never_delays_the_default_executor(monkeypatch):
    """The routes' ledger turns run on the default executor: while N slow
    parses are in flight, a turn there starts at once."""
    seen = _slow_parse(monkeypatch)
    records = _ManyRecords(N)
    parse_pool = ThreadPoolExecutor(2, thread_name_prefix="corpus-grade-parse")
    grader = CorpusGrader(job=_job(), records=records, dispatcher=_Dispatcher(PASSED, CERTIFIED),
                          renderer=R, source=SweSource([("i0", "p"), ("repo__x.1", "fix"),
                                                        ("i2", "q")]),
                          params=AuditParams(), clock=lambda: 1000.0,
                          parse_executor=parse_pool)

    async def scenario():
        grading = asyncio.gather(*(grader.grade_one(sid) for sid in records.submissions))
        await asyncio.sleep(0.2)                       # the burst is under way
        delays = []
        for _ in range(5):
            started = time.monotonic()
            await asyncio.to_thread(lambda: None)      # a route's ledger turn
            delays.append(time.monotonic() - started)
            await asyncio.sleep(0.05)
        await grading
        return delays

    delays = asyncio.run(scenario())
    parse_pool.shutdown()
    assert max(delays) < 0.1, delays
    assert seen.threads and all(t.startswith("corpus-grade-parse") for t in seen.threads), seen.threads
    assert seen.peak <= 2                               # bounded parse concurrency
    assert len(records.grades) == N


def test_the_parse_concurrency_is_bounded_without_an_executor(monkeypatch):
    seen = _slow_parse(monkeypatch)
    records = _ManyRecords(8)
    grader = CorpusGrader(job=_job(), records=records, dispatcher=_Dispatcher(PASSED, CERTIFIED),
                          renderer=R, source=SweSource([("i0", "p"), ("repo__x.1", "fix"),
                                                        ("i2", "q")]),
                          params=AuditParams(), clock=lambda: 1000.0)

    async def scenario():
        await asyncio.gather(*(grader.grade_one(sid) for sid in records.submissions))

    asyncio.run(scenario())
    assert seen.peak <= corpus_grading.GRADE_PARSE_CONCURRENCY


def test_the_replay_draw_reads_drand_on_its_executor_never_the_loop():
    on = {}
    loop_thread = []

    def beacon(round_number):
        on["beacon"] = threading.current_thread().name
        return "ab" * 32

    def round_at(t):
        on["round_at"] = threading.current_thread().name
        return 7

    pool = ThreadPoolExecutor(1, thread_name_prefix="corpus-judge-drand")
    grader = CorpusGrader(job=_job(fraction=0.5), records=_ManyRecords(1),
                          dispatcher=_Dispatcher(PASSED, CERTIFIED), renderer=R,
                          source=SweSource([("i0", "p")]), params=AuditParams(),
                          beacon=beacon, round_at=round_at, clock=lambda: 1000.0,
                          beacon_executor=pool)

    async def scenario():
        loop_thread.append(threading.current_thread().name)
        return await grader._draw("a" * 64, 100.0)

    drawn = asyncio.run(scenario())
    pool.shutdown()
    assert drawn["round"] == 8
    assert on["beacon"].startswith("corpus-judge-drand")
    assert on["round_at"].startswith("corpus-judge-drand")   # the chain lookup too


def test_an_unknown_round_still_draws_a_replay_off_the_loop():
    def round_at(t):
        raise RuntimeError("drand chain genesis/period not resolved yet")

    pool = ThreadPoolExecutor(1)
    grader = CorpusGrader(job=_job(fraction=0.5), records=_ManyRecords(1),
                          dispatcher=_Dispatcher(PASSED, CERTIFIED), renderer=R,
                          source=SweSource([("i0", "p")]), params=AuditParams(),
                          beacon=lambda r: "ab" * 32, round_at=round_at, clock=lambda: 1000.0,
                          beacon_executor=pool)
    drawn = asyncio.run(grader._draw("a" * 64, 100.0))
    pool.shutdown()
    assert drawn == {"fraction": 0.5, "drawn": True, "why": "round unknown"}


def test_the_grader_parses_with_its_own_renderer_not_the_intakes():
    from reliquary.validator.corpus_validator import wire_job_grader

    job = _job()
    intake_renderer, grade_renderer = object(), object()
    w = SimpleNamespace(entry=SimpleNamespace(task_id="t", params={}), job=job,
                        episode_intake=SimpleNamespace(renderer=intake_renderer,
                                                       source=SweSource([("i0", "p")])),
                        grade_renderer=grade_renderer)
    pool = ThreadPoolExecutor(1)
    dispatcher = SimpleNamespace(env_pin=(job.episode.env.package, job.episode.env.version))
    wire_job_grader(w, records=_ManyRecords(0), judge_records=_ManyRecords(0),
                    dispatcher=dispatcher, parse_executor=pool, beacon_executor=pool)
    pool.shutdown()
    assert w.grader._renderer is grade_renderer
    assert w.grader._parse_executor is pool and w.grader._beacon_executor is pool
