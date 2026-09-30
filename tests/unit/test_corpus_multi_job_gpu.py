"""Two jobs' auditors on one loaded model: one forward pass at a time, FIFO."""

from __future__ import annotations

import asyncio
import threading
import time

from reliquary.validator.corpus_auditor import CorpusAuditor
from tests.unit.test_corpus_auditor import _Records


class _Probe:
    """Stands in for `_judge_many`: records who ran, and how many ran at once."""

    def __init__(self, seconds: float = 0.02) -> None:
        self._seconds = seconds
        self._guard = threading.Lock()
        self.live = 0
        self.max_live = 0
        self.order: list[str] = []

    def judge(self, name: str):
        def _judge_many(records):
            with self._guard:
                self.live += 1
                self.max_live = max(self.max_live, self.live)
                self.order.append(name)
            time.sleep(self._seconds)
            with self._guard:
                self.live -= 1
            return [{"passed": True, "reason": None} for _ in records]
        return _judge_many


def _auditor(job_id, records, lock, probe):
    auditor = CorpusAuditor(job_id=job_id, records=records, model=None, tokenizer=None,
                            proof=None, gpu_lock=lock)
    auditor._judge_many = probe.judge(job_id)
    return auditor


def test_two_jobs_never_run_a_forward_pass_at_once():
    probe = _Probe()

    async def main():
        lock = asyncio.Lock()
        a = _auditor("job-a", _Records({}), lock, probe)
        b = _auditor("job-b", _Records({}), lock, probe)
        await asyncio.gather(*(x._audit_outcomes([{}]) for x in (a, b) for _ in range(4)))

    asyncio.run(main())
    assert probe.max_live == 1
    assert sorted(probe.order) == ["job-a"] * 4 + ["job-b"] * 4


def test_a_backlogged_job_does_not_starve_the_other():
    """The lock is FIFO (asyncio.Lock wakes waiters in arrival order), so a job
    that arrives while the other drains a backlog runs after at most one of
    the backlog's passes, not after all of them."""
    probe = _Probe()

    async def main():
        lock = asyncio.Lock()
        a = _auditor("job-a", _Records({}), lock, probe)
        b = _auditor("job-b", _Records({}), lock, probe)

        async def backlog():
            for _ in range(6):
                await a._audit_outcomes([{}])

        drain = asyncio.create_task(backlog())
        while not probe.order:
            await asyncio.sleep(0.001)
        await b._audit_outcomes([{}])
        await drain

    asyncio.run(main())
    assert probe.max_live == 1
    assert probe.order.index("job-b") <= 2


def test_no_lock_keeps_a_single_auditor_unchanged():
    probe = _Probe(seconds=0.0)
    auditor = CorpusAuditor(job_id="job-a", records=_Records({}), model=None,
                            tokenizer=None, proof=None)
    auditor._judge_many = probe.judge("job-a")

    assert asyncio.run(auditor._audit_outcomes([{}])) == [{"passed": True, "reason": None}]


def test_the_solo_retry_path_takes_the_lock_too():
    """A failed batch is retried one record at a time; each retry is its own
    forward pass and must queue behind the other job like any other."""
    probe = _Probe()
    calls = {"n": 0}

    async def main():
        lock = asyncio.Lock()
        a = _auditor("job-a", _Records({}), lock, probe)
        b = _auditor("job-b", _Records({}), lock, probe)
        inner = a._judge_many

        def flaky(records):
            calls["n"] += 1
            if len(records) > 1:
                raise RuntimeError("batch failed")
            return inner(records)

        a._judge_many = flaky
        await asyncio.gather(a._audit_outcomes([{}, {}, {}]),
                             *(b._audit_outcomes([{}]) for _ in range(3)))

    asyncio.run(main())
    assert probe.max_live == 1
    assert probe.order.count("job-a") == 3
