"""What the corpus settlers share: the cap split by verified tokens, and the
fixtures other tests settle a job with (the period settler; the window settler
is gone, design 2026-10-03 section 8)."""

import asyncio
import os

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
from reliquary.validator.corpus_settlement import R2Archives, rewards_for

GENESIS = 1_000_000.0
# Every fixture verdict is received in period WORK; the settler's clock stands
# far enough past it that the period is closed.
WORK = 3
RECEIVED = GENESIS + WORK * cp.PERIOD_SECONDS + 10.0
NOW = GENESIS + (WORK + 2) * cp.PERIOD_SECONDS + 600.0


def test_rewards_split_the_cap_by_passed_tokens():
    verdicts = [
        {"hotkey": "A", "token_count": 300, "passed": True},
        {"hotkey": "B", "token_count": 100, "passed": True},
        {"hotkey": "C", "token_count": 900, "passed": False},
    ]
    assert rewards_for(verdicts, 0.1) == pytest.approx({"A": 0.075, "B": 0.025})


def test_no_passed_token_pays_nobody():
    assert rewards_for([{"hotkey": "C", "token_count": 9, "passed": False}], 0.1) == {}


class _Records:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.state, self.etag = {}, None
        self.fail_state_writes_after = None

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_settlement(self, job_id):
        return dict(self.state), self.etag

    async def write_settlement(self, job_id, state, etag):
        if self.fail_state_writes_after == 0:
            raise OSError("crash")
        if self.fail_state_writes_after is not None:
            self.fail_state_writes_after -= 1
        assert etag == self.etag
        self.state, self.etag = dict(state), f"e{len(str(state))}"
        return self.etag


class _Archives:
    """Period archives in memory; ``written`` maps a work period to its archive."""

    def __init__(self, *_ignored, **_also_ignored):
        self.written = {}
        self.docs = {}

    async def write(self, task_id, work, entry, document):
        assert (work, entry) not in self.docs, "an archive is never overwritten"
        self.docs[(work, entry)] = document
        self.written[work] = document

    async def read(self, task_id, work, entry):
        return self.docs.get((work, entry))

    async def list(self, task_id):
        return sorted(self.docs)


def _settler(records, archives, now=NOW, *, task_id="corpus-math", job_id="math-v1", cap=0.1,
             **kwargs):
    return CorpusPeriodSettler(task_id=task_id, job_id=job_id, cap=cap,
                               records=records, archives=archives,
                               oldest_pending=lambda: None, genesis=lambda: GENESIS,
                               clock=lambda: now, **kwargs)


def _v(hk, n, ok=True, received=RECEIVED):
    return {"hotkey": hk, "token_count": n, "passed": ok, "received_at": received}


def test_a_settlement_writes_one_archive_and_marks_its_verdicts():
    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30)})
    archives = _Archives()
    assert asyncio.run(_settler(records, archives).settle_once()) == WORK
    archive = archives.written[WORK]
    assert archive["work_period"] == WORK and archive["cap"] == 0.1
    assert archive["rewards_by_hotkey"] == pytest.approx({"A": 0.025, "B": 0.075})
    assert sorted(records.state["settled"]) == ["1" * 64, "2" * 64]
    assert records.state["pending"] is None


def test_settled_verdicts_are_never_paid_again():
    records = _Records({"1" * 64: _v("A", 10)})
    archives = _Archives()
    asyncio.run(_settler(records, archives).settle_once())
    assert asyncio.run(_settler(records, archives, now=NOW + 9000).settle_once()) is None
    assert list(archives.written) == [WORK]


def test_a_crash_after_the_archive_finishes_the_same_archive_not_a_second_one():
    records = _Records({"1" * 64: _v("A", 10)})
    archives = _Archives()
    records.fail_state_writes_after = 1  # the pending write lands, the final one crashes
    with pytest.raises(OSError):
        asyncio.run(_settler(records, archives).settle_once())
    records.fail_state_writes_after = None
    assert asyncio.run(_settler(records, archives).settle_once()) == WORK
    assert list(archives.docs) == [(WORK, cp.period_of(NOW, GENESIS) + 1)]
    assert records.state["settled"] == ["1" * 64]


def test_an_all_failed_period_writes_no_archive_but_marks_ids_settled():
    records = _Records({"1" * 64: _v("C", 9, ok=False)})
    archives = _Archives()
    assert asyncio.run(_settler(records, archives).settle_once()) is None
    assert archives.written == {}
    assert records.state["settled"] == ["1" * 64]
    assert records.state["pending"] is None


def test_the_guard_refuses_a_task_this_process_does_not_serve(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-a,corpus-b")
    guard = R2Archives(served=lambda: {"corpus-hot"})
    guard.refuse_unserved("corpus-a")
    guard.refuse_unserved("corpus-hot")
    with pytest.raises(RuntimeError):
        guard.refuse_unserved("corpus-other")
    only = R2Archives(served=lambda: {"corpus-hot"}, served_only=True)
    only.refuse_unserved("corpus-hot")
    with pytest.raises(RuntimeError):
        only.refuse_unserved("corpus-a")
    monkeypatch.delenv("RELIQUARY_TASK_ID")
    with pytest.raises(RuntimeError):
        R2Archives().refuse_unserved("corpus-a")
    assert "RELIQUARY_TASK_ID" not in os.environ
