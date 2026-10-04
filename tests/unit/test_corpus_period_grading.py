"""A period-settled episode job: a period closes only once everything received
in it is graded (ruling P21), and nothing ungraded is ever paid."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams
from reliquary.environment.agentic_swe import SweSource
from reliquary.validator.corpus_grade_remote import GradeDecision
from reliquary.validator.corpus_grading import CorpusGrader
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler, oldest_of
from tests.unit.test_corpus_grading import (
    CERTIFIED, FORGED, PASSED, SID, _Gated, _job, _record, _Sequence,
)
from tests.unit.test_corpus_grading import _Records as _GradeRecords
from tests.unit.test_corpus_period_settlement import GENESIS, Archives, at, verdict
from tests.unit.test_trajectory_parse import R

OTHER = "b" * 64


class _Records(_GradeRecords):
    """The grader's records and the period settler's, in one store."""

    def __init__(self, received=at(3), **kwargs):
        super().__init__(**kwargs)
        self.submissions[SID]["received_at"] = received
        self.verdicts = {SID: verdict(SID, "5Hot", 10, received)}
        self.state = {}

    def add(self, sid, hotkey, received):
        self.submissions[sid] = {**_record(), "hotkey": hotkey, "received_at": received}
        self.verdicts[sid] = verdict(sid, hotkey, 10, received)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return dict(self.state), "etag"

    async def write_settlement(self, job_id, state, etag):
        self.state = dict(state)
        return "etag"


class _BySubmission:
    def __init__(self, answers):
        self.answers = answers

    async def decide(self, item):
        return self.answers[(item["submission_id"], item["mode"])]


def _grader(records, dispatcher, now):
    return CorpusGrader(job=_job(), records=records, dispatcher=dispatcher, renderer=R,
                        source=SweSource([("i0", "p"), ("repo__x.1", "fix it"), ("i2", "q")]),
                        params=AuditParams(), clock=lambda: now[0])


def _settler(records, archives, now, grader, auditor=lambda: None):
    return CorpusPeriodSettler(task_id="t", job_id="j", cap=0.1, records=records,
                               archives=archives,
                               oldest_pending=oldest_of(auditor,
                                                        grader.oldest_unready_received_at),
                               ready=grader.ready, genesis=lambda: GENESIS,
                               clock=lambda: now[0])


def test_a_period_with_an_ungraded_submission_closes_after_its_grade():
    records, archives, now = _Records(), Archives(), [at(6)]
    dispatcher = _Gated([PASSED], [CERTIFIED])
    dispatcher.release, dispatcher.hold_replays = asyncio.Event(), True
    grader = _grader(records, dispatcher, now)
    settler = _settler(records, archives, now, grader)

    async def scenario():
        await grader.rescan_once()                     # listed; its replay is in flight
        await dispatcher.until_waiting(1)
        held = await settler.settle_once()
        assert archives.docs == {}
        dispatcher.release.set()
        await grader.drain()
        return held, await settler.settle_once()

    assert asyncio.run(scenario()) == (None, 3)
    assert archives.docs[(3, 7)]["rewards_by_hotkey"] == {"5Hot": pytest.approx(0.1)}


def test_a_voided_episode_in_a_closed_period_is_not_paid():
    records, archives, now = _Records(), Archives(), [at(6)]
    records.add(OTHER, "5Other", at(3, 50))
    dispatcher = _BySubmission({(SID, "grade"): PASSED, (SID, "replay"): FORGED,
                                (OTHER, "grade"): PASSED, (OTHER, "replay"): CERTIFIED})
    grader = _grader(records, dispatcher, now)
    settler = _settler(records, archives, now, grader)

    async def scenario():
        await grader.rescan_once()
        await grader.drain()
        return await settler.settle_once()

    assert asyncio.run(scenario()) == 3
    assert records.voided[SID]["reason"] == "replay_failed"
    assert archives.docs[(3, 7)]["rewards_by_hotkey"] == {"5Other": pytest.approx(0.1)}


def test_a_submission_held_for_a_quarantine_regrade_keeps_its_period_open():
    records, archives, now = _Records(), Archives(), [at(6)]
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    grader = _grader(records, _Sequence([PASSED, regraded], [CERTIFIED, CERTIFIED]), now)
    settler = _settler(records, archives, now, grader)

    async def scenario():
        await grader.rescan_once()
        await grader.drain()                           # graded by g0 alone
        grader.hold_executor("g0")                     # g0 quarantined
        held = await settler.settle_once()
        assert archives.docs == {}
        assert await grader.regrade_executor("g0") == [SID]
        return held, await settler.settle_once()

    assert asyncio.run(scenario()) == (None, 3)


def test_a_restart_keeps_periods_open_until_the_grader_has_listed():
    records, archives, now = _Records(), Archives(), [at(6)]
    first = _grader(records, _Sequence([PASSED], [CERTIFIED]), now)
    asyncio.run(first.grade_one(SID))
    second = _grader(records, _Sequence([], []), now)
    settler = _settler(records, archives, now, second)
    with pytest.raises(LookupError):
        second.oldest_unready_received_at()

    async def scenario():
        held = await settler.settle_once()             # seeded by ready(), never listed
        await second.rescan_once()
        return held, await settler.settle_once()

    assert asyncio.run(scenario()) == (None, 3)


def test_a_quarantine_at_boot_keeps_periods_open_until_the_old_grades_are_indexed():
    records, archives, now = _Records(), Archives(), [at(6)]
    first = _grader(records, _Sequence([PASSED], [CERTIFIED]), now)
    asyncio.run(first.grade_one(SID))
    second = _grader(records, _Sequence([], []), now)
    settler = _settler(records, archives, now, second)

    async def scenario():
        await second.rescan_once()
        second.hold_executor("g9")                     # from the registry, at wiring
        with pytest.raises(LookupError):
            second.oldest_unready_received_at()
        held = await settler.settle_once()
        assert await second.regrade_executor("g9") == []
        return held, await settler.settle_once()

    assert asyncio.run(scenario()) == (None, 3)


def test_a_live_submission_counts_from_its_admission_until_its_record_is_read():
    records, now = _Records(), [at(6)]
    del records.submissions[SID]
    dispatcher = _Gated([PASSED], [CERTIFIED])
    dispatcher.release, dispatcher.hold_replays = asyncio.Event(), True
    grader = _grader(records, dispatcher, now)

    async def scenario():
        await grader.rescan_once()
        assert grader.oldest_unready_received_at() is None
        records.submissions[SID] = {**_record(), "received_at": at(3)}
        grader.enqueue(SID)                            # accepted live: not read yet
        admitted = grader.oldest_unready_received_at()
        await dispatcher.until_waiting(1)
        read = grader.oldest_unready_received_at()
        dispatcher.release.set()
        await grader.drain()
        return admitted, read, grader.oldest_unready_received_at()

    admitted, read, done = asyncio.run(scenario())
    assert admitted <= at(6) - 420.0
    assert (read, done) == (at(3), None)


def test_a_listed_submission_with_no_readable_record_holds_then_is_given_up():
    from reliquary.validator.corpus_grading import UNKNOWN_ARRIVAL_GIVE_UP_SECONDS

    class _Ghost(_Records):
        async def list_submission_ids(self, job_id):
            return sorted({*self.submissions, OTHER})

    records, now = _Ghost(), [at(6)]
    grader = _grader(records, _Sequence([PASSED], [CERTIFIED]), now)

    async def scenario():
        await grader.rescan_once()
        await grader.drain()
        return grader.oldest_unready_received_at()

    assert asyncio.run(scenario()) == float("-inf")
    now[0] += UNKNOWN_ARRIVAL_GIVE_UP_SECONDS + 1
    assert grader.oldest_unready_received_at() is None


def test_the_period_settler_never_pays_what_is_not_ready():
    records, archives = _Records(), Archives()
    graded: set[str] = set()

    async def ready(ids):
        return {sid for sid in ids if sid in graded}

    settler = CorpusPeriodSettler(task_id="t", job_id="j", cap=0.1, records=records,
                                  archives=archives, oldest_pending=lambda: None, ready=ready,
                                  genesis=lambda: GENESIS, clock=lambda: at(6))
    assert asyncio.run(settler.settle_once()) is None and archives.docs == {}
    graded.add(SID)
    assert asyncio.run(settler.settle_once()) == 3


def test_oldest_of_takes_the_earliest_and_propagates_a_lookup_error():
    def unknown():
        raise LookupError("not yet")

    assert oldest_of(lambda: None, lambda: None)() is None
    assert oldest_of(lambda: 5.0, lambda: None)() == 5.0
    assert oldest_of(lambda: 5.0, lambda: 3.0)() == 3.0
    with pytest.raises(LookupError):
        oldest_of(lambda: 5.0, unknown)()


# -- the wiring ------------------------------------------------------------------


def _wire(grader=None):
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_validator import wire_job_judge

    w = SimpleNamespace(entry=SimpleNamespace(task_id="t", params={"settlement": "period-ema-v1"}),
                        job=_job(), cap=0.1, stats=JobStats())
    if grader is not None:
        w.grader = grader
    wire_job_judge(w, records=_Records(), judge_records=_Records(),
                   judge_threads=SimpleNamespace(codec=None, gpu=None), archives=object(),
                   proof=toploc_proof(ACTIVE_PROTOCOL_PROFILE), model=None, tokenizer=None)
    return w


def test_a_period_task_without_a_grader_waits_for_its_auditor_alone():
    w = _wire()
    assert isinstance(w.settler, CorpusPeriodSettler) and w.settler._ready is None
    w.auditor.oldest_pending_received_at = lambda: 5.0
    assert w.settler._oldest_pending() == 5.0


def test_a_period_paid_episode_job_waits_for_its_auditor_and_its_grader():
    grader = _grader(_Records(), _Sequence([], []), [at(6)])
    w = _wire(grader)
    assert isinstance(w.settler, CorpusPeriodSettler) and w.settler._ready == grader.ready
    w.auditor.oldest_pending_received_at = lambda: 5.0
    grader.oldest_unready_received_at = lambda: 3.0
    assert w.settler._oldest_pending() == 3.0
    grader.oldest_unready_received_at = lambda: None
    assert w.settler._oldest_pending() == 5.0
    assert grader.on_voided == w.miners.voided
