"""The grading policy: grade everything, replay passes and drawn failures,
two executors agreeing on a failed replay void the submission once."""

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState, drawn, replay_drawn
from reliquary.corpus.job import parse_job
from reliquary.environment.agentic_swe import SweSource
from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2
from reliquary.validator.corpus_grade_remote import GradeDecision
from reliquary.validator.corpus_grading import GRADE_SCHEMA, CorpusGrader
from tests.unit.test_corpus_job_episode import _episode, _manifest
from tests.unit.test_trajectory_parse import CALL, PROMPT, TERM, TEXT, R, build

SID = "a" * 64
TURNS = [([TEXT] * 8 + [CALL, TERM], ["a.py"]), ([TEXT] * 9 + [TERM], None)]


def _job(fraction=1.0, **episode):
    return parse_job(_manifest(prompt_count=3,
                               episode=_episode(replay_fraction_failed=fraction, **episode)))


def _record(stop="agent_completed"):
    tokens, spans = build(TURNS)
    return {"schema": RECORD_SCHEMA_V2, "hotkey": "5Hot", "prompt_index": 1, "received_at": 100.0,
            "completions": [{"prompt_tokens": PROMPT, "tokens": tokens, "final_diff": "D",
                             "stop": stop,
                             "turns": [{"start": s, "end": e, "proofs": ["A" * 8]} for s, e in spans]}]}


class _Records:
    def __init__(self, verdict=None, fail_grade_writes=0, stop="agent_completed"):
        self.submissions = {SID: _record(stop)}
        self.verdicts = {} if verdict is None else {SID: verdict}
        self.grades, self.regrades, self.voided = {}, {}, {}
        self._fail = fail_grade_writes

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def list_submission_ids(self, job_id):
        return sorted(self.submissions)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def write_grade(self, job_id, sid, document):
        if self._fail:
            self._fail -= 1
            raise OSError("bucket down")
        if sid in self.grades:
            return False
        self.grades[sid] = document
        return True

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def list_grade_ids(self, job_id):
        return sorted(self.grades)

    async def write_regrade(self, job_id, sid, document, generation=1):
        # regrades[sid] is the latest generation; generations keeps them all.
        self.generations = getattr(self, "generations", {})
        if (sid, generation) in self.generations or (generation == 1 and sid in self.regrades):
            return False
        self.generations[(sid, generation)] = document
        if generation >= (self.regrades.get(sid) or {}).get("generation", 1):
            self.regrades[sid] = document
        return True

    async def read_regrade(self, job_id, sid):
        return self.regrades.get(sid)

    async def read_voided(self, job_id, sid):
        return self.voided.get(sid)

    async def write_voided(self, job_id, sid, document):
        if sid in self.voided:
            return False
        self.voided[sid] = document
        return True


class _States:
    def __init__(self):
        self.states = {}

    async def update_many(self, changes):
        for hotkey, change in changes.items():
            self.states[hotkey] = change(self.states.get(hotkey, MinerState()))
        return dict(self.states)


class _Dispatcher:
    def __init__(self, grade, replay):
        self.items, self._answers = [], {"grade": grade, "replay": replay}

    async def decide(self, item):
        self.items.append(item)
        return self._answers[item["mode"]]


PASSED = GradeDecision("ok", {"status": "ok", "diff_applied": True, "tests_passed": True},
                       ("g0",), ("p0",))
FAILED = GradeDecision("ok", {"status": "ok", "diff_applied": True, "tests_passed": False},
                       ("g0",), ("p0",))
NOT_APPLIED = GradeDecision("ok", {"status": "ok", "diff_applied": False, "tests_passed": False},
                            ("g0",), ("p0",))
CERTIFIED = GradeDecision("ok", {"status": "ok", "replay_diff_equal": True, "observations_compared": 1,
                                 "observations_mismatched": []}, ("g1",), ("p1",))
_BAD_REPLAY = {"status": "ok", "replay_diff_equal": False, "observations_compared": 1,
               "observations_mismatched": []}
FORGED = GradeDecision("ok", _BAD_REPLAY, ("g0", "g1"), ("p0", "p1"))


def _grader(records, dispatcher, *, job=None, states=None, beacon=None):
    voided = []
    grader = CorpusGrader(job=job or _job(), records=records, dispatcher=dispatcher, renderer=R,
                          source=SweSource([("i0", "p"), ("repo__x.1", "fix it"), ("i2", "q")]),
                          params=AuditParams(), miner_states=states, beacon=beacon,
                          round_at=(lambda t: 7) if beacon else None, on_voided=lambda s, d: voided.append(s),
                          clock=lambda: 1000.0)
    return grader, voided


def test_a_passing_grade_is_replayed_and_certified():
    records, dispatcher = _Records(), _Dispatcher(PASSED, CERTIFIED)
    grader, _ = _grader(records, dispatcher)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["schema"] == GRADE_SCHEMA and doc["status"] == "ok"
    assert doc["graded_success"] is True and doc["replay_certified"] is True
    grade_item, replay_item = dispatcher.items
    assert grade_item["instance_id"] == "repo__x.1" and grade_item["final_diff"] == "D"
    assert replay_item["mode"] == "replay"
    assert replay_item["actions"] == [{"tool": "bash", "arguments": '{"command": "c0"}', "observation": "a.py"}]
    assert records.grades[SID] == doc


def test_a_failing_grade_is_replayed_only_when_drawn():
    records, dispatcher = _Records(), _Dispatcher(FAILED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(fraction=0.0))
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"] == {"drawn": False, "draw": {"fraction": 0.0, "drawn": False}}
    assert [i["mode"] for i in dispatcher.items] == ["grade"]


def test_the_draw_uses_the_drand_round_after_arrival():
    randomness = "c" * 64
    records, dispatcher = _Records(), _Dispatcher(FAILED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(fraction=0.5), beacon=lambda r: randomness)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["draw"]["round"] == 8
    assert doc["replay"]["drawn"] is replay_drawn(randomness, SID, 0.5)


def test_an_unpublished_round_leaves_the_submission_pending():
    records, dispatcher = _Records(), _Dispatcher(FAILED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(fraction=0.5), beacon=lambda r: None)
    assert asyncio.run(grader.grade_one(SID)) is None and records.grades == {}


def test_a_grade_waiting_for_its_round_is_not_graded_again():
    # Ruling P7: the rescan retries the draw, never the grade.
    published = {"round": None}
    records, dispatcher = _Records(), _Dispatcher(FAILED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(fraction=0.5),
                        beacon=lambda r: published["round"])

    async def scenario():
        assert await grader.grade_one(SID) is None
        assert await grader.grade_one(SID) is None
        published["round"] = "c" * 64
        return await grader.grade_one(SID)

    doc = asyncio.run(scenario())
    assert [i["mode"] for i in dispatcher.items].count("grade") == 1
    assert doc["grade"] == FAILED.result and doc["replay"]["draw"]["round"] == 8


def test_an_agreed_failed_replay_voids_and_escalates():
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(PASSED, FORGED), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["failed"] is True and doc["replay_certified"] is False
    assert records.voided[SID]["reason"] == "replay_failed" and voided == [SID]
    assert states.states["5Hot"].suspect_until is not None
    assert states.states["5Hot"].failure_ids == [SID]


def test_a_drawn_failing_grade_whose_replay_fails_by_agreement_is_voided():
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(FAILED, FORGED), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["graded_success"] is False and doc["replay"]["failed"] is True
    assert voided == [SID] and states.states["5Hot"].failure_ids == [SID]


@pytest.mark.parametrize("decision", [
    # One executor alone, whatever it says.
    GradeDecision("ok", _BAD_REPLAY, ("g0",), ("p0",)),
    # Two executors, one provider: one vote.
    GradeDecision("ok", _BAD_REPLAY, ("g0", "g1"), ("p0",)),
    # Agreement whose providers are not known.
    GradeDecision("ok", _BAD_REPLAY, ("g0", "g1")),
], ids=["one-executor", "one-provider", "no-providers"])
def test_a_failed_replay_without_two_providers_sanctions_nobody(decision):
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(PASSED, decision), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["failed"] is False and doc["replay"]["certified"] is False
    assert doc["replay"]["unconfirmed"] is True and doc["replay_certified"] is False
    assert records.voided == {} and states.states == {} and voided == []


@pytest.mark.parametrize("grade", [FAILED, NOT_APPLIED], ids=["tests-failed", "diff-not-applied"])
def test_a_failing_grade_by_one_undrawn_executor_never_sanctions(grade):
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(grade, FORGED), job=_job(fraction=0.0),
                             states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["status"] == "ok" and doc["graded_success"] is False
    assert doc["replay_certified"] is False and doc["replay"]["drawn"] is False
    assert records.voided == {} and states.states == {} and voided == []


@pytest.mark.parametrize("status", ["timeout", "error", "ungradeable"])
def test_a_grade_without_a_decision_judges_nobody_and_certifies_nothing(status):
    records, states = _Records(), _States()
    dispatcher = _Dispatcher(GradeDecision(status, None, ()), FORGED)
    grader, voided = _grader(records, dispatcher, job=_job(fraction=0.0), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["status"] == status                         # its own outcome, on record
    assert doc["graded_success"] is False and doc["replay"]["drawn"] is False
    assert doc["replay_certified"] is False
    assert [i["mode"] for i in dispatcher.items] == ["grade"]
    assert records.voided == {} and states.states == {} and voided == []
    assert records.grades[SID] == doc


@pytest.mark.parametrize("status", ["timeout", "error", "ungradeable", "disputed", "unjudgeable"])
def test_a_grade_that_is_not_a_clean_success_gets_the_failing_replay_draw(status):
    """F2 (ruling P23 a): a forger who makes the grade fail to decide (its
    patch kills the box, its tests sleep) is still replayed when drawn, and a
    replay two providers fail sanctions it."""
    records, states = _Records(), _States()
    grade = GradeDecision(status, {"status": "box_lost"} if status == "unjudgeable" else None,
                          ("g0", "g1"), ("p0", "p1"))
    dispatcher = _Dispatcher(grade, FORGED)
    grader, voided = _grader(records, dispatcher, job=_job(fraction=1.0), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert [i["mode"] for i in dispatcher.items] == ["grade", "replay"]
    assert doc["status"] == status and doc["replay"]["failed"] is True
    assert records.voided[SID]["reason"] == "replay_failed" and voided == [SID]
    assert states.states["5Hot"].failure_ids == [SID]


UNJUDGEABLE_REPLAY = GradeDecision("unjudgeable", {"status": "box_lost", "detail": "kill -9 1"},
                                   ("g0", "g1"), ("p0", "p1"))


@pytest.mark.parametrize("grade", [PASSED, FAILED], ids=["passing-grade", "failing-grade"])
def test_an_agreed_unjudgeable_replay_voids_unpaid_without_escalation(grade):
    """F2 (ruling P23 c): `kill -9 1`, `rm -rf .git`, `sleep 99999` as the last
    action: two providers agree the replay could not judge the trajectory.
    Unpaid, never certified, and no sanction (honest infra noise must not ban)."""
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(grade, UNJUDGEABLE_REPLAY), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["status"] == "unjudgeable" and doc["replay"]["unjudgeable"] is True
    assert doc["replay"]["failed"] is False and doc["replay_certified"] is False
    void = records.voided[SID]
    assert void["reason"] == "replay_unjudgeable" and void["stage"] == "replay"
    assert void["providers"] == ["p0", "p1"] and voided == [SID]
    assert states.states == {}                              # no suspect, no ban


def test_an_unjudgeable_replay_one_provider_claims_voids_nothing():
    records, states = _Records(), _States()
    lone = GradeDecision("unjudgeable", {"status": "box_lost"}, ("g0", "g1"), ("p0",))
    grader, voided = _grader(records, _Dispatcher(PASSED, lone), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["unjudgeable"] is False and doc["replay_certified"] is False
    assert records.voided == {} and voided == [] and states.states == {}


def test_an_agreed_unjudgeable_grade_voids_unpaid_when_the_replay_does_not_sanction():
    records, states = _Records(), _States()
    grade = GradeDecision("unjudgeable", {"status": "box_timeout"}, ("g0", "g2"), ("p0", "p2"))
    for job, replay in ((_job(fraction=0.0), CERTIFIED), (_job(fraction=1.0), CERTIFIED)):
        records = _Records()
        grader, voided = _grader(records, _Dispatcher(grade, replay), job=job, states=states)
        doc = asyncio.run(grader.grade_one(SID))
        assert doc["graded_success"] is False and doc["status"] == "unjudgeable"
        assert records.voided[SID]["reason"] == "replay_unjudgeable"
        assert records.voided[SID]["stage"] == "grade" and voided == [SID]
    assert states.states == {}


@pytest.mark.parametrize("status", ["timeout", "error", "ungradeable", "disputed"])
def test_a_replay_without_a_decision_judges_nobody(status):
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(PASSED, GradeDecision(status, None, ("g0", "g1"))),
                             states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["status"] == status
    assert doc["replay"]["certified"] is False and doc["replay"]["failed"] is False
    assert doc["replay_certified"] is False and doc["graded_success"] is True
    assert records.voided == {} and states.states == {} and voided == []


def test_an_audit_failure_is_not_graded():
    records, dispatcher = _Records(verdict={"passed": False}), _Dispatcher(PASSED, CERTIFIED)
    grader, _ = _grader(records, dispatcher)
    assert asyncio.run(grader.grade_one(SID))["status"] == "audit_failed"
    assert dispatcher.items == []


def test_the_actions_are_parsed_under_the_job_turn_limit():
    # A max_turns stop is only accepted against the job's own limit.
    records, dispatcher = _Records(stop="max_turns"), _Dispatcher(PASSED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(max_turns=2))
    assert asyncio.run(grader.grade_one(SID))["replay_certified"] is True
    records, dispatcher = _Records(stop="max_turns"), _Dispatcher(PASSED, CERTIFIED)
    grader, _ = _grader(records, dispatcher, job=_job(max_turns=3))
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["status"] == "unparseable" and doc["reason"] == "bad_stop"
    assert dispatcher.items == [] and records.voided == {}


def test_a_confirmed_failure_is_counted_once_across_a_restart():
    records, states = _Records(fail_grade_writes=1), _States()
    first, _ = _grader(records, _Dispatcher(PASSED, FORGED), states=states)
    with pytest.raises(OSError):
        asyncio.run(first.grade_one(SID))              # crashed after the void, before the grade
    second, voided = _grader(records, _Dispatcher(PASSED, FORGED), states=states)
    asyncio.run(second.grade_one(SID))
    assert states.states["5Hot"].failure_ids == [SID]
    assert len(states.states["5Hot"].confirmed_failures) == 1
    assert list(records.voided) == [SID] and voided == []      # written once, announced once


def test_ready_lists_only_graded_submissions():
    records = _Records()
    records.grades["b" * 64] = {}
    grader, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    assert asyncio.run(grader.ready([SID, "b" * 64])) == {"b" * 64}


def test_ready_learns_of_a_grade_this_process_wrote():
    records = _Records()
    grader, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))

    async def scenario():
        before = await grader.ready([SID])
        await grader.grade_one(SID)
        return before, await grader.ready([SID])

    assert asyncio.run(scenario()) == (set(), {SID})


def test_the_replay_draw_is_independent_of_the_audit_draw():
    sids = [hashlib.sha256(bytes([k])).hexdigest() for k in range(64)]
    randomness = "d" * 64
    assert [replay_drawn(randomness, s, 0.5) for s in sids] != [drawn(randomness, s, 0.5) for s in sids]
    assert replay_drawn(randomness, sids[0], 1.0) and not replay_drawn(randomness, sids[0], 0.0)


# -- quarantine: what an executor decided alone is graded again ---------------


class _Sequence:
    """A dispatcher answering each mode from its own list, in order."""

    def __init__(self, grades, replays):
        self.items, self._answers = [], {"grade": list(grades), "replay": list(replays)}

    async def decide(self, item):
        self.items.append(item)
        return self._answers[item["mode"]].pop(0)


def test_a_quarantined_executor_alone_on_a_decision_is_regraded():
    records = _Records()
    regraded_pass = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    dispatcher = _Sequence([PASSED, regraded_pass], [CERTIFIED, CERTIFIED])
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        await grader.grade_one(SID)                        # g0 graded alone, g1 replayed alone
        return await grader.regrade_executor("g0"), await grader.regrade_executor("g0")

    first, again = asyncio.run(scenario())
    assert first == [SID] and again == []
    assert records.regrades[SID]["graded_by"] == ["g2"]
    assert records.grades[SID]["graded_by"] == ["g0"]      # the grade stays, superseded


def test_the_lone_replayer_is_tracked_too():
    records = _Records()
    grader, _ = _grader(records, _Sequence([PASSED, PASSED], [CERTIFIED, CERTIFIED]))

    async def scenario():
        await grader.grade_one(SID)
        return await grader.regrade_executor("g1")

    assert asyncio.run(scenario()) == [SID] and SID in records.regrades


def test_an_agreed_decision_is_not_regraded():
    records, states = _Records(), _States()
    agreed = GradeDecision("ok", PASSED.result, ("g0", "g3"), ("p0", "p3"))
    grader, _ = _grader(records, _Dispatcher(agreed, FORGED), states=states)

    async def scenario():
        await grader.grade_one(SID)
        return await grader.regrade_executor("g0"), await grader.regrade_executor("g1")

    assert asyncio.run(scenario()) == ([], [])
    assert records.regrades == {}


def test_grades_written_before_a_restart_are_regraded_too():
    records = _Records()
    first, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    asyncio.run(first.grade_one(SID))
    second, _ = _grader(records, _Dispatcher(GradeDecision("ok", PASSED.result, ("g2",), ("p2",)),
                                             CERTIFIED))

    async def scenario():
        await second.ready([SID])                          # seeded from the store
        return await second.regrade_executor("g0")

    assert asyncio.run(scenario()) == [SID]
    assert records.regrades[SID]["graded_by"] == ["g2"]


def test_a_cached_grade_of_a_quarantined_executor_is_dropped():
    published = {"round": None}
    records = _Records()
    regraded = GradeDecision("ok", FAILED.result, ("g2",), ("p2",))
    dispatcher = _Sequence([FAILED, regraded], [CERTIFIED])
    grader, _ = _grader(records, dispatcher, job=_job(fraction=0.5),
                        beacon=lambda r: published["round"])

    async def scenario():
        assert await grader.grade_one(SID) is None         # g0's grade waits for its round
        await grader.regrade_executor("g0")
        published["round"] = "c" * 64
        return await grader.grade_one(SID)

    doc = asyncio.run(scenario())
    assert doc["graded_by"] == ["g2"]


# -- the payment gate ------------------------------------------------------------


def test_settlement_waits_for_the_grade():
    from reliquary.validator.corpus_settlement import CorpusSettler
    from tests.unit.test_corpus_settlement import _Archives
    from tests.unit.test_corpus_settlement import _Records as _SettleRecords

    verdict = {"passed": True, "hotkey": "5Hot", "token_count": 10}
    records = _SettleRecords({SID: verdict, "b" * 64: dict(verdict)})
    graded = {"b" * 64}

    async def ready(ids):
        return {s for s in ids if s in graded}

    settler = CorpusSettler(task_id="t", job_id="j", cap=0.1, records=records,
                            archives=_Archives(other_max=None), ready=ready,
                            advance_every_seconds=0.0)
    asyncio.run(settler.settle_once())
    assert records.state["settled"] == ["b" * 64]
    graded.add(SID)
    asyncio.run(settler.settle_once())
    assert sorted(records.state["settled"]) == sorted([SID, "b" * 64])


def test_a_void_landing_before_payment_is_never_paid():
    from reliquary.validator.corpus_settlement import CorpusSettler
    from tests.unit.test_corpus_settlement import _Archives
    from tests.unit.test_corpus_settlement import _Records as _SettleRecords

    class _Voiding(_SettleRecords):
        async def list_voided_ids(self, job_id):
            return [SID]

    records = _Voiding({SID: {"passed": True, "hotkey": "5Hot", "token_count": 10},
                        "b" * 64: {"passed": True, "hotkey": "5Other", "token_count": 10}})

    async def ready(ids):
        return set(ids)

    archives = _Archives(other_max=None)
    settler = CorpusSettler(task_id="t", job_id="j", cap=0.1, records=records,
                            archives=archives, ready=ready)
    asyncio.run(settler.settle_once())
    (archive,) = archives.written.values()
    assert archive["rewards_by_hotkey"] == {"5Other": pytest.approx(0.1)}


# -- the control's wiring --------------------------------------------------------


def test_judge_jobs_follow_the_mode():
    from reliquary.validator.corpus_validator import judge_jobs

    calls = []

    class _Runner:
        def __init__(self, name):
            self.name = name

        def run(self):
            calls.append(self.name)
            return self.name

    w = SimpleNamespace(entry=SimpleNamespace(task_id="t"), auditor=_Runner("auditor"),
                        settler=object(), grader=_Runner("grader"))
    full = judge_jobs(w, intake_only=False, settle_every_seconds=60.0,
                      settle=lambda task_id, settler, every: "settle")
    assert full == ["auditor", "settle", "grader"]
    assert judge_jobs(w, intake_only=True, settle_every_seconds=60.0,
                      settle=lambda *a: "settle") == ["grader"]
    w.grader = None
    assert judge_jobs(w, intake_only=False, settle_every_seconds=60.0,
                      settle=lambda *a: "settle") == ["auditor", "settle"]
    w.judge_link = object()
    assert judge_jobs(w, intake_only=False, settle_every_seconds=60.0,
                      settle=lambda *a: "settle") == []


def _wired(job, intake=True):
    return SimpleNamespace(
        entry=SimpleNamespace(task_id="t", params={}), job=job,
        episode_intake=SimpleNamespace(renderer=R, source=SweSource([("i0", "p")])) if intake else None)


def test_an_episode_job_gets_a_grader_on_the_pinned_env():
    from reliquary.validator.corpus_validator import wire_job_grader

    job = _job()
    dispatcher = SimpleNamespace(env_pin=(job.episode.env.package, job.episode.env.version))
    w = _wired(job)
    wire_job_grader(w, records=_Records(), judge_records=_Records(), dispatcher=dispatcher)
    assert isinstance(w.grader, CorpusGrader)
    graders = w.grader
    wire_job_grader(w, records=_Records(), judge_records=_Records(), dispatcher=dispatcher)
    assert w.grader is graders                             # once per job


def test_a_job_without_an_episode_gets_no_grader():
    from reliquary.validator.corpus_validator import wire_job_grader

    w = _wired(parse_job(_manifest(with_episode=False)), intake=False)
    wire_job_grader(w, records=_Records(), judge_records=_Records(), dispatcher=None)
    assert getattr(w, "grader", None) is None


def test_an_episode_job_without_its_grade_dispatcher_is_refused():
    from reliquary.validator.corpus_validator import wire_job_grader

    job = _job()
    with pytest.raises(ValueError, match="restart"):
        wire_job_grader(_wired(job), records=_Records(), judge_records=_Records(), dispatcher=None)
    other = SimpleNamespace(env_pin=(job.episode.env.package, "f" * 40))
    with pytest.raises(ValueError, match="grades"):
        wire_job_grader(_wired(job), records=_Records(), judge_records=_Records(), dispatcher=other)


def test_intake_only_refuses_any_audit():
    from reliquary.validator.corpus_validator import run_corpus_validator

    for kwargs in ({"remote_audit": True}, {"split": object()}):
        with pytest.raises(RuntimeError, match="intake-only serves no audit"):
            asyncio.run(run_corpus_validator(
                wallet=None, netuid=1, signer_client=None, http_host="h", http_port=1,
                set_weights=False, intake_only=True, **kwargs))


def test_the_vocab_size_is_read_from_the_config(tmp_path):
    from reliquary.validator.corpus_validator import _config_vocab_size

    (tmp_path / "config.json").write_text('{"text_config": {"vocab_size": 248320}}')
    assert _config_vocab_size(tmp_path) == 248320
    (tmp_path / "config.json").write_text('{"vocab_size": 151936}')
    assert _config_vocab_size(tmp_path) == 151936
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="vocab_size"):
        _config_vocab_size(tmp_path)


@pytest.mark.parametrize("flag", ["1", "0"])
def test_the_cli_passes_intake_only(monkeypatch, flag):
    from reliquary.cli import main
    from reliquary.validator import corpus_validator

    seen = []

    async def fake(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(corpus_validator, "run_corpus_validator", fake)
    monkeypatch.setenv("RELIQUARY_CORPUS_INTAKE_ONLY", flag)
    for jobs in ([("e", 0.1)], [("e", 0.1), ("f", 0.2)]):
        asyncio.run(main._run_corpus(jobs=jobs, wallet=None, netuid=1, signer_client=None,
                                     http_host="h", http_port=1, set_weights=False))
    assert [k["intake_only"] for k in seen] == [flag == "1"] * 2


def test_the_cli_refuses_intake_only_with_the_split(monkeypatch):
    from reliquary.cli import main

    monkeypatch.setenv("RELIQUARY_CORPUS_INTAKE_ONLY", "1")
    monkeypatch.setenv("RELIQUARY_CORPUS_SPLIT", "1")
    with pytest.raises(RuntimeError, match="intake-only"):
        asyncio.run(main._run_corpus(jobs=[("e", 0.1)], wallet=None, netuid=1, signer_client=None,
                                     http_host="h", http_port=1, set_weights=False))


def test_the_judge_pays_only_graded_submissions_and_hears_of_voids():
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_validator import wire_job_judge

    grader, _ = _grader(_Records(), _Dispatcher(PASSED, CERTIFIED))
    w = SimpleNamespace(entry=SimpleNamespace(task_id="t", params={}), job=_job(), cap=0.1,
                        stats=JobStats(), grader=grader)
    wire_job_judge(w, records=_Records(), judge_records=_Records(),
                   judge_threads=SimpleNamespace(codec=None, gpu=None), archives=object(),
                   proof=toploc_proof(ACTIVE_PROTOCOL_PROFILE), model=None, tokenizer=None)
    assert w.settler._ready == grader.ready
    assert grader.on_voided == w.miners.voided


def test_a_failed_regrade_is_retried_by_the_rescan():
    records = _Records()

    class _Flaky(_Sequence):
        async def decide(self, item):
            if len(self.items) == 2:                       # the regrade's first dispatch
                self.items.append(None)
                raise OSError("dispatcher down")
            return await super().decide(item)

    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    grader, _ = _grader(records, _Flaky([PASSED, regraded], [CERTIFIED, CERTIFIED]))
    grader._rescan = 3600.0

    async def scenario():
        await grader.grade_one(SID)
        assert await grader.regrade_executor("g0") == [SID]
        assert SID not in records.regrades
        await grader.rescan_once()
        await grader.drain()

    asyncio.run(scenario())
    assert records.regrades[SID]["graded_by"] == ["g2"]


def test_the_void_document_is_the_auditors():
    from reliquary.validator import corpus_auditor, corpus_grading

    assert corpus_grading.VOIDED_SCHEMA == corpus_auditor.VOIDED_SCHEMA


# -- fix round 1: payment held while a quarantined executor's grades are redone --


class _Gated(_Sequence):
    """Replays wait for ``release`` once ``hold_replays`` is set; a test waits
    for them with ``until_waiting``, never by counting loop turns (a grade
    parses its trajectory in a worker thread, so turns prove nothing)."""

    def __init__(self, grades, replays):
        super().__init__(grades, replays)
        self.release = None
        self.hold_replays = False
        self.waiting = 0
        self._arrived = None

    async def decide(self, item):
        if item["mode"] == "replay" and self.hold_replays:
            self.waiting += 1
            if self._arrived is not None:
                self._arrived.set()
            try:
                await self.release.wait()
            finally:
                self.waiting -= 1
        return await super().decide(item)

    async def until_waiting(self, count):
        while self.waiting < count:
            self._arrived = asyncio.Event()
            await self._arrived.wait()


def test_a_lone_certified_submission_is_not_paid_while_regraded():
    records = _Records()
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    dispatcher = _Gated([PASSED, regraded], [CERTIFIED, CERTIFIED])
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        await grader.grade_one(SID)
        assert await grader.ready([SID]) == {SID}
        grader.hold_executor("g0")                         # the quarantine, synchronously
        assert await grader.ready([SID]) == set()
        dispatcher.release, dispatcher.hold_replays = asyncio.Event(), True
        regrade = asyncio.ensure_future(grader.regrade_executor("g0"))
        await dispatcher.until_waiting(1)                  # the regrade's replay, in flight
        assert not regrade.done() and await grader.ready([SID]) == set()
        dispatcher.release.set()
        assert await regrade == [SID]
        return await grader.ready([SID])

    assert asyncio.run(scenario()) == {SID}
    assert records.regrades[SID]["replay_certified"] is True


def test_a_regrade_that_voids_leaves_the_void_to_the_settler():
    records, states = _Records(), _States()
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    grader, voided = _grader(records, _Sequence([PASSED, regraded], [CERTIFIED, FORGED]),
                             states=states)

    async def scenario():
        await grader.grade_one(SID)
        await grader.regrade_executor("g1")                # g1 certified the replay alone

    asyncio.run(scenario())
    assert records.voided[SID]["reason"] == "replay_failed" and voided == [SID]
    assert records.regrades[SID]["replay"]["failed"] is True
    assert states.states["5Hot"].failure_ids == [SID]


def test_a_failed_regrade_holds_payment_until_it_succeeds():
    records = _Records()

    class _Flaky(_Sequence):
        async def decide(self, item):
            if len(self.items) == 2:
                self.items.append(None)
                raise OSError("dispatcher down")
            return await super().decide(item)

    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    grader, _ = _grader(records, _Flaky([PASSED, regraded], [CERTIFIED, CERTIFIED]))
    grader._rescan = 3600.0

    async def scenario():
        await grader.grade_one(SID)
        await grader.regrade_executor("g0")
        held = await grader.ready([SID])
        await grader.rescan_once()
        await grader.drain()
        return held, await grader.ready([SID])

    assert asyncio.run(scenario()) == (set(), {SID})


def test_grades_from_before_boot_are_held_while_they_are_indexed():
    records = _Records()
    first, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    asyncio.run(first.grade_one(SID))
    second, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))

    async def scenario():
        assert await second.ready([SID]) == {SID}
        second.hold_executor("g9")                         # who knows what g9 decided alone
        held = await second.ready([SID])
        assert await second.regrade_executor("g9") == []
        return held, await second.ready([SID])

    assert asyncio.run(scenario()) == (set(), {SID})


def test_regrades_run_side_by_side():
    other = "b" * 64
    records = _Records()
    records.submissions[other] = _record()
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    dispatcher = _Gated([PASSED, PASSED, regraded, regraded], [CERTIFIED] * 4)
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        await grader.grade_one(SID)
        await grader.grade_one(other)
        dispatcher.release, dispatcher.hold_replays = asyncio.Event(), True
        regrade = asyncio.ensure_future(grader.regrade_executor("g0"))
        # Both regrades are past their grade and waiting on their replay at once.
        await dispatcher.until_waiting(2)
        replays_waiting = dispatcher.waiting
        dispatcher.release.set()
        await regrade
        return replays_waiting

    assert asyncio.run(scenario()) == 2
    assert set(records.regrades) == {SID, other}


def test_an_existing_void_is_the_grade_never_a_new_replay():
    records, states = _Records(), _States()
    records.voided[SID] = {"schema": "reliquary/corpus-voided/v1", "submission_id": SID,
                           "hotkey": "5Hot", "reason": "replay_failed", "voided_at": 900.0,
                           "graded_by": ["g0", "g1"], "providers": ["p0", "p1"],
                           "replay": {"replay_diff_equal": False, "observations_compared": 1,
                                      "observations_mismatched": [], "allowed": 5}}
    dispatcher = _Dispatcher(PASSED, CERTIFIED)
    grader, voided = _grader(records, dispatcher, states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert [i["mode"] for i in dispatcher.items] == ["grade"]
    assert doc["replay"]["failed"] is True and doc["replay"]["from_void"] is True
    assert doc["replay"]["graded_by"] == ["g0", "g1"] and doc["replay_certified"] is False
    assert voided == [] and states.states == {}           # counted when it was written


def test_an_existing_regrade_is_not_overwritten_and_says_so(monkeypatch):
    from reliquary.validator import corpus_grading

    warned = []
    monkeypatch.setattr(corpus_grading.logger, "warning",
                        lambda *a, **k: warned.append(a[0] % a[1:]))
    records = _Records()
    records.regrades[SID] = {"graded_by": ["g7"]}
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    grader, _ = _grader(records, _Sequence([PASSED, regraded], [CERTIFIED, CERTIFIED]))

    async def scenario():
        await grader.grade_one(SID)
        await grader.regrade_executor("g0")

    asyncio.run(scenario())
    assert records.regrades[SID] == {"graded_by": ["g7"]}
    # Round 3: the existing regrade (by an unquarantined executor) stands; no new dispatch.
    assert [i["mode"] for i in grader._dispatcher.items] == ["grade", "replay"]


def test_the_grader_reports_its_backlog():
    records = _Records()
    records.submissions["b" * 64] = _record()
    grader, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    grader._rescan = 3600.0

    async def scenario():
        await grader.grade_one(SID)
        grader.enqueue = lambda sid: None                  # the rescan only counts here
        await grader.rescan_once()
        return grader.status()

    status = asyncio.run(scenario())
    assert status["graded"] == 1 and status["ungraded"] == 1 and status["regrading"] == 0
    assert "dispatcher_waiting" in status


def test_the_split_refuses_an_episode_job_at_startup(monkeypatch):
    from reliquary.infrastructure import corpus_job_store
    from reliquary.validator.corpus_validator import SPLIT_EPISODE_REFUSAL, run_corpus_validator

    class _Store:
        async def read_job(self, job_id):
            return _job(), None

    monkeypatch.setattr(corpus_job_store, "BucketJobStore", _Store)
    entry = SimpleNamespace(task_id="corpus-swe", job_id="swe-agentic-v1", params={})
    with pytest.raises(RuntimeError) as refused:
        asyncio.run(run_corpus_validator(
            wallet=None, netuid=1, signer_client=None, http_host="h", http_port=1,
            set_weights=False, entry=entry, cap=0.1, split=SimpleNamespace(links={})))
    assert str(refused.value) == SPLIT_EPISODE_REFUSAL


# -- fix round 2: a decision finished after its executor's quarantine ------------


@pytest.mark.parametrize("lone", ["g0", "g1"], ids=["lone-grader", "lone-replayer"])
def test_a_lone_decision_finished_after_the_quarantine_is_regraded(lone):
    records = _Records()
    first = PASSED if lone == "g0" else GradeDecision("ok", PASSED.result, ("g0", "g3"), ("p0", "p3"))
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    recertified = GradeDecision("ok", CERTIFIED.result, ("g4",), ("p4",))
    dispatcher = _Gated([first, regraded], [CERTIFIED, recertified])
    dispatcher.quarantined = set()
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        dispatcher.release, dispatcher.hold_replays = asyncio.Event(), True
        grading = asyncio.ensure_future(grader.grade_one(SID))
        await dispatcher.until_waiting(1)                  # the replay is in flight
        assert not grading.done()
        dispatcher.quarantined.add(lone)                   # quarantined meanwhile
        grader.hold_executor(lone)
        assert await grader.regrade_executor(lone) == []   # nothing written yet to regrade
        dispatcher.hold_replays = False
        dispatcher.release.set()
        await grading
        held = await grader.ready([SID])                   # held from the grade write on
        await grader.drain()                               # the regrade it scheduled
        return held, await grader.ready([SID])

    assert asyncio.run(scenario()) == (set(), {SID})
    assert SID in records.regrades and lone not in records.regrades[SID]["graded_by"]


def test_an_executor_regrade_is_launched_once_while_it_runs():
    records = _Records()
    grader, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    calls, gate = [], asyncio.Event()

    async def blocked(executor_id):
        calls.append(executor_id)
        await gate.wait()
        return []

    grader.regrade_executor = blocked
    grader._held_executors.add("g9")

    async def scenario():
        await grader.rescan_once()
        await asyncio.sleep(0)                             # the first regrade starts, blocked
        await grader.rescan_once()                         # it is still running: not again
        await grader.rescan_once()
        gate.set()
        await grader.drain()

    asyncio.run(scenario())
    assert calls == ["g9"]


def test_the_status_shows_a_job_wide_hold():
    records = _Records()
    first, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    asyncio.run(first.grade_one(SID))
    second, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))

    async def scenario():
        await second.ready([SID])
        second.hold_executor("g9")
        return second.status()

    status = asyncio.run(scenario())
    assert status["held_executors"] == ["g9"] and status["unindexed"] == 1


def test_the_quarantine_listener_logs_a_grader_failure(monkeypatch):
    from reliquary.validator import corpus_validator

    logged = []
    monkeypatch.setattr(corpus_validator.logger, "error",
                        lambda *a, **k: logged.append(a[0] % a[1:]))

    class _Good:
        async def regrade_executor(self, eid):
            return ["x"]

    class _Bad:
        _job = SimpleNamespace(job_id="swe-v1")

        async def regrade_executor(self, eid):
            raise OSError("bucket down")

    result = asyncio.run(corpus_validator.regrade_everywhere([_Good(), _Bad()], "g0"))
    assert result == [["x"]]
    assert any("g0" in line and "bucket down" in line for line in logged)


# -- fix round 3: a quarantine survives a restart ----------------------------------


def test_a_restart_before_the_regrade_write_holds_until_regraded():
    records = _Records()
    first, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    asyncio.run(first.grade_one(SID))                      # g0 graded alone; then g0 quarantined
    first.hold_executor("g0")                              # ... and the process dies here
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    second, _ = _grader(records, _Sequence([regraded], [GradeDecision("ok", CERTIFIED.result,
                                                                      ("g4",), ("p4",))]))
    second._dispatcher.quarantined = {"g0"}                # loaded from the registry at boot
    second.hold_executor("g0")                             # what the control does at wiring

    async def scenario():
        held = await second.ready([SID])
        assert await second.regrade_executor("g0") == [SID]
        return held, await second.ready([SID])

    assert asyncio.run(scenario()) == (set(), {SID})
    assert records.regrades[SID]["graded_by"] == ["g2"]


def test_a_restart_after_the_regrade_write_releases_without_regrading():
    records = _Records()
    regraded = GradeDecision("ok", PASSED.result, ("g2",), ("p2",))
    first, _ = _grader(records, _Sequence([PASSED, regraded], [CERTIFIED, CERTIFIED]))

    async def before():
        await first.grade_one(SID)
        await first.regrade_executor("g0")

    asyncio.run(before())                                  # regrade written; the process dies
    second, _ = _grader(records, _Sequence([], []))        # any dispatch would fail
    second._dispatcher.quarantined = {"g0"}
    second.hold_executor("g0")

    async def scenario():
        assert await second.regrade_executor("g0") == [SID]
        return await second.ready([SID])

    assert asyncio.run(scenario()) == {SID}
    assert second._dispatcher.items == []
    assert records.regrades[SID]["graded_by"] == ["g2"]


def test_a_regrade_that_voids_before_a_restart_is_voided_after_it():
    # The regrade landed (failed by agreement) but its void did not: written on release.
    records, states = _Records(), _States()
    first, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))
    asyncio.run(first.grade_one(SID))
    records.regrades[SID] = {"schema": "reliquary/corpus-grade/v1", "submission_id": SID,
                             "hotkey": "5Hot", "graded_by": ["g2"], "generation": 1,
                             "replay": {"failed": True, "graded_by": ["g2", "g3"],
                                        "providers": ["p2", "p3"], "replay_diff_equal": False,
                                        "observations_compared": 1,
                                        "observations_mismatched": [], "allowed": 5}}
    second, voided = _grader(records, _Sequence([], []), states=states)
    second._dispatcher.quarantined = {"g0"}
    second.hold_executor("g0")
    asyncio.run(second.regrade_executor("g0"))
    assert records.voided[SID]["reason"] == "replay_failed" and voided == [SID]
    assert states.states["5Hot"].failure_ids == [SID]


def test_a_regrade_decided_alone_by_a_later_quarantined_executor_is_redone_once():
    records = _Records()
    by = lambda e: GradeDecision("ok", PASSED.result, (e,), (f"p{e}",))  # noqa: E731
    cert = lambda e: GradeDecision("ok", CERTIFIED.result, (e,), (f"p{e}",))  # noqa: E731
    dispatcher = _Sequence([by("g0"), by("g2"), by("g5")], [cert("g1"), cert("g3"), cert("g6")])
    dispatcher.quarantined = set()
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        await grader.grade_one(SID)
        dispatcher.quarantined.add("g0")
        await grader.regrade_executor("g0")               # generation 1, by g2 alone
        dispatcher.quarantined.add("g2")
        first = await grader.regrade_executor("g2")        # generation 2, by g5 alone
        again = await grader.regrade_executor("g2")        # never twice for the same executor
        await grader.drain()
        return first, again, await grader.ready([SID])

    first, again, ready = asyncio.run(scenario())
    assert first == [SID] and again == [] and ready == {SID}
    assert records.regrades[SID]["generation"] == 2 and records.regrades[SID]["graded_by"] == ["g5"]
    assert [i["mode"] for i in dispatcher.items].count("grade") == 3


def test_regrades_stop_at_the_generation_cap():
    from reliquary.validator.corpus_grading import MAX_REGRADE_GENERATIONS

    records = _Records()
    names = [f"g{k}" for k in range(2 * MAX_REGRADE_GENERATIONS + 4)]
    grades = [GradeDecision("ok", PASSED.result, (e,), (f"p{e}",)) for e in names[0::2]]
    replays = [GradeDecision("ok", CERTIFIED.result, (e,), (f"p{e}",)) for e in names[1::2]]
    dispatcher = _Sequence(grades, replays)
    dispatcher.quarantined = set()
    grader, _ = _grader(records, dispatcher)

    async def scenario():
        await grader.grade_one(SID)
        for k in range(MAX_REGRADE_GENERATIONS + 1):
            executor = names[2 * k]
            dispatcher.quarantined.add(executor)
            await grader.regrade_executor(executor)
        return await grader.ready([SID])

    # Ruling P22: past the cap it resolves regrade_exhausted, unpaid (a void
    # without escalation) and no longer held: one stuck submission never
    # freezes a task's pay.
    assert asyncio.run(scenario()) == {SID}
    assert records.regrades[SID]["generation"] == MAX_REGRADE_GENERATIONS
    void = records.voided[SID]
    assert void["reason"] == "regrade_exhausted" and void["hotkey"] == "5Hot"
    assert grader.status()["regrading"] == 0


def test_an_exhausted_regrade_logs_an_operator_error_and_never_escalates(monkeypatch):
    from reliquary.validator import corpus_grading
    from reliquary.validator.corpus_grading import MAX_REGRADE_GENERATIONS

    errors = []
    monkeypatch.setattr(corpus_grading.logger, "error", lambda *a, **k: errors.append(a[0] % a[1:]))
    records, states = _Records(), _States()
    names = [f"g{k}" for k in range(2 * MAX_REGRADE_GENERATIONS + 4)]
    grades = [GradeDecision("ok", PASSED.result, (e,), (f"p{e}",)) for e in names[0::2]]
    replays = [GradeDecision("ok", CERTIFIED.result, (e,), (f"p{e}",)) for e in names[1::2]]
    dispatcher = _Sequence(grades, replays)
    dispatcher.quarantined = set()
    grader, voided = _grader(records, dispatcher, states=states)

    async def scenario():
        await grader.grade_one(SID)
        for k in range(MAX_REGRADE_GENERATIONS + 1):
            dispatcher.quarantined.add(names[2 * k])
            await grader.regrade_executor(names[2 * k])
        grader._listed = set()
        return grader.oldest_unready_received_at()

    assert asyncio.run(scenario()) is None                 # no longer holds any period
    assert any("regrade_exhausted" in e for e in errors)
    assert states.states == {} and voided == [SID]


def test_a_grader_wired_after_a_quarantine_holds_its_executor():
    from reliquary.validator.corpus_validator import wire_job_grader

    job = _job()
    dispatcher = SimpleNamespace(env_pin=(job.episode.env.package, job.episode.env.version),
                                 quarantined={"g0"})
    w = _wired(job)
    wire_job_grader(w, records=_Records(), judge_records=_Records(), dispatcher=dispatcher)
    assert w.grader.status()["held_executors"] == ["g0"]  # regraded by its first rescan


def test_the_grader_holds_its_background_tasks_until_done():
    records = _Records()
    grader, _ = _grader(records, _Dispatcher(PASSED, CERTIFIED))

    async def scenario():
        grader.enqueue(SID)
        held = len(grader._tasks)                          # strongly referenced, not just weak
        await grader.drain()
        return held, len(grader._tasks)

    assert asyncio.run(scenario()) == (1, 0)
    assert SID in records.grades


@pytest.mark.parametrize("decision", [
    GradeDecision("uncertified", _BAD_REPLAY, ("g0",), ("p0",)),
    GradeDecision("uncertified", {"status": "box_lost"}, ("g0", "g1"), ("p0", "p1")),
], ids=["lone-failure", "box-lost-and-failure"])
def test_a_replay_no_vote_certifies_voids_unpaid_without_a_sanction(decision):
    """Ruling P27: a forger whose trajectory kills boxes at random no longer
    collects through `disputed`: no vote certifying it is enough to withhold
    pay (never to sanction)."""
    records, states = _Records(), _States()
    grader, voided = _grader(records, _Dispatcher(PASSED, decision), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert doc["replay"]["status"] == "uncertified" and doc["replay_certified"] is False
    assert doc["replay"]["failed"] is False
    assert records.voided[SID]["reason"] == "replay_unjudgeable"
    assert records.voided[SID]["stage"] == "replay" and voided == [SID]
    assert states.states == {}



def test_a_disputed_grade_is_always_replayed():
    """Ruling P27: a grade the executors could not agree on (a box that died
    against a pass) is no clean success; its replay is drawn whatever the
    failing fraction."""
    records, states = _Records(), _States()
    dispatcher = _Dispatcher(GradeDecision("disputed", None, ("g0", "g1")), FORGED)
    grader, voided = _grader(records, dispatcher, job=_job(fraction=0.0), states=states)
    doc = asyncio.run(grader.grade_one(SID))
    assert [i["mode"] for i in dispatcher.items] == ["grade", "replay"]
    assert doc["replay"]["drawn"] is True and doc["replay"]["failed"] is True
    assert records.voided[SID]["reason"] == "replay_failed"
