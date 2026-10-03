"""Grade leases on the control: agreement replaces the local recheck."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from reliquary.validator import corpus_grade_remote
from reliquary.validator.corpus_audit_remote import ExecutorDirectory, LeaseRefused, token_sha256
from reliquary.validator.corpus_grade_protocol import (
    MAX_ACTIONS,
    MAX_ARGUMENT_CHARS,
    MAX_OBSERVATION_CHARS,
    MAX_TOOL_NAME_CHARS,
    GradeLease,
    GradeResult,
)
from reliquary.validator.corpus_grade_remote import (
    RemoteGradeDispatcher,
    build_grade_executor_router,
    decision_key,
    replay_certified,
)
from tests.unit.test_corpus_audit_remote import _Clock

PACKAGE, VERSION = "reliquary-swe", "b" * 40
TOKENS = {f"g{k}": f"token-{k}-" + "x" * 30 for k in range(4)}
SID = "a" * 64


def _docs(scope="grade"):
    return [dict(executor_id=eid, token_sha256=token_sha256(token), model_id=PACKAGE,
                 model_revision=VERSION, expires_at=1e12, status="active", scope=scope)
            for eid, token in TOKENS.items()]


class _Rng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def _dispatcher(recheck=1.0, clock=None, quarantined=None):
    clock = clock or _Clock()

    async def listed():
        return _docs()

    async def quarantine(eid, reason):
        if quarantined is not None:
            quarantined.append(eid)

    directory = ExecutorDirectory(model_id=PACKAGE, model_revision=VERSION, list_documents=listed,
                                  clock=clock, scope="grade")
    return RemoteGradeDispatcher(directory=directory, env_package=PACKAGE, env_version=VERSION,
                                 quarantine=quarantine, clock=clock, rng=_Rng(recheck))


def _item(mode="grade", **kw):
    return {"submission_id": SID, "task_index": 1, "instance_id": "repo__x.1", "mode": mode,
            "final_diff": "d", "actions": [], **kw}


# Every executor result echoes the leased submission id (corpus_grade_executor._work).
PASS = {"status": "ok", "submission_id": SID, "diff_applied": True, "tests_passed": True}
REPLAY_OK = {"status": "ok", "submission_id": SID, "replay_diff_equal": True,
             "observations_compared": 10, "observations_mismatched": [1]}
REPLAY_BAD = {"status": "ok", "submission_id": SID, "replay_diff_equal": False,
              "observations_compared": 10, "observations_mismatched": []}
TIMEOUT = {"status": "timeout", "submission_id": SID}
ERROR = {"status": "error", "submission_id": SID}


def _result(result):
    return GradeResult.model_validate({"results": [result]})


def _answer(d, eid, result):
    lease = d.claim(eid)
    assert lease is not None, f"{eid} got no lease"
    return d.result(eid, lease["lease_id"], _result(result))


def test_decision_keys():
    assert decision_key("grade", PASS) == (True, True)
    assert decision_key("replay", REPLAY_OK) == (True, True)
    assert decision_key("replay", REPLAY_BAD) == (False, False)
    over = {**REPLAY_OK, "observations_mismatched": list(range(6))}
    assert decision_key("replay", over) == (True, False)
    assert replay_certified(REPLAY_OK) and not replay_certified(over)


async def test_an_undrawn_pass_needs_one_executor():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    assert lease["env"] == {"package": PACKAGE, "version": VERSION}
    assert lease["items"][0]["instance_id"] == "repo__x.1"
    GradeLease.model_validate(lease)                      # what the executor will parse
    d.result("g0", lease["lease_id"], _result(PASS))
    assert (await decision).graded_by == ("g0",)


async def test_a_drawn_recheck_needs_a_second_executor():
    d = _dispatcher(recheck=0.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    assert not decision.done() and d.claim("g0") is None      # never the same executor twice
    _answer(d, "g1", PASS)
    assert (await decision).graded_by == ("g0", "g1")


async def test_a_failure_with_one_executor_waits_for_a_second():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    await asyncio.sleep(0)
    assert not decision.done()
    _answer(d, "g1", REPLAY_BAD)
    got = await decision
    assert got.status == "ok" and got.result["replay_diff_equal"] is False
    assert got.graded_by == ("g0", "g1")


async def test_a_disagreement_is_arbitrated_by_a_third_executor():
    quarantined = []
    d = _dispatcher(recheck=1.0, quarantined=quarantined)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)                 # a lying executor fails an honest miner
    _answer(d, "g1", REPLAY_OK)
    assert not decision.done()
    _answer(d, "g2", REPLAY_OK)
    got = await decision
    await asyncio.sleep(0)
    assert got.result["replay_diff_equal"] is True and got.graded_by == ("g1", "g2")
    assert "g0" in d.quarantined and quarantined == ["g0"]


async def test_three_results_without_two_agreeing_judge_nobody():
    d = _dispatcher(recheck=0.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    _answer(d, "g1", {**PASS, "tests_passed": False})
    _answer(d, "g2", {**PASS, "diff_applied": False, "tests_passed": False})
    got = await decision
    assert got.status == "error" and got.result is None
    assert not d.quarantined


async def test_a_quarantined_executor_loses_its_pending_vote():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)                 # waits for a second executor
    await d.quarantine("g0", "caught elsewhere")
    _answer(d, "g1", REPLAY_OK)                  # alone now: a pass needs one, undrawn
    got = await decision
    assert got.graded_by == ("g1",) and got.result["replay_diff_equal"] is True


async def test_a_quarantined_vote_on_a_leased_item_is_dropped_too():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    lease = d.claim("g1")                        # g1 works while g0 is caught
    await d.quarantine("g0", "caught elsewhere")
    assert d.claim("g2") is None                 # never leased twice at once
    d.result("g1", lease["lease_id"], _result(REPLAY_OK))
    assert (await decision).graded_by == ("g1",)


async def test_timeouts_go_to_other_executors_then_resolve_unjudged():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", TIMEOUT)
    assert d.claim("g0") is None
    _answer(d, "g1", TIMEOUT)
    assert (await decision).status == "timeout"


async def test_three_errors_resolve_as_an_error():
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    for eid in ("g0", "g1"):
        assert _answer(d, eid, ERROR) == "requeued"
        assert not decision.done()
    _answer(d, "g2", ERROR)
    got = await decision
    assert got.status == "error" and not d.quarantined


async def test_an_expired_lease_strikes_and_three_strikes_quarantine():
    clock = _Clock()
    quarantined = []
    d = _dispatcher(recheck=1.0, clock=clock, quarantined=quarantined)
    for _ in range(3):
        asyncio.ensure_future(d.decide(_item()))
        await asyncio.sleep(0)
        assert d.claim("g3") is not None
        clock.now += 10_000
        await d.sweep()
    assert "g3" in d.quarantined and quarantined == ["g3"]


async def test_an_item_whose_leases_keep_expiring_resolves_as_a_timeout():
    clock = _Clock()
    d = _dispatcher(recheck=1.0, clock=clock)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    for eid in ("g0", "g1"):
        assert d.claim(eid) is not None
        clock.now += 10_000
        await d.sweep()
    got = await decision
    assert got.status == "timeout" and got.result is None


async def test_a_late_result_is_refused_and_counts_as_an_expiry():
    clock = _Clock()
    d = _dispatcher(recheck=1.0, clock=clock)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    clock.now += 10_000
    with pytest.raises(LeaseRefused) as refused:
        d.result("g0", lease["lease_id"], _result(REPLAY_BAD))
    assert refused.value.detail == "lease_expired"
    assert d.claim("g0") is None and not decision.done()
    _answer(d, "g1", REPLAY_OK)
    assert (await decision).graded_by == ("g1",)


@pytest.mark.parametrize("echo", [None, "c" * 64])
async def test_a_result_for_another_submission_is_refused_and_struck(echo):
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    forged = {**REPLAY_BAD, "submission_id": echo}
    with pytest.raises(LeaseRefused) as refused:
        d.result("g0", lease["lease_id"], _result(forged))
    assert refused.value.status == 422 and refused.value.detail == "result_does_not_fit_the_lease"
    assert d._strikes["g0"] == 1 and d.claim("g0") is None and not decision.done()
    _answer(d, "g1", REPLAY_OK)
    assert (await decision).graded_by == ("g1",)


@pytest.mark.parametrize("mode,answer", [
    ("grade", {"status": "ok", "submission_id": SID}),
    ("grade", {"status": "ok", "submission_id": SID, "diff_applied": True}),
    ("replay", {"status": "ok", "submission_id": SID, "diff_applied": True, "tests_passed": True}),
])
async def test_an_ok_result_without_its_mode_facts_is_refused(mode, answer):
    d = _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item(mode)))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    with pytest.raises(LeaseRefused):
        d.result("g0", lease["lease_id"], _result(answer))
    assert not decision.done() and d.claim("g1") is not None


@pytest.mark.parametrize("oversized", [
    {"actions": [{"tool": "bash", "arguments": "{}", "observation": "x"}] * (MAX_ACTIONS + 1)},
    {"actions": [{"tool": "b" * (MAX_TOOL_NAME_CHARS + 1), "arguments": "{}", "observation": None}]},
    {"actions": [{"tool": "bash", "arguments": "a" * (MAX_ARGUMENT_CHARS + 1), "observation": None}]},
    {"actions": [{"tool": "bash", "arguments": "{}", "observation": "o" * (MAX_OBSERVATION_CHARS + 1)}]},
    {"instance_id": ""},
])
async def test_an_item_no_lease_can_carry_is_ungradeable_and_never_leased(oversized):
    d = _dispatcher(recheck=1.0)
    got = await asyncio.wait_for(d.decide(_item("replay", **oversized)), timeout=1)
    assert got.status == "ungradeable" and got.result is None and got.graded_by == ()
    assert d.claim("g0") is None and d.stats["ungradeable"] == 1


def test_a_lease_outlives_its_work():
    from reliquary.validator.agentic_replay import DEFAULT_EPISODE_DEADLINE
    from reliquary.validator.corpus_grade_executor import DEFAULT_SCORING_SECONDS

    seconds = corpus_grade_remote.GRADE_LEASE_SECONDS
    # A replay is bounded by its episode deadline, a grade by its scoring
    # timeout; each lease leaves room for the box and the corpus load.
    assert seconds["replay"] >= DEFAULT_EPISODE_DEADLINE + 300
    assert seconds["grade"] >= DEFAULT_SCORING_SECONDS + 300


async def test_the_lease_expiry_follows_its_mode():
    clock = _Clock()
    d = _dispatcher(clock=clock)
    asyncio.ensure_future(d.decide(_item("replay")))
    asyncio.ensure_future(d.decide(_item("grade")))
    await asyncio.sleep(0)
    replay, grade = d.claim("g0"), d.claim("g1")
    assert replay["expires_at"] == clock.now + corpus_grade_remote.GRADE_LEASE_SECONDS["replay"]
    assert grade["expires_at"] == clock.now + corpus_grade_remote.GRADE_LEASE_SECONDS["grade"]


async def test_the_router_checks_scope_and_env():
    d = _dispatcher()
    app = FastAPI()
    app.include_router(build_grade_executor_router(d, d._directory))
    await d._directory.refresh()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as http:
        auth = {"Authorization": f"Bearer {TOKENS['g0']}"}
        claim = {"executor_id": "g0", "env_package": PACKAGE, "env_version": VERSION}
        assert (await http.post("/corpus/internal/grade/claim", json=claim, headers=auth)).status_code == 204
        wrong = {**claim, "env_version": "c" * 40}
        assert (await http.post("/corpus/internal/grade/claim", json=wrong, headers=auth)).status_code == 409
        bad = {"Authorization": "Bearer nope"}
        assert (await http.post("/corpus/internal/grade/claim", json=claim, headers=bad)).status_code == 401


async def test_the_router_serves_a_whole_lease():
    d = _dispatcher(recheck=1.0)
    app = FastAPI()
    app.include_router(build_grade_executor_router(d, d._directory))
    await d._directory.refresh()
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as http:
        auth = {"Authorization": f"Bearer {TOKENS['g0']}"}
        beat = await http.post("/corpus/internal/grade/heartbeat",
                               json={"executor_id": "g0", "detail": {"leases": 0}}, headers=auth)
        # The executor learns its env pin from the heartbeat (GradeExecutor.start).
        assert beat.json() == {"executor_id": "g0", "model_id": PACKAGE, "model_revision": VERSION}
        claim = {"executor_id": "g0", "env_package": PACKAGE, "env_version": VERSION}
        lease = (await http.post("/corpus/internal/grade/claim", json=claim, headers=auth)).json()
        url = f"/corpus/internal/grade/{lease['lease_id']}/result"
        other = {"Authorization": f"Bearer {TOKENS['g1']}"}
        assert (await http.post(url, json={"results": [PASS]}, headers=other)).status_code == 410
        forged = {"results": [{**PASS, "submission_id": "c" * 64}]}
        assert (await http.post(url, json=forged, headers=auth)).status_code == 422
        lease = (await http.post("/corpus/internal/grade/claim", json={**claim, "executor_id": "g1"},
                                 headers=other)).json()
        url = f"/corpus/internal/grade/{lease['lease_id']}/result"
        answered = await http.post(url, json={"results": [PASS]}, headers=other)
        assert answered.json() == {"lease_id": lease["lease_id"], "outcome": "accepted"}
    assert (await decision).graded_by == ("g1",)


async def test_a_corpus_directory_refuses_a_grade_token():
    async def listed():
        return _docs()

    directory = ExecutorDirectory(model_id=PACKAGE, model_revision=VERSION, list_documents=listed)
    await directory.refresh()
    assert directory.authenticate(TOKENS["g0"]) == (None, "wrong_scope")


async def test_a_grade_directory_refuses_a_corpus_token():
    async def listed():
        return [{k: v for k, v in doc.items() if k != "scope"} for doc in _docs()]

    directory = ExecutorDirectory(model_id=PACKAGE, model_revision=VERSION, list_documents=listed,
                                  scope="grade")
    await directory.refresh()
    assert directory.authenticate(TOKENS["g0"]) == (None, "wrong_scope")
