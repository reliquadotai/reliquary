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


def _docs(scope="grade", providers=None):
    # Each executor on its own provider unless told otherwise (ruling P17).
    providers = providers or {eid: f"p{k}" for k, eid in enumerate(TOKENS)}
    return [dict(executor_id=eid, token_sha256=token_sha256(token), model_id=PACKAGE,
                 model_revision=VERSION, expires_at=1e12, status="active", scope=scope,
                 **({"provider_id": providers[eid]} if providers.get(eid) else {}))
            for eid, token in TOKENS.items()]


class _Rng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


async def _dispatcher(recheck=1.0, clock=None, quarantined=None, providers=None,
                      write=None, **kw):
    clock = clock or _Clock()

    async def listed():
        return _docs(providers=providers)

    async def quarantine(eid, reason):
        if write is not None:
            await write(eid, reason)
        if quarantined is not None:
            quarantined.append(eid)

    directory = ExecutorDirectory(model_id=PACKAGE, model_revision=VERSION, list_documents=listed,
                                  clock=clock, scope="grade")
    await directory.refresh()
    return RemoteGradeDispatcher(directory=directory, env_package=PACKAGE, env_version=VERSION,
                                 quarantine=quarantine, clock=clock, rng=_Rng(recheck), **kw)


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
    d = await _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    assert lease["env"] == {"package": PACKAGE, "version": VERSION}
    assert lease["items"][0]["instance_id"] == "repo__x.1"
    GradeLease.model_validate(lease)                      # what the executor will parse
    d.result("g0", lease["lease_id"], _result(PASS))
    assert (await decision).graded_by == ("g0",)


async def test_a_drawn_recheck_needs_a_second_executor():
    d = await _dispatcher(recheck=0.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    assert not decision.done() and d.claim("g0") is None      # never the same executor twice
    _answer(d, "g1", PASS)
    assert (await decision).graded_by == ("g0", "g1")


async def test_a_failure_with_one_executor_waits_for_a_second():
    d = await _dispatcher(recheck=1.0)
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
    d = await _dispatcher(recheck=1.0, quarantined=quarantined)
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
    d = await _dispatcher(recheck=0.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    _answer(d, "g1", {**PASS, "tests_passed": False})
    _answer(d, "g2", {**PASS, "diff_applied": False, "tests_passed": False})
    got = await decision
    assert got.status == "error" and got.result is None
    assert not d.quarantined


async def test_a_quarantined_executor_loses_its_pending_vote():
    d = await _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)                 # waits for a second executor
    await d.quarantine("g0", "caught elsewhere")
    _answer(d, "g1", REPLAY_OK)                  # alone now: a pass needs one, undrawn
    got = await decision
    assert got.graded_by == ("g1",) and got.result["replay_diff_equal"] is True


async def test_a_quarantined_vote_on_a_leased_item_is_dropped_too():
    d = await _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    lease = d.claim("g1")                        # g1 works while g0 is caught
    await d.quarantine("g0", "caught elsewhere")
    assert d.claim("g2") is None                 # never leased twice at once
    d.result("g1", lease["lease_id"], _result(REPLAY_OK))
    assert (await decision).graded_by == ("g1",)


async def test_timeouts_go_to_other_executors_then_resolve_unjudged():
    d = await _dispatcher(recheck=1.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", TIMEOUT)
    assert d.claim("g0") is None
    _answer(d, "g1", TIMEOUT)
    assert (await decision).status == "timeout"


async def test_three_errors_resolve_as_an_error():
    d = await _dispatcher(recheck=1.0)
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
    d = await _dispatcher(recheck=1.0, clock=clock, quarantined=quarantined)
    for _ in range(3):
        asyncio.ensure_future(d.decide(_item()))
        await asyncio.sleep(0)
        assert d.claim("g3") is not None
        clock.now += 10_000
        await d.sweep()
    assert "g3" in d.quarantined and quarantined == ["g3"]


async def test_an_item_whose_leases_keep_expiring_resolves_as_a_timeout():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock)
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
    d = await _dispatcher(recheck=1.0, clock=clock)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    clock.now += 10_000
    with pytest.raises(LeaseRefused) as refused:
        d.result("g0", lease["lease_id"], _result(REPLAY_BAD))
    assert refused.value.detail == "lease_expired"
    assert d.claim("g0") is None and not decision.done()
    assert d._strikes["g0"] == 1                  # a late result is an expiry: struck
    _answer(d, "g1", REPLAY_OK)
    assert (await decision).graded_by == ("g1",)


@pytest.mark.parametrize("echo", [None, "c" * 64])
async def test_a_result_for_another_submission_is_refused_and_struck(echo):
    d = await _dispatcher(recheck=1.0)
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
    d = await _dispatcher(recheck=1.0)
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
    d = await _dispatcher(recheck=1.0)
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
    d = await _dispatcher(clock=clock)
    asyncio.ensure_future(d.decide(_item("replay")))
    asyncio.ensure_future(d.decide(_item("grade")))
    await asyncio.sleep(0)
    replay, grade = d.claim("g0"), d.claim("g1")
    assert replay["expires_at"] == clock.now + corpus_grade_remote.GRADE_LEASE_SECONDS["replay"]
    assert grade["expires_at"] == clock.now + corpus_grade_remote.GRADE_LEASE_SECONDS["grade"]


async def test_the_router_checks_scope_and_env():
    d = await _dispatcher()
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
    d = await _dispatcher(recheck=1.0)
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


# --------------------------------------------------------------------------
# Fix round 1: bounded waits (P16), distinct providers (P17), races
# --------------------------------------------------------------------------


async def test_a_disagreement_without_a_third_executor_resolves_disputed(monkeypatch):
    warned = []
    monkeypatch.setattr(corpus_grade_remote.logger, "warning",
                        lambda *a, **k: warned.append(a[0] % a[1:]))
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, dispute_seconds=1800.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    _answer(d, "g1", REPLAY_OK)
    clock.now += 1799
    for eid in ("g0", "g1"):
        d.heartbeat(eid)                          # live, but both excluded
    await d.sweep()
    assert not decision.done()
    assert d.stats["stranded"] >= 1 and any("every live grade executor" in w for w in warned)
    clock.now += 2
    await d.sweep()
    got = await decision
    assert got.status == "disputed" and got.result is None and got.graded_by == ("g0", "g1")
    assert d.stats["disputed"] == 1 and any("disputed" in w for w in warned)
    assert not d.quarantined                      # nobody is judged


async def test_a_failing_replay_without_a_second_provider_resolves_disputed():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, dispute_seconds=1800.0,
                          providers={"g0": "hetzner", "g1": "hetzner", "g2": "hetzner",
                                     "g3": "hetzner"})
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    for eid in ("g1", "g2", "g3"):
        assert d.claim(eid) is None               # same provider: never a second vote
    clock.now += 1801
    await d.sweep()
    got = await decision
    assert got.status == "disputed" and got.graded_by == ("g0",)


async def test_a_drawn_recheck_without_a_second_executor_resolves_disputed():
    clock = _Clock()
    d = await _dispatcher(recheck=0.0, clock=clock, dispute_seconds=1800.0)
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    clock.now += 1801
    await d.sweep()
    assert (await decision).status == "disputed"


async def test_a_same_provider_vote_never_completes_an_agreement():
    d = await _dispatcher(recheck=0.0, providers={"g0": "a", "g1": "a", "g2": "b", "g3": "b"})
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    assert d.claim("g1") is None
    _answer(d, "g2", REPLAY_BAD)
    got = await decision
    assert got.status == "ok" and got.graded_by == ("g0", "g2")


async def test_an_executor_without_a_provider_gets_no_lease():
    d = await _dispatcher(providers={"g0": None, "g1": "p1", "g2": "p2", "g3": "p3"})
    asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    assert d.claim("g0") is None and d.claim("g1") is not None


def test_a_grade_executor_registers_with_its_provider(monkeypatch):
    from reliquary.infrastructure import corpus_executor_store as executors
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: _FakeMultiObjectR2())
    fields = dict(executor_id="g1", token_sha256="d" * 64, model_id="reliquary-swe",
                  model_revision="b" * 40, expires_at=2e9, now=1000.0, scope="grade")
    with pytest.raises(ValueError, match="provider_id"):
        asyncio.run(executors.register_executor(**fields))
    doc, _ = asyncio.run(executors.register_executor(**fields, provider_id="hetzner"))
    assert doc["provider_id"] == "hetzner"


def test_the_register_command_requires_a_provider(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import corpus_executor_store

    calls = []

    async def register(**kw):
        calls.append(kw)
        return {"executor_id": kw["executor_id"]}, True

    monkeypatch.setattr(corpus_executor_store, "register_executor", register)
    argv = ["corpus", "register-grade-executor", "--executor-id", "g1", "--env-version", "b" * 40]
    assert CliRunner().invoke(app, argv).exit_code == 2 and calls == []
    result = CliRunner().invoke(app, argv + ["--provider-id", "hetzner"])
    assert result.exit_code == 0, result.output
    assert calls[0]["provider_id"] == "hetzner"


async def test_a_vote_is_dropped_before_the_quarantine_write_yields():
    async def slow_write(eid, reason):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    d = await _dispatcher(recheck=1.0, write=slow_write)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    caught = asyncio.ensure_future(d.quarantine("g0", "caught elsewhere"))
    await asyncio.sleep(0)                        # the write is now pending
    assert not caught.done()
    _answer(d, "g1", REPLAY_BAD)                  # would co-sign a sanction with g0
    assert not decision.done()
    await caught
    _answer(d, "g2", REPLAY_BAD)
    got = await decision
    assert got.graded_by == ("g1", "g2") and "g0" not in got.graded_by


async def test_a_dissenter_is_quarantined_at_decision_time():
    async def slow_write(eid, reason):
        await asyncio.sleep(0)

    d = await _dispatcher(recheck=1.0, write=slow_write)
    first = asyncio.ensure_future(d.decide(_item("replay")))
    second = asyncio.ensure_future(d.decide(_item("replay", task_index=2)))
    await asyncio.sleep(0)
    lease = d.claim("g0")                         # g0 works on item 1, then lies on item 2
    _answer(d, "g0", REPLAY_BAD)
    _answer(d, "g1", REPLAY_OK)
    _answer(d, "g2", REPLAY_OK)                   # item 2 decided: g0 dissented
    assert "g0" in d.quarantined                  # before any background write ran
    assert d.claim("g0") is None
    with pytest.raises(LeaseRefused):
        d.result("g0", lease["lease_id"], _result(REPLAY_OK))
    assert (await second).graded_by == ("g1", "g2")
    assert not first.done()


# --------------------------------------------------------------------------
# Carried into Task 17: a vote always carries its executor's provider
# --------------------------------------------------------------------------


async def test_a_result_from_an_executor_that_lost_its_provider_is_refused():
    docs = _docs()

    async def listed():
        return [dict(d) for d in docs]

    directory = ExecutorDirectory(model_id=PACKAGE, model_revision=VERSION, list_documents=listed,
                                  clock=_Clock(), scope="grade")
    await directory.refresh()
    d = RemoteGradeDispatcher(directory=directory, env_package=PACKAGE, env_version=VERSION,
                              clock=_Clock(), rng=_Rng(1.0))
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    docs[0].pop("provider_id")                   # the registry no longer names its provider
    await directory.refresh()
    with pytest.raises(LeaseRefused) as refused:
        d.result("g0", lease["lease_id"], _result(REPLAY_OK))
    assert refused.value.status == 403
    assert not decision.done()                   # its answer never counted, not even as g0
    _answer(d, "g1", REPLAY_OK)
    assert (await decision).graded_by == ("g1",)


async def test_providers_count_once_whatever_their_spelling():
    d = await _dispatcher(recheck=0.0, providers={"g0": "Hetzner", "g1": " hetzner ",
                                                  "g2": "OVH", "g3": "ovh"})
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    assert d.claim("g1") is None                 # the same provider, written otherwise
    _answer(d, "g2", REPLAY_BAD)
    assert (await decision).graded_by == ("g0", "g2")


def test_a_grade_executor_provider_is_normalized_at_registration(monkeypatch):
    from reliquary.infrastructure import corpus_executor_store as executors
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: _FakeMultiObjectR2())
    fields = dict(executor_id="g1", token_sha256="d" * 64, model_id="reliquary-swe",
                  model_revision="b" * 40, expires_at=2e9, now=1000.0, scope="grade")
    with pytest.raises(ValueError, match="provider_id"):
        asyncio.run(executors.register_executor(**fields, provider_id="   "))
    doc, _ = asyncio.run(executors.register_executor(**fields, provider_id="  Hetzner "))
    assert doc["provider_id"] == "hetzner"


async def test_an_agreed_decision_names_its_providers():
    d = await _dispatcher(recheck=1.0, providers={"g0": "a", "g1": "a", "g2": "b", "g3": "c"})
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    _answer(d, "g2", REPLAY_BAD)
    got = await decision
    assert got.providers == ("a", "b") and d.env_pin == (PACKAGE, VERSION)


async def test_a_quarantine_holds_before_any_listener_runs():
    seen = []
    d = await _dispatcher()
    d.hold_on_quarantine(lambda eid: seen.append(("hold", eid)))

    async def listener(eid):
        seen.append(("listener", eid))

    d.subscribe(listener)
    await d.quarantine("g0", "caught")
    for _ in range(5):
        await asyncio.sleep(0)
    assert seen == [("hold", "g0"), ("listener", "g0")]
