"""Liveness of the grade dispatcher with a single executor (production
incident 2026-10-04): an item whose only executor failed an attempt on it
(an expired lease, an error, a misfit) must not wait forever."""

import asyncio

import pytest

from reliquary.validator import corpus_grade_remote
from tests.unit.test_corpus_audit_remote import _Clock
from tests.unit.test_corpus_grade_remote import (
    ERROR,
    REPLAY_BAD,
    REPLAY_OK,
    _answer,
    _dispatcher,
    _item,
    _result,
)

RETRY = 600.0
REPLAY_LEASE = corpus_grade_remote.GRADE_LEASE_SECONDS["replay"]


async def _expire(d, clock, eid="g0"):
    lease = d.claim(eid)
    assert lease is not None, f"{eid} got no lease"
    clock.now = lease["expires_at"] + 1
    await d.sweep()
    return lease


def test_the_retry_wait_is_a_bounded_knob(monkeypatch):
    assert corpus_grade_remote.GRADE_RETRY_EXCLUDED_SECONDS == 600.0
    monkeypatch.setenv("RELIQUARY_CORPUS_GRADE_RETRY_EXCLUDED_SECONDS", "5")
    with pytest.raises(ValueError):
        corpus_grade_remote._bounded_env("RELIQUARY_CORPUS_GRADE_RETRY_EXCLUDED_SECONDS",
                                         600.0, *corpus_grade_remote.GRADE_RETRY_EXCLUDED_BOUNDS)


async def test_a_single_executor_takes_back_its_expired_replay_after_the_wait():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    await _expire(d, clock)
    assert not decision.done()
    assert d.claim("g0") is None                  # not at once: a bounded wait first
    clock.now += RETRY - 1
    assert d.claim("g0") is None
    clock.now += 1
    lease = d.claim("g0")
    assert lease is not None                      # the same executor, again
    assert d.result("g0", lease["lease_id"], _result(REPLAY_OK)) == "accepted"
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g0",)


async def test_a_second_expiry_of_the_single_executor_resolves_timeout():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    await _expire(d, clock)
    clock.now += RETRY
    await _expire(d, clock)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "timeout" and got.graded_by == () and got.result is None


async def test_a_single_executor_erring_three_times_resolves_error():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    for attempt in range(3):
        if attempt:
            assert d.claim("g0") is None
            clock.now += RETRY
        _answer(d, "g0", ERROR)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "error" and got.graded_by == ()


async def test_an_executor_that_voted_is_never_readmitted():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY,
                          dispute_seconds=86400.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)                  # a failing replay: needs a second vote
    for _ in range(5):
        clock.now += RETRY * 3
        await d.sweep()
        assert d.claim("g0") is None
    assert not decision.done()


async def test_a_voter_that_then_failed_elsewhere_is_still_never_readmitted():
    # g0 voted, g1 then let its lease expire: g1 comes back after the wait,
    # g0 never does.
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY,
                          dispute_seconds=86400.0)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    await _expire(d, clock, "g1")
    clock.now += RETRY
    assert d.claim("g0") is None
    _answer(d, "g1", REPLAY_BAD)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g0", "g1")


async def test_multi_executor_another_executor_still_takes_it_at_once():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    await _expire(d, clock, "g0")
    assert d.claim("g0") is None
    _answer(d, "g1", REPLAY_OK)                   # no wait for an executor that never failed it
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g1",)


async def test_past_the_wait_a_claiming_executor_that_never_failed_is_preferred():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    await _expire(d, clock, "g0")
    clock.now += RETRY
    d.heartbeat("g1")
    d._claimed_at["g1"] = clock.now               # g1 claims lately, eligible
    assert d.claim("g0") is None                  # so g0 is not taken back
    _answer(d, "g1", REPLAY_OK)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g1",)


async def test_the_production_symptom_no_longer_waits_forever(monkeypatch):
    """One executor (provider "hetzner"), a replay whose 12 000 s lease
    expired: before the fix every sweep logged 'waits: every live grade
    executor is excluded from it' and the item never resolved, so no period
    could settle. The executor polls (claim every 30 s) and sweeps run."""
    warned = []
    monkeypatch.setattr(corpus_grade_remote.logger, "warning",
                        lambda *a, **k: warned.append(a[0] % a[1:]))
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock,
                          providers={"g0": "hetzner", "g1": "hetzner", "g2": "hetzner",
                                     "g3": "hetzner"})
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    first = d.claim("g0")
    assert first is not None
    clock.now = first["expires_at"] + 1
    await d.sweep()
    second = None
    for _ in range(int(3600 / 30)):                # one hour of polling at most
        clock.now += 30
        second = d.claim("g0")
        if second is not None:
            break
        await d.sweep()
    assert second is not None, "the item still waits for an executor that will never come"
    d.result("g0", second["lease_id"], _result(REPLAY_OK))
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g0",)


# --------------------------------------------------------------------------
# Recheck draws need a second provider to recheck with
# --------------------------------------------------------------------------

ONE_PROVIDER = {"g0": "hetzner", "g1": "hetzner", "g2": "hetzner", "g3": "hetzner"}


def _logged(monkeypatch):
    said = []
    for level in ("info", "warning"):
        monkeypatch.setattr(corpus_grade_remote.logger, level,
                            lambda *a, **k: said.append(a[0] % a[1:]))
    return said


@pytest.mark.parametrize("mode,answer", [("grade", None), ("replay", REPLAY_OK)])
async def test_with_one_provider_nothing_is_drawn_for_a_recheck(monkeypatch, mode, answer):
    from tests.unit.test_corpus_grade_remote import PASS

    said = _logged(monkeypatch)
    d = await _dispatcher(recheck=0.0, providers=dict(ONE_PROVIDER))    # would always draw
    decision = asyncio.ensure_future(d.decide(_item(mode)))
    await asyncio.sleep(0)
    _answer(d, "g0", answer or PASS)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g0",)          # at once, undrawn
    assert sum("rechecks disabled" in s for s in said) == 1


async def test_draws_resume_when_a_second_provider_registers(monkeypatch):
    from tests.unit.test_corpus_grade_remote import PASS

    said = _logged(monkeypatch)
    providers = dict(ONE_PROVIDER)
    d = await _dispatcher(recheck=0.0, providers=providers)
    for _ in range(2):                                              # logged once, not per item
        decision = asyncio.ensure_future(d.decide(_item()))
        await asyncio.sleep(0)
        _answer(d, "g0", PASS)
        assert (await asyncio.wait_for(decision, 5)).graded_by == ("g0",)
    assert sum("rechecks disabled" in s for s in said) == 1
    providers["g1"] = "lium"
    await d._directory.refresh()
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    await asyncio.sleep(0)
    assert not decision.done()                                      # drawn: a second provider
    _answer(d, "g1", PASS)
    got = await asyncio.wait_for(decision, 5)
    assert got.graded_by == ("g0", "g1") and got.providers == ("hetzner", "lium")
    assert sum("rechecks enabled" in s for s in said) == 1


async def test_a_quarantined_second_provider_does_not_count():
    from tests.unit.test_corpus_grade_remote import PASS

    d = await _dispatcher(recheck=0.0, providers={**ONE_PROVIDER, "g1": "lium"})
    d._mark_quarantined("g1", "test")
    decision = asyncio.ensure_future(d.decide(_item()))
    await asyncio.sleep(0)
    _answer(d, "g0", PASS)
    assert (await asyncio.wait_for(decision, 5)).graded_by == ("g0",)


async def test_with_one_provider_a_failing_replay_still_needs_two_and_is_uncertified():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, dispute_seconds=1800.0,
                          providers=dict(ONE_PROVIDER))
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    _answer(d, "g0", REPLAY_BAD)
    await asyncio.sleep(0)
    assert not decision.done()
    clock.now += 1801
    await d.sweep()
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "uncertified" and not d.quarantined


# --------------------------------------------------------------------------
# Leaked leases: a claim whose reply was lost (production: RemoteProtocolError
# several times an hour) self-heals through the executor's heartbeat
# --------------------------------------------------------------------------

GRACE = corpus_grade_remote.GRADE_LEASE_REPORT_GRACE_SECONDS


async def test_a_claim_whose_reply_is_lost_is_taken_back_after_the_grace():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lost = d.claim("g0")                          # the reply never reaches g0
    assert lost is not None
    clock.now += GRACE - 1
    d.heartbeat("g0", {"leases": 0}, held_leases=[])
    assert lost["lease_id"] in d._leases          # within the grace: kept
    clock.now += 1
    d.heartbeat("g0", {"leases": 0}, held_leases=[])
    assert lost["lease_id"] not in d._leases      # taken back, no vote
    assert d.stats["leases_unreported"] == 1 and not decision.done()
    assert d._strikes["g0"] == 0 and not d.quarantined
    with pytest.raises(corpus_grade_remote.LeaseRefused):
        d.result("g0", lost["lease_id"], _result(REPLAY_OK))
    # Requeued: another executor takes it at once...
    _answer(d, "g1", REPLAY_OK)
    got = await asyncio.wait_for(decision, 5)
    assert got.status == "ok" and got.graded_by == ("g1",)


async def test_a_lost_claim_with_a_single_executor_comes_back_to_it():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock, retry_excluded_seconds=RETRY)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    assert d.claim("g0") is not None
    clock.now += GRACE
    d.heartbeat("g0", None, held_leases=[])
    assert d.claim("g0") is None                  # an executor fault: the retry wait
    clock.now += RETRY
    _answer(d, "g0", REPLAY_OK)
    assert (await asyncio.wait_for(decision, 5)).graded_by == ("g0",)


async def test_a_held_lease_is_never_taken_back():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    for _ in range(100):
        clock.now += 100
        d.heartbeat("g0", {"leases": 0}, held_leases=[lease["lease_id"]])
    assert lease["lease_id"] in d._leases
    assert d.result("g0", lease["lease_id"], _result(REPLAY_OK)) == "accepted"
    assert (await asyncio.wait_for(decision, 5)).graded_by == ("g0",)


async def test_another_executors_report_never_takes_back_a_lease():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock)
    asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    clock.now += GRACE * 10
    d.heartbeat("g1", None, held_leases=[])
    assert lease["lease_id"] in d._leases


async def test_an_old_executor_without_the_field_behaves_as_before():
    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock)
    decision = asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    lease = d.claim("g0")
    clock.now += GRACE * 10
    d.heartbeat("g0", {"leases": 0})              # no held_leases: nothing taken back
    d.heartbeat("g0", {"leases": 0}, held_leases=None)
    assert lease["lease_id"] in d._leases and "leases_unreported" not in d.stats
    assert d.result("g0", lease["lease_id"], _result(REPLAY_OK)) == "accepted"
    assert (await asyncio.wait_for(decision, 5)).graded_by == ("g0",)


async def test_the_router_takes_the_held_leases_and_old_heartbeats_still_pass():
    import httpx
    from fastapi import FastAPI

    from tests.unit.test_corpus_grade_remote import PACKAGE, TOKENS, VERSION

    clock = _Clock()
    d = await _dispatcher(recheck=1.0, clock=clock)
    app = FastAPI()
    app.include_router(corpus_grade_remote.build_grade_executor_router(d, d._directory))
    asyncio.ensure_future(d.decide(_item("replay")))
    await asyncio.sleep(0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as http:
        auth = {"Authorization": f"Bearer {TOKENS['g0']}"}
        claim = {"executor_id": "g0", "env_package": PACKAGE, "env_version": VERSION}
        lease = (await http.post("/corpus/internal/grade/claim", json=claim,
                                 headers=auth)).json()
        clock.now += GRACE
        old = await http.post("/corpus/internal/grade/heartbeat",
                              json={"executor_id": "g0", "detail": {"leases": 0}}, headers=auth)
        assert old.status_code == 200 and lease["lease_id"] in d._leases
        new = await http.post("/corpus/internal/grade/heartbeat",
                              json={"executor_id": "g0", "detail": {"leases": 0},
                                    "held_leases": []}, headers=auth)
        assert new.status_code == 200 and lease["lease_id"] not in d._leases
