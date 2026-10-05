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
