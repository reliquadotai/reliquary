"""Audit executors sized by the work waiting: a model's orders ask the fleet for enough executors that their
expected audits drain within the drain target, at the throughput one executor was measured at."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_auditor import CorpusAuditor
from reliquary.validator.eval_control import (
    MIN_MEASURED_BUSY_SECONDS,
    THROUGHPUT_WINDOW_SECONDS,
    ExecutorSizing,
    PairedAuditDispatcher,
    executors_for_backlog,
    order_audit_drain_seconds,
    order_max_audit_executors,
)
from tests.unit.test_eval_control import ITEMS, MODEL_A, MODEL_B, _executor, _honest


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_executors_cover_the_backlog_within_the_drain_target():
    size = lambda backlog, rate: executors_for_backlog(backlog, rate, drain_seconds=1800, max_executors=3)
    assert size(0, 3.0) == 0
    assert size(5000, 3.0) == 1          # 1667 s on one executor
    assert size(6000, 3.0) == 2          # 2000 s on one, 1000 s on two
    assert size(1_000_000, 3.0) == 3     # never past the maximum
    assert size(10, 3.0) == 1
    assert size(6000, None) is None      # unmeasured: keep what is there


def test_fewer_executors_are_wanted_only_after_the_scale_down_delay():
    clock = _Clock()
    sizing = ExecutorSizing(clock=clock, scale_down_seconds=600)
    assert sizing.update("m", None) is None
    assert sizing.update("m", 1) == 1
    assert sizing.update("m", 3) == 3     # up at once
    clock.now += 300
    assert sizing.update("m", 1) == 3
    clock.now += 200
    assert sizing.update("m", 3) == 3     # back up: the delay restarts
    assert sizing.update("m", 1) == 3
    clock.now += 599
    assert sizing.update("m", 1) == 3
    clock.now += 1
    assert sizing.update("m", 1) == 1
    assert sizing.update("m", None) == 1  # unmeasured: unchanged
    sizing.forget(set())
    assert sizing.update("m", None) is None


def test_settings_from_the_environment():
    assert order_audit_drain_seconds({}) == 1800.0
    assert order_audit_drain_seconds({"RELIQUARY_ORDER_AUDIT_DRAIN_SECONDS": "600"}) == 600.0
    assert order_max_audit_executors({}) == 3
    assert order_max_audit_executors({"RELIQUARY_ORDER_MAX_AUDIT_EXECUTORS": "5"}) == 5
    with pytest.raises(ValueError):
        order_audit_drain_seconds({"RELIQUARY_ORDER_AUDIT_DRAIN_SECONDS": "0"})
    with pytest.raises(ValueError):
        order_max_audit_executors({"RELIQUARY_ORDER_MAX_AUDIT_EXECUTORS": "0"})


def _dispatcher(clock):
    return PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None), clock=clock,
                                 scorers=1)


async def _lease(d, clock, seconds, model=MODEL_A, executor="a"):
    """One batch of ITEMS (2 items) scored by ``executor`` in ``seconds``."""
    scoring = asyncio.ensure_future(d.view(*model, PROOF).score(ITEMS))
    await asyncio.sleep(0)
    lease = d.claim(_executor(executor, model=model))
    clock.now += seconds
    await d.result(executor, lease["lease_id"], _honest())
    await scoring


def test_throughput_is_items_per_second_an_executor_holds_a_lease():
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        assert d.items_per_second(MODEL_A) is None
        await _lease(d, clock, 50)
        assert d.items_per_second(MODEL_A) is None  # 50 s measured: under the minimum
        await _lease(d, clock, 50)
        clock.now += 300                           # idle: not counted
        await _lease(d, clock, 100)
        assert d.items_per_second(MODEL_A) == pytest.approx(6 / 200)
        assert d.items_per_second(MODEL_B) is None
        clock.now += THROUGHPUT_WINDOW_SECONDS + 1
        assert d.items_per_second(MODEL_A) is None  # all of it left the window

    asyncio.run(go())


def test_throughput_counts_a_lease_still_held():
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        await _lease(d, clock, 60)
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        assert d.claim(_executor("a")) is not None
        clock.now += MIN_MEASURED_BUSY_SECONDS
        # 2 items over 60 s closed and the open lease's time so far.
        assert d.items_per_second(MODEL_A) == pytest.approx(2 / (60 + MIN_MEASURED_BUSY_SECONDS))
        scoring.cancel()

    asyncio.run(go())


def test_expected_audits_are_items_at_the_jobs_audited_share():
    from collections import Counter

    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._params = AuditParams(q=0.2, hold_seconds=600)
    auditor._decided = Counter()
    auditor._completions_read = [0, 0]
    assert auditor.expected_audits(None) is None              # before the seed: unknown, not zero
    assert auditor.expected_audits(0) == 0.0
    assert auditor.expected_audits(1000) == pytest.approx(200)   # no verdict yet: q, one item per record
    auditor._completions_read = [10, 40]                          # four completions per record: four items
    assert auditor.expected_audits(1000) == pytest.approx(800)
    auditor._completions_read = [0, 0]
    auditor._decided = Counter({"audit": 950})                    # probation: everything audited
    assert auditor.expected_audits(1000) == pytest.approx(1000 * 960 / 1000)
    auditor._decided = Counter({"audit": 600, "pass_unaudited": 1400})
    assert auditor.expected_audits(1000) == pytest.approx(1000 * 610 / 2050)
    auditor._decided = Counter({"audit": 100, "pass_unaudited": 1900})  # luck under q: never below q
    assert auditor.expected_audits(1000) == pytest.approx(200)


def test_idle_executors_are_those_holding_no_lease():
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        d.heartbeat("b", _executor("b"))
        assert d.claim(_executor("a")) is not None
        live, idle = d.executor_ids(MODEL_A)
        scoring.cancel()
        return live, idle

    live, idle = asyncio.run(go())
    assert live == ["a", "b"] and idle == ["b"]


def test_an_executor_to_release_is_leased_nothing_more():
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        d.heartbeat("old", _executor("old"))
        clock.now += 10
        d.heartbeat("new", _executor("new"))
        assert d.release_beyond(MODEL_A, None) == []        # nothing wanted known: none released
        assert d.release_beyond(MODEL_A, 2) == []
        assert d.release_beyond(MODEL_A, 0) == ["new"]      # at least one kept, the newest goes
        d.set_releasing(["new"])
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        assert d.claim(_executor("new")) is None
        assert d.claim(_executor("old")) is not None
        d.set_releasing([])                                  # wanted again: leased as before
        scoring.cancel()

    asyncio.run(go())
