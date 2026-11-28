"""Audit executors sized by the work waiting: a model's orders ask the fleet for enough executors that their
expected audits drain within the drain target, at the throughput one executor was measured at."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.audit_policy import AuditParams
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditResult
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


async def _lease(d, clock, seconds, model=MODEL_A, executor="a", *, document=None, protocols=(AUDIT_PROTOCOL,)):
    """One batch of ITEMS (2 items) scored by ``executor`` in ``seconds``."""
    scoring = asyncio.ensure_future(d.view(*model, PROOF).score(ITEMS))
    await asyncio.sleep(0)
    lease = d.claim(document or _executor(executor, model=model), protocols)
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
        # ITEMS: 4 completion and 3 prompt tokens a batch.
        assert d.tokens_per_second(MODEL_A) == pytest.approx((12 / 200, 9 / 200))
        assert d.items_per_second(MODEL_B) is None
        assert d.tokens_per_second(MODEL_B) is None
        clock.now += THROUGHPUT_WINDOW_SECONDS + 1
        assert d.items_per_second(MODEL_A) is None  # all of it left the window
        assert d.tokens_per_second(MODEL_A) is None

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
    assert auditor.audit_share() == pytest.approx(0.2)            # the share the sandbox sizing reads
    auditor._decided = Counter({"audit": 600, "pass_unaudited": 1400})
    assert auditor.audit_share() == pytest.approx(610 / 2050)


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
        assert d.claim(_executor("old")) is None             # claims: its model is loaded (nothing to lease yet)
        clock.now += 10
        assert d.claim(_executor("new")) is None
        assert d.release_beyond(MODEL_A, None) == []        # nothing wanted known: none released
        assert d.release_beyond(MODEL_A, 2) == []
        assert d.release_beyond(MODEL_A, 0) == []
        await _lease(d, clock, 1, executor="new")
        assert d.release_beyond(MODEL_A, 0) == ["old"]      # at least one kept, the oldest goes
        d.set_releasing(["old"])
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        assert d.claim(_executor("old")) is None
        assert d.claim(_executor("new")) is not None
        d.set_releasing([])                                  # wanted again: leased as before
        scoring.cancel()

    asyncio.run(go())


def _rented(eid, at, model=MODEL_A):
    """An executor's registration as the directory reads it: made when the fleet rented the card."""
    return {**_executor(eid, model=model), "registered_at": at}


def test_the_oldest_rented_cards_are_released_first():
    """The newest card runs the image the fleet rents now: it is kept, so a new image replaces the old one.
    Age is the registration: a restarted control hears from its cards in any order."""
    clock = _Clock()
    d = _dispatcher(clock)
    for eid, rented_at in (("c", 300.0), ("b", 200.0), ("a", 100.0)):  # first contact: newest first
        asyncio.run(_lease(d, clock, 10, executor=eid, document=_rented(eid, rented_at)))
    assert d.release_beyond(MODEL_A, 1) == ["a", "b"]
    assert d.release_beyond(MODEL_A, 2) == ["a"]
    assert d.release_beyond(MODEL_A, 3) == []
    assert d.release_beyond(MODEL_A, 0) == ["a", "b"]                 # at least one kept
    clock.now += 1000                                                # "c" falls silent: no longer live
    for eid, rented_at in (("a", 100.0), ("b", 200.0)):
        d.claim(_rented(eid, rented_at))
    assert d.release_beyond(MODEL_A, 1) == []  # stale results after reconnection
    asyncio.run(_lease(d, clock, 1, executor="b", document=_rented("b", 200.0)))
    assert d.release_beyond(MODEL_A, 1) == ["a"]


def test_a_card_still_loading_its_model_is_neither_kept_nor_released():
    """Until the new card claims, the old one keeps serving: a card that never loads displaces nothing."""
    clock = _Clock()
    d = _dispatcher(clock)
    d.claim(_rented("old", 100.0))
    d.heartbeat("new", _rented("new", 200.0))        # live, its model still loading
    assert d.release_beyond(MODEL_A, 1) == []
    d.heartbeat("next", _rented("next", 300.0))
    assert d.release_beyond(MODEL_A, 1) == []
    d.claim(_rented("new", 200.0))                   # loaded; still no measured result
    assert d.release_beyond(MODEL_A, 1) == []
    asyncio.run(_lease(d, clock, 1, executor="new", document=_rented("new", 200.0)))
    assert d.release_beyond(MODEL_A, 1) == ["old"]


def test_the_borrowed_draw_auditor_is_a_card_the_fleet_keeps():
    clock = _Clock()
    d = _dispatcher(clock)
    draw = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3)
    d.claim(_rented("old", 100.0), draw)
    d.claim(_rented("new", 200.0), draw)
    assert d.reusable_auditor(MODEL_A, AUDIT_PROTOCOL_V3) == "new"   # the one release_beyond keeps
    d.claim(_rented("legacy", 300.0))                                 # newest, but claims no draw audit
    assert d.reusable_auditor(MODEL_A, AUDIT_PROTOCOL_V3) == "new"
    d.set_releasing(["new"])
    assert d.reusable_auditor(MODEL_A, AUDIT_PROTOCOL_V3) == "old"
    d.set_releasing(["old", "new"])
    assert d.reusable_auditor(MODEL_A, AUDIT_PROTOCOL_V3) is None
    pair = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None), clock=clock,
                                 scorers=2)
    pair.claim(_rented("new", 200.0), draw)
    assert pair.reusable_auditor(MODEL_A, AUDIT_PROTOCOL_V3) is None  # only a single scorer is borrowed


def test_a_borrowed_batch_goes_to_the_card_the_fleet_keeps():
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        d.reuse_only_jobs.add("borrow")
        d.claim(_rented("old", 100.0))
        clock.now += 10
        d.claim(_rented("new", 200.0))
        borrow = asyncio.ensure_future(d.view(*MODEL_A, PROOF, "borrow").score(ITEMS))
        await asyncio.sleep(0)
        assert d.claim(_rented("old", 100.0)) is None    # released next: it borrows nothing
        assert d.claim(_rented("new", 200.0)) is not None
        borrow.cancel()

    asyncio.run(go())


def test_new_card_must_cover_the_protocols_of_the_one_leaving():
    clock = _Clock()
    d = _dispatcher(clock)
    draw = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3)
    d.claim(_rented("old", 100.0), draw)
    asyncio.run(_lease(d, clock, 1, executor="new", document=_rented("new", 200.0)))
    assert d.release_beyond(MODEL_A, 1) == []
    d.claim(_rented("new", 200.0), draw)
    assert d.release_beyond(MODEL_A, 1) == ["old"]


def test_restart_waits_for_a_new_score_even_when_registered_age_is_known():
    clock = _Clock()
    for _ in range(2):
        d = _dispatcher(clock)
        d.claim(_rented("new", 200.0))
        d.claim(_rented("old", 100.0))
        assert d.release_beyond(MODEL_A, 1) == []
        asyncio.run(_lease(d, clock, 1, executor="new", document=_rented("new", 200.0)))
        assert d.release_beyond(MODEL_A, 1) == ["old"]

@pytest.mark.parametrize("status", ["error", "proof_undecodable"])
def test_an_answer_without_a_measure_never_displaces_a_working_card(status):
    clock = _Clock()

    async def go():
        d = _dispatcher(clock)
        await _lease(d, clock, 1, executor="old", document=_rented("old", 100.0))
        if status == "error":
            await _lease(d, clock, 1, executor="new", document=_rented("new", 200.0))
            assert d.release_beyond(MODEL_A, 1) == ["old"]
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
        await asyncio.sleep(0)
        lease = d.claim(_rented("new", 200.0))
        result = AuditResult.model_validate({"scores": [{"status": status} for _ in ITEMS]})
        await d.result("new", lease["lease_id"], result)
        assert d.release_beyond(MODEL_A, 1) == []
        scoring.cancel()

    asyncio.run(go())


def test_release_preserves_distinct_hosts_for_paired_scoring():
    clock = _Clock()
    d = PairedAuditDispatcher(directory=SimpleNamespace(), clock=clock, scorers=2)
    old = _rented("old", 100)
    replacements = [_rented("new", 200), {**_rented("next", 300), "host": "host-new"}]

    async def go():
        for replacement in replacements:
            scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF).score(ITEMS))
            await asyncio.sleep(0)
            for card in (old, replacement):
                lease = d.claim(card)
                await d.result(card["executor_id"], lease["lease_id"], _honest())
            await scoring
        assert d.release_beyond(MODEL_A, 2) == []

    asyncio.run(go())


@pytest.mark.parametrize("disconnect", [False, True])
def test_replacement_must_score_a_draw_if_the_old_card_scored_draws(disconnect):
    from reliquary.corpus.job import DEFAULT_DRAW_JUDGE
    from tests.unit.test_eval_control_long_trajectories import SAMPLED, _drawn_item

    clock = _Clock()
    d = _dispatcher(clock)
    protocols = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3)
    old, new = _rented("old", 100), _rented("new", 200)

    async def score_draw(card):
        scoring = asyncio.ensure_future(d.view(*MODEL_A, PROOF, "job", DEFAULT_DRAW_JUDGE).score(
            [_drawn_item(1, SAMPLED)]))
        await asyncio.sleep(0)
        lease = d.claim(card, protocols)
        diag = {"positions": 1, "stochastic": 1, "exact": 1, "max_miss": 0.0, "last_miss": 0.0}
        score = AuditResult.model_validate({"scores": [{"status": "ok", "draw": diag}]})
        await d.result(card["executor_id"], lease["lease_id"], score)
        await scoring

    async def go():
        await score_draw(old)
        await _lease(d, clock, 1, executor="new", document=new, protocols=protocols)
        assert d.release_beyond(MODEL_A, 1) == []
        await score_draw(new)
        assert d.release_beyond(MODEL_A, 1) == ["old"]
        # Keep the incumbent live. The replacement either stays live with stale
        # draw evidence, or reconnects under the same registration.
        for _ in range(4):
            clock.now += 30
            d.heartbeat("old", old)
            if not disconnect:
                d.heartbeat("new", new)
        await _lease(d, clock, 1, executor="new", document=new, protocols=protocols)
        assert d.release_beyond(MODEL_A, 1) == []
        await score_draw(new)
        assert d.release_beyond(MODEL_A, 1) == ["old"]
        d.heartbeat("new", new, {"loaded": False})
        assert d.release_beyond(MODEL_A, 1) == []

    asyncio.run(go())
