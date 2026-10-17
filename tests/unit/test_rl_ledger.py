"""The async RL pay (P5): flat pay per matched token of trained groups above the floor, 25 % first scans after
audit, ban forfeiture, the per-env async price controller, period settlement into weight-ready archives, and the
async ``GET /rl/pay``."""
from __future__ import annotations

import asyncio

import pytest

from reliquary.services.exploration import AUDIT_FAILURE_DETERMINISTIC
from reliquary.validator import corpus_periods as cp
from reliquary.validator.emission_price import PriceState
from reliquary.validator.rl_ledger import FORFEITED, LedgerConfig, RlPayLedger
from reliquary.validator.rl_price import (
    DESCEND, HOLD, IDLE, RISE, AsyncPriceController, AsyncPriceParams, QueueTick, advance_async, queue_level,
)
from reliquary.validator.rl_submit import AuditResult
from tests.unit.test_rl_submit import ENV, NOW, RIGHT, book, group, lease, service, small_cap  # noqa: F401

OTHER = "other"
CAP = 0.5
PLANNED = 1_000_000.0
GROUP_TOKENS = 16 * len(RIGHT.tokens)


def priced(book, **params):
    return AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED,
                                params=AsyncPriceParams(**params), clock=lambda: NOW)


def ledger_for(book, subs, prices=None, **config):
    return RlPayLedger(book, subs, prices or priced(book), LedgerConfig(task_id="rl-async", cap=CAP, **config),
                       clock=lambda: NOW)


def trained(book, subs, prompt, match=1.0, *, hotkey="A", matched=None, now=NOW):
    held = lease(book, prompt, hotkey)
    subs.submit(held.lease_id, hotkey, group(8))
    subs.report_training(held.lease_id, rollout_match=[match] * 16 if isinstance(match, float) else match,
                         accepted=True, matched_tokens=matched, now=now)
    return held


def amounts(book):
    return book.db.execute("SELECT hotkey, kind, matched_tokens, amount, status FROM rl_accruals ORDER BY seq").fetchall()


NOMINAL = CAP * 0.5 / PLANNED                     # ENV's price per matched token at multiplier 1


def test_trained_groups_are_paid_flat_per_matched_token(book):
    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs)
    trained(book, subs, 1)
    trained(book, subs, 2, matched=[3] * 16)          # the trainer's exact counts win over match x length
    assert ledger.ingest() == 2
    assert amounts(book) == [("A", "trained", GROUP_TOKENS, pytest.approx(GROUP_TOKENS * NOMINAL), "accrued"),
                             ("A", "trained", 48, pytest.approx(48 * NOMINAL), "accrued")]
    assert ledger.ingest() == 0                        # the cursor persists


def test_rollouts_under_the_floor_are_never_paid(book):
    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs)
    per = len(RIGHT.tokens)
    trained(book, subs, 1, [0.99] * 10 + [0.949] * 6, matched=[per] * 16)
    held = lease(book, 2)
    subs.submit(held.lease_id, "A", group(8))
    with pytest.raises(ValueError):                                     # more matched tokens than generated
        subs.report_training(held.lease_id, rollout_match=[1.0] * 16, accepted=True, matched_tokens=[per + 1] * 16)
    subs.report_training(held.lease_id, rollout_match=[0.5] * 16, accepted=False)     # untrained: no money
    ledger.ingest()
    assert [(k, n) for _, k, n, _, _ in amounts(book)] == [("trained", 10 * per)]


def test_a_first_scan_earns_a_quarter_of_an_in_zone_group_after_audit(book):
    subs = service(book, probation_groups=0, audit_rate=0.0)
    ledger = ledger_for(book, subs)
    zero = lease(book, 10)
    subs.submit(zero.lease_id, "A", group(0))                  # not drawn: payable at once, default reference
    ledger.ingest()
    assert amounts(book)[-1][1:] == ("first_scan", 0, pytest.approx(0.25 * 16384 * NOMINAL), "accrued")
    trained(book, subs, 11)
    trained(book, subs, 12, matched=[0] * 16)                   # reference = mean matched tokens of trained groups
    again = lease(book, 13)
    subs.submit(again.lease_id, "A", group(16))
    ledger.ingest()
    assert amounts(book)[-1][3] == pytest.approx(0.25 * (GROUP_TOKENS / 2) * NOMINAL)

    drawn = service(book, probation_groups=10**6)              # every group drawn while in probation
    paid_ledger = ledger_for(book, drawn)
    leases = [lease(book, p) for p in (20, 21, 22)]
    for held in leases:
        drawn.submit(held.lease_id, "A", group(0))
    before = len(amounts(book))
    paid_ledger.ingest()
    assert len(amounts(book)) == before                        # nothing payable before the audit
    drawn.report_audit(leases[0].lease_id, AuditResult(True))
    drawn.report_audit(leases[1].lease_id, AuditResult(None))   # inconclusive: forfeited, no sanction
    drawn.report_audit(leases[2].lease_id, AuditResult(False, "statistical"))
    paid_ledger.ingest()
    rows = amounts(book)[before:]
    assert [(r[1], r[4]) for r in rows] == [("first_scan", "accrued")]


def test_a_ban_forfeits_what_is_unsettled_and_what_lands_during_it(book):
    subs = service(book, probation_groups=0, audit_rate=1.0)
    ledger = ledger_for(book, subs)
    trained(book, subs, 1, hotkey="A")
    trained(book, subs, 2, hotkey="B")
    pending = lease(book, 3, "A")
    subs.submit(pending.lease_id, "A", group(8))
    forged = lease(book, 4, "A")
    subs.submit(forged.lease_id, "A", group(0))
    ledger.ingest()
    subs.report_audit(forged.lease_id, AuditResult(False, AUDIT_FAILURE_DETERMINISTIC), now=NOW + 10)
    subs.report_training(pending.lease_id, rollout_match=[1.0] * 16, accepted=True, now=NOW + 20)  # during the ban
    ledger.ingest()
    assert [(h, s) for h, _, _, _, s in amounts(book)] == [("A", FORFEITED), ("B", "accrued"), ("A", FORFEITED)]
    board = ledger.balances()
    assert board["A"]["accrued"] == 0 and board["A"]["forfeited"] > 0 and board["A"]["banned_until"] == NOW + 86410
    assert board["B"]["accrued"] == pytest.approx(GROUP_TOKENS * NOMINAL)


def test_the_queue_sets_the_level_and_the_arrivals_set_the_mix():
    params = AsyncPriceParams()                                     # backlog 32 groups, starved 5 % of the tick
    assert queue_level(QueueTick(batches=0, starved_s=0.0), 300, params) == IDLE     # trainer down: all hold
    assert queue_level(QueueTick(batches=3, backlog=30, starved_s=16.0), 300, params) == RISE
    assert queue_level(QueueTick(batches=3, backlog=300), 300, params) == DESCEND  # 100 left after each batch
    assert queue_level(QueueTick(batches=3, backlog=30, starved_s=10.0), 300, params) == HOLD
    assert queue_level(QueueTick(batches=0, starved_s=300.0), 300, params) == RISE  # waited the whole tick
    state = PriceState(price=1.0, last_good=1.0)
    # (level, mix) -> regime: a rise skips the long envs, a descent the short ones, a hold rebalances the mix
    for level, mix, regime in [(RISE, 1.0, RISE), (RISE, 0.5, RISE), (RISE, 1.5, HOLD),
                               (DESCEND, 1.0, DESCEND), (DESCEND, 1.5, DESCEND), (DESCEND, 0.5, HOLD),
                               (HOLD, 1.0, HOLD), (HOLD, 0.5, RISE), (HOLD, 1.5, DESCEND),
                               (IDLE, 0.5, HOLD), (IDLE, 1.5, HOLD), (RISE, None, RISE), (HOLD, None, HOLD)]:
        assert advance_async(state, level, mix, 300, params).regime == regime, (level, mix)
    assert advance_async(state, RISE, 1.0, 300, params).price == pytest.approx(1.05)
    assert advance_async(state, DESCEND, 1.0, 300, params).price == pytest.approx(0.817 ** 0.1)   # -2 % per tick
    assert advance_async(PriceState(3.9, 1.0), RISE, 0.0, 300, params).price == 4.0
    assert advance_async(PriceState(0.05, 1.0), DESCEND, 1.0, 3e6, params).price == 0.05


def test_the_controller_reads_the_queue_persists_and_prices_per_token(book):
    clock = [NOW]
    prices = AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED, clock=lambda: clock[0])
    assert prices.token_price(ENV) == pytest.approx(NOMINAL) and not prices.due()
    clock[0] += 300
    assert prices.due()
    assert all(d.regime == HOLD for d in prices.tick(QueueTick({ENV: 9_000})).values())   # no batch, no wait
    clock[0] += 300
    decisions = prices.tick(QueueTick({ENV: 9_000, OTHER: 1_000}, batches=3, backlog=30))  # level hold
    assert decisions[ENV].regime == DESCEND and decisions[OTHER].regime == RISE          # mix 1.8 and 0.2
    assert prices.token_price(OTHER) == pytest.approx(1.05 * NOMINAL) and not prices.due()
    assert prices.tick(QueueTick()) == {}                                                # no time elapsed
    again = AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED, clock=lambda: clock[0])
    assert again.multiplier(OTHER) == pytest.approx(1.05)                                # restart-safe
    signal = again.board()[OTHER]["signal"]
    assert again.board()[OTHER]["regime"] == RISE and signal["level"] == HOLD
    assert signal["mix"] == pytest.approx(0.2) and signal["backlog"] == pytest.approx(10)
    clock[0] += 300
    decisions = prices.tick(QueueTick({ENV: 9_000}, batches=3, backlog=0, starved_s=60.0))  # OTHER dry, starved
    assert decisions[ENV].regime == RISE and decisions[OTHER].regime == RISE            # ENV takes all: mix 1
    clock[0] += 300
    decisions = prices.tick(QueueTick({ENV: 5_000, OTHER: 5_000}, batches=3, backlog=600))  # 200 left: too much
    assert {d.regime for d in decisions.values()} == {DESCEND}
    with pytest.raises(ValueError):
        AsyncPriceController(book, {ENV: 0.7}, CAP, PLANNED)


class Archives:
    def __init__(self):
        self.docs = {}

    async def write(self, task_id, work, entry, doc):
        self.docs[(task_id, work, entry)] = doc


def test_a_closed_period_settles_once_and_feeds_the_period_weights(book):
    subs = service(book, probation_groups=0)
    period = cp.period_of(NOW)
    ledger = ledger_for(book, subs, slack_s=60)
    for i, hotkey in enumerate("AAB"):
        trained(book, subs, 1 + i, hotkey=hotkey)
    ledger.ingest()
    assert ledger.settle(cp.period_end(period) + 30) == []            # within the slack: not closed
    later = cp.period_end(period) + 61
    docs = ledger.settle(later)
    assert [d["work_period"] for d in docs] == [period] and docs[0]["entry_period"] == period + 2
    unit = GROUP_TOKENS * NOMINAL
    assert docs[0]["scale"] == 1.0 and docs[0]["rewards_by_hotkey"] == {"A": pytest.approx(2 * unit),
                                                                       "B": pytest.approx(unit)}
    assert ledger.settle(later) == []                                  # once
    trained(book, subs, 9, hotkey="B", now=NOW + 1)                    # lands after its period settled
    ledger.ingest()
    docs = ledger.settle(cp.period_end(period + 1) + 61)
    assert [(d["work_period"], d["rewards_by_hotkey"]) for d in docs] == [(period + 1, {"B": pytest.approx(unit)})]
    archives = Archives()
    assert asyncio.run(ledger.publish(archives, now=cp.period_end(period + 5))) == 2
    assert ledger.unpublished() == [] and asyncio.run(ledger.publish(archives)) == 0
    assert sorted(archives.docs) == [("rl-async", period, period + 7), ("rl-async", period + 1, period + 7)]
    weights = cp.replay(list(archives.docs.values()), period + 7)      # what weight_only._period_weights replays
    assert weights == {"A": pytest.approx(cp.PERIOD_ALPHA * 2 * unit), "B": pytest.approx(cp.PERIOD_ALPHA * 2 * unit)}
    assert ledger.balances()["A"] == {"accrued": 0.0, "paid": pytest.approx(2 * unit), "forfeited": 0.0}


def test_a_period_never_pays_more_than_the_cap(book):
    subs = service(book, probation_groups=0)
    prices = AsyncPriceController(book, {ENV: 1.0}, CAP, 100.0, clock=lambda: NOW)   # one group > the cap
    ledger = ledger_for(book, subs, prices)
    trained(book, subs, 1, hotkey="A")
    trained(book, subs, 2, hotkey="A")
    trained(book, subs, 3, hotkey="B")
    ledger.ingest()
    (doc,) = ledger.settle(cp.period_end(cp.period_of(NOW)) + 600)
    assert doc["accrued_total"] == pytest.approx(3 * GROUP_TOKENS * CAP / 100)
    assert doc["scale"] == pytest.approx(CAP / doc["accrued_total"])
    assert sum(doc["rewards_by_hotkey"].values()) == pytest.approx(CAP)
    assert doc["rewards_by_hotkey"]["A"] == pytest.approx(2 * doc["rewards_by_hotkey"]["B"])


def test_the_pay_route_publishes_prices_and_balances(book):
    pytest.importorskip("bittensor")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.validator.rl_lease_routes import build_rl_lease_router
    from reliquary.validator.rl_leases import LeaseCollector
    from tests.unit.test_rl_lease_routes import MINER, VALIDATOR

    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs)
    trained(book, subs, 1, hotkey=MINER.ss58_address)
    ledger.ingest()
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=VALIDATOR.ss58_address, clock=lambda: NOW,
                                             pay=ledger))
    client = TestClient(app)
    body = client.get("/rl/pay").json()
    assert body["policy"] == "pay-per-matched-token/v1" and body["floor"] == 0.95
    assert body["environments"][ENV]["token_price"] == pytest.approx(NOMINAL)
    assert body["hotkeys"][MINER.ss58_address]["accrued"] == pytest.approx(GROUP_TOKENS * NOMINAL)
    one = client.get("/rl/pay", params={"hotkey": VALIDATOR.ss58_address}).json()
    assert one["hotkeys"] == {VALIDATOR.ss58_address: {"accrued": 0.0, "paid": 0.0, "forfeited": 0.0}}
    assert client.get("/rl/pay", params={"hotkey": "nope"}).status_code == 422

    from reliquary.validator.server import ValidatorServer

    server = ValidatorServer()
    server.mount_rl_lease_router(build_rl_lease_router(
        book=book, collector=LeaseCollector(book, interval_s=0.0), validator_hotkey=VALIDATOR.ss58_address,
        clock=lambda: NOW, pay=ledger))
    assert [r.path for r in server.app.routes if getattr(r, "path", "") == "/rl/pay"] == ["/rl/pay"]
    assert TestClient(server.app).get("/rl/pay").json()["policy"] == "pay-per-matched-token/v1"


def test_the_public_board_is_rebuilt_at_most_every_15_seconds(book):
    from reliquary.validator.rl_ledger import BOARD_CACHE_S

    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs)
    first = ledger.board(NOW)
    trained(book, subs, 1)
    ledger.ingest()
    assert ledger.board(NOW + BOARD_CACHE_S - 1) is first and first["hotkeys"] == {}     # cached: no recompute
    fresh = ledger.board(NOW + BOARD_CACHE_S, hotkey="A")
    assert fresh["hotkeys"]["A"]["accrued"] > 0 and list(fresh["hotkeys"]) == ["A"]
    assert ledger.board(NOW + BOARD_CACHE_S)["hotkeys"]["A"] == fresh["hotkeys"]["A"]
