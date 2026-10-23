"""The async RL pay (P5): flat pay per matched token of trained groups above the floor, 25 % first scans after
audit, ban forfeiture, the per-env async price controller, period settlement into weight-ready archives, and the
async ``GET /rl/pay``."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.emission_price import PriceState
from reliquary.validator.rl_ledger import FORFEITED, LedgerConfig, RlPayLedger
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import LeaseBook
from reliquary.validator.rl_price import (
    DESCEND, HOLD, RISE, AsyncPriceController, AsyncPriceParams, EnvSignal, QueueTick, advance_async,
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


NOMINAL = CAP / PLANNED                     # every env's price per matched token at multiplier 1 (start pool = cap)


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
    trained(book, subs, 1, [0.99] * 10 + [0.899] * 6, matched=[per] * 16)
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
    ledger.ingest()
    subs.blacklist("A", NOW + 86410, "manual", now=NOW + 10)
    subs.report_training(pending.lease_id, rollout_match=[1.0] * 16, accepted=True, now=NOW + 20)  # during the ban
    ledger.ingest()
    assert [(h, s) for h, _, _, _, s in amounts(book)] == [("A", FORFEITED), ("B", "accrued"), ("A", FORFEITED)]
    board = ledger.balances()
    assert board["A"]["accrued"] == 0 and board["A"]["forfeited"] > 0 and board["A"]["banned_until"] == NOW + 86410
    assert board["B"]["accrued"] == pytest.approx(GROUP_TOKENS * NOMINAL)


def test_the_async_price_moves_on_supply_against_its_share_of_the_trainer_speed():
    params = AsyncPriceParams()
    state = PriceState(price=1.0, last_good=1.0)
    short = advance_async(state, EnvSignal(300, supply_tps=400, demand_tps=500), params)          # r = 0.8
    assert (short.regime, short.price, short.r) == (RISE, pytest.approx(1.05), pytest.approx(0.8))
    hold = advance_async(state, EnvSignal(300, 500, 500), params)
    assert hold.regime == HOLD and hold.price == 1.0
    over = advance_async(state, EnvSignal(300, 600, 500), params)                                 # r = 1.2
    assert over.regime == DESCEND and over.price == pytest.approx(0.817 ** 0.1)                  # -2 % per tick
    full = advance_async(state, EnvSignal(300, 500, 500, full=True), params)                      # the cap refused it
    assert full.regime == DESCEND
    short_but_full = advance_async(state, EnvSignal(300, 100, 500, full=True), params)            # the mix still moves
    assert short_but_full.regime == RISE
    # the queue's guards: a full queue raises no price, a waiting trainer lowers none
    assert advance_async(state, EnvSignal(300, 100, 500, capped=True), params).regime == HOLD
    assert advance_async(state, EnvSignal(300, 600, 500, starved=True), params).regime == HOLD
    assert advance_async(state, EnvSignal(300, 500, 500, full=True, starved=True), params).regime == HOLD
    no_speed = advance_async(state, EnvSignal(300, 100, None), params)                           # trainer unknown
    assert no_speed.regime == HOLD and no_speed.r is None
    capped = advance_async(PriceState(3.9, 1.0), EnvSignal(300, 0, 500), params)
    assert capped.price == 4.0
    floored = advance_async(PriceState(0.02, 1.0), EnvSignal(3e6, 10**9, 1), params)
    assert floored.price == 0.02
    # a waiting trainer: the mix still moves (relative = supply share / target share), within the band it does not
    assert advance_async(state, EnvSignal(300, 600, 500, starved=True, relative=1.05), params).regime == HOLD
    over = advance_async(state, EnvSignal(300, 600, 500, starved=True, relative=3.0), params)
    assert over.regime == DESCEND and over.price == pytest.approx(1 / 1.02)
    short = advance_async(state, EnvSignal(300, 100, 500, starved=True, relative=0.5), params)
    assert short.regime == RISE and short.price == pytest.approx(1.05 * 1.02)
    both_short = advance_async(state, EnvSignal(300, 100, 500, starved=True, relative=1.3), params)
    assert both_short.regime == RISE and both_short.price == pytest.approx(1.05 / 1.02)  # short, less than the rest
    assert advance_async(state, EnvSignal(300, 100, 500, capped=True, starved=True, relative=0.5),
                         params).regime == HOLD                                         # the full queue still wins


def test_the_controller_reads_the_trainer_speed_from_the_queue_persists_and_prices_per_token(book):
    clock = [NOW]
    prices = AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED, clock=lambda: clock[0])
    assert prices.token_price(ENV) == pytest.approx(NOMINAL) and not prices.due()
    clock[0] += 300
    assert prices.due()
    assert all(d.regime == HOLD for d in prices.tick(QueueTick({ENV: 9_000})).values())   # no speed yet: hold
    clock[0] += 300
    # the first take opens a batch: its work is measured at the next take
    assert all(d.regime == HOLD for d in prices.tick(QueueTick({ENV: 9_000}, 150_000)).values())
    assert prices.trainer_speed() is None
    clock[0] += 300
    # the next take, 300 s later with 10 s of waiting: 150,000 tokens over 290 s of work, 500 per env
    decisions = prices.tick(QueueTick({ENV: 300 * 1_200, OTHER: 300 * 100}, 150_000, 10.0), {ENV})  # r 2.4, 0.2
    assert prices.trainer_speed() == pytest.approx(150_000 / 290)
    assert decisions[ENV].regime == DESCEND and decisions[OTHER].regime == HOLD           # capped: no rise
    clock[0] += 300
    decisions = prices.tick(QueueTick({ENV: 300 * 1_200, OTHER: 300 * 100}, 150_000, 150.0))
    assert prices.trainer_speed() == pytest.approx(300_000 / 440, rel=0.01)               # sums over the batches
    # starved: ENV descends on the mix only (12x OTHER's supply for the same share), OTHER rises twice
    assert decisions[ENV].regime == DESCEND and decisions[OTHER].regime == RISE
    assert decisions[ENV].price == pytest.approx(0.817 ** 0.1 / 1.02)
    assert prices.token_price(OTHER) == pytest.approx(1.05 * 1.02 * NOMINAL) and not prices.due()
    assert prices.tick(QueueTick()) == {}                                               # no time elapsed
    again = AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED, clock=lambda: clock[0])
    assert again.multiplier(OTHER) == pytest.approx(1.05 * 1.02)                        # restart-safe
    signal = again.board()[OTHER]["signal"]
    assert again.board()[OTHER]["regime"] == RISE and signal["full"] is False and signal["starved_s"] == 150.0
    assert signal["ratio"] == pytest.approx(300 * 100 / 300 / signal["demand_tps"])
    clock[0] += 300
    decisions = prices.tick(QueueTick({ENV: 300 * 1_000}, 300_000))                     # OTHER dry: ENV takes all
    assert decisions[OTHER].regime == RISE and decisions[ENV].r == pytest.approx(1_000 / prices.trainer_speed())
    clock[0] += 300
    before = prices.multiplier(ENV)
    decisions = prices.tick(QueueTick({ENV: 300 * 500, OTHER: 300 * 100}, 0, 300.0), {ENV})  # waited, took nothing
    assert prices.trainer_speed() is not None                                          # alive: speed kept
    assert decisions[ENV].price == pytest.approx(before / 1.02)                         # the mix's step, full ignored
    clock[0] += 4 * 3600 + 1                                                             # neither took nor waited
    assert again.trainer_speed() is None
    with pytest.raises(ValueError):
        AsyncPriceController(book, {ENV: 0.7}, CAP, PLANNED)


def test_a_step_long_trainer_keeps_its_speed_between_takes(book):
    """The trainer takes one batch at the start of a 90 min step: the speed is that batch over the step, measured at
    the next take and held in between (no hold of the prices), never one batch over one tick
    (teutonic-rl-async-v1, 2026-10-23)."""
    clock = [NOW]
    prices = AsyncPriceController(book, {ENV: 0.5, OTHER: 0.5}, CAP, PLANNED, clock=lambda: clock[0])
    batch, ticks = 5_400_000, 18                                                        # 1,000 tokens/s over 90 min
    speeds, regimes = [], []
    for k in range(4 * ticks):
        clock[0] += 300
        if k % ticks == 0:                                                              # a take in two halves
            prices.tick(QueueTick({ENV: 300 * 400, OTHER: 300 * 400}, batch // 2, 100.0))
            clock[0] += 300
            decisions = prices.tick(QueueTick({ENV: 300 * 400, OTHER: 300 * 400}, batch - batch // 2, 200.0))
        else:
            decisions = prices.tick(QueueTick({ENV: 300 * 400, OTHER: 300 * 400}))
        speeds.append(prices.trainer_speed())
        regimes.append({d.regime for d in decisions.values()})
    assert speeds[:ticks] == [None] * ticks                                             # one batch open: no speed
    late = speeds[ticks:]
    assert all(v is not None and 950 < v < 1_150 for v in late), late                   # ~1,000, not 18,000
    assert all(r == {RISE} for r in regimes[ticks:])                                    # 400 of 500 per env: short


class Archives:
    def __init__(self):
        self.docs = {}

    async def write(self, task_id, work, entry, doc):
        self.docs[(task_id, work, entry)] = doc

    async def read(self, task_id, work, entry):
        return self.docs.get((task_id, work, entry))

    async def list(self, task_id):
        return sorted((work, entry) for task, work, entry in self.docs if task == task_id)


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


@pytest.mark.parametrize("interruption", [TimeoutError, asyncio.CancelledError])
@pytest.mark.parametrize("commit_after_retry", [False, True])
@pytest.mark.parametrize("legacy_writer", [False, True])
async def test_publication_reconciles_after_restart(book, tmp_path, interruption, commit_after_retry, legacy_writer):
    from reliquary.validator.weight_only import WeightOnlyValidator

    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs, slack_s=0)
    trained(book, subs, 1)
    ledger.ingest()
    now = cp.period_end(cp.period_of(NOW)) + 1
    (doc,) = ledger.settle(now)

    class InterruptedArchives(Archives):
        interrupted = False
        pending = None

        async def write(self, *args):
            if not self.interrupted:
                self.interrupted = True
                if commit_after_retry:
                    self.pending = args
                else:
                    await super().write(*args)
                raise interruption()
            await super().write(*args)

    archives = InterruptedArchives()
    with pytest.raises(interruption):
        if legacy_writer:
            book.db.execute("UPDATE rl_pay_periods SET published=0")
            book.db.commit()
            await archives.write("rl-async", doc["work_period"], doc["entry_period"], doc)
        else:
            await ledger.publish(archives, now=now)
    assert ledger.unpublished() == [doc]
    assert ledger.board(now)["settled_periods"][0]["published"] is False
    book.close()

    restarted = LeaseBook(tmp_path / "leases.sqlite3", book.config,
                          quota=QuotaController(QuotaConfig(initial=64, ceiling=64)), clock=lambda: NOW)
    try:
        ledger = ledger_for(restarted, service(restarted, probation_groups=0), slack_s=0)
        later = cp.period_end(doc["entry_period"]) + 1
        assert await ledger.publish(archives, now=later) == 1
        assert ledger.unpublished() == []
        if archives.pending is not None:
            await Archives.write(archives, *archives.pending)
        assert await ledger.publish(archives, now=later + cp.PERIOD_SECONDS) == 0
        declared = {"rl-async": SimpleNamespace(params={"settlement": "period-ema-v1", "cap": CAP})}
        weights = await WeightOnlyValidator._period_weights(declared, archives=archives, now=later,
                                                           genesis=cp.PERIOD_EPOCH)
        expected = cp.replay([doc], cp.period_of(later))
        assert weights["rl-async"] == pytest.approx(expected)
        assert archives.docs == {("rl-async", doc["work_period"], doc["entry_period"]): doc}
        assert ledger.board(later)["settled_periods"][0]["published"] is True
    finally:
        restarted.close()


async def test_unattempted_stale_archive_can_move_after_restart(book, tmp_path):
    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs, slack_s=0)
    trained(book, subs, 1)
    ledger.ingest()
    (doc,) = ledger.settle(cp.period_end(cp.period_of(NOW)) + 1)
    assert ledger.board()["settled_periods"][0]["published"] is False
    book.close()

    restarted = LeaseBook(tmp_path / "leases.sqlite3", book.config,
                          quota=QuotaController(QuotaConfig(initial=64, ceiling=64)), clock=lambda: NOW)
    try:
        ledger = ledger_for(restarted, service(restarted, probation_groups=0), slack_s=0)
        later = cp.period_end(doc["entry_period"]) + 1
        archives = Archives()
        assert await ledger.publish(archives, now=later) == 1
        moved = {**doc, "entry_period": cp.period_of(later) + 1}
        assert archives.docs == {("rl-async", doc["work_period"], moved["entry_period"]): moved}
    finally:
        restarted.close()


@pytest.mark.parametrize("stored", ["unreadable", "different"])
async def test_publication_requires_matching_archive_before_moving_entry(book, stored):
    subs = service(book, probation_groups=0)
    ledger = ledger_for(book, subs, slack_s=0)
    trained(book, subs, 1)
    ledger.ingest()
    (doc,) = ledger.settle(cp.period_end(cp.period_of(NOW)) + 1)

    class UnavailableArchives(Archives):
        async def read(self, *args):
            if stored == "unreadable":
                raise RuntimeError("archive unavailable")
            return {**doc, "groups": doc["groups"] + 1}

    archives = UnavailableArchives()
    with pytest.raises(RuntimeError):
        await ledger.publish(archives, now=cp.period_end(doc["entry_period"]) + 1)
    assert ledger.unpublished() == [doc]
    assert archives.docs == {}


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


@pytest.mark.parametrize("k", [None, 2.0])
def test_the_per_env_guard_holds_an_env_to_k_times_its_share_of_the_cap(book, k):
    subs = service(book, probation_groups=0)
    prices = AsyncPriceController(book, {ENV: 0.2, OTHER: 0.8}, CAP, 100.0, clock=lambda: NOW)  # ENV alone > cap
    ledger = ledger_for(book, subs, prices, env_pay_k=k)
    trained(book, subs, 1, hotkey="A")
    trained(book, subs, 2, hotkey="B")
    ledger.ingest()
    (doc,) = ledger.settle(cp.period_end(cp.period_of(NOW)) + 600)
    paid = sum(doc["rewards_by_hotkey"].values())
    if k is None:
        assert paid == pytest.approx(CAP) and "env_scale" not in doc                     # off: the archive unchanged
    else:
        assert paid == pytest.approx(2.0 * 0.2 * CAP)                                   # the surplus burns
        assert doc["env_scale"][ENV] == pytest.approx(0.4 * CAP * 100 / (2 * GROUP_TOKENS * CAP))
        assert doc["accrued_total"] == pytest.approx(0.4 * CAP) and doc["scale"] == 1.0
    assert doc["rewards_by_hotkey"]["A"] == pytest.approx(doc["rewards_by_hotkey"]["B"])
    with pytest.raises(ValueError):
        LedgerConfig(task_id="t", cap=CAP, env_pay_k=0.0)
    from reliquary.validator.rl_async_service import ENV_PAY_K_PARAM, rl_async_params

    assert rl_async_params(planned_tokens_per_period=1.0, env_pay_k=2.0)[ENV_PAY_K_PARAM] == 2.0
    assert ENV_PAY_K_PARAM not in rl_async_params(planned_tokens_per_period=1.0)


def set_multiplier(book, env, price):
    book.db.execute("INSERT INTO rl_prices(environment, price, last_good, regime, at, signal) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(environment) DO UPDATE SET price=excluded.price", (env, price, 1.0, RISE, NOW, "{}"))
    book.db.commit()


def test_every_env_has_one_nominal_and_the_target_mix_pays_the_start_pool(book):
    targets = {ENV: 0.8, OTHER: 0.2}
    prices = AsyncPriceController(book, targets, CAP, PLANNED, start_pool=0.2, clock=lambda: NOW)
    assert prices.nominal == pytest.approx(0.2 / PLANNED)
    assert prices.token_price(ENV) == prices.token_price(OTHER) == pytest.approx(prices.nominal)
    assert prices.token_price("unknown") == 0.0
    paid = {env: share * PLANNED * prices.token_price(env) for env, share in targets.items()}
    assert paid == {ENV: pytest.approx(0.8 * 0.2), OTHER: pytest.approx(0.2 * 0.2)}     # start_pool x share
    assert sum(paid.values()) == pytest.approx(0.2)
    assert prices.board()[OTHER]["per_million_tokens"] == pytest.approx(1e6 * 0.2 / PLANNED)
    legacy = AsyncPriceController(book, targets, CAP, PLANNED, clock=lambda: NOW)       # no start pool: the cap
    assert legacy.start_pool == CAP and legacy.token_price(OTHER) == pytest.approx(CAP / PLANNED)
    for bad in (0.0, -0.1, CAP * 1.01):
        with pytest.raises(ValueError, match="start_pool"):
            AsyncPriceController(book, targets, CAP, PLANNED, start_pool=bad)


def test_the_multiplier_ceiling_lets_scarcity_reach_the_cap(book):
    assert AsyncPriceController(book, {ENV: 1.0}, CAP, PLANNED, start_pool=CAP / 2).params.base.cap == 4.0
    low = AsyncPriceController(book, {ENV: 1.0}, CAP, PLANNED, start_pool=CAP / 10)
    assert low.params.base.cap == pytest.approx(10.0)
    assert advance_async(PriceState(9.9, 1.0), EnvSignal(300, 0, 500), low.params).price == pytest.approx(10.0)


@pytest.mark.parametrize("multiplier, total, scale", [(1.0, CAP / 2, 1.0), (1.5, 0.75 * CAP, 1.0),
                                                      (3.0, CAP, CAP / (1.5 * CAP))])
def test_scarcity_raises_the_total_above_the_start_pool_up_to_the_cap(book, multiplier, total, scale):
    subs = service(book, probation_groups=0)
    # The trainer plans three groups a period: at multiplier 1 they pay the start pool, half the cap.
    prices = AsyncPriceController(book, {ENV: 1.0}, CAP, 3 * GROUP_TOKENS, start_pool=CAP / 2, clock=lambda: NOW)
    set_multiplier(book, ENV, multiplier)
    ledger = ledger_for(book, subs, prices)
    for prompt, hotkey in ((1, "A"), (2, "A"), (3, "B")):
        trained(book, subs, prompt, hotkey=hotkey)
    ledger.ingest()
    (doc,) = ledger.settle(cp.period_end(cp.period_of(NOW)) + 600)
    assert doc["accrued_total"] == pytest.approx(multiplier * CAP / 2)
    assert doc["scale"] == pytest.approx(scale) and sum(doc["rewards_by_hotkey"].values()) == pytest.approx(total)
    assert ledger.board(NOW)["start_pool"] == CAP / 2 and ledger.board(NOW)["cap"] == CAP


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
    assert body["policy"] == "pay-per-matched-token/v1" and body["floor"] == 0.90
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
    assert fresh["environments"][ENV]["recent"] == {"groups": 1, "trained_share": 1.0, "first_scan_share": 0.0}
    assert first["environments"][ENV]["recent"] == {"groups": 0, "trained_share": None, "first_scan_share": None}
    assert ledger.board(NOW + BOARD_CACHE_S)["hotkeys"]["A"] == fresh["hotkeys"]["A"]
