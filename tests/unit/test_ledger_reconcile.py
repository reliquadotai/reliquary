"""Ledger obligations against what the chain paid: the schedule is the weight
setter's own replay, so a chain that paid the weights exactly reconciles to
the rao, and a hotkey paid 2 % off fails the 1 % criterion."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.ledger import obligations as ob
from reliquary.ledger import reconcile as rc
from reliquary.validator import corpus_periods as cp
from reliquary.validator.weight_only import WeightOnlyValidator

GENESIS = cp.PERIOD_EPOCH
E = 147.60082398718242
TASK = "order-gen-recon"
CEILINGS = {TASK: [ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=1e-8,
                              pricing=ob.PRICING_FIXED)]}
FIRST = 23_700


class _Archives:
    def __init__(self, refs):
        self.refs = refs

    async def list(self, task_id):
        return sorted((r.work_period, r.entry_period) for r in self.refs if r.task_id == task_id)

    async def read(self, task_id, work, entry):
        for ref in self.refs:
            if (ref.task_id, ref.work_period, ref.entry_period) == (task_id, work, entry):
                return dict(ref.document)
        return None


def _refs():
    """Thirty periods of work by three hotkeys whose shares change over time."""
    refs = []
    for i in range(30):
        work = FIRST + i
        rewards = {"5A": 0.002 + 0.0001 * i, "5B": 0.004 if i < 15 else 0.0005}
        if i % 3 == 0:
            rewards["5C"] = 0.001
        doc = {"task_id": TASK, "work_period": work, "rewards_by_hotkey": rewards, "cap": 0.02,
               "budget": 0.02, "max_price_per_token": 1e-8, "tokens": 10**6}
        refs.append(ob.ArchiveRef(task_id=TASK, work_period=work, entry_period=work + 1 + (i % 2),
                                  document=doc, settled_at=cp.period_end(work) + 500))
    return refs


def _entry():
    return SimpleNamespace(params={"cap": 0.02, "settlement": cp.SETTLEMENT_PERIOD_EMA,
                                   "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0})


def _native_weights(refs, period):
    """What a weight-set at ``period`` pays, the way ``submit_once`` combines it."""
    declared = {TASK: _entry()}
    now = cp.period_end(period - 1, GENESIS) + 60
    periods = asyncio.run(WeightOnlyValidator._period_weights(declared, archives=_Archives(refs), now=now,
                                                              genesis=GENESIS))
    return WeightOnlyValidator._replay_ema([], caps=WeightOnlyValidator._caps_by_task(declared),
                                           floors={TASK: (0.0, 0.0)}, periods=periods,
                                           period_caps=WeightOnlyValidator._period_caps_by_task(declared))


def _epochs(refs, *, lag, first_period, count, offset=700.0, skew=None):
    """Epochs whose pay is exactly the weights replayed ``lag`` epochs earlier, times E."""
    out = []
    for k in range(count):
        at = cp.period_end(first_period + k - 1, GENESIS) + offset
        period = rc.weight_period(at, lag_epochs=lag)
        pay = {hk: w * E for hk, w in _native_weights(refs, period).items()}
        if skew:
            pay = {hk: v * skew.get(hk, 1.0) for hk, v in pay.items()}
        out.append(rc.EpochPay(block=1000 + 360 * k, at=at, pay=pay))
    return out


def test_the_ema_factor_is_the_weight_setters_replay():
    for age in range(-1, cp.REPLAY_DEPTH + 3):
        replayed = cp.replay([{"entry_period": 10, "rewards_by_hotkey": {"h": 1.0}}], 10 + age)
        assert rc.ema_factor(age) == pytest.approx(replayed.get("h", 0.0), abs=1e-9)


def test_the_schedule_is_the_weight_setters_replay_in_obligation_alpha():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    for period in (FIRST + 2, FIRST + 10, FIRST + 29, FIRST + 40):
        native = _native_weights(refs, period)
        scheduled = rc.scheduled_alpha(ledger.obligations, [period])
        assert set(scheduled) == set(native)
        for hotkey, weight in native.items():
            # Obligations are rounded down to the rao: at most a rao below the share x E.
            assert scheduled[hotkey] == pytest.approx(weight * E, abs=1e-8)


def test_a_chain_paying_the_weights_exactly_reconciles_and_passes():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    epochs = _epochs(refs, lag=1, first_period=FIRST + 26, count=30)
    start, end = epochs[0].at, epochs[-1].at + 1
    result = rc.reconcile(ledger.obligations, epochs, start=start, end=end, lag_epochs=1,
                          earliest_entry=FIRST + 1)
    assert result["criterion"]["pass"] is True
    assert result["criterion"]["hotkeys_checked"] == 3
    for row in result["hotkeys"]:
        assert abs(row["gap"]) < 1e-6
    assert result["window"]["epochs"] == 30 and result["window"]["history_too_short"] is False
    fit = result["lag_fit"]
    assert min(fit, key=lambda lag: fit[lag]) == "1"


def test_a_hotkey_paid_two_percent_short_fails_the_criterion_alone():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    epochs = _epochs(refs, lag=1, first_period=FIRST + 26, count=30, skew={"5B": 0.98})
    result = rc.reconcile(ledger.obligations, epochs, start=epochs[0].at, end=epochs[-1].at + 1, lag_epochs=1)
    rows = {row["hotkey"]: row for row in result["hotkeys"]}
    assert rows["5B"]["within"] is False and rows["5B"]["gap"] == pytest.approx(-0.02, abs=1e-6)
    assert rows["5A"]["within"] is True and rows["5C"]["within"] is True
    assert result["criterion"]["pass"] is False and result["criterion"]["hotkeys_within"] == 2


def test_the_wrong_lag_shows_as_a_gap_and_the_fit_names_the_right_one():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    epochs = _epochs(refs, lag=2, first_period=FIRST + 26, count=30)
    wrong = rc.reconcile(ledger.obligations, epochs, start=epochs[0].at, end=epochs[-1].at + 1, lag_epochs=0)
    assert wrong["criterion"]["pass"] is False
    fit = wrong["lag_fit"]
    assert min(fit, key=lambda lag: fit[lag]) == "2"


def test_pay_no_obligation_explains_is_counted_apart_and_hotkeys_below_the_floor_are_not_checked():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    epochs = _epochs(refs, lag=1, first_period=FIRST + 26, count=10)
    epochs[3] = rc.EpochPay(block=epochs[3].block, at=epochs[3].at, pay={**epochs[3].pay, "5Stranger": 0.5})
    result = rc.reconcile(ledger.obligations, epochs, start=epochs[0].at, end=epochs[-1].at + 1, lag_epochs=1,
                          min_alpha=10**6)
    assert result["totals"]["onchain_unscheduled_alpha"] == pytest.approx(0.5)
    stranger = next(row for row in result["hotkeys"] if row["hotkey"] == "5Stranger")
    assert stranger["gap"] is None and stranger["checked"] is False
    assert result["criterion"]["hotkeys_checked"] == 0 and result["criterion"]["pass"] is False


def test_a_window_reaching_past_the_history_read_says_so():
    refs = _refs()
    ledger = ob.build_ledger(refs, ceilings=CEILINGS)
    epochs = _epochs(refs, lag=1, first_period=FIRST + 5, count=5)
    result = rc.reconcile(ledger.obligations, epochs, start=epochs[0].at, end=epochs[-1].at + 1, lag_epochs=1,
                          earliest_entry=FIRST + 1)
    assert result["window"]["history_too_short"] is True and result["criterion"]["pass"] is False


def test_epochs_outside_the_window_are_left_out():
    epochs = [rc.EpochPay(block=1, at=100.0, pay={"h": 1.0}), rc.EpochPay(block=2, at=200.0, pay={"h": 2.0}),
              rc.EpochPay(block=3, at=300.0, pay={"h": 4.0})]
    result = rc.reconcile([], epochs, start=150.0, end=300.0)
    assert result["window"]["epochs"] == 1 and result["totals"]["onchain_alpha"] == 2.0
    with pytest.raises(ValueError):
        rc.reconcile([], epochs, start=0.0, end=1.0, lag_epochs=-1)
