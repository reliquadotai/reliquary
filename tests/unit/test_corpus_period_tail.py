"""A period task's cap governs the periods still to be worked, never the pay
already earned: lowering it (to 0 when the job is finished) lets the earned
tail run out, and pays nothing new."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.weight_only import WeightOnlyValidator

GENESIS = 1_692_803_367.0
FLOORLESS = (0.0, 0.0)


class _PeriodArchives:
    def __init__(self, docs):
        self.docs = docs

    async def list(self, task_id):
        return sorted((w, e) for (t, w, e) in self.docs if t == task_id)

    async def read(self, task_id, work, entry):
        return self.docs.get((task_id, work, entry))


def _entry(cap, **extra):
    params = {"cap": cap, "settlement": cp.SETTLEMENT_PERIOD_EMA,
              "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0, **extra}
    return SimpleNamespace(params=params)


def _corpus_entry(cap, settlement):
    from reliquary.shared.task_registry import TaskEntry

    params = {"start": cap, "decay": 0.98, "rounds_per_step": 1, "deadband": 0.0,
              "snap": 0.0, "floor": cap, "cap": cap, "median_rounds": 1,
              "last_good_fills": 1}
    if settlement:
        params["settlement"] = settlement
    return TaskEntry(task_id="corpus-x", profile_id="p", profile_sha256="0" * 64,
                     mechanism="corpus-generation", params=params, status="active",
                     retired_at=None, job_id="job-x")


def _weights_at(period, declared, archives):
    """What one weight-set pays, the way ``submit_once`` combines it."""
    now = GENESIS + period * cp.PERIOD_SECONDS + 5
    periods = asyncio.run(WeightOnlyValidator._period_weights(
        declared, archives=archives, now=now, genesis=GENESIS))
    return WeightOnlyValidator._replay_ema(
        [], caps=WeightOnlyValidator._caps_by_task(declared),
        floors={t: FLOORLESS for t in declared}, periods=periods,
        period_caps=WeightOnlyValidator._period_caps_by_task(declared))


def test_an_archive_records_the_cap_it_was_settled_under():
    from tests.unit.test_corpus_period_settlement import Archives, Records, at, settler, verdict

    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=0.07).settle_once())
    (doc,) = archives.docs.values()
    assert doc["cap"] == 0.07
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(0.07)}


def test_a_period_settled_at_cap_zero_writes_nothing_and_pays_nothing():
    from tests.unit.test_corpus_period_settlement import Archives, Records, at, settler, verdict

    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=0.0).settle_once())
    assert archives.docs == {}
    assert records.state["settled"] == ["a"] and records.state.get("pending") is None


def test_lowering_a_period_task_cap_keeps_its_earned_cap_as_the_tail_bound():
    from reliquary.shared.task_registry import set_cap

    entry = _corpus_entry(cap=0.1, settlement=cp.SETTLEMENT_PERIOD_EMA)
    lowered = set_cap({entry.task_id: entry}, entry.task_id, 0.0)[entry.task_id]
    assert lowered.params["cap"] == 0.0 and lowered.params["floor"] == 0.0
    assert lowered.params["tail_cap"] == 0.1
    # Raised again, then lowered: the bound is the highest cap it was paid at.
    raised = set_cap({entry.task_id: lowered}, entry.task_id, 0.05)[entry.task_id]
    assert raised.params["tail_cap"] == 0.1
    again = set_cap({entry.task_id: raised}, entry.task_id, 0.0)[entry.task_id]
    assert again.params["tail_cap"] == 0.1


def test_a_window_task_cap_change_writes_no_tail_bound():
    from reliquary.shared.task_registry import set_cap

    entry = _corpus_entry(cap=0.1, settlement=None)
    lowered = set_cap({entry.task_id: entry}, entry.task_id, 0.0)[entry.task_id]
    assert "tail_cap" not in lowered.params


def test_a_tail_bound_out_of_range_is_refused():
    from reliquary.shared.task_registry import RegistryError, validate_entry

    entry = _corpus_entry(cap=0.1, settlement=cp.SETTLEMENT_PERIOD_EMA)
    for bad in (-0.1, 1.5, "0.1", float("nan")):
        with pytest.raises(RegistryError):
            validate_entry(replace(entry, params={**entry.params, "tail_cap": bad}))


def test_cap_zero_mid_tail_pays_what_was_earned_and_nothing_more():
    """The job is finished: its last archives are still entering when the cap
    goes to 0. The tail runs out at the archives' own cap, in full."""
    cap = 0.1
    docs, earned = {}, {}
    for work in range(10):
        rewards = {"a": cap * 0.75, "b": cap * 0.25} if work % 2 else {"c": cap}
        docs[("t", work, work + 2)] = {"rewards_by_hotkey": rewards, "cap": cap,
                                       "entry_period": work + 2}
        for hk, r in rewards.items():
            earned[hk] = earned.get(hk, 0.0) + r
    archives = _PeriodArchives(docs)
    open_ = {"t": _entry(cap)}
    closed = {"t": _entry(0.0, tail_cap=cap)}
    paid: dict[str, float] = {}
    for period in range(0, 80):
        declared = open_ if period < 9 else closed  # cap 0 while 2 archives still queue
        for hk, w in _weights_at(period, declared, archives).items():
            paid[hk] = paid.get(hk, 0.0) + w
    lost = (1 - cp.PERIOD_ALPHA) ** (cp.REPLAY_DEPTH + 1)
    for hk, owed in earned.items():
        assert paid[hk] <= owed * (1 + 1e-12)
        assert paid[hk] == pytest.approx(owed, rel=lost * 1.5)
    # And it ends: nothing is paid past the replay depth of the last entry.
    assert _weights_at(11 + cp.REPLAY_DEPTH + 1, closed, archives) == {}


def test_no_period_worked_after_the_cap_went_to_zero_is_paid():
    """New work settled at cap 0 writes no archive, so it can never pay."""
    from tests.unit.test_corpus_period_settlement import Archives, Records, at, settler, verdict

    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=0.1).settle_once())
    records.verdicts["b"] = verdict("b", "b", 10, at(6))
    asyncio.run(settler(records, archives, at(8, 1000), cap=0.0).settle_once())
    assert [doc["rewards_by_hotkey"] for doc in archives.docs.values()] == [
        {"a": pytest.approx(0.1)}]
    assert sorted(records.state["settled"]) == ["a", "b"]


def test_an_archive_never_pays_past_the_highest_cap_the_registry_gave_the_task():
    docs = {("t", 30, 32): {"rewards_by_hotkey": {"m": 0.5}, "cap": 0.5}}
    now = GENESIS + 32 * cp.PERIOD_SECONDS + 5
    weights = asyncio.run(WeightOnlyValidator._period_weights(
        {"t": _entry(0.0, tail_cap=0.1)}, archives=_PeriodArchives(docs), now=now,
        genesis=GENESIS))
    assert weights["t"]["m"] == pytest.approx(cp.PERIOD_ALPHA * 0.1)


def test_an_archive_never_pays_past_the_cap_it_records():
    docs = {("t", 30, 32): {"rewards_by_hotkey": {"m": 0.08}, "cap": 0.05}}
    now = GENESIS + 32 * cp.PERIOD_SECONDS + 5
    weights = asyncio.run(WeightOnlyValidator._period_weights(
        {"t": _entry(0.1)}, archives=_PeriodArchives(docs), now=now, genesis=GENESIS))
    assert weights["t"]["m"] == pytest.approx(cp.PERIOD_ALPHA * 0.05)


def test_an_archive_written_before_caps_were_recorded_is_held_to_the_tail_bound():
    docs = {("t", 30, 32): {"rewards_by_hotkey": {"m": 0.1}}}
    now = GENESIS + 32 * cp.PERIOD_SECONDS + 5
    paid = asyncio.run(WeightOnlyValidator._period_weights(
        {"t": _entry(0.0, tail_cap=0.1)}, archives=_PeriodArchives(docs), now=now,
        genesis=GENESIS))
    assert paid["t"]["m"] == pytest.approx(cp.PERIOD_ALPHA * 0.1)
    # With no tail bound (a cap lowered by an older binary): as before, cut.
    cut = asyncio.run(WeightOnlyValidator._period_weights(
        {"t": _entry(0.0)}, archives=_PeriodArchives(docs), now=now, genesis=GENESIS))
    assert cut.get("t", {}).get("m", 0.0) == 0.0


def test_a_period_task_at_cap_zero_is_bounded_by_its_tail_not_cut():
    paid = WeightOnlyValidator._replay_ema(
        [], caps={"t": 0.0}, floors={"t": FLOORLESS}, periods={"t": {"m": 0.03}},
        period_caps={"t": 0.1})
    assert paid == {"m": pytest.approx(0.03)}
    bounded = WeightOnlyValidator._replay_ema(
        [], caps={"t": 0.0}, floors={"t": FLOORLESS}, periods={"t": {"m": 0.9}},
        period_caps={"t": 0.1})
    assert bounded == {"m": pytest.approx(0.1 * cp.CATCHUP_ENTRIES)}


def test_the_registry_and_the_period_module_name_the_same_settlement_and_bound():
    from reliquary.shared import task_registry

    assert task_registry.SETTLEMENT_PERIOD_EMA == cp.SETTLEMENT_PERIOD_EMA
    assert task_registry.TAIL_CAP_PARAM == cp.TAIL_CAP_PARAM
