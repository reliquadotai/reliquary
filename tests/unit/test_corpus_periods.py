"""Corpus tasks paid on their own clock: the period arithmetic and the replay."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp

GENESIS = 1_692_803_367.0


def test_periods_are_72_minutes_of_drand_time():
    assert cp.PERIOD_SECONDS == 4320
    assert cp.period_of(GENESIS, GENESIS) == 0
    assert cp.period_of(GENESIS + 4319.9, GENESIS) == 0
    assert cp.period_of(GENESIS + 4320, GENESIS) == 1
    assert cp.period_end(0, GENESIS) == GENESIS + 4320


def test_a_period_closes_once_nothing_received_in_it_is_pending():
    p5 = GENESIS + 5 * 4320
    # Nothing pending: everything ended `slack` ago is closed.
    assert cp.closed_through(now=p5 + 1000, oldest_pending=None, genesis=GENESIS,
                             slack=420) == 4
    assert cp.closed_through(now=p5 + 300, oldest_pending=None, genesis=GENESIS,
                             slack=420) == 3
    # A submission received in period 2 is undecided: period 2 stays open.
    assert cp.closed_through(now=p5 + 1000, oldest_pending=GENESIS + 2 * 4320 + 10,
                             genesis=GENESIS, slack=420) == 1


def test_only_tasks_declared_so_are_period_tasks():
    assert cp.is_period_task(SimpleNamespace(params={"settlement": "period-ema-v1"}))
    assert not cp.is_period_task(SimpleNamespace(params={"cap": 0.1}))
    assert not cp.is_period_task(SimpleNamespace(params=None))


def test_the_replay_decays_every_period_even_with_nothing_paid():
    archive = {"entry_period": 10, "rewards_by_hotkey": {"a": 0.1}}
    a = cp.PERIOD_ALPHA
    assert cp.replay([archive], 9) == {}
    assert cp.replay([archive], 10)["a"] == pytest.approx(a * 0.1)
    assert cp.replay([archive], 13)["a"] == pytest.approx(a * (1 - a) ** 3 * 0.1)
    assert cp.replay([archive], 10 + cp.REPLAY_DEPTH + 1) == {}


def _pay(archives, first, last):
    """What each hotkey receives, in cap-periods, if weights are set once a period."""
    paid: dict[str, float] = {}
    for period in range(first, last + 1):
        for hk, w in cp.replay(archives, period).items():
            paid[hk] = paid.get(hk, 0.0) + w
    return paid


def test_every_hotkey_is_paid_what_it_earned():
    """Conservation: random work with gaps, bursts and periods closed late."""
    rng = random.Random(7)
    cap = 0.1
    archives, earned = [], {}
    for work in range(60):
        if rng.random() < 0.3:
            continue  # nobody worked: that period's cap burns
        tokens = {hk: rng.randint(1, 1000) for hk in rng.sample("abcdefgh", rng.randint(1, 5))}
        total = sum(tokens.values())
        rewards = {hk: cap * t / total for hk, t in tokens.items()}
        entry = work + rng.choice([1, 1, 2, 5])  # closed one, two or five periods late
        archives.append({"entry_period": entry, "rewards_by_hotkey": rewards})
        for hk, r in rewards.items():
            earned[hk] = earned.get(hk, 0.0) + r
    paid = _pay(archives, 0, 200)
    lost = (1 - cp.PERIOD_ALPHA) ** (cp.REPLAY_DEPTH + 1)
    for hk, owed in earned.items():
        assert paid[hk] == pytest.approx(owed, rel=lost * 1.5)
        assert paid[hk] <= owed


def test_a_finished_task_stops_paying_on_its_own():
    cap = 0.04
    archives = [{"entry_period": p, "rewards_by_hotkey": {"m": cap}} for p in range(100)]
    assert sum(cp.replay(archives, 99).values()) == pytest.approx(cap, rel=1e-3)
    # 14 periods (17 h) after the last pay entered, under 1 % of the cap is left.
    assert sum(cp.replay(archives, 99 + 14).values()) < 0.01 * cap
    assert cp.replay(archives, 99 + cp.REPLAY_DEPTH + 1) == {}


# --------------------------------------------------------------------------
# the weight setter
# --------------------------------------------------------------------------


class _PeriodArchives:
    def __init__(self, docs):
        self.docs, self.reads = docs, []

    async def list(self, task_id):
        return sorted((w, e) for (t, w, e) in self.docs if t == task_id)

    async def read(self, task_id, work, entry):
        self.reads.append((task_id, work, entry))
        return self.docs.get((task_id, work, entry))


def _entry(cap, settlement="period-ema-v1", share=0.0):
    params = {"cap": cap, "min_incentive_share": share, "min_incentive_ramp_start": 0.0}
    if settlement:
        params["settlement"] = settlement
    return SimpleNamespace(params=params)


def test_the_weight_setter_replays_period_tasks_on_their_own_clock():
    import asyncio

    from reliquary.validator.weight_only import WeightOnlyValidator

    docs = {("eval-a", 30, 31): {"rewards_by_hotkey": {"m1": 0.02}},
            ("eval-a", 31, 32): {"rewards_by_hotkey": {"m1": 0.01, "m2": 0.01}},
            ("eval-a", 33, 33): {"rewards_by_hotkey": {"m3": 0.02}},  # not entered yet
            ("eval-a", 1, 2): {"rewards_by_hotkey": {"old": 0.02}}}  # beyond the depth
    archives = _PeriodArchives(docs)
    declared = {"eval-a": _entry(0.02), "default": _entry(0.8, settlement=None),
                "corpus-old": _entry(0.04, settlement=None)}
    now = GENESIS + 32 * cp.PERIOD_SECONDS + 5
    weights = asyncio.run(WeightOnlyValidator._period_weights(
        declared, archives=archives, now=now, genesis=GENESIS))
    a = cp.PERIOD_ALPHA
    assert set(weights) == {"eval-a"}
    assert weights["eval-a"]["m1"] == pytest.approx(a * (1 - a) * 0.02 + a * 0.01)
    assert weights["eval-a"]["m2"] == pytest.approx(a * 0.01)
    assert ("eval-a", 1, 2) not in archives.reads  # never read past the depth


def test_the_weight_setter_reads_the_clock_when_no_time_is_given(monkeypatch):
    # Production calls _period_weights(declared) with no `now`: the wall clock
    # path must work (a missing import made every weight setter abstain).
    import asyncio
    import time

    from reliquary.validator.weight_only import WeightOnlyValidator

    now = GENESIS + 32 * cp.PERIOD_SECONDS + 5
    monkeypatch.setattr(time, "time", lambda: now)
    docs = {("eval-a", 31, 32): {"rewards_by_hotkey": {"m1": 0.01}}}
    weights = asyncio.run(WeightOnlyValidator._period_weights(
        {"eval-a": _entry(0.02)}, archives=_PeriodArchives(docs), genesis=GENESIS))
    assert weights["eval-a"]["m1"] == pytest.approx(cp.PERIOD_ALPHA * 0.01)


def test_period_pay_is_capped_and_added_to_the_window_replay():
    from reliquary.validator.weight_only import WeightOnlyValidator

    window = [{"task_id": "default", "window_start": 5, "rewards_by_hotkey": {"r": 0.5}}]
    combined = WeightOnlyValidator._replay_ema(
        window, caps={"default": 0.8, "eval-a": 0.02},
        floors={"default": (0.0, 0.0), "eval-a": (0.0, 0.0)},
        periods={"eval-a": {"m1": 0.05, "m2": 0.05}})
    assert combined["m1"] == pytest.approx(0.01) and combined["m2"] == pytest.approx(0.01)
    assert combined["r"] == pytest.approx(
        WeightOnlyValidator._replay_ema(window, caps={"default": 0.8},
                                        floors={"default": (0.0, 0.0)})["r"])


def test_an_epoch_with_only_period_pay_still_submits(monkeypatch):
    """No window archive at all (every task period-settled) is not 'nothing to pay'."""
    import asyncio

    import reliquary.validator.weight_only as wov_mod
    from reliquary.validator.weight_only import WeightOnlyValidator

    async def no_windows(*, task_id=None, strict=False, **kw):
        return []

    async def tasks(*, strict=False, **kw):
        return ["default"]

    async def registry():
        return {"eval-a": _entry(0.02)}, "etag"

    async def periods(declared):
        return {"eval-a": {"m1": 0.01}}

    async def subtensor():
        return object()

    async def close(_):
        return None

    monkeypatch.setattr(wov_mod.storage, "list_task_ids", tasks)
    monkeypatch.setattr(wov_mod.storage, "list_all_window_keys", no_windows)
    monkeypatch.setattr(wov_mod, "read_registry", registry)
    monkeypatch.setattr(wov_mod.chain, "get_subtensor", subtensor)
    monkeypatch.setattr(wov_mod.chain, "close_subtensor", close)
    wov = WeightOnlyValidator.__new__(WeightOnlyValidator)
    wov._active_submit_epoch = None
    monkeypatch.setattr(wov, "_period_weights", periods, raising=False)
    submitted = {}

    async def submit(sub, weights):
        submitted.update(weights)
        return True

    wov._submit_weights = submit
    assert asyncio.run(wov.submit_once()) is True
    assert submitted == {"m1": pytest.approx(0.01)}


def test_unreadable_period_pay_abstains(monkeypatch):
    import asyncio

    import reliquary.validator.weight_only as wov_mod
    from reliquary.validator.weight_only import WeightOnlyValidator

    async def tasks(*, strict=False, **kw):
        return ["default"]

    async def windows(*, task_id=None, strict=False, **kw):
        return []

    async def registry():
        return {"eval-a": _entry(0.02)}, "etag"

    async def broken(declared):
        raise ConnectionError("R2 down")

    monkeypatch.setattr(wov_mod.storage, "list_task_ids", tasks)
    monkeypatch.setattr(wov_mod.storage, "list_all_window_keys", windows)
    monkeypatch.setattr(wov_mod, "read_registry", registry)
    wov = WeightOnlyValidator.__new__(WeightOnlyValidator)
    monkeypatch.setattr(wov, "_period_weights", broken, raising=False)
    assert asyncio.run(wov.submit_once()) is False


def test_window_and_period_pay_of_one_task_add_up():
    from reliquary.validator.weight_only import WeightOnlyValidator

    window = [{"task_id": "corpus-x", "window_start": 5, "rewards_by_hotkey": {"a": 0.04}}]
    alone = WeightOnlyValidator._replay_ema(window, caps={"corpus-x": 0.04},
                                            floors={"corpus-x": (0.0, 0.0)})["a"]
    both = WeightOnlyValidator._replay_ema(window, caps={"corpus-x": 0.04},
                                           floors={"corpus-x": (0.0, 0.0)},
                                           periods={"corpus-x": {"a": 0.001, "b": 0.002}})
    assert both["a"] == pytest.approx(alone + 0.001) and both["b"] == pytest.approx(0.002)
