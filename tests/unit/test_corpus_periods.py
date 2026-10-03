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
