"""Level mode of the async price controller: one level L normalising the recent supply's pay to the budget, per-env
weights for the mix only (rl_price module doc, teutonic-rl-async-v1 2026-10-24)."""
from __future__ import annotations

import pytest

from reliquary.validator.rl_price import AsyncPriceController, AsyncPriceParams, QueueTick
from tests.unit.test_rl_price_steer import MemBook

TARGETS = {"a": 0.5, "b": 0.3, "idle": 0.2}
TICK = 300.0
SPEED = 10_000.0                     # trainer tokens per second
CAP, PLANNED, START_POOL = 0.8, 1e7, 0.5
NOMINAL = START_POOL / PLANNED
ON_TARGET = {"a": 2500, "b": 1500, "idle": 1000}        # 5000 tok/s on the target mix
# The first live speeds (x1.03 / x0.98 a 5-min tick, no band, 1 h / 10 min half-lives): the tests below that predate
# the slow defaults keep them, so their tick counts stay readable.
FAST = dict(level_rise_h=1.03 ** 12, level_decay_h=0.98 ** 12, level_band=0.0, oversupply_rise_h=1.02 ** 12,
            oversupply_decay_h=0.98 ** 12, weight_rise_h=1.02 ** 12, weight_decay_h=0.98 ** 12, weight_low=0.9,
            weight_high=1.1, supply_halflife_s=3600.0, mix_halflife_s=600.0)


def controller(clock, prices=None, slow=False, **params):
    c = AsyncPriceController(MemBook(), TARGETS, cap=CAP, planned_tokens_per_period=PLANNED, start_pool=START_POOL,
                             params=AsyncPriceParams(mode="level", **({} if slow else FAST) | params),
                             clock=lambda: clock[0])
    with c.book.transaction():
        for env, price in (prices or {}).items():
            c.book.db.execute("INSERT INTO rl_prices VALUES(?,?,?,?,?,?)", (env, price, price, "hold", clock[0], "{}"))
        c.book.db.execute("INSERT INTO rl_trainer_work VALUES(0, ?, ?, 0, 0, ?)", (SPEED * TICK, TICK, clock[0]))
    return c


def run(c, clock, ticks, tps, *, starved=0.0, full=(), lane_full=()):
    """``tps``: env -> in-zone tokens per second arriving each tick."""
    for _ in range(ticks):
        clock[0] += TICK
        c.tick(QueueTick({e: int(v * TICK) for e, v in tps.items()}, taken=int(SPEED * TICK),   # a working trainer
                         starved_s=starved * TICK), full=full, lane_full=lane_full)


def pay_per_period(c, tps):
    """What ``tps`` earns a period at the board's prices."""
    return sum(v * c.multiplier(e) * NOMINAL * 4320 for e, v in tps.items())


def test_the_switch_starts_from_the_live_board_and_moves_by_one_step_at_most():
    clock = [0.0]
    live = {"a": 0.2, "b": 1.5, "idle": 4.0}
    c = controller(clock, live)
    run(c, clock, 1, ON_TARGET)
    assert c.level() / (0.5 * 0.2 + 0.3 * 1.5 + 0.2 * 4.0) in (pytest.approx(1.03), pytest.approx(0.98))
    for env, price in live.items():
        assert 0.98 * 0.98 <= c.multiplier(env) / price <= 1.03 * 1.02


def test_the_supply_is_priced_to_earn_the_pay_target_of_the_cap():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 60, ON_TARGET)
    assert pay_per_period(c, ON_TARGET) == pytest.approx(0.95 * CAP, rel=0.02)


def test_half_the_supply_earns_the_same_budget_at_twice_the_price_per_token():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 60, ON_TARGET)
    before = c.level()
    half = {e: v / 2 for e, v in ON_TARGET.items()}
    run(c, clock, 120, half, starved=0.5)
    assert c.level() == pytest.approx(2 * before, rel=0.03)
    assert pay_per_period(c, half) == pytest.approx(0.95 * CAP, rel=0.03)


def test_one_tick_moves_the_level_by_its_step_limits_at_most():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})       # aim ~0.67: a descent
    run(c, clock, 1, ON_TARGET)
    assert c.level() == pytest.approx(0.98)
    c2 = controller([0.0], {"a": 0.01, "b": 0.01, "idle": 0.01})   # aim ~0.67: a rise
    clock2 = [0.0]
    c2.clock = lambda: clock2[0]
    run(c2, clock2, 1, ON_TARGET)
    assert c2.level() == pytest.approx(0.01 * 1.03)


def test_a_full_queue_lowers_the_budget_and_it_comes_back_after():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 60, ON_TARGET)
    run(c, clock, 36, ON_TARGET, full=("a",))                     # 3 h of leases refused for a full queue
    assert pay_per_period(c, ON_TARGET) < 0.6 * CAP
    run(c, clock, 200, ON_TARGET)
    assert pay_per_period(c, ON_TARGET) == pytest.approx(0.95 * CAP, rel=0.02)


def test_weights_move_the_mix_only_and_keep_their_target_mean():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    tps = {"a": 4000, "b": 500, "idle": 500}                      # a over target, b under
    run(c, clock, 24, tps)
    w = {e: c.multiplier(e) / c.level() for e in TARGETS}
    assert w["a"] < 1 < w["b"]
    assert sum(TARGETS[e] * w[e] for e in TARGETS) == pytest.approx(1.0)


def test_an_env_nobody_supplies_is_frozen_and_no_longer_weighs_on_the_others():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 4.0})
    run(c, clock, 12 * 7, {"a": 5000, "b": 3000})                 # 7 h: idle is out after 6 h
    frozen = c.multiplier("idle") / c.level()
    run(c, clock, 24, {"a": 5000, "b": 3000})
    assert c.multiplier("idle") / c.level() == pytest.approx(frozen)
    w = {e: c.multiplier(e) / c.level() for e in ("a", "b")}
    assert 0.5 * w["a"] + 0.3 * w["b"] == pytest.approx(0.8)      # rescaled over the supplied envs only
    board = c.board()
    assert board["idle"]["signal"]["active"] is False and board["a"]["signal"]["active"] is True


def test_an_env_at_its_own_maximum_descends_alone_and_the_budget_holds():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 60, ON_TARGET)
    run(c, clock, 12, ON_TARGET, lane_full=("b",))
    w = {e: c.multiplier(e) / c.level() for e in TARGETS}
    assert w["b"] < 1 < w["a"]
    assert pay_per_period(c, ON_TARGET) == pytest.approx(0.95 * CAP, rel=0.03)  # a lane refusal is not a full queue


def test_per_env_mode_is_still_the_default():
    assert AsyncPriceParams().mode == "per_env"
    with pytest.raises(ValueError):
        AsyncPriceParams(mode="other")


def test_a_trainer_down_holds_the_level():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0}, stale_s=600.0)
    for _ in range(5):
        clock[0] += TICK
        c.tick(QueueTick({e: int(v * TICK) for e, v in ON_TARGET.items()}))   # no take, no wait: the trainer is gone
    assert c.level() == pytest.approx(0.98 ** 2)                                # two ticks before it went stale


def test_most_envs_at_their_maximum_with_a_busy_trainer_lower_the_budget():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 60, ON_TARGET)
    run(c, clock, 36, ON_TARGET, lane_full=("a",))                 # a holds 50 % of the targets
    assert pay_per_period(c, ON_TARGET) < 0.6 * CAP
    clock2 = [0.0]
    c2 = controller(clock2, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c2, clock2, 60, ON_TARGET)
    run(c2, clock2, 36, ON_TARGET, lane_full=("a",), starved=0.5)   # the trainer waits: not a flood
    assert pay_per_period(c2, ON_TARGET) > 0.8 * CAP


def test_an_env_frozen_moves_no_price_at_once():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0})
    run(c, clock, 12 * 6 - 1, {"a": 2500, "b": 1500})              # idle about to drop out
    before = {e: c.multiplier(e) for e in TARGETS}
    run(c, clock, 2, {"a": 2500, "b": 1500})
    for env in ("a", "b"):
        assert 0.98 ** 2 * 0.96 <= c.multiplier(env) / before[env] <= 1.03 ** 2 * 1.05


# --- the slow defaults (2026-10-24: the first speeds cut the burn from ~60 % to ~10 % in one period) -------------

def test_speeds_are_per_hour_whatever_the_tick():
    for tick in (60.0, 300.0, 900.0):
        clock = [0.0]
        c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0}, slow=True, tick_s=tick)    # aim ~0.67: a descent
        for _ in range(int(3600 / tick)):
            clock[0] += tick
            c.tick(QueueTick({e: int(v * tick) for e, v in ON_TARGET.items()}, taken=int(SPEED * tick)))
        assert c.level() == pytest.approx(0.90)


def test_the_slow_level_takes_hours_to_close_a_gap_of_two():
    clock = [0.0]
    c = controller(clock, {"a": 0.5, "b": 0.5, "idle": 0.5}, slow=True)      # aim ~0.67 x 2 = ~1.35
    run(c, clock, 12, {e: v / 2 for e, v in ON_TARGET.items()}, starved=0.5)
    assert c.level() <= 0.5 * 1.10 + 1e-9                                      # at most +10 % an hour
    run(c, clock, 12 * 11, {e: v / 2 for e, v in ON_TARGET.items()}, starved=0.5)
    assert c.level() > 0.5 * 2 * 0.9


def test_the_level_holds_inside_its_band():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0}, slow=True)
    run(c, clock, 12 * 24, ON_TARGET)                  # came down from above: settled at the band's upper edge
    settled = c.level()
    run(c, clock, 12 * 6, {e: v * 0.97 for e, v in ON_TARGET.items()})        # -3 % supply: the aim stays inside
    assert c.level() == pytest.approx(settled)
    assert 0.95 / 1.05 * CAP * 0.99 <= pay_per_period(c, ON_TARGET) <= 0.95 * 1.05 * CAP * 1.01


def test_one_new_miner_moves_no_price_within_the_hour():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0}, slow=True)
    run(c, clock, 12 * 24, ON_TARGET)
    before = {e: c.multiplier(e) for e in TARGETS}
    run(c, clock, 12, {**ON_TARGET, "a": ON_TARGET["a"] * 1.2})                # a's supply +20 % for an hour
    for env in TARGETS:
        assert c.multiplier(env) == pytest.approx(before[env], rel=0.08)


def test_a_flood_lowers_the_budget_slowly():
    clock = [0.0]
    c = controller(clock, {"a": 1.0, "b": 1.0, "idle": 1.0}, slow=True)
    run(c, clock, 12 * 24, ON_TARGET)
    before = pay_per_period(c, ON_TARGET)
    run(c, clock, 12, ON_TARGET, full=("a",))                                  # an hour of a full queue
    assert pay_per_period(c, ON_TARGET) == pytest.approx(before * 0.95, rel=0.01)
