"""The async price controller steering a short fleet towards the target mix (rl_price, 2026-10-23).

Deterministic simulation from the live state of teutonic-rl-async-v1 (2026-10-22/23): every multiplier stuck at the
ceiling, the trainer always waiting, the paid mix far from the split. Miners as a whole put their tokens on env e in
proportion to ``preference(e) x multiplier(e) ** elasticity`` (preference = the live mix at equal prices) and move a
fraction ``inertia`` of their GPUs towards that each tick. Run directly for the trajectory table.
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading

import pytest

from reliquary.validator.rl_price import DESCEND, AsyncPriceController, QueueTick

TARGETS = {"swe": 0.23, "competitive-code": 0.21, "hard-math": 0.21, "tmax": 0.18, "dapo": 0.13,
           "envscaler": 0.03, "ifeval": 0.01}
# Pay share of each env since the run opened, every multiplier at 4.0 (EnvScaler not seen: 0.2 % assumed).
LIVE_MIX = {"swe": 0.011, "competitive-code": 0.065, "hard-math": 0.665, "tmax": 0.042, "dapo": 0.193,
            "envscaler": 0.002, "ifeval": 0.025}
TICK = 300.0
TRAINER_TPS = 10_000.0
SUPPLY_TPS = 7_000.0        # the fleet makes 70 % of what the trainer can take: it waits 30 % of every tick


class MemBook:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.lock = threading.RLock()

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")


def stuck_controller(clock, start=4.0):
    """A controller restarted on the live run's state: every multiplier at ``start``, regime rise."""
    prices = AsyncPriceController(MemBook(), TARGETS, cap=0.8, planned_tokens_per_period=1e9, start_pool=0.5,
                                  clock=lambda: clock[0])
    with prices.book.transaction():
        for env in TARGETS:
            prices.book.db.execute("INSERT INTO rl_prices VALUES(?,?,?,?,?,?)",
                                   (env, start, 1.0, "rise", clock[0], "{}"))
    return prices


def simulate(hours=36.0, elasticity=1.0, inertia=0.1, lump=1):
    """``lump``: SWE and TMax groups land every ``lump`` ticks, all at once (long episodes)."""
    clock = [1_000_000.0]
    prices = stuck_controller(clock)
    mix = dict(LIVE_MIX)
    rows = []
    for n in range(int(hours * 3600 / TICK)):
        clock[0] += TICK
        arrivals = {env: int(share * SUPPLY_TPS * TICK) for env, share in mix.items()}
        for env in ("swe", "tmax"):
            arrivals[env] = arrivals[env] * lump if n % lump == 0 else 0
        prices.tick(QueueTick(arrivals, sum(arrivals.values()), TICK * (1 - SUPPLY_TPS / TRAINER_TPS)))
        mult = {env: prices.multiplier(env) for env in TARGETS}
        want = {env: LIVE_MIX[env] * mult[env] ** elasticity for env in TARGETS}
        total = sum(want.values())
        mix = {env: (1 - inertia) * mix[env] + inertia * want[env] / total for env in TARGETS}
        rows.append((clock[0], dict(mult), dict(mix)))
    return rows


def gap(mix):
    return max(abs(mix[env] - TARGETS[env]) for env in TARGETS)


def converged_at(rows, within=0.03):
    """Hours until the mix enters ``within`` of every target and never leaves it again (None: never)."""
    start = rows[0][0] - TICK
    last_out = None
    for i, (_, _, mix) in enumerate(rows):
        if gap(mix) > within:
            last_out = i
    if last_out == len(rows) - 1:
        return None
    return (rows[last_out + 1 if last_out is not None else 0][0] - start) / 3600


def test_restart_on_the_stuck_board_only_lowers_the_over_supplied_envs():
    clock = [1_000_000.0]
    prices = stuck_controller(clock)
    clock[0] += TICK
    arrivals = {env: int(share * SUPPLY_TPS * TICK) for env, share in LIVE_MIX.items()}
    decisions = prices.tick(QueueTick(arrivals, sum(arrivals.values()), 0.3 * TICK))
    assert max(d.price for d in decisions.values()) == pytest.approx(4.0)          # no jump above the old board
    for env in ("hard-math", "dapo", "ifeval"):                                     # over their target share
        assert decisions[env].regime == DESCEND and decisions[env].price < 4.0 / 1.05
    for env in ("swe", "tmax", "competitive-code", "envscaler"):                    # short: held at the ceiling
        assert decisions[env].price == pytest.approx(4.0)


SCENARIOS = {   # (elasticity, inertia per tick, lump): miners moving 3-30 % of their GPUs per tick
    "base": (1.0, 0.1, 1), "elastic-lumpy": (2.0, 0.1, 3), "slow-lumpy": (1.0, 0.03, 6), "fast": (1.0, 0.3, 1),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_a_short_fleet_converges_to_the_target_mix_without_oscillating(scenario):
    elasticity, inertia, lump = SCENARIOS[scenario]
    rows = simulate(36.0, elasticity, inertia, lump)
    hours = converged_at(rows, within=0.035)
    assert hours is not None and hours < 24.0
    assert max(gap(mix) for _, _, mix in rows[-144:]) < 0.035                      # the last 12 h stay there
    _, mult, _ = rows[-1]
    assert mult["swe"] / mult["hard-math"] > 5                                      # the scarce env pays more


if __name__ == "__main__":
    for name, (elasticity, inertia, lump) in sorted(SCENARIOS.items()):
        rows = simulate(36.0, elasticity, inertia, lump)
        print(f"{name}: within 3 pts after {converged_at(rows)} h, within 3.5 after {converged_at(rows, 0.035)} h, "
              f"last 12 h max gap {max(gap(m) for _, _, m in rows[-144:]):.3f}")
        for t, mult, mix in rows[::24]:
            h = (t - rows[0][0] + TICK) / 3600
            print(f"{h:5.1f} h gap {gap(mix):.3f} " + " ".join(
                f"{env[:6]} {100 * mix[env]:4.1f}%@{mult[env]:.2f}" for env in TARGETS))
