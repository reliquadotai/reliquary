"""The async per-env price controller (rl-async-design.md §4.9; implementation map §1.12, P5).

The target is the trainer's speed: each env must supply its target share of what the trainer consumes. Per tick
(``tick_s`` of wall time, the validator's timer):

* **demand** of an env = its target share (renormalised over the envs that supplied anything in the tick: a dry
  env's share goes to the others, since the trainer takes what arrives) x the trainer's speed, in completion tokens per second of trainer work
  (``note_speed``: per step, the completion tokens the trainer processed and the seconds it worked, waits excluded;
  kept as sums decayed with a half-life of one tick, persisted);
* **supply** = the completion tokens of the env's groups queued for training during the tick
  (``TrainingBuffer.drain_arrivals``), per second;
* ``r = supply / demand``: under ``low`` the price rises (x ``rise``), over ``high`` it descends (``decay`` scaled to
  the tick's length), else it holds. An env refused leases for the global cap during the tick
  (``LeaseBook.drain_full``) descends unless it is under ``low`` (then it rises: the cap holds every env back, the
  mix still has to move), and only when the total supply covered the trainer's speed (else the cap is too small).
* No speed yet, or none for ``stale_ticks`` ticks (trainer down): every price holds.

No global price: the trainer's speed is the sum of the demands, so every env at r ~ 1 means the miners as a whole
match the trainer. The price is a multiplier of the env's nominal rate: per matched token an env pays ``multiplier x
cap x target_share / planned_tokens_per_period``. Prices above 1 (up to ``cap`` = 4 x nominal) are bounded by the
settlement's normalisation (``rl_ledger``): a period never pays more than the cap, the rest burns. State lives in the
lease book's SQLite file (``rl_prices``, ``rl_trainer_speed``) and survives a restart.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from reliquary.validator.emission_price import (
    PRODUCTION_PRICE_PARAMS, ROUND_SECONDS, PriceDecision, PriceParams, PriceState,
)

# Simulated 2026-10-16 (scratchpad price_sim.py: 3 envs, miners reallocating on last tick's prices, 72 h): the
# window's walk (x1.10 up, -0.2 % per 300 s tick down, band 0.95-1.15) ratchets up and pins the costliest env at
# the cap; x1.05 up, -2 % per tick down (decay 0.817 per 1000 rounds) and a band centred on 1 converge in 7-9 h
# with every env within its target share and 1-3 % trainer wait. Placeholders until the end-to-end run.
ASYNC_PRICE_RISE = 1.05
ASYNC_PRICE_DECAY = 0.817
ASYNC_BASE_PARAMS = replace(PRODUCTION_PRICE_PARAMS, cap=4.0, rise=ASYNC_PRICE_RISE, decay=ASYNC_PRICE_DECAY)
RISE, DESCEND, HOLD = "rise", "descend", "hold"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_prices(environment TEXT PRIMARY KEY, price REAL NOT NULL, last_good REAL NOT NULL,
    regime TEXT NOT NULL, at REAL NOT NULL, signal TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_price_clock(id INTEGER PRIMARY KEY CHECK(id=0), last_tick REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rl_trainer_speed(id INTEGER PRIMARY KEY CHECK(id=0), tokens REAL NOT NULL,
    seconds REAL NOT NULL, at REAL NOT NULL);
"""


@dataclass(frozen=True)
class AsyncPriceParams:
    base: PriceParams = ASYNC_BASE_PARAMS
    tick_s: float = 300.0
    low: float = 0.90            # placeholders: the supply/demand band the price holds in
    high: float = 1.10
    stale_ticks: float = 3.0     # no trainer speed for this many ticks: hold every price

    def __post_init__(self) -> None:
        if not (self.tick_s > 0 and 0 < self.low <= self.high and self.stale_ticks > 0):
            raise ValueError("tick_s > 0, 0 < low <= high, stale_ticks > 0")
        if not 0 < self.base.floor <= self.base.start <= self.base.cap:
            raise ValueError("floor <= start <= cap")


@dataclass(frozen=True)
class EnvSignal:
    """One env over one tick: ``supply_tps`` its queued completion tokens per second, ``demand_tps`` its target share
    of the trainer's speed (None: no usable speed), ``full`` refused leases for the global cap."""
    seconds: float
    supply_tps: float
    demand_tps: float | None
    full: bool = False

    @property
    def ratio(self) -> float | None:
        return self.supply_tps / self.demand_tps if self.demand_tps else None


def advance_async(state: PriceState, signal: EnvSignal, params: AsyncPriceParams) -> PriceDecision:
    """One env's price after one tick (pure). ``r`` carries the tick's supply/demand ratio (None: hold)."""
    base = params.base
    price = state.price
    r = signal.ratio
    if r is None:
        regime = HOLD
    elif r < params.low:
        price *= base.rise
        regime = RISE
    elif r > params.high or signal.full:
        price *= base.decay ** (signal.seconds / (base.rounds_per_step * ROUND_SECONDS))
        regime = DESCEND
    else:
        regime = HOLD
    price = min(max(price, base.floor), base.cap)
    return PriceDecision(price=price, last_good=state.last_good, recent_fill_prices=(), r=r, r_smoothed=None,
                         regime=regime)


@dataclass
class AsyncPriceController:
    """Per-env multipliers, persisted in ``book``'s file. ``targets``: env -> target share of trained tokens (the
    trainer's mix, summing to 1); ``cap`` and ``planned_tokens_per_period`` set the nominal rate per token."""
    book: object
    targets: Mapping[str, float]
    cap: float
    planned_tokens_per_period: float
    params: AsyncPriceParams = field(default_factory=AsyncPriceParams)
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not self.targets or any(v < 0 for v in self.targets.values()) or abs(sum(self.targets.values()) - 1) > 1e-6:
            raise ValueError("targets are non-negative shares summing to 1")
        if not (self.cap > 0 and self.planned_tokens_per_period > 0):
            raise ValueError("cap and planned_tokens_per_period must be positive")
        with self.book.lock:
            self.book.db.executescript(_SCHEMA)
            self.book.db.execute("INSERT OR IGNORE INTO rl_price_clock(id, last_tick) VALUES(0, ?)",
                                 (float(self.clock()),))
            self.book.db.commit()

    # --- reads ---------------------------------------------------------------------------------------------
    def multiplier(self, environment: str) -> float:
        with self.book.lock:
            row = self.book.db.execute("SELECT price FROM rl_prices WHERE environment=?", (environment,)).fetchone()
        return self.params.base.start if row is None else float(row[0])

    def token_price(self, environment: str) -> float:
        """What one matched token of ``environment`` pays now, in cap units (0 for an env without a target)."""
        nominal = self.cap * float(self.targets.get(environment, 0.0)) / self.planned_tokens_per_period
        return self.multiplier(environment) * nominal

    def board(self) -> dict[str, dict]:
        with self.book.lock:
            rows = {env: (price, regime, at, signal) for env, price, regime, at, signal in self.book.db.execute(
                "SELECT environment, price, regime, at, signal FROM rl_prices").fetchall()}
        out = {}
        for env in sorted(self.targets):
            price, regime, at, signal = rows.get(env, (self.params.base.start, HOLD, None, "{}"))
            out[env] = {"multiplier": price, "token_price": self.token_price(env),
                        "per_million_tokens": 1e6 * self.token_price(env), "target_share": self.targets[env],
                        "regime": regime, "decided_at": at, "signal": json.loads(signal)}
        return out

    # --- signals and ticks ---------------------------------------------------------------------------------
    def note_speed(self, tokens: float, seconds: float, now: float | None = None) -> None:
        """One trainer step: ``tokens`` completion tokens processed in ``seconds`` of work (its waits excluded)."""
        if not (tokens >= 0 and seconds > 0):
            return
        now = float(self.clock()) if now is None else now
        with self.book.transaction():
            row = self.book.db.execute("SELECT tokens, seconds, at FROM rl_trainer_speed WHERE id=0").fetchone()
            keep = 0.0 if row is None else 0.5 ** (max(now - row[2], 0.0) / self.params.tick_s)
            total_tokens = float(tokens) + (0.0 if row is None else keep * row[0])
            total_seconds = float(seconds) + (0.0 if row is None else keep * row[1])
            self.book.db.execute("INSERT INTO rl_trainer_speed(id, tokens, seconds, at) VALUES(0,?,?,?) ON CONFLICT(id) "
                                 "DO UPDATE SET tokens=excluded.tokens, seconds=excluded.seconds, at=excluded.at",
                                 (total_tokens, total_seconds, now))

    def trainer_speed(self, now: float | None = None) -> float | None:
        """Completion tokens per second of trainer work; None before any step or after ``stale_ticks`` without."""
        now = float(self.clock()) if now is None else now
        with self.book.lock:
            row = self.book.db.execute("SELECT tokens, seconds, at FROM rl_trainer_speed WHERE id=0").fetchone()
        if row is None or row[1] <= 0 or now - row[2] > self.params.stale_ticks * self.params.tick_s:
            return None
        return row[0] / row[1]

    def due(self, now: float | None = None) -> bool:
        now = float(self.clock()) if now is None else now
        with self.book.lock:
            last = self.book.db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
        return now - last >= self.params.tick_s

    def tick(self, arrivals: Mapping[str, int], full: Iterable[str] = (),
             now: float | None = None) -> dict[str, PriceDecision]:
        """Advance every target env over the time since the last tick. ``arrivals``: env -> completion tokens queued
        since the last tick (``TrainingBuffer.drain_arrivals``); ``full``: envs refused for the global cap
        (``LeaseBook.drain_full``)."""
        now = float(self.clock()) if now is None else now
        speed = self.trainer_speed(now)
        full = set(full)
        decisions: dict[str, PriceDecision] = {}
        with self.book.transaction():
            db = self.book.db
            last = db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
            seconds = max(now - last, 0.0)
            if seconds <= 0:
                return {}
            # An env nobody supplied gives its share to the others (the trainer takes what arrives), and still rises.
            live = sum(float(v) for env, v in self.targets.items() if arrivals.get(env, 0) > 0) or 1.0
            # The cap refusing leases while the miners as a whole supply less than the trainer takes says the cap is
            # too small, not that the miners are too many: ``full`` counts only when the supply covers the speed.
            oversupplied = speed is not None and sum(int(v) for v in arrivals.values()) / seconds >= speed
            for env in sorted(self.targets):
                row = db.execute("SELECT price, last_good FROM rl_prices WHERE environment=?", (env,)).fetchone()
                start = self.params.base.start
                state = PriceState(price=start, last_good=start) if row is None else PriceState(*row)
                share = float(self.targets[env])
                if arrivals.get(env, 0) > 0:
                    share /= live
                signal = EnvSignal(seconds, int(arrivals.get(env, 0)) / seconds,
                                   None if speed is None else share * speed, oversupplied and env in full)
                decision = advance_async(state, signal, self.params)
                decisions[env] = decision
                detail = {"seconds": seconds, "ratio": decision.r, "supply_tps": signal.supply_tps,
                          "demand_tps": signal.demand_tps, "trainer_tps": speed, "target_share": share,
                          "full": signal.full}
                db.execute("INSERT INTO rl_prices(environment, price, last_good, regime, at, signal) "
                           "VALUES(?,?,?,?,?,?) ON CONFLICT(environment) DO UPDATE SET price=excluded.price, "
                           "last_good=excluded.last_good, regime=excluded.regime, at=excluded.at, "
                           "signal=excluded.signal",
                           (env, decision.price, decision.last_good, decision.regime, now,
                            json.dumps(detail, sort_keys=True)))
            db.execute("UPDATE rl_price_clock SET last_tick=? WHERE id=0", (now,))
        return decisions


__all__ = [
    "ASYNC_BASE_PARAMS", "ASYNC_PRICE_RISE", "DESCEND", "HOLD", "RISE", "AsyncPriceController", "AsyncPriceParams", "EnvSignal",
    "advance_async",
]
