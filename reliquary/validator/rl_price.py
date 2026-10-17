"""The async per-env price controller (rl-async-design.md §4.9; implementation map §1.12, P5).

The training queue is the signal: it is what the miners deliver minus what the trainer consumes, summed over time,
so it already holds the trainer's speed, and its depth is the lag the queue adds to every group. Per tick
(``tick_s`` of wall time, the validator's timer), from the buffer (``TrainingBuffer.drain_tick``):

* **level** (one for every env): the trainer waited for groups more than ``starved`` of the tick -> too little
  supply, prices rise; else the groups left queued after its batches averaged more than ``backlog`` -> too much,
  prices descend (what is not paid burns: the emissions go down); else hold. No completed batch and no wait in the
  tick (trainer down, or a step longer than a tick): every price holds.
* **mix** (per env): its share of the tick's arrived completion tokens over its target share (renormalised over the
  envs that supplied anything: the trainer takes what arrives). Under ``low`` the env is short, over ``high`` long.
  A rise moves only the envs not long, a descent only the envs not short; on a hold the short envs rise and the
  long ones descend.

The price is a multiplier of the env's nominal rate: per matched token an env pays ``multiplier x cap x
target_share / planned_tokens_per_period``. Prices above 1 (up to ``cap`` = 4 x nominal) are bounded by the
settlement's normalisation (``rl_ledger``): a period never pays more than the cap, the rest burns. State lives in the
lease book's SQLite file (``rl_prices``) and survives a restart.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
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
RISE, DESCEND, HOLD, IDLE = "rise", "descend", "hold", "idle"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_prices(environment TEXT PRIMARY KEY, price REAL NOT NULL, last_good REAL NOT NULL,
    regime TEXT NOT NULL, at REAL NOT NULL, signal TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_price_clock(id INTEGER PRIMARY KEY CHECK(id=0), last_tick REAL NOT NULL);
"""


@dataclass(frozen=True)
class AsyncPriceParams:
    base: PriceParams = ASYNC_BASE_PARAMS
    tick_s: float = 300.0
    low: float = 0.90            # placeholders: the mix band (an env's share of arrivals over its target share)
    high: float = 1.10
    backlog: float = 32.0        # placeholder: groups left queued after a batch (tick mean) over which prices descend
    starved: float = 0.05        # placeholder: share of the tick the trainer waited over which prices rise

    def __post_init__(self) -> None:
        if not (self.tick_s > 0 and 0 < self.low <= self.high and self.backlog >= 0 and 0 <= self.starved < 1):
            raise ValueError("tick_s > 0, 0 < low <= high, backlog >= 0, 0 <= starved < 1")
        if not 0 < self.base.floor <= self.base.start <= self.base.cap:
            raise ValueError("floor <= start <= cap")


@dataclass(frozen=True)
class QueueTick:
    """The buffer over one tick (``TrainingBuffer.drain_tick``): completion tokens queued per env, the trainer's
    completed batches, the groups left queued after each (summed), the seconds it waited for groups."""
    arrivals: Mapping[str, int] = field(default_factory=dict)
    batches: int = 0
    backlog: int = 0
    starved_s: float = 0.0


def queue_level(tick: QueueTick, seconds: float, params: AsyncPriceParams) -> str:
    """Every env's direction from the queue (pure): RISE, DESCEND, HOLD, or IDLE (the trainer neither finished a
    batch nor waited: every price holds, the mix too)."""
    if not tick.batches and not tick.starved_s:
        return IDLE
    if tick.starved_s > params.starved * seconds:
        return RISE
    if tick.batches and tick.backlog / tick.batches > params.backlog:
        return DESCEND
    return HOLD


def advance_async(state: PriceState, level: str, mix: float | None, seconds: float,
                  params: AsyncPriceParams) -> PriceDecision:
    """One env's price after one tick (pure). ``mix``: its share of arrivals over its target share (None: no
    arrivals at all, the mix holds)."""
    base = params.base
    short = level != IDLE and mix is not None and mix < params.low
    long = level != IDLE and mix is not None and mix > params.high
    if (level == RISE and not long) or (level == HOLD and short):
        regime, price = RISE, state.price * base.rise
    elif (level == DESCEND and not short) or (level == HOLD and long):
        regime, price = DESCEND, state.price * base.decay ** (seconds / (base.rounds_per_step * ROUND_SECONDS))
    else:
        regime, price = HOLD, state.price
    price = min(max(price, base.floor), base.cap)
    return PriceDecision(price=price, last_good=state.last_good, recent_fill_prices=(), r=mix, r_smoothed=None,
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

    # --- ticks ---------------------------------------------------------------------------------------------
    def due(self, now: float | None = None) -> bool:
        now = float(self.clock()) if now is None else now
        with self.book.lock:
            last = self.book.db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
        return now - last >= self.params.tick_s

    def tick(self, queue: QueueTick, now: float | None = None) -> dict[str, PriceDecision]:
        """Advance every target env over the time since the last tick (``queue``: ``TrainingBuffer.drain_tick``)."""
        now = float(self.clock()) if now is None else now
        decisions: dict[str, PriceDecision] = {}
        with self.book.transaction():
            db = self.book.db
            last = db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
            seconds = max(now - last, 0.0)
            if seconds <= 0:
                return {}
            level = queue_level(queue, seconds, self.params)
            arrived = {env: int(queue.arrivals.get(env, 0)) for env in self.targets}
            total = sum(arrived.values())
            # An env nobody supplied gives its share to the others (the trainer takes what arrives), and is short.
            live = sum(float(v) for env, v in self.targets.items() if arrived[env] > 0) or 1.0
            for env in sorted(self.targets):
                row = db.execute("SELECT price, last_good FROM rl_prices WHERE environment=?", (env,)).fetchone()
                start = self.params.base.start
                state = PriceState(price=start, last_good=start) if row is None else PriceState(*row)
                share = float(self.targets[env]) / (live if arrived[env] > 0 else 1.0)
                mix = None if not total or not share else arrived[env] / total / share
                decision = advance_async(state, level, mix, seconds, self.params)
                decisions[env] = decision
                detail = {"seconds": seconds, "level": level, "mix": mix, "target_share": share,
                          "supply_tps": arrived[env] / seconds, "batches": queue.batches,
                          "backlog": queue.backlog / queue.batches if queue.batches else None,
                          "starved_s": queue.starved_s}
                db.execute("INSERT INTO rl_prices(environment, price, last_good, regime, at, signal) "
                           "VALUES(?,?,?,?,?,?) ON CONFLICT(environment) DO UPDATE SET price=excluded.price, "
                           "last_good=excluded.last_good, regime=excluded.regime, at=excluded.at, "
                           "signal=excluded.signal",
                           (env, decision.price, decision.last_good, decision.regime, now,
                            json.dumps(detail, sort_keys=True)))
            db.execute("UPDATE rl_price_clock SET last_tick=? WHERE id=0", (now,))
        return decisions


__all__ = [
    "ASYNC_BASE_PARAMS", "ASYNC_PRICE_RISE", "DESCEND", "HOLD", "IDLE", "RISE", "AsyncPriceController", "AsyncPriceParams",
    "QueueTick", "advance_async", "queue_level",
]
