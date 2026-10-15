"""The async per-env price controller (rl-async-design.md §4.9; implementation map §1.12, P5).

The window controller (``emission_price``) advances once per window on window signals (fill time, snap on an
unfilled window). The async protocol has no window: this module keeps its state (``PriceState``), its decision
record (``PriceDecision``), its parameters (``PriceParams``: ``start``, ``floor``, ``cap``, ``rise``,
``rise_wait_fraction``, and ``decay`` per ``rounds_per_step`` drand rounds) and its walk (multiply, then clamp), and
replaces only the triggers, read once per tick (``tick_s`` of wall time, or one trainer step: ``tick``):

* descend (``decay`` scaled to the tick's length) when the env's buffer overflows (``TrainingBuffer.depth`` above
  ``overflow_groups``): miners supply more than the trainer takes;
* else rise (``x rise``) when the trainer waited on the env for more than ``rise_wait_fraction`` of the tick
  (``note_wait``, fed by the trainer, P4), or when the env's share of the tick's trained matched tokens is below
  ``share_tolerance`` x its target share (only when the trainer trained anything in the tick);
* else hold.

The price is a multiplier of the env's nominal rate, as the window price multiplied the env's pool. Per matched
token an env pays ``multiplier x cap x target_share / planned_tokens_per_period``: at multiplier 1, with the
trainer consuming its planned tokens at the target mix, one settlement period pays exactly the task cap. More
consumption, or prices above 1 (an under-supplied env may rise to ``cap`` = 4 x nominal), are bounded by the
settlement's normalisation (``rl_ledger``): a period never pays more than the cap, the rest burns.

The window snap and breaker are not carried over: there is no window to fill, and a dry env's price simply sits at
its cap. State lives in the lease book's SQLite file (``rl_prices``) and survives a restart.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from reliquary.validator.emission_price import (
    PRODUCTION_PRICE_PARAMS, ROUND_SECONDS, PriceDecision, PriceParams, PriceState,
)

ASYNC_BASE_PARAMS = replace(PRODUCTION_PRICE_PARAMS, cap=4.0)
RISE, DESCEND, HOLD = "rise", "descend", "hold"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_prices(environment TEXT PRIMARY KEY, price REAL NOT NULL, last_good REAL NOT NULL,
    regime TEXT NOT NULL, at REAL NOT NULL, signal TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_price_clock(id INTEGER PRIMARY KEY CHECK(id=0), last_tick REAL NOT NULL);
"""


@dataclass(frozen=True)
class AsyncPriceParams:
    base: PriceParams = ASYNC_BASE_PARAMS
    tick_s: float = 300.0
    share_tolerance: float = 0.9
    overflow_groups: int = 64

    def __post_init__(self) -> None:
        if not (self.tick_s > 0 and 0 < self.share_tolerance <= 1 and self.overflow_groups > 0):
            raise ValueError("tick_s > 0, share_tolerance in (0, 1], overflow_groups > 0")
        if not 0 < self.base.floor <= self.base.start <= self.base.cap:
            raise ValueError("floor <= start <= cap")


@dataclass(frozen=True)
class EnvSignal:
    """One env over one tick: ``trained_tokens`` its matched tokens trained, ``total_trained_tokens`` the trainer's
    over all envs, ``trainer_wait_s`` the seconds the trainer waited on it, ``buffer_depth`` its queued groups."""
    seconds: float
    target_share: float
    trained_tokens: int = 0
    total_trained_tokens: int = 0
    trainer_wait_s: float = 0.0
    buffer_depth: int = 0

    @property
    def share(self) -> float | None:
        return self.trained_tokens / self.total_trained_tokens if self.total_trained_tokens > 0 else None


def advance_async(state: PriceState, signal: EnvSignal, params: AsyncPriceParams) -> PriceDecision:
    """One env's price after one tick (pure). ``r`` carries the tick's share of trained tokens (None: no training)."""
    base = params.base
    price = state.price
    share = signal.share
    if signal.buffer_depth > params.overflow_groups:
        price *= base.decay ** (signal.seconds / (base.rounds_per_step * ROUND_SECONDS))
        regime = DESCEND
    elif (signal.trainer_wait_s > base.rise_wait_fraction * signal.seconds
          or (share is not None and share < params.share_tolerance * signal.target_share)):
        price *= base.rise
        regime = RISE
    else:
        regime = HOLD
    price = min(max(price, base.floor), base.cap)
    return PriceDecision(price=price, last_good=state.last_good, recent_fill_prices=(), r=share, r_smoothed=None,
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
        self._waits: dict[str, float] = {}
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
    def note_wait(self, environment: str, seconds: float) -> None:
        """The trainer was ready and waited ``seconds`` on ``environment``'s supply (P4 reports it)."""
        if seconds > 0:
            self._waits[environment] = self._waits.get(environment, 0.0) + float(seconds)

    def due(self, now: float | None = None) -> bool:
        now = float(self.clock()) if now is None else now
        with self.book.lock:
            last = self.book.db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
        return now - last >= self.params.tick_s

    def tick(self, trained_tokens: Mapping[str, int], buffer_depth: Mapping[str, int],
             now: float | None = None) -> dict[str, PriceDecision]:
        """Advance every target env over the time since the last tick (a timer or one trainer step).
        ``trained_tokens``: env -> matched tokens trained since the last tick (``RlPayLedger.trained_tokens``)."""
        now = float(self.clock()) if now is None else now
        total = sum(int(v) for v in trained_tokens.values())
        decisions: dict[str, PriceDecision] = {}
        with self.book.transaction():
            db = self.book.db
            last = db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
            seconds = max(now - last, 0.0)
            if seconds <= 0:
                return {}
            for env in sorted(self.targets):
                row = db.execute("SELECT price, last_good FROM rl_prices WHERE environment=?", (env,)).fetchone()
                start = self.params.base.start
                state = PriceState(price=start, last_good=start) if row is None else PriceState(*row)
                signal = EnvSignal(seconds, float(self.targets[env]), int(trained_tokens.get(env, 0)), total,
                                   self._waits.get(env, 0.0), int(buffer_depth.get(env, 0)))
                decision = advance_async(state, signal, self.params)
                decisions[env] = decision
                detail = {"seconds": seconds, "share": decision.r, "target_share": signal.target_share,
                          "trained_tokens": signal.trained_tokens, "trainer_wait_s": signal.trainer_wait_s,
                          "buffer_depth": signal.buffer_depth}
                db.execute("INSERT INTO rl_prices(environment, price, last_good, regime, at, signal) "
                           "VALUES(?,?,?,?,?,?) ON CONFLICT(environment) DO UPDATE SET price=excluded.price, "
                           "last_good=excluded.last_good, regime=excluded.regime, at=excluded.at, "
                           "signal=excluded.signal",
                           (env, decision.price, decision.last_good, decision.regime, now,
                            json.dumps(detail, sort_keys=True)))
            db.execute("UPDATE rl_price_clock SET last_tick=? WHERE id=0", (now,))
        self._waits.clear()
        return decisions


__all__ = [
    "ASYNC_BASE_PARAMS", "DESCEND", "HOLD", "RISE", "AsyncPriceController", "AsyncPriceParams", "EnvSignal",
    "advance_async",
]
