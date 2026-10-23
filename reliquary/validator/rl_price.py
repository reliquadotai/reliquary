"""The async per-env price controller (rl-async-design.md §4.9; implementation map §1.12, P5).

The target is the trainer's speed: each env must supply its target share of what the trainer consumes. Everything
is read from the training queue (``TrainingBuffer.drain_tick``), nothing is reported by the trainer. Per tick
(``tick_s`` of wall time, the validator's timer):

* **trainer speed** = the completion tokens it took from the queue over the seconds it did not wait for groups
  (kept as sums decayed with a half-life of one tick, persisted);
* **demand** of an env = its target share (renormalised over the envs that supplied anything in the tick: a dry
  env's share goes to the others, since the trainer takes what arrives) x the trainer's speed;
* **supply** = the completion tokens of the env's groups queued for training during the tick, per second;
* ``r = supply / demand``: under ``low`` the price rises (x ``rise``), over ``high`` it descends (``decay`` scaled to
  the tick's length), else it holds. An env refused leases for the global cap during the tick
  (``LeaseBook.drain_full``) and not under ``low`` descends too.
* Two global guards, the queue's: while the cap refuses leases (the queue is at its most) no price rises, and while
  the trainer waits for groups (over ``starved`` of the tick) no price descends on its own ratio. Without them, miners
  moving between envs keep some env short at every tick: it rises x1.05 while the long ones descend 2 %, and the
  prices ratchet to the cap with the queue full (queue_sim, 2026-10-17).
* Steering under a shortage: while the trainer waits, ``relative`` = the env's share of the recent supply (sums
  decayed with a half-life of ``steer_halflife_ticks``) over its target share moves envs against each other. Outside
  ``[low, high]`` the price takes a step ``relative ** -steer_gain`` (within x/÷ ``steer_step``) on top of the level
  step above: an over-supplied env that is not short itself descends, a short one under-supplied against the mix
  out-rises the others. Otherwise a fleet short everywhere rises everywhere and pins every env at the ceiling with
  nothing moving miners between envs (teutonic-rl-async-v1, 2026-10-22: every multiplier at 4 for 10 h, hard math
  66 % of the pay for a 21 % target, SWE 1 % for 23 %).
* One ceiling for the board: when a rise would pass it, every multiplier is divided by the same factor. Ratios are
  what the miners see (settlement normalises to the cap), so a short env at the ceiling keeps gaining on the others
  instead of the others catching up with it.
* No speed yet, or the trainer neither took nor waited for ``stale_ticks`` ticks (trainer down): every price holds.

No global price: the trainer's speed is the sum of the demands, so every env at r ~ 1 means the miners as a whole
match the trainer. The price is a multiplier of one nominal rate, the same for every env: per matched token an env
pays ``multiplier x start_pool / planned_tokens_per_period``. At the target mix and multiplier 1 env e supplies
``share(e) x planned`` tokens and earns ``start_pool x share(e)``: the period pays ``start_pool`` (the starting
budget, the task's cap when not declared). Multipliers above 1 (up to 4, or ``cap / start_pool`` when larger, so
that scarcity can reach the cap) raise the total towards the cap; the settlement's normalisation (``rl_ledger``)
keeps the cap the most a period pays, the rest burns. State lives in the lease book's SQLite file (``rl_prices``,
``rl_trainer_speed``, ``rl_supply``) and survives a restart.
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
# Floor 0.02: the live mix needs ~65x between SWE and hard math at unit elasticity, hard math dipping to 0.03 on the
# way (test_rl_price_steer); 4 / 0.05 is 80x, 4 / 0.02 leaves room.
ASYNC_PRICE_FLOOR = 0.02
ASYNC_BASE_PARAMS = replace(PRODUCTION_PRICE_PARAMS, cap=4.0, rise=ASYNC_PRICE_RISE, decay=ASYNC_PRICE_DECAY,
                            floor=ASYNC_PRICE_FLOOR)
RISE, DESCEND, HOLD = "rise", "descend", "hold"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_prices(environment TEXT PRIMARY KEY, price REAL NOT NULL, last_good REAL NOT NULL,
    regime TEXT NOT NULL, at REAL NOT NULL, signal TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_price_clock(id INTEGER PRIMARY KEY CHECK(id=0), last_tick REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rl_trainer_speed(id INTEGER PRIMARY KEY CHECK(id=0), tokens REAL NOT NULL,
    seconds REAL NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rl_supply(environment TEXT PRIMARY KEY, tokens REAL NOT NULL);
"""


@dataclass(frozen=True)
class AsyncPriceParams:
    base: PriceParams = ASYNC_BASE_PARAMS
    tick_s: float = 300.0
    low: float = 0.90            # placeholders: the supply/demand band the price holds in
    high: float = 1.10
    stale_ticks: float = 3.0     # no trainer speed for this many ticks: hold every price
    starved: float = 0.05        # placeholder: the trainer waited over this share of the tick: no price descends
    # Steering under a shortage (test_rl_price_steer, 2026-10-23): the supply mix ``relative`` reads is decayed with
    # this half-life (long-episode envs arrive in lumps); the step against the mix is relative ** -gain within
    # x/÷ ``steer_step``. Larger steps or a slower mix oscillate against miners that take hours to move.
    steer_halflife_ticks: float = 2.0
    steer_step: float = 1.02
    steer_gain: float = 1.0

    def __post_init__(self) -> None:
        if not (self.tick_s > 0 and 0 < self.low <= self.high and self.stale_ticks > 0 and 0 <= self.starved < 1
                and self.steer_halflife_ticks > 0 and self.steer_step >= 1
                and self.steer_gain >= 0):
            raise ValueError("tick_s > 0, 0 < low <= high, stale_ticks > 0, 0 <= starved < 1, steering out of range")
        if not 0 < self.base.floor <= self.base.start <= self.base.cap:
            raise ValueError("floor <= start <= cap")


@dataclass(frozen=True)
class QueueTick:
    """The queue over one tick (``TrainingBuffer.drain_tick``): completion tokens queued per env, completion tokens
    the trainer took, seconds it waited for groups."""
    arrivals: Mapping[str, int] = field(default_factory=dict)
    taken: int = 0
    starved_s: float = 0.0


@dataclass(frozen=True)
class EnvSignal:
    """One env over one tick: ``supply_tps`` its queued completion tokens per second, ``demand_tps`` its target share
    of the trainer's speed (None: no usable speed), ``full`` refused leases for the global cap. The two global
    guards: ``capped`` the cap refused some lease (the queue is at its most), ``starved`` the trainer waited.
    ``relative``: the env's share of the recent supply over its target share (None: nothing supplied)."""
    seconds: float
    supply_tps: float
    demand_tps: float | None
    full: bool = False
    capped: bool = False
    starved: bool = False
    relative: float | None = None

    @property
    def ratio(self) -> float | None:
        return self.supply_tps / self.demand_tps if self.demand_tps else None


def _move(price: float, signal: EnvSignal, params: AsyncPriceParams) -> tuple[float, str]:
    base = params.base
    r = signal.ratio
    if r is None:
        return price, HOLD
    rise = base.rise if r < params.low and not signal.capped else 1.0   # a full queue raises no price: the mix waits
    if signal.starved and signal.relative is not None:
        # A shortage: the level step as below, times a step against the mix (none inside the band).
        rel = signal.relative
        step = 1.0 if params.low <= rel <= params.high else min(max(rel ** -params.steer_gain, 1 / params.steer_step),
                                                                params.steer_step)
        if signal.capped:
            step = min(step, 1.0)
        price *= rise * step
        return price, RISE if rise * step > 1 else DESCEND if rise * step < 1 else HOLD
    if rise > 1:
        return price * rise, RISE
    if (r > params.high or (signal.full and r >= params.low)) and not signal.starved:   # nor a waiting trainer
        return price * base.decay ** (signal.seconds / (base.rounds_per_step * ROUND_SECONDS)), DESCEND
    return price, HOLD


def advance_async(state: PriceState, signal: EnvSignal, params: AsyncPriceParams) -> PriceDecision:
    """One env's price after one tick (pure). ``r`` carries the tick's supply/demand ratio (None: hold)."""
    price, regime = _move(state.price, signal, params)
    price = min(max(price, params.base.floor), params.base.cap)
    return PriceDecision(price=price, last_good=state.last_good, recent_fill_prices=(), r=signal.ratio,
                         r_smoothed=None, regime=regime)


@dataclass
class AsyncPriceController:
    """Per-env multipliers, persisted in ``book``'s file. ``targets``: env -> target share of trained tokens (the
    trainer's mix, summing to 1); ``start_pool`` (None: ``cap``) and ``planned_tokens_per_period`` set the nominal
    rate per token, ``cap`` the most a period pays."""
    book: object
    targets: Mapping[str, float]
    cap: float
    planned_tokens_per_period: float
    start_pool: float | None = None
    params: AsyncPriceParams = field(default_factory=AsyncPriceParams)
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if not self.targets or any(v < 0 for v in self.targets.values()) or abs(sum(self.targets.values()) - 1) > 1e-6:
            raise ValueError("targets are non-negative shares summing to 1")
        if self.start_pool is None:
            self.start_pool = self.cap
        if not (self.cap > 0 and self.planned_tokens_per_period > 0 and 0 < self.start_pool <= self.cap):
            raise ValueError("cap and planned_tokens_per_period must be positive, 0 < start_pool <= cap")
        # A fleet short everywhere must still be priceable up to the cap: the ceiling is at least cap / start_pool.
        base = self.params.base
        if self.cap / self.start_pool > base.cap:
            self.params = replace(self.params, base=replace(base, cap=self.cap / self.start_pool))
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
        if float(self.targets.get(environment, 0.0)) <= 0:
            return 0.0
        return self.multiplier(environment) * self.nominal

    @property
    def nominal(self) -> float:
        """Every env's price per matched token at multiplier 1: the target mix pays ``start_pool`` a period."""
        return self.start_pool / self.planned_tokens_per_period

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
    def _note_speed(self, tokens: float, seconds: float, now: float) -> None:
        """Inside ``tick``'s transaction: ``tokens`` taken in ``seconds`` the trainer did not wait. A trainer that
        took nothing but waited is alive: its speed is kept, refreshed."""
        db = self.book.db
        row = db.execute("SELECT tokens, seconds, at FROM rl_trainer_speed WHERE id=0").fetchone()
        if tokens <= 0 or seconds <= 0:
            db.execute("UPDATE rl_trainer_speed SET at=? WHERE id=0", (now,))
            return
        keep = 0.0 if row is None else 0.5 ** (max(now - row[2], 0.0) / self.params.tick_s)
        db.execute("INSERT INTO rl_trainer_speed(id, tokens, seconds, at) VALUES(0,?,?,?) ON CONFLICT(id) DO UPDATE "
                   "SET tokens=excluded.tokens, seconds=excluded.seconds, at=excluded.at",
                   (tokens + (0.0 if row is None else keep * row[0]), seconds + (0.0 if row is None else keep * row[1]),
                    now))

    def _note_supply(self, arrivals: Mapping[str, int], seconds: float) -> dict[str, float]:
        """Inside ``tick``'s transaction: each env's queued tokens, decayed with a half-life of
        ``steer_halflife_ticks`` ticks (persisted). Returns the sums."""
        db = self.book.db
        keep = 0.5 ** (seconds / (self.params.steer_halflife_ticks * self.params.tick_s))
        old = dict(db.execute("SELECT environment, tokens FROM rl_supply").fetchall())
        out = {env: keep * float(old.get(env, 0.0)) + float(arrivals.get(env, 0)) for env in self.targets}
        db.executemany("INSERT INTO rl_supply(environment, tokens) VALUES(?,?) ON CONFLICT(environment) DO UPDATE "
                       "SET tokens=excluded.tokens", list(out.items()))
        return out

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

    def tick(self, queue: QueueTick, full: Iterable[str] = (), now: float | None = None) -> dict[str, PriceDecision]:
        """Advance every target env over the time since the last tick. ``queue``: ``TrainingBuffer.drain_tick``;
        ``full``: envs refused for the global cap (``LeaseBook.drain_full``)."""
        now = float(self.clock()) if now is None else now
        arrivals = queue.arrivals
        full = set(full)
        decisions: dict[str, PriceDecision] = {}
        with self.book.transaction():
            db = self.book.db
            last = db.execute("SELECT last_tick FROM rl_price_clock WHERE id=0").fetchone()[0]
            seconds = max(now - last, 0.0)
            if seconds <= 0:
                return {}
            if queue.taken or queue.starved_s:
                self._note_speed(float(queue.taken), seconds - min(float(queue.starved_s), seconds), now)
            speed = self.trainer_speed(now)
            # An env nobody supplied gives its share to the others (the trainer takes what arrives), and still rises.
            live = sum(float(v) for env, v in self.targets.items() if arrivals.get(env, 0) > 0) or 1.0
            capped, starved = bool(full), queue.starved_s > self.params.starved * seconds
            mix = self._note_supply(arrivals, seconds)
            supplied = sum(mix.values())
            moves: dict[str, tuple[PriceState, EnvSignal, float, float, str]] = {}
            for env in sorted(self.targets):
                row = db.execute("SELECT price, last_good FROM rl_prices WHERE environment=?", (env,)).fetchone()
                start = self.params.base.start
                state = PriceState(price=start, last_good=start) if row is None else PriceState(*row)
                share = float(self.targets[env])
                if arrivals.get(env, 0) > 0:
                    share /= live
                target = float(self.targets[env])
                relative = mix[env] / supplied / target if supplied > 0 and target > 0 else None
                signal = EnvSignal(seconds, int(arrivals.get(env, 0)) / seconds,
                                   None if speed is None else share * speed, env in full, capped, starved, relative)
                moves[env] = (state, signal, share, *_move(state.price, signal, self.params))
            # One ceiling for the board: the rise past it lowers every env alike (ratios kept).
            base = self.params.base
            over = max((m[3] for m in moves.values()), default=0.0) / base.cap
            rescale = over if over > 1 else 1.0
            for env, (state, signal, share, price, regime) in moves.items():
                price = min(max(price / rescale, base.floor), base.cap)
                decision = PriceDecision(price=price, last_good=state.last_good, recent_fill_prices=(),
                                         r=signal.ratio, r_smoothed=None, regime=regime)
                decisions[env] = decision
                detail = {"seconds": seconds, "ratio": decision.r, "supply_tps": signal.supply_tps,
                          "demand_tps": signal.demand_tps, "trainer_tps": speed, "target_share": share,
                          "full": signal.full, "starved_s": queue.starved_s, "relative": signal.relative,
                          "rescale": rescale}
                db.execute("INSERT INTO rl_prices(environment, price, last_good, regime, at, signal) "
                           "VALUES(?,?,?,?,?,?) ON CONFLICT(environment) DO UPDATE SET price=excluded.price, "
                           "last_good=excluded.last_good, regime=excluded.regime, at=excluded.at, "
                           "signal=excluded.signal",
                           (env, decision.price, decision.last_good, decision.regime, now,
                            json.dumps(detail, sort_keys=True)))
            db.execute("UPDATE rl_price_clock SET last_tick=? WHERE id=0", (now,))
        return decisions


__all__ = [
    "ASYNC_BASE_PARAMS", "ASYNC_PRICE_FLOOR", "ASYNC_PRICE_RISE", "DESCEND", "HOLD", "RISE", "AsyncPriceController",
    "AsyncPriceParams", "EnvSignal", "QueueTick", "advance_async",
]
