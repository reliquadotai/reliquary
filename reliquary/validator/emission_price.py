"""Discovered emission price for one task, replayed from window outcomes.

The pool is wired to ``1.0`` per window today, so the subnet pays its full
miner share regardless of supply, of what the trainer can consume, or of how
fast the window filled. This module turns that number into a price the market
discovers, and the share it does not pay burns.

The controller tracks ``r = t_collect / max(t_training, t_validation)``: paying
for collection faster than the cycle's own incompressible time buys nothing.
The target is therefore our own performance, so repairing the proof plane
lowers the price with no constant to retune.

Every quantity is in DRAND ROUNDS -- quicknet, one every 3 seconds. Not
windows: see the module's test for why that is not a stylistic choice. Not
chain blocks either, though they would serve: the window loop makes no chain
call, while ``window_open_drand_round`` is already computed at window open and
the whole seal path already reasons in rounds. Rounds are three times finer and
cost nothing extra.

Whichever it is, the unit must be named honestly. Calling a round a "block"
would be the same unit confusion that quietly turned EMA_ALPHA into a 28-hour
time constant.

Purity matters as much as the arithmetic: any weight-only node must be able to
replay this from the published archives and land on the same number, exactly as
it already replays the EMA. Nothing here may read a clock, an environment
variable, or any state that is not an argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import logging
import math
import os
import statistics

logger = logging.getLogger(__name__)

# The price rise on trainer wait (``PriceParams.rise``) is OFF unless this is set:
# 1.0 (the default) keeps the controller's price walk exactly as before the rise
# existed. The next RL run sets it to 1.10 (docs/next-rl-run-decisions.md). Only
# the controller walks prices; weight-only validators read archived rewards.
PRICE_RISE_ENV = "RELIQUARY_EMISSION_PRICE_RISE"


def price_rise_from_env(environ: Mapping[str, str] | None = None) -> float:
    """``RELIQUARY_EMISSION_PRICE_RISE`` as a finite factor >= 1.0 (default 1.0:
    no rise). A bad value refuses the import rather than pricing silently."""
    raw = (os.environ if environ is None else environ).get(PRICE_RISE_ENV, "").strip()
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{PRICE_RISE_ENV} must be a number >= 1.0, got {raw!r}") from None
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"{PRICE_RISE_ENV} must be a finite number >= 1.0, got {raw!r}")
    return value


PRICE_RISE = price_rise_from_env()


@dataclass(frozen=True, slots=True)
class PriceParams:
    """Controller settings. Versioned in the image, never env-overridable."""

    start: float
    decay: float
    rounds_per_step: int
    deadband: float
    snap: float
    floor: float
    cap: float
    median_rounds: int
    # How many of the most recent FILLING windows ``last_good`` is the rolling
    # minimum over -- a count of fills, not of windows, so a run of shortages
    # does not shrink it and a single expensive fill cannot raise it. See
    # ``PriceState.recent_fill_prices``.
    last_good_fills: int
    # How many consecutive unfilled windows before an environment's price stops
    # escalating: past this, the likelier explanation is that the environment is
    # broken on our side, not that the market is short.
    breaker_timeouts: int = 3
    # The trainer was idle and waited on this environment's supply for more
    # than ``rise_wait_fraction`` of the window: multiply the price by
    # ``rise``. Image-versioned like ``breaker_timeouts``: the registry does not
    # carry them (PRICE_PARAM_FIELDS), so these defaults ARE the production
    # values every registry-built params object gets. ``rise`` is 1.0 (off)
    # unless ``RELIQUARY_EMISSION_PRICE_RISE`` sets it.
    rise: float = PRICE_RISE
    rise_wait_fraction: float = 0.05


@dataclass(frozen=True, slots=True)
class WindowOutcome:
    """What one archived window contributes to the price.

    ``collect_ready_round`` is when the N-th ADMISSIBLE candidate arrived, not
    when it was proven. The distinction is the whole sensor: a fill-closed
    window closes on PROVEN groups, so a proof-clock reading would measure our
    own plane (11 proofs/min) and never the market.
    """

    open_round: int
    close_round: int
    collect_ready_round: int | None
    incompressible_rounds: float

    @property
    def elapsed_rounds(self) -> int:
        return int(self.close_round) - int(self.open_round)

    @property
    def filled(self) -> bool:
        """Whether the window gathered its target at all."""
        return self.collect_ready_round is not None

    @property
    def ratio(self) -> float | None:
        """``r``, or None when no ratio can be computed.

        None means "no signal", NOT "shortage": a window that filled but whose
        stage durations were never recorded carries no information either way.
        Shortage is ``filled`` being false, and only that.
        """
        if self.collect_ready_round is None:
            return None
        if self.incompressible_rounds <= 0:
            return None
        return (
            int(self.collect_ready_round) - int(self.open_round)
        ) / self.incompressible_rounds


@dataclass(frozen=True, slots=True)
class PriceState:
    """What one archive carries forward. Bounded on purpose: see ``advance``.

    ``recent_fill_prices`` is capped at ``params.last_good_fills`` entries --
    fixed-size, so it costs ``replay()`` nothing to re-fold from genesis.
    ``last_good`` stays as the reported scalar so nothing downstream breaks;
    it is DERIVED as the minimum of the tuple when the tuple is non-empty.

    An archive written before this field existed carries none: the default,
    an empty tuple, makes that state read as "no fills recorded yet", and
    ``advance`` falls back to the carried scalar rather than treating the
    empty tuple as a fresh, unfilled controller.
    """

    price: float
    last_good: float
    recent_fill_prices: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PriceDecision:
    """The price plus the inputs that produced it, so the archive can carry both."""

    price: float
    last_good: float
    recent_fill_prices: tuple[float, ...]
    r: float | None
    r_smoothed: float | None
    regime: str

    @property
    def state(self) -> PriceState:
        return PriceState(
            price=self.price,
            last_good=self.last_good,
            recent_fill_prices=self.recent_fill_prices,
        )


def _timed_out(outcome: Any) -> bool:
    """Whether this outcome's window timed out.

    Read leniently because both walks share ``advance``: only
    ``EnvironmentOutcome`` carries the field, and the scalar ``WindowOutcome``
    walk must stay byte-identical to what it was before it existed.
    """
    return bool(getattr(outcome, "timed_out", False))


def _waited_on(outcome: Any, params: PriceParams) -> bool:
    """The trainer was ready and waited on this environment's supply for more
    than ``rise_wait_fraction`` of the window: collection slowed training.

    ``r`` cannot say this: its denominator, the training span, grows with the
    very wait, so it saturates near 1 when collection is the bottleneck.
    """
    wait = getattr(outcome, "trainer_wait_seconds", None)
    if wait is None or _timed_out(outcome):
        return False
    window_seconds = outcome.elapsed_rounds * ROUND_SECONDS
    return window_seconds > 0 and float(wait) > params.rise_wait_fraction * window_seconds


def _smoothed_ratio(
    recent: Sequence[WindowOutcome], median_rounds: int
) -> float | None:
    """Median ``r`` over the trailing ``median_rounds``, or None if not finite.

    A window that never filled contributes ``inf``: it is maximally slow, and
    letting it be absent would allow a run of failures to leave the median
    claiming over-supply. When the median itself is infinite there is no usable
    ratio -- and no finite value could be archived either, since Reliquary's
    canonical JSON refuses non-finite floats.
    """
    cutoff = recent[-1].close_round - median_rounds
    samples: list[float] = []
    for outcome in recent:
        if outcome.close_round <= cutoff:
            continue
        if not outcome.filled:
            samples.append(math.inf)
            continue
        ratio = outcome.ratio
        if ratio is not None:
            samples.append(ratio)
    if not samples:
        return None
    value = statistics.median(samples)
    return value if math.isfinite(value) else None


def advance(
    state: PriceState,
    recent: Sequence[WindowOutcome],
    params: PriceParams,
) -> PriceDecision:
    """Decide the price after one window.

    ``recent`` is the trailing run of outcomes, oldest first, whose LAST element
    is the window being decided; anything outside ``median_rounds`` is dropped
    here rather than trusted from the caller.

    Taking a ``PriceState`` instead of the whole chain is what makes this
    replayable: ``_replay_ema`` reads a bounded slice of archives, so a price
    that needed folding back to genesis would put a weight-only node and the
    validator that wrote the archive on different numbers.
    """
    outcome = recent[-1]
    r = outcome.ratio
    r_smoothed = _smoothed_ratio(recent, params.median_rounds)
    price = state.price
    recent_fill_prices = state.recent_fill_prices
    # ``last_good`` is DERIVED from the rolling tuple, not carried as an
    # independent value: a fill at a raised price appends to the tuple, it
    # never overwrites it, so one expensive incident cannot lift the floor
    # the next snap escalates from. An empty tuple -- genesis, or a state
    # revived from an archive written before this field existed -- falls
    # back to the carried scalar instead of reading as "nothing ever filled".
    last_good = min(recent_fill_prices) if recent_fill_prices else state.last_good
    if not outcome.filled:
        # The window never gathered its target, so the trainer is stopped:
        # there is no graceful degradation to ride out. Escalating from the
        # CURRENT price rather than pinning to ``last_good`` is what keeps a
        # snap that fails to restore supply from parking there forever.
        price = max(price, last_good) * params.snap
        regime = "snap"
    elif params.rise > 1.0 and _waited_on(outcome, params):
        # Off (1.0): the window takes the deadband/decay path, as before the rise.
        price *= params.rise
        regime = "rise"
    elif r_smoothed is not None and r_smoothed < params.deadband:
        price *= params.decay ** (outcome.elapsed_rounds / params.rounds_per_step)
        regime = "descend"
    else:
        regime = "hold"
    price = min(max(price, params.floor), params.cap)
    # A timed-out window assembled no batch, so it is not evidence that this
    # price works no matter what its admission clock says.
    if outcome.filled and not _timed_out(outcome):
        # Only a window that actually filled proves a price works. It joins
        # the rolling window rather than replacing ``last_good`` outright, so
        # the minimum -- not the latest fill -- is what the next snap reads.
        recent_fill_prices = (*recent_fill_prices, price)[-params.last_good_fills:]
        last_good = min(recent_fill_prices)
    return PriceDecision(
        price=price,
        last_good=last_good,
        recent_fill_prices=recent_fill_prices,
        r=r,
        r_smoothed=r_smoothed,
        regime=regime,
    )


def replay(
    outcomes: Sequence[WindowOutcome], params: PriceParams
) -> PriceDecision:
    """Fold a whole history. The verification path, not the hot one."""
    state = PriceState(price=params.start, last_good=params.start)
    decision = PriceDecision(
        price=params.start,
        last_good=params.start,
        recent_fill_prices=(),
        r=None,
        r_smoothed=None,
        regime="hold",
    )
    for index in range(len(outcomes)):
        decision = advance(state, outcomes[: index + 1], params)
        state = decision.state
    return decision


# Versioned in the image on purpose, and deliberately NOT in constants.py: the
# replay has to be reproducible from a release, while constants.py is where
# env-overridable settings live. An operator turning a knob at runtime would
# put two weight-only nodes on two different prices.
#
# None of these numbers is settled. They are a starting point whose whole job
# is to produce a shadow curve legible enough to calibrate them against.
# The drand quicknet period. Versioned here rather than read from drand at
# runtime, so replaying an archive never depends on a live read.
ROUND_SECONDS = 3.0

PRODUCTION_PRICE_PARAMS = PriceParams(
    # Today's pool, so the first armed window changes nothing.
    start=1.0,
    # -2% per ~50 minutes of window time: about a week from 1.0 to 0.05 at a
    # 17-minute cycle, still ~5x slower than the ~12 h miners take to feel a price.
    decay=0.98,
    rounds_per_step=1000,
    # Collection close to the incompressible time is the target, not a signal.
    deadband=0.80,
    snap=1.20,
    # A liveness guard, not an economic opinion. Where the cliff actually sits
    # is what the shadow phase exists to find out.
    floor=0.05,
    cap=1.0,
    # ~4 hours: several windows of confirmation before spending less, against a
    # loop delay of one miner spin-up plus one window.
    median_rounds=4800,
    # A starting point, to be calibrated during the shadow phase like every
    # other number in this block: the rolling minimum tracks the last 50
    # filling windows.
    last_good_fills=50,
    # A starting point, calibrated in shadow like every other number here.
    breaker_timeouts=3,
    # +10 % per window in which the trainer waited on the environment for more
    # than 5 % of the window (~22 s of a 450 s window). In normal windows every
    # environment fills in 80-130 s while the trainer takes ~350 s, so the
    # trainer never waits on supply (measured, 1,000 windows, 10-13).
    # OFF (1.0) unless RELIQUARY_EMISSION_PRICE_RISE is set; the next run sets 1.10.
    rise=PRICE_RISE,
    rise_wait_fraction=0.05,
)
