"""Discovered emission price for one task, replayed from window outcomes.

The pool is wired to ``1.0`` per window today, so the subnet pays its full
miner share regardless of supply, of what the trainer can consume, or of how
fast the window filled. This module turns that number into a price the market
discovers, and the share it does not pay burns.

The controller tracks ``r = t_collect / max(t_training, t_validation)``: paying
for collection faster than the cycle's own incompressible time buys nothing.
The target is therefore our own performance, so repairing the proof plane
lowers the price with no constant to retune.

Every quantity is in BLOCKS. See the module's test for why that is not a
stylistic choice.

Purity matters as much as the arithmetic: any weight-only node must be able to
replay this from the published archives and land on the same number, exactly as
it already replays the EMA. Nothing here may read a clock, an environment
variable, or any state that is not an argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import math
import statistics


@dataclass(frozen=True, slots=True)
class PriceParams:
    """Controller settings. Versioned in the image, never env-overridable."""

    start: float
    decay: float
    blocks_per_step: int
    deadband: float
    snap: float
    floor: float
    cap: float
    median_blocks: int


@dataclass(frozen=True, slots=True)
class WindowOutcome:
    """What one archived window contributes to the price.

    ``collect_ready_block`` is when the N-th ADMISSIBLE candidate arrived, not
    when it was proven. The distinction is the whole sensor: a fill-closed
    window closes on PROVEN groups, so a proof-clock reading would measure our
    own plane (11 proofs/min) and never the market.
    """

    open_block: int
    close_block: int
    collect_ready_block: int | None
    training_blocks: float
    validation_blocks: float

    @property
    def elapsed_blocks(self) -> int:
        return int(self.close_block) - int(self.open_block)

    @property
    def filled(self) -> bool:
        """Whether the window gathered its target at all."""
        return self.collect_ready_block is not None

    @property
    def ratio(self) -> float | None:
        """``r``, or None when no ratio can be computed.

        None means "no signal", NOT "shortage": a window that filled but whose
        stage durations were never recorded carries no information either way.
        Shortage is ``filled`` being false, and only that.
        """
        if self.collect_ready_block is None:
            return None
        incompressible = max(self.training_blocks, self.validation_blocks)
        if incompressible <= 0:
            return None
        return (int(self.collect_ready_block) - int(self.open_block)) / incompressible


@dataclass(frozen=True, slots=True)
class PriceState:
    """What one archive carries forward. Bounded on purpose: see ``advance``."""

    price: float
    last_good: float


@dataclass(frozen=True, slots=True)
class PriceDecision:
    """The price plus the inputs that produced it, so the archive can carry both."""

    price: float
    last_good: float
    r: float | None
    r_smoothed: float | None
    regime: str

    @property
    def state(self) -> PriceState:
        return PriceState(price=self.price, last_good=self.last_good)


def _smoothed_ratio(
    recent: Sequence[WindowOutcome], median_blocks: int
) -> float | None:
    """Median ``r`` over the trailing ``median_blocks``, or None if not finite.

    A window that never filled contributes ``inf``: it is maximally slow, and
    letting it be absent would allow a run of failures to leave the median
    claiming over-supply. When the median itself is infinite there is no usable
    ratio -- and no finite value could be archived either, since Reliquary's
    canonical JSON refuses non-finite floats.
    """
    cutoff = recent[-1].close_block - median_blocks
    samples: list[float] = []
    for outcome in recent:
        if outcome.close_block <= cutoff:
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
    is the window being decided; anything outside ``median_blocks`` is dropped
    here rather than trusted from the caller.

    Taking a ``PriceState`` instead of the whole chain is what makes this
    replayable: ``_replay_ema`` reads a bounded slice of archives, so a price
    that needed folding back to genesis would put a weight-only node and the
    validator that wrote the archive on different numbers.
    """
    outcome = recent[-1]
    r = outcome.ratio
    r_smoothed = _smoothed_ratio(recent, params.median_blocks)
    price = state.price
    last_good = state.last_good
    if not outcome.filled:
        # The window never gathered its target, so the trainer is stopped:
        # there is no graceful degradation to ride out. Escalating from the
        # CURRENT price rather than pinning to ``last_good`` is what keeps a
        # snap that fails to restore supply from parking there forever.
        price = max(price, last_good) * params.snap
        regime = "snap"
    elif r_smoothed is not None and r_smoothed < params.deadband:
        price *= params.decay ** (outcome.elapsed_blocks / params.blocks_per_step)
        regime = "descend"
    else:
        regime = "hold"
    price = min(max(price, params.floor), params.cap)
    if outcome.filled:
        # Only a window that actually filled proves a price works.
        last_good = price
    return PriceDecision(
        price=price,
        last_good=last_good,
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
        r=None,
        r_smoothed=None,
        regime="hold",
    )
    for index in range(len(outcomes)):
        decision = advance(state, outcomes[: index + 1], params)
        state = decision.state
    return decision


# Absent from every archive written before this shipped. Their absence is how
# the replay tells "not instrumented" from "did not fill": the latter is an
# explicit ``collect_ready_block: null`` INSIDE an otherwise complete record.
_REQUIRED_ARCHIVE_FIELDS = (
    "window_open_block",
    "window_close_block",
    "training_blocks",
    "validation_blocks",
)


def outcome_from_archive(record: Mapping[str, Any]) -> WindowOutcome | None:
    """Read one archive's price signal, or None when it carries none.

    Aborted windows are skipped for the same reason ``_replay_ema`` skips them:
    their timings describe the abort, not the market.
    """
    if record.get("window_status", "completed") == "aborted":
        return None
    if any(record.get(field) is None for field in _REQUIRED_ARCHIVE_FIELDS):
        return None
    ready = record.get("collect_ready_block")
    return WindowOutcome(
        open_block=int(record["window_open_block"]),
        close_block=int(record["window_close_block"]),
        collect_ready_block=None if ready is None else int(ready),
        training_blocks=float(record["training_blocks"]),
        validation_blocks=float(record["validation_blocks"]),
    )
