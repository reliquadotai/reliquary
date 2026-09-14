"""The price block a miner reads from ``GET /tasks`` before committing hardware.

Pure: every input is passed in, so the route stays an attribute read and the
numbers are testable without a running validator.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from reliquary.validator.emission_price import PriceParams, WindowOutcome

# How far ahead the price is projected, in wall-clock seconds.
PROJECTION_HORIZONS_SECONDS = {"1h": 3600.0, "6h": 21600.0, "24h": 86400.0}
# How many recent windows the fill time and the in-window share are read from.
_RECENT_WINDOWS = 12


def price_view(
    *,
    shadow: Mapping[str, Any] | None,
    outcomes: Sequence[WindowOutcome],
    params: PriceParams,
    window_pool: float | None,
    places_per_window: int | None,
    round_seconds: float,
) -> dict[str, Any] | None:
    """What the price is, why it moves, what one place pays, and where it heads."""
    if not shadow:
        return None
    value = float(shadow["price"])
    applied = bool(shadow.get("applied", False))
    regime = shadow.get("regime")
    recent = list(outcomes)[-_RECENT_WINDOWS:]
    fills = [
        outcome.collect_ready_round - outcome.open_round
        for outcome in recent
        if outcome.collect_ready_round is not None
    ]
    elapsed = [outcome.elapsed_rounds for outcome in recent if outcome.elapsed_rounds > 0]
    view: dict[str, Any] = {
        "applied": applied,
        "value": value,
        "regime": regime,
        "floor": params.floor,
        "cap": params.cap,
        "decay": params.decay,
        "rounds_per_step": params.rounds_per_step,
        "deadband": params.deadband,
        "snap": params.snap,
        "fill_seconds_recent": (
            statistics.median(fills) * round_seconds if fills else None
        ),
        "fill_target_seconds": (
            params.deadband * statistics.median(elapsed) * round_seconds
            if elapsed else None
        ),
        "pay_per_place_share_of_window": (
            window_pool / places_per_window
            if window_pool is not None and places_per_window else None
        ),
        "projection": _projection(value, regime, recent, params, round_seconds),
    }
    if not applied:
        view["pay_per_place_if_applied"] = (
            value / places_per_window if places_per_window else None
        )
    by_environment = shadow.get("by_environment")
    if isinstance(by_environment, Mapping):
        view["by_environment"] = {
            environment: {"value": decision.get("price"), "regime": decision.get("regime")}
            for environment, decision in by_environment.items()
            if isinstance(decision, Mapping)
        }
    return view


def _projection(
    value: float,
    regime: str | None,
    recent: Sequence[WindowOutcome],
    params: PriceParams,
    round_seconds: float,
) -> dict[str, float] | None:
    """The price at each horizon if collection keeps its current pace.

    Descent accrues only inside a window, so wall time is scaled by the share of
    each cycle a window occupies; a snapping or frozen price is not projected.
    """
    if regime == "hold":
        return {label: value for label in PROJECTION_HORIZONS_SECONDS}
    if regime != "descend" or len(recent) < 2:
        return None
    spacings = [
        later.open_round - earlier.open_round
        for earlier, later in zip(recent, recent[1:])
        if later.open_round > earlier.open_round
    ]
    elapsed = [outcome.elapsed_rounds for outcome in recent if outcome.elapsed_rounds > 0]
    if not spacings or not elapsed:
        return None
    share = min(1.0, statistics.median(elapsed) / statistics.median(spacings))
    return {
        label: max(
            params.floor,
            value * params.decay ** (seconds / round_seconds * share / params.rounds_per_step),
        )
        for label, seconds in PROJECTION_HORIZONS_SECONDS.items()
    }
