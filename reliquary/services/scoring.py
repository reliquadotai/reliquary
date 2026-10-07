"""Service scoring, distinct from miner payment and training selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Signal:
    category: str
    sigma: float | None
    mean: float | None
    k: int | None
    expected: int
    received: int

    @property
    def in_zone(self) -> bool:
        return self.category == "in-zone"


def weighted_reward(metrics: Mapping[str, float | None], weights_bps: Mapping[str, int]) -> float | None:
    """Missing/error metrics stay unknown; they do not turn into zero rewards."""
    if not weights_bps or any(type(w) is not int or not 0 < w <= 10000 for w in weights_bps.values()) or sum(weights_bps.values()) != 10000:
        raise ValueError("weights must be positive integer basis points summing to 10000")
    total = 0.0
    for metric, weight in weights_bps.items():
        value = metrics.get(metric)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid reward metric {metric}")
        total += value * weight / 10000
    return total


def classify_signal(rewards: Sequence[float | None], *, expected: int, sigma_min_bps: int) -> Signal:
    if type(expected) is not int or not 2 <= expected <= 65536:
        raise ValueError("expected group size must be in 2..65536")
    if type(sigma_min_bps) is not int or not 0 <= sigma_min_bps <= 10000:
        raise ValueError("sigma_min_bps must be in 0..10000")
    if len(rewards) > expected:
        raise ValueError("more rewards than expected slots")
    known = []
    for value in rewards:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("rewards must be finite in [0,1] or None")
        known.append(float(value))
    if len(rewards) != expected or len(known) != expected:
        return Signal("unknown", None, None, None, expected, len(known))
    sigma, average = pstdev(known), mean(known)
    k = sum(v == 1 for v in known) if all(v in (0, 1) for v in known) else None
    if sigma < 1e-8:
        category = "uniform-low" if average == 0 else "uniform-high" if average == 1 else "uniform-intermediate"
    else:
        category = "in-zone" if sigma >= sigma_min_bps / 10000 else "diverse-below-threshold"
    return Signal(category, sigma, average, k, expected, len(known))
