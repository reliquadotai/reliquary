"""The report's arithmetic, stdlib only.

The unbiased pass@k estimator and the k-histogram are ported from
reliquary-compare ``src/eval/histogram.py`` (Chen et al. 2021); the bootstrap
is new. A problem is ``(c, n)``: ``c`` correct of ``n`` graded samples.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

BOOTSTRAP_RESAMPLES = 2000


def pass_at_k_single(c: int, n: int, k: int) -> float:
    """``1 - C(n-c, k) / C(n, k)``, as a running product so no binomial is built."""
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n={n}], got {k}")
    if not 0 <= c <= n:
        raise ValueError(f"c={c} outside [0, {n}]")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio


def pass_at_k(problems: Sequence[tuple[int, int]], k: int) -> float:
    """The mean over problems of the per-problem estimator, each at its own n."""
    if not problems:
        raise ValueError("no problems")
    return sum(pass_at_k_single(c, n, k) for c, n in problems) / len(problems)


def k_histogram(k_correct: Sequence[int], n: int) -> list[float]:
    """The fraction of problems with exactly j correct, j in 0..n."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not k_correct:
        raise ValueError("no problems to histogram")
    bins = [0.0] * (n + 1)
    for c in k_correct:
        if not 0 <= c <= n:
            raise ValueError(f"c={c} outside [0, {n}]")
        bins[c] += 1.0
    return [b / len(k_correct) for b in bins]


def report_ks(samples: int) -> list[int]:
    """1, 2, 4, 8, ... up to ``samples``."""
    ks, k = [], 1
    while k <= samples:
        ks.append(k)
        k *= 2
    return ks


def bootstrap_mean_ci(values: Sequence[float], *, seed: int,
                      resamples: int = BOOTSTRAP_RESAMPLES,
                      level: float = 0.95) -> tuple[float, float]:
    """A percentile interval on the mean, resampling problems with replacement.
    Seeded, so the same values always give the same interval."""
    if not values:
        raise ValueError("no values to bootstrap")
    rng = random.Random(seed)
    means = sorted(_resampled_mean(values, rng) for _ in range(resamples))
    tail = (1.0 - level) / 2.0

    def nearest_rank(q: float) -> float:
        return means[min(resamples - 1, max(0, math.ceil(round(q * resamples, 9)) - 1))]

    return nearest_rank(tail), nearest_rank(1.0 - tail)


def _resampled_mean(values: Sequence[float], rng: random.Random) -> float:
    n = len(values)
    return sum(values[rng.randrange(n)] for _ in range(n)) / n

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "bootstrap_mean_ci",
    "k_histogram",
    "pass_at_k",
    "pass_at_k_single",
    "report_ks",
]
