"""The report's arithmetic: the unbiased pass@k, the k-histogram and the
bootstrap interval. Ported from reliquary-compare `src/eval` with its tests."""

from __future__ import annotations

import itertools
from fractions import Fraction
from math import comb

import pytest

from reliquary.eval.metrics import (
    bootstrap_mean_ci,
    k_histogram,
    pass_at_k,
    pass_at_k_single,
    report_ks,
)


@pytest.mark.parametrize("c,n,k,expected", [
    (0, 16, 8, 0.0),
    (16, 16, 8, 1.0),
    (1, 16, 8, 0.5),     # 1 - C(15,8)/C(16,8)
    (8, 16, 1, 0.5),
    (4, 16, 2, 0.45),    # 1 - C(12,2)/C(16,2)
])
def test_pass_at_k_exact(c, n, k, expected):
    assert pass_at_k_single(c, n, k) == pytest.approx(expected, abs=1e-12)


def _brute_force(c: int, n: int, k: int) -> Fraction:
    """P(at least one correct in a k-subset drawn without replacement),
    by enumerating every subset of a vector with c ones."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for subset in subsets if any(samples[i] for i in subset))
    return Fraction(hits, len(subsets))


@pytest.mark.parametrize("n", range(1, 9))
def test_pass_at_k_equals_brute_force_on_small_cases(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k_single(c, n, k) == pytest.approx(float(_brute_force(c, n, k)),
                                                              abs=1e-12)
            # And the closed form the design names.
            assert pass_at_k_single(c, n, k) == pytest.approx(
                1 - comb(n - c, k) / comb(n, k), abs=1e-12)


@pytest.mark.parametrize("c", range(17))
def test_pass_at_1_is_the_mean(c):
    assert pass_at_k_single(c, 16, 1) == pytest.approx(c / 16, abs=1e-12)


def test_pass_at_k_is_monotone_and_bounded():
    values = [pass_at_k_single(5, 16, k) for k in range(1, 17)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    assert all(0.0 <= v <= 1.0 for v in values)


def test_no_overflow_at_large_n():
    assert 0.0 <= pass_at_k_single(1, 1024, 512) <= 1.0


def test_pass_at_k_refuses_out_of_range():
    with pytest.raises(ValueError):
        pass_at_k_single(1, 4, 5)
    with pytest.raises(ValueError):
        pass_at_k_single(5, 4, 1)


def test_suite_pass_at_k_is_the_mean_of_problems_with_their_own_n():
    # Problem n differs (a grader error drops a sample): each uses its own n.
    assert pass_at_k([(0, 16), (8, 16), (16, 16)], 1) == pytest.approx(0.5)
    assert pass_at_k([(1, 2), (2, 3)], 2) == pytest.approx((1.0 + 1.0) / 2)
    with pytest.raises(ValueError):
        pass_at_k([], 1)


def test_histogram_bins():
    h = k_histogram([0, 4, 8, 16, 8], 16)
    assert len(h) == 17 and sum(h) == pytest.approx(1.0)
    assert h[8] == pytest.approx(2 / 5)
    with pytest.raises(ValueError):
        k_histogram([], 16)


def test_report_ks_are_powers_of_two_up_to_samples():
    assert report_ks(1) == [1]
    assert report_ks(8) == [1, 2, 4, 8]
    assert report_ks(12) == [1, 2, 4, 8]


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = [0.0, 0.25, 0.5, 1.0, 1.0, 0.75, 0.0, 0.5]
    first = bootstrap_mean_ci(values, seed=7, resamples=500)
    assert first == bootstrap_mean_ci(values, seed=7, resamples=500)
    low, high = first
    assert low < sum(values) / len(values) < high
    assert bootstrap_mean_ci([0.5], seed=1) == (0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([], seed=1)


def test_bootstrap_takes_the_nearest_rank_percentiles(monkeypatch):
    # With R resamples the q-th percentile is the ceil(q*R)-th smallest mean.
    from reliquary.eval import metrics

    means = iter(range(2000))

    class Rng:
        def __init__(self, seed):
            pass

        def randrange(self, n):
            return 0

    monkeypatch.setattr(metrics.random, "Random", Rng)
    monkeypatch.setattr(metrics, "_resampled_mean", lambda values, rng: next(means))
    low, high = metrics.bootstrap_mean_ci([0.0, 1.0], seed=1, resamples=2000)
    assert (low, high) == (49, 1949)
