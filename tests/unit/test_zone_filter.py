"""Zone filter: σ ≥ SIGMA_MIN (std-based, reward-scale-agnostic)."""

import math

from reliquary.validator.verifier import (
    is_in_zone,
    rewards_std,
)


def test_sigma_zero_rejected():
    """Degenerate std=0 is always rejected."""
    assert is_in_zone(0.0) is False


def test_sigma_below_min_rejected():
    """0.3 < 0.43 → rejected."""
    assert is_in_zone(0.3) is False


def test_sigma_at_min_accepted():
    """σ = 0.43 passes the steady-state gate."""
    assert is_in_zone(0.43) is True


def test_sigma_above_min_accepted():
    """σ = 0.5 passes the steady-state gate."""
    assert is_in_zone(0.5) is True


def test_bootstrap_threshold_lower():
    """0.35 is rejected in steady state but accepted in bootstrap."""
    assert is_in_zone(0.35, bootstrap=False) is False
    assert is_in_zone(0.35, bootstrap=True) is True


def test_bootstrap_still_rejects_zero_sigma():
    """Bootstrap mode doesn't save pathological zero-std groups."""
    assert is_in_zone(0.0, bootstrap=True) is False


def test_rewards_std_binary_matches_expected():
    """For binary rewards with k successes out of M=8, σ = √(p(1-p)) with p=k/M."""
    M = 8
    for k in range(M + 1):
        rewards = [1.0] * k + [0.0] * (M - k)
        p = k / M
        expected = math.sqrt(p * (1 - p))
        assert abs(rewards_std(rewards) - expected) < 1e-9, (
            f"k={k}: expected σ={expected:.6f}, got {rewards_std(rewards):.6f}"
        )


def test_rewards_std_empty_returns_zero():
    assert rewards_std([]) == 0.0


def test_rewards_std_single_returns_zero():
    assert rewards_std([1.0]) == 0.0


def test_rewards_std_continuous():
    """[0.7, 0.5, 0.3, 0.1] — population std = sqrt(variance)."""
    rewards = [0.7, 0.5, 0.3, 0.1]
    mean = sum(rewards) / len(rewards)                          # 0.4
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    expected = math.sqrt(variance)
    assert abs(rewards_std(rewards) - expected) < 1e-9


