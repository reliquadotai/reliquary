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
    """σ = 0.43 (k=2/6 of M=8) passes the steady-state gate."""
    assert is_in_zone(0.43) is True


def test_sigma_above_min_accepted():
    """σ = 0.5 passes the steady-state gate."""
    assert is_in_zone(0.5) is True


def test_bootstrap_threshold_lower():
    """0.38 is rejected in steady state (<0.43) but accepted in bootstrap (≥0.33)."""
    assert is_in_zone(0.38, bootstrap=False) is False
    assert is_in_zone(0.38, bootstrap=True) is True


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




# ── v4: DAPO dynamic-sampling intent — admit any binary variance group (k=1..15
# at M=16) via a 0.24 floor, still filtering near-degenerate continuous clusters.

def test_v4_admits_k1_and_k15_of_16(monkeypatch):
    """For M=16 binary the extremes k=1 and k=15 have σ=√(1/16·15/16)=0.2421 —
    rejected by the 0.43 gate, admitted by the v4 0.24 floor."""
    import reliquary.constants as C
    monkeypatch.setattr(C, "SIGMA_MIN", 0.24)
    monkeypatch.setattr(C, "BOOTSTRAP_SIGMA_MIN", 0.22)

    sigma_k1 = math.sqrt((1 / 16) * (15 / 16))
    assert round(sigma_k1, 4) == 0.2421
    assert is_in_zone(sigma_k1) is True            # k=1 (and by symmetry k=15)
    assert is_in_zone(0.5) is True                 # k=8, always fine


def test_v4_still_filters_near_degenerate_continuous_clusters(monkeypatch):
    """0.24, not 0.0: a tight continuous (code) cluster with tiny σ carries no
    GRPO gradient and must still be rejected — this is why the floor isn't 0."""
    import reliquary.constants as C
    monkeypatch.setattr(C, "SIGMA_MIN", 0.24)
    monkeypatch.setattr(C, "BOOTSTRAP_SIGMA_MIN", 0.22)

    assert is_in_zone(0.0) is False                # all-same
    assert is_in_zone(0.1) is False                # tight cluster, below 0.24
    assert is_in_zone(0.1, bootstrap=True) is False  # 0.1 < 0.22 too
