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
    """0.3 < 0.33 → rejected."""
    assert is_in_zone(0.3) is False


def test_sigma_at_min_accepted():
    """σ = 0.33 passes the steady-state gate."""
    assert is_in_zone(0.33) is True


def test_sigma_above_min_accepted():
    """σ = 0.5 passes the steady-state gate."""
    assert is_in_zone(0.5) is True


def test_bootstrap_threshold_lower():
    """0.28 is rejected in steady state (<0.33) but accepted in bootstrap (≥0.25)."""
    assert is_in_zone(0.28, bootstrap=False) is False
    assert is_in_zone(0.28, bootstrap=True) is True


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


def test_truncation_penalized_reward():
    """Max-tokens penalty: subtract `penalty` from a cap-hit, cut-off (no EOS)
    rollout (correct → keeps most credit, wrong → drops below 0); terminated
    rollouts keep their reward."""
    from reliquary.shared.modeling import truncation_penalized_reward as tpr
    cap, P, eos = 100, 0.2, {99}
    def approx(a, b): return abs(a - b) < 1e-9
    # terminated correct, well under cap → keeps base
    assert tpr(1.0, [1, 2, 99], 1, 2, eos, penalty=P, cap=cap) == 1.0
    # used the full budget but ended on a real EOS → not penalized (penalize being
    # cut off, not thinking long)
    assert tpr(1.0, [1] + [2] * 98 + [99], 1, 99, eos, penalty=P, cap=cap) == 1.0
    # cap-hit, cut off, correct box mid-stream → 1.0 - 0.2 = 0.8 (keeps most credit)
    assert approx(tpr(1.0, [1] + [2] * 99, 1, 99, eos, penalty=P, cap=cap), 0.8)
    # cap-hit, cut off, wrong → 0.0 - 0.2 = -0.2 (below finished-wrong)
    assert approx(tpr(0.0, [1] + [2] * 99, 1, 99, eos, penalty=P, cap=cap), -0.2)
    # short, no EOS, under the cap → not this helper's concern
    assert tpr(0.0, [1, 2, 3], 1, 2, eos, penalty=P, cap=cap) == 0.0


