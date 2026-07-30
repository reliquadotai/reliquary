"""Exact robust valuation of one truncated rollout."""

from itertools import permutations, product

import pytest

from reliquary.validator.difficulty_auction import (
    difficulty_score,
    fractional_reward_lattice,
    gated_difficulty_utility,
    robust_truncation_utility,
)


def _oracle_utility(rewards, *, sigma_min, delta=1.0):
    values = tuple(float(reward) for reward in rewards)
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    sigma = (sum((reward - mean) ** 2 for reward in values) / len(values)) ** 0.5
    if sigma < 1e-8 or sigma < sigma_min:
        return 0.0
    return sigma * (1.0 - mean) ** delta


def _oracle_robust(
    rewards,
    *,
    truncated_index,
    attainable_rewards,
    sigma_min,
    delta=1.0,
):
    outcomes = []
    for attainable_reward in attainable_rewards:
        outcome = list(rewards)
        outcome[truncated_index] = attainable_reward
        outcomes.append(_oracle_utility(outcome, sigma_min=sigma_min, delta=delta))
    return min(outcomes)


def test_gated_utility_matches_validator_sigma_boundary():
    assert gated_difficulty_utility([0.5] * 8, sigma_min=0.0) == 0.0

    below = [1.0, 1.0] + [0.0] * 6
    sigma = difficulty_score(below).reward_std
    assert gated_difficulty_utility(below, sigma_min=sigma + 1e-12) == 0.0
    assert (
        gated_difficulty_utility(
            below,
            sigma_min=sigma,
        )
        == difficulty_score(below).value
    )


def test_fractional_reward_lattice_is_exact_and_not_binary_only():
    assert fractional_reward_lattice(1) == (0.0, 1.0)
    assert fractional_reward_lattice(4) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_fractional_interior_outcome_can_be_the_robust_minimum():
    rewards = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    sigma_min = 0.43

    endpoints_only = robust_truncation_utility(
        rewards,
        sigma_min=sigma_min,
        truncated_index=7,
        attainable_rewards=(0.0, 1.0),
    )
    exact_code = robust_truncation_utility(
        rewards,
        sigma_min=sigma_min,
        truncated_index=7,
        attainable_rewards=fractional_reward_lattice(2),
    )

    assert endpoints_only > 0.0
    assert exact_code == 0.0


def test_math_lattice_matches_exhaustive_oracle_for_all_binary_groups():
    lattice = fractional_reward_lattice(1)
    for rewards in product(lattice, repeat=8):
        for truncated_index in range(8):
            actual = robust_truncation_utility(
                rewards,
                sigma_min=0.43,
                truncated_index=truncated_index,
                attainable_rewards=lattice,
            )
            expected = _oracle_robust(
                rewards,
                sigma_min=0.43,
                truncated_index=truncated_index,
                attainable_rewards=lattice,
            )
            assert actual == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("total_tests", (2, 3, 4))
def test_code_lattices_match_exhaustive_oracle(total_tests):
    lattice = fractional_reward_lattice(total_tests)
    for rewards in product(lattice, repeat=4):
        for truncated_index in range(4):
            actual = robust_truncation_utility(
                rewards,
                sigma_min=0.2,
                truncated_index=truncated_index,
                attainable_rewards=lattice,
            )
            expected = _oracle_robust(
                rewards,
                sigma_min=0.2,
                truncated_index=truncated_index,
                attainable_rewards=lattice,
            )
            assert actual == pytest.approx(expected, abs=1e-15)


def test_robust_utility_never_exceeds_an_attainable_outcome():
    for total_tests in (2, 3, 5, 8):
        lattice = fractional_reward_lattice(total_tests)
        rewards = tuple(
            lattice[(index * 3 + total_tests) % len(lattice)] for index in range(8)
        )
        robust = robust_truncation_utility(
            rewards,
            sigma_min=0.17,
            truncated_index=3,
            attainable_rewards=lattice,
        )
        for attainable_reward in lattice:
            outcome = list(rewards)
            outcome[3] = attainable_reward
            assert robust <= gated_difficulty_utility(
                outcome,
                sigma_min=0.17,
            )


def test_permuting_rewards_and_unknown_identity_preserves_utility():
    rewards = (0.0, 0.25, 0.5, 0.75)
    lattice = fractional_reward_lattice(4)
    expected = robust_truncation_utility(
        rewards,
        sigma_min=0.2,
        truncated_index=1,
        attainable_rewards=lattice,
    )

    for order in permutations(range(len(rewards))):
        permuted = tuple(rewards[index] for index in order)
        permuted_index = order.index(1)
        assert robust_truncation_utility(
            permuted,
            sigma_min=0.2,
            truncated_index=permuted_index,
            attainable_rewards=lattice,
        ) == pytest.approx(expected, abs=1e-15)


def test_no_truncation_is_the_plain_gated_utility():
    rewards = [1.0, 1.0] + [0.0] * 6
    assert robust_truncation_utility(
        rewards,
        sigma_min=0.43,
    ) == gated_difficulty_utility(rewards, sigma_min=0.43)


@pytest.mark.parametrize("total_tests", (0, -1, 1.5, True))
def test_fractional_lattice_rejects_invalid_denominators(total_tests):
    with pytest.raises(ValueError):
        fractional_reward_lattice(total_tests)


@pytest.mark.parametrize("sigma_min", (-0.1, float("inf"), float("nan")))
def test_gated_utility_rejects_invalid_sigma_thresholds(sigma_min):
    with pytest.raises(ValueError):
        gated_difficulty_utility([0.0, 1.0], sigma_min=sigma_min)


@pytest.mark.parametrize("truncated_index", (-1, 2, 0.5, True))
def test_robust_utility_rejects_invalid_unknown_identity(truncated_index):
    with pytest.raises(ValueError):
        robust_truncation_utility(
            [0.0, 1.0],
            sigma_min=0.0,
            truncated_index=truncated_index,
            attainable_rewards=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    "attainable_rewards",
    ((), (-0.1, 1.0), (0.0, 1.1), (0.0, float("nan"))),
)
def test_robust_utility_rejects_invalid_lattices(attainable_rewards):
    with pytest.raises(ValueError):
        robust_truncation_utility(
            [0.0, 1.0],
            sigma_min=0.0,
            truncated_index=0,
            attainable_rewards=attainable_rewards,
        )


@pytest.mark.parametrize(
    "rewards",
    ([float("nan"), 0.0], [-0.1, 1.0], [0.0, 1.1]),
)
def test_robust_utility_does_not_hide_invalid_observed_rewards(rewards):
    with pytest.raises(ValueError):
        robust_truncation_utility(
            rewards,
            sigma_min=0.0,
            truncated_index=0,
            attainable_rewards=(0.0, 1.0),
        )
