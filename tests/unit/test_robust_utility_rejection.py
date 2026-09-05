"""A group whose least-favourable reading leaves the zone is refused."""
from reliquary.validator.admission import robust_utility_admits


def test_a_manufactured_zero_is_refused_not_priced():
    """All 16 rollouts correct is out of zone and worthless. Break one by
    suppressing EOS and the observed vector looks in-zone -- but the truncated
    rollout may have been correct, and that reading is out of zone again."""
    rewards = [1.0] * 15 + [0.0]

    assert robust_utility_admits(
        rewards,
        sigma_min=0.24,
        truncated_indices=(15,),
        attainable_rewards=(0.0, 1.0),
    ) is False


def test_an_honest_in_zone_group_is_admitted():
    rewards = [1.0] * 8 + [0.0] * 8

    assert robust_utility_admits(
        rewards,
        sigma_min=0.24,
        truncated_indices=(),
        attainable_rewards=(0.0, 1.0),
    ) is True
