"""Characterize the token accounting retained for fill qualification."""
from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool


def _g(hotkey, tokens, operator=None):
    return AcceptedGroup(
        hotkey=hotkey, operator_id=operator or hotkey, eos_tokens=tokens
    )


def test_the_pool_is_divided_in_proportion_to_tokens():
    rewards = split_environment_pool(
        [_g("short", 1_000), _g("long", 9_000)],
        pool=1.0,
    )

    assert rewards["short"] == 0.1
    assert rewards["long"] == 0.9


def test_only_eos_terminated_tokens_count():
    """A group with no counted tokens receives no allocation."""
    rewards = split_environment_pool(
        [_g("terminated", 1_000), _g("padded", 0)],
        pool=1.0,
    )

    assert rewards["terminated"] == 1.0
    assert rewards.get("padded", 0.0) == 0.0


def test_an_empty_environment_pays_nothing_rather_than_dividing_by_zero():
    assert split_environment_pool([], pool=1.0) == {}


def test_rewards_sum_to_the_pool_across_mixed_eos_and_padded_groups():
    """The deterministic split conserves the configured pool."""
    groups = [
        _g("a", 4_000),
        _g("b", 3_000),
        _g("c", 0),
        _g("d", 1_000),
    ]

    rewards = split_environment_pool(groups, pool=1.0)

    assert abs(sum(rewards.values()) - 1.0) < 1e-12
    assert "c" not in rewards


def test_two_hotkeys_under_the_same_operator_are_paid_independently():
    """This isolated policy does not aggregate by operator identity."""
    rewards = split_environment_pool(
        [
            _g("operator-a-1", 9_000, operator="operator-a"),
            _g("operator-a-2", 1_000, operator="operator-a"),
            _g("operator-b-1", 10_000, operator="operator-b"),
        ],
        pool=1.0,
    )

    assert abs(rewards["operator-a-1"] - 0.45) < 1e-9
    assert abs(rewards["operator-a-2"] - 0.05) < 1e-9
    assert abs(rewards["operator-b-1"] - 0.5) < 1e-9
