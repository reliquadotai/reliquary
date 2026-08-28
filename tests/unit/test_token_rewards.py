"""Emission is divided by EOS-terminated completion tokens, not by slot."""
from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool


def _g(hotkey, tokens, operator=None):
    return AcceptedGroup(
        hotkey=hotkey, operator_id=operator or hotkey, eos_tokens=tokens
    )


def test_the_pool_is_divided_in_proportion_to_tokens():
    """Under a flat per-slot share, revenue per GPU-second is proportional to
    1/L: halving response length doubles income. Dividing by tokens makes it
    independent of length, so the policy decides how long to reason."""
    rewards = split_environment_pool(
        [_g("short", 1_000), _g("long", 9_000)],
        pool=1.0,
    )

    assert rewards["short"] == 0.1
    assert rewards["long"] == 0.9


def test_only_eos_terminated_tokens_count():
    """Callers pass EOS-terminated tokens only. A group with none earns
    nothing, which is what keeps padding a strictly negative margin."""
    rewards = split_environment_pool(
        [_g("terminated", 1_000), _g("padded", 0)],
        pool=1.0,
    )

    assert rewards["terminated"] == 1.0
    assert rewards.get("padded", 0.0) == 0.0


def test_an_empty_environment_pays_nothing_rather_than_dividing_by_zero():
    assert split_environment_pool([], pool=1.0) == {}


def test_rewards_sum_to_the_pool_across_mixed_eos_and_padded_groups():
    """No cap, no reflow loop: the split is a plain proportion, so summing
    to ``pool`` holds trivially rather than needing a convergence loop to
    get there. Pinned so a future change cannot silently leak emission."""
    groups = [
        _g("a", 4_000),
        _g("b", 3_000),
        _g("c", 0),  # padded, contributes nothing and takes nothing
        _g("d", 1_000),
    ]

    rewards = split_environment_pool(groups, pool=1.0)

    assert abs(sum(rewards.values()) - 1.0) < 1e-12
    assert "c" not in rewards


def test_two_hotkeys_under_the_same_operator_are_paid_independently():
    """No per-operator cap means no operator-level pooling either: each
    hotkey is paid strictly by its own tokens, regardless of how many other
    hotkeys share its ``operator_id``. This is the property that replaced
    the cap -- a shared operator can no longer buy either hotkey a better
    rate than its own tokens earn."""
    rewards = split_environment_pool(
        [
            _g("whale-a", 9_000, operator="whale"),
            _g("whale-b", 1_000, operator="whale"),
            _g("solo", 10_000, operator="solo"),
        ],
        pool=1.0,
    )

    assert abs(rewards["whale-a"] - 0.45) < 1e-9
    assert abs(rewards["whale-b"] - 0.05) < 1e-9
    assert abs(rewards["solo"] - 0.5) < 1e-9
