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
        max_operator_share=1.0,
    )

    assert rewards["short"] == 0.1
    assert rewards["long"] == 0.9


def test_only_eos_terminated_tokens_count():
    """Callers pass EOS-terminated tokens only. A group with none earns
    nothing, which is what keeps padding a strictly negative margin."""
    rewards = split_environment_pool(
        [_g("terminated", 1_000), _g("padded", 0)],
        pool=1.0,
        max_operator_share=1.0,
    )

    assert rewards["terminated"] == 1.0
    assert rewards.get("padded", 0.0) == 0.0


def test_an_operator_over_the_cap_is_clipped_and_the_rest_reflows():
    """Bounded in TOKENS, not groups: under per-token payment a group count
    bounds nothing, since an operator can take few very long groups."""
    rewards = split_environment_pool(
        [
            _g("whale-a", 9_000, operator="whale"),
            _g("whale-b", 9_000, operator="whale"),
            _g("small", 2_000, operator="small"),
        ],
        pool=1.0,
        max_operator_share=0.5,
    )

    whale = rewards["whale-a"] + rewards["whale-b"]
    assert abs(whale - 0.5) < 1e-9
    assert abs(rewards["small"] - 0.5) < 1e-9
    assert abs(rewards["whale-a"] - rewards["whale-b"]) < 1e-9


def test_an_empty_environment_pays_nothing_rather_than_dividing_by_zero():
    assert split_environment_pool([], pool=1.0, max_operator_share=1.0) == {}


def test_reflow_can_push_a_second_operator_over_the_cap_and_the_loop_fixes_it():
    """A single reflow pass is not enough: clipping "A" (the operator raw-
    entitled to 60% of the pool) and handing its surplus to "B" and "C" in
    proportion to their own raw shares would, in one pass alone, leave B at
    0.45 -- itself over a 0.4 cap. Only a second pass (clip B, hand its own
    surplus back to whoever still has room) settles it. The correct fixed
    point is A = B = 0.4 (both pinned to the cap) and C absorbing the
    remainder.

    "spectator*" are many small never-capped operators; they exist only to
    give the reflow loop -- bounded by the operator count -- enough passes
    to converge to that fixed point to float precision, since convergence
    here is geometric rather than exact-in-one-more-step.
    """
    groups = [
        _g("A", 6_000),
        _g("B", 3_000),
        _g("C", 1_000),
    ] + [_g(f"spectator{i}", 1) for i in range(20)]

    rewards = split_environment_pool(groups, pool=1.0, max_operator_share=0.4)

    assert abs(rewards["A"] - 0.4) < 1e-4
    assert abs(rewards["B"] - 0.4) < 1e-4
    assert rewards["C"] < 0.4
    assert rewards["C"] > 0.19
    assert abs(sum(rewards.values()) - 1.0) < 1e-9
