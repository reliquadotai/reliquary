from reliquary.validator.reward_shape import detect_reward_shape_manipulation


def test_detects_repeated_truncated_zero_tail():
    metrics = detect_reward_shape_manipulation(
        [1, 1, 1, 1, 0, 0, 0, 0],
        [239, 212, 213, 232, 120, 120, 120, 120],
        [False, False, False, False, True, True, True, True],
    )

    assert metrics.suspicious is True
    assert metrics.reward_vector == "11110000"
    assert metrics.zero_length_mode == 120
    assert metrics.zero_mode_truncated_count == 4
    assert metrics.repeated_truncated_losers is True


def test_detects_exact_zero_tail_even_when_eos_terminated():
    metrics = detect_reward_shape_manipulation(
        [1, 1, 1, 1, 1, 0, 0, 0],
        [310, 298, 305, 288, 321, 150, 150, 150],
        [False] * 8,
    )

    assert metrics.suspicious is True
    assert metrics.reward_vector == "11111000"
    assert metrics.repeated_exact_losers is True


def test_ignores_varied_loser_lengths():
    metrics = detect_reward_shape_manipulation(
        [1, 1, 1, 1, 0, 0, 0, 0],
        [239, 212, 213, 232, 120, 133, 147, 161],
        [False, False, False, False, True, True, True, True],
    )

    assert metrics.suspicious is False
    assert metrics.zero_length_mode_count == 1


def test_ignores_mixed_reward_order():
    metrics = detect_reward_shape_manipulation(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [239, 120, 213, 120, 232, 120, 244, 120],
        [False, True, False, True, False, True, False, True],
    )

    assert metrics.suspicious is False
    assert metrics.ordered_prefix is False


def test_short_fixture_lengths_are_not_suspicious():
    metrics = detect_reward_shape_manipulation(
        [1, 1, 1, 1, 0, 0, 0, 0],
        [32, 32, 32, 32, 32, 32, 32, 32],
        [False] * 8,
    )

    assert metrics.suspicious is False
