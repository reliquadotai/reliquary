from dataclasses import dataclass, field

import pytest

from reliquary.validator.difficulty_auction import (
    difficulty_score,
    select_shadow_auction,
)


@dataclass
class FakeRollout:
    reward: float


@dataclass
class FakeSubmission:
    hotkey: str
    prompt_idx: int
    drand_round: int
    rewards: list[float]
    selection_digest: bytes = field(default_factory=lambda: b"x" * 32)
    merkle_root: bytes = field(default_factory=lambda: b"x" * 32)
    in_cooldown: bool = False

    @property
    def source_id(self):
        return id(self)


def _select(submissions, *, b=8, **kwargs):
    return select_shadow_auction(
        submissions,
        b=b,
        delta=1.0,
        **kwargs,
    )


def test_binary_curve_peaks_at_two_of_eight():
    scores = {
        k: difficulty_score([1.0] * k + [0.0] * (8 - k)).value
        for k in range(9)
    }

    assert max(scores, key=scores.get) == 2
    assert scores[2] > scores[6]
    assert scores[0] == scores[8] == 0.0


@pytest.mark.parametrize(
    "rewards",
    ([float("nan"), 0.0], [-0.1, 1.0], [0.0, 1.1]),
)
def test_invalid_reward_domain_is_not_silently_ranked(rewards):
    with pytest.raises(ValueError):
        difficulty_score(rewards)


def test_harder_candidate_beats_faster_easy_candidate():
    fast_easy = FakeSubmission("fast", 1, 1, [1.0] * 6 + [0.0] * 2)
    slow_hard = FakeSubmission("slow", 2, 9, [1.0] * 2 + [0.0] * 6)

    result = _select([fast_easy, slow_hard], b=1)

    assert result.selected == (slow_hard,)


def test_same_prompt_is_resolved_at_selection_not_admission():
    weak = FakeSubmission("weak", 7, 1, [1.0] * 6 + [0.0] * 2)
    strong = FakeSubmission("strong", 7, 2, [1.0] * 2 + [0.0] * 6)

    result = _select([weak, strong])

    assert result.selected == (strong,)
    assert result.distinct_prompt_count == 1
    assert len(result.candidates) == 2


def test_requested_operator_cap_is_disabled_when_mapping_is_incomplete():
    submissions = [
        FakeSubmission(f"hk{i}", i, 1, [1.0] * 2 + [0.0] * 6)
        for i in range(3)
    ]

    result = _select(
        submissions,
        b=3,
        max_slots_per_operator=1,
        operator_of=lambda hotkey: "owner" if hotkey != "hk2" else None,
    )

    assert result.operator_cap_requested == 1
    assert result.operator_mapping_complete is False
    assert result.operator_cap_applied is False
    assert len(result.selected) == 3


def test_complete_operator_mapping_applies_the_counterfactual_cap():
    submissions = [
        FakeSubmission(f"hk{i}", i, 1, [1.0] * 2 + [0.0] * 6)
        for i in range(3)
    ]

    result = _select(
        submissions,
        b=3,
        max_slots_per_operator=1,
        operator_of=lambda _hotkey: "owner",
    )

    assert result.operator_mapping_complete is True
    assert result.operator_cap_applied is True
    assert len(result.selected) == 1


def test_rank_key_breaks_score_ties_by_arrival_round():
    from reliquary.validator.difficulty_auction import (
        ShadowSubmission, _rank_key, difficulty_score,
    )

    def _sub(source_id, hotkey, arrival):
        return ShadowSubmission(
            source_id=source_id, hotkey=hotkey, prompt_idx=source_id,
            drand_round=999, merkle_root=b"\x00" * 32,
            selection_digest=hotkey.encode().ljust(32, b"\x00"),
            rewards=(1.0, 1.0) + (0.0,) * 6,
            arrival_drand_round=arrival,
        )

    slow = _sub(1, "slow", 105)
    fast = _sub(2, "fast", 103)
    ranked = sorted(
        ((s, difficulty_score(s.rewards, delta=1.0)) for s in (slow, fast)),
        key=_rank_key,
    )
    assert [s.hotkey for s, _ in ranked] == ["fast", "slow"]


def test_max_difficulty_value_is_the_binary_k2_peak_for_8_rollouts():
    """delta=1: v(p)=sqrt(p(1-p))*(1-p) peaks at p=1/4 -> k=2 of 8; the
    constant must be the exact float difficulty_score emits for that profile
    (ranking compares with ==, no epsilon)."""
    from reliquary.validator.difficulty_auction import (
        difficulty_score, max_difficulty_value,
    )

    expected = difficulty_score([1.0, 1.0] + [0.0] * 6, delta=1.0).value
    assert max_difficulty_value(8, delta=1.0) == expected
    assert all(
        difficulty_score([1.0] * k + [0.0] * (8 - k), delta=1.0).value
        <= max_difficulty_value(8, delta=1.0)
        for k in range(9)
    )


def test_no_fractional_profile_exceeds_the_binary_maximum():
    """For fixed mean, std is maximized only by extremal (0/1) rewards, so no
    in-[0,1] profile can beat the binary max. Grid-check 4-rollout profiles
    exhaustively on a 0/0.25/0.5/0.75/1 lattice."""
    from itertools import product

    from reliquary.validator.difficulty_auction import (
        difficulty_score, max_difficulty_value,
    )

    cap = max_difficulty_value(4, delta=1.0)
    lattice = (0.0, 0.25, 0.5, 0.75, 1.0)
    for profile in product(lattice, repeat=4):
        assert difficulty_score(list(profile), delta=1.0).value <= cap


def test_max_difficulty_value_zero_and_one_rollout_degenerate_to_zero():
    from reliquary.validator.difficulty_auction import max_difficulty_value

    assert max_difficulty_value(0, delta=1.0) == 0.0
    assert max_difficulty_value(1, delta=1.0) == 0.0
