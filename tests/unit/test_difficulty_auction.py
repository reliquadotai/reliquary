from dataclasses import dataclass, field

import pytest

from reliquary.validator.cooldown import CooldownMap
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

    @property
    def rollouts(self):
        return [FakeRollout(reward) for reward in self.rewards]


def _select(submissions, *, b=8, **kwargs):
    return select_shadow_auction(
        submissions,
        b=b,
        cooldown_map=CooldownMap(cooldown_windows=0),
        current_window=100,
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
