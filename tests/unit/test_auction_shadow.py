"""Shadow wiring: score every submission, and compute the batch the difficulty
auction WOULD have selected — without changing what production actually does.

Arming the auction is blocked on the free-negative measurement (M5): paying the
most for a low-k group also pays the most for a FALSE negative, and the #1 math
earner already farms those. So the auction ships dark first: we archive the batch
it would have picked next to the batch drand-FCFS really picked, and compare.
See docs/superpowers/specs/2026-07-14-difficulty-auction-design.md §9-10
"""

from reliquary.validator.batcher import ValidSubmission


class FakeRollout:
    def __init__(self, reward: float):
        self.reward = reward


def _submission(hotkey: str, prompt_idx: int, k: int, m: int = 8):
    """A submission whose group has k correct rollouts out of m."""
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        merkle_root_bytes=hotkey.encode().ljust(32, b"\x00"),
        rollouts=[FakeRollout(1.0)] * k + [FakeRollout(0.0)] * (m - k),
    )


def test_value_is_scored_from_the_groups_rewards():
    """k=2 of 8 sits at the peak of the curve the design commits to."""
    assert _submission("a", 1, k=2).value > 0.31
    assert _submission("a", 1, k=2).value < 0.33


def test_hard_group_scores_above_its_easy_mirror():
    """The whole point: identical sigma, opposite pedagogical value."""
    assert _submission("a", 1, k=2).value > _submission("b", 2, k=6).value


def test_unanimous_group_scores_zero():
    assert _submission("a", 1, k=0).value == 0.0
    assert _submission("a", 1, k=8).value == 0.0


def test_submission_without_rollouts_scores_zero():
    """Defensive: the archive replay path constructs bare submissions."""
    assert ValidSubmission(
        hotkey="a", prompt_idx=1, merkle_root_bytes=b"\x00" * 32,
    ).value == 0.0
