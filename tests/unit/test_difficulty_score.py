"""Difficulty-auction score v(k).

The score answers "how much does the model learn from this group?" and, unlike
sigma, it is NOT symmetric: a group the model mostly FAILS is worth more than a
group it mostly PASSES, because GRPO's dominant per-sample advantage flips sign
across k=M/2. See docs/superpowers/specs/2026-07-14-difficulty-auction-design.md
"""

import pytest

from reliquary.validator.verifier import submission_value


def _binary(k: int, m: int = 8) -> list[float]:
    """A group with k correct rollouts out of m."""
    return [1.0] * k + [0.0] * (m - k)


def test_unanimous_groups_score_zero():
    """All-wrong and all-right carry no GRPO signal: every advantage cancels."""
    assert submission_value(_binary(0)) == 0.0
    assert submission_value(_binary(8)) == 0.0


def test_peaks_at_k2_with_default_delta():
    """delta=1.0 places the peak at 2-correct-of-8 — the exploration frontier."""
    scores = {k: submission_value(_binary(k)) for k in range(9)}
    assert max(scores, key=scores.__getitem__) == 2


def test_hard_group_outscores_easy_mirror():
    """THE point of the design: k=2 and k=6 have identical sigma, but k=2 is
    where GRPO amplifies a rare discovery instead of suppressing noise."""
    assert submission_value(_binary(2)) > submission_value(_binary(6))


def test_delta_zero_collapses_to_symmetric_sigma():
    """With no difficulty tilt the score degenerates to sigma — k=2 and k=6 tie,
    which is exactly the blindness the delta term exists to fix."""
    assert submission_value(_binary(2), delta=0.0) == pytest.approx(
        submission_value(_binary(6), delta=0.0)
    )


def test_larger_delta_tilts_harder():
    """delta is the difficulty dial: raising it widens the gap hard-over-easy."""
    gap_1 = submission_value(_binary(2)) / submission_value(_binary(6))
    gap_2 = (
        submission_value(_binary(2), delta=2.0)
        / submission_value(_binary(6), delta=2.0)
    )
    assert gap_2 > gap_1


def test_degenerate_groups_score_zero():
    assert submission_value([]) == 0.0
    assert submission_value([1.0]) == 0.0


def test_continuous_rewards_use_real_dispersion():
    """Code rewards need not be binary. A spread-out group still scores; a
    unanimous one still scores zero."""
    assert submission_value([0.5] * 8) == 0.0
    assert submission_value([0.9, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1]) > 0.0


def test_matches_the_published_curve():
    """Pin the exact values the design doc commits to (delta=1.0, M=8)."""
    expected = {1: 0.29, 2: 0.32, 3: 0.30, 4: 0.25, 5: 0.18, 6: 0.11, 7: 0.04}
    for k, want in expected.items():
        assert submission_value(_binary(k)) == pytest.approx(want, abs=0.005)
