"""Flat auction valuation: every in-zone group ranks at the same value.

With ``DIFFICULTY_AUCTION_FLAT_VALUE`` on, selection reduces to the sigma
gate plus the throughput/arrival tie-breaks: the batch k-distribution becomes
the natural corpus-times-model distribution (DAPO-style uniform + filter)
instead of concentrating on the k=2 value peak. It also removes the
manufactured-zero premium (a broken rollout can no longer inflate a prompt's
value) and the k=2.00 Sybil duel target.

The constant is read lazily inside ``difficulty_auction``, so tests patch
``reliquary.constants``.
"""

from types import SimpleNamespace

import reliquary.constants as C
from reliquary.validator.difficulty_auction import (
    auction_difficulty_score,
    auction_value,
    flat_auction_value,
)


def _k(k: int) -> list[float]:
    """Binary M=8 reward vector with k correct rollouts."""
    return [1.0] * k + [0.0] * (8 - k)


def test_default_off_preserves_difficulty_ranking():
    assert auction_value(_k(2)) > auction_value(_k(4)) > auction_value(_k(6))


def test_flat_collapses_all_nondegenerate_k_to_equal_value(monkeypatch):
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    values = {k: auction_value(_k(k)) for k in range(1, 8)}
    assert all(v == 1.0 for v in values.values()), values


def test_flat_keeps_zero_for_degenerate_groups(monkeypatch):
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    assert auction_value(_k(0)) == 0.0
    assert auction_value(_k(8)) == 0.0


def test_flat_removes_manufactured_zero_premium(monkeypatch):
    # Breaking one correct rollout (k=6 -> k=5) must gain exactly nothing.
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    assert auction_value(_k(6)) == auction_value(_k(5))


def test_flat_preserves_mean_and_std_telemetry(monkeypatch):
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    score = auction_difficulty_score(_k(2))
    assert score.value == 1.0
    assert abs(score.mean_reward - 0.25) < 1e-9
    assert score.reward_std > 0.4


def test_flat_helper_collapses_positive_only(monkeypatch):
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    assert flat_auction_value(0.1234) == 1.0
    assert flat_auction_value(0.0) == 0.0
    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", False)
    assert flat_auction_value(0.1234) == 0.1234


def test_pending_ranking_flat_with_robust_utility_override(monkeypatch):
    # The conservative-truncation robust_utility bypasses auction_value, so
    # the ranking adapter must flatten it too.
    from reliquary.validator.batcher import _pending_difficulty_score

    monkeypatch.setattr(C, "DIFFICULTY_AUCTION_FLAT_VALUE", True)
    pending = SimpleNamespace(
        rewards=_k(6), truncated_count=1, robust_utility=0.0421,
    )
    assert _pending_difficulty_score(pending).value == 1.0
