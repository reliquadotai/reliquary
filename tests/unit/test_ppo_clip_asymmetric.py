"""Clip-Higher: the surrogate clamps with separate low and high epsilons.

DAPO §3.1 — the *upper* clip is what lets low-probability tokens grow. At a
symmetric 0.2 a token at p=0.01 can reach at most 0.012 while one at p=0.9
reaches 1.08, so exploitation tokens grow freely and exploration tokens cannot;
the paper's Figure 2b shows entropy collapsing to ~0 without the fix.

The band is only interpretable as a true ratio bound because v4 sampling makes
warp() the identity (see test_forced_sampling.py). Under v3 the trainer's ratio
lives in a different space from the one sampled, by a factor that varies token
by token.

Both surrogate paths — per-rollout and micro-batched — go through the single
helper tested here, so the band cannot diverge between them.

PPO_CLIP_EPSILON_LOW/HIGH are module-top imports in training.py, so these tests
patch the consuming module rather than reliquary.constants.
"""
import torch

import reliquary.validator.training as T


def test_defaults_are_symmetric_on_pre_v4_profiles():
    """The test-default profile is v2: behaviour must be byte-identical to the
    single-epsilon band it replaces."""
    assert T.PPO_CLIP_EPSILON_LOW == 0.2
    assert T.PPO_CLIP_EPSILON_HIGH == 0.2


def test_clipped_surrogate_clamps_the_upside_at_epsilon_high(monkeypatch):
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_LOW", 0.2)
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.28)

    surr, clip_active = T._clipped_surrogate(
        torch.tensor([1.5]), torch.tensor([1.0])
    )

    assert surr.item() == torch.tensor(1.28).item()
    assert bool(clip_active.item()) is True


def test_clipped_surrogate_clamps_the_downside_at_epsilon_low(monkeypatch):
    """A negative advantage is bounded by the LOW side, which stays at 0.2 —
    raising only the ceiling is the whole point of the asymmetry."""
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_LOW", 0.2)
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.28)

    surr, clip_active = T._clipped_surrogate(
        torch.tensor([0.5]), torch.tensor([-1.0])
    )

    assert surr.item() == torch.tensor(-0.8).item()
    assert bool(clip_active.item()) is True


def test_clipped_surrogate_leaves_ratios_inside_the_band_untouched(monkeypatch):
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_LOW", 0.2)
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.28)

    surr, clip_active = T._clipped_surrogate(
        torch.tensor([1.1]), torch.tensor([1.0])
    )

    assert surr.item() == torch.tensor(1.1).item()
    assert bool(clip_active.item()) is False


def test_a_ratio_between_the_two_epsilons_is_clipped_only_when_symmetric(monkeypatch):
    """1.24 sits above 1+0.2 but below 1+0.28: it is exactly the exploration
    headroom Clip-Higher buys, so the two settings must disagree here."""
    adv = torch.tensor([1.0])
    ratio = torch.tensor([1.24])

    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_LOW", 0.2)
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.2)
    symmetric, symmetric_clipped = T._clipped_surrogate(ratio, adv)

    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.28)
    higher, higher_clipped = T._clipped_surrogate(ratio, adv)

    assert bool(symmetric_clipped.item()) is True
    assert symmetric.item() == torch.tensor(1.2).item()
    assert bool(higher_clipped.item()) is False
    assert higher.item() == torch.tensor(1.24).item()
