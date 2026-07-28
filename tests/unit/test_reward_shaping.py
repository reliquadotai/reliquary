from types import SimpleNamespace

import pytest

from reliquary import constants as C
from reliquary.validator.training import (
    _compute_advantages,
    _plan_from_batches,
    _shape_advantages,
    _shaping_training_metrics,
    _training_environment_metrics,
)


@pytest.fixture
def shaping_enabled(monkeypatch):
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.5)


def _roll(
    reward,
    completion_length,
    *,
    env_name="openmathinstruct",
    forced=False,
    truncated=False,
):
    return SimpleNamespace(
        reward=reward,
        env_name=env_name,
        commit={"rollout": {
            "prompt_length": 1,
            "completion_length": completion_length,
            "token_logprobs": [-1.0] * completion_length,
            "forced": forced,
            "truncated": truncated,
        }},
    )


def test_shaping_penalizes_under_thinking_only(shaping_enabled):
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    rollouts = [
        _roll(0.0, early),                    # finished-early + wrong → penalize
        _roll(1.0, early),                    # finished-early + correct → keep
        _roll(0.0, C.BFT_THINKING_BUDGET),    # long + wrong (tried hard) → keep
    ]
    out = _shape_advantages(rollouts, [0.3, 0.3, 0.3])
    assert out[0] == -C.SHAPE_PENALTY
    assert out[1] == 0.3
    assert out[2] == 0.3


def test_shaping_masks_forced_instead_of_penalising_it(shaping_enabled):
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    # forced + finished-early + wrong: never length-penalised (its length is a
    # budget artefact) — it is masked out of the loss entirely instead.
    out = _shape_advantages([_roll(0.0, early, forced=True)], [0.5])
    assert out[0] == 0.0


def test_shaping_penalizes_truncated_overlong(shaping_enabled):
    # overlong side penalises a cap-truncated rollout regardless of correctness
    out = _shape_advantages([_roll(1.0, C.BFT_THINKING_BUDGET, truncated=True)], [0.4])
    assert out[0] == -C.SHAPE_PENALTY


def test_mask_forced_zeroes_gradient_even_with_shaping_off(monkeypatch):
    # DAPO overlong filtering (now unconditional): a forced rollout's advantage
    # is masked to 0, independent of SHAPE_PENALTY; others are left alone.
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.0)
    out = _shape_advantages(
        [_roll(1.0, 100, forced=True), _roll(0.0, 100)], [0.7, -0.7]
    )
    assert out[0] == 0.0        # forced → masked from the gradient
    assert out[1] == -0.7       # non-forced untouched (shaping off)


def test_mask_forced_keeps_baseline_over_full_group(monkeypatch):
    # The forced 0 stays in the group mean/std (baseline unchanged, per DAPO Eq.9);
    # only its own gradient is masked. The correct rollouts keep the advantages
    # that were computed WITH the forced sample in the mean.
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.0)
    adv = _compute_advantages([1.0, 1.0, 0.0])          # mean over all 3
    rollouts = [_roll(1.0, 100), _roll(1.0, 100), _roll(0.0, 100, forced=True)]
    out = _shape_advantages(rollouts, adv)
    assert out[0] == adv[0] and out[1] == adv[1]        # correct advantages intact
    assert out[2] == 0.0                                 # forced masked


def test_natural_zero_advantage_is_not_treated_as_masked(monkeypatch):
    """REGRESSION: a legitimate advantage is exactly 0 whenever a reward equals
    its group mean (common with fractional code rewards). Such a rollout must NOT
    be mistaken for a masked one — it still owes its KL term and its tokens to
    N_e. Masking is decided by the rollout's metadata, never by the value 0.0."""
    from reliquary.validator.training import _is_masked_from_loss

    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.0)
    rewards = [1.0, 0.5, 0.0]
    advantages = _compute_advantages(rewards)
    assert advantages[1] == 0.0                      # exactly zero, legitimately
    middle = _roll(0.5, 100)                         # not forced
    assert _is_masked_from_loss(middle) is False     # so it is NOT skipped
    assert _is_masked_from_loss(_roll(0.0, 100, forced=True)) is True


def test_mask_forced_composes_with_length_shaping(monkeypatch):
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.5)
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    rollouts = [
        _roll(0.0, 100, forced=True),                        # forced → masked 0
        _roll(1.0, C.BFT_THINKING_BUDGET, truncated=True),   # overlong → −penalty
        _roll(0.0, early),                                   # under-thinking → −penalty
    ]
    out = _shape_advantages(rollouts, [0.3, 0.3, 0.3])
    assert out[0] == 0.0
    assert out[1] == -C.SHAPE_PENALTY
    assert out[2] == -C.SHAPE_PENALTY


def test_shaping_off_when_penalty_zero(monkeypatch):
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.0)
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    out = _shape_advantages([_roll(0.0, early)], [0.3])
    assert out == [0.3]


def test_plan_keeps_all_wrong_group_when_shape_adds_signal(shaping_enabled):
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    group = SimpleNamespace(
        rollouts=[
            _roll(0.0, early),
            _roll(0.0, C.BFT_THINKING_BUDGET),
        ],
        prompt_idx=0,
    )

    plan, n_skipped = _plan_from_batches([[group]])

    assert n_skipped == 0
    assert len(plan) == 1
    _group, advantages, _scale = plan[0]
    assert advantages[0] == -C.SHAPE_PENALTY
    assert advantages[1] == 0.0


def test_shaping_metrics_separate_overlong_underthinking_and_forced(
    shaping_enabled,
):
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    group = SimpleNamespace(
        rollouts=[
            _roll(0.0, C.BFT_THINKING_BUDGET, truncated=True),
            _roll(0.0, early),
            _roll(0.0, early, forced=True),
            _roll(1.0, C.BFT_THINKING_BUDGET),
        ],
        prompt_idx=0,
    )
    raw = _compute_advantages([rollout.reward for rollout in group.rollouts])
    shaped = _shape_advantages(group.rollouts, raw)

    metrics = _shaping_training_metrics([(group, shaped, 1.0)])

    assert metrics["train/shaping_overlong_ratio"] == 0.25
    assert metrics["train/shaping_underthinking_ratio"] == 0.25
    assert metrics["train/shaping_forced_exempt_ratio"] == 0.25
    assert metrics["train/shaping_changed_ratio"] == 0.75   # forced now masked too


def test_training_environment_metrics_separate_domains_and_plan_signal():
    math_group = SimpleNamespace(
        rollouts=[_roll(1.0, 4), _roll(0.0, 4)],
        prompt_idx=0,
    )
    code_group = SimpleNamespace(
        rollouts=[
            _roll(0.75, 6, env_name="opencodeinstruct"),
            _roll(0.75, 6, env_name="opencodeinstruct"),
        ],
        prompt_idx=1,
    )
    batches = [[math_group], [code_group]]
    plan, n_skipped = _plan_from_batches(batches)

    metrics = _training_environment_metrics(batches, plan)

    assert n_skipped == 1
    assert metrics["train/env/openmathinstruct/reward_mean"] == 0.5
    assert metrics["train/env/openmathinstruct/reward_std"] == 0.5
    assert metrics["train/env/openmathinstruct/reward_nonzero_ratio"] == 0.5
    assert metrics["train/env/openmathinstruct/plan_groups"] == 1.0
    assert metrics["train/env/openmathinstruct/plan_rollouts"] == 2.0
    assert metrics["train/env/opencodeinstruct/reward_mean"] == 0.75
    assert metrics["train/env/opencodeinstruct/raw_completion_tokens"] == 12.0
    assert metrics["train/env/opencodeinstruct/plan_groups"] == 0.0
    assert metrics["train/env/opencodeinstruct/plan_rollouts"] == 0.0
