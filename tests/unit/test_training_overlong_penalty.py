"""Soft overlong punishment (DAPO Eq. 13) applied to TRAINING rewards.

The invariant every test guards: ``rollout.reward`` — the graded reward that
drives the sigma gate, auction and emission — is never modified. Only the
rewards fed to advantage computation change.
"""

from types import SimpleNamespace

import pytest

from reliquary import constants as C
from reliquary.validator import training
from reliquary.validator.training import (
    _overlong_bounds,
    _overlong_penalty,
    _overlong_training_metrics,
    _plan_from_batches,
    _training_rewards,
)

CAP = 16384
SOFT = CAP - 4096  # 12288


@pytest.fixture(autouse=True)
def _v3_defaults(monkeypatch):
    monkeypatch.setattr(C, "OVERLONG_PENALTY_FACTOR", 0.5)
    monkeypatch.setattr(C, "OVERLONG_PENALTY_CACHE_TOKENS", 4096)
    monkeypatch.setattr(C, "TRAIN_FORCED_REWARD_ZERO", True)
    monkeypatch.setattr(
        C, "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV",
        {"openmathinstruct": CAP, "opencodeinstruct": CAP},
    )


def _roll(reward, length, *, forced=False, env="openmathinstruct"):
    return SimpleNamespace(
        reward=reward,
        env_name=env,
        commit={"rollout": {
            "prompt_length": 1,
            "completion_length": length,
            "token_logprobs": [-1.0] * length,
            "forced": forced,
        }},
    )


def _group(rollouts, prompt_idx=0):
    return SimpleNamespace(rollouts=rollouts, prompt_idx=prompt_idx)


def test_penalty_ramp_shape():
    assert _overlong_penalty(1000, SOFT, CAP) == 0.0
    assert _overlong_penalty(SOFT, SOFT, CAP) == 0.0
    assert _overlong_penalty(SOFT + 2048, SOFT, CAP) == pytest.approx(-0.5)
    assert _overlong_penalty(CAP, SOFT, CAP) == -1.0
    assert _overlong_penalty(CAP + 5000, SOFT, CAP) == -1.0


def test_bounds_follow_env_cap():
    assert _overlong_bounds("openmathinstruct") == (SOFT, CAP)
    assert _overlong_bounds("") == (
        C.MAX_NEW_TOKENS_PROTOCOL_CAP - 4096,
        C.MAX_NEW_TOKENS_PROTOCOL_CAP,
    )


def test_short_rollout_unchanged():
    assert _training_rewards([_roll(1.0, 5000)]) == [1.0]


def test_ramp_hits_correct_rollouts_too():
    # A correct answer in the overlong band must score below the same answer
    # short — otherwise convergence is never taught.
    out = _training_rewards([_roll(1.0, SOFT + 2048)])
    assert out[0] == pytest.approx(1.0 - 0.5 * 0.5)


def test_forced_lottery_is_neutralized():
    # Lucky (grade 1.0) and unlucky (grade 0.0) forced rollouts train
    # identically: base zeroed, flat full penalty.
    lucky, unlucky = _roll(1.0, 16136, forced=True), _roll(0.0, 16136, forced=True)
    assert _training_rewards([lucky, unlucky]) == [-0.5, -0.5]


def test_forced_flat_penalty_ignores_answer_length_jitter():
    # +-30 tokens of sampled answer between forced rollouts must not become
    # full-size advantages through group normalization.
    a, b = _roll(0.0, 16110, forced=True), _roll(0.0, 16140, forced=True)
    out = _training_rewards([a, b])
    assert out[0] == out[1]


def test_forced_zero_disabled_keeps_grade(monkeypatch):
    monkeypatch.setattr(C, "TRAIN_FORCED_REWARD_ZERO", False)
    assert _training_rewards([_roll(1.0, 16136, forced=True)]) == [0.5]


def test_factor_zero_restores_legacy(monkeypatch):
    monkeypatch.setattr(C, "OVERLONG_PENALTY_FACTOR", 0.0)
    monkeypatch.setattr(C, "TRAIN_FORCED_REWARD_ZERO", False)
    rollouts = [_roll(1.0, 16136, forced=True), _roll(0.0, CAP), _roll(1.0, 100)]
    assert _training_rewards(rollouts) == [1.0, 0.0, 1.0]


def test_graded_reward_attribute_untouched():
    rollouts = [_roll(1.0, 16136, forced=True), _roll(0.0, CAP)]
    _training_rewards(rollouts)
    assert [r.reward for r in rollouts] == [1.0, 0.0]


def test_all_forced_group_is_degenerate():
    # The BFT lottery gave this group sigma on paper (raw k=4); with every
    # base zeroed it carries no training signal and must be skipped.
    group = _group([_roll(float(i % 2), 16136, forced=True) for i in range(8)])
    plan, skipped = _plan_from_batches([[group]])
    assert plan == [] and skipped == 1


def test_mixed_group_advantage_ordering():
    rollouts = (
        [_roll(1.0, 4200)]
        + [_roll(0.0, 5000), _roll(0.0, 3900), _roll(0.0, 6100)]
        + [_roll(1.0, 16136, forced=True), _roll(1.0, 16136, forced=True)]
        + [_roll(0.0, 16136, forced=True), _roll(0.0, 16136, forced=True)]
    )
    plan, _ = _plan_from_batches([[_group(rollouts)]])
    ((_, advantages, _scale),) = plan
    correct_short, wrong_short = advantages[0], advantages[1]
    forced = advantages[4:]
    # correct+terminated >> wrong+terminated > every ruminator, lucky or not.
    assert correct_short > wrong_short > max(forced)
    assert all(a == forced[0] for a in forced)
    assert all(a < 0 for a in forced)


def test_code_truncation_reaches_full_penalty():
    # No BFT in code: a cap-length rollout hits the ramp floor naturally.
    out = _training_rewards([_roll(0.0, CAP, env="opencodeinstruct")])
    assert out[0] == pytest.approx(-0.5)


def test_overlong_metrics_report_reshaping():
    rollouts = [_roll(1.0, 4200), _roll(0.0, 16136, forced=True)]
    plan, _ = _plan_from_batches([[_group(rollouts)]])
    metrics = _overlong_training_metrics(plan)
    assert metrics["train/overlong_rollout_count"] == 2.0
    assert metrics["train/overlong_forced_zeroed_ratio"] == 0.5
    assert metrics["train/overlong_changed_ratio"] == 0.5
    assert metrics["train/overlong_reward_delta_mean"] == pytest.approx(-0.25)


def test_masked_rollout_still_skipped_when_kill_switch_on(monkeypatch):
    # The mask flags remain functional as kill switches on top of the penalty.
    monkeypatch.setattr(C, "MASK_MATH_FORCED_FROM_LOSS", True)
    forced = _roll(0.0, 16136, forced=True)
    assert training._is_masked_from_loss(forced) is True
    monkeypatch.setattr(C, "MASK_MATH_FORCED_FROM_LOSS", False)
    assert training._is_masked_from_loss(forced) is False
