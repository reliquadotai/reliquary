from types import SimpleNamespace

from reliquary import constants as C
from reliquary.validator.training import (
    _compute_advantages,
    _plan_from_batches,
    _shape_advantages,
    _shaping_training_metrics,
    _training_environment_metrics,
)


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


def test_shaping_penalizes_under_thinking_only():
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


def test_shaping_leaves_forced_untouched():
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    # forced + finished-early + wrong → still untouched (E7)
    out = _shape_advantages([_roll(0.0, early, forced=True)], [0.5])
    assert out[0] == 0.5


def test_shaping_penalizes_truncated_overlong():
    # overlong side penalises a cap-truncated rollout regardless of correctness
    out = _shape_advantages([_roll(1.0, C.BFT_THINKING_BUDGET, truncated=True)], [0.4])
    assert out[0] == -C.SHAPE_PENALTY


def test_shaping_off_when_penalty_zero(monkeypatch):
    monkeypatch.setattr(C, "SHAPE_PENALTY", 0.0)
    early = int(C.SHAPE_LEN_FRAC * C.BFT_THINKING_BUDGET) - 1
    out = _shape_advantages([_roll(0.0, early)], [0.3])
    assert out == [0.3]


def test_plan_keeps_all_wrong_group_when_shape_adds_signal():
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


def test_shaping_metrics_separate_overlong_underthinking_and_forced():
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
    assert metrics["train/shaping_changed_ratio"] == 0.5


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
