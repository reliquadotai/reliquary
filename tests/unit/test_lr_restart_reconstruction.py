"""LR schedule position survives restarts via the EXACT step counter
recorded in the checkpoint profile at publish — never derived from
checkpoint numbers (this run inherited its counter across a weight reset;
publish cadence changed across protocol versions). Fail-closed: missing
field or foreign run id => full warmup, exactly like before the feature.
"""
import json

import pytest
import torch

import reliquary.validator.training as training
from reliquary.constants import (
    LR_RESTART_REWARMUP_WINDOWS, LR_WARMUP_WINDOWS,
    LEARNING_RATE, TRAINING_RUN_ID,
)
from reliquary.validator.checkpoint_profile import (
    CHECKPOINT_PROFILE_NAME, active_checkpoint_profile,
    validate_checkpoint_profile, write_checkpoint_profile,
)
from reliquary.validator.service import ValidationService


@pytest.fixture(autouse=True)
def _fresh_training_state():
    training.reset_training_state()
    yield
    training.reset_training_state()


def _tiny_model():
    return torch.nn.Linear(4, 4)


def _lr():
    return training._optimizer.param_groups[0]["lr"]


def _unrestarted_lambda(step):
    """The schedule a run that never restarted would see: linear warmup,
    then flat. Written out independently so a change to the production
    lambda has to be made twice, deliberately."""
    if step < LR_WARMUP_WINDOWS:
        return (step + 1) / LR_WARMUP_WINDOWS
    return 1.0


def test_fresh_run_warms_up_then_holds():
    assert training._lazy_init(_tiny_model(), global_step_hint=0)
    lam = training._scheduler.lr_lambdas[0]
    for step in range(0, 12_000, 7):
        assert lam(step) == _unrestarted_lambda(step), step


def test_schedule_is_flat_after_warmup():
    """DAPO §4.1 runs a constant LR. The old cosine over 10k windows was
    justified as "≈ constant at realistic window counts"; at ~94 s/window a
    run crosses 10k in ~11 days and the factor reaches 0 mid-run."""
    assert training._lazy_init(_tiny_model(), global_step_hint=0)
    lam = training._scheduler.lr_lambdas[0]
    for step in range(LR_WARMUP_WINDOWS, 60_000, 137):
        assert lam(step) == 1.0, step


def test_restored_position_reported_exactly():
    assert training._lazy_init(_tiny_model(), global_step_hint=337)
    assert training.current_lr_schedule_step() == 337
    for _ in range(5):
        training._scheduler.step()
    assert training.current_lr_schedule_step() == 342


def test_step_counter_none_before_init():
    assert training.current_lr_schedule_step() is None


def test_restart_runs_exactly_r_windows_below_full_lr():
    hint = 300  # past warmup: base factor is 1.0
    assert training._lazy_init(_tiny_model(), global_step_hint=hint)
    lam = training._scheduler.lr_lambdas[0]
    below = [
        k for k in range(hint, hint + LR_RESTART_REWARMUP_WINDOWS + 5)
        if lam(k) < _unrestarted_lambda(k)
    ]
    assert len(below) == LR_RESTART_REWARMUP_WINDOWS


def test_after_ramp_exact_equality_with_unrestarted_run():
    hint = 337
    assert training._lazy_init(_tiny_model(), global_step_hint=hint)
    lam = training._scheduler.lr_lambdas[0]
    for k in range(hint + LR_RESTART_REWARMUP_WINDOWS, hint + 500):
        assert lam(k) == _unrestarted_lambda(k), k


def test_first_window_uses_ramped_restored_position():
    hint = 337
    assert training._lazy_init(_tiny_model(), global_step_hint=hint)
    expected = LEARNING_RATE * _unrestarted_lambda(hint) * (
        1 / (LR_RESTART_REWARMUP_WINDOWS + 1)
    )
    assert _lr() == pytest.approx(expected, rel=1e-6)


def test_hint_consumed_only_at_first_init():
    model = _tiny_model()
    assert training._lazy_init(model, global_step_hint=100)
    assert training._lazy_init(model, global_step_hint=5000)
    assert training.current_lr_schedule_step() == 100


class _HintStub:
    _lr_global_step_hint = ValidationService._lr_global_step_hint

    def __init__(self, resumed_step=None, resumed_run_id=None):
        self._resumed_lr_schedule_step = resumed_step
        self._resumed_training_run_id = resumed_run_id


def test_hint_is_the_recorded_step_never_a_derivation():
    stub = _HintStub(resumed_step=337, resumed_run_id=TRAINING_RUN_ID)
    assert stub._lr_global_step_hint() == 337


def test_hint_missing_step_field_fails_closed_to_full_warmup():
    """All checkpoints published before this feature: full warmup, exactly
    the pre-feature behavior — the counter-inherited-across-reset trap
    (repo starts at ckpt 486 with zero real steps) cannot fire."""
    stub = _HintStub(resumed_step=None, resumed_run_id=TRAINING_RUN_ID)
    assert stub._lr_global_step_hint() == 0
    stub2 = _HintStub(resumed_step=None, resumed_run_id=None)
    assert stub2._lr_global_step_hint() == 0


def test_hint_new_run_id_forces_full_warmup():
    stub = _HintStub(resumed_step=337, resumed_run_id="some-other-run")
    assert stub._lr_global_step_hint() == 0


def test_profile_roundtrip_records_and_restores_step(tmp_path):
    write_checkpoint_profile(tmp_path, extra={"lr_schedule_step": 337})
    value = validate_checkpoint_profile(tmp_path, required=True)
    assert value["lr_schedule_step"] == 337
    assert value["training_run_id"] == TRAINING_RUN_ID


def test_profile_validation_tolerates_legacy_without_new_fields(tmp_path):
    legacy = active_checkpoint_profile()
    legacy.pop("training_run_id")
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(legacy))
    value = validate_checkpoint_profile(tmp_path, required=True)
    assert value is not None
    assert value.get("lr_schedule_step") is None


def test_resume_capture_rejects_garbage_step_values():
    from reliquary.validator.service import _coerce_lr_schedule_step

    assert _coerce_lr_schedule_step(337) == 337
    assert _coerce_lr_schedule_step(0) == 0
    for garbage in ("337", -5, True, False, 3.5, None, [3]):
        assert _coerce_lr_schedule_step(garbage) is None
