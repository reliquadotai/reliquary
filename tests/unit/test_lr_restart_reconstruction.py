"""LR schedule survives restarts within a run; a NEW run still warms up.

Position is reconstructed from the published checkpoint count (already
restart-proof), gated on the run id carried by the checkpoint profile.
Adam moments are NOT persisted, so a same-run restart applies a short
re-warmup ramp instead of jumping straight to full LR.
"""
import math

import pytest
import torch

import reliquary.validator.training as training
from reliquary.constants import (
    LR_RESTART_REWARMUP_WINDOWS, LR_WARMUP_WINDOWS, LEARNING_RATE,
    TRAINING_RUN_ID,
)
from reliquary.validator.checkpoint_profile import active_checkpoint_profile
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


def test_fresh_run_full_warmup():
    assert training._lazy_init(_tiny_model(), global_step_hint=0)
    # step 0 of a fresh run: first rung of the 10-window warmup
    assert _lr() == pytest.approx(LEARNING_RATE / LR_WARMUP_WINDOWS)


def test_same_run_restart_fast_forwards_with_rewarmup():
    hint = 499 * 16  # checkpoint_n x publish interval
    assert training._lazy_init(_tiny_model(), global_step_hint=hint)
    sched = training._scheduler
    assert sched.last_epoch == hint
    # first window post-boot: cosine position x restart ramp 1/R
    progress = (hint - LR_WARMUP_WINDOWS) / (10_000 - LR_WARMUP_WINDOWS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    expected_first = LEARNING_RATE * cosine * (1 / LR_RESTART_REWARMUP_WINDOWS)
    assert _lr() == pytest.approx(expected_first, rel=1e-6)
    # after R scheduler steps the ramp is exhausted: full cosine LR
    for _ in range(LR_RESTART_REWARMUP_WINDOWS):
        sched.step()
    progress2 = (
        hint + LR_RESTART_REWARMUP_WINDOWS - LR_WARMUP_WINDOWS
    ) / (10_000 - LR_WARMUP_WINDOWS)
    cosine2 = 0.5 * (1 + math.cos(math.pi * progress2))
    assert _lr() == pytest.approx(LEARNING_RATE * cosine2, rel=1e-6)


def test_restart_mid_warmup_composes_both_ramps():
    assert training._lazy_init(_tiny_model(), global_step_hint=4)
    # warmup rung 5/10 x restart ramp 1/R — never MORE than the fresh path
    expected = LEARNING_RATE * (5 / LR_WARMUP_WINDOWS) * (
        1 / LR_RESTART_REWARMUP_WINDOWS
    )
    assert _lr() == pytest.approx(expected, rel=1e-6)


def test_hint_consumed_only_at_first_init():
    model = _tiny_model()
    assert training._lazy_init(model, global_step_hint=100)
    stepped = training._scheduler.last_epoch
    # later calls with a different hint must not rebuild or move anything
    assert training._lazy_init(model, global_step_hint=5000)
    assert training._scheduler.last_epoch == stepped


class _HintStub:
    _lr_global_step_hint = ValidationService._lr_global_step_hint

    def __init__(self, ckpt_n, since=3, resumed_run_id=None):
        self._checkpoint_n = ckpt_n
        self._publish_every = 16
        self._trained_windows_since_publish = since
        self._resumed_training_run_id = resumed_run_id


def test_hint_same_run_id():
    stub = _HintStub(499, since=3, resumed_run_id=TRAINING_RUN_ID)
    assert stub._lr_global_step_hint() == 499 * 16 + 3


def test_hint_missing_field_treated_as_same_run():
    stub = _HintStub(499, since=0, resumed_run_id=None)
    assert stub._lr_global_step_hint() == 499 * 16


def test_hint_new_run_id_forces_full_warmup():
    stub = _HintStub(499, resumed_run_id="some-other-run-2026")
    assert stub._lr_global_step_hint() == 0


def test_hint_fresh_repo_forces_full_warmup():
    stub = _HintStub(0, resumed_run_id=TRAINING_RUN_ID)
    assert stub._lr_global_step_hint() == 0


def test_checkpoint_profile_carries_run_id():
    profile = active_checkpoint_profile()
    assert profile["training_run_id"] == TRAINING_RUN_ID


def test_profile_validation_tolerates_missing_run_id(tmp_path):
    """Historical checkpoints predate the field: still valid, both ways."""
    import json
    from reliquary.validator.checkpoint_profile import (
        CHECKPOINT_PROFILE_NAME, validate_checkpoint_profile,
    )
    legacy = active_checkpoint_profile()
    legacy.pop("training_run_id")
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(legacy))
    value = validate_checkpoint_profile(tmp_path, required=True)
    assert value is not None
    assert value.get("training_run_id") is None
