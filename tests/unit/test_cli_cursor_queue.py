"""v6.1 (trainer-paced picks): the trainer-host queue instance that
carries the per-step pacing cursor to R2 via the same transport as
payloads/tombstones.

``run_train_worker`` itself loads a real model and cannot be unit-tested
end-to-end; this covers the one piece of its wiring that is a pure
function of its inputs -- the factory that builds the queue instance.
"""

from pathlib import Path

import pytest

from reliquary.infrastructure.training_payload_queue import TrainingPayloadQueue
from reliquary.trainer.cli import (
    _build_cursor_queue,
    _environment_positive_int,
    _profile_nonnegative_int,
)


def test_build_cursor_queue_returns_a_training_payload_queue(tmp_path):
    queue = _build_cursor_queue(tmp_path / "trainer-state")
    assert isinstance(queue, TrainingPayloadQueue)


def test_build_cursor_queue_is_scoped_under_the_given_state_dir(tmp_path):
    state_dir = tmp_path / "trainer-state"
    queue = _build_cursor_queue(state_dir)
    assert Path(queue.queue_dir).is_relative_to(state_dir)


def test_build_cursor_queue_uses_a_subdir_distinct_from_the_validator_default(
    tmp_path,
):
    """The validator's own TrainingPayloadQueue defaults to
    ``pending_training_payloads`` under RELIQUARY_STATE_DIR. Even if
    RELIQUARY_TRAINER_STATE_DIR ever pointed at the same root, the two
    queues must not glob each other's window-*/epoch-*/step-cursor files."""
    queue = _build_cursor_queue(tmp_path / "trainer-state")
    assert Path(queue.queue_dir).name != "pending_training_payloads"


def test_build_cursor_queue_round_trips_the_step_cursor(tmp_path):
    queue = _build_cursor_queue(tmp_path / "trainer-state")
    assert queue.read_step_cursor() is None
    queue.write_step_cursor(30142)
    assert queue.read_step_cursor() == 30142


@pytest.mark.parametrize("value", [True, 7.0, "7", -1])
def test_trainer_profile_cursor_does_not_coerce_durable_values(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        _profile_nonnegative_int({"trained_window_cursor": value}, "trained_window_cursor")


def test_trainer_environment_stride_is_explicitly_parsed(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TRAINER_WINDOW_STRIDE", " 2 ")

    assert _environment_positive_int("RELIQUARY_TRAINER_WINDOW_STRIDE", "1") == 2


@pytest.mark.parametrize("value", ["0", "-1", "1.0", "true"])
def test_trainer_environment_stride_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("RELIQUARY_TRAINER_WINDOW_STRIDE", value)

    with pytest.raises(ValueError):
        _environment_positive_int("RELIQUARY_TRAINER_WINDOW_STRIDE", "1")
