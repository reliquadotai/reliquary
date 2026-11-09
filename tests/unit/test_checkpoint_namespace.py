"""Checkpoint namespaces isolate publication, recovery and local state."""

import asyncio
import json
from pathlib import Path

import pytest

from reliquary.shared.checkpoint_namespace import CheckpointNamespace, active_checkpoint_namespace
from reliquary.trainer.publisher import MIRROR_COMPLETE, checkpoint_key
from reliquary.validator.checkpoint_profile import (
    CHECKPOINT_PROFILE_NAME, CheckpointProfileMismatch,
    active_checkpoint_profile, validate_checkpoint_profile,
)


BASE = "0" * 40
REV = "1" * 40


def _publish(pub, n, parent=BASE):
    return asyncio.run(pub.publish(
        object(), checkpoint_n=n, trained_window_cursor=n * 10,
        lr_schedule_step=5, reason="cadence", parent_revision=parent,
    ))


def test_disabled_mode_preserves_legacy_even_for_named_run(tmp_path):
    legacy = active_checkpoint_namespace({"RELIQUARY_TASK_ID": "another-task",
                                          "RELIQUARY_TRAINING_RUN_ID": "old/run"})
    assert legacy.identity == {}
    assert legacy.local_path(tmp_path) == tmp_path
    assert legacy.candidate_manifest_key == "reliquary/training/candidate-manifest.json"
    assert checkpoint_key(REV, "config.json", namespace=legacy) == f"reliquary/checkpoints/{REV}/config.json"
    assert active_checkpoint_profile(namespace=legacy) == active_checkpoint_profile()


@pytest.mark.parametrize("task,run", [(None, "run"), ("task", None), ("../task", "run"),
                                      ("task", "../run"), ("task", "run/a"),
                                      ("task\n", "run"), ("task", " run"), ("task", "")])
def test_requested_scope_never_falls_back_for_bad_identity(task, run):
    with pytest.raises(ValueError):
        active_checkpoint_namespace({"RELIQUARY_TASK_SCOPED_CHECKPOINTS": "1",
                                     "RELIQUARY_TASK_ID": task, "RELIQUARY_TRAINING_RUN_ID": run})


def test_profile_without_runtime_scope_is_not_silently_legacy(tmp_path):
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(
        active_checkpoint_profile(namespace=CheckpointNamespace("task", "run"))))
    with pytest.raises(CheckpointProfileMismatch):
        validate_checkpoint_profile(tmp_path, required=True,
                                    expected=active_checkpoint_profile(namespace=CheckpointNamespace()))


def test_local_scope_refuses_aliases_but_allows_configured_root(tmp_path):
    scope = CheckpointNamespace("task", "run-a")
    real_root = tmp_path / "real"
    real_root.mkdir()
    configured = tmp_path / "configured"
    configured.symlink_to(real_root, target_is_directory=True)
    own = scope.local_path(configured)
    own.parent.mkdir(parents=True)
    other = own.parent / "run-b"
    other.mkdir()
    own.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scope.local_path(configured)
    own.unlink()
    own.mkdir()
    (own / "resume").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scope.child_path(own, "resume", REV)
    assert not (other / REV).exists()


def test_runtime_checkpoint_policy_cannot_disagree_with_mode():
    scoped = CheckpointNamespace("task", "run")
    scoped.require_policy({"kind": "trainer-driven/v1", "task_scoped": 1})
    CheckpointNamespace().require_policy({"kind": "trainer-driven/v1", "task_scoped": 0})
    for policy in ({"kind": "frozen/v1"}, {"kind": "trainer-driven/v1", "task_scoped": 0},
                   {"kind": "trainer-driven/v1", "task_scoped": True}):
        with pytest.raises(ValueError):
            scoped.require_policy(policy)


def _enable_scope(monkeypatch, scope):
    monkeypatch.setenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "1")
    monkeypatch.setenv("RELIQUARY_TASK_ID", scope.task_id)
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", scope.run_id)


