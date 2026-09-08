"""A new curriculum is explicit, isolated and monotonically numbered."""

import os
import subprocess
import sys

import pytest

from scripts import prepare_v1_bootstrap as prepare
from tests.unit.test_prepare_v5_fill_checkpoint import _profile


def test_new_curriculum_bootstrap_preserves_parent_but_resets_run_and_cursor(monkeypatch):
    source = {**_profile("qwen3-4b-base-dapo-reasoning-v5"),
              "trained_window_cursor": 50, "lr_schedule_step": 800}
    target = {**_profile(prepare.PROFILE), "training_run_id": "reliquary-v1-test"}
    monkeypatch.setattr(prepare, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(prepare, "PROTOCOL_PROFILE_ID", prepare.PROFILE)
    monkeypatch.setattr(prepare, "active_checkpoint_profile", lambda: dict(target))
    import reliquary.constants as constants
    monkeypatch.setattr(constants, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    args = dict(repo_id="owner/model", revision="a" * 40, checkpoint_n=100,
                last_archived_window=50, source_bucket="old-run", target_bucket="new-run")
    plan = prepare.prepare_bootstrap(source, **args)
    profile = plan["files"][prepare.CHECKPOINT_PROFILE_NAME]
    transition = plan["files"][prepare.TRANSITION]
    assert profile["trained_window_cursor"] + 1 == 51 * 16
    assert profile["lr_schedule_step"] == 0
    assert transition["target_checkpoint_n"] == 101
    assert plan["parent_commit"] == "a" * 40
    assert len(transition["environment_targets"]) == 3
    assert "private_storage_migration" not in transition
    for changes in ({"target_bucket": "old-run"}, {"last_archived_window": 51},
                    {"checkpoint_n": True}, {"lr_start_step": -1}):
        with pytest.raises(ValueError):
            prepare.prepare_bootstrap(source, **{**args, **changes})
    target["training_run_id"] = source["training_run_id"]
    with pytest.raises(ValueError, match="new explicit run"):
        prepare.prepare_bootstrap(source, **args)


def test_v1_profile_needs_fill_capability_and_rejects_retired_epoch_flag():
    base = {key: value for key, value in os.environ.items() if not key.startswith("RELIQUARY_")}
    base["RELIQUARY_PROTOCOL_PROFILE"] = prepare.PROFILE
    script = "from reliquary import constants as c; assert len(c.ENVIRONMENT_MIX)==3; assert c.M_ROLLOUTS==16; assert c.FILL_CLOSED_ENABLED"
    result = subprocess.run([sys.executable, "-c", script], env=base, capture_output=True, text=True, check=False)
    assert result.returncode and "explicit" in result.stderr
    base["RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED"] = "1"
    subprocess.run([sys.executable, "-c", script], env=base, check=True)
    base["RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED"] = "1"
    result = subprocess.run([sys.executable, "-c", script], env=base, capture_output=True, text=True, check=False)
    assert result.returncode and "retired" in result.stderr
