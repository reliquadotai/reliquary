from __future__ import annotations

from scripts.report_training_recovery import (
    _markdown,
    build_report,
    deduplicate_history,
)


def test_deduplicate_history_removes_resume_duplicates():
    row = {
        "_step": 10,
        "_timestamp": 100.0,
        "train/grad_norm": 1.5,
        "train/kl": 0.01,
    }
    richer = {**row, "train/ppo_loss": -0.2}

    result = deduplicate_history([row, richer, {**row, "_timestamp": 101.0}])

    assert len(result) == 2
    assert result[0]["train/ppo_loss"] == -0.2


def test_build_report_assigns_steps_to_checkpoint_intervals():
    history = [
        {
            "_step": 10,
            "_timestamp": 110.0,
            "train/grad_norm": 2.0,
            "train/kl": 0.01,
            "train/ppo_loss": -0.2,
            "train/rollouts_processed": 128,
            "bft/forced_rollout_ratio": 0.0,
            "train/ppo_ratio_outside_clip_ratio": 0.001,
            "train/ppo_clip_active_ratio": 0.0005,
            "train/kl_to_ppo_abs_ratio": 0.01,
            "train/shaping_changed_ratio": 0.0,
            "train/env/openmathinstruct/reward_mean": 0.5,
            "train/env/openmathinstruct/reward_nonzero_ratio": 0.5,
            "train/env/openmathinstruct/plan_groups": 8,
        },
        {
            "_step": 11,
            "_timestamp": 120.0,
            "train/grad_norm": 10432.0,
            "train/kl": 19.6,
            "train/ppo_loss": -0.1,
            "train/rollouts_processed": 128,
            "bft/forced_rollout_ratio": 0.0,
            "train/ppo_ratio_outside_clip_ratio": 0.06,
            "train/ppo_ratio_outside_clip_skip_threshold": 0.05,
            "train/step_skipped_policy_ratio_drift": 1.0,
            "train/env/openmathinstruct/reward_mean": 0.4,
            "train/env/openmathinstruct/reward_nonzero_ratio": 0.4,
            "train/env/openmathinstruct/plan_groups": 8,
        },
        {
            "_step": 12,
            "_timestamp": 210.0,
            "train/grad_norm": 1.0,
            "train/kl": 0.0,
        },
    ]
    checkpoints = [
        {
            "id": "a" * 40,
            "title": "initial commit",
            "date": "1970-01-01T00:01:40Z",
        },
        {
            "id": "b" * 40,
            "title": "checkpoint 1",
            "date": "1970-01-01T00:03:20Z",
        },
    ]
    termination = [
        {
            "event": "termination_shadow",
            "checkpoint_hash": "a" * 40,
            "window_start": 10,
            "miner_hotkey": "hk",
            "termination_ok": False,
            "cap_truncated": True,
            "terminal_boundary_compatible": False,
            "natural_close_boundary_compatible": False,
        }
    ]

    report = build_report(history, checkpoints, termination)

    assert report["history_steps_unique"] == 3
    assert report["global"]["gradient_norm"]["gt_100"] == 1
    assert report["anomalies"][0]["window"] == 11
    assert report["anomalies"][0]["forced_rollout_ratio"] == 0.0
    assert report["checkpoint_intervals"][0]["steps"] == 2
    assert report["checkpoint_intervals"][1]["steps"] == 1
    assert report["termination_by_checkpoint"]["a" * 40]["termination_failures"] == 1
    assert report["canary_policy"]["ppo_ratio_outside_clip"]["latest"] == 0.06
    assert (
        report["training_environments"]["openmathinstruct"]["reward_mean"]["min"] == 0.4
    )
    assert report["training_health_gate_events"] == [
        {
            "window": 11,
            "timestamp": "1970-01-01T00:02:00Z",
            "reasons": ["policy_ratio_drift"],
            "gradient_norm": 10432.0,
            "ppo_ratio_outside_clip": 0.06,
            "ppo_ratio_threshold": 0.05,
        }
    ]

    markdown = _markdown(report)
    assert "## Canary Policy Health" in markdown
    assert "| ppo_ratio_outside_clip | 2 | 0.001 | 0.06 |" in markdown
    assert "| 11 | 1970-01-01T00:02:00Z | policy_ratio_drift |" in markdown
    assert "| openmathinstruct | reward_mean | 2 | 0.5 | 0.4 |" in markdown
