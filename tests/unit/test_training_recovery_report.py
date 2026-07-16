from __future__ import annotations

from scripts.report_training_recovery import (
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
        },
        {
            "_step": 11,
            "_timestamp": 120.0,
            "train/grad_norm": 10432.0,
            "train/kl": 19.6,
            "train/ppo_loss": -0.1,
            "train/rollouts_processed": 128,
            "bft/forced_rollout_ratio": 0.0,
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
    assert report["termination_by_checkpoint"]["a" * 40][
        "termination_failures"
    ] == 1
