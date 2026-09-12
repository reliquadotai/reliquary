"""Adding a task must not move the weights of the task already running."""

from __future__ import annotations

from reliquary.validator.weight_only import WeightOnlyValidator


def _archive(window, task, rewards):
    return {"window_start": window, "task_id": task, "rewards_by_hotkey": rewards}


def test_archives_are_merged_in_window_then_task_order():
    merged = WeightOnlyValidator._merge_archives({
        "logic-probe": [_archive(2, "logic-probe", {}), _archive(1, "logic-probe", {})],
        "default": [_archive(1, "default", {}), _archive(2, "default", {})],
    })

    assert [(a["window_start"], a["task_id"]) for a in merged] == [
        (1, "default"), (1, "logic-probe"), (2, "default"), (2, "logic-probe"),
    ]


def test_a_task_that_pays_nothing_leaves_the_other_untouched():
    alone = [_archive(w, "default", {"hk_a": 1.0}) for w in range(1, 200)]
    beside = WeightOnlyValidator._merge_archives({
        "default": alone,
        "logic-probe": [_archive(w, "logic-probe", {"hk_b": 0.0}) for w in range(1, 200)],
    })

    assert WeightOnlyValidator._replay_ema(beside)["hk_a"] == WeightOnlyValidator._replay_ema(alone)["hk_a"]
    assert "hk_b" not in WeightOnlyValidator._replay_ema(beside)


def test_a_paying_task_does_take_a_share():
    both = WeightOnlyValidator._merge_archives({
        "default": [_archive(w, "default", {"hk_a": 1.0}) for w in range(1, 400)],
        "logic-probe": [_archive(w, "logic-probe", {"hk_b": 1.0}) for w in range(1, 400)],
    })

    ema = WeightOnlyValidator._replay_ema(both)

    # Equal archive rates, equal pools: the two tasks split the mass exactly.
    # Each task replays on its own decay clock, so within-window ordering
    # between tasks cannot affect either value.
    assert ema["hk_a"] == ema["hk_b"]
    assert abs(ema["hk_a"] + ema["hk_b"] - 1.0) < 1e-6
