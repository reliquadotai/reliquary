"""Adding a task must not move the weights of the task already running."""

from __future__ import annotations

import pytest

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


def test_a_task_with_archives_but_no_registry_entry_is_flagged():
    """Paying a task nobody declared is paying under rules nobody agreed to."""
    undeclared = WeightOnlyValidator._undeclared_tasks(
        {"default": [], "ghost": []}, {"default": object()}
    )

    assert undeclared == ["ghost"]


def test_declared_tasks_are_not_flagged():
    assert WeightOnlyValidator._undeclared_tasks(
        {"a": [], "b": []}, {"a": object(), "b": object()}
    ) == []


def test_every_archived_task_missing_from_the_registry_is_named():
    undeclared = WeightOnlyValidator._undeclared_tasks(
        {"b": [], "a": [], "ok": []}, {"ok": object()}
    )

    assert undeclared == ["a", "b"]


# --- A retired task's cap is only reserved while its EMA decays; the decay
# is what makes retirement mean anything. ---

def test_a_declared_cap_clamps_the_task_that_exceeds_it():
    """`window_pool` binds only the validator that produced the archive. A box
    on a stale image, or one declared at a lower cap after it started, archives
    a full pool regardless — the cap has to bind where money is assigned."""
    merged = WeightOnlyValidator._merge_archives({
        "greedy": [_archive(w, "greedy", {"hk_b": 1.0}) for w in range(1, 400)],
    })

    ema = WeightOnlyValidator._replay_ema(merged, caps={"greedy": 0.25})

    assert sum(ema.values()) == pytest.approx(0.25)


def test_a_task_capped_at_zero_pays_nobody():
    merged = WeightOnlyValidator._merge_archives({
        "shadow": [_archive(w, "shadow", {"hk_b": 1.0}) for w in range(1, 400)],
    })

    assert WeightOnlyValidator._replay_ema(merged, caps={"shadow": 0.0}) == {}


def test_a_task_over_its_cap_does_not_move_default():
    """Without the per-task clamp the combined total reaches ~2.0 and the
    global backstop rescales EVERYONE: `default`'s miners lose half their
    emission to a task declared at cap 0.0."""
    alone = [_archive(w, "default", {"hk_a": 1.0}) for w in range(1, 400)]
    beside = WeightOnlyValidator._merge_archives({
        "default": list(alone),
        "shadow": [_archive(w, "shadow", {"hk_b": 1.0}) for w in range(1, 400)],
    })
    caps = {"default": 1.0, "shadow": 0.0}

    with_shadow = WeightOnlyValidator._replay_ema(beside, caps=caps)
    without = WeightOnlyValidator._replay_ema(alone, caps=caps)

    assert with_shadow["hk_a"] == without["hk_a"]
    assert "hk_b" not in with_shadow


def test_a_task_inside_its_cap_is_left_exactly_alone():
    archives = [_archive(w, "default", {"hk_a": 0.4}) for w in range(1, 400)]

    assert (
        WeightOnlyValidator._replay_ema(archives, caps={"default": 1.0})
        == WeightOnlyValidator._replay_ema(archives)
    )


def test_caps_come_from_the_registry_entries():
    from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry

    entry = TaskEntry(
        task_id="logic-probe", profile_id="p", profile_sha256="a" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params={"start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
                "deadband": 0.8, "snap": 1.2, "floor": 0.05, "cap": 0.3,
                "median_rounds": 4800},
        status="active", retired_at=None,
    )

    assert WeightOnlyValidator._caps_by_task({"logic-probe": entry}) == {"logic-probe": 0.3}


def test_an_entry_with_no_readable_cap_is_left_unclamped_not_guessed():
    """Only reachable with a non-registry stand-in: `read_registry` validates
    every entry, so a real one always carries a finite numeric cap."""
    assert WeightOnlyValidator._caps_by_task({"default": object()}) == {}
    assert WeightOnlyValidator._caps_by_task({}) == {}
