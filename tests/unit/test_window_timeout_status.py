"""A timeout describes the market; a crash describes us."""

from __future__ import annotations

from types import SimpleNamespace

from reliquary.constants import B_BATCH
from reliquary.validator.emission_price import (
    outcome_from_archive,
    outcomes_by_environment_from_archive,
)
from reliquary.validator.fill_closed_batch_assembler import (
    FillClosedBatchAssembler,
)
from tests.unit.test_training_payload_codec import _roll

ENV_ORDER = ["openmathinstruct", "opencodeinstruct"]


def _record(status: str) -> dict:
    return {
        "window_status": status,
        "window_open_round": 0,
        "window_close_round": 1000,
        "collect_ready_round": None,
        "collect_ready_round_by_environment": {"math": None, "code": 400},
        "training_rounds": 0.0,
        "validation_rounds": 500.0,
    }


def test_an_aborted_window_carries_no_price_signal():
    assert outcome_from_archive(_record("aborted")) is None
    assert outcomes_by_environment_from_archive(_record("aborted")) is None


def test_a_timed_out_window_carries_its_signal():
    outcomes = outcomes_by_environment_from_archive(_record("timed_out"))

    assert outcomes is not None
    assert outcomes["math"].filled is False
    assert outcomes["code"].filled is True


def test_a_timed_out_window_yields_no_ratio_for_anyone():
    """No training ran, so the denominator is undefined. Shortage is carried
    by `filled`, and inventing a denominator would inflate the ratio exactly
    on the worst windows."""
    outcomes = outcomes_by_environment_from_archive(_record("timed_out"))

    assert outcomes["math"].ratio is None
    assert outcomes["code"].ratio is None


def test_a_completed_window_does_yield_a_ratio():
    outcomes = outcomes_by_environment_from_archive(_record("completed"))

    assert outcomes["code"].ratio is not None


def test_a_completed_window_still_carries_its_signal():
    assert outcomes_by_environment_from_archive(_record("completed")) is not None


def _paid_group(hotkey: str, tag: int, env: str) -> SimpleNamespace:
    return SimpleNamespace(
        rollouts=[_roll(1.0, 4, env=env)], prompt_idx=tag, hotkey=hotkey,
    )


def _full_cycle(hotkeys: list, env: str) -> list:
    return [_paid_group(hk, i, env) for i, hk in enumerate(hotkeys)]


def _assembler(window: int) -> FillClosedBatchAssembler:
    return FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
    )



def test_a_timed_out_windows_partial_remainder_is_paid_but_not_trained():
    """Every accepted group is paid, even when the window times out. What a
    timed-out window must not do is hand its unbalanced trailing cycle to the
    trainer, so ``close(train_partial_remainder=False)`` pays that cycle and
    tombstones it instead of enqueueing it."""
    window = 42
    math_hotkeys = [f"math-{i}" for i in range(B_BATCH)]
    code_hotkeys = [f"code-{i}" for i in range(B_BATCH)]
    enqueued: list[int] = []
    assembler = FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: enqueued.append(key),
        tombstone_fn=lambda key, data: None,
    )
    assembler.accept(
        "openmathinstruct", _full_cycle(math_hotkeys, "openmathinstruct"), window, "rev",
    )
    assembler.accept(
        "opencodeinstruct", _full_cycle(code_hotkeys, "opencodeinstruct"), window, "rev",
    )
    assembler.accept(
        "openmathinstruct", [_paid_group("math-straggler", 999, "openmathinstruct")],
        window, "rev",
    )
    assembler.accept(
        "opencodeinstruct", [_paid_group("code-straggler", 999, "opencodeinstruct")],
        window, "rev",
    )
    # Both environments hold a group in the final cycle, so a normal close()
    # would train on it: confirm the scenario exercises that branch.
    assert assembler._accumulator.has_groups_for_all_targets
    trained_before_close = list(enqueued)

    assembler.close(train_partial_remainder=False)

    paid = assembler.reward_map()
    assert paid["math-straggler"] == paid["math-0"]
    assert paid["code-straggler"] == paid["code-0"]
    assert enqueued == trained_before_close
    assert len(assembler.paid_groups()["openmathinstruct"]) == B_BATCH + 1
    assert len(assembler.paid_groups()["opencodeinstruct"]) == B_BATCH + 1


