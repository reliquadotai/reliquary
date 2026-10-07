"""A corpus task is an ordinary registry entry whose price is pinned by
declaring floor == cap."""

import pytest

from reliquary.shared.task_registry import (
    MECHANISM_CORPUS_GENERATION,
    RegistryError,
    TaskEntry,
    add_task,
    validate_entry,
    parse_registry,
    render_registry,
    retire_task,
    set_admission,
    set_cap,
)


def _entry(**overrides):
    params = {
        "start": 0.10,
        "decay": 0.98,
        "rounds_per_step": 1000,
        "deadband": 0.80,
        "snap": 1.20,
        "floor": 0.10,
        "cap": 0.10,
        "median_rounds": 4800,
        "last_good_fills": 50,
    }
    params.update(overrides.pop("params", {}))
    payload = {
        "task_id": "corpus-math",
        "profile_id": "corpus-v1",
        "profile_sha256": "a" * 64,
        "mechanism": MECHANISM_CORPUS_GENERATION,
        "params": params,
        "status": "active",
        "retired_at": None,
        # A corpus entry names the job it generates for; an entry without one
        # is a share of the pool with no work attached to it.
        "job_id": "corpus-math-v1",
    }
    payload.update(overrides)
    return TaskEntry(**payload)


def test_the_corpus_mechanism_is_declarable():
    validate_entry(_entry())


def test_pause_resume_preserves_contract_bytes_and_economics_and_retirement_is_terminal():
    entries = {"corpus-math": _entry()}
    original = render_registry(entries)
    assert b'"admission"' not in original
    assert parse_registry(original)["corpus-math"].admission == "open"
    paused = set_admission(entries, "corpus-math", "paused")
    assert paused["corpus-math"].params == entries["corpus-math"].params
    assert paused["corpus-math"].profile_sha256 == entries["corpus-math"].profile_sha256
    assert paused["corpus-math"].status == "active"
    assert parse_registry(render_registry(paused))["corpus-math"].admission == "paused"
    assert render_registry(set_admission(paused, "corpus-math", "open")) == original
    assert set_cap(entries, "corpus-math", 0)["corpus-math"].admission == "open"
    retired = retire_task(paused, "corpus-math", 1000)
    with pytest.raises(RegistryError, match="retired"):
        set_admission(retired, "corpus-math", "open")
    with pytest.raises(RegistryError, match="admission"):
        set_admission(entries, "corpus-math", "invalid")


def test_a_pinned_price_is_legal():
    # floor == cap makes advance() mathematically inert: V0 pays the whole share.
    validate_entry(_entry(params={"floor": 0.10, "cap": 0.10}))


def test_a_floor_above_the_cap_is_still_refused():
    with pytest.raises(RegistryError):
        validate_entry(_entry(params={"floor": 0.20, "cap": 0.10}))


def test_an_unknown_mechanism_is_still_refused():
    with pytest.raises(RegistryError):
        validate_entry(_entry(mechanism="corpus-whatever"))


def test_a_corpus_task_may_sit_beside_the_rl_task_within_one_pool():
    rl = TaskEntry(
        task_id="default",
        profile_id="v6",
        profile_sha256="b" * 64,
        mechanism="rl-discovered-price",
        params={
            "start": 1.0, "decay": 0.98, "rounds_per_step": 1000, "deadband": 0.80,
            "snap": 1.20, "floor": 0.05, "cap": 0.90, "median_rounds": 4800,
            "last_good_fills": 50,
        },
        status="active",
        retired_at=None,
    )
    merged = add_task({"default": rl}, _entry())
    assert set(merged) == {"default", "corpus-math"}


def test_the_two_caps_may_not_exceed_the_single_pool():
    rl = TaskEntry(
        task_id="default",
        profile_id="v6",
        profile_sha256="b" * 64,
        mechanism="rl-discovered-price",
        params={
            "start": 1.0, "decay": 0.98, "rounds_per_step": 1000, "deadband": 0.80,
            "snap": 1.20, "floor": 0.05, "cap": 1.0, "median_rounds": 4800,
            "last_good_fills": 50,
        },
        status="active",
        retired_at=None,
    )
    with pytest.raises(RegistryError):
        add_task({"default": rl}, _entry())
