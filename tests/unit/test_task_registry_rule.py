"""The sum of the caps is the invariant, and it lives in one place."""

from __future__ import annotations

import json

import pytest

from dataclasses import replace

from reliquary.shared.task_registry import (
    MECHANISM_RL_DISCOVERED_PRICE,
    RegistryError,
    TaskEntry,
    add_task,
    parse_registry,
    render_registry,
    retire_task,
    total_cap,
    validate_registry,
)

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(task_id: str, cap: float, status: str = "active") -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        profile_sha256="a" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params={**PARAMS, "cap": cap},
        status=status,
        retired_at=None,
    )


def test_caps_summing_to_exactly_one_are_allowed():
    entries = {"a": _entry("a", 0.6), "b": _entry("b", 0.4)}

    validate_registry(entries)

    assert total_cap(entries) == pytest.approx(1.0)


def test_caps_over_one_are_refused():
    entries = {"a": _entry("a", 0.6), "b": _entry("b", 0.5)}

    with pytest.raises(RegistryError, match="1.1"):
        validate_registry(entries)


def test_a_retired_task_still_reserves_its_cap():
    entries = {"a": _entry("a", 0.6, status="retired"), "b": _entry("b", 0.4)}

    assert total_cap(entries) == pytest.approx(1.0)

    with pytest.raises(RegistryError):
        validate_registry(add_task(entries, _entry("c", 0.1)))


def test_adding_a_task_that_would_overflow_is_refused():
    entries = {"a": _entry("a", 0.8)}

    with pytest.raises(RegistryError):
        add_task(entries, _entry("b", 0.3))


def test_adding_a_duplicate_id_is_refused():
    entries = {"a": _entry("a", 0.5)}

    with pytest.raises(RegistryError, match="already"):
        add_task(entries, _entry("a", 0.1))


def test_retiring_marks_the_entry_and_keeps_the_cap():
    entries = retire_task({"a": _entry("a", 0.5)}, "a", retired_at=12345)

    assert entries["a"].status == "retired"
    assert entries["a"].retired_at == 12345
    assert total_cap(entries) == pytest.approx(0.5)


@pytest.mark.parametrize("cap", [-0.1, 1.5, True])
def test_an_impossible_cap_is_refused(cap):
    with pytest.raises(RegistryError):
        add_task({}, _entry("a", cap))


def test_an_unknown_mechanism_is_refused():
    broken = replace(_entry("a", 0.5), mechanism="vibes")

    with pytest.raises(RegistryError, match="vibes"):
        add_task({}, broken)


def test_a_missing_price_parameter_is_refused():
    broken = replace(_entry("a", 0.5), params={"cap": 0.5})

    with pytest.raises(RegistryError, match="start"):
        add_task({}, broken)


def test_round_trip_is_canonical():
    entries = {"b": _entry("b", 0.4), "a": _entry("a", 0.6)}

    raw = render_registry(entries)

    assert parse_registry(raw) == entries
    assert list(json.loads(raw)["tasks"]) == ["a", "b"]


def test_a_registry_that_is_not_json_is_refused():
    with pytest.raises(RegistryError):
        parse_registry(b"{ not json")


def test_an_empty_registry_parses_to_nothing():
    assert parse_registry(render_registry({})) == {}


def test_parsing_an_oversubscribed_registry_is_refused():
    raw = render_registry({"a": _entry("a", 0.9), "b": _entry("b", 0.9)})

    with pytest.raises(RegistryError, match="1.8"):
        parse_registry(raw)


def test_an_oversubscribed_registry_can_still_be_inspected():
    raw = render_registry({"a": _entry("a", 0.9), "b": _entry("b", 0.9)})

    assert set(parse_registry(raw, strict=False)) == {"a", "b"}


def test_an_unusable_task_id_is_a_registry_error():
    raw = json.dumps({
        "registry_version": 1,
        "tasks": {"BadID": {
            "profile_id": "p",
            "profile_sha256": "a" * 64,
            "incentive": {
                "mechanism": MECHANISM_RL_DISCOVERED_PRICE,
                "params": {**PARAMS, "cap": 0.5},
            },
            "status": "active",
            "retired_at": None,
        }},
    }).encode()

    with pytest.raises(RegistryError, match="BadID"):
        parse_registry(raw)


def test_a_non_integer_retired_at_is_a_registry_error():
    with pytest.raises(RegistryError, match="retired_at"):
        retire_task({"a": _entry("a", 0.5)}, "a", retired_at=None)


def test_a_retired_entry_round_trips_still_reserving_its_cap():
    entries = retire_task({"a": _entry("a", 0.5)}, "a", retired_at=12345)

    restored = parse_registry(render_registry(entries))

    assert restored == entries
    assert total_cap(restored) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", ["0.99", None, True])
def test_a_non_numeric_price_parameter_is_refused(bad):
    entry = replace(_entry("a", 0.5), params={**PARAMS, "cap": 0.5, "decay": bad})

    with pytest.raises(RegistryError, match="decay"):
        add_task({}, entry)


def test_a_fractional_round_count_is_refused():
    entry = replace(
        _entry("a", 0.5), params={**PARAMS, "cap": 0.5, "rounds_per_step": 10.5}
    )

    with pytest.raises(RegistryError, match="rounds_per_step"):
        add_task({}, entry)


def test_an_infinite_retired_at_is_a_registry_error():
    with pytest.raises(RegistryError, match="retired_at"):
        retire_task({"a": _entry("a", 0.5)}, "a", retired_at=float("inf"))


def test_a_fractional_retired_at_in_the_file_is_refused():
    raw = json.dumps({
        "registry_version": 1,
        "tasks": {"a": {
            "profile_id": "p",
            "profile_sha256": "a" * 64,
            "incentive": {
                "mechanism": MECHANISM_RL_DISCOVERED_PRICE,
                "params": {**PARAMS, "cap": 0.5},
            },
            "status": "retired",
            "retired_at": 1.5,
        }},
    }).encode()

    with pytest.raises(RegistryError, match="retired_at"):
        parse_registry(raw)


def test_a_number_too_large_for_a_float_is_refused():
    huge = int("9" * 400)
    entry = replace(_entry("a", 0.5), params={**PARAMS, "cap": 0.5, "decay": huge})

    with pytest.raises(RegistryError, match="decay"):
        add_task({}, entry)


def test_an_oversized_cap_is_refused_when_summing():
    entry = replace(_entry("a", 0.5), params={**PARAMS, "cap": int("9" * 400)})

    with pytest.raises(RegistryError, match="cap"):
        total_cap({"a": entry})


def test_summing_an_entry_with_no_cap_is_a_registry_error():
    entry = replace(
        _entry("a", 0.5), params={k: v for k, v in PARAMS.items() if k != "cap"}
    )

    with pytest.raises(RegistryError, match="cap"):
        total_cap({"a": entry})
