"""A restart resumes the price walk from the archives instead of starting over."""

from __future__ import annotations

import json

import pytest

from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS, restore_walk

PARAMS = PRODUCTION_PRICE_PARAMS

# (window_status, window readiness offset, per-environment readiness offsets).
# Mixes fast fills, a timed-out window, a starved environment, an environment
# missing from one window, and a slow fill that holds.
SCRIPT = [
    ("completed", 20, {"math": 20, "code": 15}),
    ("completed", 25, {"math": 25, "code": 10}),
    ("timed_out", 30, {"math": 30, "code": 12}),
    ("completed", None, {"math": None, "code": 14}),
    ("completed", 22, {"math": 22}),
    ("completed", 18, {"math": 18, "code": 11}),
    ("completed", 200, {"math": 200, "code": 190}),
    ("completed", 21, {"math": 21, "code": 9}),
]


def _stub_service():
    from reliquary.validator.service import ValidationService

    service = ValidationService.__new__(ValidationService)
    service._price_params = PARAMS
    return service


def _signal(index, ready, ready_by_environment):
    open_round = 1000 + index * 336
    return {
        "window_open_round": open_round,
        "window_close_round": open_round + 244,
        "collect_ready_round": None if ready is None else open_round + ready,
        "collect_ready_round_by_environment": {
            environment: None if offset is None else open_round + offset
            for environment, offset in ready_by_environment.items()
        },
    }


def _walk(service, script, start_index=0):
    """Run the live walk and return the archives it would have written."""
    records = []
    for offset, (status, ready, by_environment) in enumerate(script):
        index = start_index + offset
        signal = _signal(index, ready, by_environment)
        shadow = service._advance_price_shadow(signal, window_status=status)
        record = {"window_start": index, "window_status": status, **signal}
        if shadow:
            record["emission_price_shadow"] = shadow
        records.append(json.loads(json.dumps(record)))
    return records


def test_a_restored_walk_is_the_walk_that_was_interrupted():
    live = _stub_service()
    records = _walk(live, SCRIPT)

    walk = restore_walk(records, PARAMS)

    assert walk.state == live._price_shadow_state
    assert list(walk.history) == list(live._price_shadow_outcomes)
    assert walk.states_by_environment == live._price_shadow_states_by_environment
    assert {env: list(trail) for env, trail in walk.history_by_environment.items()} == {
        env: list(trail) for env, trail in live._price_shadow_outcomes_by_environment.items()
    }
    assert walk.latest_shadow == records[-1]["emission_price_shadow"]


def test_a_restarted_validator_prices_the_next_windows_as_if_it_never_stopped():
    uninterrupted = _stub_service()
    records = _walk(uninterrupted, SCRIPT)
    restarted = _stub_service()
    restarted._seed_price_walk(restore_walk(records, PARAMS))

    tail = [
        ("completed", 19, {"math": 19, "code": 13}),
        ("completed", None, {"math": 23, "code": None}),
        ("completed", 17, {"math": 17, "code": 16}),
    ]

    assert _walk(restarted, tail, start_index=len(SCRIPT)) == _walk(
        uninterrupted, tail, start_index=len(SCRIPT)
    )


def test_archives_without_a_price_start_a_fresh_walk():
    walk = restore_walk([{"window_start": 1, "window_status": "completed"}], PARAMS)

    assert walk.state is None
    assert walk.states_by_environment == {}
    assert list(walk.history) == []
    assert walk.latest_shadow is None


def test_an_aborted_archive_neither_moves_the_price_nor_joins_the_history():
    records = _walk(_stub_service(), SCRIPT[:2])
    aborted = json.loads(json.dumps(records[-1]))
    aborted.update(window_start=99, window_status="aborted")
    aborted["emission_price_shadow"].update(price=0.07, last_good=0.07)

    assert restore_walk(records + [aborted], PARAMS) == restore_walk(records, PARAMS)


@pytest.mark.parametrize("unreadable", ["n/a", True, None, [0.5], float("inf")])
def test_an_unreadable_archived_price_is_skipped_not_trusted(unreadable):
    records = _walk(_stub_service(), SCRIPT[:2])
    broken = json.loads(json.dumps(records[-1]))
    broken["emission_price_shadow"]["price"] = unreadable

    assert restore_walk([records[0], broken], PARAMS).state == (
        restore_walk([records[0]], PARAMS).state
    )


@pytest.mark.parametrize("archived, expected", [(7.0, PARAMS.cap), (-1.0, PARAMS.floor)])
def test_an_out_of_range_archived_price_is_clamped(archived, expected):
    records = _walk(_stub_service(), SCRIPT[:1])
    records[0]["emission_price_shadow"].update(price=archived, last_good=archived)

    assert restore_walk(records, PARAMS).state.price == expected


class _Server:
    _price_view = None


def _restorable_service(records, *, window_n):
    service = _stub_service()
    service._window_n = window_n
    service.server = _Server()
    service._fill_closed_assembler = None
    service._emission_cap = 1.0
    service.env_mix = [("math", 16), ("code", 16)]
    calls = []

    async def load_range(**kwargs):
        calls.append(kwargs)
        return records

    service._load_archive_range = load_range
    return service, calls


async def _failing_load_range(**_kwargs):
    raise RuntimeError("archive lookup failed")


@pytest.mark.asyncio
async def test_startup_resumes_the_walk_from_the_archived_windows():
    live = _stub_service()
    records = _walk(live, SCRIPT)
    service, calls = _restorable_service(records, window_n=len(SCRIPT) - 1)

    await service._restore_price_walk()

    assert service._price_shadow_state == live._price_shadow_state
    assert service._price_shadow_states_by_environment == live._price_shadow_states_by_environment
    assert service.server._price_view["value"] == records[-1]["emission_price_shadow"]["price"]
    assert calls[0]["end_window"] == len(SCRIPT) - 1
    assert calls[0]["require_all"] is False
    assert "emission_price_shadow" in calls[0]["fields"]


@pytest.mark.asyncio
async def test_a_validator_with_no_window_yet_has_nothing_to_restore():
    service, calls = _restorable_service([], window_n=0)

    await service._restore_price_walk()

    assert calls == []
    assert getattr(service, "_price_shadow_state", None) is None


@pytest.mark.asyncio
async def test_an_unreadable_history_refuses_startup_while_the_price_is_armed():
    """Starting over would hand miners back the starting pool."""
    service, _ = _restorable_service([], window_n=10)
    service._load_archive_range = _failing_load_range

    with pytest.raises(RuntimeError):
        await service._restore_price_walk()


@pytest.mark.asyncio
async def test_an_unreadable_history_only_restarts_the_walk_when_disarmed(monkeypatch):
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)
    service, _ = _restorable_service([], window_n=10)
    service._load_archive_range = _failing_load_range

    await service._restore_price_walk()

    assert getattr(service, "_price_shadow_state", None) is None
