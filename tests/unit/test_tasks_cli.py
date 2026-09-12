"""Launching a task is writing one entry, and the defaults come from the image."""

from __future__ import annotations

import pytest

from reliquary.cli.main import build_task_entry
from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS


def test_defaults_come_from_the_shipped_controller():
    entry = build_task_entry(
        task_id="logic-probe",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=0.25,
        overrides={},
    )

    assert entry.params["decay"] == PRODUCTION_PRICE_PARAMS.decay
    assert entry.params["median_rounds"] == PRODUCTION_PRICE_PARAMS.median_rounds
    assert entry.params["cap"] == 0.25
    assert entry.status == "active"
    assert len(entry.profile_sha256) == 64


def test_an_override_replaces_one_parameter_only():
    entry = build_task_entry(
        task_id="logic-probe",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=0.25,
        overrides={"start": 0.2},
    )

    assert entry.params["start"] == 0.2
    assert entry.params["decay"] == PRODUCTION_PRICE_PARAMS.decay


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="unknown protocol profile"):
        build_task_entry(
            task_id="logic-probe", profile_id="no-such-profile",
            cap=0.25, overrides={},
        )


def test_a_built_entry_passes_registry_validation():
    from reliquary.shared.task_registry import validate_entry

    entry = build_task_entry(
        task_id="logic-probe",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=0.25,
        overrides={"start": 0.2, "decay": 0.95},
    )

    validate_entry(entry)


def test_list_reads_a_registry_whose_declared_caps_exceed_one(monkeypatch):
    """``tasks list`` exists to let an operator see a broken registry so they
    can repair it, so it must not itself raise on an oversubscribed one."""
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry

    params = {
        "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
        "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.7,
        "median_rounds": 4800,
    }
    oversubscribed = {
        "a": TaskEntry(
            task_id="a", profile_id="qwen3-4b-base-dapo-fill-closed-v6",
            profile_sha256="a" * 64, mechanism=MECHANISM_RL_DISCOVERED_PRICE,
            params=params, status="active", retired_at=None,
        ),
        "b": TaskEntry(
            task_id="b", profile_id="qwen3-4b-base-dapo-fill-closed-v6",
            profile_sha256="b" * 64, mechanism=MECHANISM_RL_DISCOVERED_PRICE,
            params={**params, "cap": 0.6}, status="active", retired_at=None,
        ),
    }

    async def _fake_read_registry(**kwargs):
        assert kwargs.get("strict") is False
        return oversubscribed, '"etag"'

    monkeypatch.setattr(
        "reliquary.infrastructure.task_registry_store.read_registry",
        _fake_read_registry,
    )

    result = CliRunner().invoke(app, ["tasks", "list"])

    assert result.exit_code == 0, result.output
    assert "a" in result.output and "b" in result.output
    assert "1.3000" in result.output


# --- The id an operator types becomes the registry KEY. ---

def test_a_padded_task_id_is_normalised_before_it_becomes_a_key():
    """`" default"` keyed under the raw string validates but can never be
    looked up again, so the padding dies here, at the one place typing
    becomes an entry."""
    entry = build_task_entry(
        task_id=" default",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=1.0,
        overrides={},
    )

    assert entry.task_id == "default"


def test_an_unusable_task_id_is_refused_by_the_builder():
    with pytest.raises(ValueError, match="unusable task id"):
        build_task_entry(
            task_id="Logic Probe",
            profile_id="qwen3-4b-base-dapo-fill-closed-v6",
            cap=0.25, overrides={},
        )


def test_a_non_canonical_entry_can_never_be_written():
    """The builder normalises, but the registry refuses anyway: a
    hand-constructed entry does not get a different rule."""
    from dataclasses import replace

    from reliquary.shared.task_registry import RegistryError, add_task

    entry = replace(
        build_task_entry(
            task_id="default",
            profile_id="qwen3-4b-base-dapo-fill-closed-v6",
            cap=1.0, overrides={},
        ),
        task_id=" default",
    )

    with pytest.raises(RegistryError, match="canonical"):
        add_task({}, entry)


# --- Declaring the first task is the one CLI command that can stop the fleet. ---

def _fake_store(monkeypatch, state):
    """Wire `tasks create` onto an in-memory registry."""
    from reliquary.infrastructure import task_registry_store as store

    async def _read(**kwargs):
        return dict(state["entries"]), state["etag"]

    async def _write(entries, etag, **kwargs):
        from reliquary.shared.task_registry import validate_registry

        validate_registry(entries)
        state["entries"] = dict(entries)
        state["etag"] = '"v2"'
        return state["etag"]

    monkeypatch.setattr(store, "read_registry", _read)
    monkeypatch.setattr(store, "write_registry", _write)


def test_declaring_a_non_default_task_first_is_refused_by_the_cli(monkeypatch):
    """`reliquary tasks create --task-id foo` is the first command anyone
    runs. On an empty registry it must not write: from that moment every
    training validator exits 4 on restart and every submitter abstains."""
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    state = {"entries": {}, "etag": None}
    _fake_store(monkeypatch, state)

    result = CliRunner().invoke(app, [
        "tasks", "create", "--task-id", "logic-probe",
        "--profile-id", "qwen3-4b-base-dapo-fill-closed-v6", "--cap", "0.3",
    ])

    assert result.exit_code == 1, result.output
    assert "default" in (result.output + str(result.exception))
    assert state["entries"] == {}


def test_declaring_default_then_a_second_task_both_succeed(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    state = {"entries": {}, "etag": None}
    _fake_store(monkeypatch, state)
    runner = CliRunner()

    first = runner.invoke(app, [
        "tasks", "create", "--task-id", "default",
        "--profile-id", "qwen3-4b-base-dapo-fill-closed-v6", "--cap", "0.7",
    ])
    second = runner.invoke(app, [
        "tasks", "create", "--task-id", "logic-probe",
        "--profile-id", "qwen3-4b-base-dapo-fill-closed-v6", "--cap", "0.3",
    ])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert set(state["entries"]) == {"default", "logic-probe"}


# --- A transient R2 error at startup is not a boot failure. ---

def test_the_startup_registry_read_retries_a_raising_client():
    import asyncio

    from reliquary.cli.main import read_task_registry_with_retry

    calls = {"n": 0}

    async def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return {"default": object()}, '"etag"'

    entries, etag = asyncio.run(
        read_task_registry_with_retry(_flaky, attempts=4, backoff_seconds=0.0)
    )

    assert calls["n"] == 3
    assert set(entries) == {"default"}


def test_the_startup_registry_read_still_refuses_after_the_bound():
    import asyncio

    from reliquary.cli.main import read_task_registry_with_retry

    calls = {"n": 0}

    async def _dead():
        calls["n"] += 1
        raise RuntimeError("503 Service Unavailable")

    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(
            read_task_registry_with_retry(_dead, attempts=3, backoff_seconds=0.0)
        )

    assert calls["n"] == 3


def test_an_absent_registry_is_not_an_error_and_is_never_retried():
    """`read_registry` reports an absent object as ({}, None), which is the
    legacy fallback's own signal — retrying it would add 14s to every boot
    of every validator running today."""
    import asyncio

    from reliquary.cli.main import read_task_registry_with_retry

    calls = {"n": 0}

    async def _absent():
        calls["n"] += 1
        return {}, None

    entries, etag = asyncio.run(
        read_task_registry_with_retry(_absent, attempts=4, backoff_seconds=0.0)
    )

    assert (entries, etag) == ({}, None)
    assert calls["n"] == 1
