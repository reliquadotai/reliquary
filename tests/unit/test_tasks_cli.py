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
