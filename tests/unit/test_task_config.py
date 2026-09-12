"""A validator that cannot find itself in the registry does not run."""

from __future__ import annotations

import pytest

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
from reliquary.validator.task_config import TaskConfigError, resolve_task_config

CONTRACT = {"model_id": "demo", "environments": {}}
DIGEST = canonical_sha256(CONTRACT)
PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(**overrides) -> TaskEntry:
    base = dict(
        task_id="default",
        profile_id="demo-profile",
        profile_sha256=DIGEST,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=dict(PARAMS),
        status="active",
        retired_at=None,
    )
    return TaskEntry(**{**base, **overrides})


def _resolve(entries, task_id="default"):
    return resolve_task_config(
        entries, task_id, profile_id="demo-profile", generation_contract=CONTRACT
    )


def test_a_declared_task_yields_its_price_parameters():
    config = _resolve({"default": _entry()})

    assert config.emission_cap == 0.6
    assert config.price_params.cap == 0.6
    assert config.price_params.median_rounds == 4800


def test_an_undeclared_task_refuses():
    with pytest.raises(TaskConfigError, match="not declared"):
        _resolve({"other": _entry(task_id="other")}, task_id="default")


def test_an_oversubscribed_registry_refuses():
    entries = {
        "default": _entry(params={**PARAMS, "cap": 0.8}),
        "other": _entry(task_id="other", params={**PARAMS, "cap": 0.5}),
    }

    with pytest.raises(TaskConfigError, match="1.3"):
        _resolve(entries)


def test_a_profile_the_binary_does_not_match_refuses():
    with pytest.raises(TaskConfigError, match="profile"):
        _resolve({"default": _entry(profile_sha256="b" * 64)})


def test_a_profile_id_mismatch_refuses():
    with pytest.raises(TaskConfigError, match="demo-profile"):
        resolve_task_config(
            {"default": _entry()},
            "default",
            profile_id="another-profile",
            generation_contract=CONTRACT,
        )


def test_an_unknown_mechanism_refuses():
    with pytest.raises(TaskConfigError, match="vibes"):
        _resolve({"default": _entry(mechanism="vibes")})


def test_a_retired_task_refuses_to_start():
    with pytest.raises(TaskConfigError, match="retired"):
        _resolve({"default": _entry(status="retired")})
