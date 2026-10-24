"""RL service tasks run on service-contract/v2 only; legacy bytes stay identical."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared.task_registry import (
    MECHANISM_SERVICE_RL, RegistryError, TaskEntry, parse_registry, render_registry, validate_entry,
)
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2_dict
from tests.unit.test_task_registry_contract import PARAMS, _entry


def _service_entry(contract: dict, **overrides) -> TaskEntry:
    generation = {"model_id": contract["checkpoint"]["repo"], "model_revision": contract["checkpoint"]["revision"],
                  "environments": {name: {} for name in (MATH, CODE)}}
    contract = {**contract, "generation_contract_sha256": canonical_sha256(generation)}
    entry = _entry(task_id="next-rl", profile_id="teutonic-test", profile_sha256=contract["generation_contract_sha256"],
                   mechanism=MECHANISM_SERVICE_RL, contract=generation, service_contract=contract,
                   params={**PARAMS, "cap": 0.5, "floor": 0.5, "start": 0.5,
                           "min_incentive_share": 0.0, "min_incentive_ramp_start": 0.0})
    return replace(entry, **overrides)


def service_entry():
    return _service_entry(contract_v2_dict())


def test_v2_service_entry_validates():
    validate_entry(_service_entry(contract_v2_dict()))


def test_service_registry_roundtrip_requires_explicit_economic_and_generation_binding():
    entry = service_entry()
    validate_entry(entry)
    assert parse_registry(render_registry({entry.task_id: entry}))[entry.task_id] == entry
    with pytest.raises(RegistryError, match="floor"):
        validate_entry(replace(entry, params=PARAMS))
    with pytest.raises(RegistryError, match="explicit service"):
        validate_entry(replace(entry, mechanism="rl-discovered-price"))
    bad = deepcopy(entry.service_contract)
    bad["checkpoint"]["revision"] = "f" * 40
    with pytest.raises(RegistryError, match="checkpoint"):
        validate_entry(replace(entry, service_contract=bad))
    legacy = _entry()
    assert b'"service_contract"' not in render_registry({legacy.task_id: legacy})


def test_v1_adaptive_training_is_refused_for_rl():
    v1 = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    v1["service_kind"] = "adaptive_training"
    v1["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    with pytest.raises(RegistryError, match="service-contract/v2"):
        validate_entry(_service_entry(contract_v2_dict(), service_contract=v1))


def test_service_entry_refuses_env_split():
    with pytest.raises(RegistryError, match="env_split"):
        validate_entry(_service_entry(contract_v2_dict(), env_split={MATH: 0.5, CODE: 0.5}))


def test_every_contract_env_must_be_in_the_generation_contract():
    contract = contract_v2_dict(envs=(MATH, "reliquary_logic_v2"), shares={MATH: 5000, "reliquary_logic_v2": 5000})
    with pytest.raises(RegistryError, match="absent from generation contract"):
        validate_entry(_service_entry(contract))
