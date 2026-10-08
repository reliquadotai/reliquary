"""New services require an explicit mechanism; legacy bytes stay identical."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_SERVICE_RL, RegistryError, parse_registry, render_registry, validate_entry
from tests.unit.test_task_registry_contract import CONTRACT, PARAMS, _entry


def service_entry():
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    generation = deepcopy(CONTRACT)
    generation.update(model_id=value["checkpoint"]["repo"], model_revision=value["checkpoint"]["revision"],
                      environments={value["environment"]["id"]: {}})
    value["generation_contract_sha256"] = canonical_sha256(generation)
    return _entry(task_id="service", contract=generation, service_contract=value, mechanism=MECHANISM_SERVICE_RL,
                  params={**PARAMS, "min_incentive_share": 0, "min_incentive_ramp_start": 0})


def test_service_registry_roundtrip_requires_explicit_economic_and_generation_binding():
    entry = service_entry()
    validate_entry(entry)
    assert parse_registry(render_registry({entry.task_id: entry}))[entry.task_id] == entry
    with pytest.raises(RegistryError, match="floor"):
        validate_entry(replace(entry, params=PARAMS))
    with pytest.raises(RegistryError, match="explicit service"):
        validate_entry(replace(entry, mechanism="rl-discovered-price"))
    bad = deepcopy(entry.service_contract); bad["checkpoint"]["revision"] = "f" * 40
    with pytest.raises(RegistryError, match="checkpoint"):
        validate_entry(replace(entry, service_contract=bad))
    legacy = _entry()
    assert b'"service_contract"' not in render_registry({legacy.task_id: legacy})
