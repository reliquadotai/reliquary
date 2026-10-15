import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError, parse_service_contract

FIXTURE = Path(__file__).parents[1] / "fixtures" / "service_contract_v1.json"


def example():
    return json.loads(FIXTURE.read_text())


def test_order_snapshot_canonical_and_immutable():
    value = example()
    ordered = ServiceContract.from_dict(value)
    value["scoring"]["sigma_min_bps"] = 1
    assert ordered.to_dict()["scoring"]["sigma_min_bps"] == 2400
    assert parse_service_contract(ordered.canonical) == ordered
    with pytest.raises(ServiceContractError):
        parse_service_contract(json.dumps(example(), indent=2).encode())
    # No field is added to the historical generation/release serializers.
    assert "service_contract" not in ordered.to_dict()


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(unknown=True),
    lambda x: x.update(service_kind=[]),
    lambda x: x.update(visibility=[]),
    lambda x: x["limits"].update(max_tokens=True),
    lambda x: x["limits"].update(max_tokens=2**53),
    lambda x: x["checkpoint"].update(revision="main"),
    lambda x: x["scoring"].update(sigma_min_bps=2400.0),
    lambda x: x["scoring"]["weights_bps"].update(reward=9999),
    lambda x: x["policies"]["sampling"].update(kind="unknown/v1"),
    lambda x: x["policies"]["sampling"].update(kind=[]),
    lambda x: x["policies"]["reward"].update(kind="exploration-discount/v1",divisor=4,budget_bps=100,refresh_windows=10,max_tokens_per_group=1000),
])
def test_refuses_ambiguous_or_unsupported_contract(mutate):
    value = deepcopy(example())
    mutate(value)
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(value)


def test_rejects_duplicate_key_and_nonfinite_and_capability_mismatch():
    raw = canonical_json_bytes(example()).replace(b'"schema":', b'"schema":"service-contract/v1","schema":')
    with pytest.raises(ServiceContractError):
        parse_service_contract(raw)
    value = example()
    value["limits"]["max_tokens"] = float("nan")
    with pytest.raises(ServiceContractError):
        ServiceContract.from_dict(value)
    value = example()
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind":"trainer-driven/v1","task_scoped":1}
    contract = ServiceContract.from_dict(value)
    caps = {"environment-reward/v1", "legacy/v1", "all/v1", "static/v1", "trainer-driven/v1"}
    with pytest.raises(ServiceContractError, match="task-scoped/v1"):
        contract.require_capabilities(caps)
    contract.require_capabilities(caps | {"task-scoped/v1"})
