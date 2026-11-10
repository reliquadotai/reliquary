import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_contract import ServiceContract, ServiceContractError

FIXTURE = Path(__file__).parents[1] / "fixtures" / "service_contract_v1.json"


def example():
    return json.loads(FIXTURE.read_text())


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


