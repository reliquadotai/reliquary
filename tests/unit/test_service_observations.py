from copy import deepcopy

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import ObservationStore, observation_signal
from tests.unit.service_v1_fixtures import contract, observation


def test_durable_idempotent_deltas_and_conflicting_evidence(tmp_path):
    c = contract()
    path = tmp_path / "journal.sqlite"
    store = ObservationStore(path,c)
    row = observation(c)
    identifier, created = store.append(row)
    assert created and not store.append(row)[1]
    watermark = store.snapshot()["watermark"]
    store.close()
    store = ObservationStore(path,c)
    assert list(store.deltas(after=watermark)) == []
    assert len(list(store.rows())) == 1
    other = deepcopy(row); other["rewards_bps"] = [0,0]
    with pytest.raises(ValueError,match="different evidence"):
        store.append(other)
    store.append(observation(c,"row-2",rewards=(None,0)))
    assert len(list(store.deltas(after=watermark))) == 1
    value = c.to_dict(); value["checkpoint"]["sha256"] = "a"*64
    changed = ServiceContract.from_dict(value)
    with pytest.raises(ValueError,match="context"):
        ObservationStore(path,changed).append(row)
    store.close()


def test_grader_error_does_not_become_zero_signal():
    c=contract()
    row=observation(c,rewards=(0,0));row["verification"]["grading"]="error"
    assert observation_signal(row,c).category == "unknown"
