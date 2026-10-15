import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import ObservationStore, observation_signal


def contract(source=None):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    if source is not None:
        value["dataset"]["sha256"] = hashlib.sha256(source).hexdigest()
    return ServiceContract.from_dict(value)


def observation(c, row_id="row-1", rewards=(0, 10000), **kw):
    return {"schema":"prompt-observation/v1","context_sha256":c.context_sha256,
            "row_id":row_id,"group_id":"group-0","expected_samples":2,
            "sample_ids":[f"sample-{i}" for i in range(len(rewards))],"rewards_bps":list(rewards),
            "tokens":[10]*len(rewards),"window":1,
            "verification":{"generation":"verified","sampling":"unverified","grading":"graded"},
            "source_sha256":"f"*64,**kw}


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
