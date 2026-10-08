import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.mapping import mapping_artifact
from reliquary.services.curation import curate_rows
from reliquary.services.observations import ObservationStore, observation_signal

CATALOG_CARD = {"source_kind": "catalog", "source": "reliquary_dapo_math_v1", "split": "train",
                "index_range": [0, 3], "set_id": "dapo-train-slice",
                "disjointness": {"external_benchmark": False, "held_out": []}}


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


def test_mapping_and_curated_slice_preserve_source_and_confidence():
    source = b'{"row_id":"row-1","prompt":"a"}\n{"row_id":"row-2","prompt":"b"}\n'
    c = contract(source)
    rows = [observation(c),observation(c,"row-2",rewards=(10000,10000))]
    body, manifest = mapping_artifact(c,rows,expected_rows=2)
    assert manifest["complete"] and manifest["generation_verified"] and not manifest["sampling_verified"]
    output, curated = curate_rows(source,body,manifest,c, set_card=CATALOG_CARD)
    assert output == source.splitlines(keepends=True)[0]
    assert curated["rows"] == 1 and not curated["sampling_verified"]
    assert source.count(b"prompt") == 2
    with pytest.raises(ValueError,match="sampling"):
        curate_rows(source,body,manifest,c,require_sampling=True, set_card=CATALOG_CARD)
    with pytest.raises(ValueError,match="digest"):
        curate_rows(source+b" ",body,manifest,c, set_card=CATALOG_CARD)
    bad = deepcopy(rows);bad[1]["rewards_bps"]=[None,10000]
    partial, partial_manifest = mapping_artifact(c,bad,expected_rows=2)
    assert not partial_manifest["complete"] and partial_manifest["category_counts"]["unknown"]==1
    with pytest.raises(ValueError,match="complete"):
        curate_rows(source,partial,partial_manifest,c, set_card=CATALOG_CARD)


def test_curation_rechecks_row_confidence_and_preserves_original_line_bytes():
    source = b'{"row_id":"row-1","prompt":"a"}\r\n{"row_id":"row-2","prompt":"b"}'
    c = contract(source)
    rows = [observation(c), observation(c,"row-2",rewards=(10000,10000))]
    body, manifest = mapping_artifact(c,rows,expected_rows=2)
    output, _ = curate_rows(source,body,manifest,c, set_card=CATALOG_CARD)
    assert output == source.splitlines(keepends=True)[0]
    rows[0]["verification"]["generation"] = "unverified"
    body, manifest = mapping_artifact(c,rows,expected_rows=2)
    manifest["generation_verified"] = True
    with pytest.raises(ValueError,match="every mapped row"):
        curate_rows(source,body,manifest,c, set_card=CATALOG_CARD)
