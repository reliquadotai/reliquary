from copy import deepcopy

import pytest

from reliquary.services.mapping import mapping_artifact
from reliquary.services.curation import curate_rows
from tests.unit.service_v1_fixtures import contract, observation

CATALOG_CARD = {"source_kind": "catalog", "source": "reliquary_dapo_math_v1", "split": "train",
                "index_range": [0, 3], "set_id": "dapo-train-slice",
                "disjointness": {"external_benchmark": False, "held_out": []}}


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
