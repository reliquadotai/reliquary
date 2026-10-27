import gzip
import io
import json
import zipfile

import numpy as np
import pytest

from reliquary.integrations.affine_scoring_contract import (
    create_fixture,
    validate_inputs,
    validate_output,
)


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "inputs"
    prepared = create_fixture(root)
    output = prepared.pop("expectedNativeOutput")
    return root, prepared, output


def write(path, value):
    path.write_text(json.dumps(value))


def document(root):
    with gzip.open(root / "docs.jsonl.gz", "rt") as stream:
        return json.loads(stream.readline())


def documents(root, rows):
    with gzip.open(root / "docs.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_fixture_native_golden_and_safe_summary(native, tmp_path):
    root, job, output = native
    context = validate_inputs(root, job)
    # Golden from the pinned upstream pure build_chunks function, not its signer.
    assert context["chunks"] == [
        dict(kind="subject", subject="Algebra", half=0, c=0, tasks=[7, 6, 1, 2]),
        dict(kind="subject", subject="Algebra", half=1, c=0, tasks=[0, 4, 5, 3]),
    ]
    assert context["outputDimensions"] == dict(
        documents=1, rolloutsPerDocument=8, chunks=2, referenceTasks=8
    )
    path = tmp_path / "dots-0.json"
    write(path, output)
    summary = validate_output(path, job, context)
    assert summary["outputValidated"] and summary["assurance"] == "fixture"
    assert summary["documents"] == 1 and summary["rollouts"] == 8
    assert not summary["upstreamAccepted"] and not summary["nativeRewardVerified"]
    assert (
        not summary["chainSeedVerified"]
        and not summary["independentNumericalVerification"]
    )
    assert "a" * 16 not in json.dumps(summary)
    assert (root / "ref/gen-0.npz").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", [dict(shard=1), dict(nshards=2), dict(layers=3)])
def test_refuses_unsupported_partition(native, change):
    root, job, _ = native
    with pytest.raises(ValueError):
        validate_inputs(root, dict(job, **change))


@pytest.mark.parametrize(
    "field,value", [("margin_blocks", 9), ("seed", "0" * 64), ("anchor_ts_ms", 1)]
)
def test_seed_binding_and_canonical_margin(native, field, value):
    root, job, _ = native
    path = root / "seed.json"
    seed = json.loads(path.read_text())
    seed[field] = value
    write(path, seed)
    with pytest.raises(ValueError):
        validate_inputs(root, job)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "length", "base", "subject"]
)
def test_reference_coverage_and_native_binding(native, change):
    root, job, _ = native
    path = root / "ref/gen-0.json"
    ref = json.loads(path.read_text())
    if change == "duplicate":
        ref["meta"][1]["j"] = 0
    if change == "missing":
        ref["meta"].pop()
    if change == "length":
        ref["meta"][0]["len"] += 1
    if change == "base":
        ref["base"] = "0" * 64
    if change == "subject":
        ref["meta"][0]["subject"] = "Other"
    write(path, ref)
    with pytest.raises(ValueError):
        validate_inputs(root, job)


def test_object_arrays_never_deserialized(native):
    root, job, _ = native
    np.savez(
        root / "ref/gen-0.npz",
        tokens=np.asarray([object()], dtype=object),
        offsets=np.asarray([0, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="integer NPZ"):
        validate_inputs(root, job)


def test_huge_npy_header_rejected_before_allocation(native):
    root, job, _ = native
    bad = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        bad, dict(descr="<i4", fortran_order=False, shape=(1 << 40,))
    )
    offsets = io.BytesIO()
    np.save(offsets, np.asarray([0], dtype=np.int64))
    with zipfile.ZipFile(root / "ref/gen-0.npz", "w") as archive:
        archive.writestr("tokens.npy", bad.getvalue())
        archive.writestr("offsets.npy", offsets.getvalue())
    with pytest.raises(ValueError, match="shape/size"):
        validate_inputs(root, job)


def test_identity_prefix_collision_even_outside_todo(native):
    root, job, _ = native
    first = document(root)
    second = dict(first, miner_full="a" * 16 + "d" * 48, slot=1)
    documents(root, [first, second])
    with pytest.raises(ValueError, match="prefix collision"):
        validate_inputs(root, job)


@pytest.mark.parametrize("change", ["missing", "duplicate", "bad-rollouts", "context"])
def test_document_scope_and_shape(native, change):
    root, job, _ = native
    row = document(root)
    if change == "missing":
        write(root / "todo.json", [["a" * 16, 1]])
    if change == "duplicate":
        documents(root, [row, row])
    if change == "bad-rollouts":
        row["rollouts"].pop()
        documents(root, [row])
    if change == "context":
        row["prompt"] = [1] * 12288
        documents(root, [row])
    with pytest.raises(ValueError):
        validate_inputs(root, job)


@pytest.mark.parametrize(
    "change",
    [
        "epoch-int",
        "coverage",
        "identity",
        "lens",
        "dimensions",
        "nan",
        "bool",
        "chunks",
        "timings",
    ],
)
def test_result_binding_and_finite_dimensions(native, tmp_path, change):
    root, job, output = native
    context = validate_inputs(root, job)
    if change == "epoch-int":
        output["epoch"] = job["epoch"]
    if change == "coverage":
        output["docs"] = []
    if change == "identity":
        output["docs"][0]["miner"] = "d" * 16
    if change == "lens":
        output["docs"][0]["lens"][0] = 99
    if change == "dimensions":
        output["docs"][0]["dots"].pop()
    if change == "nan":
        output["docs"][0]["dots"][0][0] = float("nan")
    if change == "bool":
        output["docs"][0]["dots"][0][0] = True
    if change == "chunks":
        output["chunks"][0]["tasks"] = [0, 1, 2, 3]
    if change == "timings":
        output["timings"]["load"] = float("inf")
    path = tmp_path / "output.json"
    write(path, output)
    with pytest.raises(ValueError):
        validate_output(path, job, context)


def test_context_cannot_be_reused_for_changed_job(native, tmp_path):
    root, job, output = native
    context = validate_inputs(root, job)
    path = tmp_path / "output.json"
    write(path, output)
    with pytest.raises(ValueError, match="validated job changed"):
        validate_output(path, dict(job, assurance="historical-v9-compute"), context)


def test_symlink_output_rejected(native, tmp_path):
    root, job, output = native
    context = validate_inputs(root, job)
    path = tmp_path / "output.json"
    write(path, output)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="ordinary input"):
        validate_output(link, job, context)
