"""Harmless CPU transport fixture; no model, gradients, registration or payment."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliquary.integrations.affine_scoring import (
    SCHEMA,
    SOURCE_REVISION,
    export,
    inventory,
    load_job,
    run,
    summary,
    verify,
    write_json,
)
from reliquary.integrations.affine_scoring_contract import create_fixture
from reliquary.protocol.release_contract import canonical_sha256


def execute(bundle, workspace):
    job, context = load_job(bundle)
    if job["assurance"] != "fixture":
        raise ValueError("CPU fixture cannot execute a native job")
    result = dict(
        epoch=str(job["epoch"]),
        base=job["baseCheckpoint"],
        seed=context["seed"],
        shard=0,
        chunks=context["chunks"],
        timings=dict(load=0.0, ref_grads=0.0, batch_dots=0.0),
        docs=[
            dict(row, dots=[[0.0] * len(context["chunks"]) for _ in range(8)])
            for row in context["expectedDocs"]
        ],
    )
    write_json(Path(workspace) / "native/dots-0.json", result)


def rehearse(destination):
    from reliquary.integrations.affine import _directory, _private_path

    destination = Path(destination)
    _private_path(destination)
    if destination.exists():
        raise ValueError("rehearsal requires a new private directory")
    _directory(destination)
    bundle = destination / "bundle"
    _directory(bundle)
    fields = create_fixture(bundle / "inputs")
    fields.pop("expectedNativeOutput")
    source = bundle / "source"
    _directory(source)
    for name in ("grad_docs_v9.py", "v9common.py"):
        path = source / name
        path.write_text(
            "# Synthetic transport fixture. Never executes native GPU code.\n"
        )
        path.chmod(0o600)
    (source / "data" / f"e{fields['epoch']}").mkdir(mode=0o700, parents=True)
    checkpoint = destination / "checkpoint"
    _directory(checkpoint)
    (checkpoint / "config.json").write_text('{"fixture":true}')
    (checkpoint / "config.json").chmod(0o600)
    runtime = dict(
        python="3.12",
        packages={},
        gpu=dict(name="none", sm=[0, 0]),
        upstreamRevision=SOURCE_REVISION,
    )
    job = dict(
        schema=SCHEMA,
        operation="gradient-dot",
        **fields,
        runtime=runtime,
        sourceFiles=inventory(source),
        inputFiles=inventory(bundle / "inputs"),
        checkpointFiles=inventory(checkpoint),
        limits=dict(
            maxSeconds=30,
            finalizationReserveSeconds=2,
            maxLogBytes=1024**2,
            maxOutputBytes=1024**2,
            maxWorkspaceBytes=16 * 1024**2,
        ),
    )
    for name, value in (
        ("sourceClosureDigest", job["sourceFiles"]),
        ("runtimeDigest", runtime),
        ("inputInventoryDigest", job["inputFiles"]),
        ("checkpointInventoryDigest", job["checkpointFiles"]),
    ):
        job[name] = canonical_sha256(value)
    job["jobDigest"] = canonical_sha256(job)
    write_json(bundle / "job.json", job)
    workspace = destination / "work"
    run(bundle, workspace, fixture=True)
    result = verify(bundle, workspace)
    artifact = export(bundle, workspace, destination / "scoring-result.zip")
    report = {
        **summary(job, "rehearsal_completed"),
        **{k: result[k] for k in ("documents", "rollouts", "referenceChunks")},
        "artifactSha256": artifact["artifactSha256"],
        "artifactBytes": artifact["artifactBytes"],
        "apiDeliveryVerified": False,
        "numericalComparisonPassed": False,
    }
    write_json(destination / "rehearsal-result.json", report)
    return report


if __name__ == "__main__":
    execute(Path(sys.argv[1]), Path(sys.argv[2]))
