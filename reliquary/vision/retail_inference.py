"""Local inference over frozen retail windows using the existing vision runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import time

from reliquary.vision import retail, runner, runtime
from reliquary.vision.data import load_manifest, prepare_jsonl


def infer(
    benchmark_path,
    model_dir,
    output,
    *,
    device,
    max_pixels=200704,
    max_context_tokens=8192,
    max_new_tokens=128,
    seed=17,
    max_seconds=300,
    checkpoint=None,
    checkpoint_sha256=None,
):
    """Verify local bytes; cooperative deadlines do not guarantee a host hard stop."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("an explicit cpu or cuda device is required")
    for value, low, high, name in (
        (max_pixels, 1, 4194304, "max_pixels"),
        (max_context_tokens, 32, 32768, "max_context_tokens"),
        (max_new_tokens, 1, 128, "max_new_tokens"),
        (seed, 0, 2**32 - 1, "seed"),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"invalid {name}")
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 3600
    ):
        raise ValueError("max_seconds must be finite and within (0,3600]")
    if (checkpoint is None) != (checkpoint_sha256 is None):
        raise ValueError("checkpoint and its SHA256 must be supplied together")
    raw, benchmark = retail._read(benchmark_path)
    retail._keys(
        benchmark,
        (
            "schema",
            "source_manifest_sha256",
            "max_frames",
            "labels",
            "recordings",
            "mode",
            "rights_assurance",
        ),
        "benchmark",
    )
    if (
        benchmark["schema"] != retail.BENCHMARK_SCHEMA
        or benchmark["mode"] != "timestamped_frames_v1"
    ):
        raise ValueError("unsupported benchmark")
    retail._integer(benchmark["max_frames"], 1, 64, "max_frames")
    value = retail._normalize(
        {
            "schema_version": 1,
            "labels": benchmark["labels"],
            "recordings": benchmark["recordings"],
        },
        Path(benchmark_path).parent,
    )
    if any(r["split"] != "test" for r in value["recordings"]):
        raise ValueError("inference benchmark must contain only final-test recordings")
    rows = [row for _, _, row in retail._windows(value, benchmark["max_frames"])]
    model_dir = Path(model_dir)
    source = runner.read_json(model_dir / "source.json", 1024 * 1024)
    runner._digest(source["revision"], runner.REVISION, "model revision")
    runner._digest(source["processorRevision"], runner.REVISION, "processor revision")
    profile = {
        "modelId": source["modelId"],
        "modelRevision": source["revision"],
        "environmentDigest": runner.checksum(model_dir / "training.py"),
        "vision": {
            "schemaVersion": 1,
            "processorRevision": source["processorRevision"],
            "processorDigest": runner.canonical_sha256(source["processorFiles"]),
            "mode": "timestamped_frames_v1",
            "task": "retail_events",
            "maxFrames": benchmark["max_frames"],
            "maxPixels": max_pixels,
            "maxSequenceLength": max_context_tokens,
        },
    }
    weights = runner.verify_model(model_dir, profile)
    source_sha = runner.checksum(model_dir / "source.json")
    if checkpoint is not None:
        runner._digest(checkpoint_sha256, runner.SHA256, "checkpoint SHA256")
        checkpoint = Path(checkpoint)
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or checkpoint.stat().st_size > 1024**3
        ):
            raise ValueError("checkpoint must be a bounded regular archive")
        if runner.checksum(checkpoint) != checkpoint_sha256:
            raise ValueError("checkpoint SHA256 mismatch")
    output = retail._private_parent(output)
    if os.path.lexists(output):
        raise ValueError("output exists; use a new directory")
    output.mkdir(mode=0o700)
    retail._write(output / "input.private.jsonl", retail._jsonl(rows))
    manifest_path = prepare_jsonl(output / "input.private.jsonl", output / "input")
    dataset = load_manifest(manifest_path)
    job = output / "job"
    job.mkdir(mode=0o700)
    adapter = None
    if checkpoint is not None:
        shutil.copyfile(checkpoint, job / "parent.checkpoint")
        (job / "parent.checkpoint").chmod(0o600)
        adapter = runner.load_parent(
            job, output, {"parent": {"sha256": checkpoint_sha256}}, profile
        )
    deadline = time.monotonic() + max_seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("local inference cooperative deadline exceeded")

    request = {
        "processor_mode": "timestamped_frames_v1",
        "method": "generate",
        "device": device,
        "precision": "float32" if device == "cpu" else "bf16",
        "seed": seed,
        "max_eval_examples": 256,
        "max_pixels": max_pixels,
        "max_context_tokens": max_context_tokens,
        "max_new_tokens": max_new_tokens,
        "model_id": source["modelId"],
        "model_revision": source["revision"],
    }
    try:
        report = runtime.run_job(
            request,
            dataset,
            manifest_path.parent,
            weights,
            output / "run",
            adapter,
            check_deadline=check_deadline,
        )
    finally:
        for item in output.rglob("*"):
            item.chmod(0o700 if item.is_dir() else 0o600)
    if report["steps"] != 0 or report["samples"] != len(rows):
        raise ValueError("inference unexpectedly trained or omitted windows")
    _, predictions = retail._read(output / "run" / "predictions.json")
    model = {
        "id": source["modelId"],
        "revision": source["revision"],
        "processor_revision": source["processorRevision"],
    }
    envelope = {
        "schema": retail.PREDICTION_SCHEMA,
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
        "model": model,
        "predictions": [
            {
                "window_id": row["id"],
                "status": "failed" if row["truncated"] else "succeeded",
                "completion": row["answer"],
                "latency_ms": row["latency_ms"],
            }
            for row in predictions
        ],
    }
    if (
        runner.checksum(model_dir / "source.json") != source_sha
        or Path(benchmark_path).read_bytes() != raw
    ):
        raise ValueError("frozen inference inputs changed")
    runner.verify_model(model_dir, profile)
    payload = retail._json(envelope)
    retail._write(output / "predictions.private.json", payload)
    evidence = {
        "schema": "retail-video-inference/v1",
        "benchmark_sha256": envelope["benchmark_sha256"],
        "model": model,
        "source_sha256": source_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "windows": len(rows),
        "device": device,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "Pillow")
        },
        "runtime_module_sha256": runner.checksum(Path(runtime.__file__)),
        "limits": {
            "max_pixels": max_pixels,
            "max_context_tokens": max_context_tokens,
            "max_new_tokens": max_new_tokens,
            "max_seconds": max_seconds,
            "seed": seed,
        },
        "model_identity_assurance": "caller-declared-local-bytes-verified",
        "deadline_assurance": "cooperative-generation-only-not-host-hard-deadline",
        "gpu_qualification": "not_run",
        "customer_acceptance": False,
    }
    retail._write(output / "inference.private.json", retail._json(evidence))
    for item in output.rglob("*"):
        item.chmod(0o700 if item.is_dir() else 0o600)
    return evidence
