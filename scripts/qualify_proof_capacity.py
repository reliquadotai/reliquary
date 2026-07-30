#!/usr/bin/env python3
"""Build an immutable proof-capacity manifest from staging measurements.

Input is JSONL with one completed proof per row:

    {"environment":"openmathinstruct","seconds":61.2}

Use only end-to-end validator proofs generated against the exact release
candidate and model revision. Synthetic forward-pass estimates are not valid
qualification evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import quantiles
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reliquary.constants import (
    B_BATCH,
    FORENSIC_SAMPLE_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_PROFILE_ID,
)
from reliquary.validator.proof_capacity import (
    ProofCapacityQualification,
)


ENVIRONMENTS = ("openmathinstruct", "opencodeinstruct")


def _p95(values: list[float]) -> float:
    if len(values) < 20:
        raise ValueError("at least 20 proof samples are required per environment")
    return float(quantiles(values, n=100, method="inclusive")[94])


def _load_samples(path: Path) -> dict[str, list[float]]:
    samples = {environment: [] for environment in ENVIRONMENTS}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            environment = str(row["environment"])
            seconds = float(row["seconds"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid sample at line {line_number}"
            ) from exc
        if environment not in samples:
            raise ValueError(
                f"unknown environment at line {line_number}: {environment}"
            )
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                f"invalid proof duration at line {line_number}"
            )
        samples[environment].append(seconds)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software-revision", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--benchmark-device-count", type=int, required=True)
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--headroom", type=float, default=0.2)
    args = parser.parse_args()

    samples = _load_samples(args.samples)
    p95_by_environment = {
        environment: _p95(samples[environment])
        for environment in ENVIRONMENTS
    }
    proofs_per_environment = {
        environment: B_BATCH + FORENSIC_SAMPLE_PER_WINDOW
        for environment in ENVIRONMENTS
    }
    manifest = {
        "schema_version": 1,
        "profile_id": PROTOCOL_PROFILE_ID,
        "model_revision": PROTOCOL_MODEL_REVISION,
        "software_revision": args.software_revision,
        "hardware_class": args.hardware_class,
        "benchmark_device_count": args.benchmark_device_count,
        "proof_wall_seconds": MAX_PROOF_WALL_SECONDS,
        "headroom_fraction": args.headroom,
        "proofs_per_environment": proofs_per_environment,
        "p95_seconds_per_proof": p95_by_environment,
        "measured_at": args.measured_at,
        "qualified": True,
    }
    qualification = ProofCapacityQualification.from_mapping(manifest)
    # Validate the benchmark fleet itself. A manifest that already needs more
    # devices than were exercised is not qualification evidence.
    qualification.validate(
        profile_id=PROTOCOL_PROFILE_ID,
        model_revision=PROTOCOL_MODEL_REVISION,
        software_revision=args.software_revision,
        configured_devices=tuple(
            f"benchmark:{index}"
            for index in range(args.benchmark_device_count)
        ),
        configured_hardware=tuple(
            args.hardware_class
            for _index in range(args.benchmark_device_count)
        ),
        proof_wall_seconds=MAX_PROOF_WALL_SECONDS,
        minimum_proofs_per_environment=(
            B_BATCH + FORENSIC_SAMPLE_PER_WINDOW
        ),
    )

    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"path={args.output}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"p95={json.dumps(p95_by_environment, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
