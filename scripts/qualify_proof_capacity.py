#!/usr/bin/env python3
"""Build an immutable proof-capacity manifest from staging measurements.

Input is JSONL with one completed proof per row:

    {
      "environment":"openmathinstruct",
      "seconds":61.2,
      "proof_passed":true,
      "profile_id":"qwen35-4b-auction-v3",
      "model_revision":"<40-char SHA>",
      "software_revision":"<40-char SHA>",
      "checkpoint_revision":"<40-char SHA>",
      "hardware_class":"NVIDIA H100 80GB HBM3",
      "device_uuid":"GPU-...",
      "rollout_count":8,
      "completion_token_lengths":[... eight values ...]
    }

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

from reliquary.constants import (  # noqa: E402
    FORENSIC_SAMPLE_PER_WINDOW,
    MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
    MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    MAX_PROOF_WALL_SECONDS,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
)
from reliquary.validator.proof_capacity import (  # noqa: E402
    ProofCapacityQualification,
)


ENVIRONMENTS = ("openmathinstruct", "opencodeinstruct")
ROLLOUTS_PER_PROOF = 8
MINIMUM_CAP_FRACTION = 0.9


def _p95(values: list[float]) -> float:
    if len(values) < 20:
        raise ValueError("at least 20 proof samples are required per environment")
    return float(quantiles(values, n=100, method="inclusive")[94])


def _load_samples(
    path: Path,
    *,
    software_revision: str,
    checkpoint_revision: str,
    hardware_class: str,
    benchmark_device_count: int,
) -> tuple[
    dict[str, list[float]],
    tuple[str, ...],
    dict[str, int],
]:
    samples = {environment: [] for environment in ENVIRONMENTS}
    devices_by_environment = {
        environment: set() for environment in ENVIRONMENTS
    }
    minimum_completion_tokens = {
        environment: math.ceil(
            MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]
            * MINIMUM_CAP_FRACTION
        )
        for environment in ENVIRONMENTS
    }
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
        if row.get("proof_passed") is not True:
            raise ValueError(
                f"proof did not pass at line {line_number}"
            )
        expected_fields = {
            "profile_id": PROTOCOL_PROFILE_ID,
            "model_revision": PROTOCOL_MODEL_REVISION,
            "software_revision": software_revision,
            "checkpoint_revision": checkpoint_revision,
            "hardware_class": hardware_class,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"{field} mismatch at line {line_number}"
                )
        if row.get("rollout_count") != ROLLOUTS_PER_PROOF:
            raise ValueError(
                f"rollout_count mismatch at line {line_number}"
            )
        completion_lengths = row.get("completion_token_lengths")
        if (
            not isinstance(completion_lengths, list)
            or len(completion_lengths) != ROLLOUTS_PER_PROOF
            or any(
                not isinstance(length, int)
                or isinstance(length, bool)
                or length < minimum_completion_tokens[environment]
                or length
                > MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]
                for length in completion_lengths
            )
        ):
            raise ValueError(
                "completion_token_lengths are not representative at "
                f"line {line_number}"
            )
        device_uuid = str(row.get("device_uuid", "")).strip().casefold()
        if not device_uuid:
            raise ValueError(
                f"missing device_uuid at line {line_number}"
            )
        samples[environment].append(seconds)
        devices_by_environment[environment].add(device_uuid)

    benchmark_devices = set().union(*devices_by_environment.values())
    if len(benchmark_devices) != benchmark_device_count:
        raise ValueError(
            "observed GPU UUID count differs from --benchmark-device-count"
        )
    for environment, devices in devices_by_environment.items():
        if devices != benchmark_devices:
            raise ValueError(
                f"{environment} did not exercise every benchmark GPU"
            )
    return (
        samples,
        tuple(sorted(benchmark_devices)),
        minimum_completion_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--software-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--benchmark-device-count", type=int, required=True)
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--headroom", type=float, default=0.2)
    args = parser.parse_args()

    if PROTOCOL_VERSION < 3:
        parser.error(
            "set RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3"
        )
    for label, revision in (
        ("software", args.software_revision),
        ("checkpoint", args.checkpoint_revision),
    ):
        if (
            len(revision) != 40
            or revision != revision.lower()
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            parser.error(f"{label} revision must be a full lowercase commit SHA")
    if args.benchmark_device_count <= 0:
        parser.error("--benchmark-device-count must be positive")

    source_payload = args.samples.read_bytes()
    samples, benchmark_device_uuids, minimum_completion_tokens = _load_samples(
        args.samples,
        software_revision=args.software_revision,
        checkpoint_revision=args.checkpoint_revision,
        hardware_class=args.hardware_class,
        benchmark_device_count=args.benchmark_device_count,
    )
    p95_by_environment = {
        environment: _p95(samples[environment])
        for environment in ENVIRONMENTS
    }
    proofs_per_environment = {
        environment: (
            MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
            + FORENSIC_SAMPLE_PER_WINDOW
        )
        for environment in ENVIRONMENTS
    }
    manifest = {
        "schema_version": 2,
        "profile_id": PROTOCOL_PROFILE_ID,
        "model_revision": PROTOCOL_MODEL_REVISION,
        "software_revision": args.software_revision,
        "checkpoint_revision": args.checkpoint_revision,
        "samples_sha256": hashlib.sha256(source_payload).hexdigest(),
        "hardware_class": args.hardware_class,
        "benchmark_device_count": args.benchmark_device_count,
        "benchmark_device_uuids": list(benchmark_device_uuids),
        "proof_wall_seconds": MAX_PROOF_WALL_SECONDS,
        "headroom_fraction": args.headroom,
        "proofs_per_environment": proofs_per_environment,
        "p95_seconds_per_proof": p95_by_environment,
        "sample_count_by_environment": {
            environment: len(samples[environment])
            for environment in ENVIRONMENTS
        },
        "minimum_completion_tokens_by_environment": (
            minimum_completion_tokens
        ),
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
            f"cuda:{index}"
            for index in range(args.benchmark_device_count)
        ),
        configured_hardware=tuple(
            args.hardware_class
            for _index in range(args.benchmark_device_count)
        ),
        configured_device_uuids=benchmark_device_uuids,
        proof_wall_seconds=MAX_PROOF_WALL_SECONDS,
        minimum_proofs_per_environment=(
            MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
            + FORENSIC_SAMPLE_PER_WINDOW
        ),
        minimum_completion_tokens_per_environment=minimum_completion_tokens,
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
