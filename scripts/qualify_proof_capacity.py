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
      "rollout_count":"<active profile rollout count>",
      "completion_token_lengths":[... one value per rollout ...]
    }

Use only end-to-end validator proofs generated against the exact release
candidate and model revision. Synthetic forward-pass estimates are not valid qualification evidence alone.
With --stress-samples, schema 4 combines genuine natural E2E groups with
separately labeled measured full-context GPU/transport/post-proof CPU work.
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
    MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
    M_ROLLOUTS,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
)
from reliquary.validator.proof_capacity import (  # noqa: E402
    ProofCapacityQualification,
    capacity_budget,
    compute_proof_path_hash,
)


ENVIRONMENTS = tuple(MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)
ROLLOUTS_PER_PROOF = M_ROLLOUTS
MINIMUM_CAP_FRACTION = 0.9
MINIMUM_SAMPLES_PER_DEVICE_PER_ENVIRONMENT = 20


def _p95(values: list[float]) -> float:
    if len(values) < MINIMUM_SAMPLES_PER_DEVICE_PER_ENVIRONMENT:
        raise ValueError(
            "at least 20 proof samples are required per GPU and environment"
        )
    return float(quantiles(values, n=100, method="inclusive")[94])


def _load_samples(
    path: Path,
    *,
    software_revision: str,
    checkpoint_revision: str,
    runtime_fingerprint_hash: str,
    hardware_class: str,
    benchmark_device_count: int,
    remote_proof: dict | None = None,
    natural_completions: bool = False,
) -> tuple[
    dict[str, dict[str, list[float]]],
    tuple[str, ...],
    dict[str, int],
]:
    samples = {
        environment: {} for environment in ENVIRONMENTS
    }
    minimum_completion_tokens = {
        environment: (MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment] if natural_completions else math.ceil(
            MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]
            * MINIMUM_CAP_FRACTION
        ))
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
            "runtime_fingerprint_hash": runtime_fingerprint_hash,
            "hardware_class": hardware_class,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"{field} mismatch at line {line_number}"
                )
        if remote_proof is not None and row.get("remote_proof") != remote_proof:
            raise ValueError(f"remote proof measurement mismatch at line {line_number}")
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
                or length < (1 if natural_completions else minimum_completion_tokens[environment])
                or length
                > MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]
                for length in completion_lengths
            )
        ):
            raise ValueError(
                "completion_token_lengths are not representative at "
                f"line {line_number}"
            )
        if natural_completions:
            minimum_completion_tokens[environment] = min(minimum_completion_tokens[environment], *completion_lengths)
        device_uuid = str(row.get("device_uuid", "")).strip().casefold()
        if not device_uuid:
            raise ValueError(
                f"missing device_uuid at line {line_number}"
            )
        samples[environment].setdefault(device_uuid, []).append(seconds)

    benchmark_devices = set().union(
        *(set(device_samples) for device_samples in samples.values())
    )
    if len(benchmark_devices) != benchmark_device_count:
        raise ValueError(
            "observed GPU UUID count differs from --benchmark-device-count"
        )
    for environment, device_samples in samples.items():
        if set(device_samples) != benchmark_devices:
            raise ValueError(
                f"{environment} did not exercise every benchmark GPU"
            )
        for device_uuid, values in device_samples.items():
            if len(values) < MINIMUM_SAMPLES_PER_DEVICE_PER_ENVIRONMENT:
                raise ValueError(
                    f"{environment} GPU {device_uuid} has fewer than 20 samples"
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
    parser.add_argument("--runtime-fingerprint-hash", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--benchmark-device-count", type=int, required=True)
    parser.add_argument("--measured-at", required=True)
    parser.add_argument("--stress-samples", type=Path,
                        help="Explicit v4: real passing natural groups plus separately measured full-envelope stress")
    parser.add_argument("--natural-corpus", type=Path, help="Original signed corpus retained by combined-natural measurement")
    parser.add_argument("--maximum-context-tokens", type=int,
                        help="Exact adopted model context; required for combined v4 and checked again at runtime")
    parser.add_argument("--headroom", type=float, default=0.2)
    parser.add_argument("--remote-proof-worker-id",
                        help="Require every source sample to cover validator end-to-end mTLS proof execution")
    args = parser.parse_args()

    if PROTOCOL_VERSION < 3:
        parser.error(
            "select a protocol profile with proof-capacity qualification"
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
    if (
        len(args.runtime_fingerprint_hash) != 64
        or args.runtime_fingerprint_hash
        != args.runtime_fingerprint_hash.lower()
        or any(
            character not in "0123456789abcdef"
            for character in args.runtime_fingerprint_hash
        )
    ):
        parser.error(
            "--runtime-fingerprint-hash must be lowercase SHA-256"
        )

    remote_proof = None
    if args.remote_proof_worker_id:
        from reliquary.validator.remote_proof_protocol import RemoteProofMeasurement, transport_hash
        remote_proof = RemoteProofMeasurement(
            worker_id=args.remote_proof_worker_id, transport_sha256=transport_hash(),
        ).model_dump()
    if args.stress_samples and (remote_proof is None or not args.maximum_context_tokens or not args.natural_corpus):
        parser.error("combined v4 requires --remote-proof-worker-id, --maximum-context-tokens and --natural-corpus")
    source_payload = args.samples.read_bytes()
    samples, benchmark_device_uuids, minimum_completion_tokens = _load_samples(
        args.samples,
        software_revision=args.software_revision,
        checkpoint_revision=args.checkpoint_revision,
        runtime_fingerprint_hash=args.runtime_fingerprint_hash,
        hardware_class=args.hardware_class,
        benchmark_device_count=args.benchmark_device_count,
        remote_proof=remote_proof,
        natural_completions=args.stress_samples is not None,
    )
    p95_by_environment_and_device = {
        environment: {
            device_uuid: _p95(values)
            for device_uuid, values in device_samples.items()
        }
        for environment, device_samples in samples.items()
    }
    p95_by_environment = {
        environment: max(device_p95.values())
        for environment, device_p95 in (
            p95_by_environment_and_device.items()
        )
    }
    sample_count_by_environment = {
        environment: sum(
            len(values) for values in samples[environment].values()
        )
        for environment in ENVIRONMENTS
    }
    sample_count_by_environment_and_device = {
        environment: {
            device_uuid: len(values)
            for device_uuid, values in device_samples.items()
        }
        for environment, device_samples in samples.items()
    }
    budget = capacity_budget()
    proofs_per_environment = {environment: budget["proofs_per_environment"] for environment in ENVIRONMENTS}
    manifest = {
        "schema_version": 3,
        "profile_id": PROTOCOL_PROFILE_ID,
        "model_revision": PROTOCOL_MODEL_REVISION,
        "software_revision": args.software_revision,
        "checkpoint_revision": args.checkpoint_revision,
        "samples_sha256": hashlib.sha256(source_payload).hexdigest(),
        "runtime_fingerprint_hash": args.runtime_fingerprint_hash,
        "hardware_class": args.hardware_class,
        "benchmark_device_count": args.benchmark_device_count,
        "benchmark_device_uuids": list(benchmark_device_uuids),
        "proof_wall_seconds": budget["wall_seconds"],
        "headroom_fraction": args.headroom,
        "proofs_per_environment": proofs_per_environment,
        "p95_seconds_per_proof": p95_by_environment,
        "p95_seconds_per_proof_by_environment_and_device": (
            p95_by_environment_and_device
        ),
        "sample_count_by_environment": sample_count_by_environment,
        "sample_count_by_environment_and_device": (
            sample_count_by_environment_and_device
        ),
        "minimum_samples_per_device_per_environment": (
            MINIMUM_SAMPLES_PER_DEVICE_PER_ENVIRONMENT
        ),
        "minimum_completion_tokens_by_environment": (
            minimum_completion_tokens
        ),
        "measured_at": args.measured_at,
        "qualified": True,
        # Content hash of the proof path at benchmark time: lets a later
        # image deploy that does not touch the proof path carry this
        # qualification over instead of re-benchmarking.
        "proof_path_hash": compute_proof_path_hash(),
    }
    if remote_proof is not None:
        manifest["remote_proof"] = remote_proof
    if args.stress_samples is not None:
        from reliquary.validator.proof_capacity_combined import load_stress_samples
        manifest["schema_version"] = 4
        manifest["combined_evidence"] = load_stress_samples(
            args.stress_samples, natural_path=args.samples, corpus_path=args.natural_corpus,
            expected_identity={key: manifest[key] for key in (
                "profile_id", "model_revision", "software_revision", "checkpoint_revision",
                "runtime_fingerprint_hash", "hardware_class")},
            device_uuids=benchmark_device_uuids,
            completion_caps=MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
            maximum_context_tokens=args.maximum_context_tokens, remote_proof=remote_proof,
            natural_samples=samples, natural_samples_sha256=manifest["samples_sha256"])

    qualification = ProofCapacityQualification.from_mapping(manifest)
    # Validate the benchmark fleet itself. A manifest that already needs more
    # devices than were exercised is not qualification evidence.
    qualification.validate(
        profile_id=PROTOCOL_PROFILE_ID,
        model_revision=PROTOCOL_MODEL_REVISION,
        software_revision=args.software_revision,
        checkpoint_revision=args.checkpoint_revision,
        runtime_fingerprint_hash=args.runtime_fingerprint_hash,
        configured_devices=tuple(
            f"cuda:{index}"
            for index in range(args.benchmark_device_count)
        ),
        configured_hardware=tuple(
            args.hardware_class
            for _index in range(args.benchmark_device_count)
        ),
        configured_device_uuids=benchmark_device_uuids,
        proof_wall_seconds=budget["wall_seconds"],
        minimum_proofs_per_environment=budget["proofs_per_environment"],
        minimum_completion_tokens_per_environment=minimum_completion_tokens,
        configured_slots=manifest.get("combined_evidence", {}).get("configured_slots"),
        maximum_context_tokens=args.maximum_context_tokens,
        full_completion_tokens_per_environment=MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
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
