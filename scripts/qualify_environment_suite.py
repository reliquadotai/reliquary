#!/usr/bin/env python3
"""Offline qualification for ``reliquaryverifiable_v1``.

The report is deterministic except for measured latency and the explicit
working-tree state.  It performs no network access and does not load a model.
Model frontier and release-host proof qualification remain separate canary
gates described in the runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.environment.records_tasks import (  # noqa: E402
    GENERATOR_VERSION,
)
from reliquary.environment.registry import get_environment_spec  # noqa: E402
from reliquary.environment.reliquaryverifiable import (  # noqa: E402
    ReliquaryVerifiableEnvironment,
)
from reliquary.protocol.profiles import resolve_protocol_profile  # noqa: E402


ENVIRONMENT = "reliquaryverifiable_v1"
DEFAULT_PROFILE = "qwen3-4b-reliquary-verifiable-v6-dev1"
FIXTURE = ROOT / "tests/fixtures/reliquaryverifiable_v1.jsonl"
MANIFEST = (
    ROOT
    / "reliquary/environment/manifests/reliquaryverifiable_v1.json"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(_percentile(values, 0.50), 6),
        "p95_ms": round(_percentile(values, 0.95), 6),
        "p99_ms": round(_percentile(values, 0.99), 6),
        "max_ms": round(max(values, default=0.0), 6),
    }


def _git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def qualify(*, sample_count: int, profile_id: str, allow_dirty: bool) -> dict:
    if sample_count < 100:
        raise ValueError("sample_count must be at least 100")

    environment = ReliquaryVerifiableEnvironment()
    profile = resolve_protocol_profile(profile_id)
    profile_environment = profile.environments.get(ENVIRONMENT)
    spec = get_environment_spec(ENVIRONMENT)
    manifest_document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixture_bytes = FIXTURE.read_bytes()
    fixture_rows = [
        json.loads(line)
        for line in fixture_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    ids: set[str] = set()
    operation_counts: dict[str, int] = {}
    generation_latency: list[float] = []
    deterministic = True
    for index in range(sample_count):
        started = time.perf_counter()
        first = environment.get_problem(index)
        generation_latency.append((time.perf_counter() - started) * 1000.0)
        second = environment.get_problem(index)
        deterministic = deterministic and first == second
        ids.add(first["id"])
        operation = str(first["operation_id"])
        operation_counts[operation] = operation_counts.get(operation, 0) + 1

    accepted = 0
    rejected = 0
    reward_latency: list[float] = []
    for row in fixture_rows:
        problem = environment.get_problem(int(row["index"]))
        started = time.perf_counter()
        accepted += int(
            environment.compute_reward(
                problem, str(row["accepted_completion"])
            )
            == 1.0
        )
        reward_latency.append((time.perf_counter() - started) * 1000.0)
        for completion in row["rejected_completions"]:
            started = time.perf_counter()
            rejected += int(
                environment.compute_reward(problem, str(completion)) == 0.0
            )
            reward_latency.append((time.perf_counter() - started) * 1000.0)

    malformed_count = max(10_000, sample_count)
    malformed_fail_closed = 0
    malformed_problem = environment.get_problem(0)
    for index in range(malformed_count):
        completion = "garbled-" + _sha256(str(index).encode("ascii"))
        malformed_fail_closed += int(
            environment.compute_reward(malformed_problem, completion) == 0.0
        )

    group_size = int(profile.sampling.rollouts)
    binary_frontier = []
    for success_count in range(1, group_size):
        probability = success_count / group_size
        binary_frontier.append(
            {
                "successes": success_count,
                "failures": group_size - success_count,
                "population_sigma": math.sqrt(
                    probability * (1.0 - probability)
                ),
            }
        )
    minimum_mixed_sigma = min(
        row["population_sigma"] for row in binary_frontier
    )

    manifest_digest = _canonical_sha256(manifest_document)
    fixture_digest = _sha256(fixture_bytes)
    software_revision, working_tree_dirty = _git_state()
    expected_operations = set(manifest_document["operation_templates"])
    gates = {
        "profile_declares_environment": profile_environment is not None,
        "profile_manifest_matches_registry": bool(
            profile_environment is not None
            and profile_environment.environment_manifest_sha256
            == spec.environment_manifest_sha256
        ),
        "manifest_digest_matches_registry": (
            manifest_digest == spec.environment_manifest_sha256
        ),
        "fixture_digest_matches_manifest": (
            fixture_digest
            == manifest_document.get("golden_fixture_sha256")
        ),
        "generator_is_deterministic": deterministic,
        "sample_ids_are_unique": len(ids) == sample_count,
        "all_operations_are_covered": set(operation_counts) == expected_operations,
        "golden_accepts_all_score_one": accepted == len(fixture_rows),
        "golden_near_misses_all_score_zero": rejected
        == sum(len(row["rejected_completions"]) for row in fixture_rows),
        "malformed_inputs_fail_closed": malformed_fail_closed == malformed_count,
        "binary_frontier_clears_sigma_min": minimum_mixed_sigma >= 0.24,
        "code_revision_is_pinned": (
            len(software_revision) == 40
            and (not working_tree_dirty or allow_dirty)
        ),
    }

    report = {
        "schema": "reliquary/environment-qualification/v1",
        "environment": ENVIRONMENT,
        "profile_id": profile.profile_id,
        "software_revision": software_revision,
        "working_tree_dirty": working_tree_dirty,
        "dirty_override": bool(allow_dirty and working_tree_dirty),
        "environment_manifest_sha256": manifest_digest,
        "registry_manifest_sha256": spec.environment_manifest_sha256,
        "golden_fixture_sha256": fixture_digest,
        "generator_version": GENERATOR_VERSION,
        "counts": {
            "generated_indices": sample_count,
            "unique_ids": len(ids),
            "golden_accepted": accepted,
            "golden_rejected": rejected,
            "malformed_inputs": malformed_count,
            "malformed_failed_closed": malformed_fail_closed,
            "operations": dict(sorted(operation_counts.items())),
        },
        "reward_frontier": {
            "group_size": group_size,
            "reward_lattice": list(spec.attainable_rewards),
            "minimum_mixed_population_sigma": minimum_mixed_sigma,
            "sigma_gate": 0.24,
            "binary_groups": binary_frontier,
        },
        "latency": {
            "problem_generation": _latency_summary(generation_latency),
            "golden_reward": _latency_summary(reward_latency),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record and explicitly permit a dirty development checkout",
    )
    args = parser.parse_args()
    report = qualify(
        sample_count=args.samples,
        profile_id=args.profile,
        allow_dirty=args.allow_dirty,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
