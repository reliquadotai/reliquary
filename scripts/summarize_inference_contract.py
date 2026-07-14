#!/usr/bin/env python3
"""Aggregate inference-contract benchmark artifacts with Wilson intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _load(paths: list[Path]) -> list[dict]:
    artifacts = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        artifacts.append(row)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifacts = _load(args.artifacts)
    revisions = {a.get("model_revision_resolved") for a in artifacts}
    corpora = {a.get("prompts_sha256") for a in artifacts}
    checkpoints = {a.get("checkpoint_hash") for a in artifacts}
    if len(revisions) != 1 or len(corpora) != 1 or len(checkpoints) != 1:
        raise ValueError(
            "artifacts mix model revisions, prompt corpora, or checkpoint hashes"
        )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for artifact in artifacts:
        config = artifact["config"]
        key = (
            artifact["profile_label"],
            config["batch_size"],
            config["max_new_tokens"],
            config["dtype"],
            config["generation_use_cache"],
            config.get("bft_thinking_budget", 0),
            config.get("bft_answer_budget", 0),
        )
        groups[key].append(artifact)

    summaries = []
    for key, runs in sorted(groups.items()):
        positions = sum(int(a["summary"]["n_positions"]) for a in runs)
        hard = sum(int(a["summary"]["n_hard_mismatch"]) for a in runs)
        stochastic = sum(
            sum(int(row["n_stochastic"]) for row in a["rollouts"])
            for a in runs
        )
        exact = sum(
            sum(int(row["n_exact_match"]) for row in a["rollouts"])
            for a in runs
        )
        rollouts = [row for a in runs for row in a["rollouts"]]
        generated = sum(int(a["summary"]["generated_tokens"]) for a in runs)
        elapsed = sum(float(a["summary"]["elapsed_seconds"]) for a in runs)
        generation_elapsed = sum(
            float(a["summary"].get("generation_seconds", a["summary"]["elapsed_seconds"]))
            for a in runs
        )
        teacher_force_elapsed = sum(
            float(a["summary"].get("teacher_force_seconds", 0.0))
            for a in runs
        )
        run_rates = [
            a["summary"]["n_hard_mismatch"] / a["summary"]["n_positions"]
            for a in runs
            if a["summary"]["n_positions"]
        ]
        completion_maps = [
            {
                (int(row["prompt_idx"]), int(row["rollout_idx"])): row.get(
                    "completion_sha256"
                )
                for row in artifact["rollouts"]
            }
            for artifact in runs
        ]
        comparable_completion_pairs = 0
        matching_completion_pairs = 0
        if len(completion_maps) > 1:
            baseline = completion_maps[0]
            for candidate in completion_maps[1:]:
                for identity in baseline.keys() & candidate.keys():
                    if baseline[identity] is None or candidate[identity] is None:
                        continue
                    comparable_completion_pairs += 1
                    matching_completion_pairs += (
                        baseline[identity] == candidate[identity]
                    )
        summaries.append(
            {
                "profile_label": key[0],
                "batch_size": key[1],
                "max_new_tokens": key[2],
                "dtype": key[3],
                "generation_use_cache": key[4],
                "bft_thinking_budget": key[5],
                "bft_answer_budget": key[6],
                "replicates": len(runs),
                "runtime_profile_hashes": sorted(
                    {a["runtime_profile"]["profile_hash"] for a in runs}
                ),
                "positions": positions,
                "hard_mismatches": hard,
                "hard_mismatch_rate": hard / positions if positions else None,
                "hard_mismatch_wilson95": _wilson(hard, positions),
                "cross_process_rate_stddev": (
                    pstdev(run_rates) if len(run_rates) > 1 else 0.0
                ),
                "cross_process_completion_agreement": (
                    matching_completion_pairs / comparable_completion_pairs
                    if comparable_completion_pairs
                    else None
                ),
                "stochastic_agreement": exact / stochastic if stochastic else None,
                "stochastic_agreement_wilson95": _wilson(exact, stochastic),
                "eos_rate": (
                    mean(bool(row["ended_eos"]) for row in rollouts)
                    if rollouts
                    else None
                ),
                "forced_rate": (
                    mean(bool(row.get("forced")) for row in rollouts)
                    if rollouts
                    else None
                ),
                "bft_termination_paths": {
                    path: sum(
                        row.get("bft_termination_path") == path
                        for row in rollouts
                    )
                    for path in sorted(
                        {
                            row.get("bft_termination_path")
                            for row in rollouts
                            if row.get("bft_termination_path") is not None
                        }
                    )
                },
                "mean_unique_token_ratio": (
                    mean(float(row["unique_token_ratio"]) for row in rollouts)
                    if rollouts
                    else None
                ),
                "mean_repeated_ngram_fraction": (
                    mean(float(row["repeated_ngram_fraction"]) for row in rollouts)
                    if rollouts
                    else None
                ),
                "generation_tokens_per_second": (
                    generated / generation_elapsed if generation_elapsed else None
                ),
                "pipeline_tokens_per_second": generated / elapsed if elapsed else None,
                "teacher_force_seconds": teacher_force_elapsed,
                "max_cuda_peak_allocated_bytes": max(
                    int(a["summary"]["cuda_peak_allocated_bytes"] or 0) for a in runs
                ),
            }
        )

    pairwise_profiles = []
    for left_index, left in enumerate(artifacts):
        for right in artifacts[left_index + 1:]:
            if left["profile_label"] == right["profile_label"]:
                continue
            if left.get("replicate") != right.get("replicate"):
                continue
            left_config = left["config"]
            right_config = right["config"]
            config_fields = (
                "batch_size",
                "max_new_tokens",
                "dtype",
                "generation_use_cache",
                "bft_thinking_budget",
                "bft_answer_budget",
            )
            if any(
                left_config.get(field, 0) != right_config.get(field, 0)
                for field in config_fields
            ):
                continue
            left_rows = {
                (int(row["prompt_idx"]), int(row["rollout_idx"])): row
                for row in left["rollouts"]
            }
            right_rows = {
                (int(row["prompt_idx"]), int(row["rollout_idx"])): row
                for row in right["rollouts"]
            }
            identities = left_rows.keys() & right_rows.keys()
            comparable = [
                identity for identity in identities
                if left_rows[identity].get("completion_sha256") is not None
                and right_rows[identity].get("completion_sha256") is not None
            ]
            pairwise_profiles.append(
                {
                    "left_profile": left["profile_label"],
                    "right_profile": right["profile_label"],
                    "replicate": left.get("replicate"),
                    "comparable_rollouts": len(comparable),
                    "completion_agreement": (
                        sum(
                            left_rows[identity]["completion_sha256"]
                            == right_rows[identity]["completion_sha256"]
                            for identity in comparable
                        ) / len(comparable)
                        if comparable
                        else None
                    ),
                }
            )

    output = {
        "schema_version": 1,
        "model_revision_resolved": next(iter(revisions)),
        "checkpoint_hash": next(iter(checkpoints)),
        "prompts_sha256": next(iter(corpora)),
        "artifacts": len(artifacts),
        "groups": summaries,
        "pairwise_profile_completion_agreement": pairwise_profiles,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
