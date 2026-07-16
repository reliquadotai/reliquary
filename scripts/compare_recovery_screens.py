#!/usr/bin/env python3
"""Compare protocol-parity checkpoint screens with paired uncertainty.

Every candidate must use the same tasks, forced draws, tokenizer, budgets, and
runtime contract as the baseline. Confidence intervals resample whole tasks so
the multiple rollouts for one prompt remain a single statistical cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any


CONTRACT_FIELDS = (
    "environment",
    "tokenizer_repo",
    "tokenizer_revision",
    "dataset_sha256",
    "dataset_repo",
    "dataset_revision",
    "n_prompts",
    "prompt_offset",
    "samples_per_prompt",
    "thinking_budget",
    "answer_budget",
    "seed_domain",
    "attention_implementation",
)

OPTIONAL_IDENTITY_FIELDS = (
    "reliquary_revision",
    "screen_script_sha256",
)

RUNTIME_CONTRACT_FIELDS = (
    "python_version",
    "gpu_name",
    "gpu_compute_capability",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "transformers_version",
    "flash_linear_attention_version",
    "flash_attn_version",
    "causal_conv1d_version",
    "bitsandbytes_version",
)

METRIC_DIRECTIONS = {
    "pass_at_1": 1,
    "pass_at_k": 1,
    "pass_average": 1,
    "termination_rate": 1,
    "forced_rate": -1,
    "rambling_proxy_rate": -1,
    # Length has no monotonic direction: lower can mean concise reasoning or
    # the under-thinking collapse this recovery process is meant to catch.
    "mean_completion_length": 0,
}


def load_screen(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError(f"invalid recovery screen: {path}")
    return data


def _sample_key(sample: dict[str, Any]) -> tuple[str, int]:
    return str(sample["task_id"]), int(sample["sample_index"])


def validate_paired_contract(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    mismatches = [
        field
        for field in CONTRACT_FIELDS
        if baseline.get(field) != candidate.get(field)
    ]
    if mismatches:
        raise ValueError(
            "screen contracts differ for: " + ", ".join(mismatches)
        )
    identity_mismatches = [
        field
        for field in OPTIONAL_IDENTITY_FIELDS
        if baseline.get(field) is not None
        and candidate.get(field) is not None
        and baseline.get(field) != candidate.get(field)
    ]
    if identity_mismatches:
        raise ValueError(
            "screen source identities differ for: "
            + ", ".join(identity_mismatches)
        )
    runtime_mismatches = [
        field
        for field in RUNTIME_CONTRACT_FIELDS
        if (baseline.get("runtime") or {}).get(field)
        != (candidate.get("runtime") or {}).get(field)
    ]
    if runtime_mismatches:
        raise ValueError(
            "screen runtimes differ for: " + ", ".join(runtime_mismatches)
        )

    baseline_keys = [_sample_key(row) for row in baseline["samples"]]
    candidate_keys = [_sample_key(row) for row in candidate["samples"]]
    if len(set(baseline_keys)) != len(baseline_keys):
        raise ValueError("baseline contains duplicate task/sample keys")
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("candidate contains duplicate task/sample keys")
    if set(baseline_keys) != set(candidate_keys):
        raise ValueError("screen task/sample keys differ")


def task_metrics(screen: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in screen["samples"]:
        grouped.setdefault(str(sample["task_id"]), []).append(sample)

    expected_prompts = int(screen["n_prompts"])
    if len(grouped) != expected_prompts:
        raise ValueError(
            f"expected {expected_prompts} task groups, observed {len(grouped)}"
        )

    result: dict[str, dict[str, float]] = {}
    for task_id, samples in grouped.items():
        ordered = sorted(samples, key=lambda row: int(row["sample_index"]))
        rewards = [float(row["reward"]) for row in ordered]
        denominator = len(ordered)
        if denominator != int(screen["samples_per_prompt"]):
            raise ValueError(
                f"task {task_id} has {denominator} samples, expected "
                f"{screen['samples_per_prompt']}"
            )
        result[task_id] = {
            "pass_at_1": rewards[0],
            "pass_at_k": max(rewards),
            "pass_average": statistics.fmean(rewards),
            "termination_rate": sum(
                bool(row["terminated"]) for row in ordered
            ) / denominator,
            "forced_rate": sum(bool(row["forced"]) for row in ordered)
            / denominator,
            "rambling_proxy_rate": sum(
                bool(row["rambling_proxy"]) for row in ordered
            ) / denominator,
            "mean_completion_length": statistics.fmean(
                float(row["completion_length"]) for row in ordered
            ),
        }
    return result


def paired_sample_transitions(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Describe exact trajectory identity and discrete outcome flips."""
    baseline_rows = {
        _sample_key(row): row for row in baseline["samples"]
    }
    candidate_rows = {
        _sample_key(row): row for row in candidate["samples"]
    }
    keys = sorted(baseline_rows)
    reward_improved = reward_regressed = reward_unchanged = 0
    hash_available = hash_matches = 0
    for key in keys:
        before = baseline_rows[key]
        after = candidate_rows[key]
        reward_delta = float(after["reward"]) - float(before["reward"])
        if reward_delta > 0.0:
            reward_improved += 1
        elif reward_delta < 0.0:
            reward_regressed += 1
        else:
            reward_unchanged += 1
        before_hash = before.get("completion_sha256")
        after_hash = after.get("completion_sha256")
        if isinstance(before_hash, str) and isinstance(after_hash, str):
            hash_available += 1
            hash_matches += int(before_hash == after_hash)
    return {
        "sample_count": len(keys),
        "reward_improved": reward_improved,
        "reward_regressed": reward_regressed,
        "reward_unchanged": reward_unchanged,
        "completion_hash_available": hash_available,
        "completion_hash_matches": hash_matches,
        "completion_hash_match_rate": (
            hash_matches / hash_available if hash_available else None
        ),
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare_screens(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    iterations: int = 30_000,
    seed: int = 0,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    validate_paired_contract(baseline, candidate)
    baseline_tasks = task_metrics(baseline)
    candidate_tasks = task_metrics(candidate)
    task_ids = sorted(baseline_tasks)
    rng = random.Random(seed)

    metrics: dict[str, Any] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        paired_deltas = [
            candidate_tasks[task_id][metric]
            - baseline_tasks[task_id][metric]
            for task_id in task_ids
        ]
        observed = statistics.fmean(paired_deltas)
        bootstrapped = []
        for _ in range(iterations):
            bootstrapped.append(statistics.fmean(
                paired_deltas[rng.randrange(len(task_ids))]
                for _ in task_ids
            ))
        ci_low = _quantile(bootstrapped, 0.025)
        ci_high = _quantile(bootstrapped, 0.975)
        favorable_probability = None
        if direction:
            favorable_probability = sum(
                direction * delta > 0.0 for delta in bootstrapped
            ) / iterations
        metrics[metric] = {
            "baseline": statistics.fmean(
                baseline_tasks[task_id][metric] for task_id in task_ids
            ),
            "candidate": statistics.fmean(
                candidate_tasks[task_id][metric] for task_id in task_ids
            ),
            "delta": observed,
            "ci_95": [ci_low, ci_high],
            "direction": (
                "higher_is_better"
                if direction > 0
                else "lower_is_better" if direction < 0 else "context_only"
            ),
            "probability_favorable": favorable_probability,
        }

    return {
        "baseline_label": baseline["checkpoint_label"],
        "candidate_label": candidate["checkpoint_label"],
        "baseline_model": {
            "repo": baseline.get("model_repo"),
            "revision": baseline.get("model_revision"),
            "path": baseline.get("model_path"),
        },
        "candidate_model": {
            "repo": candidate.get("model_repo"),
            "revision": candidate.get("model_revision"),
            "path": candidate.get("model_path"),
        },
        "task_clusters": len(task_ids),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "paired_sample_transitions": paired_sample_transitions(
            baseline, candidate
        ),
        "metrics": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    transitions = report["paired_sample_transitions"]
    hash_rate = transitions["completion_hash_match_rate"]
    hash_text = (
        "n/a"
        if hash_rate is None
        else (
            f"{transitions['completion_hash_matches']}/"
            f"{transitions['completion_hash_available']} ({hash_rate:.3f})"
        )
    )
    lines = [
        f"## {report['candidate_label']} vs {report['baseline_label']}",
        "",
        f"Paired task clusters: {report['task_clusters']}; "
        f"bootstrap iterations: {report['bootstrap_iterations']}.",
        "",
        "Paired sample transitions: "
        f"reward +{transitions['reward_improved']} / "
        f"-{transitions['reward_regressed']} / "
        f"={transitions['reward_unchanged']}; exact completion hash "
        f"matches: {hash_text}.",
        "",
        "| Metric | Baseline | Candidate | Delta | 95% CI | Favorable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in report["metrics"].items():
        ci_low, ci_high = values["ci_95"]
        favorable = values["probability_favorable"]
        favorable_text = "n/a" if favorable is None else f"{favorable:.3f}"
        lines.append(
            f"| {metric} | {values['baseline']:.6g} | "
            f"{values['candidate']:.6g} | {values['delta']:+.6g} | "
            f"[{ci_low:+.6g}, {ci_high:+.6g}] | "
            f"{favorable_text} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    baseline = load_screen(args.baseline)
    comparisons = [
        compare_screens(
            baseline,
            load_screen(path),
            iterations=args.iterations,
            seed=args.seed,
        )
        for path in args.candidate
    ]
    payload = {
        "schema_version": 1,
        "baseline_path": str(args.baseline.resolve()),
        "candidate_paths": [str(path.resolve()) for path in args.candidate],
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        "# Recovery Screen Comparisons\n\n"
        + "\n".join(render_markdown(report) for report in comparisons),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
