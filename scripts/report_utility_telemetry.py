#!/usr/bin/env python3
"""Summarize validator-private auction utility telemetry without raw vectors."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DIRECTORY = Path("/root/reliquary/state/utility_telemetry")
MIN_WINDOWS = 256
MIN_CHECKPOINTS = 3
MIN_COMPLETE_FIELD_RATE = 0.99


def _paths(inputs: Iterable[str | Path]) -> list[Path]:
    discovered: set[Path] = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            discovered.update(path.glob("window-*.json.gz"))
        elif path.is_file():
            discovered.add(path)
    return sorted(discovered)


def load_bundles(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    bundles = []
    for path in _paths(inputs):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: telemetry bundle must be an object")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"{path}: unsupported telemetry schema")
        bundles.append(payload)
    return bundles


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _pearson(pairs: Iterable[tuple[float | None, float | None]]) -> float | None:
    clean = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(clean) < 3:
        return None
    xs, ys = zip(*clean)
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in clean)
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    denominator = x_norm * y_norm
    return numerator / denominator if denominator else None


def _rollout_mean(group: dict[str, Any], field: str) -> float | None:
    return _mean(_finite(row.get(field)) for row in group.get("rollouts", []))


def _nested_rollout_mean(
    group: dict[str, Any], field: str, statistic: str = "mean"
) -> float | None:
    return _mean(
        _finite((row.get(field) or {}).get(statistic))
        for row in group.get("rollouts", [])
    )


def _group_features(group: dict[str, Any]) -> dict[str, Any]:
    rollouts = list(group.get("rollouts") or [])
    rewards = [_finite(row.get("reward")) for row in rollouts]
    reward_values = [value for value in rewards if value is not None]
    complete_rollouts = sum(
        (row.get("chosen_nll") or {}).get("mean") is not None
        and (row.get("full_policy_entropy") or {}).get("mean") is not None
        and row.get("hidden_delta_f16_b64") is not None
        and row.get("termination_path") is not None
        for row in rollouts
    )
    return {
        "window": int(group["_window"]),
        "checkpoint_revision": str(group.get("checkpoint_revision", "")),
        "candidate_id": str(group.get("candidate_id", "")),
        "operator_pseudonym": str(group.get("operator_pseudonym", "")),
        "prompt_content_sha256": str(group.get("prompt_content_sha256", "")),
        "role": str(group.get("role", "unknown")),
        "forensic_role": group.get("forensic_role"),
        "rollouts": len(rollouts),
        "complete_rollouts": complete_rollouts,
        "positive_reward_count": sum(
            value > 0.0 for value in reward_values
        ),
        "reward_mean": _mean(reward_values),
        "completion_length_mean": _rollout_mean(
            group, "completion_length"
        ),
        "chosen_nll_mean": _nested_rollout_mean(group, "chosen_nll"),
        "full_policy_entropy_mean": _nested_rollout_mean(
            group, "full_policy_entropy"
        ),
        "representation_shift_l2_mean": _rollout_mean(
            group, "representation_shift_l2"
        ),
        "natural_eos_rate": _mean(
            float(bool(row.get("natural_eos", False))) for row in rollouts
        ),
        "repeated_ngram_fraction_mean": _mean(
            _finite((row.get("token_degeneracy") or {}).get(
                "repeated_ngram_fraction"
            ))
            for row in rollouts
        ),
    }


def summarize(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    by_environment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    windows: set[int] = set()
    for bundle in bundles:
        window = int(bundle["window"])
        windows.add(window)
        for environment, groups in (bundle.get("environments") or {}).items():
            for group in groups or []:
                row = dict(group)
                row["_window"] = window
                by_environment[str(environment)].append(_group_features(row))

    environments = {}
    for environment, groups in sorted(by_environment.items()):
        env_windows = {row["window"] for row in groups}
        checkpoints = {
            row["checkpoint_revision"]
            for row in groups
            if row["checkpoint_revision"]
        }
        total_rollouts = sum(row["rollouts"] for row in groups)
        complete_rollouts = sum(row["complete_rollouts"] for row in groups)
        complete_rate = (
            complete_rollouts / total_rollouts if total_rollouts else None
        )
        reasons = []
        if len(env_windows) < MIN_WINDOWS:
            reasons.append(f"windows<{MIN_WINDOWS}")
        if len(checkpoints) < MIN_CHECKPOINTS:
            reasons.append(f"checkpoints<{MIN_CHECKPOINTS}")
        if complete_rate is None or complete_rate < MIN_COMPLETE_FIELD_RATE:
            reasons.append(f"complete_field_rate<{MIN_COMPLETE_FIELD_RATE}")
        if not any(row["role"] == "forensic" for row in groups):
            reasons.append("no_forensic_counterfactuals")

        content_counts = Counter(
            row["prompt_content_sha256"]
            for row in groups
            if row["prompt_content_sha256"]
        )
        feature_names = (
            "completion_length_mean",
            "chosen_nll_mean",
            "full_policy_entropy_mean",
            "representation_shift_l2_mean",
            "natural_eos_rate",
            "repeated_ngram_fraction_mean",
        )
        environments[environment] = {
            "decision": (
                "INSUFFICIENT_TELEMETRY"
                if reasons
                else "READY_FOR_CAUSAL_LABELING_NOT_ACTIVATION"
            ),
            "blocking_reasons": reasons,
            "windows": len(env_windows),
            "checkpoints": len(checkpoints),
            "groups": len(groups),
            "winner_groups": sum(row["role"] == "winner" for row in groups),
            "forensic_groups": sum(
                row["role"] == "forensic" for row in groups
            ),
            "counterfactual_groups": sum(
                row["forensic_role"] == "counterfactual" for row in groups
            ),
            "random_watch_groups": sum(
                row["forensic_role"] == "random_watch" for row in groups
            ),
            "operators": len({
                row["operator_pseudonym"]
                for row in groups
                if row["operator_pseudonym"]
            }),
            "rollouts": total_rollouts,
            "complete_field_rate": complete_rate,
            "positive_reward_count_distribution": dict(sorted(Counter(
                row["positive_reward_count"] for row in groups
            ).items())),
            "duplicate_content_groups": sum(
                count - 1 for count in content_counts.values() if count > 1
            ),
            "feature_means": {
                name: _mean(row[name] for row in groups)
                for name in feature_names
            },
            "outcome_association_not_training_utility": {
                name: _pearson(
                    (row[name], row["reward_mean"]) for row in groups
                )
                for name in feature_names
            },
        }

    ordered_windows = sorted(windows)
    missing_windows = (
        (ordered_windows[-1] - ordered_windows[0] + 1 - len(ordered_windows))
        if ordered_windows
        else 0
    )
    return {
        "schema_version": 1,
        "bundles": len(bundles),
        "windows": len(windows),
        "window_min": ordered_windows[0] if ordered_windows else None,
        "window_max": ordered_windows[-1] if ordered_windows else None,
        "missing_windows_in_observed_span": missing_windows,
        "environments": environments,
        "activation_allowed": False,
        "activation_note": (
            "This report checks telemetry readiness only. Auction-v3 activation "
            "requires out-of-checkpoint U_step/U_align labels and shadow replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(DEFAULT_DIRECTORY)],
        help="Bundle file(s) or directories (default: validator state path)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    report = summarize(load_bundles(args.paths))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
