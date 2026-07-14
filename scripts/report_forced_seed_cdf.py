#!/usr/bin/env python3
"""Summarize validator-private forced-seed CDF calibration telemetry."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_PATH = Path(
    "/root/reliquary/state/auth_forensics/forced-seed-shadow.jsonl"
)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = index - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _rollout_cohort(items: list[dict]) -> dict:
    positions = sum(int(item.get("n_positions", 0) or 0) for item in items)
    hard = sum(int(item.get("n_hard_mismatch", 0) or 0) for item in items)

    def values(field: str) -> list[float]:
        return [
            float(item[field])
            for item in items
            if item.get(field) is not None
        ]

    return {
        "rollouts": len(items),
        "positions": positions,
        "hard_mismatches": hard,
        "hard_mismatch_rate": hard / positions if positions else None,
        "repeated_ngram_fraction_mean": (
            statistics.fmean(values("repeated_ngram_fraction"))
            if values("repeated_ngram_fraction")
            else None
        ),
        "tail_repeated_ngram_fraction_mean": (
            statistics.fmean(values("tail_repeated_ngram_fraction"))
            if values("tail_repeated_ngram_fraction")
            else None
        ),
        "max_same_token_run_p95": _quantile(
            values("max_same_token_run"), 0.95,
        ),
        "first_hard_mismatch_offset_p50": _quantile(
            values("first_hard_mismatch_offset"), 0.50,
        ),
    }


def summarize(rows: list[dict]) -> dict:
    v2 = [row for row in rows if int(row.get("schema_version", 1)) >= 2]
    v3 = [row for row in rows if int(row.get("schema_version", 1)) >= 3]
    v4 = [row for row in rows if int(row.get("schema_version", 1)) >= 4]
    rollouts_v4 = [
        item
        for row in v4
        for item in (row.get("per_rollout") or [])
        if isinstance(item, dict)
    ]
    scores = [float(row.get("score", 0.0)) for row in v2]
    cdf_clean = [
        row
        for row in v2
        if not row.get("ratio_group_would_reject", False)
        and not row.get("ratio_rollout_would_reject", False)
    ]
    hard_clean = [row for row in cdf_clean if row.get("cdf_would_reject", False)]
    by_hotkey: dict[str, list[dict]] = defaultdict(list)
    by_environment: dict[str, list[dict]] = defaultdict(list)
    for row in v2:
        by_hotkey[str(row.get("miner_hotkey", ""))].append(row)
        if int(row.get("schema_version", 1)) >= 3:
            by_environment[str(row.get("env_name", "") or "unknown")].append(row)

    timestamps = [float(row["ts_unix"]) for row in v2 if "ts_unix" in row]
    span_hours = (
        (max(timestamps) - min(timestamps)) / 3600.0
        if len(timestamps) >= 2
        else 0.0
    )
    # One unexplained hard mismatch is enough to stop activation. Volume and
    # duration thresholds are only evidence requirements for enabling a gate,
    # not prerequisites for recognizing unsafe behavior.
    if hard_clean:
        decision = "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"
    elif len(v2) < 1000 or len(by_hotkey) < 5 or span_hours < 24:
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        decision = "ELIGIBLE_FOR_BOUNDED_ENFORCEMENT_CANARY"

    ratio_clean_positions = sum(
        int(row.get("n_positions", 0) or 0) for row in cdf_clean
    )
    ratio_clean_hard_mismatches = sum(
        int(row.get("n_hard_mismatch", 0) or 0) for row in cdf_clean
    )
    termination_paths: dict[str, list[dict]] = defaultdict(list)
    for item in rollouts_v4:
        termination_paths[str(item.get("termination_path") or "unknown")].append(
            item
        )
    mismatch_rollouts = [
        item
        for item in rollouts_v4
        if int(item.get("n_hard_mismatch", 0) or 0) > 0
    ]
    clean_rollouts = [
        item
        for item in rollouts_v4
        if int(item.get("n_hard_mismatch", 0) or 0) == 0
    ]
    directionality = [
        item
        for item in rollouts_v4
        if item.get("first_hard_mismatch_offset") is not None
        and item.get("first_repeated_ngram_offset") is not None
    ]

    return {
        "records_total": len(rows),
        "records_schema_v2": len(v2),
        "records_schema_v3": len(v3),
        "records_schema_v4": len(v4),
        "hotkeys_schema_v2": len(by_hotkey),
        "schema_v2_span_hours": span_hours,
        "windows_schema_v3": len(
            {
                int(row["window_start"])
                for row in v3
                if row.get("window_start") is not None
            }
        ),
        "checkpoints_schema_v3": sorted(
            {
                str(row.get("checkpoint_hash", ""))
                for row in v3
                if row.get("checkpoint_hash")
            }
        ),
        "ratio_score": {
            "min": min(scores) if scores else None,
            "p01": _quantile(scores, 0.01),
            "p50": statistics.median(scores) if scores else None,
            "p99": _quantile(scores, 0.99),
            "max": max(scores) if scores else None,
        },
        "ratio_group_would_reject": sum(
            bool(row.get("ratio_group_would_reject", False)) for row in v2
        ),
        "ratio_rollout_would_reject": sum(
            bool(row.get("ratio_rollout_would_reject", False)) for row in v2
        ),
        "cdf_would_reject": sum(
            bool(row.get("cdf_would_reject", False)) for row in v2
        ),
        "cdf_hard_mismatch_groups_among_ratio_clean": len(hard_clean),
        "cdf_positions_among_ratio_clean": ratio_clean_positions,
        "cdf_hard_mismatch_positions_among_ratio_clean": (
            ratio_clean_hard_mismatches
        ),
        "cdf_hard_mismatch_rate_among_ratio_clean": (
            float(ratio_clean_hard_mismatches) / float(ratio_clean_positions)
            if ratio_clean_positions
            else None
        ),
        "cdf_miss_severity_schema_v3": {
            "gt_0_01": sum(int(row.get("n_miss_gt_0_01", 0) or 0) for row in v3),
            "gt_0_05": sum(int(row.get("n_miss_gt_0_05", 0) or 0) for row in v3),
            "gt_0_10": sum(int(row.get("n_miss_gt_0_10", 0) or 0) for row in v3),
        },
        "max_cdf_miss_p99_ratio_clean": _quantile(
            [float(row.get("max_cdf_miss", 0.0)) for row in cdf_clean],
            0.99,
        ),
        "by_hotkey": sorted(
            (
                {
                    "hotkey": hotkey,
                    "records": len(items),
                    "ratio_mean": statistics.fmean(
                        float(item.get("score", 0.0)) for item in items
                    ),
                    "cdf_would_reject": sum(
                        bool(item.get("cdf_would_reject", False))
                        for item in items
                    ),
                }
                for hotkey, items in by_hotkey.items()
            ),
            key=lambda item: (-item["cdf_would_reject"], item["ratio_mean"]),
        ),
        "by_environment_schema_v3": sorted(
            (
                {
                    "environment": environment,
                    "records": len(items),
                    "cdf_would_reject": sum(
                        bool(item.get("cdf_would_reject", False))
                        for item in items
                    ),
                    "n_hard_mismatch": sum(
                        int(item.get("n_hard_mismatch", 0) or 0)
                        for item in items
                    ),
                }
                for environment, items in by_environment.items()
            ),
            key=lambda item: item["environment"],
        ),
        "rollouts_schema_v4": len(rollouts_v4),
        "by_termination_path_schema_v4": sorted(
            (
                {
                    "termination_path": path,
                    **_rollout_cohort(items),
                }
                for path, items in termination_paths.items()
            ),
            key=lambda item: item["termination_path"],
        ),
        "cdf_degeneracy_cohorts_schema_v4": {
            "hard_mismatch": _rollout_cohort(mismatch_rollouts),
            "no_hard_mismatch": _rollout_cohort(clean_rollouts),
        },
        "directionality_schema_v4": {
            "both_offsets_observed": len(directionality),
            "cdf_mismatch_at_or_before_repetition": sum(
                int(item["first_hard_mismatch_offset"])
                <= int(item["first_repeated_ngram_offset"])
                for item in directionality
            ),
            "repetition_before_cdf_mismatch": sum(
                int(item["first_repeated_ngram_offset"])
                < int(item["first_hard_mismatch_offset"])
                for item in directionality
            ),
        },
        "decision": decision,
        "activation_rule": (
            "Do not set FORCED_SEED_CDF_ENFORCE=true until at least 24h, "
            "1000 schema-v2+ groups and 5 hotkeys show zero unexplained hard "
            "mismatches among ratio-clean traffic, then run a bounded canary "
            "and confirm an adversarial branch-token control still rejects."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(rows)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Forced-seed CDF calibration")
    print(f"decision: {report['decision']}")
    print(
        "records: "
        f"{report['records_schema_v2']} v2+ "
        f"({report['records_schema_v3']} v3, "
        f"{report['records_schema_v4']} v4) / "
        f"{report['records_total']} total; "
        f"{report['hotkeys_schema_v2']} hotkeys; "
        f"{report['schema_v2_span_hours']:.1f}h"
    )
    print(
        "would reject: "
        f"ratio-group={report['ratio_group_would_reject']} "
        f"ratio-rollout={report['ratio_rollout_would_reject']} "
        f"cdf={report['cdf_would_reject']}"
    )
    print(
        "ratio-clean CDF hard mismatches: "
        f"{report['cdf_hard_mismatch_groups_among_ratio_clean']} groups, "
        f"{report['cdf_hard_mismatch_positions_among_ratio_clean']} / "
        f"{report['cdf_positions_among_ratio_clean']} positions"
    )
    print(
        "v3 severity: "
        f">.01={report['cdf_miss_severity_schema_v3']['gt_0_01']} "
        f">.05={report['cdf_miss_severity_schema_v3']['gt_0_05']} "
        f">.10={report['cdf_miss_severity_schema_v3']['gt_0_10']}"
    )
    print(
        "v4 causal telemetry: "
        f"{report['rollouts_schema_v4']} rollouts; "
        f"{report['directionality_schema_v4']['both_offsets_observed']} "
        "with both CDF and repetition onsets"
    )
    print(report["activation_rule"])


if __name__ == "__main__":
    main()
