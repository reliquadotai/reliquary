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


def summarize(rows: list[dict]) -> dict:
    v2 = [row for row in rows if int(row.get("schema_version", 1)) >= 2]
    scores = [float(row.get("score", 0.0)) for row in rows]
    cdf_clean = [
        row
        for row in v2
        if not row.get("ratio_group_would_reject", False)
        and not row.get("ratio_rollout_would_reject", False)
    ]
    hard_clean = [row for row in cdf_clean if row.get("cdf_would_reject", False)]
    by_hotkey: dict[str, list[dict]] = defaultdict(list)
    for row in v2:
        by_hotkey[str(row.get("miner_hotkey", ""))].append(row)

    timestamps = [float(row["ts_unix"]) for row in v2 if "ts_unix" in row]
    span_hours = (
        (max(timestamps) - min(timestamps)) / 3600.0
        if len(timestamps) >= 2
        else 0.0
    )
    if len(v2) < 1000 or len(by_hotkey) < 5 or span_hours < 24:
        decision = "INSUFFICIENT_EVIDENCE"
    elif hard_clean:
        decision = "HOLD_AND_REVIEW_CDF_HARD_MISMATCHES"
    else:
        decision = "ELIGIBLE_FOR_BOUNDED_ENFORCEMENT_CANARY"

    return {
        "records_total": len(rows),
        "records_schema_v2": len(v2),
        "hotkeys_schema_v2": len(by_hotkey),
        "schema_v2_span_hours": span_hours,
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
        "decision": decision,
        "activation_rule": (
            "Do not set FORCED_SEED_CDF_ENFORCE=true until at least 24h, "
            "1000 schema-v2 groups and 5 hotkeys show zero unexplained hard "
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
        f"{report['records_schema_v2']} v2 / {report['records_total']} total; "
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
        f"{report['cdf_hard_mismatch_groups_among_ratio_clean']}"
    )
    print(report["activation_rule"])


if __name__ == "__main__":
    main()
