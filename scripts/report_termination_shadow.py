#!/usr/bin/env python3
"""Summarize validator-private termination boundary diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


DEFAULT_PATH = Path(
    "/root/reliquary/state/auth_forensics/termination-shadow.jsonl"
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


def summarize(
    rows: list[dict],
    *,
    min_records: int = 100,
    min_hotkeys: int = 5,
    min_windows: int = 24,
) -> dict:
    records = [row for row in rows if row.get("event") == "termination_shadow"]
    hotkeys = {str(row.get("miner_hotkey", "")) for row in records}
    windows = {row.get("window_start") for row in records}
    environments = Counter(
        str(row.get("env_name") or "unknown") for row in records
    )
    terminal_boundary = [
        row for row in records
        if row.get("terminal_boundary_compatible", False)
    ]
    natural_boundary = [
        row for row in records
        if row.get("natural_close_boundary_compatible", False)
    ]
    boundary_candidates = terminal_boundary + natural_boundary
    if boundary_candidates:
        decision = "REVIEW_BOUNDARY_CANDIDATES_KEEP_GATE_UNCHANGED"
    elif (
        len(records) < min_records
        or len(hotkeys) < min_hotkeys
        or len(windows) < min_windows
    ):
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        decision = "NO_BOUNDARY_FALSE_POSITIVE_SIGNAL"

    terminal_misses = [
        float(row["terminal_pick_cdf_miss"])
        for row in records
        if row.get("terminal_pick_cdf_miss") is not None
    ]
    natural_misses = [
        float(row["natural_close_pick_cdf_miss"])
        for row in records
        if row.get("natural_close_pick_cdf_miss") is not None
    ]
    return {
        "records": len(records),
        "distinct_hotkeys": len(hotkeys),
        "distinct_windows": len(windows),
        "environments": dict(sorted(environments.items())),
        "checkpoints": sorted(
            {
                str(row.get("checkpoint_hash"))
                for row in records
                if row.get("checkpoint_hash")
            }
        ),
        "exact_terminal_pick_rescues": sum(
            row.get("terminal_pick_ok") is True for row in records
        ),
        "terminal_boundary_candidates": len(terminal_boundary),
        "natural_close_exact_rescues": sum(
            row.get("natural_close_pick_ok") is True for row in records
        ),
        "natural_close_boundary_candidates": len(natural_boundary),
        "termination_failures_observed": sum(
            not row.get("termination_ok", False) for row in records
        ),
        "cap_truncations_observed": sum(
            bool(row.get("cap_truncated", False)) for row in records
        ),
        "would_exceed_truncation_budget": sum(
            bool(row.get("would_exceed_truncation_budget", False))
            for row in records
        ),
        "terminal_cdf_miss": {
            "min": min(terminal_misses) if terminal_misses else None,
            "p50": statistics.median(terminal_misses) if terminal_misses else None,
            "p99": _quantile(terminal_misses, 0.99),
            "max": max(terminal_misses) if terminal_misses else None,
        },
        "natural_close_cdf_miss": {
            "min": min(natural_misses) if natural_misses else None,
            "p50": statistics.median(natural_misses) if natural_misses else None,
            "p99": _quantile(natural_misses, 0.99),
            "max": max(natural_misses) if natural_misses else None,
        },
        "decision": decision,
        "operator_rule": (
            "A near-boundary record is not proof of an honest rollout and must "
            "not automatically relax termination. Reproduce candidates with "
            "the matching checkpoint and generation stack, compare by "
            "environment, and run injected-stop controls before any gate change."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--min-records", type=int, default=100)
    parser.add_argument("--min-hotkeys", type=int, default=5)
    parser.add_argument("--min-windows", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize(
        rows,
        min_records=args.min_records,
        min_hotkeys=args.min_hotkeys,
        min_windows=args.min_windows,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Termination shadow calibration")
    print(f"decision: {report['decision']}")
    print(
        f"records={report['records']} hotkeys={report['distinct_hotkeys']} "
        f"windows={report['distinct_windows']}"
    )
    print(
        "boundary candidates: "
        f"terminal={report['terminal_boundary_candidates']} "
        f"natural_close={report['natural_close_boundary_candidates']}"
    )
    print(
        "exact rescues: "
        f"terminal={report['exact_terminal_pick_rescues']} "
        f"natural_close={report['natural_close_exact_rescues']}"
    )
    print(report["operator_rule"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
