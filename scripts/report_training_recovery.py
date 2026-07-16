#!/usr/bin/env python3
"""Correlate optimizer telemetry, checkpoint lineage, and termination records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


_CHECKPOINT_RE = re.compile(r"^checkpoint\s+(\d+)$", re.IGNORECASE)


def _timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _iso(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _quantile(values: list[float], q: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return finite[lo]
    fraction = position - lo
    return finite[lo] * (1.0 - fraction) + finite[hi] * fraction


def deduplicate_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove W&B resume duplicates while retaining distinct same-window logs."""
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("train/grad_norm") is None:
            continue
        key = (
            row.get("_step"),
            row.get("_timestamp"),
            row.get("train/grad_norm"),
            row.get("train/kl"),
        )
        existing = unique.get(key)
        if existing is None or len(row) > len(existing):
            unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (float(row.get("_timestamp", 0)), int(row.get("_step", 0))),
    )


def parse_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = []
    for row in rows:
        title = str(row.get("title") or "")
        match = _CHECKPOINT_RE.match(title)
        if match is not None:
            checkpoint_n = int(match.group(1))
        elif title == "initial commit":
            checkpoint_n = 0
        else:
            continue
        checkpoints.append({
            "checkpoint_n": checkpoint_n,
            "revision": str(row["id"]),
            "published_at": str(row["date"]),
            "published_ts": _timestamp(str(row["date"])),
        })
    return sorted(checkpoints, key=lambda row: row["published_ts"])


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gradients = [float(row["train/grad_norm"]) for row in rows]
    kls = [
        float(row["train/kl"])
        for row in rows
        if row.get("train/kl") is not None
    ]
    return {
        "steps": len(rows),
        "gradient_norm": {
            "p50": _quantile(gradients, 0.50),
            "p95": _quantile(gradients, 0.95),
            "p99": _quantile(gradients, 0.99),
            "max": max(gradients) if gradients else None,
            "gt_100": sum(value > 100 for value in gradients),
        },
        "kl": {
            "p50": _quantile(kls, 0.50),
            "p95": _quantile(kls, 0.95),
            "p99": _quantile(kls, 0.99),
            "max": max(kls) if kls else None,
        },
    }


def correlate_intervals(
    history: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    intervals = []
    for index, checkpoint in enumerate(checkpoints):
        start = float(checkpoint["published_ts"])
        end = (
            float(checkpoints[index + 1]["published_ts"])
            if index + 1 < len(checkpoints)
            else float("inf")
        )
        rows = [
            row for row in history
            if start <= float(row.get("_timestamp", 0)) < end
        ]
        interval = {
            key: value
            for key, value in checkpoint.items()
            if key != "published_ts"
        }
        interval.update(_metric_summary(rows))
        interval["first_window"] = (
            min(int(row["_step"]) for row in rows) if rows else None
        )
        interval["last_window"] = (
            max(int(row["_step"]) for row in rows) if rows else None
        )
        intervals.append(interval)
    return intervals


def summarize_termination(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event") != "termination_shadow":
            continue
        grouped[str(row.get("checkpoint_hash") or "unknown")].append(row)
    report = {}
    for checkpoint, records in grouped.items():
        report[checkpoint] = {
            "records": len(records),
            "windows": len({row.get("window_start") for row in records}),
            "hotkeys": len({row.get("miner_hotkey") for row in records}),
            "termination_failures": sum(
                not bool(row.get("termination_ok", False)) for row in records
            ),
            "cap_truncations": sum(
                bool(row.get("cap_truncated", False)) for row in records
            ),
            "terminal_boundary_candidates": sum(
                bool(row.get("terminal_boundary_compatible", False))
                for row in records
            ),
            "natural_boundary_candidates": sum(
                bool(row.get("natural_close_boundary_compatible", False))
                for row in records
            ),
        }
    return dict(sorted(report.items()))


def build_report(
    history_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    history = deduplicate_history(history_rows)
    checkpoints = parse_checkpoints(checkpoint_rows)
    anomalies = [
        {
            "window": int(row["_step"]),
            "timestamp": _iso(float(row["_timestamp"])),
            "gradient_norm": float(row["train/grad_norm"]),
            "kl": float(row.get("train/kl", 0)),
            "ppo_loss": float(row.get("train/ppo_loss", 0)),
            "rollouts": int(row.get("train/rollouts_processed", 0)),
            "forced_rollout_ratio": row.get("bft/forced_rollout_ratio"),
        }
        for row in history
        if (
            not math.isfinite(float(row["train/grad_norm"]))
            or float(row["train/grad_norm"]) > 100
        )
    ]
    return {
        "schema_version": 1,
        "history_rows_raw": len(history_rows),
        "history_steps_unique": len(history),
        "global": _metric_summary(history),
        "anomalies": anomalies,
        "checkpoint_intervals": correlate_intervals(history, checkpoints),
        "termination_by_checkpoint": summarize_termination(termination_rows),
    }


def _markdown(report: dict[str, Any]) -> str:
    global_stats = report["global"]
    lines = [
        "# Training Recovery Lineage Report",
        "",
        f"Unique optimizer steps: {report['history_steps_unique']}",
        "",
        "## Global Health",
        "",
        "| Metric | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|",
        (
            "| Gradient norm | {p50:.6g} | {p95:.6g} | {p99:.6g} | "
            "{max:.6g} |"
        ).format(**global_stats["gradient_norm"]),
        (
            "| KL | {p50:.6g} | {p95:.6g} | {p99:.6g} | {max:.6g} |"
        ).format(**global_stats["kl"]),
        "",
        "## Health-Gate Events",
        "",
    ]
    if not report["anomalies"]:
        lines.append("No non-finite or gradient-norm > 100 events.")
    else:
        lines.extend([
            "| Window | Timestamp | Grad norm | KL | BFT forced ratio |",
            "|---:|---|---:|---:|---:|",
        ])
        for row in report["anomalies"]:
            lines.append(
                f"| {row['window']} | {row['timestamp']} | "
                f"{row['gradient_norm']:.6g} | {row['kl']:.6g} | "
                f"{row['forced_rollout_ratio']} |"
            )
    lines.extend([
        "",
        "## Checkpoint Intervals",
        "",
        "| Ckpt | Revision | Steps | First | Last | Max grad | Max KL |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["checkpoint_intervals"]:
        lines.append(
            f"| {row['checkpoint_n']} | `{row['revision'][:12]}` | "
            f"{row['steps']} | {row['first_window']} | {row['last_window']} | "
            f"{row['gradient_norm']['max']} | {row['kl']['max']} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--termination", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    history_rows = json.loads(args.history.read_text(encoding="utf-8"))
    checkpoint_rows = json.loads(args.checkpoints.read_text(encoding="utf-8"))
    termination_rows = [
        json.loads(line)
        for line in args.termination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_report(history_rows, checkpoint_rows, termination_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["global"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
