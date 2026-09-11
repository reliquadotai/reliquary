#!/usr/bin/env python3
"""Private, conditional policy comparison. Never submits or changes live state."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def report(paths):
    counts, eligible, picks, gaps = Counter(), {}, [], []
    sequences, checkpoints, windows = {}, set(), set()
    malformed = 0
    seen = set()
    baselines = {}
    for path in sorted(paths):
        for line in Path(path).read_text().splitlines():
            try:
                e = json.loads(line)
            except (ValueError, TypeError):
                malformed += 1
                continue
            if not isinstance(e, dict):
                malformed += 1
                continue
            run = e.get("run_id")
            seq = e.get("sequence")
            if run and isinstance(seq, int):
                if (run, seq) in seen:
                    continue
                seen.add((run, seq))
                sequences.setdefault(run, set()).add(seq)
            counts[e.get("event", "unknown")] += 1
            if e.get("event") == "external_sample":
                baselines.setdefault(run, []).append(e)
            if e.get("checkpoint"):
                checkpoints.add(e["checkpoint"])
            if e.get("window") is not None:
                windows.add(e["window"])
            if e.get("event") == "candidate_eligible":
                key = (run, e["window"], e["environment"], e["receipt_id"])
                if key in eligible:
                    gaps.append("duplicate_candidate_identity")
                eligible[key] = e
            if e.get("event") in {"pick_ready_set", "proof_ready_set"}:
                picks.append(e)
            status = e.get("writer_status", {})
            if status.get("errors") or status.get("dropped"):
                gaps.append("writer_loss")
    comparisons = []
    for e in picks:
        candidates, selected = e["candidates"], e["chosen_receipts"]

        def identity(c):
            return (e.get("run_id"), e["window"], e["environment"], c["receipt_id"])

        if (
            not selected
            or any(
                not c["receipt_id"] or identity(c) not in eligible for c in candidates
            )
            or len({c["receipt_id"] for c in candidates}) != len(candidates)
        ):
            gaps.append("ready_set_without_unique_eligible_identity")
            continue
        fifo = sorted(candidates, key=lambda c: eligible[identity(c)]["ordinal"])[
            : len(selected)
        ]
        alternate = [c["receipt_id"] for c in fifo]
        comparisons.append(
            dict(
                window=e["window"],
                environment=e["environment"],
                stage=e["event"],
                candidates=len(candidates),
                selected=len(selected),
                replaced=len(set(selected) - set(alternate)),
                selected_receipts=selected,
                eligible_order_receipts=alternate,
            )
        )
    holes = {run: max(seq) - min(seq) + 1 - len(seq) for run, seq in sequences.items()}
    external = []
    for run, rows in baselines.items():
        rows = sorted({r["time"]: r for r in rows}.values(), key=lambda r: r["time"])
        utilization = []
        for row in rows:
            for line in row.get("gpu", {}).get("stdout", "").splitlines():
                try:
                    utilization.append(float(line.split(",")[1]))
                except (ValueError, IndexError):
                    pass
        external.append(
            dict(
                run_id=run,
                container=rows[0]["container"],
                samples=len(rows),
                observed_span_seconds=rows[-1]["time"] - rows[0]["time"],
                gaps_over_30s=sum(
                    b["time"] - a["time"] > 30 for a, b in zip(rows, rows[1:])
                ),
                gpu_samples=len(utilization),
                mean_sampled_gpu_utilization=statistics.fmean(utilization)
                if utilization
                else None,
                zero_utilization_sample_fraction=(
                    sum(v == 0 for v in utilization) / len(utilization)
                    if utilization
                    else None
                ),
                log_tail_limit_samples=sum(
                    bool(r["logs"].get("tail_limit_reached")) for r in rows
                ),
                note="Sampled utilization, not generation timing or proof of wasted compute.",
            )
        )
    return dict(
        event_counts=dict(counts),
        observed_windows=sorted(windows),
        observed_checkpoints=len(checkpoints),
        sequence_holes=holes,
        malformed_lines=malformed,
        external_baselines=external,
        coverage_issues=dict(Counter(gaps)),
        comparisons=comparisons,
        scope="Same observed ready set only; not a replay of alternate proof dispatch or miner behavior.",
        complete_observation=False,
        limits=[
            "Run prefixes/tails and unobserved attempts are unknown, never zero.",
            "Observed windows are not necessarily consecutive or complete.",
            "Reward variance, bytes and tokens do not establish learning utility.",
            "Actual optimizer inclusion requires reconciled optimizer_group_receipt events.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    print(json.dumps(report(args.files), indent=2))
