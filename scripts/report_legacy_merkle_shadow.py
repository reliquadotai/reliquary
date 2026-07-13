#!/usr/bin/env python3
"""Summarize wire-v1 Merkle shadow checks from validator structured logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def _payload_from_line(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _submission_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload.get("hotkey"),
        payload.get("window_n"),
        payload.get("prompt_idx"),
        payload.get("merkle_root_lead"),
        payload.get("t_arrival"),
    )


def summarize(
    lines: Iterable[str],
    *,
    min_checks: int = 500,
    min_hotkeys: int = 5,
    min_windows: int = 24,
    required_envs: Iterable[str] = (),
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    outcomes: dict[tuple[Any, ...], str] = {}
    for line in lines:
        payload = _payload_from_line(line)
        if payload is None:
            continue
        if payload.get("event") != "validator_submit_lifecycle":
            continue
        stage = payload.get("stage")
        if stage == "legacy_merkle_checked":
            checks.append(payload)
        elif stage in {"candidate_accepted", "candidate_rejected"}:
            outcome = (
                "accepted"
                if stage == "candidate_accepted"
                else str(payload.get("reject_reason") or "rejected_unknown")
            )
            outcomes[_submission_key(payload)] = outcome

    statuses = Counter(str(row.get("legacy_merkle_status")) for row in checks)
    environments = Counter(
        str(row.get("submission_env_name"))
        for row in checks
        if row.get("submission_env_name")
    )
    versions = Counter(
        str(row.get("protocol_version"))
        for row in checks
        if row.get("protocol_version") is not None
    )
    windows = {row.get("window_n") for row in checks}
    hotkeys = {str(row.get("hotkey")) for row in checks if row.get("hotkey")}
    mismatch_outcomes: Counter[str] = Counter()
    mismatch_hotkeys: Counter[str] = Counter()
    for row in checks:
        if row.get("legacy_merkle_status") == "match":
            continue
        mismatch_outcomes[outcomes.get(_submission_key(row), "pending_or_unknown")] += 1
        hotkey = str(row.get("hotkey") or "unknown")
        mismatch_hotkeys[hotkey[:12]] += 1

    required = {str(env) for env in required_envs}
    missing_envs = sorted(required - set(environments))
    total = len(checks)
    blockers = []
    if total < min_checks:
        blockers.append(f"checks<{min_checks}")
    if len(hotkeys) < min_hotkeys:
        blockers.append(f"hotkeys<{min_hotkeys}")
    if len(windows) < min_windows:
        blockers.append(f"windows<{min_windows}")
    if statuses["mismatch"]:
        blockers.append("unresolved_mismatches")
    if statuses["error"]:
        blockers.append("compute_errors")
    if missing_envs:
        blockers.append("missing_required_environments")

    return {
        "checks": total,
        "matches": statuses["match"],
        "mismatches": statuses["mismatch"],
        "errors": statuses["error"],
        "match_rate": (statuses["match"] / total if total else None),
        "distinct_hotkeys": len(hotkeys),
        "distinct_windows": len(windows),
        "environments": dict(sorted(environments.items())),
        "protocol_versions": dict(sorted(versions.items())),
        "mismatch_outcomes": dict(sorted(mismatch_outcomes.items())),
        "mismatch_hotkey_prefixes": dict(sorted(mismatch_hotkeys.items())),
        "missing_required_environments": missing_envs,
        "ready_to_enforce": not blockers,
        "enforcement_blockers": blockers,
    }


def _iter_inputs(paths: list[str]) -> Iterable[str]:
    if not paths:
        yield from sys.stdin
        return
    for value in paths:
        with Path(value).open(encoding="utf-8", errors="replace") as handle:
            yield from handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="validator log files; stdin if omitted")
    parser.add_argument("--min-checks", type=int, default=500)
    parser.add_argument("--min-hotkeys", type=int, default=5)
    parser.add_argument("--min-windows", type=int, default=24)
    parser.add_argument("--required-env", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = summarize(
        _iter_inputs(args.logs),
        min_checks=args.min_checks,
        min_hotkeys=args.min_hotkeys,
        min_windows=args.min_windows,
        required_envs=args.required_env,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "legacy-merkle shadow: "
            f"checks={result['checks']} matches={result['matches']} "
            f"mismatches={result['mismatches']} errors={result['errors']} "
            f"hotkeys={result['distinct_hotkeys']} "
            f"windows={result['distinct_windows']}"
        )
        print(f"environments={result['environments']}")
        print(f"protocol_versions={result['protocol_versions']}")
        print(f"mismatch_outcomes={result['mismatch_outcomes']}")
        print(f"ready_to_enforce={result['ready_to_enforce']}")
        if result["enforcement_blockers"]:
            print(f"blockers={result['enforcement_blockers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
