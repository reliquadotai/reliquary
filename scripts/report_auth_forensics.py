#!/usr/bin/env python3
"""Summarize validator-private token-auth forensics JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reliquary.validator.auth_forensics import auth_forensics_path


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _fmt_prob(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return "-"


def summarize(records: list[dict[str, Any]], *, top_n: int) -> str:
    windows = {r.get("window_start") for r in records}
    envs = {r.get("env_name") for r in records}
    positive = [r for r in records if r.get("reward_positive")]
    lines = [
        "Token-auth private forensics report",
        f"records: {len(records)}",
        f"windows: {len(windows)}",
        f"envs: {', '.join(sorted(str(e) for e in envs if e)) or '-'}",
        f"reward_positive_records: {len(positive)}",
        "",
        "Top hotkeys",
    ]
    by_hotkey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hotkey[str(record.get("miner_hotkey", ""))].append(record)

    ranked = sorted(
        by_hotkey.items(),
        key=lambda item: (
            sum(1 for r in item[1] if r.get("reward_positive")),
            len(item[1]),
        ),
        reverse=True,
    )
    for hotkey, hotkey_records in ranked[:top_n]:
        h_windows = {r.get("window_start") for r in hotkey_records}
        h_positive = [r for r in hotkey_records if r.get("reward_positive")]
        min_p = min(
            (
                float(r["p_chosen"])
                for r in hotkey_records
                if r.get("p_chosen") is not None
            ),
            default=None,
        )
        example = min(
            hotkey_records,
            key=lambda r: (
                float(r.get("p_chosen", 1.0) or 1.0),
                int(r.get("window_start", 0) or 0),
            ),
        )
        swap = (
            f"{example.get('token_text')!r}"
            f" -> {example.get('argmax_text')!r}"
        )
        lines.append(
            f"- {hotkey}: records={len(hotkey_records)} "
            f"positive={len(h_positive)} windows={len(h_windows)} "
            f"min_p={_fmt_prob(min_p)} example="
            f"w{example.get('window_start')} p{example.get('prompt_idx')} "
            f"r{example.get('rollout_idx')} pos={example.get('completion_pos')} "
            f"{swap}"
        )
    if not ranked:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validator-private all-token auth forensic JSONL files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSONL paths. Defaults to RELIQUARY_AUTH_FORENSICS_PATH/state dir.",
    )
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    paths = args.paths or [auth_forensics_path()]
    records = _load_records(paths)
    print(summarize(records, top_n=max(1, args.top_n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
