#!/usr/bin/env python3
"""Summarize validator-private token-auth forensics JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from reliquary.validator.auth_forensics import (
    auth_forensics_path,
    code_semantic_auth_forensics_path,
    code_semantic_signal_bucket,
)

_SWEEP_THRESHOLDS = (
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    3e-6,
    1e-6,
    3e-7,
    1e-7,
    3e-8,
    1e-8,
    3e-9,
    1e-9,
    3e-10,
    1e-10,
)


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


def _prob(record: dict[str, Any]) -> float | None:
    value = record.get("p_chosen")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _window(record: dict[str, Any]) -> int | None:
    value = record.get("window_start")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _signal_bucket(record: dict[str, Any]) -> str:
    return str(
        record.get("signal_bucket")
        or code_semantic_signal_bucket(record.get("label"))
    )


def _label_matches(label: str | None, patterns: Iterable[str]) -> bool:
    if label is None:
        return False
    for pattern in patterns:
        if pattern.endswith("*") and label.startswith(pattern[:-1]):
            return True
        if label == pattern:
            return True
    return False


def filter_records(
    records: list[dict[str, Any]],
    *,
    surface: str,
    code_signal: str,
    labels: list[str],
    hotkeys: list[str],
    reward_positive_only: bool,
    min_prob_lt: float | None,
    since_window: int | None,
    last_n_windows: int | None,
) -> list[dict[str, Any]]:
    out = list(records)
    if since_window is not None:
        out = [r for r in out if (_window(r) or -1) >= since_window]
    if last_n_windows is not None and last_n_windows > 0:
        windows = sorted({w for r in out if (w := _window(r)) is not None})
        if windows:
            keep = set(windows[-last_n_windows:])
            out = [r for r in out if _window(r) in keep]
    if hotkeys:
        wanted = set(hotkeys)
        out = [r for r in out if str(r.get("miner_hotkey", "")) in wanted]
    if reward_positive_only:
        out = [r for r in out if bool(r.get("reward_positive"))]
    if min_prob_lt is not None:
        out = [
            r for r in out
            if (p := _prob(r)) is not None and p < min_prob_lt
        ]
    if labels:
        out = [
            r for r in out
            if _label_matches(str(r.get("label", "")), labels)
        ]
    if surface == "code-semantic" and code_signal != "all":
        if code_signal == "non-low":
            out = [r for r in out if _signal_bucket(r) != "low"]
        elif code_signal == "high-or-review":
            out = [
                r for r in out
                if _signal_bucket(r) in {"high", "review"}
            ]
        else:
            out = [r for r in out if _signal_bucket(r) == code_signal]
    return out


def _count_positive(records: list[dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("reward_positive"))


def _count_counterfactual_flips(records: list[dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("counterfactual_reward_flipped"))


def _labels_text(records: list[dict[str, Any]], *, limit: int = 5) -> str:
    labels = Counter(str(r.get("label", "-")) for r in records)
    if not labels:
        return "-"
    parts = [f"{label}:{count}" for label, count in labels.most_common(limit)]
    if len(labels) > limit:
        parts.append("...")
    return ", ".join(parts)


def _compact_context(record: dict[str, Any], *, max_chars: int = 180) -> str:
    raw = record.get("code_context") or record.get("completion_context") or ""
    text = " ".join(str(raw).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _threshold_sweep(records: list[dict[str, Any]]) -> list[str]:
    lines = ["", "Threshold sweep"]
    for threshold in _SWEEP_THRESHOLDS:
        rows = [
            r for r in records
            if (p := _prob(r)) is not None and p < threshold
        ]
        if not rows:
            continue
        lines.append(
            f"- p<{threshold:g}: records={len(rows)} "
            f"positive={_count_positive(rows)} "
            f"hotkeys={len({r.get('miner_hotkey') for r in rows})}"
        )
    if len(lines) == 1:
        lines.append("- none")
    return lines


def _interpretation(records: list[dict[str, Any]], *, surface: str) -> list[str]:
    lines = ["", "Interpretation"]
    if not records:
        lines.append(
        "- No records in this view. Wait for more live windows before "
        "drawing conclusions."
        )
        return lines

    windows = len({r.get("window_start") for r in records})
    positive = _count_positive(records)
    if surface == "all-token-shadow":
        lines.append(
            "- All-token shadow is a broad anomaly smoke test. Do not use it "
            "as a punitive gate while reward-positive records remain."
        )
        lines.append(
            f"- Current view has {positive} reward-positive records across "
            f"{windows} windows; keep it telemetry-only."
        )
        lines.append(
            "- Use it to spot repeated hotkeys or sudden distribution shifts, "
            "not to slash individual submissions."
        )
        return lines

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[_signal_bucket(record)].append(record)
    high = buckets.get("high", [])
    review = buckets.get("review", [])
    low = buckets.get("low", [])
    flips = _count_counterfactual_flips(records)
    checked = sum(1 for r in records if r.get("counterfactual_checked"))
    repeated_flip_hotkeys = [
        hotkey for hotkey, rows in _group_by_hotkey(records).items()
        if _count_counterfactual_flips(rows) >= 2
        and len({r.get("window_start") for r in rows}) >= 2
    ]

    lines.append(
        f"- Read raw counts carefully: high={len(high)}, "
        f"review={len(review)}, low={len(low)}, "
        f"reward_positive={positive}, windows={windows}."
    )
    if low and len(low) >= max(1, len(records) // 2):
        lines.append(
            "- Low-signal constant:string dominates this view; that is "
            "inspection material, not enforcement evidence."
        )
    if flips:
        lines.append(
            f"- Counterfactual reward flips found: {flips}/{checked} checked. "
            "Inspect these first."
        )
        if repeated_flip_hotkeys:
            lines.append(
                "- Repeated flip hotkeys crossed the investigation threshold: "
                + ", ".join(sorted(repeated_flip_hotkeys))
            )
        else:
            lines.append(
                "- A single flip is not enough for global enforcement; look "
                "for repeats across windows/hotkeys."
            )
    elif high or review:
        lines.append(
            "- High/review records exist but no reward-flip evidence is "
            "present in this view; keep shadow mode and inspect examples."
        )
    else:
        lines.append(
            "- No high/review records in this view; do not act on it."
        )
    lines.append(
        "- Cadence: run this after at least 50 fresh windows or daily; react "
        "immediately only when the same hotkey has 2+ counterfactual reward "
        "flips in 2+ windows."
    )
    return lines


def _group_by_hotkey(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_hotkey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hotkey[str(record.get("miner_hotkey", ""))].append(record)
    return by_hotkey


def summarize(
    records: list[dict[str, Any]],
    *,
    top_n: int,
    title: str = "Token-auth private forensics report",
    surface: str = "all-token-shadow",
    examples: int = 0,
    include_interpretation: bool = True,
    threshold_sweep: bool = False,
) -> str:
    windows = {r.get("window_start") for r in records}
    envs = {r.get("env_name") for r in records}
    positive = [r for r in records if r.get("reward_positive")]
    lines = [
        title,
        f"records: {len(records)}",
        f"windows: {len(windows)}",
        f"envs: {', '.join(sorted(str(e) for e in envs if e)) or '-'}",
        f"reward_positive_records: {len(positive)}",
    ]

    if surface == "code-semantic":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            buckets[_signal_bucket(record)].append(record)
        lines.extend(["", "Code signal buckets"])
        for bucket in ("high", "review", "low", "unknown"):
            rows = buckets.get(bucket, [])
            if not rows:
                continue
            lines.append(
                f"- {bucket}: records={len(rows)} "
                f"positive={_count_positive(rows)} "
                f"windows={len({r.get('window_start') for r in rows})} "
                f"labels={_labels_text(rows)}"
            )
        checked = sum(1 for r in records if r.get("counterfactual_checked"))
        if checked:
            lines.extend([
                "",
                "Counterfactual regrade",
                f"- checked={checked} reward_flips={_count_counterfactual_flips(records)}",
            ])

    lines.extend(["", "Top hotkeys"])
    ranked = sorted(
        _group_by_hotkey(records).items(),
        key=lambda item: (
            _count_counterfactual_flips(item[1]),
            _count_positive(item[1]),
            len(item[1]),
        ),
        reverse=True,
    )
    for hotkey, hotkey_records in ranked[:top_n]:
        h_windows = {r.get("window_start") for r in hotkey_records}
        min_p = min(
            (p for r in hotkey_records if (p := _prob(r)) is not None),
            default=None,
        )
        example = min(
            hotkey_records,
            key=lambda r: (
                _prob(r) if _prob(r) is not None else 1.0,
                int(r.get("window_start", 0) or 0),
            ),
        )
        swap = (
            f"{example.get('token_text')!r}"
            f" -> {example.get('argmax_text')!r}"
        )
        label = example.get("label")
        label_text = f" label={label}" if label else ""
        signal_text = (
            f" signal={_signal_bucket(example)}"
            if surface == "code-semantic"
            else ""
        )
        cf_text = ""
        if any(r.get("counterfactual_checked") for r in hotkey_records):
            cf_text = (
                f" cf_checked={sum(1 for r in hotkey_records if r.get('counterfactual_checked'))}"
                f" cf_flips={_count_counterfactual_flips(hotkey_records)}"
            )
        lines.append(
            f"- {hotkey}: records={len(hotkey_records)} "
            f"positive={_count_positive(hotkey_records)} windows={len(h_windows)} "
            f"min_p={_fmt_prob(min_p)}{cf_text} example="
            f"w{example.get('window_start')} p{example.get('prompt_idx')} "
            f"r{example.get('rollout_idx')} pos={example.get('completion_pos')} "
            f"{swap}{label_text}{signal_text}"
        )
    if not ranked:
        lines.append("- none")

    if examples > 0 and records:
        rows = sorted(
            records,
            key=lambda r: (
                not bool(r.get("counterfactual_reward_flipped")),
                not bool(r.get("reward_positive")),
                _prob(r) if _prob(r) is not None else 1.0,
                int(r.get("window_start", 0) or 0),
            ),
        )
        lines.extend(["", "Examples"])
        for record in rows[:examples]:
            cf_text = ""
            if record.get("counterfactual_checked"):
                cf_text = (
                    f" cf_reward={_fmt_prob(record.get('counterfactual_reward'))}"
                    f" cf_flip={bool(record.get('counterfactual_reward_flipped'))}"
                )
            lines.append(
                f"- w{record.get('window_start')} {record.get('miner_hotkey')} "
                f"reward={record.get('rollout_reward')} "
                f"positive={bool(record.get('reward_positive'))} "
                f"label={record.get('label')} signal={_signal_bucket(record)} "
                f"p={_fmt_prob(record.get('p_chosen'))} "
                f"{record.get('token_text')!r}->{record.get('argmax_text')!r}"
                f"{cf_text} ctx={_compact_context(record)}"
            )

    if threshold_sweep:
        lines.extend(_threshold_sweep(records))
    if include_interpretation:
        lines.extend(_interpretation(records, surface=surface))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize validator-private token-auth forensic JSONL files."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSONL paths. Defaults to the selected private forensics path.",
    )
    parser.add_argument(
        "--surface",
        choices=("all-token-shadow", "code-semantic"),
        default="all-token-shadow",
        help="Private forensic surface to read when no paths are supplied.",
    )
    parser.add_argument(
        "--code-signal",
        choices=(
            "all",
            "high",
            "review",
            "low",
            "unknown",
            "non-low",
            "high-or-review",
        ),
        default="all",
        help="Filter code-semantic records by signal bucket.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Filter by exact label, or prefix with '*' suffix, e.g. keyword:*.",
    )
    parser.add_argument("--hotkey", action="append", default=[])
    parser.add_argument("--reward-positive-only", action="store_true")
    parser.add_argument("--min-prob-lt", type=float)
    parser.add_argument("--since-window", type=int)
    parser.add_argument("--last-n-windows", type=int)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--no-interpretation", action="store_true")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    default_path = (
        code_semantic_auth_forensics_path()
        if args.surface == "code-semantic"
        else auth_forensics_path()
    )
    paths = args.paths or [default_path]
    records = filter_records(
        _load_records(paths),
        surface=args.surface,
        code_signal=args.code_signal,
        labels=args.label,
        hotkeys=args.hotkey,
        reward_positive_only=args.reward_positive_only,
        min_prob_lt=args.min_prob_lt,
        since_window=args.since_window,
        last_n_windows=args.last_n_windows,
    )
    title = (
        "Code-semantic private forensics report"
        if args.surface == "code-semantic"
        else "All-token private forensics report"
    )
    print(
        summarize(
            records,
            top_n=max(1, args.top_n),
            title=title,
            surface=args.surface,
            examples=max(0, args.examples),
            include_interpretation=not args.no_interpretation,
            threshold_sweep=args.threshold_sweep,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
