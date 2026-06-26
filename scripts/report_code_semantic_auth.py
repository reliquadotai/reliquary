#!/usr/bin/env python3
"""Summarize OpenCode semantic-token shadow telemetry from archives.

Usage examples:

    python scripts/report_code_semantic_auth.py /tmp/window-*.json.gz

    python scripts/report_code_semantic_auth.py --from-r2 --current-window 1234 --n 24

The report reads archive fields written by PR #94:

    code_semantic_auth_findings
    code_semantic_auth_min_prob

It is intentionally read-only. A non-zero finding count is not automatically a
ban signal while CODE_SEMANTIC_AUTH_ENFORCE is false; it is calibration data.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Bucket:
    submissions: int = 0
    flagged_submissions: int = 0
    findings: int = 0
    min_prob: float | None = None
    positive_findings: int = 0
    positive_min_prob: float | None = None

    def add(
        self,
        findings: int,
        min_prob: float | None,
        *,
        positive_findings: int = 0,
        positive_min_prob: float | None = None,
    ) -> None:
        self.submissions += 1
        if findings > 0:
            self.flagged_submissions += 1
        self.findings += int(findings)
        self.positive_findings += int(positive_findings)
        if min_prob is not None:
            p = float(min_prob)
            if self.min_prob is None or p < self.min_prob:
                self.min_prob = p
        if positive_min_prob is not None:
            p = float(positive_min_prob)
            if self.positive_min_prob is None or p < self.positive_min_prob:
                self.positive_min_prob = p

    def as_dict(self) -> dict[str, Any]:
        rate = (
            self.flagged_submissions / self.submissions
            if self.submissions
            else 0.0
        )
        return {
            "submissions": self.submissions,
            "flagged_submissions": self.flagged_submissions,
            "flagged_rate": rate,
            "findings": self.findings,
            "min_prob": self.min_prob,
            "positive_findings": self.positive_findings,
            "positive_min_prob": self.positive_min_prob,
        }


@dataclass
class CodeSemanticAuthSummary:
    windows: int = 0
    entries_seen: int = 0
    entries_matching_env: int = 0
    selected: Bucket = field(default_factory=Bucket)
    runners_up: Bucket = field(default_factory=Bucket)
    by_hotkey: dict[str, Bucket] = field(default_factory=dict)
    by_window: dict[int, Bucket] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "windows": self.windows,
            "entries_seen": self.entries_seen,
            "entries_matching_env": self.entries_matching_env,
            "selected": self.selected.as_dict(),
            "runners_up": self.runners_up.as_dict(),
            "by_hotkey": {
                hotkey: bucket.as_dict()
                for hotkey, bucket in sorted(self.by_hotkey.items())
            },
            "by_window": {
                str(window): bucket.as_dict()
                for window, bucket in sorted(self.by_window.items())
            },
        }


def _read_archive(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def load_archives_from_paths(paths: Iterable[str]) -> list[dict[str, Any]]:
    archives: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.name.endswith((".json", ".json.gz")):
                    archives.append(_read_archive(child))
            continue
        archives.append(_read_archive(path))
    return archives


async def load_archives_from_r2(current_window: int, n: int) -> list[dict[str, Any]]:
    from reliquary.infrastructure.storage import list_recent_datasets

    return await list_recent_datasets(current_window=current_window, n=n)


def _entry_env(archive: dict[str, Any], entry: dict[str, Any]) -> str:
    env = entry.get("env_name")
    if isinstance(env, str) and env:
        return env
    env = archive.get("environment")
    return env if isinstance(env, str) else ""


def _entry_findings(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("code_semantic_auth_findings", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _entry_min_prob(entry: dict[str, Any]) -> float | None:
    raw = entry.get("code_semantic_auth_min_prob")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _entry_positive_findings(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("code_semantic_auth_positive_findings", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _entry_positive_min_prob(entry: dict[str, Any]) -> float | None:
    raw = entry.get("code_semantic_auth_positive_min_prob")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarize_archives(
    archives: Iterable[dict[str, Any]],
    *,
    env_name: str = "opencodeinstruct",
    include_runners_up: bool = True,
) -> CodeSemanticAuthSummary:
    summary = CodeSemanticAuthSummary()
    for archive in archives:
        summary.windows += 1
        try:
            window = int(archive.get("window_start", -1))
        except (TypeError, ValueError):
            window = -1

        groups: list[tuple[str, list[dict[str, Any]]]] = [
            ("selected", list(archive.get("batch") or [])),
        ]
        if include_runners_up:
            groups.append(("runners_up", list(archive.get("runners_up") or [])))

        for group_name, entries in groups:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                summary.entries_seen += 1
                if _entry_env(archive, entry) != env_name:
                    continue
                summary.entries_matching_env += 1
                findings = _entry_findings(entry)
                min_prob = _entry_min_prob(entry)
                positive_findings = _entry_positive_findings(entry)
                positive_min_prob = _entry_positive_min_prob(entry)
                target = (
                    summary.runners_up
                    if group_name == "runners_up"
                    else summary.selected
                )
                target.add(
                    findings,
                    min_prob,
                    positive_findings=positive_findings,
                    positive_min_prob=positive_min_prob,
                )

                hotkey = str(entry.get("hotkey") or "")
                if hotkey:
                    summary.by_hotkey.setdefault(hotkey, Bucket()).add(
                        findings,
                        min_prob,
                        positive_findings=positive_findings,
                        positive_min_prob=positive_min_prob,
                    )
                if window >= 0:
                    summary.by_window.setdefault(window, Bucket()).add(
                        findings,
                        min_prob,
                        positive_findings=positive_findings,
                        positive_min_prob=positive_min_prob,
                    )
    return summary


def _fmt_prob(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def format_text_report(summary: CodeSemanticAuthSummary, *, top_n: int = 12) -> str:
    data = summary.as_dict()
    selected = data["selected"]
    runners_up = data["runners_up"]
    total_submissions = selected["submissions"] + runners_up["submissions"]
    total_flagged = (
        selected["flagged_submissions"] + runners_up["flagged_submissions"]
    )
    total_findings = selected["findings"] + runners_up["findings"]
    total_positive_findings = (
        selected["positive_findings"] + runners_up["positive_findings"]
    )
    min_probs = [
        p for p in (selected["min_prob"], runners_up["min_prob"]) if p is not None
    ]
    min_prob = min(min_probs) if min_probs else None
    positive_min_probs = [
        p for p in (
            selected["positive_min_prob"],
            runners_up["positive_min_prob"],
        ) if p is not None
    ]
    positive_min_prob = min(positive_min_probs) if positive_min_probs else None
    flagged_rate = total_flagged / total_submissions if total_submissions else 0.0

    lines = [
        "OpenCode semantic-token shadow report",
        f"windows: {summary.windows}",
        f"entries_seen: {summary.entries_seen}",
        f"opencode_entries: {summary.entries_matching_env}",
        "",
        (
            "total: "
            f"submissions={total_submissions} "
            f"flagged={total_flagged} "
            f"rate={flagged_rate:.2%} "
            f"findings={total_findings} "
            f"min_prob={_fmt_prob(min_prob)} "
            f"positive_findings={total_positive_findings} "
            f"positive_min_prob={_fmt_prob(positive_min_prob)}"
        ),
        (
            "selected: "
            f"submissions={selected['submissions']} "
            f"flagged={selected['flagged_submissions']} "
            f"rate={selected['flagged_rate']:.2%} "
            f"findings={selected['findings']} "
            f"min_prob={_fmt_prob(selected['min_prob'])} "
            f"positive_findings={selected['positive_findings']} "
            f"positive_min_prob={_fmt_prob(selected['positive_min_prob'])}"
        ),
        (
            "runners_up: "
            f"submissions={runners_up['submissions']} "
            f"flagged={runners_up['flagged_submissions']} "
            f"rate={runners_up['flagged_rate']:.2%} "
            f"findings={runners_up['findings']} "
            f"min_prob={_fmt_prob(runners_up['min_prob'])} "
            f"positive_findings={runners_up['positive_findings']} "
            f"positive_min_prob={_fmt_prob(runners_up['positive_min_prob'])}"
        ),
    ]

    hotkeys = sorted(
        summary.by_hotkey.items(),
        key=lambda item: (
            item[1].flagged_submissions,
            item[1].findings,
            -(item[1].min_prob or 0.0),
        ),
        reverse=True,
    )
    if hotkeys:
        lines.extend(["", f"top_hotkeys_by_findings (top {top_n}):"])
        for hotkey, bucket in hotkeys[:top_n]:
            d = bucket.as_dict()
            lines.append(
                f"- {hotkey}: submissions={d['submissions']} "
                f"flagged={d['flagged_submissions']} "
                f"rate={d['flagged_rate']:.2%} "
                f"findings={d['findings']} "
                f"min_prob={_fmt_prob(d['min_prob'])} "
                f"positive_findings={d['positive_findings']} "
                f"positive_min_prob={_fmt_prob(d['positive_min_prob'])}"
            )

    if total_submissions == 0:
        recommendation = "no OpenCode submissions found in this archive sample"
    elif total_flagged == 0:
        recommendation = "no shadow findings in sample; keep collecting before enforcement"
    else:
        recommendation = "shadow findings present; inspect examples before enabling enforcement"
    lines.extend(["", f"recommendation: {recommendation}"])
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize OpenCode semantic-token shadow telemetry.",
    )
    parser.add_argument("archives", nargs="*", help="Local .json or .json.gz archives")
    parser.add_argument("--env", default="opencodeinstruct", help="Environment name")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--top", type=int, default=12, help="Hotkeys to show")
    parser.add_argument(
        "--no-runners-up",
        action="store_true",
        help="Ignore archive runners_up entries",
    )
    parser.add_argument("--from-r2", action="store_true", help="Fetch recent R2 archives")
    parser.add_argument("--current-window", type=int, default=0, help="Exclusive upper window")
    parser.add_argument("--n", type=int, default=24, help="Number of recent windows")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.from_r2:
        if args.current_window <= 0:
            raise SystemExit("--from-r2 requires --current-window > 0")
        archives = asyncio.run(load_archives_from_r2(args.current_window, args.n))
    else:
        if not args.archives:
            raise SystemExit("pass archive paths or use --from-r2")
        archives = load_archives_from_paths(args.archives)

    summary = summarize_archives(
        archives,
        env_name=args.env,
        include_runners_up=not args.no_runners_up,
    )
    if args.json:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_text_report(summary, top_n=args.top), end="")


if __name__ == "__main__":
    main()
