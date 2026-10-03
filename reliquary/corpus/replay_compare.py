"""Compare a replayed agentic episode with the trajectory a miner submitted.

Pure: the actions come from a verifiers trace, the replayed observations from
``reliquary.validator.agentic_replay``. A replay certifies an episode when its
final diff is identical and few observations differ after normalization (gate
M2 sets the rules and the tolerance, see ``within_tolerance``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# Durations and timestamps a re-run cannot reproduce. Extended from M2's
# observed mismatches; each rule names what it hides.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\r\n?"), "\n"),                                         # line endings
    (re.compile(r"\b\d+(\.\d+)?s\b"), "<dur>"),                           # "0.75s"
    (re.compile(r"\b\d+(\.\d+)? ?(ms|seconds?|secs?)\b"), "<dur>"),       # "12 ms"
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?"), "<ts>"),  # ISO timestamps
    (re.compile(r"\b0x[0-9a-f]{6,}\b"), "<addr>"),                       # object addresses in reprs
    (re.compile(r"\b[0-9a-f]{7,40}(?= base\b)"), "<commit>"),             # the box's own "base" commit
    (re.compile(r"\.g[0-9a-f]{7,40}\b"), ".g<commit>"),                  # version strings built from it
    (re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r" +\d{1,2} \d{2}:\d{2}:\d{2} [A-Z]{2,5} \d{4}\b"), "<date>"),  # `date`
    (re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r" +\d{1,2} \d{2}:\d{2}:\d{2} \d{4} [+-]\d{4}\b"), "<gitdate>"),   # git log/show Date:
    (re.compile(r"(?<=commit )[0-9a-f]{40}\b"), "<commit>"),             # git log/show full hash
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +\d{2}:\d{2}\b"),
     "<mtime>"),                                                         # ls -l mtimes (checkout time)
)


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: str
    observation: str


@dataclass(frozen=True)
class ReplayReport:
    compared: int
    mismatched: list[int] = field(default_factory=list)
    diff_equal: bool = False

    @property
    def mismatch_share(self) -> float:
        return len(self.mismatched) / self.compared if self.compared else 0.0


def _call_fields(call: dict) -> tuple[str, str]:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or ""), str(function.get("arguments") or "")
    return str(call.get("name") or ""), str(call.get("arguments") or "")


def actions_from_trace(trace: dict) -> list[Action]:
    """Every tool call with the tool message that answered it, in order.

    A tool call with no answering tool message is dropped silently. That is
    fine for gate M2 (honest traces); production must parse actions from the
    TOPLOC-verified tokens with the pinned renderer and refuse such calls
    (spec section 5, N5).
    """
    actions: list[Action] = []
    pending: list[tuple[str, str]] = []
    for node in trace.get("nodes", []):
        message = node.get("message") or {}
        role = message.get("role")
        if role == "assistant":
            pending.extend(_call_fields(c) for c in message.get("tool_calls") or [])
        elif role == "tool" and pending:
            tool, arguments = pending.pop(0)
            content = message.get("content")
            actions.append(Action(tool, arguments, content if isinstance(content, str) else str(content)))
    return actions


def normalize(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


# Per-episode replay tolerance, derived from gate M2's 66 honest episodes
# (docs/design/measurements/2026-10-03-m2-replay-agreement.json): at most 4
# mismatched observations in any episode, at most 20 % (3 of 15) in a short
# one, at most 9.5 % (4 of 42) in one of 40+ observations. The floor covers
# short episodes, the share covers long ones; every M2 episode passes with
# at least 2 observations to spare. It is also the forgery budget: a miner
# may forge up to this many observations in an episode without failing it.
TOLERANCE_FLOOR = 5
TOLERANCE_SHARE = 0.12


def allowed_mismatches(observations: int) -> int:
    """Mismatched observations an episode of ``observations`` may carry."""
    return max(TOLERANCE_FLOOR, math.ceil(TOLERANCE_SHARE * observations))


def within_tolerance(report: ReplayReport) -> bool:
    """True when a replay certifies the episode: identical final diff and no
    more mismatched observations than ``allowed_mismatches`` allows."""
    return report.diff_equal and len(report.mismatched) <= allowed_mismatches(report.compared)


def compare(recorded: Sequence[Action], replayed: Sequence[str],
            recorded_diff: str, replayed_diff: str) -> ReplayReport:
    mismatched = [
        i for i, action in enumerate(recorded)
        if i >= len(replayed) or normalize(action.observation) != normalize(replayed[i])
    ]
    return ReplayReport(compared=len(recorded), mismatched=mismatched,
                        diff_equal=recorded_diff == replayed_diff)
