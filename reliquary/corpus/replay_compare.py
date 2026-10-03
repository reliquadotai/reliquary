"""Compare a replayed agentic episode with the trajectory a miner submitted.

Pure: the actions come from a verifiers trace, the replayed observations from
``reliquary.validator.agentic_replay``. A replay certifies an episode when its
final diff is identical and few observations differ after normalization (gate
M2 sets the rules and the tolerance).
"""

from __future__ import annotations

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
    """Every tool call with the tool message that answered it, in order."""
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


def compare(recorded: Sequence[Action], replayed: Sequence[str],
            recorded_diff: str, replayed_diff: str) -> ReplayReport:
    mismatched = [
        i for i, action in enumerate(recorded)
        if i >= len(replayed) or normalize(action.observation) != normalize(replayed[i])
    ]
    return ReplayReport(compared=len(recorded), mismatched=mismatched,
                        diff_equal=recorded_diff == replayed_diff)
