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
    (re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:s| ?(?:ms|secs?|seconds?))(?![\w.])"), "<dur>"),  # "0.75s", "12 s"
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?"), "<ts>"),  # ISO timestamps
    (re.compile(r"(?<=at )0x[0-9a-f]{6,}\b"), "<addr>"),                 # object addresses in reprs only
    (re.compile(r"\b[0-9a-f]{7,40}(?= base\b)"), "<commit>"),             # the box's own "base" commit
    (re.compile(r"\.g[0-9a-f]{7,40}\b"), ".g<commit>"),                  # version strings built from it
    (re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r" +\d{1,2} \d{2}:\d{2}:\d{2} [A-Z]{2,5} \d{4}\b"), "<date>"),  # `date`
    (re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r" +\d{1,2} \d{2}:\d{2}:\d{2} \d{4} [+-]\d{4}\b"), "<gitdate>"),   # git log/show Date:
    (re.compile(r"(?m)^commit [0-9a-f]{40}\b"), "commit <commit>"),      # git log/show header hash
    (re.compile(r"(?m)^([-dlcbps][-rwxsStT]{9}[.+@]?[ \t]+\d+[ \t]+\S+[ \t]+\S+[ \t]+\d+[ \t]+)"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +\d{2}:\d{2}(?!\d)"),
     r"\1<mtime>"),                                                      # ls -l mtime column (checkout time)
    # B1 (2026-10-04), from honest 27B mismatches replayed twice on one host:
    (re.compile(r"(?m)^Took \d+m\d+(?:\.\d+)?s$"), "Took <dur>"),        # behave's run time
    (re.compile(r"(?<=<dur>) \(\d+:\d{2}:\d{2}\)"), ""),                  # pytest's clock past 60 s
    (re.compile(r"(?m)^<dur> (?:call|setup|teardown) +\S+\.py::\S*?(?:\[[^\]\n]*\])?$"),
     "<dur> <slowest test>"),                                            # pytest --durations entries
    (re.compile(r"(?m)^\(\d+ durations < <dur> hidden\.  Use -vv to show these durations\.\)$"),
     "(<n> durations < <dur> hidden.  Use -vv to show these durations.)"),  # ...and how many it hid
    (re.compile(r"(?m)^\[(\d+)/(\d+)\] (?:Compiling (?:C|C\+\+|Cython|Fortran) (?:object|source) \S+"
                r"|Linking (?:static )?target \S+"
                r"|Generating \S+ with a custom command(?: \(wrapped by meson to [^)\n]*\))?)$"),
     r"[\1/\2] <ninja step>"),                                            # parallel build: order of steps
    # verifiers' bash harness installs the latest uv in the box (unpinned), so a
    # replay after a uv release lists another version: `pip list`'s uv row only.
    (re.compile(r"(?m)^uv( {2,})\d+(?:\.\d+)+(?:(?:a|b|rc|\.post|\.dev)\d+)?$"), r"uv\1<uv version>"),
)

# A line `grep -r`/`find` prints: a path with a directory part, then nothing,
# or a `:`/`-` field (line number, matched text). Directory order is the host
# filesystem's (xfs keeps insertion order, ext4 hashes with a per-filesystem
# seed), so another host lists the same hits in another order.
_PATH_LINE = re.compile(r"(?:\.{1,2}/|/)?[\w.@+-]+(?:/[\w.@+-]+)+/?(?:[:-].*)?")


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: str
    # None: executed in the box but never answered in the tokens (a final
    # turn cut by context_length), so replayed and not compared.
    observation: str | None


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


def canonical(text: str) -> str:
    """``normalize``, then each maximal run of consecutive path lines sorted:
    hits may move within their run, never across another line, and none may
    change, appear or vanish."""
    lines = normalize(text).split("\n")
    out: list[str] = []
    run: list[str] = []
    for line in lines:
        if _PATH_LINE.fullmatch(line):
            run.append(line)
            continue
        out.extend(sorted(run))
        run = []
        out.append(line)
    out.extend(sorted(run))
    return "\n".join(out)


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
    compared = [i for i, action in enumerate(recorded) if action.observation is not None]
    mismatched = [
        i for i in compared
        if i >= len(replayed) or canonical(recorded[i].observation) != canonical(replayed[i])
    ]
    return ReplayReport(compared=len(compared), mismatched=mismatched,
                        diff_equal=recorded_diff == replayed_diff)
