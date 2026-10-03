"""A trajectory's actions and observations, read from its proven tokens.

Spec §5 N5 / F5: production never reads a miner's trace. Every tool call is
parsed from an assistant span with the pinned renderer, every observation
from the segment after it, and that segment must be exactly what the pinned
renderer emits for those observations, so the tokens the student trains on
and the actions the replay runs are one and the same thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from reliquary.corpus.checks import REASON_BAD_TURNS
from reliquary.corpus.replay_compare import Action

REASON_BAD_OBSERVATION = "bad_observation"
REASON_UNANSWERED_TOOL_CALL = "unanswered_tool_call"
REASON_BAD_STOP = "bad_stop"


class TrajectoryRefused(ValueError):
    def __init__(self, reason: str, detail: dict | None = None) -> None:
        super().__init__(f"{reason}: {detail or {}}")
        self.reason = reason
        self.detail = dict(detail or {})


class TurnRenderer(Protocol):
    terminator_id: int
    stop_ids: frozenset[int]

    def initial_ids(self, prompt: str) -> list[int]: ...

    def tool_calls(self, completion_ids: Sequence[int]) -> list[tuple[str, str]]: ...

    def observations(self, segment_ids: Sequence[int]) -> list[str]: ...

    def next_prompt(self, prompt_ids: Sequence[int], completion_ids: Sequence[int],
                    observations: Sequence[str]) -> list[int] | None: ...

    def assistant_message(self, completion_ids: Sequence[int]) -> dict: ...


@dataclass(frozen=True)
class ParsedTurn:
    calls: tuple[tuple[str, str], ...]
    observations: tuple[str, ...]


@dataclass(frozen=True)
class ParsedTrajectory:
    turns: tuple[ParsedTurn, ...]
    actions: tuple[Action, ...]


def parse_trajectory(renderer: TurnRenderer, *, prompt_ids: Sequence[int],
                     tokens: Sequence[int], spans: Sequence[tuple[int, int]],
                     stop: str) -> ParsedTrajectory:
    """The trajectory's turns and actions, or ``TrajectoryRefused``.

    ``spans`` are assistant spans in ``tokens`` coordinates, already checked
    ordered and in bounds (``checks.check_turn_spans``)."""
    if not spans:
        raise TrajectoryRefused(REASON_BAD_TURNS, {"turns": 0})
    full = list(prompt_ids) + list(tokens)
    offset = len(prompt_ids)
    turns: list[ParsedTurn] = []
    actions: list[Action] = []
    for k, (start, end) in enumerate(spans):
        completion = list(tokens[start:end])
        calls = tuple(renderer.tool_calls(completion))
        if k == len(spans) - 1:
            if end != len(tokens):
                raise TrajectoryRefused(REASON_BAD_TURNS, {"trailing_tokens": len(tokens) - end})
            if stop == "agent_completed" and calls:
                raise TrajectoryRefused(REASON_BAD_STOP, {"turn": k, "calls": len(calls)})
            if stop == "context_length":
                actions += [Action(name, arguments, None) for name, arguments in calls]
            turns.append(ParsedTurn(calls, ()))
            continue
        next_start = spans[k + 1][0]
        observations = tuple(renderer.observations(tokens[end:next_start]))
        if not calls:
            raise TrajectoryRefused(REASON_BAD_OBSERVATION, {"turn": k, "why": "no tool call"})
        if len(observations) < len(calls):
            raise TrajectoryRefused(REASON_UNANSWERED_TOOL_CALL,
                                    {"turn": k, "calls": len(calls), "observations": len(observations)})
        if len(observations) > len(calls):
            raise TrajectoryRefused(REASON_BAD_OBSERVATION,
                                    {"turn": k, "calls": len(calls), "observations": len(observations)})
        expected = renderer.next_prompt(full[:offset + start], completion, observations)
        if expected is None or expected != full[:offset + next_start]:
            raise TrajectoryRefused(REASON_BAD_OBSERVATION,
                                    {"turn": k, "why": "segment is not the pinned rendering"})
        turns.append(ParsedTurn(calls, observations))
        actions += [Action(name, arguments, observation)
                    for (name, arguments), observation in zip(calls, observations)]
    return ParsedTrajectory(tuple(turns), tuple(actions))
