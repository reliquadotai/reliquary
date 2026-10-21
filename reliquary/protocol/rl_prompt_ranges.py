"""The rows of each environment's train split that RL may draw prompts from.

A service task's window offers miners one contiguous ``[start, end)`` slice per
environment; the validator publishes it in its miner state and refuses any
prompt outside it. Without an entry the whole train split is offered.

An entry exists where the train split is shared with something RL must not
repeat: rows a distillation corpus already consumed (the policy was trained on
their answers), or rows an evaluation set holds out. Each one records the split
length it was cut against, so a repinned environment of another length fails
closed instead of offering a shifted slice.

A signed-episode env of an async task declares its rows in its own block instead
(``rl_async_episodes``: ``rows = {start, end, exclude}``): ``PromptRows``, a slice
minus excluded rows and intervals, which the validator enforces at lease time and
the miner draws from.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RlPromptRange:
    start: int
    # None: to the end of the split.
    end: int | None
    source_length: int
    why: str


RL_PROMPT_RANGES: dict[str, RlPromptRange] = {
    "reliquary_hard_math_v1": RlPromptRange(
        15_749, None, 23_227,
        "Rows [0, 15749) (tiers 1 and 2) are the hard-math corpus jobs' prompts; the "
        "third tier is RL's."),
    "reliquary_instruction_following_v1": RlPromptRange(
        20_000, None, 29_435,
        "Rows [0, 20000) are the instruction-following corpus job's prompts."),
    "reliquary_code_v1": RlPromptRange(
        200_000, 2_381_806, 2_481_806,
        "Rows [0, 200000) are the two code corpus jobs' prompts, and the last 100,000 "
        "rows are the code evaluation set (reliquary.eval.sets.HELD_OUT)."),
    "reliquary_competitive_code_v1": RlPromptRange(
        4_181, 6_899, 8_792,
        "Train is SFT [0, 4181), RL [4181, 6899), then the v1.1 problems [6899, 8792), "
        "all SFT (the package's layout.use_ranges); the SFT rows are the competitive-code "
        "corpus jobs' prompts."),
    "envscaler_tools_v2": RlPromptRange(
        0, 1_296, 7_825,
        "The scenarios are split by world (the package's corpus layout): RL [0, 1296), 26 "
        "worlds no SFT row comes from; the rest are the SFT ranges and the corpus jobs' prompts."),
}


# Environments whose RL rows knowingly include corpus rows, each with why. The
# science corpus jobs distilled the whole train split, so no row of it is left
# that the policy has not seen answered; it is offered whole until a decision
# reserves part of it or drops the environment from the run.
RL_REUSES_CORPUS_ROWS: dict[str, str] = {
    "reliquary_science_v1": (
        "Both science corpus jobs took the whole train split [0, 10871)."),
}


def rl_prompt_range(environment: str, universe_n: int) -> tuple[int, int]:
    """The ``[start, end)`` slice of ``environment``'s train split RL may draw from."""
    universe_n = int(universe_n)
    entry = RL_PROMPT_RANGES.get(environment)
    if entry is None:
        return (0, universe_n)
    if universe_n != entry.source_length:
        raise ValueError(
            f"{environment}: the RL prompt range was cut against {entry.source_length} "
            f"rows, the installed train split has {universe_n}"
        )
    end = universe_n if entry.end is None else entry.end
    if not 0 <= entry.start < end <= universe_n:
        raise ValueError(f"{environment}: empty or invalid RL prompt range")
    return (entry.start, end)


MAX_EXCLUDED_ENTRIES = 4096


@dataclass(frozen=True)
class PromptRows:
    """The rows RL may draw from: ``[start, end)`` minus ``exclude`` (sorted, disjoint ``[a, b)`` inside it)."""
    start: int
    end: int
    exclude: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError("prompt rows must be a non-empty [start, end)")
        floor = self.start
        for a, b in self.exclude:
            if not floor <= a < b <= self.end:
                raise ValueError("excluded rows must be sorted, disjoint intervals inside [start, end)")
            floor = b
        if len(self) == 0:
            raise ValueError("every row of [start, end) is excluded")

    @classmethod
    def of(cls, value: PromptRows | tuple[int, int]) -> PromptRows:
        """A ``PromptRows``, or a plain ``(start, end)`` slice."""
        return value if isinstance(value, PromptRows) else cls(int(value[0]), int(value[1]))

    def __contains__(self, row: object) -> bool:
        if type(row) is not int or not self.start <= row < self.end:
            return False
        i = bisect.bisect_right(self.exclude, (row, math.inf)) - 1
        return i < 0 or not self.exclude[i][0] <= row < self.exclude[i][1]

    def __len__(self) -> int:
        return self.end - self.start - sum(b - a for a, b in self.exclude)


def parse_prompt_rows(value: Any, where: str) -> PromptRows:
    """``{"start", "end", "exclude"?}``: ``exclude`` lists rows (an int) and ``[a, b)`` intervals, which may come in
    any order but never overlap. ValueError otherwise."""
    def integer(v: Any, label: str) -> int:
        if type(v) is not int or v < 0:
            raise ValueError(f"{where}.{label} must be a non-negative integer")
        return v

    if not isinstance(value, Mapping) or not {"start", "end"} <= set(value) <= {"start", "end", "exclude"}:
        raise ValueError(f"{where} is {{start, end, exclude?}}")
    start, end = integer(value["start"], "start"), integer(value["end"], "end")
    raw = value.get("exclude", [])
    if not isinstance(raw, list) or len(raw) > MAX_EXCLUDED_ENTRIES:
        raise ValueError(f"{where}.exclude must be a list of at most {MAX_EXCLUDED_ENTRIES} rows or [a, b) intervals")
    spans = []
    for item in raw:
        if isinstance(item, list):
            if len(item) != 2:
                raise ValueError(f"{where}.exclude: an interval is [a, b)")
            a, b = integer(item[0], "exclude"), integer(item[1], "exclude")
        else:
            a = integer(item, "exclude")
            b = a + 1
        if not start <= a < b <= end:
            raise ValueError(f"{where}.exclude: {item} is empty or outside [{start}, {end})")
        spans.append((a, b))
    spans.sort()
    for (_, b), (a, _) in itertools.pairwise(spans):
        if a < b:
            raise ValueError(f"{where}.exclude: overlapping entries at row {a}")
    try:
        return PromptRows(start, end, tuple(spans))
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


# Worlds of a world-split environment whose RL rows RL never draws, each with why. The rows stay in the RL range
# (every index keeps naming its scenario); ``rl_prompt_rows`` excludes them from the loaded environment, so the
# miner skips them and the validator refuses a lease of one. A stronger policy can be measured on them later.
RL_EXCLUDED_WORLDS: dict[str, tuple[frozenset[str], str]] = {
    "envscaler_tools_v2": (
        frozenset({"env_147_rl", "env_153_rl", "env_161_rl", "env_163_rl",
                   "env_169_rl", "env_173_rl", "env_181_rl", "env_187_rl"}),
        "Measured on 2026-10-21 on the RL starting policy (teutonic-i-sft-v3, RL sampling, ~8 scenarios x 8 "
        "rollouts per world, 1,741 episodes): these eight RL worlds won none of their episodes, so every group of "
        "theirs is too hard and only costs leases."),
}


def rl_excluded_prompts(environment: str, env: Any) -> frozenset[int]:
    """Rows of ``environment``'s RL range RL never draws: those whose scenario (``env.scenario(row).env_id``) is a
    world of ``RL_EXCLUDED_WORLDS``; empty for an environment without an entry."""
    entry = RL_EXCLUDED_WORLDS.get(environment)
    if entry is None:
        return frozenset()
    start, end = rl_prompt_range(environment, len(env))
    return frozenset(row for row in range(start, end) if env.scenario(row).env_id in entry[0])


def rl_prompt_rows(environment: str, env: Any) -> tuple[int, int] | PromptRows:
    """``rl_prompt_range`` of the loaded ``env``, as ``PromptRows`` minus ``rl_excluded_prompts`` when it has any."""
    start, end = rl_prompt_range(environment, len(env))
    excluded = sorted(rl_excluded_prompts(environment, env))
    if not excluded:
        return (start, end)
    spans: list[tuple[int, int]] = []
    for row in excluded:
        if spans and spans[-1][1] == row:
            spans[-1] = (spans[-1][0], row + 1)
        else:
            spans.append((row, row + 1))
    return PromptRows(start, end, tuple(spans))


__all__ = ["MAX_EXCLUDED_ENTRIES", "RL_EXCLUDED_WORLDS", "RL_PROMPT_RANGES", "RL_REUSES_CORPUS_ROWS", "PromptRows",
           "RlPromptRange", "parse_prompt_rows", "rl_excluded_prompts", "rl_prompt_range", "rl_prompt_rows"]
