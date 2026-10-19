"""The rows of each environment's train split that RL may draw prompts from.

A service task's window offers miners one contiguous ``[start, end)`` slice per
environment; the validator publishes it in its miner state and refuses any
prompt outside it. Without an entry the whole train split is offered.

An entry exists where the train split is shared with something RL must not
repeat: rows a distillation corpus already consumed (the policy was trained on
their answers), or rows an evaluation set holds out. Each one records the split
length it was cut against, so a repinned environment of another length fails
closed instead of offering a shifted slice.
"""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["RL_PROMPT_RANGES", "RL_REUSES_CORPUS_ROWS", "RlPromptRange", "rl_prompt_range"]
