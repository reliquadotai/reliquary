"""One verdict for one submission, from state passed in explicitly.

The order of the checks is the order of their cost: identity and bookkeeping
first, then the cheap content checks, and a slot is only consumed once the
submission has earned it.
"""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

from reliquary.corpus.checks import (
    check_completion_count,
    check_duplicates,
    check_termination,
    check_token_budget,
)
from reliquary.corpus.job import PROMPT_ORDER_MINER_WALK, JobSpec
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, walk_index


@dataclass(frozen=True, slots=True)
class Verdict:
    accepted: bool
    reason: str
    slots_remaining: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def admit(
    job: JobSpec,
    *,
    hotkey: str,
    cursor: int,
    prompt_index: int,
    checkpoint_sha256: str,
    token_counts: Sequence[int],
    terminations: Sequence[str],
    digests: Sequence[str],
    slots: SlotLedger,
    cursors: CursorLedger,
    seen: AbstractSet[str],
) -> Verdict:
    """Decide one submission, consuming a slot and a cursor step when earned."""
    if slots.is_complete:
        return Verdict(False, "job_complete")

    if checkpoint_sha256 != job.checkpoint_sha256:
        return Verdict(
            False,
            "checkpoint_mismatch",
            detail={"expected": job.checkpoint_sha256, "got": checkpoint_sha256},
        )

    if job.prompt_order == PROMPT_ORDER_MINER_WALK:
        expected_cursor = cursors.expected(hotkey)
        if cursor != expected_cursor:
            return Verdict(
                False,
                "bad_cursor",
                detail={"expected": expected_cursor, "got": cursor},
            )
        expected_index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
        if prompt_index != expected_index:
            return Verdict(
                False,
                "prompt_mismatch",
                detail={"expected": expected_index, "got": prompt_index},
            )
    elif prompt_index < 0 or prompt_index >= job.prompt_count:
        return Verdict(
            False,
            "prompt_mismatch",
            detail={"prompt_count": job.prompt_count, "got": prompt_index},
        )

    for result in (
        check_completion_count(len(token_counts), job.sampling),
        check_token_budget(token_counts, job.sampling),
        check_termination(terminations),
        check_duplicates(digests, seen),
    ):
        if not result.ok:
            return Verdict(False, result.reason or "", detail=dict(result.detail))

    # Past this point the miner really did answer the prompt its walk named,
    # so the cursor moves whether or not a slot was still free. Skipping a
    # prompt must never be cheaper than answering it.
    if slots.is_full(prompt_index):
        _advance(job, cursors, hotkey)
        return Verdict(
            False,
            "prompt_full",
            slots_remaining=0,
            detail={"prompt_index": prompt_index},
        )

    remaining = slots.consume(prompt_index)
    _advance(job, cursors, hotkey)
    return Verdict(True, "accepted", slots_remaining=remaining)


def _advance(job: JobSpec, cursors: CursorLedger, hotkey: str) -> None:
    if job.prompt_order == PROMPT_ORDER_MINER_WALK:
        cursors.advance(hotkey)
