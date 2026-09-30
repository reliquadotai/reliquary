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
    REASON_PROMPT_NOT_FULL,
    CheckResult,
    check_completion_count,
    check_proof_shape,
    check_duplicates,
    check_termination,
    check_token_budget,
)
from reliquary.corpus.job import PROMPT_ORDER_MINER_WALK, JobSpec
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, job_walk_index


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
    last_token_ids: Sequence[int],
    digests: Sequence[str],
    slots: SlotLedger,
    cursors: CursorLedger,
    seen: AbstractSet[str],
    proof_counts: Sequence[int] | None = None,
    proof_chunk_tokens: int | None = None,
) -> Verdict:
    """Decide one submission, consuming a slot and a cursor step when earned.

    REQUIRED of the caller: ``token_counts``, ``last_token_ids`` and
    ``digests`` must be derived by the validator from the submitted token
    arrays, never copied from what the miner declared. Pass declared digests
    and the duplicate check is decorative.

    These are facts about the tokens, never a label ABOUT them: there is no
    ``terminations`` parameter, so "the caller forwarded the miner's declared
    termination" is unrepresentable rather than merely tested for.
    ``check_termination`` derives the label from ``last_token_ids`` and the
    job's ``eos_token_id``. ``proof_counts`` must likewise be the lengths of the
    received proof lists, counted by the validator.
    """
    # The three sequences describe the same completions, so a disagreement in
    # length means some completion would be paid for without ever being checked.
    lengths = {
        "token_counts": len(token_counts),
        "digests": len(digests),
        "last_token_ids": len(last_token_ids),
    }
    if len(set(lengths.values())) != 1:
        return Verdict(False, "malformed_submission", detail=lengths)

    if proof_chunk_tokens is not None and proof_counts is None:
        return Verdict(False, "malformed_submission", detail={"proof_counts": None})

    if slots.is_complete:
        return Verdict(False, "job_complete")

    if checkpoint_sha256 != job.checkpoint_sha256:
        return Verdict(
            False,
            "checkpoint_mismatch",
            detail={"expected": job.checkpoint_sha256, "got": checkpoint_sha256},
        )

    if job.prompt_order == PROMPT_ORDER_MINER_WALK:
        refused = _walk_position_refusal(job, cursors, hotkey, cursor, prompt_index)
        if refused is not None:
            return refused
    elif not job.owns(prompt_index):
        return Verdict(False, "prompt_mismatch", detail=out_of_range_detail(job, prompt_index))

    # Callables, not results: a junk submission is refused on its first failing
    # check rather than walked once per check.
    for check in (
        lambda: check_completion_count(len(token_counts), job.sampling),
        lambda: check_token_budget(token_counts, job.sampling),
        lambda: check_termination(
            token_counts,
            last_token_ids,
            sampling=job.sampling,
            eos_token_id=job.eos_token_id,
        ),
        lambda: check_duplicates(digests, seen),
        lambda: (
            check_proof_shape(token_counts, proof_counts, proof_chunk_tokens)
            if proof_chunk_tokens is not None
            else CheckResult(ok=True)
        ),
    ):
        result = check()
        if not result.ok:
            return Verdict(False, result.reason or "", detail=dict(result.detail))

    # Past this point the miner really did answer the prompt its walk named,
    # so the cursor moves whether or not a slot was still free. This prices a
    # step at n * min_new_tokens tokens of work rather than making it free; it
    # does NOT make skipping cost what answering costs, because a miner can
    # always emit exactly the minimum. What closes the gap is the audit tier:
    # junk completions fail token authenticity. See the spec's threat model.
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


# The most walk steps one skip may cover, and how far `skip_target` looks.
# Bounds the validator's work per skip; a longer run of full prompts is
# crossed in several skips.
MAX_SKIP_STEPS = 256


def skip_target(job: JobSpec, hotkey: str, cursor: int, slots: SlotLedger) -> int:
    """The first cursor after ``cursor`` whose walk position has a free slot,
    looking at most ``MAX_SKIP_STEPS`` ahead; ``cursor + MAX_SKIP_STEPS`` if
    every one of those is full."""
    for step in range(cursor + 1, cursor + MAX_SKIP_STEPS):
        if not slots.is_full(job_walk_index(job, hotkey, step)):
            return step
    return cursor + MAX_SKIP_STEPS


def skip_refusal(
    job: JobSpec,
    *,
    hotkey: str,
    cursor: int,
    prompt_index: int,
    to_cursor: int,
    slots: SlotLedger,
    cursors: CursorLedger,
) -> Verdict | None:
    """Why this skip may not happen, or None. Reads the ledgers, never moves
    them, so it can be asked of a shared cached state."""
    if slots.is_complete:
        return Verdict(False, "job_complete")
    if job.prompt_order != PROMPT_ORDER_MINER_WALK:
        # A free job's cursor never moves, so there is no step to give up.
        return Verdict(False, "malformed_submission", detail={"prompt_order": job.prompt_order})
    refused = _walk_position_refusal(job, cursors, hotkey, cursor, prompt_index)
    if refused is not None:
        return refused
    if not 1 <= to_cursor - cursor <= MAX_SKIP_STEPS:
        return Verdict(
            False,
            "malformed_submission",
            detail={"cursor": cursor, "to_cursor": to_cursor, "max_skip_steps": MAX_SKIP_STEPS},
        )
    # EVERY position crossed must be full: a skip never steps over a prompt
    # the miner could have answered.
    for step in range(cursor, to_cursor):
        index = job_walk_index(job, hotkey, step)
        remaining = slots.remaining(index)
        if remaining:
            return Verdict(
                False,
                REASON_PROMPT_NOT_FULL,
                slots_remaining=remaining,
                detail={"cursor": step, "prompt_index": index, "slots_remaining": remaining},
            )
    return None


def skip(
    job: JobSpec,
    *,
    hotkey: str,
    cursor: int,
    prompt_index: int,
    to_cursor: int,
    slots: SlotLedger,
    cursors: CursorLedger,
) -> Verdict:
    """Step over the full walk positions [cursor, to_cursor), landing on
    ``to_cursor``: what that many ``prompt_full`` refusals do, one ``_advance``
    per position, without the generations they used to cost. No slot is
    consumed and no digest is seen."""
    refused = skip_refusal(
        job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
        to_cursor=to_cursor, slots=slots, cursors=cursors,
    )
    if refused is not None:
        return refused
    for _ in range(cursor, to_cursor):
        _advance(job, cursors, hotkey)
    return Verdict(True, "accepted", slots_remaining=0)


def _walk_position_refusal(
    job: JobSpec, cursors: CursorLedger, hotkey: str, cursor: int, prompt_index: int
) -> Verdict | None:
    """The walk's own rule, shared by ``admit`` and ``skip``: the cursor is the
    ledger's, and the index is the one the walk names there."""
    expected_cursor = cursors.expected(hotkey)
    if cursor != expected_cursor:
        return Verdict(
            False,
            "bad_cursor",
            detail={"expected": expected_cursor, "got": cursor},
        )
    expected_index = job_walk_index(job, hotkey, cursor)
    if prompt_index != expected_index:
        return Verdict(
            False,
            "prompt_mismatch",
            detail={"expected": expected_index, "got": prompt_index},
        )
    return None


def out_of_range_detail(job: JobSpec, prompt_index: int) -> dict[str, int]:
    """What a ``prompt_mismatch`` for an index outside the job names. The start
    appears only when it is set, so a job at 0 answers exactly as before."""
    detail = {"prompt_count": job.prompt_count, "got": prompt_index}
    if job.prompt_start:
        detail = {"prompt_start": job.prompt_start, **detail}
    return detail


def _advance(job: JobSpec, cursors: CursorLedger, hotkey: str) -> None:
    if job.prompt_order == PROMPT_ORDER_MINER_WALK:
        cursors.advance(hotkey)
