"""The checks that cost no GPU, so they run on every submission.

Only provenance needs a forward pass and therefore a lottery; everything here
is cheap enough to apply to the whole flow, and it removes whole classes of
rubbish before the audit ever draws.
"""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass, field
import hashlib
from typing import Any

from reliquary.corpus.job import Sampling
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, expected_chunks, span_chunk_count


@dataclass(frozen=True, slots=True)
class CheckResult:
    ok: bool
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _ok() -> CheckResult:
    """A fresh result every time: ``CheckResult`` is frozen but its default
    detail dict is not, so one shared OK would carry a caller's edit forward."""
    return CheckResult(ok=True)


# No check function produces this: the ban lookup needs the validator's own
# miner state and is applied by the route itself. The string lives here so it
# has one name, the way ``corpus_text.py`` names its own reasons.
REASON_MINER_BANNED = "miner_banned"
# Applied by the route from the subnet's registrations, like the ban.
REASON_HOTKEY_NOT_REGISTERED = "hotkey_not_registered"
# A skip of a prompt that still has a slot; produced by `admission.skip`.
REASON_PROMPT_NOT_FULL = "prompt_not_full"
# Agentic trajectories (Task 8 produces them).
REASON_BAD_TURNS = "bad_turns"
REASON_SHORT_TURNS = "short_turns"


def completion_digest(prompt_index: int, tokens: Sequence[int]) -> str:
    """Bind the tokens to the prompt they answer, so the same text under two
    prompts is two different completions."""
    digest = hashlib.sha256()
    digest.update(int(prompt_index).to_bytes(8, "big", signed=False))
    for token in tokens:
        digest.update(int(token).to_bytes(4, "big", signed=False))
    return digest.hexdigest()


def check_completion_count(count: int, sampling: Sampling) -> CheckResult:
    if count != sampling.n:
        return CheckResult(
            ok=False,
            reason="bad_completion_count",
            detail={"expected": sampling.n, "got": count},
        )
    return _ok()


def check_token_budget(token_counts: Sequence[int], sampling: Sampling) -> CheckResult:
    for position, tokens in enumerate(token_counts):
        if tokens > sampling.max_new_tokens:
            return CheckResult(
                ok=False,
                reason="token_budget_exceeded",
                detail={
                    "position": position,
                    "tokens": int(tokens),
                    "max_new_tokens": sampling.max_new_tokens,
                },
            )
        # The floor catches the completion that is only its terminator, because
        # `parse_job` refuses a floor below 2; calling a short completion a cap
        # breach would read as the opposite failure in reject-reason telemetry.
        if tokens < sampling.min_new_tokens:
            return CheckResult(
                ok=False,
                reason="token_budget_underrun",
                detail={
                    "position": position,
                    "tokens": int(tokens),
                    "min_new_tokens": sampling.min_new_tokens,
                },
            )
    return _ok()


def check_termination(
    token_counts: Sequence[int],
    last_token_ids: Sequence[int],
    *,
    sampling: Sampling,
    eos_token_id: int,
) -> CheckResult:
    """A completion ends on EOS or on the cap; anything else is a silent
    truncation we will not pay for.

    The label is DERIVED here, not taken. This check used to accept a
    ``terminations`` sequence and test each label against the structure it
    claimed, which made the miner's own word an input — and any caller
    deriving the label and the last token id from the same array turned the
    EOS case into a tautology no test could ever discriminate. What is left is
    the one rule that carries evidence: a completion that did not stop on the
    terminator must have run out of budget.
    """
    for position, last_token_id in enumerate(last_token_ids):
        if last_token_id == eos_token_id:
            continue
        if token_counts[position] != sampling.max_new_tokens:
            return CheckResult(
                ok=False,
                reason="bad_termination",
                detail={
                    "position": position,
                    "termination": "cap",
                    "tokens": int(token_counts[position]),
                    "max_new_tokens": sampling.max_new_tokens,
                    "last_token_id": int(last_token_id),
                },
            )
    return _ok()


def check_duplicates(digests: Sequence[str], seen: AbstractSet[str]) -> CheckResult:
    """Verbatim copying, whether of another miner's work or of one's own."""
    within = set()
    for position, digest in enumerate(digests):
        if digest in seen or digest in within:
            return CheckResult(
                ok=False,
                reason="hash_duplicate",
                detail={"position": position, "digest": digest},
            )
        within.add(digest)
    return _ok()


def check_proof_shape(
    token_counts: Sequence[int], proof_counts: Sequence[int], chunk_tokens: int
) -> CheckResult:
    """One proof per chunk of every completion, counted before any GPU work."""
    if len(token_counts) != len(proof_counts):
        return CheckResult(
            ok=False,
            reason="bad_proof_shape",
            detail={"completions": len(token_counts), "proof_lists": len(proof_counts)},
        )
    for position, (tokens, proofs) in enumerate(zip(token_counts, proof_counts)):
        expected = expected_chunks(tokens, chunk_tokens)
        if proofs != expected:
            return CheckResult(
                ok=False,
                reason="bad_proof_shape",
                detail={"completion": position, "expected": expected, "got": proofs},
            )
    return _ok()


# A turn shorter than MIN_CHUNK_TOKENS is proven by one chunk that is never
# judged alone (Task 3 / spec §7 M1); this many of them per trajectory at most,
# so unproven tokens stay a handful.
MAX_SHORT_TURNS = 2


def check_turn_spans(spans: Sequence[tuple[int, int]], length: int, max_turns: int) -> CheckResult:
    """Ordered assistant spans in `tokens`, a segment between each two, the
    first right after the prompt, the last ending the tokens."""
    def refuse(why: str) -> CheckResult:
        return CheckResult(ok=False, reason=REASON_BAD_TURNS, detail={"why": why})

    if not spans:
        return refuse("no turns")
    if len(spans) > max_turns:
        return refuse(f"{len(spans)} turns, at most {max_turns}")
    if spans[0][0] != 0:
        return refuse("the first turn does not follow the prompt")
    previous_end = None
    for start, end in spans:
        if not start < end:
            return refuse(f"empty turn [{start}, {end})")
        if previous_end is not None and start <= previous_end:
            return refuse(f"turn [{start}, {end}) is not after the previous one's segment")
        previous_end = end
    if previous_end != length:
        return refuse(f"the last turn ends at {previous_end}, the tokens at {length}")
    return _ok()


def check_short_turns(spans: Sequence[tuple[int, int]], min_chunk_tokens: int = MIN_CHUNK_TOKENS,
                      max_short: int = MAX_SHORT_TURNS) -> CheckResult:
    short = sum(1 for start, end in spans if end - start < min_chunk_tokens)
    if short > max_short:
        return CheckResult(ok=False, reason=REASON_SHORT_TURNS,
                           detail={"short_turns": short, "max": max_short})
    return _ok()


def check_turn_budget(spans: Sequence[tuple[int, int]], *, prompt_len: int, length: int,
                      max_tokens_per_turn: int, max_total_tokens: int) -> CheckResult:
    if prompt_len + length > max_total_tokens:
        return CheckResult(ok=False, reason="token_budget_exceeded",
                           detail={"tokens": prompt_len + length, "max_total_tokens": max_total_tokens})
    for position, (start, end) in enumerate(spans):
        if end - start > max_tokens_per_turn:
            return CheckResult(ok=False, reason="token_budget_exceeded",
                               detail={"turn": position, "tokens": end - start,
                                       "max_tokens_per_turn": max_tokens_per_turn})
    return _ok()


def check_turn_termination(tokens: Sequence[int], spans: Sequence[tuple[int, int]], *,
                           prompt_len: int, terminator_id: int, stop_ids: AbstractSet[int],
                           max_tokens_per_turn: int, max_total_tokens: int) -> CheckResult:
    """Each turn ends where generation stops: on the turn terminator (any
    stop id for the final turn) or exactly at the cap the endpoint applied,
    ``min(max_tokens_per_turn, max_total_tokens - prompt so far)``."""
    last = len(spans) - 1
    for position, (start, end) in enumerate(spans):
        cap = min(max_tokens_per_turn, max_total_tokens - (prompt_len + start))
        token = tokens[end - 1]
        ended = token in stop_ids if position == last else token == terminator_id
        if not ended and end - start != cap:
            return CheckResult(ok=False, reason="bad_termination",
                               detail={"turn": position, "tokens": end - start, "cap": cap,
                                       "last_token_id": int(token)})
    return _ok()


def check_turn_proof_shape(spans: Sequence[tuple[int, int]], proof_counts: Sequence[int],
                           chunk_tokens: int, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> CheckResult:
    if len(spans) != len(proof_counts):
        return CheckResult(ok=False, reason="bad_proof_shape",
                           detail={"turns": len(spans), "proof_lists": len(proof_counts)})
    for position, ((start, end), got) in enumerate(zip(spans, proof_counts)):
        expected = span_chunk_count(end - start, chunk_tokens, min_chunk_tokens)
        if got != expected:
            return CheckResult(ok=False, reason="bad_proof_shape",
                               detail={"turn": position, "expected": expected, "got": got})
    return _ok()
