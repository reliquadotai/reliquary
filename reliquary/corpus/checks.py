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
from reliquary.protocol.toploc import expected_chunks


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
