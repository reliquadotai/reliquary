"""The corpus audit's identity check: re-run the pinned model over a sampled
submission and verify the miner's TOPLOC proofs against its own activations.

A drawn submission that fails is paid nothing and voids the miner's epoch
credit (spec section 9); this module only returns the verdict.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC, ProofProfile
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, ChunkResult, sequence_verdict, span_chunk_count
from reliquary.protocol.toploc_proof import verify_chunk_proofs, verify_span_proofs


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    passed: bool
    reason: str | None
    results: tuple[ChunkResult, ...] = ()


def _decoder(model):
    # The base model returns only the last (normed) hidden state; asking the LM
    # head model for all of them costs ~20 GB at 32k tokens on a 27B model.
    return model.model if hasattr(model, "model") else model.get_decoder()


@torch.no_grad()
def completion_hidden_states(model, tokens: Sequence[int], prompt_len: int) -> torch.Tensor:
    """Final hidden state at every position that produced a completion token."""
    return batch_completion_hidden_states(model, [(list(tokens), prompt_len)])[0]


@torch.no_grad()
def batch_completion_hidden_states(
    model, sequences: Sequence[tuple[Sequence[int], int]]
) -> list[torch.Tensor]:
    """The same rows for several sequences, right-padded into one forward pass."""
    vocabulary = model.get_input_embeddings().num_embeddings
    for tokens, prompt_len in sequences:
        if not 0 < prompt_len < len(tokens):
            raise ValueError(f"prompt_len {prompt_len} leaves no completion in {len(tokens)} tokens")
        if min(tokens) < 0 or max(tokens) >= vocabulary:
            # On CUDA an out-of-range embedding index kills the device context.
            raise ValueError(f"a token id is outside the vocabulary of {vocabulary}")
    device = next(model.parameters()).device
    width = max(len(tokens) for tokens, _ in sequences)
    ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, (tokens, _) in enumerate(sequences):
        ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
        mask[row, : len(tokens)] = 1
    hidden = _decoder(model)(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return [hidden[row, n - 1 : len(tokens) - 1] for row, (tokens, n) in enumerate(sequences)]


def check_spans(spans: Sequence[tuple[int, int]], length: int) -> None:
    """Spans are ordered, non-empty, non-overlapping and inside the sequence,
    and none starts at 0 (a turn always follows at least one prompt token)."""
    previous_end = 1
    for start, end in spans:
        if not (previous_end <= start < end <= length):
            raise ValueError(f"span [{start}, {end}) is malformed for a {length}-token sequence")
        previous_end = end


@torch.no_grad()
def span_hidden_states(
    model, tokens: Sequence[int], spans: Sequence[tuple[int, int]]
) -> list[torch.Tensor]:
    """One prefill over the whole trajectory, then for each assistant span the
    rows that predicted its tokens: positions ``start - 1 .. end - 2``."""
    check_spans(spans, len(tokens))
    (rows,) = batch_completion_hidden_states(model, [(list(tokens), 1)])
    # ``rows[i]`` is the state at position ``i`` (it predicted token ``i + 1``).
    return [rows[start - 1 : end - 1] for start, end in spans]


SCORE_OK = "ok"
SCORE_PROOF_UNDECODABLE = "proof_undecodable"
SCORE_BAD_PROOF_SHAPE = "bad_proof_shape"


def completion_chunk_scores(
    hidden: torch.Tensor, proofs_b64: Sequence[str], *, chunk_tokens: int, topk: int
) -> tuple[str, tuple[ChunkResult, ...]]:
    """The per-chunk comparison of a completion's proofs against ``hidden``,
    before any verdict: what a remote executor returns and the control judges.

    A proof the miner sent malformed is a status, not an error; a configuration
    the validator got wrong raises.
    """
    if hidden.dim() != 2:
        raise ValueError(f"configuration: expected [rows, width] activations, got {tuple(hidden.shape)}")
    if topk > hidden.shape[1]:
        raise ValueError(
            f"configuration: topk {topk} exceeds the model width {hidden.shape[1]}"
        )
    try:
        raw = [base64.b64decode(p, validate=True) for p in proofs_b64]
    except (binascii.Error, ValueError):
        return SCORE_PROOF_UNDECODABLE, ()
    try:
        results = verify_chunk_proofs(hidden, raw, chunk_tokens=chunk_tokens, topk=topk)
    except ValueError:
        return SCORE_BAD_PROOF_SHAPE, ()
    return SCORE_OK, tuple(results)


def rows_of_items(items: Sequence[dict]) -> list[tuple]:
    """``score_sequences`` rows from ``{tokens, prompt_len, proofs[, spans]}``
    dicts: a trajectory's spans must reach the scorer or it is read as a
    single-turn row."""
    return [(i["tokens"], i["prompt_len"], i["proofs"])
            if i.get("spans") is None
            else (i["tokens"], i["prompt_len"], i["proofs"], i["spans"]) for i in items]


def trajectory_chunk_scores(
    span_rows: Sequence[torch.Tensor], proofs_b64: Sequence[str], *, chunk_tokens: int,
    topk: int, min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> tuple[str, tuple[ChunkResult, ...]]:
    """``completion_chunk_scores`` for every assistant span of a trajectory:
    the proofs are the spans' lists concatenated in span order, and the
    results come back the same way."""
    for rows in span_rows:
        if rows.dim() != 2:
            raise ValueError(f"configuration: expected [rows, width] activations, got {tuple(rows.shape)}")
        if topk > rows.shape[1]:
            raise ValueError(f"configuration: topk {topk} exceeds the model width {rows.shape[1]}")
    counts = [span_chunk_count(rows.shape[0], chunk_tokens, min_chunk_tokens) for rows in span_rows]
    if sum(counts) != len(proofs_b64):
        return SCORE_BAD_PROOF_SHAPE, ()
    try:
        raw = [base64.b64decode(p, validate=True) for p in proofs_b64]
    except (binascii.Error, ValueError):
        return SCORE_PROOF_UNDECODABLE, ()
    results: list[ChunkResult] = []
    at = 0
    for rows, count in zip(span_rows, counts):
        try:
            results += verify_span_proofs(rows, raw[at:at + count], chunk_tokens=chunk_tokens,
                                          topk=topk, min_chunk_tokens=min_chunk_tokens)
        except ValueError:
            return SCORE_BAD_PROOF_SHAPE, ()
        at += count
    return SCORE_OK, tuple(results)


def trajectory_outcome(
    status: str, results: Sequence[ChunkResult], span_lengths: Sequence[int], proof: ProofProfile,
    *, min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> AuditOutcome:
    """The decision for a trajectory: each judged span through the proof's
    thresholds, the first failing one fails it. A span shorter than
    ``min_chunk_tokens`` is one chunk too short to judge alone (spec 7 M1,
    plan Task 3); a trajectory with no judged span fails closed."""
    if proof.scheme != PROOF_SCHEME_TOPLOC:
        raise ValueError(f"the corpus audit verifies toploc, not {proof.scheme!r}")
    if status != SCORE_OK:
        return AuditOutcome(False, status)
    thresholds = proof.thresholds()
    at, judged = 0, 0
    for length in span_lengths:
        count = span_chunk_count(length, proof.chunk_tokens, min_chunk_tokens)
        chunks = results[at:at + count]
        at += count
        if length < min_chunk_tokens:
            continue
        judged += 1
        passed, reason = sequence_verdict(chunks, thresholds)
        if not passed:
            return AuditOutcome(False, reason, tuple(results))
    if not judged:
        return AuditOutcome(False, "no_judged_spans", tuple(results))
    return AuditOutcome(True, None, tuple(results))


def outcome_from_scores(
    status: str, results: Sequence[ChunkResult], proof: ProofProfile
) -> AuditOutcome:
    """The decision: the proof's thresholds over the chunk comparisons."""
    if proof.scheme != PROOF_SCHEME_TOPLOC:
        raise ValueError(f"the corpus audit verifies toploc, not {proof.scheme!r}")
    if status != SCORE_OK:
        return AuditOutcome(False, status)
    passed, reason = sequence_verdict(results, proof.thresholds())
    return AuditOutcome(passed, reason, tuple(results))


def score_sequences(
    model, sequences: Sequence[tuple], *,
    chunk_tokens: int, topk: int, batch_tokens: int,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> tuple[list[tuple[str, tuple[ChunkResult, ...]]], float, float]:
    """``completion_chunk_scores`` for many ``(tokens, prompt_len, proofs)``,
    packed shortest first into forward passes under ``batch_tokens`` padded
    tokens. Returns the scores in input order and the forward and verify seconds.

    A row may carry a fourth field, its assistant spans in ``tokens``
    coordinates; such a row's ``proofs`` are its spans' lists concatenated, and
    it is prefilled alone, never packed: one 60k-token trajectory already fills
    an 80 GB card (spec 7 M3).

    One function for the control and the executor, so both compute alike.
    """
    import time

    def spans_of(k):
        return sequences[k][3] if len(sequences[k]) > 3 else None

    order = sorted((k for k in range(len(sequences)) if spans_of(k) is None),
                   key=lambda k: (len(sequences[k][0]), k))
    sub_batches: list[list[int]] = []
    current: list[int] = []
    current_width = 0
    for k in order:
        length = len(sequences[k][0])
        width = max(current_width, length)
        if current and (len(current) + 1) * width > batch_tokens:
            sub_batches.append(current)
            current, width = [], length
        current.append(k)
        current_width = width
    if current:
        sub_batches.append(current)

    scores: list = [None] * len(sequences)
    forward_seconds = verify_seconds = 0.0
    for sub_batch in sub_batches:
        rows = [(list(sequences[k][0]), sequences[k][1]) for k in sub_batch]
        mark = time.perf_counter()
        hidden_states = batch_completion_hidden_states(model, rows)
        if hidden_states and hidden_states[0].is_cuda:
            # Kernels are queued asynchronously: without this the forward's
            # time would be billed to the first verification that reads it.
            torch.cuda.synchronize(hidden_states[0].device)
        forward_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        for k, hidden in zip(sub_batch, hidden_states):
            scores[k] = completion_chunk_scores(
                hidden, sequences[k][2], chunk_tokens=chunk_tokens, topk=topk)
        verify_seconds += time.perf_counter() - mark
        # Drop this sub-batch's padded activations before the next one is
        # computed: two final-hidden-state tensors must never be live at once.
        del hidden_states, rows, hidden
    for k in range(len(sequences)):
        spans = spans_of(k)
        if spans is None:
            continue
        tokens, _prompt_len, proofs = sequences[k][:3]
        mark = time.perf_counter()
        rows = span_hidden_states(model, list(tokens), [tuple(s) for s in spans])
        if rows and rows[0].is_cuda:
            torch.cuda.synchronize(rows[0].device)
        forward_seconds += time.perf_counter() - mark
        mark = time.perf_counter()
        scores[k] = trajectory_chunk_scores(rows, proofs, chunk_tokens=chunk_tokens, topk=topk,
                                            min_chunk_tokens=min_chunk_tokens)
        verify_seconds += time.perf_counter() - mark
        del rows
    return scores, forward_seconds, verify_seconds


def audit_completion(
    hidden: torch.Tensor, proofs_b64: Sequence[str], proof: ProofProfile
) -> AuditOutcome:
    if proof.scheme != PROOF_SCHEME_TOPLOC:
        raise ValueError(f"the corpus audit verifies toploc, not {proof.scheme!r}")
    # Raised, not returned as a verdict: these are the validator's own errors,
    # and a failed audit would void an honest miner's epoch credit.
    status, results = completion_chunk_scores(
        hidden, proofs_b64, chunk_tokens=proof.chunk_tokens, topk=proof.topk)
    return outcome_from_scores(status, results, proof)
