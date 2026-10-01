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
from reliquary.protocol.toploc import ChunkResult, sequence_verdict
from reliquary.protocol.toploc_proof import verify_chunk_proofs


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
    model, sequences: Sequence[tuple[Sequence[int], int, Sequence[str]]], *,
    chunk_tokens: int, topk: int, batch_tokens: int,
) -> tuple[list[tuple[str, tuple[ChunkResult, ...]]], float, float]:
    """``completion_chunk_scores`` for many ``(tokens, prompt_len, proofs)``,
    packed shortest first into forward passes under ``batch_tokens`` padded
    tokens. Returns the scores in input order and the forward and verify seconds.

    One function for the control and the executor, so both compute alike.
    """
    import time

    order = sorted(range(len(sequences)), key=lambda k: (len(sequences[k][0]), k))
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
