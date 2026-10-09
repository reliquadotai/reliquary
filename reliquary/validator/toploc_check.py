"""TOPLOC beside GRAIL in the RL proof: one verdict from the hidden states the
GRAIL check already computed.

The thresholds come from the ``toploc_spec`` the validator put in its own copy
of the commit, from its task contract; a miner-sent spec never reaches here,
because CommitModel refuses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from reliquary.protocol.profiles import ProofProfile
from reliquary.validator.corpus_audit import (
    audit_completion, check_spans, trajectory_chunk_scores, trajectory_outcome,
)


@dataclass(frozen=True, slots=True)
class ToplocVerdict:
    passed: bool
    reason: str | None
    worst_exp: int
    worst_mant_mean: float
    worst_mant_median: float


def toploc_verdict(
    hidden: torch.Tensor, commit: Mapping, prompt_length: int
) -> ToplocVerdict | None:
    spec = commit.get("toploc_spec")
    if spec is None:
        return None
    try:
        proof = ProofProfile(**spec)
        proofs = commit.get("toploc_proofs")
        if proofs is None:
            return ToplocVerdict(False, "missing", 0, 0.0, 0.0)
        rows = hidden[prompt_length - 1 : hidden.shape[0] - 1] if prompt_length > 0 else hidden[:0]
        outcome = audit_completion(rows, proofs, proof)
    except Exception as exc:
        # The validator's own error. Enforced, it must stay loud rather than
        # fail an honest miner; in shadow it must not cost the GRAIL proof.
        if spec.get("mode") != "shadow":
            raise
        return ToplocVerdict(False, f"error:{type(exc).__name__}", 0, 0.0, 0.0)
    results = outcome.results
    return ToplocVerdict(
        outcome.passed,
        outcome.reason,
        max((r.exp_mismatches for r in results), default=0),
        max((r.mant_err_mean for r in results), default=0.0),
        max((r.mant_err_median for r in results), default=0.0),
    )


def toploc_span_verdict(
    hidden: torch.Tensor, commit: Mapping, spans
) -> ToplocVerdict | None:
    """Plan 2C: TOPLOC over every model span of a signed episode, from the one full-sequence prefill
    the GRAIL check already ran. Span ``[start, end)``'s rows are ``hidden[start - 1 : end - 1]`` (the
    corpus audit's ``span_hidden_states`` rule); its proofs are the commit's list, span after span. A
    span too short to judge alone is skipped (``trajectory_outcome``); a trajectory with no judged span
    fails, and so does an empty span list."""
    spec = commit.get("toploc_spec")
    if spec is None:
        return None
    try:
        proof = ProofProfile(**spec)
        proofs = commit.get("toploc_proofs")
        if proofs is None:
            return ToplocVerdict(False, "missing", 0, 0.0, 0.0)
        if not spans:
            return ToplocVerdict(False, "no_spans", 0, 0.0, 0.0)
        check_spans(spans, hidden.shape[0])
        rows = [hidden[start - 1 : end - 1] for start, end in spans]
        status, results = trajectory_chunk_scores(
            rows, proofs, chunk_tokens=proof.chunk_tokens, topk=proof.topk)
        outcome = trajectory_outcome(status, results, [end - start for start, end in spans], proof)
    except Exception as exc:
        # The validator's own error (as in ``toploc_verdict``): loud when enforced, free in shadow.
        if spec.get("mode") != "shadow":
            raise
        return ToplocVerdict(False, f"error:{type(exc).__name__}", 0, 0.0, 0.0)
    results = outcome.results
    return ToplocVerdict(
        outcome.passed,
        outcome.reason,
        max((r.exp_mismatches for r in results), default=0),
        max((r.mant_err_mean for r in results), default=0.0),
        max((r.mant_err_median for r in results), default=0.0),
    )
