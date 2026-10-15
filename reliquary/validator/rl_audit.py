"""The out-of-zone audit's proof forward (rl-async-design.md §4.1, §4.4, §4.6; implementation map §1.6, §9).

``audit_forward(model)`` is the ``ProofForward`` that ``RlSubmissions.run_audits`` drains: for each of a group's
rollouts it runs the proof path's forward (``verifier.proof_forward_rows``: batch 1, the model handed in, which is
the latest checkpoint, fp32 head when the profile says so) and checks every completion position against the
forced draw with the verifier's own diagnostics (``verifier._gpu_seed_consistency_diagnostics``: the profile's
draw, Gumbel-max under v10), the uniforms being the lease's (``SubmittedGroup.draw_uniforms``). A BFT force span is
excluded, as in the window path.

Verdict:

* rollout match = exact picks / stochastic positions (max p < ``FORCED_SEED_STOCHASTIC_MAXPROB``), 1.0 without
  stochastic positions;
* ``deterministic`` failure: a rollout that follows the draw (enough stochastic positions, match >= the floor) yet
  carries more than ``RL_AUDIT_FORGERY_MISMATCH_TOLERANCE`` hard mismatches at near-deterministic positions. Its
  miner held the model and the noise; tokens no draw produces there were edited afterwards;
* ``statistical`` failure: the pooled match over the group's stochastic positions is under the floor
  (``RL_ROLLOUT_MATCH_FLOOR``, the trainer's and the ledger's). A rollout under the floor is excluded from pay
  (§4.6), it does not fail the group by itself. Below ``FORCED_SEED_MIN_STOCH_POSITIONS`` pooled stochastic
  positions the check abstains (passes), as the window path does;
* no model: ``passed=None`` (inconclusive: unpaid, unsanctioned).
"""
from __future__ import annotations

from typing import Any

from reliquary.constants import (
    FORCED_SEED_MIN_STOCH_POSITIONS,
    FORCED_SEED_ROLLOUT_MIN_STOCH,
    RL_AUDIT_FORGERY_MISMATCH_TOLERANCE,
    RL_ROLLOUT_MATCH_FLOOR,
)
from reliquary.services.exploration import AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL
from reliquary.validator import verifier
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_submit import AuditResult, ProofForward

DEFAULT_AUDIT_LIMIT = 8


def rollout_draw_diagnostics(model: Any, group: SubmittedGroup, index: int):
    """The verifier's seed-consistency diagnostics of rollout ``index`` against ``model``."""
    prompt, completion = list(group.prompt_tokens), list(group.completions[index])
    span = group.force_spans[index]
    start, end = (int(span[0]), int(span[1])) if span is not None else (0, 0)
    uniforms = group.draw_uniforms(index)
    offsets = [j for j in range(len(completion)) if not start <= j < end]
    _hidden, logits = verifier.proof_forward_rows(model, prompt + completion)
    return verifier._gpu_seed_consistency_diagnostics(
        logits, [completion[j] for j in offsets], [uniforms[j] for j in offsets],
        position_offsets=offsets, logit_positions=[len(prompt) + j - 1 for j in offsets])


def judge(diagnostics, *, floor: float = RL_ROLLOUT_MATCH_FLOOR,
          tolerance: int = RL_AUDIT_FORGERY_MISMATCH_TOLERANCE) -> AuditResult:
    """The group verdict from its rollouts' diagnostics (see the module doc)."""
    match = tuple(d.n_exact_match / d.n_stochastic if d.n_stochastic else 1.0 for d in diagnostics)
    if any(d.n_stochastic >= FORCED_SEED_ROLLOUT_MIN_STOCH and m >= floor
           and d.n_deterministic_hard_mismatch > tolerance for d, m in zip(diagnostics, match)):
        return AuditResult(False, AUDIT_FAILURE_DETERMINISTIC, match)
    stochastic = sum(d.n_stochastic for d in diagnostics)
    if stochastic >= FORCED_SEED_MIN_STOCH_POSITIONS and sum(d.n_exact_match for d in diagnostics) < floor * stochastic:
        return AuditResult(False, AUDIT_FAILURE_STATISTICAL, match)
    return AuditResult(True, None, match)


def audit_forward(model: Any) -> ProofForward:
    """The ``ProofForward`` over ``model`` (the latest checkpoint; None = unavailable, every audit inconclusive)."""
    def forward(group: SubmittedGroup) -> AuditResult:
        if model is None:
            return AuditResult(None)
        return judge([rollout_draw_diagnostics(model, group, i) for i in range(len(group.completions))])
    return forward


def run_rl_audits(model: Any, submissions, *, limit: int = DEFAULT_AUDIT_LIMIT) -> int:
    """The validator service's hook: drain up to ``limit`` queued audits against ``model`` (its verify model, the
    latest published checkpoint). Returns how many were concluded. Audit promptly: a later checkpoint costs match."""
    return submissions.run_audits(audit_forward(model), limit)


__all__ = ["DEFAULT_AUDIT_LIMIT", "audit_forward", "judge", "rollout_draw_diagnostics", "run_rl_audits"]
