"""The out-of-zone audit's proof forward (rl-async-design.md §4.1, §4.4, §4.6; implementation map §1.6, §9).

``audit_forward(model)`` is the ``ProofForward`` that ``RlSubmissions.run_audits`` drains: for each of a group's
rollouts it runs the proof path's forward (``verifier.proof_forward_rows``: batch 1, the model handed in, which is
the latest checkpoint, fp32 head when the profile says so) and checks every completion position against the
forced draw with the verifier's own diagnostics (``verifier._gpu_seed_consistency_diagnostics``: the profile's
draw, Gumbel-max under v10), the uniforms being the lease's (``SubmittedGroup.draw_uniforms``). A BFT force span is
excluded, as in the window path. An episode's tool observations are in the forward but never judged: only its
model positions are, each against the draw of its model-token index (§4.10).

Verdict (each rollout's tallies read as the trainer reads them, ``rl_draw.rollout_rate`` / ``pooled_rate``):

* rollout match = exact picks / stochastic positions (max p < ``FORCED_SEED_STOCHASTIC_MAXPROB``) when it has
  ``FORCED_SEED_ROLLOUT_MIN_STOCH`` of them, else exact picks / every drawn position;
* ``deterministic`` failure: a rollout that follows the draw (enough stochastic positions, match >= the floor) yet
  carries more than ``RL_AUDIT_FORGERY_MISMATCH_TOLERANCE`` hard mismatches at near-deterministic positions. Its
  miner held the model and the noise; tokens no draw produces there were edited afterwards;
* fewer than ``FORCED_SEED_MIN_STOCH_POSITIONS`` drawn positions in the whole group: the audit abstains,
  ``passed=None`` with ``suspect`` (nothing verified: unpaid, no probation pass, the hotkey stays in probation);
* ``statistical`` failure: the pooled match is under the floor (``RL_GROUP_MATCH_FLOOR``, the trainer's), or a
  rollout is under ``RL_FORGERY_MATCH_FLOOR`` (no honest noise). Unlike the trainer, the audit rates short rollouts
  too (``min_drawn=0``): it has no Gumbel gap or final-EOS test to judge them by. A rollout between the two floors is excluded from
  pay (§4.6), it does not fail the group by itself;
* no model: ``passed=None`` (inconclusive: unpaid, unsanctioned).

The forward runs against the latest checkpoint; a failure is unpaid, never sanctioned (``RlSubmissions``).
"""
from __future__ import annotations

from typing import Any

from reliquary.constants import (
    FORCED_SEED_MIN_STOCH_POSITIONS,
    FORCED_SEED_ROLLOUT_MIN_STOCH,
    RL_AUDIT_FORGERY_MISMATCH_TOLERANCE,
    RL_FORGERY_MATCH_FLOOR,
    RL_GROUP_MATCH_FLOOR,
    RL_ROLLOUT_MATCH_FLOOR,
)
from reliquary.protocol.rl_draw import pooled_rate, rollout_rate
from reliquary.services.exploration import AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL
from reliquary.validator import verifier
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_submit import AuditResult, ProofForward

DEFAULT_AUDIT_LIMIT = 8


def rollout_draw_diagnostics(model: Any, group: SubmittedGroup, index: int):
    """The verifier's seed-consistency diagnostics of rollout ``index`` against ``model``. An episode is one
    forward over its whole sequence (prompt, model turns and tool observations); only its model positions are
    judged, draw ``t`` being the model-token index across turns (``SubmittedGroup.model_positions``)."""
    prompt, completion = list(group.prompt_tokens), list(group.completions[index])
    span = group.force_spans[index]
    start, end = (int(span[0]), int(span[1])) if span is not None else (0, 0)
    uniforms = group.draw_uniforms(index)
    drawn = [(t, p) for t, p in enumerate(group.model_positions(index)) if not start <= p < end]
    _hidden, logits = verifier.proof_forward_rows(model, prompt + completion)
    return verifier._gpu_seed_consistency_diagnostics(
        logits, [completion[p] for _, p in drawn], [uniforms[t] for t, _ in drawn],
        position_offsets=[t for t, _ in drawn], logit_positions=[len(prompt) + p - 1 for _, p in drawn])


def judge(diagnostics, *, floor: float = RL_ROLLOUT_MATCH_FLOOR, group_floor: float = RL_GROUP_MATCH_FLOOR,
          tolerance: int = RL_AUDIT_FORGERY_MISMATCH_TOLERANCE,
          forgery_floor: float = RL_FORGERY_MATCH_FLOOR) -> AuditResult:
    """The group verdict from its rollouts' diagnostics (see the module doc)."""
    tallies = [(d.n_stochastic, d.n_exact_match, d.n_positions, d.n_drawn_exact_match) for d in diagnostics]
    match = tuple(rollout_rate(t, min_drawn=0) for t in tallies)      # no gap test here: short rollouts get a rate
    if any(d.n_stochastic >= FORCED_SEED_ROLLOUT_MIN_STOCH and m >= floor
           and d.n_deterministic_hard_mismatch > tolerance for d, m in zip(diagnostics, match)):
        return AuditResult(False, AUDIT_FAILURE_DETERMINISTIC, match)
    if sum(t[2] for t in tallies) < FORCED_SEED_MIN_STOCH_POSITIONS:
        return AuditResult(None, None, match, suspect=True)
    if min(match) < forgery_floor or pooled_rate(tallies) < group_floor:
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
