"""Out-of-zone audits run by the async trainer (rl-async-design.md §4.4).

The validator keeps the audit queue and the verdicts (``RlSubmissions.take_audits`` / ``report_audit``); the trainer,
which already holds the latest weights, runs the forward. ``audit_group(model, group)`` is ``rl_audit``'s judgement
(``rl_audit.judge``) over diagnostics computed through the trainer's own model: the base model's last hidden state,
then the profile's head (fp32 when ``sampling.lm_head_fp32``) on the rows the draw reads, and the verifier's
``seed_consistency_diagnostics`` on them. Under FSDP2 the head is a sharded ``FusedLogprobHead``: the rows are
projected inside one module call (``audit=``), where its weight is gathered, and every rank runs the same forward
(``broadcasting_audit``: rank 0 sends ``AUDIT`` with the group, the followers run it and drop the result).

``group_diagnostics`` computes a whole group as the training forward does: the rollouts packed into right-padded
micro-batches under the trainer's token budget (``_pack_by_token_budget``), one base forward per micro-batch and one
head call per micro-batch (one weight gather under FSDP2) projecting only the rows the draw reads; each rollout's
diagnostics come from its own rows. ``rollout_diagnostics`` (one rollout, its own forward) is the reference it matches.

``TrainerAudits`` is the trainer's side of the loop: take up to ``limit`` jobs, judge them within ``budget_s``,
report each verdict; counters feed the step's metrics.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


class _ProjectedRows:
    """Read-only rows of ``project(rows)``: the duck-typed logits surface ``seed_consistency_diagnostics`` reads
    (slices, ``index_select``, ``shape``, ``device``), projecting only the rows asked for."""

    def __init__(self, rows: torch.Tensor, project: Callable[[torch.Tensor], torch.Tensor], vocab_size: int) -> None:
        self._rows, self._project = rows, project
        self.shape = torch.Size((int(rows.shape[0]), int(vocab_size)))

    @property
    def device(self) -> Any:
        return self._rows.device

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError("rows only")
        with torch.no_grad():
            return self._project(self._rows.index_select(0, index))

    def __getitem__(self, key: Any) -> torch.Tensor:
        with torch.no_grad():
            return self._project(self._rows[key])


def head_projection(weight: torch.Tensor, bias: torch.Tensor | None) -> Callable[[torch.Tensor], torch.Tensor]:
    """The draw's logits from hidden rows: the verifier's fp32 head when the profile says so, else the plain head."""
    from reliquary.validator.verifier import fp32_head_matmul, lm_head_fp32

    if lm_head_fp32():
        if bias is not None:
            raise RuntimeError("fp32 LM head: a biased head is not supported")
        return lambda rows: fp32_head_matmul(rows.reshape(-1, rows.shape[-1]), weight)
    return lambda rows: torch.nn.functional.linear(rows, weight, bias)


def rollout_diagnostics(model: Any, group, index: int):
    """``rl_audit.rollout_draw_diagnostics`` through the trainer's model (single GPU or FSDP2)."""
    from reliquary.validator import verifier
    from reliquary.validator.rl_audit import rollout_draw_diagnostics
    from reliquary.validator.training import _base_model_and_lm_head, _last_hidden_state

    base, lm_head = _base_model_and_lm_head(model)
    if base is None:
        return rollout_draw_diagnostics(model, group, index)          # no separable head: the proof path's forward
    prompt, completion = list(group.prompt_tokens), list(group.completions[index])
    uniforms = group.draw_uniforms(index)
    drawn = _drawn(group, index)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt + completion], device=device)
    with torch.no_grad():
        hidden = _last_hidden_state(base(input_ids, use_cache=False))[0]
        rows = hidden.index_select(0, torch.tensor([len(prompt) + p - 1 for _, p in drawn], device=device,
                                                   dtype=torch.long))
    vocab = int(getattr(lm_head, "out_features", 0) or lm_head.weight.shape[0])

    def diagnose(project):
        return verifier._gpu_seed_consistency_diagnostics(
            _ProjectedRows(rows, project, vocab), [completion[p] for _, p in drawn], [uniforms[t] for t, _ in drawn],
            position_offsets=[t for t, _ in drawn])

    if getattr(lm_head, "fused_selected_logprobs", False):
        with torch.no_grad():
            return lm_head(rows, audit=diagnose)                        # FSDP2: the weight gathered for this call
    return diagnose(head_projection(lm_head.weight, getattr(lm_head, "bias", None)))


def _drawn(group, index: int) -> list[tuple[int, int]]:
    """(draw offset, completion position) of every position the draw reads, the force span excluded."""
    span = group.force_spans[index]
    start, end = (int(span[0]), int(span[1])) if span is not None else (0, 0)
    return [(t, p) for t, p in enumerate(group.model_positions(index)) if not start <= p < end]


def group_diagnostics(model: Any, group, *, token_budget: int | None = None) -> list:
    """Every rollout's diagnostics of ``group`` through the trainer's model, packed like a training micro-batch:
    the same values as ``rollout_diagnostics`` one rollout at a time, in a few forwards and head calls."""
    from reliquary.validator import verifier
    from reliquary.validator.rl_audit import rollout_draw_diagnostics
    from reliquary.validator.training import (
        _base_model_and_lm_head, _last_hidden_state, _microbatch_budget, _pack_by_token_budget,
    )

    n = len(group.completions)
    base, lm_head = _base_model_and_lm_head(model)
    if base is None:
        return [rollout_draw_diagnostics(model, group, i) for i in range(n)]
    prompt = [int(t) for t in group.prompt_tokens]
    sequences = [prompt + [int(t) for t in group.completions[i]] for i in range(n)]
    drawn = [_drawn(group, i) for i in range(n)]
    device = next(model.parameters()).device
    vocab = int(getattr(lm_head, "out_features", 0) or lm_head.weight.shape[0])
    budget = int(token_budget) if token_budget is not None else _microbatch_budget()
    out: list = [None] * n
    for batch in _pack_by_token_budget([len(s) for s in sequences], budget):
        width = max(len(sequences[i]) for i in batch)
        input_ids = torch.zeros(len(batch), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch), width, dtype=torch.long, device=device)
        for row, i in enumerate(batch):
            input_ids[row, :len(sequences[i])] = torch.tensor(sequences[i], device=device)
            mask[row, :len(sequences[i])] = 1
        padded = any(len(sequences[i]) < width for i in batch)
        with torch.no_grad():
            outputs = (base(input_ids, attention_mask=mask, use_cache=False) if padded
                       else base(input_ids, use_cache=False))
            hidden = _last_hidden_state(outputs)
            pieces = [hidden[row].index_select(0, torch.tensor([len(prompt) + p - 1 for _, p in drawn[i]],
                                                                device=device, dtype=torch.long))
                      for row, i in enumerate(batch)]
        sizes = [len(drawn[i]) for i in batch]
        rows = torch.cat(pieces) if pieces else hidden.new_zeros((0, hidden.shape[-1]))

        def diagnose(project, batch=batch, rows=rows, sizes=sizes):
            found, start = [], 0
            for i, size in zip(batch, sizes):
                completion, uniforms = group.completions[i], group.draw_uniforms(i)
                found.append(verifier._gpu_seed_consistency_diagnostics(
                    _ProjectedRows(rows[start:start + size], project, vocab),
                    [int(completion[p]) for _, p in drawn[i]], [uniforms[t] for t, _ in drawn[i]],
                    position_offsets=[t for t, _ in drawn[i]]))
                start += size
            return found

        if getattr(lm_head, "fused_selected_logprobs", False):
            with torch.no_grad():
                found = lm_head(rows, audit=diagnose)                  # FSDP2: one gather per micro-batch
        else:
            found = diagnose(head_projection(lm_head.weight, getattr(lm_head, "bias", None)))
        for i, diagnostics in zip(batch, found):
            out[i] = diagnostics
    return out


def audit_group(model: Any, group):
    """The group's ``AuditResult`` (``rl_audit.judge``) against the trainer's current weights."""
    from reliquary.validator.rl_audit import judge

    model_was_training = getattr(model, "training", False)
    if model_was_training:
        model.eval()
    try:
        return judge(group_diagnostics(model, group))
    finally:
        if model_was_training:
            model.train()


def broadcasting_audit(channel, audit_fn: Callable = audit_group) -> Callable:
    """Rank 0's audit under FSDP2: the followers run the same forward on the same group."""

    def audit(model, group):
        channel.send("AUDIT", {"group": group})
        return audit_fn(model, group)

    return audit


class TrainerAudits:
    """Take audit jobs from ``source`` (``take_audits(limit)`` / ``report_audit(lease_id, result)``: the trainer
    link, or ``RlSubmissions`` in process) and judge them with ``audit_fn(model, group)``."""

    def __init__(self, source, *, audit_fn: Callable = audit_group, clock: Callable[[], float] = time.monotonic) -> None:
        self.source, self.audit_fn, self.clock = source, audit_fn, clock
        self.done = 0
        self.seconds = 0.0

    def run(self, model, *, limit: int, budget_s: float) -> int:
        """Judge up to ``limit`` queued audits within about ``budget_s`` seconds; returns how many were reported."""
        if limit <= 0:
            return 0
        began = self.clock()
        jobs = self.source.take_audits(limit)
        done = 0
        for group in jobs:
            if done and self.clock() - began > budget_s:
                break                           # the rest stay queued for the next round
            result = self.audit_fn(model, group)
            try:
                self.source.report_audit(group.lease_id, result)
            except Exception as exc:
                if not (isinstance(exc, LookupError) or getattr(exc, "status", None) == 404):
                    raise
                # concluded meanwhile (dropped as stale): nothing to report
            done += 1
        spent = self.clock() - began
        self.done += done
        self.seconds += spent
        if done:
            logger.info("trainer audits: %d judged in %.1fs", done, spent)
        return done

    def metrics(self) -> dict[str, float]:
        """Since the last call: audits judged and the seconds they took."""
        out = {"async/audits": float(self.done), "async/audit_s": self.seconds}
        self.done, self.seconds = 0, 0.0
        return out


__all__ = ["TrainerAudits", "audit_group", "broadcasting_audit", "group_diagnostics", "head_projection",
           "rollout_diagnostics"]
