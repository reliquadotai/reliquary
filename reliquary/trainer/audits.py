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


def diagnostics_from_rows(*, match, gap, stochastic, offsets, boundary_epsilon: float):
    """``SeedConsistencyDiagnostics`` (Gumbel draw) from a gateway's per drawn position rows: ``match`` (the pick is
    the miner's token), ``gap`` (Gumbel score gap, 0 when matched), ``stochastic`` (max p < the threshold),
    ``offsets`` (draw index). The miss is ``forced_sampling``'s ``1 - exp(-gap)``."""
    import math

    from reliquary.environment.forced_sampling import _FAR_MISS, SeedConsistencyDiagnostics

    miss = [0.0 if m else -math.expm1(-float(g)) for m, g in zip(match, gap, strict=True)]
    exact = [bool(m) for m in match]
    stoch = [bool(s) for s in stochastic]
    hard = [not (e or x <= boundary_epsilon) for e, x in zip(exact, miss)]
    first = [o for o, h in zip(offsets, hard) if h]
    if not miss:
        return SeedConsistencyDiagnostics()
    return SeedConsistencyDiagnostics(
        n_positions=len(miss), n_stochastic=sum(stoch), n_exact_match=sum(s and e for s, e in zip(stoch, exact)),
        n_boundary_match=len(miss) - sum(hard), n_hard_mismatch=sum(hard),
        n_deterministic_hard_mismatch=sum(h and not s for h, s in zip(hard, stoch)),
        n_miss_gt_0_01=sum(x > 0.01 for x in miss), n_miss_gt_0_05=sum(x > 0.05 for x in miss),
        n_miss_gt_0_10=sum(x > 0.10 for x in miss), max_cdf_miss=max(miss),
        first_hard_mismatch_offset=min(first) if first else None, n_drawn_exact_match=sum(exact),
        last_miss=miss[-1], n_far_miss=sum(x > _FAR_MISS for x in miss))


def gateway_audit(client, *, eos_ids) -> Callable:
    """``TrainerAudits``' ``audit_fn`` on a Miles gateway (``RELIQUARY_TRAINER_BACKEND=miles``): a ``forward`` with the
    ``reliquary_gate`` loss (a forward trains nothing and is never backed out), its per-position rows turned into
    the verifier's diagnostics, then ``rl_audit.judge`` per group.

    ``audit(model, group)`` judges one group; ``audit.many(model, groups)`` judges several in one ``forward`` per
    draw config (single-turn groups together, episodes together: they take no final-EOS test), both submitted
    before either is awaited. Each rollout's rows depend on its own datum only, so a group's verdict is the same
    alone or batched; ``TrainerAudits`` decides how many groups share a call (``max_rollouts_per_call``)."""
    from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX
    from reliquary.trainer.async_trainer import training_group
    from reliquary.trainer.miles_datums import _draw_positions, rollout_datum
    from reliquary.validator.rl_audit import judge
    from reliquary.validator.verifier import draw_settings

    settings = draw_settings()
    if settings.draw != DRAW_GUMBEL_MAX or settings.top_k not in (0, -1) or settings.top_p < 1.0:
        raise RuntimeError("the gateway audit implements the unwarped Gumbel draw only")
    config = {"eos_ids": ",".join(str(int(t)) for t in eos_ids)} if eos_ids else {}
    if float(settings.t) != 1.0:
        config["temperature"] = float(settings.t)

    def values(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    def diagnostics(rollout, output):
        p = int(rollout.commit["rollout"]["prompt_length"])
        drawn = _draw_positions(rollout)
        rows = [p + q - 1 for _, q in drawn]
        match, gap, stoch = values(output["match"]), values(output["gap"]), values(output["stochastic"])
        return diagnostics_from_rows(
            match=[match[j] for j in rows], gap=[gap[j] for j in rows], stochastic=[stoch[j] for j in rows],
            offsets=[t for t, _ in drawn], boundary_epsilon=settings.boundary_epsilon)

    def many(_model, groups):
        prepared = []
        for group in groups:
            uniforms = [group.draw_uniforms(i) for i in range(len(group.completions))]
            rollouts = training_group(group, slot0=0, index=0, uniforms=uniforms).rollouts
            prepared.append((rollouts, any("episode" in r.commit["rollout"] for r in rollouts)))
        pending = []                                # (group indices, future): one forward per draw config
        for episode in (False, True):
            members = [g for g, (_, e) in enumerate(prepared) if e == episode]
            if members:
                datums = [rollout_datum(r, 0.0, 0) for g in members for r in prepared[g][0]]
                pending.append((members, client.forward(
                    datums, "reliquary_gate", {**config, "terminal_check": 0} if episode else config)))
        found: list = [None] * len(groups)
        for members, future in pending:
            outputs = iter(future.result().loss_fn_outputs)
            for g in members:
                found[g] = judge([diagnostics(r, next(outputs)) for r in prepared[g][0]])
            if next(outputs, None) is not None:
                raise RuntimeError("the gateway returned more rows than datums")
        return found

    def audit(model, group):
        return many(model, [group])[0]

    audit.many = many
    return audit


def broadcasting_audit(channel, audit_fn: Callable = audit_group) -> Callable:
    """Rank 0's audit under FSDP2: the followers run the same forward on the same group."""

    def audit(model, group):
        channel.send("AUDIT", {"group": group})
        return audit_fn(model, group)

    return audit


AUDIT_CHUNK = 8   # audits taken per call
AUDIT_TAKE_MAX = 32                 # largest take of a batched round (group bodies are large)
AUDIT_ROLLOUTS_PER_CALL = 64        # rollouts per batched audit forward (``RELIQUARY_RL_TRAINER_AUDIT_ROLLOUTS_PER_CALL``)
GROUP_ROLLOUTS = 16                 # a group's usual size: sizes the take of a batched round


def pack_by_rollouts(groups: list, max_rollouts: int) -> list[list]:
    """Consecutive groups packed into calls of at most ``max_rollouts`` rollouts (a larger group goes alone)."""
    calls: list[list] = []
    size = 0
    for group in groups:
        n = len(group.completions)
        if calls and size + n <= max_rollouts:
            calls[-1].append(group)
            size += n
        else:
            calls.append([group])
            size = n
    return calls


class TrainerAudits:
    """Take audit jobs from ``source`` (``take_audits(limit)`` / ``report_audit(lease_id, result)``: the trainer
    link, or ``RlSubmissions`` in process) and judge them with ``audit_fn(model, group)``, or, when the audit
    function has ``many(model, groups)`` (the Miles gateway's), several groups per call: consecutive groups up to
    ``max_rollouts_per_call`` rollouts, the budget checked between calls."""

    def __init__(self, source, *, audit_fn: Callable = audit_group, clock: Callable[[], float] = time.monotonic,
                 max_rollouts_per_call: int = AUDIT_ROLLOUTS_PER_CALL) -> None:
        if max_rollouts_per_call < 1:
            raise ValueError("max_rollouts_per_call must be positive")
        self.source, self.audit_fn, self.clock = source, audit_fn, clock
        self.max_rollouts_per_call = int(max_rollouts_per_call)
        self.done = 0
        self.seconds = 0.0
        self.calls = 0

    def run(self, model, *, limit: int, budget_s: float) -> int:
        """Judge up to ``limit`` queued audits within about ``budget_s`` seconds; returns how many were reported."""
        if limit <= 0:
            return 0
        began = self.clock()
        many = getattr(self.audit_fn, "many", None)
        # taken a few at a time: a group body is large, and what the budget leaves is downloaded again. A take
        # returns the queue's head until it is reported, so each take is judged (or left to the budget) whole.
        chunk = AUDIT_CHUNK if many is None else min(
            AUDIT_TAKE_MAX, max(AUDIT_CHUNK, 2 * -(-self.max_rollouts_per_call // GROUP_ROLLOUTS)))
        done = 0
        while done < limit and not (done and self.clock() - began > budget_s):
            jobs = self.source.take_audits(min(chunk, limit - done))
            if not jobs:
                break
            calls = [[g] for g in jobs] if many is None else pack_by_rollouts(jobs, self.max_rollouts_per_call)
            for call in calls:
                if done and self.clock() - began > budget_s:
                    break                       # the rest stay queued for the next round
                results = [self.audit_fn(model, call[0])] if many is None else many(model, call)
                self.calls += 1
                for group, result in zip(call, results, strict=True):
                    self._report(group, result)
                done += len(call)
        spent = self.clock() - began
        self.done += done
        self.seconds += spent
        if done:
            logger.info("trainer audits: %d judged in %.1fs", done, spent)
        return done

    def _report(self, group, result) -> None:
        try:
            self.source.report_audit(group.lease_id, result)
        except Exception as exc:
            if not (isinstance(exc, LookupError) or getattr(exc, "status", None) == 404):
                raise
            # concluded meanwhile (dropped as stale): nothing to report

    def metrics(self) -> dict[str, float]:
        """Since the last call: audits judged, the seconds they took (and the forward calls, when batched)."""
        out = {"async/audits": float(self.done), "async/audit_s": self.seconds}
        if self.calls != self.done:
            out["async/audit_calls"] = float(self.calls)
        self.done, self.seconds, self.calls = 0, 0.0, 0
        return out


__all__ = ["TrainerAudits", "audit_group", "pack_by_rollouts", "broadcasting_audit", "diagnostics_from_rows", "gateway_audit",
           "group_diagnostics", "head_projection",
           "rollout_diagnostics"]
