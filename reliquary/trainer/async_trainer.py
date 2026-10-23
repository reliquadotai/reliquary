"""The async trainer loop (rl-async-design.md §1, §3, §4.6-4.9; implementation map §1.8-1.10, P4).

Next to the window trainer (``worker.py`` / ``train_runner.py``, unchanged), one step of ``AsyncTrainer.run_once``:

1. **Batch in arrival order.** ``batch_groups`` groups of any env are taken from the ``TrainingBuffer`` oldest
   first; the step waits until the batch is full (as prime-rl, AReaL, PipelineRL, verl, SkyRL). The mix is not the
   trainer's business: the per-env prices, driven by this queue, move each env's supply to its target share
   (``rl_price``), and the global lease cap holds the miners back when they outrun the trainer. Groups generated
   more than ``max_lag`` checkpoints ago are dropped unpaid without a forward (the cap keeps that rare).
2. **One forward = verification + loss.** Each ``SubmittedGroup`` becomes the trainer's group object; its rollouts
   carry the lease's Gumbel uniforms (``draw_uniforms``). ``train_step(..., gate=DrawGate)`` runs the substitution
   loss: per micro-batch the forward gives v*, each rollout's tallies and the loss; rollouts under
   ``rollout_floor`` leave the loss before the backward. Once every micro-batch ran, ``DrawGate.judge`` pools each
   group's tallies (§4.6: one bad rollout does not kill its group); a group under ``group_floor`` is rejected and
   its admitted rollouts are backed out by a second pass with negated advantages before the optimizer step (only
   failed groups pay it, honest groups never do). pi_old is the policy itself: ratio 1, exact because every step is
   published as a checkpoint (``steps_per_checkpoint`` must be 1: a second step would train with no clip and verify
   miners against weights never published).
3. **After the step.** ``report_training`` for every group (accepted with per-rollout match and matched tokens;
   rejected, unpaid and unsanctioned: honest drift fails groups too), a checkpoint every step: a CPU snapshot on the step path, then ``DeltaPublisher.publish`` and ``LeaseBook.adopt_checkpoint`` in a
   publish thread (one in flight: the next publication waits for it, and its error stops the loop), a Hugging Face
   backup every ``hf_backup_every`` steps (``BackgroundHfBackup``, never blocking).

A step that raises (e.g. out of GPU memory) reports nothing: its groups go back to the buffer, behind the fresh ones
and one per batch, and a group handed out ``max_attempts`` times without a report is reported untrained (unpaid,
unsanctioned), so one group that breaks every step cannot stall training.

Groups are reported (an accepted one becomes payable) before their step's checkpoint is published: pay is for
work trained into the weights the step computed. If the process dies between the report and the publication,
that step's weights are lost, but the miners did the work and keep its pay (deliberate, not a race to fix).

Floors are PLACEHOLDERS to calibrate at GPU qualification: ``ASYNC_ROLLOUT_FLOOR`` =
``constants.RL_ROLLOUT_MATCH_FLOOR`` (0.90) and
``ASYNC_GROUP_FLOOR`` 0.90 (pooled); a group with a forwarded rollout under ``ASYNC_FORGERY_FLOOR`` (0.5) is
rejected whatever its pool (one forged rollout must not ride on honest ones). A rollout's match
(``rl_draw.rollout_rate``) is its rate on stochastic positions (max p < 0.99) when it has
``FORCED_SEED_ROLLOUT_MIN_STOCH`` of them, else its rate on every drawn position (a peaked honest rollout matches
~1 there; arbitrary tokens do not), so the trainer's mask, the audit and the ledger's pay re-check
(``RlPayLedger``, same floor) read one number. The step is one batch: the loss is the global token mean, and
masked rollouts stay in its denominator (it is planned before the forward).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from reliquary.constants import (
    RL_FORGERY_MATCH_FLOOR,
    RL_GAP_TOLERANCE,
    RL_GROUP_MATCH_FLOOR,
    RL_ROLLOUT_MATCH_FLOOR,
    RL_SUBSTITUTE_GAP,
    RL_TERMINAL_GAP,
)
from reliquary.protocol.rl_draw import pooled_rate, rollout_rate
from reliquary.validator import telemetry
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_leases import SECRET_REVEAL_LAG
from reliquary.validator.training import (
    DRIFT_ENTROPY_EDGES,
    DRIFT_GAP_EDGES,
    DRIFT_POSITION_EDGES,
    DRIFT_WIDTH,
    TrainingStepSkipped,
    train_step as _default_train_step,
)

logger = logging.getLogger(__name__)

ASYNC_ROLLOUT_FLOOR = RL_ROLLOUT_MATCH_FLOOR   # placeholder (design §4.6), the ledger's floor too
# Placeholder: pooled match (every forwarded rollout) a group needs to be trained and paid. Under the rollout floor
# so that one noisy honest rollout does not kill its group (§4.6): 15 at 0.986 + 1 at 0.8 pool to 0.974; one under
# ``ASYNC_FORGERY_FLOOR`` (no honest noise: a forged rollout) rejects the group whatever the pool. Calibrate with the
# rollout floor at qualification.
ASYNC_GROUP_FLOOR = RL_GROUP_MATCH_FLOOR
ASYNC_FORGERY_FLOOR = RL_FORGERY_MATCH_FLOOR


class _Replay:
    """The gate of the back-out pass: same ratio-1 loss, every rollout admitted, nothing recorded."""

    def admit(self, rollouts, tallies, suspects=None):
        return None


class DrawGate:
    """Per-step verification state, carried through ``train_step`` (picklable: the FSDP2 followers get a copy and
    ``judge`` all-reduces the tallies, so every rank decides the same groups). Rollouts are known by
    ``_async_slot``; ``groups[g]`` lists group g's slots.

    A rollout is admitted when its match reaches ``rollout_floor`` and none of its tokens is further than
    ``gap_tolerance`` from the pick (``RL_GAP_TOLERANCE``), and, when it ends on one of ``eos_ids`` (not an
    episode), that EOS has a gap of at most ``terminal_gap``: a cut-short rollout keeps no trace for the re-grade. Its mismatches over ``substitute_gap`` are kept
    (``substitutions``): the validator re-grades the rollout with the trainer's picks there."""

    def __init__(self, groups: list[list[int]], *, rollout_floor: float = ASYNC_ROLLOUT_FLOOR,
                 group_floor: float = ASYNC_GROUP_FLOOR, forgery_floor: float = ASYNC_FORGERY_FLOOR,
                 gap_tolerance: float = RL_GAP_TOLERANCE, substitute_gap: float = RL_SUBSTITUTE_GAP,
                 terminal_gap: float = RL_TERMINAL_GAP, eos_ids=(), envs=None, lags=None) -> None:
        self.groups = [list(g) for g in groups]
        # Per-group labels for the drift breakdowns (telemetry only).
        self.envs = list(envs) if envs is not None else None
        self.lags = [int(v) for v in lags] if lags is not None else None
        self.rollout_floor, self.group_floor = float(rollout_floor), float(group_floor)
        self.forgery_floor = float(forgery_floor)
        self.gap_tolerance, self.substitute_gap = float(gap_tolerance), float(substitute_gap)
        self.terminal_gap = float(terminal_gap)
        self.eos_ids = frozenset(int(t) for t in eos_ids)
        self.extra: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        n = sum(len(g) for g in self.groups)
        self.tallies = [[0, 0, 0, 0] for _ in range(n)]
        self.max_gap = [0.0] * n
        self.substitutions: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        self.forwarded = [0] * n
        self.bad_terminal = [0] * n
        self.admitted = [0] * n
        self.drift = [[0.0] * DRIFT_WIDTH for _ in range(n)]
        self.accepted: list[bool] | None = None
        self.undone = 0

    def admit(self, rollouts, tallies, suspects=None) -> list[bool]:
        out = []
        for i, (rollout, tally) in enumerate(zip(rollouts, tallies)):
            slot = int(rollout._async_slot)
            self.tallies[slot] = [int(v) for v in tally]
            self.forwarded[slot] = 1
            row = getattr(rollout, "_drift_row", None)
            if row is not None and len(row) == DRIFT_WIDTH:
                self.drift[slot] = [float(v) for v in row]
            if suspects is not None:
                self.max_gap[slot], self.substitutions[slot] = float(suspects[i][0]), list(suspects[i][1])
                tokens, meta = rollout.commit["tokens"], rollout.commit.get("rollout", {})
                end = suspects[i][2] if len(suspects[i]) > 2 else None
                if end is not None and tokens and int(tokens[-1]) in self.eos_ids and "episode" not in meta:
                    self.bad_terminal[slot] = int(end > self.terminal_gap)
            ok = (rollout_rate(tally) >= self.rollout_floor and self.max_gap[slot] <= self.gap_tolerance
                  and not self.bad_terminal[slot])
            self.admitted[slot] = int(ok)
            out.append(ok)
        return out

    def replay(self) -> _Replay:
        return _Replay()

    def rate(self, slot: int) -> float:
        return rollout_rate(self.tallies[slot]) if self.forwarded[slot] else 0.0

    def pooled(self, g: int) -> float:
        return pooled_rate([self.tallies[s] for s in self.groups[g] if self.forwarded[s]])

    def forged(self, g: int) -> bool:
        """A forwarded rollout of group g under the forgery floor: no honest noise, the group is rejected."""
        return any(self.forwarded[s] and self.rate(s) < self.forgery_floor for s in self.groups[g])

    def judge(self, plan, dist=None) -> list:
        """Decide every group; return the back-out plan (rejected groups' admitted rollouts, negated advantages,
        same per-token scale) for ``train_step``'s second pass."""
        if dist is not None:
            import torch

            from reliquary.trainer.distributed import all_reduce_tensor_sum

            flat = torch.tensor([t + [f, a, m, b] + d for t, f, a, m, b, d in zip(self.tallies, self.forwarded,
                                                                                   self.admitted, self.max_gap,
                                                                                   self.bad_terminal, self.drift)],
                                dtype=torch.float64, device=dist.device)
            rows = all_reduce_tensor_sum(flat).cpu().tolist()      # each slot is forwarded on one rank only
            self.tallies = [[int(v) for v in row[:4]] for row in rows]
            self.forwarded = [int(row[4]) for row in rows]
            self.admitted = [int(row[5]) for row in rows]
            self.max_gap = [float(row[6]) for row in rows]
            self.bad_terminal = [int(row[7]) for row in rows]
            self.drift = [[float(v) for v in row[8:]] for row in rows]
            if torch.distributed.is_initialized():
                every = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(every, self.substitutions)
                self.substitutions = [[pair for rank in every for pair in rank[s]] for s in range(len(self.tallies))]
        self.accepted = [any(self.admitted[s] for s in slots) and self.pooled(g) >= self.group_floor
                         and not self.forged(g) for g, slots in enumerate(self.groups)]
        undo = []
        for group, advantages, scale in plan:
            g = int(group.rollouts[0]._async_group)
            if self.accepted[g]:
                continue
            kept = [(r, -a) for r, a in zip(group.rollouts, advantages) if self.admitted[int(r._async_slot)]]
            if kept:
                self.undone += 1
                undo.append((SimpleNamespace(rollouts=[r for r, _ in kept], prompt_idx=group.prompt_idx),
                             [a for _, a in kept], scale))
        return undo

    def metrics(self) -> dict[str, float]:
        forwarded = sum(self.forwarded)
        gaps = [g for g, f in zip(self.max_gap, self.forwarded) if f]
        out = {"async/rollouts_forwarded": forwarded,
               "async/rollouts_below_floor": forwarded - sum(self.admitted),
               "async/groups_undone": self.undone,
               "async/gap_max": max(gaps, default=0.0),
               "async/rollouts_over_gap_tolerance": sum(g > self.gap_tolerance for g in gaps),
               "async/rollouts_bad_termination": sum(self.bad_terminal),
               "async/rollouts_regraded": sum(bool(s) for s in self.substitutions),
               "async/substituted_tokens_regraded": sum(len(s) for s in self.substitutions),
               **{f"async/rollouts_gap_over_{e}": sum(g > e for g in gaps) for e in (1, 2, 4, 8)}}
        if self.accepted is not None:
            out["async/groups_accepted"] = sum(self.accepted)
            out["async/groups_rejected"] = len(self.accepted) - sum(self.accepted)
        return {**out, **self.drift_metrics(), **self.extra}

    def drift_metrics(self) -> dict[str, float]:
        """Miner-vs-trainer drift broken down by lag bucket, env, token position, entropy and Gumbel gap."""
        out: dict[str, float] = {}
        slots = [s for s, f in enumerate(self.forwarded) if f]
        if not slots:
            return out
        gaps = sorted(self.max_gap[s] for s in slots)
        for q in (50, 90, 99):
            out[f"async/rollout_gap_p{q}"] = gaps[min(len(gaps) - 1, int(len(gaps) * q / 100))]
        total = [sum(self.drift[s][i] for s in slots) for i in range(DRIFT_WIDTH)]
        out.update(_drift_breakdown(total, "async/drift"))
        if self.lags is not None:
            lags = sorted(self.lags)
            out["async/lag_max"] = lags[-1]
            out["async/lag_p90"] = lags[min(len(lags) - 1, int(len(lags) * 0.9))]
        labels = []
        if self.lags is not None:
            labels.append(("by_lag", [_lag_bucket(v) for v in self.lags]))
        if self.envs is not None:
            labels.append(("by_env", self.envs))
        for kind, names in labels:
            for name in sorted(set(names)):
                members = [g for g, n in enumerate(names) if n == name]
                out.update(self._slice_metrics(f"async/{kind}/{name}", members))
        return out

    def _slice_metrics(self, prefix: str, members: list[int]) -> dict[str, float]:
        slots = [s for g in members for s in self.groups[g] if self.forwarded[s]]
        tallies = [sum(self.tallies[s][i] for s in slots) for i in range(4)]
        out = {f"{prefix}/groups": len(members), f"{prefix}/rollouts": len(slots),
               f"{prefix}/match_rate": tallies[1] / tallies[0] if tallies[0] else 1.0,
               f"{prefix}/substitution_rate": 1.0 - tallies[3] / tallies[2] if tallies[2] else 0.0}
        if slots:
            n = len(slots)
            out[f"{prefix}/below_floor_ratio"] = sum(1 - self.admitted[s] for s in slots) / n
            out[f"{prefix}/over_gap_tolerance_ratio"] = sum(self.max_gap[s] > self.gap_tolerance for s in slots) / n
            out[f"{prefix}/regraded_ratio"] = sum(bool(self.substitutions[s]) for s in slots) / n
            rows = sum(self.drift[s][1] for s in slots)
            out[f"{prefix}/entropy"] = sum(self.drift[s][0] for s in slots) / rows if rows else 0.0
        if self.accepted is not None and members:
            out[f"{prefix}/groups_rejected_ratio"] = sum(not self.accepted[g] for g in members) / len(members)
        return out


def _lag_bucket(lag: int) -> str:
    return str(lag) if lag < 3 else "3plus"


def _edge_names(edges, unit: str = "") -> list[str]:
    fmt = lambda v: f"{v // 1024}k" if unit == "k" else f"{v:g}".replace(".", "p")  # noqa: E731 (no dots in wandb keys)
    return ([f"lt{fmt(edges[0])}"] + [f"{fmt(a)}_{fmt(b)}" for a, b in zip(edges, edges[1:])]
            + [f"ge{fmt(edges[-1])}"])


def _drift_breakdown(total: list[float], prefix: str) -> dict[str, float]:
    """Match rate per position and entropy bucket, and mismatch-gap quantiles, from summed drift rows."""
    out: dict[str, float] = {}
    col = 2
    for kind, edges, unit in (("match_by_position", DRIFT_POSITION_EDGES, "k"),
                              ("match_by_entropy", DRIFT_ENTROPY_EDGES, "")):
        for i, name in enumerate(_edge_names(edges, unit)):
            n_stoch, n_match = total[col + 2 * i], total[col + 2 * i + 1]
            out[f"{prefix}/{kind}/{name}"] = n_match / n_stoch if n_stoch else 1.0
            out[f"{prefix}/stochastic_tokens/{kind.split('_by_')[1]}/{name}"] = n_stoch
        col += 2 * (len(edges) + 1)
    bins = total[col:col + len(DRIFT_GAP_EDGES) + 1]
    missed = sum(bins)
    out[f"{prefix}/mismatched_tokens"] = missed
    for q in (50, 90, 99):
        # Upper edge of the bin holding the quantile (inf past the last edge is reported as twice it).
        value, seen = 0.0, 0.0
        for i, count in enumerate(bins):
            seen += count
            if missed and seen >= missed * q / 100:
                value = DRIFT_GAP_EDGES[i] if i < len(DRIFT_GAP_EDGES) else 2 * DRIFT_GAP_EDGES[-1]
                break
        out[f"{prefix}/mismatch_gap_p{q}"] = value
    return out


def _env_length_metrics(submitted: list[SubmittedGroup]) -> dict[str, float]:
    """Per env: mean completion length and the share of truncated rollouts in the batch."""
    tokens: dict[str, list[int]] = {}
    cut: dict[str, int] = {}
    for group in submitted:
        tokens.setdefault(group.environment, []).extend(len(c) for c in group.completions)
        cut[group.environment] = cut.get(group.environment, 0) + sum(bool(t) for t in group.truncated)
    return {**{f"async/by_env/{env}/completion_tokens_mean": sum(v) / len(v) for env, v in tokens.items() if v},
            **{f"async/by_env/{env}/truncated_ratio": cut[env] / len(v) for env, v in tokens.items() if v}}


def batch_uniforms(submitted) -> list[list[list[float]]]:
    """Every rollout's lease uniforms of a batch (``SubmittedGroup.draw_uniforms``), computed in parallel: ~190M
    positions a step kept the 8 GPUs idle ~8 min on one core (teutonic-rl-async-v1, 2026-10-23)."""
    from reliquary.protocol.rl_draw import lease_uniforms_many

    jobs = [(g.nonce, g.prompt_idx, g.checkpoint, i, g.model_tokens(i)) for g in submitted
            for i in range(len(g.completions))]
    flat = iter(lease_uniforms_many(jobs))
    return [[next(flat) for _ in g.completions] for g in submitted]


def training_group(group: SubmittedGroup, *, slot0: int, index: int,
                   uniforms: list[list[float]] | None = None) -> SimpleNamespace:
    """A buffered group as the trainer's group object (the window decoder's shape): full token rows, the BFT span
    in absolute positions, the lease's uniforms, and the gate's slot ids. Miners send no logprobs; the
    ``token_logprobs`` placeholder only carries the policy-position count (pi_old is the policy under the gate).
    An episode (§4.10) is the window path's Episode v1 shape: its validated model spans (absolute) are the policy
    mask (``_validated_assistant_spans``), so tool observations get no loss, no substitution and no match, and
    its uniforms are one per model token across turns. ``uniforms``: the rollouts' uniforms already computed
    (``batch_uniforms``), else each is recomputed here."""
    p = len(group.prompt_tokens)
    rollouts = []
    for i, completion in enumerate(group.completions):
        n = len(completion)
        span = group.force_spans[i]
        spans = None if group.spans is None else tuple((p + int(s), p + int(e)) for s, e in group.spans[i])
        meta = {"prompt_length": p, "completion_length": n, "truncated": bool(group.truncated[i]),
                "forced": span is not None, "token_logprobs": [0.0] * group.model_tokens(i)}
        if spans is not None:
            meta["episode"] = {"assistant_spans": [list(s) for s in spans]}
        rollout = SimpleNamespace(
            reward=float(group.rewards[i]), env_name=group.environment,
            commit={"tokens": list(group.prompt_tokens) + list(completion), "rollout": meta},
            _forced_draw_uniforms=group.draw_uniforms(i) if uniforms is None else uniforms[i],
            _async_slot=slot0 + i, _async_group=index)
        if spans is not None:
            rollout._validated_assistant_spans = spans
        if span is not None:
            rollout._validated_force_span = (p + int(span[0]), p + int(span[1]))
        rollouts.append(rollout)
    return SimpleNamespace(rollouts=rollouts, prompt_idx=group.prompt_idx, lease_id=group.lease_id,
                           environment=group.environment)


def _reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory() -> dict[str, float]:
    """This process's peak CUDA memory over the step (rank 0 under FSDP2)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"async/gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "async/gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}


def save_snapshot(snapshot: Mapping[str, Any], folder: Path, *, lr_schedule_step: int | None) -> None:
    from reliquary.shared.ckpt_delta import write_anchor_shards

    write_anchor_shards(snapshot, Path(folder))


@dataclass(frozen=True)
class AsyncTrainerConfig:
    batch_groups: int                        # groups per optimizer step
    steps_per_checkpoint: int = 1            # publish + adopt every step: the only safe value (pi_old = the policy)
    hf_backup_every: int | None = None       # M: full HF backup every M steps (None: off)
    rollout_floor: float = ASYNC_ROLLOUT_FLOOR
    group_floor: float = ASYNC_GROUP_FLOOR
    forgery_floor: float = ASYNC_FORGERY_FLOOR
    gap_tolerance: float = RL_GAP_TOLERANCE         # a token further than this from the pick: rollout refused
    substitute_gap: float = RL_SUBSTITUTE_GAP       # mismatches beyond this are re-graded with the pick
    terminal_gap: float = RL_TERMINAL_GAP           # the final EOS's gap
    eos_ids: tuple[int, ...] = ()                   # the model's stop tokens (the termination test's subject)
    max_lag: int = SECRET_REVEAL_LAG         # no quality cap (the floor decides); bounds the buffer only
    max_attempts: int = 3                    # hand-outs of a group without a report before it is dropped untrained
    poll_s: float = 0.5                      # while the batch is not full
    audits_per_step: int = 8                 # out-of-zone audits judged before each step (``trainer.audits``)
    audit_budget_s: float = 120.0            # their time budget per round; waiting for a batch runs more rounds

    def __post_init__(self) -> None:
        if self.batch_groups < 1 or (self.hf_backup_every or 1) < 1 or self.max_attempts < 1:
            raise ValueError("batch_groups, hf_backup_every and max_attempts must be positive")
        if self.audits_per_step < 0 or self.audit_budget_s < 0:
            raise ValueError("audits_per_step and audit_budget_s must be non-negative")
        if self.steps_per_checkpoint != 1:
            # pi_old is the policy under the gate: a second step before publishing trains with ratio 1 (no clip,
            # no dual clip, no drift breaker) and verifies miners against weights they never saw.
            raise ValueError("the async trainer publishes every step: steps_per_checkpoint must be 1")
        if not all(0 <= f <= 1 for f in (self.rollout_floor, self.group_floor, self.forgery_floor)) or self.max_lag < 0:
            raise ValueError("floors in [0, 1], max_lag >= 0")


class AsyncTrainer:
    """``run_once`` -> "trained" | "skipped" (health gate: groups reported untrained, unsanctioned); it waits for a
    full batch. ``state_fn`` returns the full state dict to publish (FSDP2: the rank-0 gather);
    ``train_step_fn`` may be ``distributed.broadcasting_train_step``. The model must hold checkpoint
    ``book.latest_checkpoint`` (or, on a fresh book, the initial weights, published and adopted as 0)."""

    def __init__(self, model: Any, *, config: AsyncTrainerConfig, buffer, submissions, book,
                 publisher=None, state_fn: Callable[[], Mapping[str, Any]] | None = None, backup=None,
                 backup_save: Callable[..., None] = save_snapshot,
                 train_step_fn: Callable = _default_train_step, ref_model: Any = None,
                 global_step_hint: int | None = None, audits=None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model, self.config = model, config
        self.buffer, self.submissions, self.book = buffer, submissions, book
        self.publisher, self.backup, self._backup_save = publisher, backup, backup_save
        self._state_fn = state_fn or (lambda: self.model.state_dict())
        self._train_step, self.ref_model, self.global_step_hint = train_step_fn, ref_model, global_step_hint
        self._clock, self._sleep = clock, sleep
        self.audits = audits                    # trainer.audits.TrainerAudits: out-of-zone audits on this model
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-publish")
        self._pending: Future | None = None
        self.since_checkpoint = 0
        self.published: list[int] = []
        self.last_step: dict = {}
        self.skipped = 0
        self.last_publication: dict = {}
        self.version = book.latest_checkpoint
        if self.version is None:
            self.version = 0
            self._publish(0, self._snapshot())
            self.wait_published()
        # One step per checkpoint: the book's latest checkpoint is the number of steps trained, so the step resumes
        # from it (HF backup names, wandb steps), unique and increasing across restarts. A step lost before its
        # publication is lost with its weights.
        self.steps = self.version * config.steps_per_checkpoint

    # --- checkpoints ---------------------------------------------------------------------------------------
    def _snapshot(self) -> dict:
        return {name: t.detach().to("cpu", copy=True).contiguous() for name, t in self._state_fn().items()}

    def _publish(self, version: int, snapshot: dict) -> None:
        self.wait_published()                         # one in flight; a failed publication stops the loop here
        self._pending = self._executor.submit(self._publish_job, version, snapshot)

    def _publish_job(self, version: int, snapshot: dict) -> None:
        started = time.monotonic()
        if self.publisher is not None:
            self.publisher.publish(version, snapshot)
        self.book.adopt_checkpoint(version)
        self.published.append(version)
        seconds = time.monotonic() - started
        deltas = [d["size"] for d in getattr(self.publisher, "deltas", []) if d["to"] == version and d["from"] == version - 1]
        self.last_publication = {"async/published_checkpoint": version, "async/publish_s": seconds,
                                 **({"async/delta_bytes": deltas[-1]} if deltas else {})}
        logger.info("async checkpoint %d published and adopted in %.1fs", version, seconds)
        measure = getattr(self.publisher, "measure_update", None)
        if measure is not None:                       # after adoption: miners never wait for it
            self.last_publication = {**self.last_publication, **measure()}

    def wait_published(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def close(self) -> None:
        try:
            self.wait_published()
        finally:
            self._executor.shutdown(wait=True)

    # --- batch ---------------------------------------------------------------------------------------------
    def _report_untrained(self, groups, failure_class=None) -> None:
        for group in groups:
            self.submissions.report_training(group.lease_id, rollout_match=[0.0] * len(group.completions),
                                             accepted=False, failure_class=failure_class, judged=False)

    def drop_stale(self) -> int:
        below = self.version - self.config.max_lag
        dropped = 0
        while True:
            stale = self.buffer.take_older(below, 256)
            if not stale:
                return dropped
            self._report_untrained(stale)
            dropped += len(stale)

    def drop_exhausted(self) -> int:
        """Groups handed out ``max_attempts`` times without a report (their steps raised): reported untrained."""
        take = getattr(self.buffer, "take_exhausted", None)
        if take is None:
            return 0
        dropped = 0
        while True:
            exhausted = take(self.config.max_attempts, 256)
            if not exhausted:
                return dropped
            logger.error("%d groups dropped untrained after %d failed steps: %s", len(exhausted),
                         self.config.max_attempts, ", ".join(g.lease_id[:12] for g in exhausted))
            self._report_untrained(exhausted)
            dropped += len(exhausted)

    def assemble(self) -> list[SubmittedGroup]:
        """``batch_groups`` groups in arrival order; waits until the batch is full, judging queued audits meanwhile."""
        taken: list[SubmittedGroup] = []
        while True:
            taken += self.buffer.take(self.config.batch_groups - len(taken))
            if len(taken) >= self.config.batch_groups:
                return taken
            if not self.run_audits():
                self._sleep(self.config.poll_s)

    def run_audits(self) -> int:
        """One round of out-of-zone audits on the current weights (none without ``audits``)."""
        if self.audits is None:
            return 0
        return self.audits.run(self.model, limit=self.config.audits_per_step, budget_s=self.config.audit_budget_s)

    # --- step ----------------------------------------------------------------------------------------------
    def run_once(self) -> str:
        began = self._clock()
        telemetry.hold()                # train_step's row and this step's timings go to wandb as one row
        self.buffer.requeue()           # between steps the trainer holds nothing: a take whose answer was lost
        dropped = self.drop_stale()
        exhausted = self.drop_exhausted()
        self.run_audits()                       # bounded: audits progress even when batches are always full
        submitted = self.assemble()
        started = self._clock()
        _reset_peak_memory()
        batch, slot = [], 0
        uniforms = batch_uniforms(submitted)
        for i, group in enumerate(submitted):
            batch.append(training_group(group, slot0=slot, index=i, uniforms=uniforms[i]))
            slot += len(group.completions)
        gate = DrawGate([[r._async_slot for r in g.rollouts] for g in batch],
                        rollout_floor=self.config.rollout_floor, group_floor=self.config.group_floor,
                        forgery_floor=self.config.forgery_floor, gap_tolerance=self.config.gap_tolerance,
                        substitute_gap=self.config.substitute_gap, terminal_gap=self.config.terminal_gap, eos_ids=self.config.eos_ids,
                        envs=[g.environment for g in submitted], lags=[self.version - g.checkpoint for g in submitted])
        envs: dict[str, int] = {}
        for group in submitted:
            envs[group.environment] = envs.get(group.environment, 0) + 1
        gate.extra = {"async/checkpoint": self.version, "async/stale_dropped": dropped,
                      "async/exhausted_dropped": exhausted,
                      "async/lag_mean": sum(self.version - g.checkpoint for g in submitted) / len(submitted),
                      **{f"async/batch_groups/{env}": n for env, n in envs.items()},
                      **_env_length_metrics(submitted)}
        try:
            self.model = self._train_step(self.model, [batch], ref_model=self.ref_model,
                                          window_index=self.steps + 1, global_step_hint=self.global_step_hint,
                                          gate=gate)
        except TrainingStepSkipped as exc:
            logger.warning("async train step skipped (%s): %d groups unpaid", exc.reason, len(submitted))
            self._report_untrained(submitted)
            self.skipped += 1           # its rows stay held and go out with the next trained step
            self.last_step = {"result": "skipped", "reason": exc.reason, **gate.metrics()}
            return "skipped"
        except BaseException:
            # Nothing reported: the groups go back (each hand-out counted), behind the fresh ones and one per batch.
            logger.exception("async train step raised; %d groups back to the buffer", len(submitted))
            self.buffer.release([g.lease_id for g in submitted])
            raise
        trained = self._clock()
        self.steps += 1
        self._report(submitted, gate)
        self.since_checkpoint += 1
        snapshot = None
        timings = {}
        if self.since_checkpoint >= self.config.steps_per_checkpoint:
            self.version += 1
            self.since_checkpoint = 0
            at = self._clock()
            snapshot = self._snapshot()
            timings["async/snapshot_s"] = self._clock() - at
            at = self._clock()
            self._publish(self.version, snapshot)
            timings["async/publish_wait_s"] = self._clock() - at
        if self.backup is not None and self.config.hf_backup_every and self.steps % self.config.hf_backup_every == 0:
            from reliquary.validator.training import current_lr_schedule_step

            frozen = snapshot if snapshot is not None else self._snapshot()
            lr_step = current_lr_schedule_step()
            self.backup.submit(self.steps, lambda folder, s=frozen, lr=lr_step:
                               self._backup_save(s, folder, lr_schedule_step=lr))
        completion = sum(len(c) for g in submitted for c in g.completions)
        worked = self._clock() - started
        self.last_step = {"result": "trained", "step": self.steps, "checkpoint": self.version, **gate.metrics()}
        audit_metrics = self.audits.metrics() if self.audits is not None else {}
        telemetry.flush({**gate.metrics(), **timings, **self.last_publication, **audit_metrics,
                         "async/wait_s": started - began, "async/train_s": trained - started,
                         "async/step_s": worked, "async/completion_tokens": completion,
                         "async/completion_tokens_per_s": completion / worked if worked > 0 else 0.0,
                         "async/steps_skipped_total": self.skipped, **_peak_memory()},
                        self.steps)
        return "trained"

    def _report(self, submitted: list[SubmittedGroup], gate: DrawGate) -> dict[str, int]:
        """``report_training`` for every group of the step; returns the matched tokens trained per env."""
        trained: dict[str, int] = {}
        slot = 0
        for g, group in enumerate(submitted):
            slots = range(slot, slot + len(group.completions))
            slot += len(group.completions)
            match = [gate.rate(s) for s in slots]
            if gate.accepted is not None and gate.accepted[g]:
                matched = [gate.tallies[s][3] if gate.admitted[s] else 0 for s in slots]
                self.submissions.report_training(group.lease_id, rollout_match=match, accepted=True,
                                                 matched_tokens=matched, judged=True,
                                                 substitutions=[gate.substitutions[s] for s in slots])
                trained[group.environment] = trained.get(group.environment, 0) + sum(matched)
                continue
            self.submissions.report_training(group.lease_id, rollout_match=match, accepted=False,
                                             judged=any(gate.forwarded[s] for s in slots))
        return trained


__all__ = [
    "ASYNC_FORGERY_FLOOR", "ASYNC_GROUP_FLOOR", "ASYNC_ROLLOUT_FLOOR", "AsyncTrainer", "AsyncTrainerConfig", "DrawGate",
    "rollout_rate", "save_snapshot", "training_group",
]
