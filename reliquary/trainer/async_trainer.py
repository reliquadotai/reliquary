"""The async trainer loop (rl-async-design.md §1, §3, §4.6-4.9; implementation map §1.8-1.10, P4).

Next to the window trainer (``worker.py`` / ``train_runner.py``, unchanged), one step of ``AsyncTrainer.run_once``:

1. **Batch at the target mix.** ``config.env_mix`` (the price controller's target shares) splits
   ``batch_groups`` across envs (largest remainder); each env's quota is taken from the ``TrainingBuffer``
   freshest first. A short env does not block: the loop polls for at most ``wait_s``, proceeds with what it has
   and reports the wait to the price controller (``note_wait``), which raises that env's price. Groups generated
   more than ``max_lag`` checkpoints ago are dropped unpaid without a forward (they would fail the floor anyway).
2. **One forward = verification + loss.** Each ``SubmittedGroup`` becomes the trainer's group object; its rollouts
   carry the lease's Gumbel uniforms (``draw_uniforms``). ``train_step(..., gate=DrawGate)`` runs the substitution
   loss: per micro-batch the forward gives v*, each rollout's tallies and the loss; rollouts under
   ``rollout_floor`` leave the loss before the backward. Once every micro-batch ran, ``DrawGate.judge`` pools each
   group's tallies (§4.6: one bad rollout does not kill its group); a group under ``group_floor`` is rejected and
   its admitted rollouts are backed out by a second pass with negated advantages before the optimizer step (only
   failed groups pay it, honest groups never do). pi_old is the policy itself: ratio 1, exact with one step per
   checkpoint (with ``steps_per_checkpoint > 1`` the forward verifies against the trainer's in-between weights, a
   documented approximation of "the published checkpoint").
3. **After the step.** ``report_training`` for every group (accepted with per-rollout match and matched tokens;
   rejected with ``statistical`` when the group's lag is at most ``sanction_max_lag``, else unsanctioned
   staleness), a checkpoint every ``steps_per_checkpoint`` steps: a CPU snapshot on the step path, then
   ``DeltaPublisher.publish`` and ``LeaseBook.adopt_checkpoint`` in a publish thread (one in flight: the next
   publication waits for it, and its error stops the loop), a Hugging Face backup every ``hf_backup_every`` steps
   (``BackgroundHfBackup``, never blocking), and one price tick with the step's trained tokens and buffer depth.

Floors are PLACEHOLDERS to calibrate at GPU qualification: ``ASYNC_ROLLOUT_FLOOR`` =
``constants.RL_ROLLOUT_MATCH_FLOOR`` (the design's 0.95 with bf16 heads, about 0.96 with fp32 heads) and
``ASYNC_GROUP_FLOOR`` 0.90 (pooled). A rollout's match is its rate on stochastic positions (max p < 0.99) when it has ``FORCED_SEED_ROLLOUT_MIN_STOCH`` of them, else its rate on every drawn position (a
peaked honest rollout matches ~1 there; arbitrary tokens do not), so the trainer's mask and the ledger's pay
re-check (``RlPayLedger``, same floor) read one number. Masked rollouts stay in their env's token-level
denominator (it is planned before the forward): the env is slightly under-weighted on such steps.
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
    FORCED_SEED_MIN_STOCH_POSITIONS,
    FORCED_SEED_ROLLOUT_MIN_STOCH,
    RL_ROLLOUT_MATCH_FLOOR,
)
from reliquary.services.exploration import AUDIT_FAILURE_STATISTICAL
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.training import TrainingStepSkipped, train_step as _default_train_step

logger = logging.getLogger(__name__)

ASYNC_ROLLOUT_FLOOR = RL_ROLLOUT_MATCH_FLOOR   # placeholder (design §4.6), the ledger's floor too
# Placeholder: pooled match (every forwarded rollout) a group needs to be trained and paid. Under the rollout floor
# so that one bad rollout does not kill its group (§4.6): 15 honest rollouts at 0.986 + 1 at 0 pool to 0.924; two
# such rollouts (0.863) do. Calibrate with the rollout floor at qualification.
ASYNC_GROUP_FLOOR = 0.90


def rollout_rate(tally, min_stoch: int = FORCED_SEED_ROLLOUT_MIN_STOCH) -> float:
    """Match of one rollout from its (stochastic, stochastic matches, drawn, drawn matches) tally; 0 when nothing
    was drawn (no public uniforms: nothing verified)."""
    n_stoch, m_stoch, n_drawn, m_drawn = tally
    if n_stoch >= min_stoch:
        return m_stoch / n_stoch
    return m_drawn / n_drawn if n_drawn else 0.0


class _Replay:
    """The gate of the back-out pass: same ratio-1 loss, every rollout admitted, nothing recorded."""

    def admit(self, rollouts, tallies):
        return None


class DrawGate:
    """Per-step verification state, carried through ``train_step`` (picklable: the FSDP2 followers get a copy and
    ``judge`` all-reduces the tallies, so every rank decides the same groups). Rollouts are known by
    ``_async_slot``; ``groups[g]`` lists group g's slots."""

    def __init__(self, groups: list[list[int]], *, rollout_floor: float = ASYNC_ROLLOUT_FLOOR,
                 group_floor: float = ASYNC_GROUP_FLOOR) -> None:
        self.groups = [list(g) for g in groups]
        self.rollout_floor, self.group_floor = float(rollout_floor), float(group_floor)
        self.extra: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        n = sum(len(g) for g in self.groups)
        self.tallies = [[0, 0, 0, 0] for _ in range(n)]
        self.forwarded = [0] * n
        self.admitted = [0] * n
        self.accepted: list[bool] | None = None
        self.undone = 0

    def admit(self, rollouts, tallies) -> list[bool]:
        out = []
        for rollout, tally in zip(rollouts, tallies):
            slot = int(rollout._async_slot)
            self.tallies[slot] = [int(v) for v in tally]
            self.forwarded[slot] = 1
            ok = rollout_rate(tally) >= self.rollout_floor
            self.admitted[slot] = int(ok)
            out.append(ok)
        return out

    def replay(self) -> _Replay:
        return _Replay()

    def rate(self, slot: int) -> float:
        return rollout_rate(self.tallies[slot]) if self.forwarded[slot] else 0.0

    def pooled(self, g: int) -> float:
        slots = [s for s in self.groups[g] if self.forwarded[s]]
        n_stoch = sum(self.tallies[s][0] for s in slots)
        if n_stoch >= FORCED_SEED_MIN_STOCH_POSITIONS:
            return sum(self.tallies[s][1] for s in slots) / n_stoch
        n_drawn = sum(self.tallies[s][2] for s in slots)
        return sum(self.tallies[s][3] for s in slots) / n_drawn if n_drawn else 0.0

    def judge(self, plan, dist=None) -> list:
        """Decide every group; return the back-out plan (rejected groups' admitted rollouts, negated advantages,
        same per-token scale) for ``train_step``'s second pass."""
        if dist is not None:
            import torch

            from reliquary.trainer.distributed import all_reduce_tensor_sum

            flat = torch.tensor([t + [f, a] for t, f, a in zip(self.tallies, self.forwarded, self.admitted)],
                                dtype=torch.float64, device=dist.device)
            rows = all_reduce_tensor_sum(flat).cpu().tolist()
            self.tallies = [[int(v) for v in row[:4]] for row in rows]
            self.forwarded = [int(row[4]) for row in rows]
            self.admitted = [int(row[5]) for row in rows]
        self.accepted = [any(self.admitted[s] for s in slots) and self.pooled(g) >= self.group_floor
                         for g, slots in enumerate(self.groups)]
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
        out = {"async/rollouts_forwarded": forwarded,
               "async/rollouts_below_floor": forwarded - sum(self.admitted),
               "async/groups_undone": self.undone}
        if self.accepted is not None:
            out["async/groups_accepted"] = sum(self.accepted)
            out["async/groups_rejected"] = len(self.accepted) - sum(self.accepted)
        return {**out, **self.extra}


def training_group(group: SubmittedGroup, *, slot0: int, index: int) -> SimpleNamespace:
    """A buffered group as the trainer's group object (the window decoder's shape): full token rows, the BFT span
    in absolute positions, the lease's uniforms, and the gate's slot ids. Miners send no logprobs; the
    ``token_logprobs`` placeholder only carries the completion length (pi_old is the policy under the gate)."""
    p = len(group.prompt_tokens)
    rollouts = []
    for i, completion in enumerate(group.completions):
        n = len(completion)
        span = group.force_spans[i]
        rollout = SimpleNamespace(
            reward=float(group.rewards[i]), env_name=group.environment,
            commit={"tokens": list(group.prompt_tokens) + list(completion),
                    "rollout": {"prompt_length": p, "completion_length": n, "truncated": bool(group.truncated[i]),
                                "forced": span is not None, "token_logprobs": [0.0] * n}},
            _forced_draw_uniforms=group.draw_uniforms(i), _async_slot=slot0 + i, _async_group=index)
        if span is not None:
            rollout._validated_force_span = (p + int(span[0]), p + int(span[1]))
        rollouts.append(rollout)
    return SimpleNamespace(rollouts=rollouts, prompt_idx=group.prompt_idx, lease_id=group.lease_id,
                           environment=group.environment)


def mix_quotas(mix: Mapping[str, float], total: int) -> dict[str, int]:
    """``total`` groups split by target share (largest remainder, ties by env name)."""
    weight = sum(mix.values())
    exact = {env: total * share / weight for env, share in mix.items() if share > 0}
    quotas = {env: int(v) for env, v in exact.items()}
    rest = sorted(exact, key=lambda env: (-(exact[env] - quotas[env]), env))
    for env in rest[: total - sum(quotas.values())]:
        quotas[env] += 1
    return quotas


def save_snapshot(snapshot: Mapping[str, Any], folder: Path) -> None:
    from reliquary.shared.ckpt_delta import write_anchor_shards

    write_anchor_shards(snapshot, Path(folder))


@dataclass(frozen=True)
class AsyncTrainerConfig:
    env_mix: Mapping[str, float]             # target share per env (the price controller's targets)
    batch_groups: int                        # groups per optimizer step at full supply
    steps_per_checkpoint: int = 1            # N: publish + adopt every N steps (design §4.8: 1)
    hf_backup_every: int | None = None       # M: full HF backup every M steps (None: off)
    rollout_floor: float = ASYNC_ROLLOUT_FLOOR
    group_floor: float = ASYNC_GROUP_FLOOR
    max_lag: int = 8                         # older groups are dropped unpaid without a forward
    sanction_max_lag: int = 2                # a failed group this fresh is evidence (honest lag <= 2 passes)
    wait_s: float = 5.0                      # bounded wait on short envs before stepping with what is there
    poll_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.env_mix or any(v < 0 for v in self.env_mix.values()) or sum(self.env_mix.values()) <= 0:
            raise ValueError("env_mix: non-negative shares, at least one positive")
        if self.batch_groups < 1 or self.steps_per_checkpoint < 1 or (self.hf_backup_every or 1) < 1:
            raise ValueError("batch_groups, steps_per_checkpoint and hf_backup_every must be positive")
        if not (0 <= self.rollout_floor <= 1 and 0 <= self.group_floor <= 1) or self.max_lag < 0:
            raise ValueError("floors in [0, 1], max_lag >= 0")


class AsyncTrainer:
    """``run_once`` -> "trained" | "skipped" (health gate: groups reported untrained, unsanctioned) | "waited"
    (no group at all). ``state_fn`` returns the full state dict to publish (FSDP2: the rank-0 gather);
    ``train_step_fn`` may be ``distributed.broadcasting_train_step``. The model must hold checkpoint
    ``book.latest_checkpoint`` (or, on a fresh book, the initial weights, published and adopted as 0)."""

    def __init__(self, model: Any, *, config: AsyncTrainerConfig, buffer, submissions, book, prices=None,
                 publisher=None, state_fn: Callable[[], Mapping[str, Any]] | None = None, backup=None,
                 backup_save: Callable[[Mapping[str, Any], Path], None] = save_snapshot,
                 train_step_fn: Callable = _default_train_step, ref_model: Any = None,
                 global_step_hint: int | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model, self.config = model, config
        self.buffer, self.submissions, self.book, self.prices = buffer, submissions, book, prices
        self.publisher, self.backup, self._backup_save = publisher, backup, backup_save
        self._state_fn = state_fn or (lambda: self.model.state_dict())
        self._train_step, self.ref_model, self.global_step_hint = train_step_fn, ref_model, global_step_hint
        self._clock, self._sleep = clock, sleep
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt-publish")
        self._pending: Future | None = None
        self.steps = 0
        self.since_checkpoint = 0
        self.published: list[int] = []
        self.last_step: dict = {}
        self.version = book.latest_checkpoint
        if self.version is None:
            self.version = 0
            self._publish(0, self._snapshot())
            self.wait_published()

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
        logger.info("async checkpoint %d published and adopted in %.1fs", version, time.monotonic() - started)

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
                                             accepted=False, failure_class=failure_class)

    def drop_stale(self) -> int:
        below = self.version - self.config.max_lag
        dropped = 0
        while True:
            stale = self.buffer.take_older(below, 256)
            if not stale:
                return dropped
            self._report_untrained(stale)
            dropped += len(stale)

    def assemble(self) -> tuple[dict[str, list[SubmittedGroup]], dict[str, float]]:
        """Groups per env at the target mix, freshest first; (groups, wait seconds of each short env)."""
        quotas = mix_quotas(self.config.env_mix, self.config.batch_groups)
        taken: dict[str, list[SubmittedGroup]] = {env: [] for env in quotas}
        started = self._clock()
        while True:
            for env, quota in quotas.items():
                if len(taken[env]) < quota:
                    taken[env] += self.buffer.take(env, quota - len(taken[env]))
            short = [env for env, quota in quotas.items() if len(taken[env]) < quota]
            if not short or self._clock() - started >= self.config.wait_s:
                break
            self._sleep(self.config.poll_s)
        waited = self._clock() - started
        waits = {env: waited for env in short} if waited > 0 else {}
        if self.prices is not None:
            for env, seconds in waits.items():
                self.prices.note_wait(env, seconds)
        return taken, waits

    # --- step ----------------------------------------------------------------------------------------------
    def run_once(self) -> str:
        dropped = self.drop_stale()
        taken, waits = self.assemble()
        submitted = [g for env in taken for g in taken[env]]
        if not submitted:
            return "waited"
        batches, slot = [], 0
        for env in taken:
            batch = []
            for group in taken[env]:
                batch.append(training_group(group, slot0=slot, index=len(batch) + sum(len(b) for b in batches)))
                slot += len(group.completions)
            batches.append(batch)
        gate = DrawGate([[r._async_slot for r in g.rollouts] for b in batches for g in b],
                        rollout_floor=self.config.rollout_floor, group_floor=self.config.group_floor)
        depth = self.buffer.depth()
        gate.extra = {"async/checkpoint": self.version, "async/stale_dropped": dropped,
                      "async/lag_mean": sum(self.version - g.checkpoint for g in submitted) / len(submitted),
                      **{f"async/buffer_depth/{env}": depth.get(env, 0) for env in self.config.env_mix},
                      **{f"async/batch_groups/{env}": len(taken[env]) for env in taken},
                      **{f"async/wait_s/{env}": waits.get(env, 0.0) for env in taken}}
        try:
            self.model = self._train_step(self.model, batches, ref_model=self.ref_model,
                                          window_index=self.steps + 1, global_step_hint=self.global_step_hint,
                                          gate=gate)
        except TrainingStepSkipped as exc:
            logger.warning("async train step skipped (%s): %d groups unpaid", exc.reason, len(submitted))
            self._report_untrained(submitted)
            self.last_step = {"result": "skipped", "reason": exc.reason, **gate.metrics()}
            return "skipped"
        except BaseException:
            self.buffer.release([g.lease_id for g in submitted])
            raise
        self.steps += 1
        trained_tokens = self._report(submitted, gate)
        self.since_checkpoint += 1
        snapshot = None
        if self.since_checkpoint >= self.config.steps_per_checkpoint:
            self.version += 1
            self.since_checkpoint = 0
            snapshot = self._snapshot()
            self._publish(self.version, snapshot)
        if self.backup is not None and self.config.hf_backup_every and self.steps % self.config.hf_backup_every == 0:
            frozen = snapshot if snapshot is not None else self._snapshot()
            self.backup.submit(self.steps, lambda folder, s=frozen: self._backup_save(s, folder))
        if self.prices is not None:
            self.prices.tick(trained_tokens, self.buffer.depth())
        self.last_step = {"result": "trained", "step": self.steps, "checkpoint": self.version, **gate.metrics()}
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
                                                 matched_tokens=matched)
                trained[group.environment] = trained.get(group.environment, 0) + sum(matched)
                continue
            forwarded = any(gate.forwarded[s] for s in slots)
            evidence = forwarded and self.version - group.checkpoint <= self.config.sanction_max_lag
            self.submissions.report_training(group.lease_id, rollout_match=match, accepted=False,
                                             failure_class=AUDIT_FAILURE_STATISTICAL if evidence else None)
        return trained


__all__ = [
    "ASYNC_GROUP_FLOOR", "ASYNC_ROLLOUT_FLOOR", "AsyncTrainer", "AsyncTrainerConfig", "DrawGate", "mix_quotas",
    "rollout_rate", "save_snapshot", "training_group",
]
