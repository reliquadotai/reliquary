"""The async trainer's step on a Miles Tinker gateway (``RELIQUARY_TRAINER_BACKEND=miles``).

``MilesTrainStep`` has ``training.train_step``'s signature and update rule; the weights and Adam live in the gateway
(Megatron, full fine-tuning). One step: the plan (``_plan_from_batches``), every forwarded rollout as a
``reliquary_gate`` datum (verification + substitution loss in one forward), ``forward_backward`` in chunks, then one
``optim_step``: the gateway judges the groups, backs the rejected ones out, and steps unless the gradient is not finite
or over ``GRAD_NORM_SKIP_THRESHOLD`` (its ``--tinker-grad-norm-skip``). The learning-rate schedule is the trainer's
(linear warmup, then flat; a short re-warmup after a restart: Adam starts fresh).
"""
from __future__ import annotations

import logging
import math

from reliquary.constants import (
    ADAM_BETA2,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LR_RESTART_REWARMUP_WINDOWS,
    LR_WARMUP_WINDOWS,
)
from reliquary.trainer.miles_datums import step_datums
from reliquary.trainer.miles_gate import apply_gateway_verdicts, fill_gate
from reliquary.validator import telemetry
from reliquary.validator.training import (
    TrainingStepSkipped,
    _bft_training_metrics,
    _overlong_training_metrics,
    _plan_from_batches,
    _shaping_training_metrics,
    _training_environment_metrics,
)

logger = logging.getLogger(__name__)

WEIGHT_DECAY = 0.01                       # train_step's AdamW
ADAM_EPS = 1e-8


class MilesTrainStep:
    def __init__(self, client, *, eos_ids, chunk: int = 64, global_step_hint: int | None = None) -> None:
        self.client, self.chunk = client, int(chunk)
        self.config = {"eos_ids": ",".join(str(int(t)) for t in eos_ids)} if eos_ids else {}
        self.hint = max(0, int(global_step_hint or 0))
        self.step = self.hint                 # scheduler steps taken (train_step's ``_scheduler.last_epoch``)

    def lr_for(self, step: int) -> float:
        factor = (step + 1) / LR_WARMUP_WINDOWS if step < LR_WARMUP_WINDOWS else 1.0
        if self.hint > 0 and step >= self.hint:
            factor *= min(1.0, (step - self.hint + 1) / (LR_RESTART_REWARMUP_WINDOWS + 1))
        return LEARNING_RATE * factor

    def __call__(self, model, batches, *, ref_model=None, behavior_model=None, window_index=None,
                 global_step_hint=None, gate):
        if ref_model is not None or behavior_model is not None:
            raise RuntimeError("the Miles backend trains with KL_BETA == 0 and no ref/behavior model")
        import tinker

        plan, n_skipped = _plan_from_batches(batches)
        if not plan:
            logger.info("miles step: no trainable groups")
            return model
        datums, slots = step_datums(plan)
        by_slot = {int(r._async_slot): r for group, _, _ in plan for r in group.rollouts}
        futures = [self.client.forward_backward(datums[i:i + self.chunk], "reliquary_gate", self.config)
                   for i in range(0, len(datums), self.chunk)]
        outputs = [out for future in futures for out in future.result().loss_fn_outputs]
        fill_gate(gate, slots, outputs, [by_slot[s] for _, s in slots])
        lr = self.lr_for(self.step)
        adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=ADAM_BETA2, eps=ADAM_EPS,
                                 weight_decay=WEIGHT_DECAY, grad_clip_norm=GRAD_CLIP_NORM)
        result = dict(self.client.optim_step(adam).result().metrics or {})
        apply_gateway_verdicts(gate, result)
        grad_norm = float(result.get("grad_norm", math.nan))
        spike, nonfinite = bool(result.get("skipped_grad_spike")), bool(result.get("skipped_nonfinite"))
        metrics = {
            "train/grad_norm": grad_norm, "train/lr_applied": lr,
            "train/step_skipped_nonfinite": float(nonfinite), "train/step_skipped_grad_spike": float(spike),
            "train/rollouts_processed": len(datums),
            "train/rollouts_total": sum(len(g.rollouts) for b in batches for g in b),
            "batch/n_degenerate_groups": n_skipped,
            **{f"miles/{k}": float(v) for k, v in result.items() if not k.startswith("gate_group_rejected/")},
            **_bft_training_metrics(plan), **_shaping_training_metrics(plan), **_overlong_training_metrics(plan),
            **_training_environment_metrics(batches, plan), **gate.metrics(),
        }
        if spike or nonfinite:
            telemetry.log_training_step(metrics, step=window_index)
            reason = "nonfinite_gradient" if nonfinite else "gradient_spike"
            logger.warning("miles step rejected by the gateway: %s grad_norm=%s", reason, grad_norm)
            raise TrainingStepSkipped(reason, grad_norm, metrics=metrics)
        self.step += 1
        rewards = [r.reward for b in batches for g in b for r in g.rollouts]
        mean = sum(rewards) / len(rewards)
        metrics.update({
            "train/lr": self.lr_for(self.step), "train/lr_next": self.lr_for(self.step),
            "train/grad_clip_ratio": grad_norm / GRAD_CLIP_NORM, "train/grad_was_clipped": float(grad_norm > GRAD_CLIP_NORM),
            "rewards/mean": mean, "rewards/std": (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5,
            "rewards/min": min(rewards), "rewards/max": max(rewards),
            "batch/n_groups": sum(len(b) for b in batches),
        })
        telemetry.log_training_step(metrics, step=window_index)
        logger.info("miles step: lr=%.2e grad_norm=%.3f rollouts=%d", lr, grad_norm, len(datums))
        return model


__all__ = ["MilesTrainStep"]
