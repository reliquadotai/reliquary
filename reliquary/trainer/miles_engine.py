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
        # episodes in their own chunks: their last assistant EOS gets no final-EOS test (DrawGate.admit)
        order, futures = [], []
        for episodes in (False, True):
            picked = [i for i, (_, s) in enumerate(slots) if ("episode" in by_slot[s].commit["rollout"]) == episodes]
            config = {**self.config, "terminal_check": 0} if episodes else self.config
            for start in range(0, len(picked), self.chunk):
                part = picked[start:start + self.chunk]
                order += part
                futures.append(self.client.forward_backward([datums[i] for i in part], "reliquary_gate", config))
        outputs = [out for future in futures for out in future.result().loss_fn_outputs]
        fill_gate(gate, [slots[i] for i in order], outputs, [by_slot[slots[i][1]] for i in order])
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



_DTYPES = {"bfloat16": "BF16", "float32": "F32", "float16": "F16"}


def _torch_dtypes() -> dict:
    import torch

    return {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def _safetensors_header(path) -> dict:
    import json
    import struct

    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(size))


def expected_weight_specs(model_dir) -> dict[str, tuple[str, list[int]]]:
    """The HF model's weights as ``{name: (dtype, shape)}``, read from its safetensors headers (what miners load)."""
    from pathlib import Path

    specs = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        for name, info in _safetensors_header(shard).items():
            if name != "__metadata__":
                specs[name] = (info["dtype"], list(info["shape"]))
    if not specs:
        raise RuntimeError(f"{model_dir} holds no safetensors weights")
    return specs


class MilesState:
    """``AsyncTrainer``'s ``state_fn``: the gateway exports the weights (``save_weights_for_sampler``, HF safetensors
    under ``<checkpoint_root>/**/<model_id>/sampler_weights/<name>/hf`` on this box's disk), loaded on CPU. The names
    and shapes must be exactly the HF model's (what miners load), or nothing is published; each tensor takes the
    model's dtype (Qwen3.5 keeps its linear-attention ``A_log`` and norms in fp32, Megatron exports them in bf16).
    A head Miles does not train (``mtp.*``, Qwen3.5's multi-token prediction, unused by generation) is taken from
    the base model (``base_dir``). Exports older than the previous one are deleted through ``delete(tinker_path)``."""

    KEEP = 2

    UNTRAINED_PREFIXES = ("mtp.",)

    def __init__(self, client, *, checkpoint_root, expected: dict[str, tuple[str, list[int]]], delete,
                 base_dir=None) -> None:
        from pathlib import Path

        self.client, self.root = client, Path(checkpoint_root)
        self.expected, self.delete = dict(expected), delete
        self.base_dir = None if base_dir is None else Path(base_dir)
        self.n, self.paths = 0, []
        self._untrained: dict | None = None

    def _base_untrained(self) -> dict:
        """The base model's weights under ``UNTRAINED_PREFIXES``, read once."""
        from safetensors import safe_open

        if self._untrained is None:
            self._untrained = {}
            for shard in sorted(self.base_dir.glob("*.safetensors")) if self.base_dir else ():
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        if key.startswith(self.UNTRAINED_PREFIXES):
                            self._untrained[key] = handle.get_tensor(key)
        return self._untrained

    def __call__(self) -> dict:
        from safetensors.torch import load_file

        name = f"snapshot-{self.n}"
        self.n += 1
        path = self.client.save_weights_for_sampler(name).result().path
        model_id, kind, _ = path.removeprefix("tinker://").split("/", 2)
        # the launcher nests the gateway's checkpoint root under its output dir (``checkpoints/<run id>``)
        found = sorted(self.root.glob(f"**/{model_id}/{kind}/{name}"))
        if len(found) != 1:
            raise RuntimeError(f"gateway export {path}: {len(found)} matches under {self.root}")
        hf = found[0].resolve() / "hf"
        state: dict = {}
        for shard in sorted(hf.glob("*.safetensors")):
            state.update(load_file(str(shard), device="cpu"))
        for key in set(self.expected) - set(state):
            if key.startswith(self.UNTRAINED_PREFIXES) and key in self._base_untrained():
                state[key] = self._base_untrained()[key]
        for key, tensor in state.items():
            want = _torch_dtypes().get(self.expected.get(key, ("",))[0])
            if want is not None and tensor.dtype != want:
                state[key] = tensor.to(want)
        got = {k: (_DTYPES.get(str(t.dtype).removeprefix("torch."), str(t.dtype)), list(t.shape))
               for k, t in state.items()}
        missing, extra = set(self.expected) - set(got), set(got) - set(self.expected)
        changed = sorted(k for k in set(got) & set(self.expected) if got[k] != tuple(self.expected[k]))
        if missing or extra or changed:
            raise RuntimeError(f"gateway export {path} does not match the model: missing {sorted(missing)[:5]}, "
                               f"unexpected {sorted(extra)[:5]}, dtype/shape changed "
                               f"{[(k, got[k], self.expected[k]) for k in changed[:5]]}")
        self.paths.append(path)
        while len(self.paths) > self.KEEP:
            self.delete(self.paths.pop(0))
        return state


__all__ = ["MilesState", "MilesTrainStep", "expected_weight_specs"]
