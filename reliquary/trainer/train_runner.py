"""Accumulate decoded windows and run train_step — the detached
counterpart of _train_and_publish's training block. Window-level
quarantine arrives precomputed in the payload; only the accumulated-batch
quarantine runs here."""

from __future__ import annotations

from reliquary.shared.decision_telemetry import capture as decision_capture, group_ref as decision_group, observe as decision_observe, annotate_origins

import logging
from typing import Any, Callable

from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.training import (
    TrainingStepSkipped,
    train_step as _default_train_step,
)
from reliquary.validator.training_accumulator import (
    BalancedTrainingAccumulator,
)

logger = logging.getLogger(__name__)


class TrainerStateUnsafe(RuntimeError):
    """An optimizer call failed; its in-memory mutations cannot be retried."""


class TrainRunner:
    def __init__(
        self,
        model: Any,
        *,
        env_targets: dict[str, int],
        env_order: list[str],
        ref_model: Any = None,
        train_step_fn: Callable = _default_train_step,
        assess_fn: Callable = assess_training_batch,
        global_step_hint: int | None = None,
    ) -> None:
        from reliquary.constants import KL_BETA

        if ref_model is None and float(KL_BETA) != 0.0:
            raise RuntimeError(
                "TrainRunner without ref_model requires KL_BETA == 0.0; "
                "pin RELIQUARY_KL_BASE_MODEL and pass it as ref_model"
            )
        self.model = model
        self.ref_model = ref_model
        self.env_order = list(env_order)
        self.env_targets = {
            str(name): int(target) for name, target in env_targets.items()
        }
        if set(self.env_targets) != set(self.env_order):
            raise ValueError("trainer environment targets must match env_order")
        self._train_step = train_step_fn
        self._assess = assess_fn
        self._base_targets = self.env_targets
        self._accumulator = BalancedTrainingAccumulator(self._base_targets)
        # Restored LR position; passed on every call like the validator's
        # _lr_global_step_hint (the schedule advances internally after
        # _lazy_init consumes it once).
        self.global_step_hint = global_step_hint
        self.groups_dropped_missing_pi_old = 0
        self._last_decoded = None

    def _filter_missing_pi_old(self, batches: dict) -> dict:
        """Drop whole groups lacking validator pi_old on any rollout.

        The in-process ladder falls back to a behavior-model forward; the
        detached trainer has no frozen replica, and the next rung down is
        the MINER-CLAIMED logprobs — exactly the trust RECOMPUTE was
        deployed to remove. Dropping the full group (not the rollout)
        keeps group-relative advantages intact for everything trained.
        """
        from reliquary.constants import (
            PI_OLD_FROM_VERIFY_LOGPROBS,
            RECOMPUTE_PI_OLD_FROM_VERIFY,
            T_PROTO,
        )

        if float(T_PROTO) != 1.0:
            # Shipped pi_old cannot exist off the identity-warp profile;
            # the gate mirrors the encoder's.
            return batches
        if not (PI_OLD_FROM_VERIFY_LOGPROBS and RECOMPUTE_PI_OLD_FROM_VERIFY):
            return batches
        out: dict = {}
        for env, groups in batches.items():
            kept = []
            for group in groups:
                if all(
                    getattr(r, "_validated_completion_logprobs", None)
                    is not None
                    for r in group.rollouts
                ):
                    kept.append(group)
                else:
                    self.groups_dropped_missing_pi_old += 1
                    decision_capture("trainer_group_filtered", lambda: dict(reason="missing_pi_old", group=decision_group(group, environment=env)))
                    logger.warning(
                        "dropping group prompt_idx=%s (%s): missing "
                        "validator pi_old; refusing miner-claim fallback",
                        getattr(group, "prompt_idx", "?"),
                        env,
                    )
            out[env] = kept
        return out



    def _run_ready_step(
        self,
        decoded: Any,
        *,
        allow_partial: bool = False,
    ) -> bool:
        batches = self._accumulator.training_batches(
            self.env_order,
            allow_partial=allow_partial,
        )
        verdict = self._assess(
            [group for batch in batches for group in batch],
            reject_counts={},
        )
        decision_capture("trainer_batch_assessed", lambda: dict(
            window=decoded.window_start, quarantined=bool(verdict.quarantined),
            groups=[decision_group(g) for batch in batches for g in batch]))
        if verdict.quarantined:
            logger.warning(
                "accumulated batch quarantined: %s",
                getattr(verdict, "reasons", None),
            )
            self._accumulator.reset()
            return False
        try:
            self.model = self._train_step(
                self.model,
                batches,
                ref_model=self.ref_model,
                window_index=decoded.window_start,
                global_step_hint=self.global_step_hint,
            )
        except TrainingStepSkipped:
            raise  # worker handles health gates (adaptive publication)
        except Exception as exc:
            raise TrainerStateUnsafe(
                f"optimizer failed at window {decoded.window_start}; "
                "reload the last published checkpoint before resuming"
            ) from exc
        finally:
            decision_capture("trainer_accumulator_released", lambda: dict(
                window=decoded.window_start,
                groups=[decision_group(g) for batch in batches for g in batch]))
            self._accumulator.reset()
        return True


    @decision_observe("trainer_payload")
    def step(self, decoded: Any) -> bool:
        """Feed one journal lane; return True only when an optimizer step ran."""
        payload_targets = dict(getattr(decoded, "env_targets", {}) or {})
        if payload_targets and payload_targets != self._base_targets:
            raise ValueError("training payload environment targets do not match trainer")
        if list(getattr(decoded, "env_order", self.env_order)) != self.env_order:
            raise ValueError("training payload environment order does not match trainer")
        self._accumulator.add_window(
            (
                {}
                if bool(decoded.window_quarantine.get("quarantined"))
                else self._filter_missing_pi_old(annotate_origins(decoded.batches(), decoded))
            ),
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        self._last_decoded = decoded
        if not self._accumulator.ready:
            return False
        return self._run_ready_step(decoded)

    def finish(self) -> bool:
        """Flush a balanced final partial step; never silently discard a tail."""
        if not any(self._accumulator.snapshot()["counts"].values()):
            return False
        if not self._accumulator.has_groups_for_all_targets:
            raise RuntimeError("drain blocked: incomplete environment mix in accumulator")
        return self._run_ready_step(self._last_decoded, allow_partial=True)



    def snapshot(self) -> dict[str, Any]:
        return {"accumulator": self._accumulator.snapshot()}
