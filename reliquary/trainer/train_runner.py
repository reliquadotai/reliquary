"""Accumulate decoded windows and run train_step — the detached
counterpart of _train_and_publish's training block. Window-level
quarantine arrives precomputed in the payload; only the accumulated-batch
quarantine runs here."""

from __future__ import annotations

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
        self._train_step = train_step_fn
        self._assess = assess_fn
        self._accumulator = BalancedTrainingAccumulator(env_targets)
        # Restored LR position; passed on every call like the validator's
        # _lr_global_step_hint (the schedule advances internally after
        # _lazy_init consumes it once).
        self.global_step_hint = global_step_hint
        self.groups_dropped_missing_pi_old = 0

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
                    logger.warning(
                        "dropping group prompt_idx=%s (%s): missing "
                        "validator pi_old; refusing miner-claim fallback",
                        getattr(group, "prompt_idx", "?"), env,
                    )
            out[env] = kept
        return out

    def step(self, decoded: Any) -> bool:
        """Feed one decoded window; returns True when a train step ran.

        TrainingStepSkipped propagates to the caller (the worker turns
        policy_ratio_drift into an adaptive publication) — the
        accumulator is reset first, matching the validator's finally.
        """
        self._accumulator.add_window(
            self._filter_missing_pi_old(decoded.batches()),
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        if not self._accumulator.ready:
            return False
        batches = self._accumulator.training_batches(self.env_order)
        verdict = self._assess(
            [g for batch in batches for g in batch], reject_counts={},
        )
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
        except Exception:
            # Parity with the in-process path (service.py: "train_step
            # failed; archiving anyway"): a CUDA OOM or kernel fault on
            # one window must not become a crash loop replaying it.
            logger.exception(
                "train_step failed for window %s; skipping this batch",
                decoded.window_start,
            )
            return False
        finally:
            self._accumulator.reset()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {"accumulator": self._accumulator.snapshot()}
