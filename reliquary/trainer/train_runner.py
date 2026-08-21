"""Accumulate decoded windows and run train_step — the detached
counterpart of _train_and_publish's training block. Window-level
quarantine arrives precomputed in the payload; only the accumulated-batch
quarantine runs here."""

from __future__ import annotations

import logging
from typing import Any, Callable

from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.training import train_step as _default_train_step
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

    def step(self, decoded: Any) -> bool:
        """Feed one decoded window; returns True when a train step ran.

        TrainingStepSkipped propagates to the caller (the worker turns
        policy_ratio_drift into an adaptive publication) — the
        accumulator is reset first, matching the validator's finally.
        """
        self._accumulator.add_window(
            decoded.batches(),
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
        finally:
            self._accumulator.reset()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {"accumulator": self._accumulator.snapshot()}
