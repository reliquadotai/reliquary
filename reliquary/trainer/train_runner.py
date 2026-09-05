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
        self.env_targets = {
            str(name): int(target) for name, target in env_targets.items()
        }
        if set(self.env_targets) != set(self.env_order):
            raise ValueError("trainer environment targets must match env_order")
        self._train_step = train_step_fn
        self._assess = assess_fn
        self._base_targets = self.env_targets
        self._accumulator = BalancedTrainingAccumulator(self._base_targets)
        self._aggregate_epoch_key: tuple | None = None
        self._aggregate_next_offset = 0
        self._sequential_epoch_key: tuple | None = None
        self._sequential_next_offset = 0
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
                        getattr(group, "prompt_idx", "?"),
                        env,
                    )
            out[env] = kept
        return out

    def _epoch_key(self, decoded: Any) -> tuple:
        binding = decoded.checkpoint_epoch
        return (
            binding.epoch_id,
            binding.manifest_sha256,
            binding.training_run_id,
            binding.training_mode,
            binding.first_window,
            binding.window_count,
            binding.target_groups_per_environment_lane,
            decoded.checkpoint_revision,
        )

    def _validate_epoch_targets(self, binding: Any) -> None:
        if any(
            target != binding.target_groups_per_environment_lane
            for target in self._base_targets.values()
        ):
            raise RuntimeError(
                "checkpoint epoch targets differ from trainer configuration"
            )

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
            # Parity with the in-process path: a failed step is consumed rather
            # than replayed ambiguously against partially-mutated optimizer state.
            logger.exception(
                "train_step failed for window %s; skipping this batch",
                decoded.window_start,
            )
            return False
        finally:
            self._accumulator.reset()
        return True

    def _step_aggregate_epoch(self, decoded: Any) -> bool:
        binding = decoded.checkpoint_epoch
        self._validate_epoch_targets(binding)
        key = self._epoch_key(decoded)
        if self._aggregate_epoch_key is None:
            if binding.lane_offset != 0:
                raise RuntimeError("aggregate checkpoint epoch must begin at lane zero")
            targets = {
                name: target * binding.window_count
                for name, target in self._base_targets.items()
            }
            self._accumulator = BalancedTrainingAccumulator(targets)
            self._aggregate_epoch_key = key
            self._aggregate_next_offset = 0
        if (
            key != self._aggregate_epoch_key
            or binding.lane_offset != self._aggregate_next_offset
        ):
            raise RuntimeError(
                "aggregate checkpoint epoch journal is stale or non-consecutive"
            )

        batches = (
            {}
            if bool(decoded.window_quarantine.get("quarantined"))
            else self._filter_missing_pi_old(decoded.batches())
        )
        self._accumulator.add_window(
            batches,
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        self._aggregate_next_offset += 1
        if not binding.final_lane:
            return False

        try:
            if not self._accumulator.has_groups_for_all_targets:
                logger.warning(
                    "aggregate checkpoint epoch %s has an empty environment; "
                    "discarding without a train step",
                    binding.epoch_id[:12],
                )
                self._accumulator.reset()
                return False
            return self._run_ready_step(decoded, allow_partial=True)
        finally:
            self._aggregate_epoch_key = None
            self._aggregate_next_offset = 0
            self._accumulator = BalancedTrainingAccumulator(self._base_targets)

    def step(self, decoded: Any) -> bool:
        """Feed one journal lane; return True only when an optimizer step ran."""
        payload_targets = dict(getattr(decoded, "env_targets", {}) or {})
        if payload_targets and payload_targets != self._base_targets:
            raise ValueError("training payload environment targets do not match trainer")
        if list(getattr(decoded, "env_order", self.env_order)) != self.env_order:
            raise ValueError("training payload environment order does not match trainer")
        binding = getattr(decoded, "checkpoint_epoch", None)
        if binding is not None and binding.training_mode == "aggregate_one_step":
            return self._step_aggregate_epoch(decoded)
        if self._aggregate_epoch_key is not None:
            raise RuntimeError(
                "aggregate checkpoint epoch was interrupted by another payload"
            )
        if binding is not None:
            return self._step_sequential_epoch(decoded)
        if self._sequential_epoch_key is not None:
            raise RuntimeError(
                "sequential checkpoint epoch was interrupted by another payload"
            )
        self._accumulator.add_window(
            (
                {}
                if bool(decoded.window_quarantine.get("quarantined"))
                else self._filter_missing_pi_old(decoded.batches())
            ),
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        if not self._accumulator.ready:
            return False
        return self._run_ready_step(decoded)

    def _step_sequential_epoch(self, decoded: Any) -> bool:
        binding = decoded.checkpoint_epoch
        self._validate_epoch_targets(binding)
        key = self._epoch_key(decoded)
        if self._sequential_epoch_key is None:
            if binding.lane_offset != 0:
                raise RuntimeError(
                    "sequential checkpoint epoch must begin at lane zero"
                )
            self._sequential_epoch_key = key
            self._sequential_next_offset = 0
        if (
            key != self._sequential_epoch_key
            or binding.lane_offset != self._sequential_next_offset
        ):
            raise RuntimeError(
                "sequential checkpoint epoch journal is stale or non-consecutive"
            )
        self._accumulator.add_window(
            (
                {}
                if bool(decoded.window_quarantine.get("quarantined"))
                else self._filter_missing_pi_old(decoded.batches())
            ),
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        self._sequential_next_offset += 1
        try:
            if not self._accumulator.ready:
                return False
            return self._run_ready_step(decoded)
        finally:
            if binding.final_lane:
                self._accumulator.reset()
                self._sequential_epoch_key = None
                self._sequential_next_offset = 0

    def abort_epoch(self, tombstone: Any) -> None:
        """Discard aggregate state when an explicit epoch tombstone is read."""
        binding = tombstone.get("checkpoint_epoch")
        if binding is None:
            return
        tombstone_key = (
            binding.epoch_id,
            binding.manifest_sha256,
            binding.training_run_id,
            binding.training_mode,
            binding.first_window,
            binding.window_count,
            binding.target_groups_per_environment_lane,
        )
        if self._aggregate_epoch_key is not None:
            if tombstone_key != self._aggregate_epoch_key[:-1]:
                raise RuntimeError("checkpoint epoch tombstone binding differs")
            if binding.lane_offset != self._aggregate_next_offset:
                raise RuntimeError("checkpoint epoch tombstone is non-consecutive")
            self._accumulator.reset()
            self._aggregate_epoch_key = None
            self._aggregate_next_offset = 0
            self._accumulator = BalancedTrainingAccumulator(self._base_targets)
        if self._sequential_epoch_key is not None:
            if tombstone_key != self._sequential_epoch_key[:-1]:
                raise RuntimeError("checkpoint epoch tombstone binding differs")
            if binding.lane_offset != self._sequential_next_offset:
                raise RuntimeError("checkpoint epoch tombstone is non-consecutive")
            self._accumulator.reset()
            self._sequential_epoch_key = None
            self._sequential_next_offset = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "accumulator": self._accumulator.snapshot(),
            "aggregate_epoch_key": self._aggregate_epoch_key,
            "aggregate_next_offset": self._aggregate_next_offset,
            "sequential_epoch_key": self._sequential_epoch_key,
            "sequential_next_offset": self._sequential_next_offset,
        }
