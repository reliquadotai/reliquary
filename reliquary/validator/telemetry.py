"""Validator-side wandb telemetry.

Opt-in: requires ``WANDB_API_KEY`` env var. Fails soft on any exception
and on missing ``wandb`` package. Never propagates to the validator main
loop. Run id is deterministic (``{hotkey[:8]}-{version}``) so restarts
resume the same wandb run; bump ``WANDB_TRAINING_VERSION`` (or set
``RELIQUARY_WANDB_VERSION``) to start a new run.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from reliquary.constants import WANDB_PROJECT, WANDB_TRAINING_VERSION

logger = logging.getLogger(__name__)

_run: Any = None
_enabled: bool = False
_log_warned: bool = False  # suppress repeated log-path warnings


def is_active() -> bool:
    return _enabled


def init(hotkey_ss58: str, config: dict) -> None:
    """Initialise wandb if ``WANDB_API_KEY`` is set. No-op otherwise.

    Fail-soft: any exception (ImportError, network, auth) disables
    telemetry for the rest of the session and logs a warning.
    """
    global _run, _enabled
    if not os.getenv("WANDB_API_KEY"):
        logger.info("wandb: WANDB_API_KEY not set, telemetry disabled")
        return

    version = os.getenv("RELIQUARY_WANDB_VERSION", WANDB_TRAINING_VERSION)
    project = os.getenv("WANDB_PROJECT", WANDB_PROJECT)
    entity = os.getenv("WANDB_ENTITY")
    run_id = f"{hotkey_ss58[:8]}-{version}"

    try:
        import wandb  # lazy: optional dependency
        _run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume="allow",
            config=config,
        )
        _enabled = True
        logger.info(
            "wandb: initialised (project=%s id=%s entity=%s)",
            project, run_id, entity or "<default>",
        )
    except Exception as e:  # noqa: BLE001 — fail-soft by design
        logger.warning("wandb: init failed (%s), telemetry disabled", e)
        _enabled = False


def log_training_step(metrics: dict, step: int | None) -> None:
    """Forward a metrics dict to wandb.log. No-op if disabled. Fail-soft.

    Only the first exception is logged — subsequent failures in the same
    session are silenced to avoid flooding the log on prolonged outages.
    """
    global _log_warned
    if not _enabled:
        return
    try:
        import wandb  # already imported successfully in init()
        # ``step=`` disables W&B's implicit commit in some client versions.
        # Commit every optimizer row so the final step remains queryable when
        # a checkpoint ceiling or emergency freeze prevents a later log call.
        wandb.log(metrics, step=step, commit=True)
    except Exception as e:  # noqa: BLE001
        if not _log_warned:
            logger.warning("wandb: log failed (%s), suppressing further warnings", e)
            _log_warned = True


def update_config(values: dict) -> None:
    """Record runtime-resolved configuration after startup discovery.

    Some values, notably a Hugging Face snapshot's resolved commit and device,
    only exist after ``wandb.init``. Keep this fail-soft like the metric path.
    """
    global _log_warned
    if not _enabled or _run is None:
        return
    try:
        _run.config.update(values, allow_val_change=True)
    except Exception as e:  # noqa: BLE001
        if not _log_warned:
            logger.warning(
                "wandb: config update failed (%s), suppressing further warnings",
                e,
            )
            _log_warned = True


def finish() -> None:
    """Close the wandb run. No-op if disabled. Fail-soft."""
    global _run, _enabled, _log_warned
    if not _enabled:
        return
    try:
        import wandb
        wandb.finish()
    except Exception as e:  # noqa: BLE001
        logger.warning("wandb: finish failed (%s)", e)
    finally:
        _run = None
        _enabled = False
        _log_warned = False


def _reset_for_tests() -> None:
    """Test-only helper: reset module state so each test starts fresh."""
    global _run, _enabled, _log_warned
    _run = None
    _enabled = False
    _log_warned = False
