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
_held: dict | None = None  # rows collected between hold() and flush() (one row per async trainer step)
_private_mode = False
_private_writer: Any = None


def is_active() -> bool:
    return bool(_private_writer and _private_writer.active) if _private_mode else _enabled


def init(hotkey_ss58: str, config: dict, *, role: str | None = None, version: str | None = None) -> None:
    """Initialise wandb if ``WANDB_API_KEY`` is set. No-op otherwise. ``role`` (async RL: ``trainer`` /
    ``controller``) gives each process its own run, ``{hotkey[:8]}-{version}-{role}``, grouped by
    ``{hotkey[:8]}-{version}``; ``version`` (the async run name) stands in for ``WANDB_TRAINING_VERSION``.

    Fail-soft: any exception (ImportError, network, auth) disables
    telemetry for the rest of the session and logs a warning.
    """
    global _run, _enabled, _private_mode, _private_writer, _held
    was_private = _private_mode
    if _private_writer is not None:
        _private_writer.close(timeout=0)
    _private_writer = None
    _private_mode = False
    mode = os.getenv("RELIQUARY_TRAINING_WANDB", "off")
    if was_private or mode not in ("", "off"):
        _held = None  # Never carry a previous identity's unfinished update into a new scope.
    if mode not in ("", "off"):
        # A malformed private opt-in must never fall back to the shared hotkey run.
        _private_mode = True
        _enabled = False
        try:
            from reliquary.shared.private_training_wandb import private_wandb_from_environment
            from reliquary.validator.private_training_telemetry import PrivateTrainingWriter

            identity = private_wandb_from_environment(config, role, os.environ)
            if not os.getenv("WANDB_API_KEY"):
                raise ValueError("private telemetry credential not configured")
            _private_writer = PrivateTrainingWriter(identity)
        except Exception:
            logger.warning("wandb: private training telemetry not configured; shared fallback disabled")
        return
    if not os.getenv("WANDB_API_KEY"):
        logger.info("wandb: WANDB_API_KEY not set, telemetry disabled")
        return

    version = os.getenv("RELIQUARY_WANDB_VERSION") or version or WANDB_TRAINING_VERSION
    project = os.getenv("WANDB_PROJECT", WANDB_PROJECT)
    entity = os.getenv("WANDB_ENTITY")
    group = f"{hotkey_ss58[:8]}-{version}"
    run_id = group if role is None else f"{group}-{role}"

    try:
        import wandb  # lazy: optional dependency
        _run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            group=None if role is None else group,
            job_type=role,
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
    if _held is not None:
        _held.update(metrics)
        return
    if _private_mode:
        return  # Only a captured, successfully published checkpoint can enter the private run.
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


def hold() -> None:
    """Collect the next ``log_training_step`` rows instead of sending them, until ``flush``."""
    global _held
    if _held is None:           # a skipped step's rows stay held for the next one
        _held = {}


def flush(extra: dict, step: int | None) -> dict:
    """Send what ``hold`` collected and ``extra`` as one row at ``step``."""
    global _held
    held, _held = _held or {}, None
    row = {**held, **extra}
    if not _private_mode:
        log_training_step(row, step)
    return row


def prepare_published_checkpoint(metrics: dict, checkpoint: int, observed_at: float):
    """Capture an independent row now; returned nonblocking sink runs only after publication."""
    if not _private_mode or _private_writer is None:
        return None
    try:
        return _private_writer.prepare(metrics, checkpoint, observed_at)
    except Exception:
        logger.warning("wandb: private checkpoint telemetry capture failed; training continues")
        return None


def update_config(values: dict) -> None:
    """Record runtime-resolved configuration after startup discovery.

    Some values, notably a Hugging Face snapshot's resolved commit and device,
    only exist after ``wandb.init``. Keep this fail-soft like the metric path.
    """
    global _log_warned
    if _private_mode:
        return  # Private run identity is immutable and never updated from arbitrary config fields.
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
    global _run, _enabled, _log_warned, _private_mode, _private_writer, _held
    if _private_mode:
        if _private_writer is not None:
            _private_writer.close()
        _private_writer = None
        _private_mode = False
        _enabled = False
        _held = None
        return
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
    global _run, _enabled, _log_warned, _held, _private_mode, _private_writer
    if _private_writer is not None:
        _private_writer.close()
    _run = None
    _enabled = False
    _log_warned = False
    _held = None
    _private_mode = False
    _private_writer = None
