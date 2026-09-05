"""Resume-point resolution for the detached trainer.

The candidate manifest is a hint for WHICH checkpoint to load; the
checkpoint PROFILE inside the snapshot is authoritative for the cursor
and LR position once downloaded. First-run bootstrap requires an explicit
cursor — the trainer refuses to guess where the journal starts.
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping

from reliquary.shared.checkpoint_identity import (
    canonical_checkpoint_identity,
    require_checkpoint_number,
    require_immutable_checkpoint_revision,
)
from reliquary.shared.strict_json import strict_json_loads
from reliquary.trainer.publisher import CANDIDATE_MANIFEST_KEY

logger = logging.getLogger(__name__)


def _environment_string(
    env: Mapping[str, str],
    key: str,
) -> str:
    value = env.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be configured as a string")
    return value.strip()


def _environment_nonnegative_int(
    env: Mapping[str, str],
    key: str,
) -> int:
    raw = _environment_string(env, key)
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise ValueError(f"{key} must contain a base-10 integer") from exc
    return require_checkpoint_number(value, field=key)


def resolve_resume_point(
    fetch_fn: Callable[[str], bytes | None],
    *,
    env: Mapping[str, str],
    expected_identity: Mapping[str, object] | None = None,
) -> tuple[str | None, int, int]:
    """Return ``(revision, cursor, checkpoint_n)``: the checkpoint
    revision to load (None = bootstrap), the journal cursor to resume
    after, and the last published checkpoint number (0 = none yet —
    checkpoint numbering must never regress across restarts)."""
    raw = fetch_fn(CANDIDATE_MANIFEST_KEY)
    if raw is not None:
        manifest = strict_json_loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("trainer resume manifest must be a JSON object")
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in (expected_identity or {}).items()
            if manifest.get(key) != expected
        }
        if not mismatches:
            checkpoint_n, _, revision = canonical_checkpoint_identity(
                manifest.get("checkpoint_n"),
                manifest.get("repo_id"),
                manifest.get("revision"),
                field="trainer resume manifest checkpoint",
            )
            return (
                revision,
                require_checkpoint_number(
                    manifest.get("trained_window_cursor"),
                    field="trainer resume manifest cursor",
                ),
                checkpoint_n,
            )
        logger.warning(
            "candidate manifest belongs to another protocol/run (%s); "
            "using the explicit bootstrap configuration",
            ", ".join(sorted(mismatches)),
        )
    bootstrap = _environment_string(
        env,
        "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR",
    )
    if not bootstrap:
        logger.critical(
            "no candidate manifest in R2 and no "
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR set — refusing to guess "
            "the journal start"
        )
        raise SystemExit(2)
    # Mid-run bootstrap (shadow start, cutover from in-process training):
    # begin from the validator's last PUBLISHED checkpoint, not the base
    # model, so the shadow comparison and the cutover are seamless.
    raw_revision = _environment_string(
        env,
        "RELIQUARY_TRAINER_BOOTSTRAP_REVISION",
    )
    revision = (
        require_immutable_checkpoint_revision(
            raw_revision,
            field="trainer bootstrap revision",
        )
        if raw_revision
        else None
    )
    raw_n = _environment_string(env, "RELIQUARY_TRAINER_CHECKPOINT_N")
    checkpoint_n = (
        _environment_nonnegative_int(env, "RELIQUARY_TRAINER_CHECKPOINT_N")
        if raw_n
        else 0
    )
    return (
        revision,
        _environment_nonnegative_int(
            env,
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR",
        ),
        checkpoint_n,
    )
