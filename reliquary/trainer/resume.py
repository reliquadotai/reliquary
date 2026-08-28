"""Resume-point resolution for the detached trainer.

The candidate manifest is a hint for WHICH checkpoint to load; the
checkpoint PROFILE inside the snapshot is authoritative for the cursor
and LR position once downloaded. First-run bootstrap requires an explicit
cursor — the trainer refuses to guess where the journal starts.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Mapping

from reliquary.trainer.publisher import CANDIDATE_MANIFEST_KEY

logger = logging.getLogger(__name__)


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
        manifest = json.loads(raw.decode("utf-8"))
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in (expected_identity or {}).items()
            if manifest.get(key) != expected
        }
        if not mismatches:
            return (
                str(manifest["revision"]),
                int(manifest["trained_window_cursor"]),
                int(manifest.get("checkpoint_n", 0)),
            )
        logger.warning(
            "candidate manifest belongs to another protocol/run (%s); "
            "using the explicit bootstrap configuration",
            ", ".join(sorted(mismatches)),
        )
    bootstrap = str(env.get("RELIQUARY_TRAINER_BOOTSTRAP_CURSOR", "")).strip()
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
    revision = str(
        env.get("RELIQUARY_TRAINER_BOOTSTRAP_REVISION", "")
    ).strip() or None
    raw_n = str(env.get("RELIQUARY_TRAINER_CHECKPOINT_N", "")).strip()
    return revision, int(bootstrap), int(raw_n) if raw_n else 0
