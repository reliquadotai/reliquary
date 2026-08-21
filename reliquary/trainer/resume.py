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
) -> tuple[str | None, int]:
    """Return ``(revision, cursor)``: the checkpoint revision to load
    (None = bootstrap from the pinned base model) and the journal cursor
    to resume after."""
    raw = fetch_fn(CANDIDATE_MANIFEST_KEY)
    if raw is not None:
        manifest = json.loads(raw.decode("utf-8"))
        return (
            str(manifest["revision"]),
            int(manifest["trained_window_cursor"]),
        )
    bootstrap = str(env.get("RELIQUARY_TRAINER_BOOTSTRAP_CURSOR", "")).strip()
    if not bootstrap:
        logger.critical(
            "no candidate manifest in R2 and no "
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR set — refusing to guess "
            "the journal start"
        )
        raise SystemExit(2)
    return None, int(bootstrap)
