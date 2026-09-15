"""The one rule for what a task id may be, shared by constants and storage."""

from __future__ import annotations

import re

DEFAULT_TASK_ID = "default"
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def normalise_task_id(value: str | None) -> str:
    """Unset or empty means the legacy task; anything else must be a usable slug."""
    resolved = (value or "").strip() or DEFAULT_TASK_ID
    if not TASK_ID_RE.match(resolved):
        raise ValueError(f"unusable task id {resolved!r}")
    return resolved
