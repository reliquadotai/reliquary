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


def parse_task_ids(value: str | None) -> tuple[str, ...]:
    """A comma-separated list of task ids, in order; one id is the usual case.

    Several ids are only meaningful to a corpus validator serving several jobs
    on one loaded model; a duplicate or empty member is a typo, not a request.
    """
    if value is None or "," not in value:
        return (normalise_task_id(value),)
    ids = tuple(member.strip() for member in value.split(","))
    if any(not member for member in ids):
        raise ValueError(f"task id list {value!r} has an empty member")
    if len(set(ids)) != len(ids):
        raise ValueError(f"task id list {value!r} names a task twice")
    return tuple(normalise_task_id(member) for member in ids)
