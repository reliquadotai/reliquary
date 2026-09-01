"""Canonical validation for publicly addressable checkpoints."""

from __future__ import annotations

import re


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def require_checkpoint_repository(
    value: object,
    *,
    field: str = "checkpoint repository",
) -> str:
    """Return a non-empty canonical repository identifier."""

    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty canonical string")
    return value


def is_immutable_checkpoint_revision(value: object) -> bool:
    """Return whether *value* is a full lowercase Git/HF commit OID."""

    return isinstance(value, str) and _IMMUTABLE_REVISION.fullmatch(value) is not None


def require_immutable_checkpoint_revision(
    value: object,
    *,
    field: str = "checkpoint revision",
) -> str:
    """Return a validated immutable revision or raise ``ValueError``."""

    if not is_immutable_checkpoint_revision(value):
        raise ValueError(f"{field} must be a lowercase 40-character commit OID")
    return value
