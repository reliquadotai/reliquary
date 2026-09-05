"""Canonical validation for publicly addressable checkpoints."""

from __future__ import annotations

import re


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def require_checkpoint_number(
    value: object,
    *,
    field: str = "checkpoint number",
) -> int:
    """Return a canonical non-negative checkpoint number."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


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


def canonical_checkpoint_identity(
    number: object,
    repository: object,
    revision: object,
    *,
    field: str = "checkpoint",
) -> tuple[int, str, str]:
    """Validate and return ``(number, repository, revision)``."""

    return (
        require_checkpoint_number(number, field=f"{field} number"),
        require_checkpoint_repository(repository, field=f"{field} repository"),
        require_immutable_checkpoint_revision(revision, field=f"{field} revision"),
    )


def require_checkpoint_successor(
    current: tuple[int, str, str] | None,
    candidate: tuple[int, str, str],
    *,
    field: str = "checkpoint",
) -> bool:
    """Reject rollback/rebinding; return whether candidate is idempotent."""

    if current is None:
        return False
    if candidate[0] < current[0]:
        raise ValueError(f"{field} cannot roll back")
    if candidate[0] == current[0] and candidate != current:
        raise ValueError(f"{field} cannot rebind an existing checkpoint number")
    return candidate == current
