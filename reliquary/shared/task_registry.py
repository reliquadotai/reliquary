"""The one rule for what tasks may exist and what they may cost.

Kept free of I/O so the sum invariant is testable without R2, the same way
``task_id`` keeps the id rule in one dependency-light place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from reliquary.shared.task_id import normalise_task_id

REGISTRY_VERSION = 1
MECHANISM_RL_DISCOVERED_PRICE = "rl-discovered-price"
KNOWN_MECHANISMS = frozenset({MECHANISM_RL_DISCOVERED_PRICE})

# Every field PriceParams needs. A missing one is refused rather than defaulted:
# a half-specified controller is not a controller.
PRICE_PARAM_FIELDS = (
    "start", "decay", "rounds_per_step", "deadband",
    "snap", "floor", "cap", "median_rounds",
)

# Float addition of exact decimals is not exact; 1.0 must not fail by 1e-16.
_SUM_TOLERANCE = 1e-9


class RegistryError(ValueError):
    """The registry, or a change to it, breaks the rule."""


@dataclass(frozen=True, slots=True)
class TaskEntry:
    task_id: str
    profile_id: str
    profile_sha256: str
    mechanism: str
    params: Mapping[str, float]
    status: str
    retired_at: int | None


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{field} must be a number, got {value!r}")
    return float(value)


def validate_entry(entry: TaskEntry) -> None:
    """Everything checkable about one entry without reading the image or R2."""
    try:
        normalise_task_id(entry.task_id)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    if entry.mechanism not in KNOWN_MECHANISMS:
        raise RegistryError(f"unknown incentive mechanism {entry.mechanism!r}")
    if entry.status not in {"active", "retired"}:
        raise RegistryError(f"unknown status {entry.status!r}")
    missing = [f for f in PRICE_PARAM_FIELDS if f not in entry.params]
    if missing:
        raise RegistryError(f"missing price parameters: {', '.join(missing)}")
    cap = _number(entry.params["cap"], "cap")
    if not 0.0 <= cap <= 1.0:
        raise RegistryError(f"cap must be between 0.0 and 1.0, got {cap}")
    floor = _number(entry.params["floor"], "floor")
    if floor > cap:
        raise RegistryError(f"floor {floor} exceeds cap {cap}")


def total_cap(entries: Mapping[str, TaskEntry]) -> float:
    """Every entry counts, retired included: a retired task keeps paying while
    its EMA decays, so its budget is not free yet."""
    return sum(float(e.params["cap"]) for e in entries.values())


def validate_registry(entries: Mapping[str, TaskEntry]) -> None:
    for entry in entries.values():
        validate_entry(entry)
    total = total_cap(entries)
    if total > 1.0 + _SUM_TOLERANCE:
        raise RegistryError(
            f"declared caps total {total:.4f}, above the single available pool "
            f"of 1.0; retire a task or lower a cap first"
        )


def add_task(
    entries: Mapping[str, TaskEntry], entry: TaskEntry
) -> dict[str, TaskEntry]:
    validate_entry(entry)
    if entry.task_id in entries:
        raise RegistryError(f"task {entry.task_id!r} already exists")
    merged = {**entries, entry.task_id: entry}
    validate_registry(merged)
    return merged


def retire_task(
    entries: Mapping[str, TaskEntry], task_id: str, retired_at: int
) -> dict[str, TaskEntry]:
    if task_id not in entries:
        raise RegistryError(f"task {task_id!r} is not in the registry")
    try:
        stamp = int(retired_at)
    except (TypeError, ValueError) as exc:
        raise RegistryError(
            f"retired_at must be an integer round, got {retired_at!r}"
        ) from exc
    retired = replace(entries[task_id], status="retired", retired_at=stamp)
    return {**entries, task_id: retired}


def parse_registry(raw: bytes, *, strict: bool = True) -> dict[str, TaskEntry]:
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise RegistryError(f"registry is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RegistryError("registry must be an object")
    tasks = document.get("tasks", {})
    if not isinstance(tasks, dict):
        raise RegistryError("registry 'tasks' must be an object")
    entries: dict[str, TaskEntry] = {}
    for task_id, body in tasks.items():
        if not isinstance(body, dict):
            raise RegistryError(f"task {task_id!r} is not an object")
        incentive = body.get("incentive")
        if not isinstance(incentive, dict):
            raise RegistryError(f"task {task_id!r} has no incentive block")
        params = incentive.get("params")
        if not isinstance(params, dict):
            raise RegistryError(f"task {task_id!r} has no incentive parameters")
        entries[task_id] = TaskEntry(
            task_id=task_id,
            profile_id=str(body.get("profile_id", "")),
            profile_sha256=str(body.get("profile_sha256", "")),
            mechanism=str(incentive.get("mechanism", "")),
            params=dict(params),
            status=str(body.get("status", "active")),
            retired_at=body.get("retired_at"),
        )
    if strict:
        validate_registry(entries)
    else:
        for entry in entries.values():
            validate_entry(entry)
    return entries


def render_registry(entries: Mapping[str, TaskEntry]) -> bytes:
    """Canonical bytes: sorted keys, so two writers produce the same object."""
    document = {
        "registry_version": REGISTRY_VERSION,
        "tasks": {
            task_id: {
                "profile_id": entry.profile_id,
                "profile_sha256": entry.profile_sha256,
                "incentive": {
                    "mechanism": entry.mechanism,
                    "params": dict(entry.params),
                },
                "status": entry.status,
                "retired_at": entry.retired_at,
            }
            for task_id, entry in sorted(entries.items())
        },
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
