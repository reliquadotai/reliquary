"""The one rule for what tasks may exist and what they may cost.

Kept free of I/O so the sum invariant is testable without R2, the same way
``task_id`` keeps the id rule in one dependency-light place.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from reliquary.shared.task_id import DEFAULT_TASK_ID, normalise_task_id

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
    """The one guarded coercion every numeric field goes through.

    Non-finite values are refused here rather than by the range checks below:
    every comparison against NaN is False, so ``floor > cap`` would wave a NaN
    floor straight through. Infinity is refused for the same reason and
    because ``json.dumps`` would then write a literal only Python can read.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{field} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RegistryError(f"{field} is too large to be a weight: {value!r}") from exc
    if not math.isfinite(number):
        raise RegistryError(f"{field} must be a finite number, got {value!r}")
    return number


def validate_entry(entry: TaskEntry) -> None:
    """Everything checkable about one entry without reading the image or R2."""
    try:
        canonical = normalise_task_id(entry.task_id)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    # `parse_registry` builds every entry with task_id=<the file's key>, so
    # this is also the check on the key itself. Refuse rather than rewrite:
    # " default" and "" both normalise to "default" while staying keyed under
    # the raw string, producing a registry that validates but that
    # `resolve_task_config` can never look up.
    if canonical != entry.task_id:
        raise RegistryError(
            f"task id {entry.task_id!r} is not canonical; write it as {canonical!r}"
        )
    if entry.mechanism not in KNOWN_MECHANISMS:
        raise RegistryError(f"unknown incentive mechanism {entry.mechanism!r}")
    if entry.status not in {"active", "retired"}:
        raise RegistryError(f"unknown status {entry.status!r}")
    missing = [f for f in PRICE_PARAM_FIELDS if f not in entry.params]
    if missing:
        raise RegistryError(f"missing price parameters: {', '.join(missing)}")
    # Every controller parameter must be a number here: a string that survives
    # to live arithmetic fails a window instead of refusing a start.
    for field in PRICE_PARAM_FIELDS:
        _number(entry.params[field], field)
    for field in ("rounds_per_step", "median_rounds"):
        value = entry.params[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RegistryError(
                f"{field} must be a whole number of rounds, got {value!r}"
            )
    if entry.retired_at is not None and (
        isinstance(entry.retired_at, bool) or not isinstance(entry.retired_at, int)
    ):
        raise RegistryError(
            f"retired_at must be an integer round or null, got {entry.retired_at!r}"
        )
    cap = _number(entry.params["cap"], "cap")
    if not 0.0 <= cap <= 1.0:
        raise RegistryError(f"cap must be between 0.0 and 1.0, got {cap}")
    floor = _number(entry.params["floor"], "floor")
    if floor > cap:
        raise RegistryError(f"floor {floor} exceeds cap {cap}")


def total_cap(entries: Mapping[str, TaskEntry]) -> float:
    """Every entry counts, retired included: a retired task keeps paying while
    its EMA decays, so its budget is not free yet."""
    return sum(_number(e.params.get("cap"), "cap") for e in entries.values())


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


def require_default_declared_first(
    entries: Mapping[str, TaskEntry], entry: TaskEntry
) -> None:
    """Refuse to make ``default`` the task nobody declared.

    Every validator running today is the legacy ``default`` task, and both
    legacy fallbacks are armed by an EMPTY registry (see
    ``reliquary.validator.task_config.legacy_registry_fallback``). Writing any
    other task first therefore un-arms them fleet-wide: trainers exit 4 on
    their next restart and submitters abstain, with no registry entry for the
    task actually running. Deliberately NOT part of ``add_task``: this is the
    bootstrap ordering of a live subnet, not an invariant of the object.
    """
    if DEFAULT_TASK_ID in entries or entry.task_id == DEFAULT_TASK_ID:
        return
    raise RegistryError(
        f"refusing to declare {entry.task_id!r} while {DEFAULT_TASK_ID!r} is "
        f"absent from the registry: that would stop every validator running "
        f"today. Declare {DEFAULT_TASK_ID!r} first, then add this task."
    )


def retire_task(
    entries: Mapping[str, TaskEntry], task_id: str, retired_at: int
) -> dict[str, TaskEntry]:
    if task_id not in entries:
        raise RegistryError(f"task {task_id!r} is not in the registry")
    try:
        stamp = int(retired_at)
    except (TypeError, ValueError, OverflowError) as exc:
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
    # allow_nan=False: the Python default emits bare NaN/Infinity literals that
    # no other JSON reader accepts, so an unreadable object would reach R2.
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
