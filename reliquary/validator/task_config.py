"""What this process is allowed to be, according to the registry.

Separate from the store so the refusals are testable without R2, and so the
decision to exit the process stays in the CLI where every other fatal startup
condition already lives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_id import DEFAULT_TASK_ID
from reliquary.shared.task_registry import (
    PRICE_PARAM_FIELDS,
    RegistryError,
    TaskEntry,
    validate_registry,
)
from reliquary.validator.emission_price import PriceParams


class TaskConfigError(RuntimeError):
    """The registry does not describe a task this binary may run."""


@dataclass(frozen=True, slots=True)
class TaskConfig:
    task_id: str
    entry: TaskEntry | None
    price_params: PriceParams
    emission_cap: float


def resolve_task_config(
    entries: Mapping[str, TaskEntry],
    task_id: str,
    *,
    profile_id: str,
    generation_contract: Any,
) -> TaskConfig:
    """This task's settings, or a refusal naming exactly what disagrees."""
    try:
        validate_registry(entries)
    except RegistryError as exc:
        raise TaskConfigError(f"task registry is unusable: {exc}") from exc

    entry = entries.get(task_id)
    if entry is None:
        declared = ", ".join(sorted(entries)) or "none"
        raise TaskConfigError(
            f"task {task_id!r} is not declared in the registry (declared: {declared})"
        )
    if entry.status != "active":
        raise TaskConfigError(f"task {task_id!r} is {entry.status}, not active")
    if entry.profile_id != profile_id:
        raise TaskConfigError(
            f"task {task_id!r} declares profile {entry.profile_id!r} but this "
            f"process runs {profile_id!r}"
        )
    digest = canonical_sha256(generation_contract)
    if entry.profile_sha256 != digest:
        raise TaskConfigError(
            f"task {task_id!r} pins profile contract {entry.profile_sha256[:12]}… "
            f"but this build computes {digest[:12]}…"
        )

    params = PriceParams(**{f: entry.params[f] for f in PRICE_PARAM_FIELDS})
    return TaskConfig(
        task_id=task_id,
        entry=entry,
        price_params=params,
        emission_cap=float(entry.params["cap"]),
    )


def legacy_task_config() -> TaskConfig:
    """The pre-registry behaviour of the single task that predates it.

    Only for a wholly absent registry: a present-but-wrong one still refuses.
    """
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    return TaskConfig(
        task_id=DEFAULT_TASK_ID,
        entry=None,
        price_params=PRODUCTION_PRICE_PARAMS,
        emission_cap=1.0,
    )
