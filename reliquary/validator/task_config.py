"""What this process is allowed to be, according to the registry.

Separate from the store so the refusals are testable without R2, and so the
decision to exit the process stays in the CLI where every other fatal startup
condition already lives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    # Per-environment share of `emission_cap`: `cap * env_split_e`.
    env_caps: dict[str, float]
    # The replica this task's validators verify with, or None to let each derive it.
    verification: str | None = None


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
    environments = list(generation_contract.get("environments") or ())
    if entry.env_split is not None:
        unknown = set(entry.env_split) - set(environments)
        if unknown:
            raise TaskConfigError(
                f"task {task_id!r} declares env_split for "
                f"{sorted(unknown)}, which profile {profile_id!r} does not "
                f"have; it declares {sorted(environments)}"
            )
        # A partial split is refused here, at startup where an operator sees
        # it, rather than at the first window: FillClosedBatchAssembler
        # raises on a window_pool map missing an environment it must run,
        # which would otherwise jam every window open attempt silently.
        uncovered = set(environments) - set(entry.env_split)
        if uncovered:
            raise TaskConfigError(
                f"task {task_id!r} declares env_split but it does not cover "
                f"{sorted(uncovered)}, which profile {profile_id!r} also "
                f"declares; env_split must name every profile environment"
            )

    params = PriceParams(**{f: entry.params[f] for f in PRICE_PARAM_FIELDS})
    cap = float(entry.params["cap"])
    if entry.env_split is not None:
        env_caps = {
            environment: cap * float(share)
            for environment, share in entry.env_split.items()
        }
    elif environments:
        env_caps = {environment: cap / len(environments) for environment in environments}
    else:
        env_caps = {}
    return TaskConfig(
        task_id=task_id,
        entry=entry,
        price_params=params,
        emission_cap=cap,
        env_caps=env_caps,
        verification=entry.verification,
    )


def legacy_registry_fallback(
    entries: Mapping[str, TaskEntry], task_ids: Iterable[str]
) -> bool:
    """True iff the legacy pre-registry behaviour is the right one to take.

    One predicate, called by both the startup path (which passes the single
    task this process is) and the weight submitter (which passes the set of
    tasks that have archives). Written independently they drifted, and the
    only safe answer is the narrow one: no registry object exists AT ALL and
    the only task in play is the legacy ``default``. A registry that exists
    but does not name us is a refusal, not a fallback -- otherwise a task
    nobody declared gets paid, which is what the check is for.
    """
    if entries:
        return False
    return set(task_ids) == {DEFAULT_TASK_ID}


def legacy_task_config() -> TaskConfig:
    """The pre-registry behaviour of the single task that predates it.

    Only for a wholly absent registry: a present-but-wrong one still refuses.
    """
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    cap = 1.0
    environments = list(ACTIVE_PROTOCOL_PROFILE.environments)
    env_caps = (
        {environment: cap / len(environments) for environment in environments}
        if environments
        else {}
    )
    return TaskConfig(
        task_id=DEFAULT_TASK_ID,
        entry=None,
        price_params=PRODUCTION_PRICE_PARAMS,
        emission_cap=cap,
        env_caps=env_caps,
    )
