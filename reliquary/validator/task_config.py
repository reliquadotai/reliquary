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
    if entry.contract is not None:
        from reliquary.constants import SUPPORTED_MODEL_ARCHITECTURES
        from reliquary.environment.registry import ENVIRONMENT_SPECS
        from reliquary.protocol.profiles import profile_from_contract

        # Shape is the registry's job; whether this binary can execute the
        # contract is ours, and it must fail here rather than mid-window.
        declared = set(entry.contract.get("environments") or ())
        missing = sorted(declared - set(ENVIRONMENT_SPECS))
        if missing:
            raise TaskConfigError(
                f"task {task_id!r} names environments this binary does not "
                f"install: {missing}"
            )
        # Read from the REBUILT profile, not from the raw contract: what the
        # process will actually run is the round trip, so a field only the raw
        # mapping carries is a field nothing enforces.
        try:
            architecture = profile_from_contract(entry.contract).model_architecture
        except ValueError as exc:
            raise TaskConfigError(
                f"task {task_id!r} carries a contract this binary cannot "
                f"read: {exc}"
            ) from exc
        # Contract-LESS entries are the historical form and keep working; a
        # carried contract without an architecture is a state nothing
        # produces, and accepting it leaves the check below unreachable for
        # exactly the entries it guards.
        if architecture is None:
            raise TaskConfigError(
                f"task {task_id!r} carries a contract that names no model "
                f"architecture; seal one into the contract so this image can "
                f"refuse a model it cannot run"
            )
        if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
            raise TaskConfigError(
                f"task {task_id!r} names model architecture "
                f"{architecture!r}, which this image cannot run; it supports "
                f"{sorted(SUPPORTED_MODEL_ARCHITECTURES)}"
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


def merge_corpus_contracts(contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One contract a process can run for several corpus tasks at once.

    Each corpus contract is its template narrowed to one environment, so the
    merge is the union of the environments; every other field (model, proofs,
    sampling, ...) must already agree, or one process cannot serve them all.
    """
    items = list(contracts.items())
    first_task, first = items[0]
    environments: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for task_id, contract in items:
        for key in sorted(set(first) | set(contract)):
            if key != "environments" and first.get(key) != contract.get(key):
                raise ValueError(
                    f"tasks {first_task!r} and {task_id!r} carry different {key!r}; "
                    "one process can only serve contracts that differ in their environments"
                )
        for name, body in (contract.get("environments") or {}).items():
            if name in environments and environments[name] != body:
                raise ValueError(
                    f"tasks {owner[name]!r} and {task_id!r} declare environment {name!r} differently"
                )
            environments.setdefault(name, body)
            owner.setdefault(name, task_id)
    return {**first, "environments": environments}


def resolve_corpus_task_configs(
    entries: Mapping[str, TaskEntry],
    task_ids: Iterable[str],
    *,
    profile_id: str,
    generation_contract: Any,
) -> list[TaskConfig]:
    """Each corpus task one process serves, in ``task_ids`` order, or a refusal.

    Every entry is resolved against its own carried contract, and that contract
    must be exactly this process's contract narrowed to its environments: the
    process renders and proves with ONE contract, the merge of all of them.
    """
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

    task_ids = tuple(task_ids)
    configs = []
    for task_id in task_ids:
        entry = entries.get(task_id)
        if entry is not None and getattr(entry, "mechanism", None) != MECHANISM_CORPUS_GENERATION:
            raise TaskConfigError(
                f"task {task_id!r} is {entry.mechanism!r}, not a corpus-generation task; "
                "only a corpus validator serves several task ids"
            )
        if entry is not None and entry.contract is None:
            raise TaskConfigError(
                f"task {task_id!r} carries no contract; a corpus validator serving several "
                "tasks needs each one's contract to check it against the one it runs"
            )
        contract = entry.contract if entry is not None else generation_contract
        config = resolve_task_config(
            entries, task_id, profile_id=profile_id, generation_contract=contract
        )
        narrowed = {
            **generation_contract,
            "environments": {
                name: (generation_contract.get("environments") or {}).get(name)
                for name in (contract.get("environments") or {})
            },
        }
        if canonical_sha256(narrowed) != canonical_sha256(contract):
            differing = sorted(
                key for key in set(narrowed) | set(contract)
                if narrowed.get(key) != contract.get(key)
            )
            raise TaskConfigError(
                f"task {task_id!r}'s contract differs from the one this process runs in "
                f"{differing}; start it with RELIQUARY_TASK_CONTRACT from `reliquary tasks "
                f"contract --task-id {' --task-id '.join(task_ids)}`"
            )
        configs.append(config)
    return configs
