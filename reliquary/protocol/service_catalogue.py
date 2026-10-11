"""Envs added to a running run (spec §6; amends phase 1 ruling 3).

An added env travels inside the schedule: a schedule row may carry
``added = {"environment": <a v2 env entry>, "catalogue": {package, options, rl_split, image_lock_sha256,
conformance_sha256}}``. The order's contract and its sha256 never change. ``catalogued(contract, schedule)``
overlays the added entries, each with its catalogue merged in (``CataloguedContract``): ``environments``,
``environment``, ``episode_policy``, ``episode_environments`` and ``catalogue_of`` see them; ``sha256``,
``canonical`` and ``to_dict`` stay the order's. Rows only accumulate (an added env is switched off by the
schedule, never removed), so the latest schedule's view is a superset of every window's. A schedule without
added rows yields the plain contract: a run that adds nothing keeps every byte.

Only signed-episode envs served through the verifiers bridge are added, and only to a run that launched with
one: the order's largest ``max_episode_tokens`` and ``max_turns`` bound the added env's (the values the trainer
was qualified for). An added env is a new distribution under a new sandbox env name (a new version of an env
cannot share a validator's venv or a gateway's env name with the old one). Its catalogue passes the same checks
as a launch env's (``validate_catalogue``: bridged identity, split and tools, no credential-looking option) and
must name its conformance report."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_contract import (
    ServiceContract,
    ServiceContractError,
    _object,
    validate_environment,
)

ADDED_FIELDS = frozenset({"environment", "catalogue"})
MAX_ADDED_ENVIRONMENTS = 16


def distribution_of(env_package: str) -> str:
    """The distribution name of a pinned env package (``name==version+...``), normalised as pip compares
    names (PEP 503: runs of ``-``, ``_`` and ``.`` are one ``-``, case-insensitive)."""
    return re.sub(r"[-_.]+", "-", str(env_package).split("==", 1)[0]).lower()


def _merged(added: Mapping[str, Any]) -> dict:
    return {**added["environment"], "catalogue": added["catalogue"]}


def validate_added(name: str, added: Any, existing: Mapping[str, Mapping[str, Any]],
                   order_environments: Mapping[str, Mapping[str, Any]]) -> None:
    """One ``added`` block of a schedule row (``ServiceContractError`` otherwise). ``existing``: every other env
    of the run already validated (the order's and the other added ones, catalogues merged);
    ``order_environments``: the order's."""
    _object(added, set(ADDED_FIELDS), f"{name}.added")
    if name in order_environments:
        raise ServiceContractError(f"{name} is already an environment of the order")
    env = added["environment"]
    if not isinstance(env, dict) or "episode" not in env:
        raise ServiceContractError(f"{name}: only signed-episode environments (an episode block) are added "
                                   f"to a running run")
    if "catalogue" in env:
        raise ServiceContractError(f"{name}: the catalogue of an added environment is the block's, not the entry's")
    if not isinstance(added["catalogue"], dict) or "conformance_sha256" not in added["catalogue"]:
        raise ServiceContractError(f"{name}.catalogue: an added environment names its conformance report "
                                   f"(conformance_sha256)")
    validate_environment(name, _merged(added))
    if env["share_bps"] != 0:
        raise ServiceContractError(f"{name}: an added environment's share is the schedule's; share_bps must be 0")
    from reliquary.constants import M_ROLLOUTS

    if env["sampling"]["group_size"] != M_ROLLOUTS:
        raise ServiceContractError(f"{name}: a public seed pool group_size must be M_ROLLOUTS")
    episode = env["episode"]
    bounds = [e["episode"] for e in order_environments.values() if isinstance(e, Mapping) and "episode" in e]
    if not bounds:
        raise ServiceContractError(f"{name}: the order has no episode environment to bound an added one")
    # Bounded by the order's largest values. A miner's engine serves only envs with exactly the same caps
    # (``check_engine_caps``): an added env whose caps differ from a launch env's needs its own engine.
    for field in ("max_episode_tokens", "max_turns", "max_tokens_per_turn"):
        limit = max(int(bound[field]) for bound in bounds)
        if int(episode[field]) > limit:
            raise ServiceContractError(f"{name}: episode.{field} {episode[field]} is above the order's {limit} "
                                       f"(the trainer is qualified for the order's episodes)")
    for other, entry in existing.items():
        if other == name or not isinstance(entry, Mapping) or "episode" not in entry:
            continue
        if entry["episode"]["sandbox_env"] == episode["sandbox_env"]:
            raise ServiceContractError(f"{name}: sandbox env {episode['sandbox_env']} already serves {other}; an "
                                       f"added environment (a new version too) takes a new sandbox env name")
        if distribution_of(entry["episode"]["env_package"]) == distribution_of(episode["env_package"]):
            raise ServiceContractError(f"{name}: distribution {distribution_of(episode['env_package'])} already "
                                       f"serves {other}; a new version is a new distribution name")


@dataclass(frozen=True, slots=True)
class CataloguedContract(ServiceContract):
    """The order's contract with the envs its schedule added. ``added`` is the canonical JSON of
    ``{name: env entry with its catalogue}``; ``canonical`` stays the order's bytes."""

    added: bytes = b"{}"

    def __post_init__(self) -> None:
        ServiceContract.__post_init__(self)
        if not isinstance(self.added, bytes):
            raise ServiceContractError("added environments must be bytes")
        try:
            value = json.loads(self.added)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ServiceContractError("added environments are not JSON") from exc
        if not isinstance(value, dict) or not 1 <= len(value) <= MAX_ADDED_ENVIRONMENTS:
            raise ServiceContractError(f"1..{MAX_ADDED_ENVIRONMENTS} added environments required")
        if canonical_json_bytes(value) != self.added:
            raise ServiceContractError("added environments must be canonical")
        order = ServiceContract.environments.fget(self)
        for name, env in value.items():
            if name in order:
                raise ServiceContractError(f"{name} is already an environment of the order")
            validate_environment(name, env)

    @property
    def order_environments(self) -> dict:
        return ServiceContract.environments.fget(self)

    @property
    def environments(self) -> dict:
        return {**ServiceContract.environments.fget(self), **json.loads(self.added)}

    def require_capabilities(self, supported: set[str]) -> None:
        ServiceContract.require_capabilities(self, supported)
        required = set()
        for env in json.loads(self.added).values():
            required.add(env["sampling"]["kind"])
            if "episode" in env:
                required.add(env["episode"]["kind"])
        missing = required - set(supported)
        if missing:
            raise ServiceContractError(f"runtime lacks capabilities: {sorted(missing)}")


def order_of(contract: ServiceContract) -> ServiceContract:
    """The plain order contract of ``contract`` (a view is rebuilt from the order's bytes)."""
    return ServiceContract(contract.canonical) if isinstance(contract, CataloguedContract) else contract


def overlay(contract: ServiceContract, added: Mapping[str, Mapping[str, Any]]) -> ServiceContract:
    """``contract``'s order with ``added`` (``{name: added block}`` of an already validated schedule)."""
    base = order_of(contract)
    if not added:
        return base
    return CataloguedContract(base.canonical, canonical_json_bytes({name: _merged(block)
                                                                    for name, block in added.items()}))


def catalogued(contract: ServiceContract, schedule: Any) -> ServiceContract:
    """``contract`` with the envs ``schedule`` (a schedule dict) added; the plain order contract when no row
    carries ``added``. A schedule with an ``added`` row is parsed in full (``ScheduleError`` otherwise): the
    view is never laxer than the schedule, and a malformed row is an error, never skipped."""
    base = order_of(contract)
    rows = schedule.get("environments") if isinstance(schedule, Mapping) else None
    if not isinstance(rows, Mapping) or not any(isinstance(row, Mapping) and "added" in row
                                                for row in rows.values()):
        return base
    from reliquary.protocol.service_schedule import ServiceSchedule

    return overlay(base, ServiceSchedule.from_dict(schedule, base).added_environments())


def announced_contract(announcement: Mapping[str, Any]) -> ServiceContract:
    """The contract a window's announcement serves: the order's, with the envs its schedule added."""
    return catalogued(ServiceContract.from_dict(announcement["contract"]), announcement.get("schedule"))


__all__ = ["ADDED_FIELDS", "MAX_ADDED_ENVIRONMENTS", "CataloguedContract", "announced_contract", "catalogued",
           "distribution_of", "order_of", "overlay", "validate_added"]
