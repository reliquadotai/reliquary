"""The live, revisioned schedule of a v2 service order: which envs run, their share, their cooldown."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_catalogue import (
    MAX_ADDED_ENVIRONMENTS,
    catalogued,
    validate_added,
)
from reliquary.protocol.service_contract import (
    ServiceContract,
    ServiceContractError,
    _integer,
    _object,
    _sha,
)

SCHEDULE_SCHEMA = "service-schedule/v1"


class ScheduleError(ValueError):
    pass


def _validate(value: dict, contract: ServiceContract) -> None:
    _object(value, {"schema", "order_sha256", "revision", "environments"}, "schedule")
    if value["schema"] != SCHEDULE_SCHEMA:
        raise ScheduleError("unknown schedule schema")
    _sha(value["order_sha256"], "order_sha256")
    if value["order_sha256"] != contract.sha256:
        raise ScheduleError("schedule belongs to another order")
    _integer(value["revision"], "revision", 0)
    environments = value["environments"]
    if not isinstance(environments, dict):
        raise ScheduleError("schedule must list exactly the order's environments")
    order = contract.order_environments
    added = {name for name, row in environments.items() if isinstance(row, dict) and "added" in row}
    if set(environments) - added != set(order) or added & set(order):
        raise ScheduleError("schedule must list exactly the order's environments and the ones added to it")
    if len(added) > MAX_ADDED_ENVIRONMENTS:
        raise ScheduleError(f"at most {MAX_ADDED_ENVIRONMENTS} environments can be added to a run")
    # Each added row is checked against the order's envs and the added rows already checked (every pair once).
    existing = dict(order)
    total, active = 0, 0
    for name, row in environments.items():
        _object(row, {"active", "share_bps", "cooldown_windows"} | ({"added"} if name in added else set()),
                f"schedule.{name}")
        if name in added:
            try:
                validate_added(name, row["added"], existing, order)
            except ServiceContractError as exc:
                raise ScheduleError(str(exc)) from exc
            existing[name] = {**row["added"]["environment"], "catalogue": row["added"]["catalogue"]}
        if type(row["active"]) is not int or row["active"] not in (0, 1):
            raise ScheduleError(f"{name}.active must be 0 or 1")
        share = _integer(row["share_bps"], f"{name}.share_bps", 0, 10000)
        _integer(row["cooldown_windows"], f"{name}.cooldown_windows", 0, 1_000_000)
        if row["active"]:
            if share == 0:
                raise ScheduleError(f"active environment {name} needs a positive share")
            total += share
            active += 1
        elif share != 0:
            raise ScheduleError(f"inactive environment {name} must have share 0")
    if active == 0:
        raise ScheduleError("a schedule needs at least one active environment")
    if total != 10000:
        raise ScheduleError("active shares must sum to 10000")


@dataclass(frozen=True, slots=True)
class ServiceSchedule:
    canonical: bytes

    @classmethod
    def from_dict(cls, value: Mapping, contract: ServiceContract) -> "ServiceSchedule":
        try:
            raw = canonical_json_bytes(dict(value))
        except (TypeError, ValueError) as exc:
            raise ScheduleError(str(exc)) from exc
        decoded = json.loads(raw)
        try:
            _validate(decoded, contract)
        except ScheduleError:
            raise
        except ValueError as exc:
            raise ScheduleError(str(exc)) from exc
        return cls(raw)

    def to_dict(self) -> dict:
        return json.loads(self.canonical)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def revision(self) -> int:
        return self.to_dict()["revision"]

    @property
    def order_sha256(self) -> str:
        return self.to_dict()["order_sha256"]

    def active_environments(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, row in self.to_dict()["environments"].items() if row["active"]))

    def share_bps(self, name: str) -> int:
        return self.to_dict()["environments"][name]["share_bps"]

    def cooldown_windows(self, name: str) -> int:
        return self.to_dict()["environments"][name]["cooldown_windows"]

    def added_environments(self) -> dict[str, dict]:
        """``{name: {"environment", "catalogue"}}`` of the envs added to the run."""
        return {name: row["added"] for name, row in self.to_dict()["environments"].items() if "added" in row}

    def contract_view(self, contract: ServiceContract) -> ServiceContract:
        """The order's contract with the envs this schedule added (the plain contract when none)."""
        return catalogued(contract, self.to_dict())


def initial_schedule(contract: ServiceContract) -> ServiceSchedule:
    environments = {
        name: {"active": int(env["share_bps"] > 0), "share_bps": env["share_bps"],
               "cooldown_windows": env["cooldown_windows"]}
        for name, env in contract.order_environments.items()
    }
    return ServiceSchedule.from_dict({"schema": SCHEDULE_SCHEMA, "order_sha256": contract.sha256,
                                      "revision": 0, "environments": environments}, contract)


def next_schedule(contract: ServiceContract, current: ServiceSchedule, *,
                  active: Iterable[str] | None = None, shares: Mapping[str, int] | None = None,
                  cooldowns: Mapping[str, int] | None = None,
                  add: Mapping[str, Mapping] | None = None) -> ServiceSchedule:
    """The next revision of ``current``. ``add`` adds envs to the run: each enters inactive with share 0 and its
    entry's cooldown, unless ``active``/``shares`` of the same call switch it on."""
    value = current.to_dict()
    environments = value["environments"]
    if add:
        from reliquary.environment.registry import ENVIRONMENT_SPECS
    for name, block in (add or {}).items():
        if name in environments:
            raise ScheduleError(f"{name} is already in the schedule")
        if name in ENVIRONMENT_SPECS:
            # As for an order's catalogue env: a bridged env never takes the name of an env of this build.
            raise ScheduleError(f"{name} is installed in this build; an added environment needs a name of its own")
        try:
            cooldown = int(block["environment"]["cooldown_windows"])
            added = {"environment": block["environment"], "catalogue": block["catalogue"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise ScheduleError(f"{name}: malformed added environment") from exc
        environments[name] = {"active": 0, "share_bps": 0, "cooldown_windows": cooldown, "added": added}
    names = set(environments)
    for given in (set(active or ()), set(shares or {}), set(cooldowns or {})):
        unknown = given - names
        if unknown:
            raise ScheduleError(f"unknown environment(s): {sorted(unknown)}")
    if active is not None:
        chosen = set(active)
        if shares is None:
            raise ScheduleError("changing the active set needs explicit shares")
        for name in names:
            environments[name]["active"] = int(name in chosen)
            environments[name]["share_bps"] = 0
    if shares is not None:
        for name, share in shares.items():
            environments[name]["share_bps"] = share
    for name, windows in (cooldowns or {}).items():
        environments[name]["cooldown_windows"] = windows
    value["revision"] = current.revision + 1
    return ServiceSchedule.from_dict(value, contract)
