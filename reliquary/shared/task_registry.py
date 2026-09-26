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

# Imported from `release_contract`, whose imports are stdlib only, so this
# module stays free of `reliquary.environment.abi` and the I/O behind it.
from reliquary.corpus.audit_policy import validate_audit_params
from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared.task_id import DEFAULT_TASK_ID, normalise_task_id

REGISTRY_VERSION = 1
MECHANISM_RL_DISCOVERED_PRICE = "rl-discovered-price"
# Corpus generation on a frozen checkpoint: paid per verified token, with the
# price pinned by declaring floor == cap.
MECHANISM_CORPUS_GENERATION = "corpus-generation"
KNOWN_MECHANISMS = frozenset(
    {MECHANISM_RL_DISCOVERED_PRICE, MECHANISM_CORPUS_GENERATION}
)
# What a task may pin as the way its rollouts are verified. Declaring one makes every validator
# run the same path whatever its card; declaring none lets each derive it, which is the default.
KNOWN_VERIFICATION = frozenset({"resident", "streamed"})

# Every field the REGISTRY declares per task. A missing one is refused rather
# than defaulted: a half-specified controller is not a controller.
#
# Deliberately NOT every field ``PriceParams`` has: ``breaker_timeouts`` (how
# many consecutive unfilled windows before an environment's price stops
# escalating) is a controller-tuning knob versioned in the image alongside
# ``PRODUCTION_PRICE_PARAMS``, not a per-task economic declaration, and
# ``PriceParams`` carries its own default for it. Completing this tuple to
# match ``PriceParams`` field-for-field would make every registry entry
# written before ``breaker_timeouts`` existed suddenly "missing price
# parameters" -- and an invalid registry makes the validator refuse to start
# and the weight submitter abstain.
PRICE_PARAM_FIELDS = (
    "start", "decay", "rounds_per_step", "deadband",
    "snap", "floor", "cap", "median_rounds", "last_good_fills",
)

# Optional per-task minimum-incentive floor: a hotkey's share is measured within
# its own task, and absent keys fall back to the protocol-wide floor.
INCENTIVE_FLOOR_FIELDS = ("min_incentive_share", "min_incentive_ramp_start")

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
    # How the task's cap divides between its environments, e.g.
    # {"math": 0.6, "code": 0.4}. None means "not declared".
    env_split: Mapping[str, float] | None = None
    # The generation contract this task runs, carried in full. None means the
    # legacy form, where the binary holds the contract and the entry pins its
    # hash in `profile_sha256`.
    contract: Mapping[str, Any] | None = None
    # Which replica the task's validators verify with, when the task pins one.
    # None means "not declared": each validator derives it from its own card.
    verification: str | None = None
    # The corpus job this task generates for. Beside the contract, never inside
    # it: the contract says how generation happens, the job says which work to
    # do, and putting it inside would move every RL contract's digest.
    job_id: str | None = None


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


def _validate_incentive_floor(params: Mapping[str, Any]) -> None:
    share = params.get("min_incentive_share")
    start = params.get("min_incentive_ramp_start")
    for field, value in (("min_incentive_share", share), ("min_incentive_ramp_start", start)):
        if value is not None and not 0.0 <= _number(value, field) < 1.0:
            raise RegistryError(f"{field} must be in [0.0, 1.0), got {value}")
    if start is not None and share is None:
        # A ramp start with no share of its own would silently borrow the
        # protocol's, which is not what the declaration says.
        raise RegistryError("min_incentive_ramp_start needs min_incentive_share")
    if start is not None and float(start) > float(share):
        raise RegistryError(
            f"min_incentive_ramp_start {start} exceeds min_incentive_share {share}"
        )


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
    if entry.verification is not None and entry.verification not in KNOWN_VERIFICATION:
        raise RegistryError(
            f"unknown verification replica {entry.verification!r}; "
            f"declare one of {', '.join(sorted(KNOWN_VERIFICATION))}, or nothing to derive it"
        )
    if entry.status not in {"active", "retired"}:
        raise RegistryError(f"unknown status {entry.status!r}")
    missing = [f for f in PRICE_PARAM_FIELDS if f not in entry.params]
    if missing:
        raise RegistryError(f"missing price parameters: {', '.join(missing)}")
    # Every controller parameter must be a number here: a string that survives
    # to live arithmetic fails a window instead of refusing a start.
    for field in PRICE_PARAM_FIELDS:
        _number(entry.params[field], field)
    for field in ("rounds_per_step", "median_rounds", "last_good_fills"):
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
    _validate_incentive_floor(entry.params)
    try:
        validate_audit_params(entry.params)
    except ValueError as exc:
        raise RegistryError(str(exc)) from exc
    cap = _number(entry.params["cap"], "cap")
    if not 0.0 <= cap <= 1.0:
        raise RegistryError(f"cap must be between 0.0 and 1.0, got {cap}")
    floor = _number(entry.params["floor"], "floor")
    if floor > cap:
        raise RegistryError(f"floor {floor} exceeds cap {cap}")
    if entry.mechanism == MECHANISM_CORPUS_GENERATION:
        # V0 has no price discovery: an unpinned floor would animate advance()
        # with nothing driving it.
        if floor != cap:
            raise RegistryError(
                f"corpus task {entry.task_id!r} must pin its price: "
                f"floor {floor} must equal cap {cap}"
            )
        if not entry.job_id:
            raise RegistryError(f"corpus task {entry.task_id!r} names no job")
    elif entry.job_id is not None:
        raise RegistryError(
            f"task {entry.task_id!r} names a job but its mechanism is "
            f"{entry.mechanism!r}, which does not run one"
        )
    if entry.env_split is not None:
        if not isinstance(entry.env_split, Mapping) or not entry.env_split:
            raise RegistryError("env_split must be a non-empty object")
        total = 0.0
        for environment, share in entry.env_split.items():
            value = _number(share, f"env_split[{environment}]")
            if not 0.0 <= value <= 1.0:
                raise RegistryError(
                    f"env_split[{environment}] must be between 0.0 and 1.0"
                )
            total += value
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise RegistryError(
                f"env_split shares total {total:.4f}, which is not 1.0"
            )
    if entry.contract is not None:
        if not isinstance(entry.contract, Mapping):
            raise RegistryError(
                f"task {entry.task_id!r} carries a contract that is not an object"
            )
        # `profile_sha256` means the digest of the contract this task runs, so
        # an entry pinning one contract and carrying another attests work
        # nobody signed for. Every startup refusal downstream assumes this.
        try:
            digest = canonical_sha256(entry.contract)
        except (TypeError, ValueError) as exc:
            raise RegistryError(
                f"task {entry.task_id!r} carries a contract that cannot be "
                f"hashed: {exc}"
            ) from exc
        if digest != entry.profile_sha256:
            raise RegistryError(
                f"task {entry.task_id!r} carries a contract digesting to "
                f"{digest[:12]}… but pins profile_sha256 "
                f"{entry.profile_sha256[:12]}…"
            )


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
    # The other money invariant, and it lives here for the same reason: one
    # entry cannot see it. Two tasks naming one job would each pay their own
    # share for the SAME submissions.
    #
    # Retired entries are excluded: they accept no submission, so they cannot
    # double-pay for one, and counting them would make a cancelled job
    # undeclarable forever. Their cap stays guarded by the sum above.
    claimed: dict[str, str] = {}
    for task_id, entry in sorted(entries.items()):
        if not entry.job_id or entry.status != "active":
            continue
        if entry.job_id in claimed:
            raise RegistryError(
                f"tasks {claimed[entry.job_id]!r} and {task_id!r} both name "
                f"job {entry.job_id!r}; one job is paid for by one task"
            )
        claimed[entry.job_id] = task_id


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


def require_fleet_knows_corpus_generation(
    entry: TaskEntry, *, acknowledged: bool
) -> None:
    """Refuse to make the registry unreadable to validators that predate the
    corpus mechanism.

    ``validate_registry`` runs ``validate_entry`` over EVERY entry, and a
    binary whose ``KNOWN_MECHANISMS`` lacks ``corpus-generation`` refuses the
    whole registry -- so the first corpus entry stops every validator still on
    an older image, not just the corpus task. No CLI can read the fleet's
    version, so the operator states it. Deliberately NOT part of ``add_task``,
    for the same reason as ``require_default_declared_first``: this is a
    deployment precondition of a live subnet, not an invariant of the object.
    """
    if entry.mechanism != MECHANISM_CORPUS_GENERATION or acknowledged:
        return
    raise RegistryError(
        f"refusing to declare {entry.task_id!r}: a {MECHANISM_CORPUS_GENERATION!r} "
        f"entry makes the WHOLE registry unreadable to any validator whose "
        f"binary does not know that mechanism, and those validators refuse to "
        f"start. Confirm every validator already runs a binary that knows it, "
        f"then declare the task again with the acknowledgement flag."
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


def set_cap(
    entries: Mapping[str, TaskEntry],
    task_id: str,
    cap: float,
    floor: float | None = None,
    min_incentive_share: float | None = None,
    audit_q: float | None = None,
    audit_probation_submissions: int | None = None,
    audit_hold_seconds: float | None = None,
    audit_suspect_seconds: float | None = None,
    audit_ban_after_failures: int | None = None,
    audit_ban_window_seconds: float | None = None,
    audit_ban_seconds: float | None = None,
) -> dict[str, TaskEntry]:
    """Change one live entry's cap (and optionally floor), nothing else.

    The contract and its digest stay as they are: a cap is what the task may
    pay, not how it generates. A corpus task's price follows its cap unless a
    floor is named, and a named floor that breaks the pin is refused below.

    Every ``audit_*`` argument is optional and independent: an omitted one
    (``None``) leaves that entry's existing value untouched, so an operator
    can raise the ban window without also having to restate the hold.
    """
    if task_id not in entries:
        raise RegistryError(f"task {task_id!r} is not in the registry")
    entry = entries[task_id]
    if entry.status != "active":
        # A retired cap is reserved while its EMA decays; changing it would
        # hand out budget that is still being paid.
        raise RegistryError(f"task {task_id!r} is {entry.status}; its cap cannot change")
    params = {**entry.params, "cap": float(cap)}
    if floor is not None:
        params["floor"] = float(floor)
    elif entry.mechanism == MECHANISM_CORPUS_GENERATION:
        params["floor"] = float(cap)
    if min_incentive_share is not None:
        params["min_incentive_share"] = float(min_incentive_share)
        if float(params.get("min_incentive_ramp_start", 0.0)) > float(min_incentive_share):
            params["min_incentive_ramp_start"] = float(min_incentive_share)
    if audit_q is not None:
        params["audit_q"] = float(audit_q)
    if audit_probation_submissions is not None:
        params["audit_probation_submissions"] = int(audit_probation_submissions)
    if audit_hold_seconds is not None:
        params["audit_hold_seconds"] = float(audit_hold_seconds)
    if audit_suspect_seconds is not None:
        params["audit_suspect_seconds"] = float(audit_suspect_seconds)
    if audit_ban_after_failures is not None:
        params["audit_ban_after_failures"] = int(audit_ban_after_failures)
    if audit_ban_window_seconds is not None:
        params["audit_ban_window_seconds"] = float(audit_ban_window_seconds)
    if audit_ban_seconds is not None:
        params["audit_ban_seconds"] = float(audit_ban_seconds)
    updated = {**entries, task_id: replace(entry, params=params)}
    validate_registry(updated)
    return updated


def _verification_of(task_id: str, body: Mapping[str, Any]) -> str | None:
    """Read the declared replica, refusing a block this reader does not understand."""
    declared = body.get("verification")
    if declared is None:
        return None
    if not isinstance(declared, Mapping):
        raise RegistryError(f"task {task_id!r} verification must be an object")
    unknown = set(declared) - {"replica"}
    if unknown:
        raise RegistryError(
            f"task {task_id!r} verification declares {', '.join(sorted(unknown))}, "
            "which this validator does not know how to honour"
        )
    replica = declared.get("replica")
    if replica is not None and not isinstance(replica, str):
        raise RegistryError(f"task {task_id!r} verification replica must be a string")
    return replica


def _job_id_of(task_id: str, body: Mapping[str, Any]) -> str | None:
    """Read the declared job, refusing a value that is not a name.

    The id is interpolated into a bucket key by the job store, so a number or
    an object must be refused here rather than reach that interpolation.
    """
    declared = body.get("job_id")
    if declared is not None and not isinstance(declared, str):
        raise RegistryError(
            f"task {task_id!r} job_id must be a string or null, got {declared!r}"
        )
    return declared


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
        contract = body.get("contract")
        entries[task_id] = TaskEntry(
            task_id=task_id,
            profile_id=str(body.get("profile_id", "")),
            profile_sha256=str(body.get("profile_sha256", "")),
            mechanism=str(incentive.get("mechanism", "")),
            params=dict(params),
            status=str(body.get("status", "active")),
            retired_at=body.get("retired_at"),
            env_split=body.get("env_split"),
            # Copied on ingestion: the entry must own the mapping its digest
            # was checked against, or validate-then-mutate reopens the gap.
            contract=(
                dict(contract) if isinstance(contract, Mapping) else contract
            ),
            verification=_verification_of(task_id, body),
            job_id=_job_id_of(task_id, body),
        )
    if strict:
        validate_registry(entries)
    else:
        for entry in entries.values():
            validate_entry(entry)
    return entries


def render_registry(entries: Mapping[str, TaskEntry]) -> bytes:
    """Canonical bytes: sorted keys, so two writers produce the same object."""
    def _body(entry: TaskEntry) -> dict[str, Any]:
        body = {
            "profile_id": entry.profile_id,
            "profile_sha256": entry.profile_sha256,
            "incentive": {
                "mechanism": entry.mechanism,
                "params": dict(entry.params),
            },
            "status": entry.status,
            "retired_at": entry.retired_at,
            "env_split": (
                None if entry.env_split is None else dict(entry.env_split)
            ),
            # Always written, null when undeclared, like `env_split`. `contract`
            # below is the one field that is omitted instead — see its comment.
            "verification": (
                None if entry.verification is None
                else {"replica": entry.verification}
            ),
            # Always written, null when undeclared, like `verification`: one
            # convention for the fields that sit beside the contract.
            "job_id": entry.job_id,
        }
        # Omitted rather than written as null, so a registry holding only legacy
        # entries renders exactly as it did before contracts existed.
        if entry.contract is not None:
            body["contract"] = dict(entry.contract)
        return body

    document = {
        "registry_version": REGISTRY_VERSION,
        "tasks": {
            task_id: _body(entry) for task_id, entry in sorted(entries.items())
        },
    }
    # allow_nan=False: the Python default emits bare NaN/Infinity literals that
    # no other JSON reader accepts, so an unreadable object would reach R2.
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
