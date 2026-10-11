"""Live operator control of a v2 service order's schedule: no restart, applied at the next window.

The operator writes a request file into a local directory they own; the validator picks it up at a
window boundary (``ServiceRuntime.apply_pending_schedule_request``), validates it again (the file
store is not trusted) and appends the next schedule revision. The running window keeps its frozen
envelope (R6). No network endpoint.

    python -m reliquary.services.schedule --folder DIR show
    python -m reliquary.services.schedule --folder DIR set --active a,b --share a=6000 --share b=4000
    python -m reliquary.services.schedule --folder DIR set --cooldown a=200
    python -m reliquary.services.schedule --folder DIR add NAME --entry block.json --attestation report.json

``add`` adds a bridged signed-episode env to the running run: its added block (``--entry``) and the
operator's conformance report, a file of the request folder whose sha256 the block names. The validator
runs its activation checks; when it cannot settle them yet (a stale fleet view) the request stays pending
("waiting", with the reason) for at most ``RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S`` seconds, then is refused.

``--folder`` (or ``RELIQUARY_SERVICE_SCHEDULE_FOLDER``) is required: there is no default location.
The runtime database is read read-only, from ``<folder>/runtime.sqlite3`` unless ``--db`` is given.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from reliquary.protocol.service_contract import ServiceContract, _identifier
from reliquary.protocol.service_schedule import ServiceSchedule, next_schedule
from reliquary.shared.strict_json import strict_json_loads
from reliquary.validator.control import write_json

REQUEST_SCHEMA = "service-schedule-request/v1"
REQUEST_FIELDS = {"schema", "request_id", "order_sha256", "revision", "active", "shares", "cooldowns"}
OPTIONAL_REQUEST_FIELDS = frozenset({"add"})   # envs added to the running run
ATTESTATION_MAX_BYTES = 16 * 1024 * 1024
_ATTESTATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# Read errors of an attestation that say nothing about the file and may pass: the request waits.
TRANSIENT_READ_ERRNOS = frozenset({errno.EIO, errno.EMFILE, errno.ENFILE, errno.EAGAIN, errno.EINTR,
                                   errno.ENOMEM, errno.EBUSY})
MAX_FILE_BYTES = 64 * 1024
FOLDER_ENV = "RELIQUARY_SERVICE_SCHEDULE_FOLDER"
DB_NAME = "runtime.sqlite3"


class ScheduleRefused(ValueError):
    """A schedule request that must not be applied; the message is shown to the operator."""


class ScheduleNotYet(Exception):
    """An added env's activation checks cannot be settled now (a stale or incomplete fleet view, an I/O
    error): the request stays pending and is checked again at the next window boundary. Not a ValueError,
    so no refusal path takes it for one."""


def _wait_max_s() -> int:
    raw = os.environ.get("RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S", "3600")
    if not (raw.isascii() and raw.isdigit()) or not 0 < int(raw) <= 7 * 86400:
        raise ValueError("RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S must be a whole number of seconds in [1, 604800]")
    return int(raw)


# How long a request that adds an env may wait on activation checks that cannot be settled yet, counted
# from the first boundary that could not settle them; past it the request is refused (final). An operator
# setting: one hour by default, enough for a gateway restart or a missed report, short enough that a dead
# machine still listed in the directory does not hold the request (and its other changes) for long.
ADD_WAIT_MAX_S = _wait_max_s()


def raise_activation_problem(name: str, problem: str | None) -> None:
    """Turn an activation answer (``validator.catalogue_activation``: None, a final problem, or a ``NotYet``)
    into ``check_request``'s ``catalogue_check`` outcome."""
    if problem is None:
        return
    from reliquary.validator.catalogue_activation import NotYet

    if isinstance(problem, NotYet):
        raise ScheduleNotYet(f"cannot add {name} yet: {problem}")
    raise ScheduleRefused(f"cannot add {name}: {problem}")


# ---------------------------------------------------------------- file store

class UnsafeLocation(ValueError):
    """The folder or request file has the wrong owner or can be written by others."""


def _check_owner_and_mode(info, label: str) -> None:
    """The operator's folder and request file belong to the validator's user and nobody else can write them."""
    if info.st_uid != os.geteuid():
        raise UnsafeLocation(f"{label} is not owned by the user running the validator (uid {os.geteuid()})")
    if info.st_mode & 0o022:
        raise UnsafeLocation(f"{label} is writable by group or others")


def _read_json_file(folder: Path, name: str):
    """Strict JSON of ``folder/name`` or None if absent. Refuses symlinks, non-regular files,
    group/world-writable files, files over ``MAX_FILE_BYTES`` and anything but strict JSON."""
    raw = _read_bytes(folder, name, MAX_FILE_BYTES)
    if raw is None:
        return None
    try:
        return strict_json_loads(raw)
    except ValueError as exc:  # includes bad UTF-8, duplicate keys, NaN
        raise ValueError(f"{name} is not strict JSON: {exc}") from exc


def _read_bytes(folder: Path, name: str, max_bytes: int, *, transient_raises: bool = False) -> bytes | None:
    """The bytes of ``folder/name`` or None if absent, with ``_read_json_file``'s safety checks. With
    ``transient_raises``, opening the file with an errno of ``TRANSIENT_READ_ERRNOS`` raises that OSError
    (otherwise every open error is a ValueError)."""
    try:
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    try:
        _check_owner_and_mode(os.fstat(directory), f"folder {folder}")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
                         dir_fd=directory)
        except FileNotFoundError:
            return None
        except OSError as exc:  # ELOOP for a symlink
            if transient_raises and exc.errno in TRANSIENT_READ_ERRNOS:
                raise
            raise ValueError(f"{name} cannot be opened safely ({exc.strerror})") from exc
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{name} is not a regular file")
        _check_owner_and_mode(info, name)
        if info.st_size > max_bytes:
            raise ValueError(f"{name} is larger than {max_bytes} bytes")
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{name} is larger than {max_bytes} bytes")
    return raw


def build_request(*, order_sha256: str, revision: int, active=None, shares=None, cooldowns=None,
                  add=None) -> dict:
    """A request file's content; ``add`` (``{env: {"environment", "catalogue", "attestation"}}``) only when
    given, so a request that adds nothing keeps its exact fields."""
    request = {"schema": REQUEST_SCHEMA, "request_id": uuid.uuid4().hex,
               "order_sha256": order_sha256, "revision": revision,
               "active": None if active is None else sorted(active),
               "shares": None if shares is None else dict(shares),
               "cooldowns": None if cooldowns is None else dict(cooldowns)}
    if add is not None:
        request["add"] = {name: dict(value) for name, value in add.items()}
    return request


class ScheduleRequestStore:
    """``<folder>/schedule-request.json`` (operator -> validator) and ``schedule-status.json`` (back)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.request_path = self.folder / "schedule-request.json"
        self.status_path = self.folder / "schedule-status.json"

    def submit(self, *, order_sha256: str, revision: int, active=None, shares=None, cooldowns=None) -> dict:
        request = build_request(order_sha256=order_sha256, revision=revision, active=active, shares=shares,
                                cooldowns=cooldowns)
        self.write(request)
        return request

    def _ensure_folder(self) -> None:
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)   # umask may only narrow it

    def write(self, request: dict) -> None:
        self._ensure_folder()
        write_json(self.request_path, request)  # temp file + fsync + rename + directory fsync

    def take(self) -> dict | None:
        """The request file as a dict (None if there is none); ValueError if it is unsafe or not JSON.
        Schema validation is ``check_request``'s."""
        value = _read_json_file(self.folder, self.request_path.name)
        if value is not None and not isinstance(value, dict):
            raise ValueError("schedule request must be a JSON object")
        return value

    def read_attestation(self, name: str) -> tuple[bytes, Any]:
        """An attestation file of this folder (a plain name): its bytes and its strict JSON. ValueError
        when it is absent, unsafe or not JSON; OSError when it cannot be read now (``TRANSIENT_READ_ERRNOS``)."""
        if not isinstance(name, str) or not _ATTESTATION_NAME.fullmatch(name):
            raise ValueError("an attestation is a plain file name of the request folder")
        raw = _read_bytes(self.folder, name, ATTESTATION_MAX_BYTES, transient_raises=True)
        if raw is None:
            raise ValueError(f"attestation {name} is not in the request folder")
        try:
            return raw, strict_json_loads(raw)
        except ValueError as exc:
            raise ValueError(f"attestation {name} is not strict JSON: {exc}") from exc

    def report(self, request_id: str, *, status: str, detail: str, revision: int) -> None:
        self._ensure_folder()
        write_json(self.status_path, {"request_id": request_id, "status": status, "detail": detail,
                                      "revision": revision, "at": time.time()})

    def status(self) -> dict | None:
        value = _read_json_file(self.folder, self.status_path.name)
        if value is not None and not isinstance(value, dict):
            raise ValueError("schedule status must be a JSON object")
        return value


# ---------------------------------------------------------------- validation (CLI and runtime)

def default_installed_version(name: str) -> str | None:
    """The manifest hash of the installed env ``name`` (None if not installed), as
    ``validator.task_config._service_env_caps`` reads it."""
    from reliquary.environment.registry import find_environment_spec
    return getattr(find_environment_spec(name), "environment_manifest_sha256", None)


def _integer_map(value, name: str) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ScheduleRefused(f"{name} must be an object of environment -> integer")
    for env, number in value.items():
        if not isinstance(env, str) or type(number) is not int:
            raise ScheduleRefused(f"{name} must map environment names to integers")
    return dict(value)


def check_request(request, *, contract: ServiceContract, current: ServiceSchedule,
                  installed_version: Callable[[str], str | None] | None = None,
                  catalogue_check: Callable[[str, dict, Any], None] | None = None,
                  read_attestation: Callable[[str], tuple[bytes, Any]] | None = None) -> ServiceSchedule:
    """The schedule that ``request`` asks for, or ``ScheduleRefused``/``ValueError`` with the reason.

    Checks: schema and exact fields; the order hash is the running order's; ``revision`` is exactly
    ``current + 1`` (no replay, no gap); only declared envs; active shares sum to 10000; cooldowns
    within the contract's advice bounds; and, for each env that becomes active, R7: installed, and
    installed version equal to the pinned one. Nothing is mutated.

    An env added by the request (``add``) needs ``read_attestation(file name) -> (bytes, json)``, which reads
    the operator's attestation (its sha256 must be the block's ``conformance_sha256``), and
    ``catalogue_check(name, added, attestation)``, the validator's activation checks: it raises
    ``ScheduleRefused`` when the env cannot be activated, or ``ScheduleNotYet`` when that cannot be settled now
    (raised from here, after every final check passed). ``contract`` may be the contract view
    (``ServiceSchedule.contract_view``): an env added earlier is then known.
    """
    installed_version = installed_version or default_installed_version
    if not isinstance(request, dict) or not REQUEST_FIELDS <= set(request) <= REQUEST_FIELDS | OPTIONAL_REQUEST_FIELDS:
        if not isinstance(request, dict) or not OPTIONAL_REQUEST_FIELDS & set(request):
            raise ScheduleRefused(f"a schedule request has exactly the fields {sorted(REQUEST_FIELDS)}")
        raise ScheduleRefused(f"a schedule request has exactly the fields {sorted(REQUEST_FIELDS)}, and "
                              f"optionally {sorted(OPTIONAL_REQUEST_FIELDS)}")
    if request["schema"] != REQUEST_SCHEMA:
        raise ScheduleRefused("unknown schedule request schema")
    try:
        _identifier(request["request_id"], "request_id")
    except ValueError as exc:
        raise ScheduleRefused(str(exc)) from exc
    if request["order_sha256"] != contract.sha256:
        raise ScheduleRefused("request was written for another order (order hash differs from the running order)")
    revision = request["revision"]
    if type(revision) is not int:
        raise ScheduleRefused("revision must be an integer")
    if revision <= current.revision:
        raise ScheduleRefused(f"stale or replayed request: revision {revision}, the schedule is already at "
                              f"{current.revision}")
    if revision != current.revision + 1:
        raise ScheduleRefused(f"revision gap: expected {current.revision + 1}, got {revision}")
    active = request["active"]
    if active is not None:
        if not isinstance(active, list) or not all(isinstance(x, str) for x in active) or len(set(active)) != len(active):
            raise ScheduleRefused("active must be a list of distinct environment names")
    shares = _integer_map(request["shares"], "shares")
    cooldowns = _integer_map(request["cooldowns"], "cooldowns")
    add = request.get("add")
    added_blocks: dict[str, dict] = {}
    if "add" in request:
        if not isinstance(add, dict) or not add:
            raise ScheduleRefused("add must name at least one environment")
        for name, entry in add.items():
            if not isinstance(entry, dict) or set(entry) != {"environment", "catalogue", "attestation"}:
                raise ScheduleRefused(f"add.{name} needs exactly environment, catalogue and attestation")
            if not isinstance(entry["attestation"], str) or not _ATTESTATION_NAME.fullmatch(entry["attestation"]):
                raise ScheduleRefused(f"add.{name}.attestation must be a plain file name of the request folder")
            added_blocks[name] = {"environment": entry["environment"], "catalogue": entry["catalogue"]}
    if active is None and shares is None and cooldowns is None and not added_blocks:
        raise ScheduleRefused("the request changes nothing")
    advice = contract.advice_policy
    for env, windows in (cooldowns or {}).items():
        if (env in contract.environments or env in added_blocks) and not advice["min_windows"] <= windows <= advice["max_windows"]:
            raise ScheduleRefused(f"cooldown {windows} of {env} is outside the contract's bounds "
                                  f"[{advice['min_windows']}, {advice['max_windows']}]")
    try:
        changed = next_schedule(contract, current, active=active, shares=shares, cooldowns=cooldowns,
                                add=added_blocks or None)
    except ValueError as exc:
        raise ScheduleRefused(str(exc)) from exc
    for name, block in added_blocks.items():
        # The entry's own cooldown, unless the request's ``cooldowns`` (checked above) sets it.
        windows = block["environment"]["cooldown_windows"]
        if name not in (cooldowns or {}) and not advice["min_windows"] <= windows <= advice["max_windows"]:
            raise ScheduleRefused(f"cannot add {name}: its cooldown {windows} is outside the contract's bounds "
                                  f"[{advice['min_windows']}, {advice['max_windows']}]")
    # Every added env is evaluated before deciding: any final refusal wins over any wait, and names its env.
    final: ScheduleRefused | None = None
    waits: list[str] = []
    attestations = {}
    for name, block in added_blocks.items():
        if catalogue_check is None or read_attestation is None:
            raise ScheduleRefused(f"cannot add {name}: this process cannot check an added environment")
        try:
            raw, attestations[name] = read_attestation(add[name]["attestation"])
        except OSError as exc:
            if exc.errno in TRANSIENT_READ_ERRNOS:
                waits.append(f"cannot add {name} yet: its attestation cannot be read now ({exc})")
                continue
            final = final or ScheduleRefused(f"cannot add {name}: its attestation cannot be read ({exc})")
            continue
        except ValueError as exc:
            final = final or ScheduleRefused(f"cannot add {name}: its attestation cannot be read ({exc})")
            continue
        if hashlib.sha256(raw).hexdigest() != block["catalogue"].get("conformance_sha256"):
            final = final or ScheduleRefused(f"cannot add {name}: the attestation file is not the one the entry "
                                             f"names (conformance_sha256)")
    if final is not None:
        raise final
    for env in changed.active_environments():
        # An env added by this request is checked by ``catalogue_check``; one added earlier and switched on
        # now is checked like any env, against the view's entry.
        if env in current.active_environments() or env in added_blocks:
            continue
        installed = installed_version(env)
        if installed is None:
            raise ScheduleRefused(f"cannot activate {env}: the environment is not installed on this validator")
        if installed != contract.environments[env]["version"]:
            raise ScheduleRefused(f"cannot activate {env}: the installed version differs from the version pinned "
                                  f"by the order")
    # Last: the activation checks are the costly ones and may answer "not yet"; every final refusal above wins
    # over a wait, and all of them run (an env that cannot be settled yet never hides another's refusal).
    for name, block in added_blocks.items():
        if name not in attestations:
            continue                        # its attestation is waiting already
        try:
            catalogue_check(name, block, attestations[name])
        except ScheduleNotYet as exc:
            waits.append(_naming(name, str(exc), "cannot add {name} yet: "))
        except ScheduleRefused as exc:
            final = final or ScheduleRefused(_naming(name, str(exc), "cannot add {name}: "))
    if final is not None:
        raise final
    if waits:
        raise ScheduleNotYet("; ".join(waits))
    return changed


def _naming(name: str, reason: str, prefix: str) -> str:
    """``reason``, prefixed so that it names ``name`` when it does not already."""
    return reason if name in reason else prefix.format(name=name) + reason


# ---------------------------------------------------------------- reading the runtime database

def _open_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise SystemExit(f"error: no runtime database at {db_path} (use --folder/--db of the running validator)")
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def read_runtime(db: sqlite3.Connection) -> dict:
    """The order's contract, latest schedule, request log and cooldown advice (read-only)."""
    row = db.execute("SELECT id, contract FROM service_orders LIMIT 1").fetchone()
    if row is None:
        raise SystemExit("error: the runtime database holds no order yet")
    contract = ServiceContract.from_dict(json.loads(row[1]))
    latest = db.execute(
        "SELECT payload FROM service_schedules WHERE order_id=? ORDER BY revision DESC LIMIT 1", (row[0],)).fetchone()
    if latest is None:
        raise SystemExit("error: the runtime database holds no schedule yet (the validator has not started this order)")
    schedule = ServiceSchedule.from_dict(json.loads(latest[0]), contract)
    requests = {}
    with contextlib.suppress(sqlite3.OperationalError):
        for rid, status, detail, revision, window in db.execute(
                "SELECT request_id, status, detail, revision, window FROM service_schedule_requests WHERE order_id=?",
                (row[0],)):
            requests[rid] = {"status": status, "detail": detail, "revision": revision, "window": window}
    advice = {}
    with contextlib.suppress(sqlite3.OperationalError):
        advice = {env: json.loads(payload) for env, payload in
                  db.execute("SELECT environment, payload FROM service_cooldown_advice")}
    return {"contract": contract, "schedule": schedule, "requests": requests, "advice": advice}


def _pairs(values: list[str] | None, option: str) -> dict[str, int] | None:
    if not values:
        return None
    result = {}
    for item in values:
        name, _, number = item.partition("=")
        if not name or not (number.isascii() and number.isdigit()):
            raise SystemExit(f"error: {option} expects env=integer, got {item!r}")
        if name in result:
            raise SystemExit(f"error: {option} given twice for {name}")
        result[name] = int(number)
    return result


def _unhandled_request(store: ScheduleRequestStore, known: dict):
    """The request file's content if the runtime has neither applied nor refused its id, else None.

    A file that cannot be read safely raises ValueError, except for content problems, which are
    returned as ``{"unreadable": reason}`` (an unsafe owner or mode is always refused) (the validator will refuse it too; ``--replace`` clears it).
    """
    try:
        pending = store.take()
    except UnsafeLocation:
        raise
    except ValueError as exc:
        return {"unreadable": str(exc)}
    if pending is None:
        return None
    if isinstance(pending, dict) and pending.get("request_id") in known:
        return None
    return pending


def _local_catalogue_check(name: str, block: dict, attestation) -> None:
    """The CLI's part of the activation checks: the attestation (the validator checks the package allowlist,
    the install and the machines again, and is the authority)."""
    from reliquary.validator.catalogue_activation import attestation_problem

    problem = attestation_problem(attestation, block)
    if problem is not None:
        raise ScheduleRefused(f"cannot add {name}: the attestation: {problem}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m reliquary.services.schedule", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=os.environ.get(FOLDER_ENV),
                        help=f"operator-owned request folder (or ${FOLDER_ENV}); required")
    parser.add_argument("--db", help=f"runtime SQLite (default <folder>/{DB_NAME}), opened read-only")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print the schedule, the pending request, its status and the cooldown advice")
    setter = sub.add_parser("set", help="write a request, applied by the validator at the next window")
    setter.add_argument("--active", help="comma-separated environments that run (needs --share for each)")
    setter.add_argument("--share", action="append", help="env=bps (active shares sum to 10000)")
    setter.add_argument("--cooldown", action="append", help="env=windows")
    setter.add_argument("--replace", action="store_true",
                        help="overwrite a pending request the validator has not handled yet (it is printed)")
    adder = sub.add_parser("add", help="add a bridged signed-episode env to the running run (from the next "
                                       "window; add it inactive, let miners see it announced, then `set --active`)")
    adder.add_argument("name")
    adder.add_argument("--entry", required=True, help="JSON file: the env's added block (bridged_episode describe)")
    adder.add_argument("--attestation", required=True, help="the conformance report's file name in --folder")
    adder.add_argument("--active", help="comma-separated environments that run (needs --share for each)")
    adder.add_argument("--share", action="append", help="env=bps (active shares sum to 10000)")
    adder.add_argument("--replace", action="store_true", help="overwrite a pending, unhandled request")
    args = parser.parse_args(argv)
    if not args.folder:
        raise SystemExit(f"error: --folder (or ${FOLDER_ENV}) is required")
    folder = Path(args.folder)
    store = ScheduleRequestStore(folder)
    state = read_runtime(_open_readonly(Path(args.db) if args.db else folder / DB_NAME))
    contract, schedule = state["contract"], state["schedule"]
    if args.command in ("set", "add"):
        active = None if args.active is None else [x for x in args.active.split(",") if x]
        add = None
        if args.command == "add":
            try:
                entry = strict_json_loads(Path(args.entry).read_bytes())
            except (OSError, ValueError) as exc:
                raise SystemExit(f"error: --entry cannot be read as strict JSON: {exc}") from exc
            if not isinstance(entry, dict):
                raise SystemExit("error: --entry must hold a JSON object (the env's added block)")
            add = {args.name: {**entry, "attestation": args.attestation}}
        request = build_request(order_sha256=contract.sha256, revision=schedule.revision + 1, active=active,
                                shares=_pairs(args.share, "--share"),
                                cooldowns=_pairs(getattr(args, "cooldown", None), "--cooldown"), add=add)
        # Checked against the order with the envs the latest schedule added, so an env added earlier can be
        # switched on, given a share or a cooldown.
        added = schedule.added_environments()
        view = schedule.contract_view(contract)

        def installed(name):
            # An env added earlier: the validator checks its install at the boundary.
            return added[name]["environment"]["version"] if name in added else default_installed_version(name)

        try:
            check_request(request, contract=view, current=schedule, installed_version=installed,
                          catalogue_check=_local_catalogue_check, read_attestation=store.read_attestation)
        except ValueError as exc:
            print(f"error: request refused: {exc}", file=sys.stderr)
            raise SystemExit(2)
        try:
            unhandled = _unhandled_request(store, state["requests"])
        except ValueError as exc:
            print(f"error: request refused: {exc}", file=sys.stderr)
            raise SystemExit(2)
        if unhandled is not None and not args.replace:
            print("error: request refused: a pending request has not been handled by the validator yet "
                  f"(id {unhandled.get('request_id') if isinstance(unhandled, dict) else None}); "
                  "use --replace to overwrite it. Pending request: "
                  + json.dumps(unhandled, sort_keys=True), file=sys.stderr)
            raise SystemExit(2)
        if unhandled is not None:
            print("replacing the unhandled request: " + json.dumps(unhandled, sort_keys=True), file=sys.stderr,
                  flush=True)
        store.write(request)
        out = {"written": request, "applies": "at the next window boundary"}
        if unhandled is not None:
            out["replaced"] = unhandled
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    try:
        pending = store.take()
    except ValueError as exc:
        pending = {"unreadable": str(exc)}
    try:
        last = store.status()
    except ValueError as exc:
        last = {"unreadable": str(exc)}
    if isinstance(pending, dict) and pending.get("request_id") in state["requests"]:
        pending = None  # already handled by the validator
    print(json.dumps({"schedule": schedule.to_dict(), "pending_request": pending, "last_request": last,
                      "cooldown_advice": state["advice"],
                      "cooldown_advice_note": "recommendation only, never applied automatically"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
