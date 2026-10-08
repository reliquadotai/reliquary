"""Live operator control of a v2 service order's schedule: no restart, applied at the next window.

The operator writes a request file into a local directory they own; the validator picks it up at a
window boundary (``ServiceRuntime.apply_pending_schedule_request``), validates it again (the file
store is not trusted) and appends the next schedule revision. The running window keeps its frozen
envelope (R6). No network endpoint.

    python -m reliquary.services.schedule --folder DIR show
    python -m reliquary.services.schedule --folder DIR set --active a,b --share a=6000 --share b=4000
    python -m reliquary.services.schedule --folder DIR set --cooldown a=200

``--folder`` (or ``RELIQUARY_SERVICE_SCHEDULE_FOLDER``) is required: there is no default location.
The runtime database is read read-only, from ``<folder>/runtime.sqlite3`` unless ``--db`` is given.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

from reliquary.protocol.service_contract import ServiceContract, _identifier
from reliquary.protocol.service_schedule import ServiceSchedule, next_schedule
from reliquary.shared.strict_json import strict_json_loads
from reliquary.validator.control import write_json

REQUEST_SCHEMA = "service-schedule-request/v1"
REQUEST_FIELDS = {"schema", "request_id", "order_sha256", "revision", "active", "shares", "cooldowns"}
MAX_FILE_BYTES = 64 * 1024
FOLDER_ENV = "RELIQUARY_SERVICE_SCHEDULE_FOLDER"
DB_NAME = "runtime.sqlite3"


class ScheduleRefused(ValueError):
    """A schedule request that must not be applied; the message is shown to the operator."""


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
            raise ValueError(f"{name} cannot be opened safely ({exc.strerror})") from exc
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{name} is not a regular file")
        _check_owner_and_mode(info, name)
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"{name} is larger than {MAX_FILE_BYTES} bytes")
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"{name} is larger than {MAX_FILE_BYTES} bytes")
    try:
        return strict_json_loads(raw)
    except ValueError as exc:  # includes bad UTF-8, duplicate keys, NaN
        raise ValueError(f"{name} is not strict JSON: {exc}") from exc


def build_request(*, order_sha256: str, revision: int, active=None, shares=None, cooldowns=None) -> dict:
    return {"schema": REQUEST_SCHEMA, "request_id": uuid.uuid4().hex,
            "order_sha256": order_sha256, "revision": revision,
            "active": None if active is None else sorted(active),
            "shares": None if shares is None else dict(shares),
            "cooldowns": None if cooldowns is None else dict(cooldowns)}


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
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    return getattr(ENVIRONMENT_SPECS.get(name), "environment_manifest_sha256", None)


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
                  installed_version: Callable[[str], str | None] | None = None) -> ServiceSchedule:
    """The schedule that ``request`` asks for, or ``ScheduleRefused``/``ValueError`` with the reason.

    Checks: schema and exact fields; the order hash is the running order's; ``revision`` is exactly
    ``current + 1`` (no replay, no gap); only declared envs; active shares sum to 10000; cooldowns
    within the contract's advice bounds; and, for each env that becomes active, R7: installed, and
    installed version equal to the pinned one. Nothing is mutated.
    """
    installed_version = installed_version or default_installed_version
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise ScheduleRefused(f"a schedule request has exactly the fields {sorted(REQUEST_FIELDS)}")
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
    if active is None and shares is None and cooldowns is None:
        raise ScheduleRefused("the request changes nothing")
    advice = contract.advice_policy
    for env, windows in (cooldowns or {}).items():
        if env in contract.environments and not advice["min_windows"] <= windows <= advice["max_windows"]:
            raise ScheduleRefused(f"cooldown {windows} of {env} is outside the contract's bounds "
                                  f"[{advice['min_windows']}, {advice['max_windows']}]")
    try:
        changed = next_schedule(contract, current, active=active, shares=shares, cooldowns=cooldowns)
    except ValueError as exc:
        raise ScheduleRefused(str(exc)) from exc
    for env in changed.active_environments():
        if env in current.active_environments():
            continue
        installed = installed_version(env)
        if installed is None:
            raise ScheduleRefused(f"cannot activate {env}: the environment is not installed on this validator")
        if installed != contract.environments[env]["version"]:
            raise ScheduleRefused(f"cannot activate {env}: the installed version differs from the version pinned "
                                  f"by the order")
    return changed


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
    args = parser.parse_args(argv)
    if not args.folder:
        raise SystemExit(f"error: --folder (or ${FOLDER_ENV}) is required")
    folder = Path(args.folder)
    store = ScheduleRequestStore(folder)
    state = read_runtime(_open_readonly(Path(args.db) if args.db else folder / DB_NAME))
    contract, schedule = state["contract"], state["schedule"]
    if args.command == "set":
        active = None if args.active is None else [x for x in args.active.split(",") if x]
        request = build_request(order_sha256=contract.sha256, revision=schedule.revision + 1, active=active,
                                shares=_pairs(args.share, "--share"), cooldowns=_pairs(args.cooldown, "--cooldown"))
        try:
            check_request(request, contract=contract, current=schedule)
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
