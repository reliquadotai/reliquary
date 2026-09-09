"""Local, durable operator requests. The state directory is the admin boundary."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from reliquary.shared.strict_json import strict_json_loads


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ControlStore:
    def __init__(self, state_dir: str | Path, *, start_closed: bool = False):
        self.path = Path(state_dir) / "control.json"
        self.status_path = self.path.with_name("control-status.json")
        self.start_closed = start_closed

    def request(self) -> dict:
        try:
            request = strict_json_loads(self.path.read_bytes())
        except FileNotFoundError:
            return {"schema_version": 1, "request_id": "startup", "mode":
                    "drain" if self.start_closed else "run", "target_cursor": None}
        if not isinstance(request, dict) or set(request) != {
            "schema_version", "request_id", "mode", "target_cursor",
        }:
            raise ValueError("invalid control request fields")
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("unsupported control request schema")
        if not isinstance(request["request_id"], str) or not request["request_id"]:
            raise ValueError("control request needs an identity")
        if request["mode"] not in {"run", "drain"}:
            raise ValueError("unsupported control mode")
        cursor = request["target_cursor"]
        if cursor is not None and (type(cursor) is not int or cursor < 0):
            raise ValueError("target cursor must be a non-negative integer")
        if request["mode"] == "run" and cursor is not None:
            raise ValueError("run cannot specify a drain cursor")
        return request

    def set_mode(self, mode: str, *, target_cursor: int | None = None) -> dict:
        if mode not in {"run", "drain"} or (
            target_cursor is not None and (
                mode != "drain" or type(target_cursor) is not int or target_cursor < 0
            )
        ):
            raise ValueError("invalid control mode or drain cursor")
        request = {"schema_version": 1, "request_id": uuid.uuid4().hex,
                   "mode": mode, "target_cursor": target_cursor}
        write_json(self.path, request)
        return request

    def report(self, request: dict, *, phase: str, **details) -> None:
        write_json(self.status_path, {"schema_version": 1,
                   "request_id": request["request_id"], "phase": phase,
                   "updated_at": time.time(), "pid": os.getpid(), **details})

    def status(self) -> dict:
        request = self.request()
        try:
            status = strict_json_loads(self.status_path.read_bytes())
        except FileNotFoundError:
            status = {}
        fresh = (status.get("request_id") == request["request_id"]
                 and 0 <= time.time() - status.get("updated_at", 0) <= 15)
        return {"request": request, "status": status, "fresh": fresh,
                "drained": fresh and status.get("phase") == "drained"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "drain", "status"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--target-cursor", type=int, help="trainer drain boundary, inclusive")
    args = parser.parse_args()
    store = ControlStore(args.state_dir, start_closed=os.getenv(
        "RELIQUARY_CONTROL_START_CLOSED", "0").lower() in {"1", "true", "yes", "on"})
    if args.mode == "status":
        if args.target_cursor is not None:
            parser.error("--target-cursor requires drain")
        print(json.dumps(store.status(), indent=2))
    else:
        print(json.dumps(store.set_mode(args.mode, target_cursor=args.target_cursor), indent=2))


if __name__ == "__main__":
    main()
