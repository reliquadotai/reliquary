"""Grader worker — runs inside the gVisor sandbox.

Reads JSON requests from stdin (one per line), evaluates the
(code, tests) pair against a fresh module namespace, writes the
result back to stdout. Stays warm between requests so the gVisor
sandbox is reused.

Each request gets a fresh dict for `code`'s namespace, and each
test runs in a shallow copy of that namespace so per-test
mutations don't leak.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
from typing import Tuple


def evaluate_code(code: str, tests: list[str], timeout_s: float) -> Tuple[int, int, str]:
    """Run `code` to populate a namespace, then run each test in a deep copy.

    Returns (passed, total, status). The user's print() and any stdout
    writes are sent to a discarded buffer so they cannot pollute the
    IPC pipe shared with the grader server. `timeout_s` is informational;
    real wall-clock enforcement is in the parent's subprocess.run timeout.
    This function never raises.
    """
    total = len(tests)
    if not code or not code.strip():
        return 0, total, "ok"

    ns: dict = {}
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(code, "<miner_code>", "exec"), ns)
        except BaseException:
            # Must catch BaseException — miner code can raise SystemExit /
            # KeyboardInterrupt; we cannot let it terminate the worker.
            return 0, total, "ok"

        passed = 0
        for i, t in enumerate(tests):
            try:
                exec(compile(t, f"<test_{i}>", "exec"), copy.deepcopy(ns))
                passed += 1
            except BaseException:
                # Must catch BaseException — miner code can raise SystemExit /
                # KeyboardInterrupt; we cannot let it terminate the worker.
                pass
    return passed, total, "ok"


def _serve_stdin() -> None:
    """Read JSON-line requests, write JSON-line responses. Loop forever.

    Used when this module is run as a subprocess (`python -m
    reliquary.environment.grader.worker`). The parent process pipes
    requests via stdin and reads results from stdout.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            passed, total, status = evaluate_code(
                req.get("code", ""),
                req.get("tests", []),
                float(req.get("timeout_s", 5.0)),
            )
            resp = {
                "req_id": req.get("req_id", ""),
                "passed": passed,
                "total": total,
                "status": status,
            }
        except BaseException as e:
            resp = {
                "req_id": "",
                "passed": 0,
                "total": 0,
                "status": "crash",
                "error": str(e),
            }
        sys.__stdout__.write(json.dumps(resp) + "\n")
        sys.__stdout__.flush()


if __name__ == "__main__":
    _serve_stdin()
