#!/usr/bin/env python3
"""Exercise grader recycling and fail on unreaped direct workers.

The default mode uses plain Python workers and is suitable for CI or a local
host.  ``--use-runsc`` exercises the production OCI bundle from inside the
privileged validator image.  A production soak can use, for example:

    python scripts/stress_grader_recycling.py \
      --use-runsc --evaluations 10000 --parallel 32 \
      --pool-size 32 --recycle-after 64 --max-zombie-delta 32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

from reliquary.environment.grader.server import (
    GraderServer,
    runsc_worker_argv,
)
from reliquary.infrastructure.process_health import collect_process_health


def _request(socket_path: str, index: int, timeout_s: float) -> dict[str, Any]:
    payload = {
        "req_id": f"soak-{index}",
        "code": "def add(a, b): return a + b",
        "cases": [
            {
                "entry": {"kind": "function", "name": "add"},
                "args": [index, 1],
                "kwargs": {},
                "expected": index + 1,
                "compare": "exact",
            }
        ],
        "timeout_s": timeout_s,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_s + 10.0)
        client.connect(socket_path)
        client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        response = b""
        while b"\n" not in response:
            chunk = client.recv(8192)
            if not chunk:
                break
            response += chunk
    return json.loads(response.split(b"\n", 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--recycle-after", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-zombie-delta", type=int, default=0)
    parser.add_argument("--use-runsc", action="store_true")
    parser.add_argument(
        "--bundle",
        default=os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        ),
    )
    args = parser.parse_args()

    if args.evaluations < 1 or args.parallel < 1 or args.pool_size < 1:
        parser.error("evaluations, parallel and pool-size must be positive")
    if args.recycle_after < 1:
        parser.error("recycle-after must be positive")

    worker_argv = (
        runsc_worker_argv(args.bundle)
        if args.use_runsc
        else [sys.executable, "-m", "reliquary.environment.grader.worker"]
    )
    before = collect_process_health()
    responses: list[dict[str, Any]] = []
    request_errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="grader-soak-", dir="/tmp") as tmp:
        socket_path = str(Path(tmp) / "grader.sock")
        health_path = str(Path(tmp) / "grader-health.json")
        server = GraderServer(
            socket_path=socket_path,
            pool_size=args.pool_size,
            worker_argv=worker_argv,
            eval_timeout_s=args.timeout,
            recycle_after_evals=args.recycle_after,
            metrics_port=0,
            health_path=health_path,
        )
        server.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel
            ) as executor:
                futures = [
                    executor.submit(
                        _request, socket_path, index, args.timeout
                    )
                    for index in range(args.evaluations)
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        responses.append(future.result())
                    except Exception as exc:
                        request_errors.append(type(exc).__name__)
        finally:
            server.stop()
        grader = server.health_snapshot()

    after = collect_process_health()
    zombie_delta = int(after["zombie_processes"]) - int(
        before["zombie_processes"]
    )
    successes = sum(
        response.get("status") == "ok" and response.get("passed") == 1
        for response in responses
    )
    report = {
        "mode": "runsc" if args.use_runsc else "python",
        "requested": args.evaluations,
        "successful": successes,
        "request_errors": request_errors,
        "zombies_before": before["zombie_processes"],
        "zombies_after": after["zombie_processes"],
        "zombie_delta": zombie_delta,
        "grader": grader,
    }
    print(json.dumps(report, sort_keys=True))

    direct_worker_leak = (
        grader["workers_spawned_total"] != grader["worker_reaped_total"]
        or grader["worker_reap_failures_total"] != 0
    )
    return int(
        successes != args.evaluations
        or bool(request_errors)
        or direct_worker_leak
        or zombie_delta > args.max_zombie_delta
    )


if __name__ == "__main__":
    raise SystemExit(main())
