#!/usr/bin/env python3
"""Run one bound mTLS execution against a Reliquary CPU executor."""

from __future__ import annotations

import argparse
import json

from reliquary.environment.grader.executor import (
    RemoteSandboxExecutor,
    grader_runtime_id,
    make_sandbox_batch_request,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", help="https:// private CPU executor endpoint")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--runtime-id", default=grader_runtime_id())
    args = parser.parse_args()

    executor = RemoteSandboxExecutor(
        args.endpoint,
        runtime_id=args.runtime_id,
        ca_cert=args.ca,
        client_cert=args.cert,
        client_key=args.key,
    )
    try:
        request = make_sandbox_batch_request(
            runtime_id=args.runtime_id,
            code="def reliquary_smoke(a, b): return a + b",
            cases=[
                {
                    "entry": {"kind": "function", "name": "reliquary_smoke"},
                    "args": [20, 22],
                    "kwargs": {},
                }
            ],
            timeout_s=5.0,
        )
        result = executor.execute(request)
        case = result.results[0]
        if case.status != "ok" or case.output != 42:
            raise SystemExit(
                f"smoke execution failed: status={case.status!r} output={case.output!r}"
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "job_id": result.job_id,
                    "runtime_id": result.runtime_id,
                    "executor_id": result.executor_id,
                    "wall_ms": result.wall_ms,
                    "transport": executor.health_snapshot(),
                },
                sort_keys=True,
            )
        )
    finally:
        executor.close()


if __name__ == "__main__":
    main()
