#!/usr/bin/env python3
"""Malicious corpus gate for a dedicated, disposable cpu-exec-01 only.

The corpus proves the current Python policy and gVisor lifecycle fail closed.
It intentionally does not claim kernel-escape resistance; host and network
checks in the deployment playbook cover the surrounding trust boundary.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from reliquary.environment.grader.executor import (
    RemoteSandboxExecutor,
    make_sandbox_batch_request,
)


CORPUS = {
    "host_file": "def attack():\n import pathlib\n return pathlib.Path('/etc/shadow').read_text()",
    "proc_environ": "def attack():\n return open('/proc/self/environ').read()",
    "socket": "def attack():\n import socket\n return socket.create_connection(('127.0.0.1', 22)).recv(1)",
    "subprocess": "def attack():\n import subprocess\n return subprocess.check_output(['id']).decode()",
    "ctypes": "def attack():\n import ctypes\n return ctypes.CDLL(None).getuid()",
    "fork": "def attack():\n import os\n return os.fork()",
    "dynamic_import": "def attack():\n return __import__('os').getuid()",
    "builtin_open": "def attack():\n return open('/etc/passwd').read()",
    "introspection": "def attack():\n return globals()",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument(
        "--confirm-dedicated-host",
        action="store_true",
        help="required acknowledgement that this targets only cpu-exec-01",
    )
    args = parser.parse_args()
    if not args.confirm_dedicated_host:
        parser.error("--confirm-dedicated-host is required")

    tls = httpx.create_ssl_context(verify=args.ca, cert=(args.cert, args.key))
    with httpx.Client(verify=tls, timeout=10.0) as client:
        before = client.get(f"{args.endpoint.rstrip('/')}/v1/health")
        before.raise_for_status()
        runtime_id = str(before.json()["runtime_id"])

    executor = RemoteSandboxExecutor(
        args.endpoint,
        runtime_id=runtime_id,
        ca_cert=args.ca,
        client_cert=args.cert,
        client_key=args.key,
    )
    results: dict[str, str] = {}
    try:
        for index, (name, code) in enumerate(CORPUS.items()):
            request = make_sandbox_batch_request(
                runtime_id=runtime_id,
                code=code,
                cases=[{
                    "entry": {"kind": "function", "name": "attack"},
                    "args": [],
                    "kwargs": {},
                }],
                timeout_s=2.0,
                attempt=index,
            )
            result = executor.execute(request)
            case = result.results[0]
            if case.status == "ok":
                raise RuntimeError(f"attack unexpectedly succeeded: {name}")
            results[name] = case.status

        timeout_request = make_sandbox_batch_request(
            runtime_id=runtime_id,
            code="def attack():\n while True:\n  pass",
            cases=[{
                "entry": {"kind": "function", "name": "attack"},
                "args": [],
                "kwargs": {},
            }],
            timeout_s=1.0,
            attempt=len(CORPUS),
        )
        timeout_result = executor.execute(timeout_request)
        if timeout_result.results[0].status != "timeout":
            raise RuntimeError("infinite loop did not hit the sandbox timeout")
        results["cpu_timeout"] = "timeout"

        smoke = make_sandbox_batch_request(
            runtime_id=runtime_id,
            code="def clean(): return 42",
            cases=[{
                "entry": {"kind": "function", "name": "clean"},
                "args": [],
                "kwargs": {},
            }],
            timeout_s=2.0,
            attempt=len(CORPUS) + 1,
        )
        smoke_result = executor.execute(smoke)
        smoke_case = smoke_result.results[0]
        if smoke_case.status != "ok" or smoke_case.output != 42:
            raise RuntimeError("clean job failed after malicious corpus")
    finally:
        executor.close()

    time.sleep(1.0)
    with httpx.Client(verify=tls, timeout=10.0) as client:
        after = client.get(f"{args.endpoint.rstrip('/')}/v1/health")
        after.raise_for_status()
    pool = after.json()["pool"]
    if pool["workers_alive"] != pool["pool_size"]:
        raise RuntimeError("worker pool did not recover after malicious corpus")
    print(json.dumps({"status": "passed", "corpus": results, "health": after.json()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
