#!/usr/bin/env python3
"""Bounded mTLS capacity test for a dedicated Reliquary CPU executor."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import ssl
import statistics
import time

import httpx

from reliquary.environment.grader.executor import (
    RemoteSandboxExecutor,
    make_sandbox_batch_request,
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("--ca", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-p95-ms", type=float, default=1000.0)
    args = parser.parse_args()
    if args.requests < 1 or args.parallel < 1:
        parser.error("requests and parallel must be positive")

    tls = ssl.create_default_context(cafile=args.ca)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(certfile=args.cert, keyfile=args.key)
    with httpx.Client(verify=tls, timeout=10.0, trust_env=False) as client:
        health = client.get(f"{args.endpoint.rstrip('/')}/v1/health")
        health.raise_for_status()
        runtime_id = str(health.json()["runtime_id"])

    executor = RemoteSandboxExecutor(
        args.endpoint,
        runtime_id=runtime_id,
        ca_cert=args.ca,
        client_cert=args.cert,
        client_key=args.key,
    )

    def one(index: int) -> tuple[bool, float, str | None]:
        request = make_sandbox_batch_request(
            runtime_id=runtime_id,
            code="def work(a, b): return (a * 31 + b) % 1000003",
            cases=[{
                "entry": {"kind": "function", "name": "work"},
                "args": [index, 7],
                "kwargs": {},
            }],
            timeout_s=args.timeout,
            attempt=index,
        )
        started = time.perf_counter()
        try:
            result = executor.execute(request)
            expected = (index * 31 + 7) % 1_000_003
            case = result.results[0]
            return (
                case.status == "ok" and case.output == expected,
                (time.perf_counter() - started) * 1000.0,
                None,
            )
        except Exception as exc:
            return False, (time.perf_counter() - started) * 1000.0, type(exc).__name__

    started = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(one, range(args.requests)))
    finally:
        executor.close()
    elapsed = time.perf_counter() - started
    latencies = [result[1] for result in results]
    failures = [result[2] or "wrong_result" for result in results if not result[0]]
    report = {
        "requests": args.requests,
        "parallel": args.parallel,
        "successful": args.requests - len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "requests_per_second": args.requests / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "runtime_id": runtime_id,
    }
    print(json.dumps(report, sort_keys=True))
    return int(bool(failures) or report["latency_ms"]["p95"] > args.max_p95_ms)


if __name__ == "__main__":
    raise SystemExit(main())
