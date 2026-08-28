#!/usr/bin/env python3
"""Paced, dependency-free load test for the local ctrl-01 snapshot edge."""

from __future__ import annotations

import argparse
import http.client
import json
import queue
import statistics
import threading
import time
import urllib.parse


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081/state")
    parser.add_argument("--requests", type=int, default=15_000)
    parser.add_argument("--rate", type=float, default=750.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--allowed-status",
        action="append",
        type=int,
        default=[],
        help="May be repeated; defaults to only 200",
    )
    args = parser.parse_args()
    if args.requests <= 0 or args.rate <= 0 or args.concurrency <= 0:
        raise SystemExit("requests, rate, and concurrency must be positive")
    allowed_statuses = set(args.allowed_status or [200])

    parsed = urllib.parse.urlsplit(args.url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise SystemExit("only an explicit http URL is supported")
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    port = parsed.port or 80
    work: queue.Queue[int | None] = queue.Queue(maxsize=args.concurrency * 4)
    lock = threading.Lock()
    latencies_ms: list[float] = []
    statuses: dict[int, int] = {}
    failures: list[str] = []
    response_bytes = 0

    def worker() -> None:
        nonlocal response_bytes
        connection: http.client.HTTPConnection | None = None
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                break
            started = time.perf_counter()
            try:
                if connection is None:
                    connection = http.client.HTTPConnection(parsed.hostname, port, timeout=args.timeout)
                connection.request(
                    "GET",
                    path,
                    headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
                )
                response = connection.getresponse()
                body = response.read()
                status = response.status
                if status == 200 and response.getheader("Content-Encoding") != "gzip":
                    raise RuntimeError("state response was not served from the gzip artifact")
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with lock:
                    latencies_ms.append(elapsed_ms)
                    statuses[status] = statuses.get(status, 0) + 1
                    response_bytes += len(body)
            except Exception as exc:  # load diagnostics must retain all failures
                if connection is not None:
                    connection.close()
                    connection = None
                with lock:
                    failures.append(type(exc).__name__)
            finally:
                work.task_done()
        if connection is not None:
            connection.close()

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for thread in workers:
        thread.start()

    started = time.perf_counter()
    interval = 1.0 / args.rate
    for index in range(args.requests):
        target_time = started + index * interval
        remaining = target_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        work.put(index)
    for _ in workers:
        work.put(None)
    work.join()
    elapsed = time.perf_counter() - started
    for thread in workers:
        thread.join(timeout=1.0)

    report = {
        "requests_scheduled": args.requests,
        "requests_completed": len(latencies_ms),
        "failures": len(failures),
        "failure_types": {name: failures.count(name) for name in sorted(set(failures))},
        "statuses": statuses,
        "elapsed_seconds": round(elapsed, 3),
        "achieved_requests_per_second": round(len(latencies_ms) / elapsed, 2),
        "response_megabytes": round(response_bytes / 1_000_000, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
            "p50": round(percentile(latencies_ms, 0.50), 3),
            "p95": round(percentile(latencies_ms, 0.95), 3),
            "p99": round(percentile(latencies_ms, 0.99), 3),
            "max": round(max(latencies_ms), 3) if latencies_ms else 0.0,
        },
    }
    print(json.dumps(report, sort_keys=True))
    observed_statuses = set(statuses)
    return (
        0
        if not failures
        and sum(statuses.values()) == args.requests
        and observed_statuses <= allowed_statuses
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
