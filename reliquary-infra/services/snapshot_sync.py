#!/usr/bin/env python3
"""Continuously publish immutable validator state snapshots for nginx.

The process has no wallet, database, Docker, or provider credentials.  Each
successful upstream response is validated as bounded JSON, precompressed, and
published by one atomic symlink replacement.  nginx therefore serves complete
identity/gzip pairs without entering the validator event loop.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final


USER_AGENT: Final = "reliquary-snapshot-sync/1"
MAX_STATE_BYTES: Final = 2 * 1024 * 1024
MAX_HEALTH_BYTES: Final = 1024 * 1024
KEEP_GENERATIONS: Final = 32


@dataclass(frozen=True)
class Target:
    name: str
    path: str
    interval_s: float
    stale_after_s: float
    max_bytes: int


@dataclass(frozen=True)
class FetchResult:
    status: int
    body: bytes | None
    elapsed_s: float
    error: str | None = None


class SnapshotPublisher:
    def __init__(self, root: Path, metrics_dir: Path) -> None:
        self.root = root
        self.metrics_dir = metrics_dir
        self.generations = root / "generations"
        self.current = root / "current"
        self.status = root / "status"
        for directory in (
            self.generations,
            self.current,
            self.status,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o755)

    def publish(self, target: Target, body: bytes) -> str:
        digest = hashlib.sha256(body).hexdigest()
        target_root = self.generations / target.name
        generation = target_root / digest
        if not generation.is_dir():
            target_root.mkdir(parents=True, exist_ok=True, mode=0o755)
            temporary_generation = target_root / f".{digest}.{os.getpid()}.tmp"
            shutil.rmtree(temporary_generation, ignore_errors=True)
            temporary_generation.mkdir(mode=0o755)
            self._write_new(temporary_generation / "payload.json", body, 0o644)
            self._write_new(
                temporary_generation / "payload.json.gz",
                gzip.compress(body, compresslevel=1, mtime=0),
                0o644,
            )
            try:
                os.rename(temporary_generation, generation)
            except FileExistsError:
                shutil.rmtree(temporary_generation, ignore_errors=True)

        link = self.current / target.name
        expected = Path("..") / "generations" / target.name / digest
        if not link.is_symlink() or os.readlink(link) != str(expected):
            temporary = self.current / f".{target.name}.{os.getpid()}.tmp"
            temporary.unlink(missing_ok=True)
            os.symlink(expected, temporary)
            os.replace(temporary, link)
        self._prune(target_root, keep=digest)
        return digest

    def retire(self, target: Target) -> None:
        (self.current / target.name).unlink(missing_ok=True)

    def record_status(
        self,
        target: Target,
        result: FetchResult,
        *,
        digest: str | None,
        last_success_wall: float | None,
        body_bytes: int,
        compressed_bytes: int,
    ) -> None:
        now = time.time()
        age = -1.0 if last_success_wall is None else max(0.0, now - last_success_wall)
        payload = {
            "target": target.name,
            "upstream_status": result.status,
            "fetch_seconds": round(result.elapsed_s, 6),
            "last_success_unix": last_success_wall,
            "age_seconds": round(age, 6),
            "published": (self.current / target.name).is_symlink(),
            "sha256": digest,
            "bytes": body_bytes,
            "compressed_bytes": compressed_bytes,
            "error": result.error,
            "updated_unix": now,
        }
        self._atomic_write(
            self.status / f"{target.name}.json",
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
            0o644,
        )

        label = target.name.replace("\\", "_").replace('"', "_")
        published = 1 if payload["published"] else 0
        success = 1 if result.status == 200 and result.error is None else 0
        metric = (
            "# HELP reliquary_snapshot_up Whether the snapshot is currently published.\n"
            "# TYPE reliquary_snapshot_up gauge\n"
            f'reliquary_snapshot_up{{target="{label}"}} {published}\n'
            "# HELP reliquary_snapshot_fetch_success Whether the last fetch succeeded.\n"
            "# TYPE reliquary_snapshot_fetch_success gauge\n"
            f'reliquary_snapshot_fetch_success{{target="{label}"}} {success}\n'
            "# HELP reliquary_snapshot_age_seconds Age of the last good snapshot.\n"
            "# TYPE reliquary_snapshot_age_seconds gauge\n"
            f'reliquary_snapshot_age_seconds{{target="{label}"}} {age}\n'
            "# HELP reliquary_snapshot_fetch_seconds Last upstream fetch duration.\n"
            "# TYPE reliquary_snapshot_fetch_seconds gauge\n"
            f'reliquary_snapshot_fetch_seconds{{target="{label}"}} {result.elapsed_s}\n'
            "# HELP reliquary_snapshot_bytes Last identity response size.\n"
            "# TYPE reliquary_snapshot_bytes gauge\n"
            f'reliquary_snapshot_bytes{{target="{label}"}} {body_bytes}\n'
            "# HELP reliquary_snapshot_compressed_bytes Last gzip response size.\n"
            "# TYPE reliquary_snapshot_compressed_bytes gauge\n"
            f'reliquary_snapshot_compressed_bytes{{target="{label}"}} {compressed_bytes}\n'
        ).encode()
        self._atomic_write(
            self.metrics_dir / f"reliquary_snapshot_{target.name}.prom",
            metric,
            0o644,
        )

    @staticmethod
    def _write_new(path: Path, content: bytes, mode: int) -> None:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)

    @staticmethod
    def _atomic_write(path: Path, content: bytes, mode: int) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                handle.write(content)
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _prune(target_root: Path, *, keep: str) -> None:
        generations = sorted(
            (item for item in target_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for item in generations[KEEP_GENERATIONS:]:
            if item.name != keep:
                shutil.rmtree(item, ignore_errors=True)


def fetch_json(url: str, *, timeout_s: float, max_bytes: int) -> FetchResult:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "Accept-Encoding": "identity", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        return FetchResult(
            status=int(exc.code),
            body=None,
            elapsed_s=time.monotonic() - started,
            error=f"http_{exc.code}",
        )
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return FetchResult(
            status=0,
            body=None,
            elapsed_s=time.monotonic() - started,
            error=type(exc).__name__,
        )

    if status != 200:
        return FetchResult(status=status, body=None, elapsed_s=time.monotonic() - started)
    if len(body) > max_bytes:
        return FetchResult(
            status=0,
            body=None,
            elapsed_s=time.monotonic() - started,
            error="response_too_large",
        )
    try:
        json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return FetchResult(
            status=0,
            body=None,
            elapsed_s=time.monotonic() - started,
            error="invalid_json",
        )
    return FetchResult(status=200, body=body, elapsed_s=time.monotonic() - started)


def target_loop(
    *,
    upstream: str,
    target: Target,
    publisher: SnapshotPublisher,
    stop: threading.Event,
    timeout_s: float,
) -> None:
    last_success_wall: float | None = None
    last_success_mono: float | None = None
    digest: str | None = None
    body_bytes = 0
    compressed_bytes = 0
    url = urllib.parse.urljoin(upstream.rstrip("/") + "/", target.path.lstrip("/"))
    next_fetch = time.monotonic()

    while not stop.is_set():
        result = fetch_json(url, timeout_s=timeout_s, max_bytes=target.max_bytes)
        now_mono = time.monotonic()
        if result.status == 200 and result.body is not None:
            digest = publisher.publish(target, result.body)
            last_success_wall = time.time()
            last_success_mono = now_mono
            body_bytes = len(result.body)
            compressed_bytes = len(gzip.compress(result.body, compresslevel=1, mtime=0))
        elif result.status in {404, 503}:
            publisher.retire(target)
        elif last_success_mono is None or now_mono - last_success_mono > target.stale_after_s:
            publisher.retire(target)

        publisher.record_status(
            target,
            result,
            digest=digest,
            last_success_wall=last_success_wall,
            body_bytes=body_bytes,
            compressed_bytes=compressed_bytes,
        )
        next_fetch += target.interval_s
        if next_fetch <= time.monotonic():
            next_fetch = time.monotonic() + target.interval_s
        stop.wait(max(0.0, next_fetch - time.monotonic()))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--root", type=Path, default=Path("/var/lib/reliquary-edge/snapshots"))
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("/var/lib/prometheus/node-exporter"),
    )
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--state-interval", type=float, default=0.5)
    parser.add_argument("--health-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--state-stale-after", type=float, default=3.0)
    parser.add_argument("--health-stale-after", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.upstream.startswith(("http://", "https://")):
        raise SystemExit("upstream must use http or https")
    for value in (
        args.state_interval,
        args.health_interval,
        args.timeout,
        args.state_stale_after,
        args.health_stale_after,
    ):
        if value <= 0:
            raise SystemExit("intervals, timeout, and stale limits must be positive")

    os.umask(0o027)
    publisher = SnapshotPublisher(args.root, args.metrics_dir)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    targets = [
        Target("state-default", "/state", args.state_interval, args.state_stale_after, MAX_STATE_BYTES),
        Target("health", "/health", args.health_interval, args.health_stale_after, MAX_HEALTH_BYTES),
        Target("checkpoint", "/checkpoint", args.state_interval, args.state_stale_after, MAX_HEALTH_BYTES),
        Target("runtime-contract", "/runtime-contract", args.state_interval, args.state_stale_after, MAX_HEALTH_BYTES),
    ]
    for environment in args.environment:
        if not environment or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in environment):
            raise SystemExit(f"invalid environment name: {environment!r}")
        query = urllib.parse.urlencode({"env": environment})
        targets.append(
            Target(
                f"state-{environment}",
                f"/state?{query}",
                args.state_interval,
                args.state_stale_after,
                MAX_STATE_BYTES,
            )
        )

    threads = [
        threading.Thread(
            target=target_loop,
            kwargs={
                "upstream": args.upstream,
                "target": target,
                "publisher": publisher,
                "stop": stop,
                "timeout_s": args.timeout,
            },
            name=f"snapshot-{target.name}",
            daemon=True,
        )
        for target in targets
    ]
    for thread in threads:
        thread.start()
    while not stop.wait(1.0):
        if any(not thread.is_alive() for thread in threads):
            stop.set()
            for thread in threads:
                thread.join(timeout=2.0)
            return 1
    for thread in threads:
        thread.join(timeout=max(args.timeout + 1.0, 2.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
