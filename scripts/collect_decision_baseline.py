#!/usr/bin/env python3
"""Bounded external observations; no application restart or private credentials."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import uuid

LOG = re.compile(
    r"train_step|trainer_step|trainer_publish|checkpoint.*publish|submitted window=|discard|skipping|generated .*prompt|checkpoint.*loaded|miner_generation",
    re.I,
)


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return dict(
            returncode=r.returncode,
            stdout=r.stdout[-262144:],
            stderr=r.stderr[-262144:],
            truncated=len(r.stdout) + len(r.stderr) > 262144,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=type(exc).__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--health-url")
    p.add_argument("--seconds", type=int, default=172800)
    a = p.parse_args()
    os.umask(0o077)
    root = Path(a.directory)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "collector.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = uuid.uuid4().hex
    deadline = time.monotonic() + a.seconds
    previous = int(time.time())
    while time.monotonic() < deadline:
        now = time.time()
        event = dict(
            event="external_sample",
            run_id=run,
            time=now,
            since=previous,
            container=a.container,
            coverage="external logs only; not full candidate or optimizer lineage",
        )
        logs = command(
            [
                "docker",
                "logs",
                "--timestamps",
                "--tail",
                "1000",
                "--since",
                str(previous),
                a.container,
            ]
        )
        raw = logs.pop("stdout", "") + logs.pop("stderr", "")
        event["logs"] = {
            **logs,
            "tail_limit_reached": len(raw.splitlines()) >= 1000,
            "lines": [line for line in raw.splitlines() if LOG.search(line)],
        }
        event["runtime"] = command(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Id}} {{.Image}} {{.State.StartedAt}} {{.State.Running}}",
                a.container,
            ]
        )
        if not a.health_url:
            event["gpu"] = command(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
        else:
            try:
                with urllib.request.urlopen(a.health_url, timeout=5) as response:
                    health = json.load(response)
                event["health"] = {
                    k: v
                    for k, v in health.items()
                    if k
                    in {
                        "status",
                        "active_window",
                        "image_revision",
                        "queue_depth_by_environment",
                        "recent_reject_counts_by_reason",
                        "utility_telemetry",
                        "training_accumulator_counts",
                        "archive_last_enqueued_window",
                        "archive_queue_depth",
                    }
                }
            except Exception as exc:
                event["health_error"] = type(exc).__name__
        name = datetime.fromtimestamp(now, timezone.utc).strftime(
            "baseline-%Y%m%d-%H.jsonl"
        )
        path = root / name
        if path.exists() and path.stat().st_size > 16_777_216:
            # ponytail: hourly cap; record overflow, expand only after measured need.
            event = dict(event="collector_hourly_cap", time=now)
            path = root / "overflow.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
        for old in sorted(root.glob("baseline-*.jsonl"))[:-48]:
            old.unlink()
        previous = int(
            now
        )  # second overlap is intentional; deduplicate log timestamps offline.
        time.sleep(10)


if __name__ == "__main__":
    main()
