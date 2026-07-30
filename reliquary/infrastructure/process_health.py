"""Secret-free process lifecycle telemetry for long-lived validators.

The production trainer runs with ``cgroup: host`` so the unified cgroup path
must be resolved from ``/proc/self/cgroup`` instead of assuming
``/sys/fs/cgroup/pids.current`` exists at the mount root.  Collection is kept
off the request path by :class:`~reliquary.validator.server.ValidatorServer`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_GRADER_HEALTH_PATH = "/tmp/reliquary-grader-health.json"
PID_USAGE_WARNING_FRACTION = 0.40
PID_USAGE_RESTART_FRACTION = 0.50
ZOMBIE_WARNING_COUNT = 128
GRADER_HEALTH_STALE_SECONDS = 120.0

_GRADER_FIELDS = frozenset(
    {
        "server_pid",
        "started_at",
        "updated_at",
        "shutdown_complete",
        "pool_size",
        "workers_alive",
        "workers_idle",
        "workers_busy",
        "workers_spawned_total",
        "worker_restarts_total",
        "worker_recycles_total",
        "worker_terminations_total",
        "worker_reaped_total",
        "worker_reap_failures_total",
        "container_deletes_total",
        "container_delete_failures_total",
    }
)
_KNOWN_INIT_NAMES = frozenset({"docker-init", "tini", "dumb-init"})


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _parse_limit(value: str) -> int | None:
    if value == "max":
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _unified_cgroup_path(proc_root: Path) -> str | None:
    try:
        for line in _read_text(proc_root / "self" / "cgroup").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                return path
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _cgroup_pids(
    proc_root: Path,
    cgroup_root: Path,
) -> tuple[int | None, int | None]:
    relative = _unified_cgroup_path(proc_root)
    candidates: list[Path] = []
    if relative is not None:
        candidates.append(cgroup_root / relative.lstrip("/"))
    candidates.append(cgroup_root)

    for candidate in candidates:
        try:
            current = int(_read_text(candidate / "pids.current"))
            maximum = _parse_limit(_read_text(candidate / "pids.max"))
            return current, maximum
        except (FileNotFoundError, OSError, ValueError):
            continue
    return None, None


def _process_counts(proc_root: Path) -> tuple[int, int]:
    processes = 0
    zombies = 0
    try:
        entries = os.scandir(proc_root)
    except OSError:
        return processes, zombies
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            processes += 1
            try:
                stat = _read_text(Path(entry.path) / "stat")
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue
            # ``comm`` is parenthesized and may contain spaces or ``)``.
            close = stat.rfind(")")
            if close >= 0 and stat[close + 2 : close + 3] == "Z":
                zombies += 1
    return processes, zombies


def _grader_snapshot(path: Path, now: float) -> dict[str, Any]:
    try:
        raw = json.loads(_read_text(path))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    safe = {key: raw[key] for key in _GRADER_FIELDS if key in raw}
    updated_at = safe.get("updated_at")
    if isinstance(updated_at, (int, float)):
        safe["age_seconds"] = round(max(0.0, now - float(updated_at)), 3)
        safe["stale"] = safe["age_seconds"] > GRADER_HEALTH_STALE_SECONDS
    else:
        safe["age_seconds"] = None
        safe["stale"] = True
    return safe


def collect_process_health(
    *,
    proc_root: str | os.PathLike[str] = "/proc",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    grader_health_path: str | os.PathLike[str] = DEFAULT_GRADER_HEALTH_PATH,
    now: float | None = None,
) -> dict[str, Any]:
    """Collect one bounded, JSON-safe process snapshot.

    No command lines, environment variables, container ids, or filesystem
    paths are returned.  This keeps the public validator ``/health`` endpoint
    useful without exposing operator secrets.
    """

    collected_at = time.time() if now is None else float(now)
    proc = Path(proc_root)
    cgroup = Path(cgroup_root)
    process_count, zombie_count = _process_counts(proc)
    pids_current, pids_max = _cgroup_pids(proc, cgroup)

    try:
        pid1_name = _read_text(proc / "1" / "comm")
    except (FileNotFoundError, PermissionError, OSError):
        pid1_name = None
    init_present = (
        pid1_name in _KNOWN_INIT_NAMES if pid1_name is not None else None
    )

    utilization = (
        float(pids_current) / float(pids_max)
        if pids_current is not None and pids_max not in (None, 0)
        else None
    )
    restart_recommended = (
        utilization is not None
        and utilization >= PID_USAGE_RESTART_FRACTION
    )

    grader = _grader_snapshot(Path(grader_health_path), collected_at)
    warning_reasons: list[str] = []
    if init_present is False:
        warning_reasons.append("init_subreaper_missing")
    if zombie_count >= ZOMBIE_WARNING_COUNT:
        warning_reasons.append("zombie_count_high")
    if (
        utilization is not None
        and utilization >= PID_USAGE_WARNING_FRACTION
    ):
        warning_reasons.append("cgroup_pid_usage_high")
    if grader and grader.get("stale") is True:
        warning_reasons.append("grader_health_stale")

    collector_available = (
        pid1_name is not None
        or process_count > 0
        or pids_current is not None
    )
    status = "ok" if collector_available else "unknown"
    if restart_recommended:
        status = "critical"
    elif warning_reasons and collector_available:
        status = "warning"

    return {
        "status": status,
        "collected_at": collected_at,
        "pid_namespace_processes": process_count,
        "zombie_processes": zombie_count,
        "pid1_name": pid1_name,
        "init_subreaper_present": init_present,
        "cgroup_pids_current": pids_current,
        "cgroup_pids_max": pids_max,
        "cgroup_pids_utilization": (
            round(utilization, 6) if utilization is not None else None
        ),
        "pids_warning_fraction": PID_USAGE_WARNING_FRACTION,
        "pids_restart_fraction": PID_USAGE_RESTART_FRACTION,
        "restart_recommended": restart_recommended,
        "warning_reasons": warning_reasons,
        "grader": grader,
    }
