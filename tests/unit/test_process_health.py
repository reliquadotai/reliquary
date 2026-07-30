from __future__ import annotations

import json
import subprocess
from pathlib import Path

from reliquary.infrastructure.process_health import collect_process_health

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_collect_process_health_resolves_unified_cgroup_and_zombies(tmp_path):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    grader = tmp_path / "grader-health.json"

    _write(proc / "self" / "cgroup", "0::/system.slice/trainer.scope\n")
    _write(proc / "1" / "comm", "docker-init\n")
    _write(proc / "1" / "stat", "1 (docker-init) S 0 0 0\n")
    _write(proc / "42" / "stat", "42 (runsc worker) Z 1 1 1\n")
    _write(proc / "43" / "stat", "43 (python) S 1 1 1\n")
    _write(
        cgroup / "system.slice" / "trainer.scope" / "pids.current",
        "400\n",
    )
    _write(
        cgroup / "system.slice" / "trainer.scope" / "pids.max",
        "1000\n",
    )
    grader.write_text(
        json.dumps(
            {
                "server_pid": 2,
                "updated_at": 990.0,
                "workers_spawned_total": 12,
                "worker_restarts_total": {"recycle": 4},
                "secret": "must-not-escape",
            }
        ),
        encoding="utf-8",
    )

    snapshot = collect_process_health(
        proc_root=proc,
        cgroup_root=cgroup,
        grader_health_path=grader,
        now=1000.0,
    )

    assert snapshot["pid_namespace_processes"] == 3
    assert snapshot["zombie_processes"] == 1
    assert snapshot["pid1_name"] == "docker-init"
    assert snapshot["init_subreaper_present"] is True
    assert snapshot["cgroup_pids_current"] == 400
    assert snapshot["cgroup_pids_max"] == 1000
    assert snapshot["cgroup_pids_utilization"] == 0.4
    assert snapshot["status"] == "warning"
    assert snapshot["restart_recommended"] is False
    assert snapshot["grader"]["age_seconds"] == 10.0
    assert snapshot["grader"]["workers_spawned_total"] == 12
    assert "secret" not in snapshot["grader"]


def test_collect_process_health_marks_missing_init_and_restart_threshold(
    tmp_path,
):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(proc / "self" / "cgroup", "0::/\n")
    _write(proc / "1" / "comm", "python\n")
    _write(proc / "1" / "stat", "1 (python) S 0 0 0\n")
    _write(cgroup / "pids.current", "51\n")
    _write(cgroup / "pids.max", "100\n")

    snapshot = collect_process_health(
        proc_root=proc,
        cgroup_root=cgroup,
        grader_health_path=tmp_path / "missing.json",
        now=1000.0,
    )

    assert snapshot["status"] == "critical"
    assert snapshot["restart_recommended"] is True
    assert "init_subreaper_missing" in snapshot["warning_reasons"]
    assert "cgroup_pid_usage_high" in snapshot["warning_reasons"]


def test_collect_process_health_handles_unlimited_or_missing_cgroup(tmp_path):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(proc / "self" / "cgroup", "0::/\n")
    _write(proc / "1" / "comm", "tini\n")
    _write(proc / "1" / "stat", "1 (tini) S 0 0 0\n")
    _write(cgroup / "pids.current", "5\n")
    _write(cgroup / "pids.max", "max\n")

    snapshot = collect_process_health(
        proc_root=proc,
        cgroup_root=cgroup,
        grader_health_path=tmp_path / "missing.json",
    )

    assert snapshot["status"] == "ok"
    assert snapshot["cgroup_pids_current"] == 5
    assert snapshot["cgroup_pids_max"] is None
    assert snapshot["cgroup_pids_utilization"] is None


def test_trainer_compose_enables_docker_init_subreaper():
    compose = (
        REPO_ROOT / "docker" / "docker-compose.trainer.yml"
    ).read_text(encoding="utf-8")

    assert "\n    init: true\n" in compose


def test_validator_entrypoint_is_valid_supervisor_script():
    entrypoint = REPO_ROOT / "docker" / "entrypoint.sh"
    completed = subprocess.run(
        ["bash", "-n", str(entrypoint)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    script = entrypoint.read_text(encoding="utf-8")
    assert "wait -n" in script
    assert "terminate_children" in script
    assert "exec reliquary validate" not in script
