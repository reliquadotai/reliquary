"""Compose accepts PID strings that Docker Engine may reject at creation."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = sorted((ROOT / "docker").glob("*compose*.yml"))


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda path: path.name)
def test_core_compose_uses_docker_private_pid_default(path):
    # https://docs.docker.com/reference/cli/docker/container/run/#pid-settings---pid
    # Docker's explicit PID modes are host/container:<id>; unlike ipc/cgroup,
    # "private" is invalid. These roles must keep the isolated, omitted default.
    for name, service in yaml.safe_load(path.read_text())["services"].items():
        assert service.get("pid", "") == "", f"{path.name}:{name}: omit pid for isolation"


@pytest.mark.parametrize("role", ("cpu-executor", "signer"))
def test_cpu_role_compose_render_keeps_private_pid_default(role, tmp_path):
    docker = shutil.which("docker")
    standalone = shutil.which("docker-compose")
    command = [docker, "compose"] if docker else [standalone] if standalone else None
    if command is None:
        pytest.skip("Docker Compose is not installed")
    version = subprocess.run([*command, "version"], capture_output=True)
    if version.returncode:
        pytest.skip("Docker Compose plugin is not installed")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")
    environ = {**os.environ,
               "CPU_EXECUTOR_IMAGE": "sha256:" + "0" * 64,
               "SIGNER_IMAGE": "sha256:" + "0" * 64,
               "RELIQUARY_CPU_EXECUTOR_ENV_FILE": str(empty_env),
               "RELIQUARY_SIGNER_ENV_FILE": str(empty_env)}
    result = subprocess.run(
        [*command, "-f", str(ROOT / "docker" / f"docker-compose.{role}.yml"),
         "config", "--format", "json"],
        env=environ, capture_output=True, text=True, check=True,
    )
    service = json.loads(result.stdout)["services"][f"reliquary-{role}"]
    assert service.get("pid", "") == ""
    assert service["ipc"] == "private"
