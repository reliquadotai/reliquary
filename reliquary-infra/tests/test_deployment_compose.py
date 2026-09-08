"""Render the same split config/Compose layout installed by the role playbooks."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import jinja2
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker Compose required")
@pytest.mark.parametrize("role", ["cpu-executor", "signer"])
def test_role_compose_reads_the_deployed_configuration(tmp_path, role):
    from jinja2 import StrictUndefined

    context = {
        "cpu_exec_image_id": "sha256:" + "a" * 64,
        "cpu_exec_image_revision": "b" * 40,
        "cpu_exec_bind_ip": "127.0.0.1",
        "cpu_exec_runtime_id": "grader-sha256:" + "c" * 64,
        "signer_image_id": "sha256:" + "d" * 64,
        "signer_network": "finney", "signer_netuid": 81,
        "signer_wallet_name": "test", "signer_hotkey_name": "test",
        "signer_expected_hotkey": "test-public-identity",
        "signer_repo_id": "test/repository", "signer_bind_ip": "127.0.0.1",
    }
    etc = tmp_path / "etc" / "reliquary"
    opt = tmp_path / "opt" / role
    etc.mkdir(parents=True)
    opt.mkdir(parents=True)
    template = ROOT / "reliquary-infra" / "templates" / f"{role}.env.j2"
    rendered = jinja2.Template(template.read_text(), undefined=StrictUndefined).render(context)
    env_file = etc / f"{role}.env"
    env_file.write_text(rendered.replace("/etc/reliquary/", f"{etc}/"))
    compose = opt / "compose.yml"
    shutil.copy2(ROOT / "docker" / f"docker-compose.{role}.yml", compose)
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_file), "--file", str(compose),
         "config", "--format", "json"],
        env={key: value for key, value in os.environ.items()
             if not key.startswith(("RELIQUARY_", "BT_", "CPU_EXECUTOR_", "SIGNER_"))},
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    service = json.loads(result.stdout)["services"][f"reliquary-{role}"]
    assert service["environment"][f"RELIQUARY_{role.upper().replace('-', '_')}_HOST"] == "127.0.0.1"
    assert service["pull_policy"] == "never"
