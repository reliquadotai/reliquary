"""Smoke-check a non-editable installation, without source imports or live services."""

import json
import os
from pathlib import Path
import subprocess
import sys

import reliquary
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from reliquary.cli.main import app


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    installed = Path(reliquary.__file__).resolve()
    assert source not in installed.parents, f"source checkout imported: {installed}"
    assert "PYTHONPATH" not in os.environ, "unset PYTHONPATH for package checks"
    cli = Path(sys.executable).parent / "reliquary"
    def installed_command(*args, env=None):
        return subprocess.run(
            [str(cli), *args], capture_output=True, text=True, timeout=30, env=env,
        )

    result = installed_command("--help")
    assert result.returncode == 0, result.stderr
    result = installed_command("--version")
    assert result.returncode == 0 and result.stdout.strip() == reliquary.__version__, result.stderr

    sensitive = "package-check-secret"
    diagnostic_env = {
        **os.environ,
        "RELIQUARY_ADMIN_SECRET": sensitive,
        "RELIQUARY_EXECUTOR_TOKEN": sensitive,
        "RELIQUARY_ADMIN_URL": f"https://user:{sensitive}@example.invalid:443/path?key={sensitive}",
        "RELIQUARY_API_KEY": sensitive,
        "RELIQUARY_API_URL": f"https://user:{sensitive}@api.example.invalid:443/path?key={sensitive}",
    }
    result = installed_command("context", "--json", env=diagnostic_env)
    assert result.returncode == 0 and sensitive not in result.stdout + result.stderr, result.stderr
    context = json.loads(result.stdout)
    assert context["schema"] == "reliquary/cli/v1"
    assert context["data"]["admin_origin"] == "https://example.invalid:443"
    assert context["data"]["api_origin"] == "https://api.example.invalid:443"
    result = installed_command("doctor", "--role", "client", "--json")
    assert result.returncode == 0 and json.loads(result.stdout)["data"]["ok"], result.stderr

    invalid_env = {**diagnostic_env, "RELIQUARY_PROTOCOL_PROFILE": "missing-package-check-profile"}
    invalid_env.pop("RELIQUARY_TASK_CONTRACT", None)
    for args in [("--help",), ("--version",), ("context", "--json"), ("platform", "--help")]:
        result = installed_command(*args, env=invalid_env)
        assert result.returncode == 0, result.stderr
        assert sensitive not in result.stdout + result.stderr
    result = installed_command("doctor", "--json", env=invalid_env)
    assert result.returncode == 1, result.stdout + result.stderr
    diagnosis = json.loads(result.stdout)["data"]
    assert not diagnosis["ok"] and any(
        check["name"] == "runtime_configuration" and not check["ok"]
        for check in diagnosis["checks"]
    )
    result = installed_command("envs", "list", env=invalid_env)
    assert result.returncode == 1 and "doctor" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr and sensitive not in result.stderr
    customer_env = {**invalid_env, "RELIQUARY_API_KEY": "", "JOBS_API_KEY": "",
                    "RELIQUARY_API_URL": "http://127.0.0.1:1"}
    result = installed_command("platform", "capabilities", "--json", env=customer_env)
    assert result.returncode == 1 and not result.stdout, result.stdout + result.stderr
    assert json.loads(result.stderr)["error"]["code"] == "credential_required"

    runner = CliRunner()
    count = 0

    def check_help(command, path=()):
        nonlocal count
        result = runner.invoke(app, [*path, "--help"])
        assert result.exit_code == 0, f"{' '.join(path)}: {result.output}"
        count += 1
        if isinstance(command, TyperGroup):
            for name, child in command.commands.items():
                check_help(child, (*path, name))

    check_help(get_command(app))
    result = runner.invoke(app, ["envs", "list"])
    assert result.exit_code == 0 and "openmathinstruct" in result.stdout, result.output
    result = runner.invoke(app, ["envs", "show", "openmathinstruct"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["name"] == "openmathinstruct"
    assert runner.invoke(app, ["not-a-command"]).exit_code == 2

    fixture = installed.parent.parent / "tests/fixtures/reliquarylogic_v1.jsonl"
    original = fixture.read_bytes()
    try:
        fixture.write_bytes(original + b"\n")
        result = subprocess.run(
            [sys.executable, "-c", "import reliquary.environment.registry"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, "changed fixture was accepted"
        assert "golden_fixture digest mismatch" in result.stderr, result.stderr
    finally:
        fixture.write_bytes(original)
    print(f"Package smoke passed: {count} help paths, diagnostics, environment catalog, fixture integrity")


if __name__ == "__main__":
    main()
