"""Shell results, startup diagnostics and validation before writes."""

import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.cli.output import SCHEMA
from tests.unit.test_jobs_cli import bucket, registry, _StatusRecords  # noqa: F401


@pytest.mark.parametrize("flag,value", [
    ("--model-revision", "revision"), ("--model-architecture", "architecture"),
    ("--from-profile", "template"), ("--envs", "logic"),
])
def test_model_flags_cannot_be_silently_dropped(registry, flag, value):
    result = CliRunner().invoke(app, ["tasks", "create", "--task-id", "default",
        "--profile-id", "qwen3-4b-base-dapo-fill-closed-v6", "--cap", "0.2",
        flag, value, "--json"])
    assert result.exit_code == 1 and result.stdout == ""
    error = json.loads(result.stderr)
    assert error["schema"] == SCHEMA and "require --model" in error["error"]["message"]
    assert registry["entries"] == {}


def test_task_results_and_validation_are_machine_readable(registry):
    runner = CliRunner()
    result = runner.invoke(app, ["tasks", "create", "--task-id", "default",
        "--profile-id", "qwen3-4b-base-dapo-fill-closed-v6", "--cap", "0.2", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"]["task_id"] == "default"
    listing = runner.invoke(app, ["tasks", "list", "--json"])
    assert json.loads(listing.stdout)["data"]["total_cap"] == 0.2
    bad = runner.invoke(app, ["tasks", "create", "--cap", "oops", "--json"])
    assert bad.exit_code == 2 and bad.stdout == ""
    assert json.loads(bad.stderr)["error"]["code"] == "invalid_input"


def test_unknown_json_option_has_structured_error():
    result = CliRunner().invoke(app, ["jobs", "list", "--unknown", "--json"])
    assert result.exit_code == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "invalid_input"


def test_storage_failure_is_redacted_and_debug_retains_details(monkeypatch):
    async def broken(**kwargs):
        raise RuntimeError("secret-token-from-library")

    monkeypatch.setattr("reliquary.infrastructure.task_registry_store.read_registry", broken)
    result = CliRunner().invoke(app, ["tasks", "list", "--json"])
    assert result.exit_code == 1 and result.stdout == ""
    assert "secret-token-from-library" not in result.stderr
    assert "RuntimeError" in json.loads(result.stderr)["error"]["message"]
    debug = CliRunner().invoke(app, ["--debug", "tasks", "list", "--json"])
    assert isinstance(debug.exception, RuntimeError)


def test_missing_operator_dependency_has_actionable_error(monkeypatch):
    async def broken(**kwargs):
        raise ModuleNotFoundError("private-library-detail")

    monkeypatch.setattr("reliquary.infrastructure.task_registry_store.read_registry", broken)
    result = CliRunner().invoke(app, ["tasks", "list", "--json"])
    assert result.exit_code == 1 and result.stdout == ""
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "dependency_missing" and "[operator]" in error["message"]
    assert "private-library-detail" not in result.stderr


def test_nonexistent_job_is_not_certified_drained(bucket, monkeypatch):
    records = _StatusRecords([], [], {})

    async def missing(job_id):
        return None, None

    monkeypatch.setattr(records, "read_job", missing)
    monkeypatch.setattr(records, "read_ledgers", missing)
    monkeypatch.setattr("reliquary.infrastructure.corpus_record_store.BucketRecordStore",
                        lambda: records)
    result = CliRunner().invoke(app, ["jobs", "status", "missing", "--json"])
    assert result.exit_code == 2 and result.stdout == ""
    assert "no job" in json.loads(result.stderr)["error"]["message"]


def test_existing_empty_job_can_be_drained(monkeypatch):
    records = _StatusRecords([], [], {})

    async def absent_ledgers(job_id):
        return None, None

    monkeypatch.setattr(records, "read_ledgers", absent_ledgers)
    monkeypatch.setattr("reliquary.infrastructure.corpus_record_store.BucketRecordStore",
                        lambda: records)
    result = CliRunner().invoke(app, ["jobs", "status", "swe-v1", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["data"]["drained"] is True


@pytest.mark.parametrize("args,exit_code", [
    (["--help"], 0), (["--version"], 0), (["--debug", "--version"], 0),
    (["context", "--json"], 0), (["--debug", "context", "--json"], 0),
    (["doctor", "--json"], 1), (["--debug", "doctor", "--json"], 1),
    (["jobs", "list", "--json"], 1),
])
def test_diagnostics_survive_invalid_runtime_configuration(args, exit_code, tmp_path):
    env = {**os.environ, "RELIQUARY_PROTOCOL_PROFILE": "unavailable-profile",
           "RELIQUARY_ADMIN_SECRET": "private-secret-value",
           "RELIQUARY_ADMIN_URL": "https://user:private-password@admin.example/path?token=secret"}
    env.pop("RELIQUARY_TASK_CONTRACT", None)
    result = subprocess.run([sys.executable, "-c",
        "from reliquary.cli.entrypoint import main; main()", *args],
        env=env, capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == exit_code, result.stderr
    assert all(value not in result.stdout + result.stderr for value in
               ("private-secret-value", "private-password", "token=secret"))
    if "--json" in args:
        payload = json.loads(result.stdout if args[-2] != "list" else result.stderr)
        assert payload["schema"] == SCHEMA
        if "context" in args:
            assert payload["data"]["admin_origin"] == "https://admin.example"
        if "doctor" in args:
            assert payload["data"]["ok"] is False


def test_nonblocking_eval_flags_reach_operator(monkeypatch):
    from reliquary.eval import operator

    monkeypatch.setenv("RELIQUARY_ADMIN_SECRET", "s" * 32)
    monkeypatch.setattr(operator, "read_set_card", lambda set_id: {"set_id": set_id})
    seen = {}

    def create(client, **kwargs):
        seen.update(kwargs)
        return [{"job_id": "order-eval-a", "set_id": "a", "qualification_id": "order-q-a",
                 "state": "qualification_requested"}]

    monkeypatch.setattr(operator, "create_evaluations", create)
    result = CliRunner().invoke(app, ["eval", "create", "--set", "a", "--model", "org/m@" + "a" * 40,
        "--samples", "2", "--max-new-tokens", "64", "--temperature", "0.6",
        "--no-wait", "--timeout", "20", "--json"])
    assert result.exit_code == 0, result.output
    assert seen["wait"] is False and seen["timeout_seconds"] == 20
    assert json.loads(result.stdout)["data"][0]["state"] == "qualification_requested"
