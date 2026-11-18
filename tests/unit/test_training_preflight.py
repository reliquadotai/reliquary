"""Local prerequisites never start a runtime or turn fixture evidence into qualification."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from reliquary.admin.training import CreateTraining, training_entry
from reliquary.rl_orders import preflight as p
from reliquary.rl_orders import supervisor as s
from reliquary.shared.task_registry import render_registry


def private_json(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from reliquary.protocol.profiles import PROFILES
    from reliquary.validator.corpus_periods import usd_price_params

    profile_id = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"
    body = CreateTraining(training_id="preflight", profile_id=profile_id,
                          environments=[{"name": "reliquary_dapo_math_v1", "share_bps": 10000}],
                          limits={"max_steps": 1, "deadline_seconds": 600, "max_tokens": 4096},
                          max_price_usd_per_mtoken=0.01, batch_groups=1, lag_target=1,
                          quote_id="qualification-quote", quote_sha256="b" * 64,
                          billing_basis="matched_token_ceiling_and_trainer_time_v1")
    prices = usd_price_params(usd_per_mtoken_input=0., usd_per_mtoken_output=.01, alpha_usd=1.,
                              tao_usd=1., miner_alpha_per_period=1., set_at=1000.)
    entry = training_entry(body, task_prefix="order-", created_at=1000, price_params=prices)
    registry = tmp_path / "registry.json"
    private_json(registry, json.loads(render_registry({entry.task_id: entry})))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    (model / "model.safetensors").write_bytes(b"synthetic test weights")
    profile = PROFILES[profile_id]
    manifest = tmp_path / "manifest.json"
    private_json(manifest, {"schema": "reliquary/local-model/v1", "profile_id": profile_id,
                            "model_id": profile.model_id, "revision": profile.model_revision,
                            "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in model.iterdir()}})
    credentials = tmp_path / "credentials.json"
    hotkey = "5" + "A" * 47
    secrets = {"R2_ACCESS_KEY_ID": "keep-this-secret", "R2_SECRET_ACCESS_KEY": "keep-this-secret",
               "R2_BUCKET_ID": "private-test", "R2_ACCOUNT_ID": "fixture",
               "R2_PUBLIC_ENDPOINT": "https://fixture.r2.cloudflarestorage.com",
               "R2_PUBLIC_ACCESS_KEY_ID": "keep-this-secret", "R2_PUBLIC_SECRET_ACCESS_KEY": "keep-this-secret",
               "R2_PUBLIC_BUCKET": "public-test", "RELIQUARY_SIGNER_URL": "https://signer.invalid",
               "RELIQUARY_SIGNER_EXPECTED_HOTKEY": hotkey}
    for key in ("RELIQUARY_SIGNER_CA", "RELIQUARY_SIGNER_CERT", "RELIQUARY_SIGNER_KEY"):
        path = tmp_path / (key + ".pem")
        path.write_bytes(b"presence-only fixture, not a usable certificate")
        path.chmod(0o600)
        secrets[key] = str(path)
    private_json(credentials, secrets)
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "cgroup.controllers").write_text("cpu memory")
    parent = tmp_path / "runtime"
    parent.mkdir(mode=0o700)
    config = s.SupervisorConfig(entry.task_id, entry.params["training_order"]["order_sha256"], profile_id,
                                entry.task_id + "-0123456789abcdef", str(parent / entry.task_id), str(credentials),
                                str(model), str(manifest), hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                str(cg), hotkey, 18080, 18889, 0, "local", 1, 120, 30, 90, 10)
    config_file = tmp_path / "config.json"
    private_json(config_file, {"schema": s.SCHEMA, **asdict(config)})
    monkeypatch.setattr(s.sys, "platform", "linux")  # Only the prerequisite file check is simulated.
    monkeypatch.setattr(p, "_cuda_build", lambda: {"torch": "fixture", "torch_cuda_build": "13.0",
                                                 "transformers": "fixture", "cuda_initialized": False})
    return config_file, registry, config, entry


def run(prepared):
    config_file, registry, _, _ = prepared
    return p.preflight(config_file, registry, now=1100)


def test_local_evidence_never_grants_launch_or_paid_qualification(prepared):
    report = run(prepared)
    assert report["status"] == "local_prerequisites_checked"
    assert all(check["status"] == "passed" for check in report["checks"])
    assert report["launch_authorized"] is report["qualified"] is report["paid_admission"] is False
    assert set(report["unverified"]) == set(p.UNVERIFIED)
    assert report["limits"]["effective_runtime_seconds"] == 120
    assert report["limits"]["nominal_process_budgets_seconds"] == 220
    assert "post_stop_checkpoint_finalization" in report["limits"]["nominal_budget_excludes"]
    assert report["limits"]["max_steps"] == 1 and report["limits"]["max_tokens"] == 4096
    assert report["limits"]["step_and_token_limits_close_admission"] is True
    assert prepared[3].env_split["reliquary_competitive_code_v1"] == 0
    assert report["controller_grader"] == {"required_at_startup": True, "availability_verified": False}
    raw = json.dumps(report)
    assert "keep-this-secret" not in raw and str(prepared[0].parent) not in raw


def test_controller_grader_is_optional_for_an_all_math_split(prepared):
    config, entry = prepared[2:]
    entry = replace(entry, env_split={"reliquary_dapo_math_v1": 1.0})
    assert p._async_settings(config, entry) == {"required_at_startup": False, "availability_verified": False}


def test_preflight_has_no_runtime_storage_network_or_filesystem_side_effects(prepared, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("preflight must not mutate, connect, claim, signal or start any work")

    before = {str(f): f.read_bytes() for f in prepared[0].parent.rglob("*") if f.is_file()}
    for name in ("mkdir", "kill", "killpg"):
        monkeypatch.setattr(p.os, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("run_supervisor", "reserve_namespace", "verify_empty_namespace", "_checkpoint_client"):
        monkeypatch.setattr(s, name, forbidden)
    assert run(prepared)["status"] == "local_prerequisites_checked"
    after = {str(f): f.read_bytes() for f in prepared[0].parent.rglob("*") if f.is_file()}
    assert before == after and not prepared[2].runtime_dir.exists()


@pytest.mark.parametrize("kind", ["expired", "wrong_order", "paused", "missing_token_cap", "retired"])
def test_unbound_or_unbounded_order_blocks_before_model_verification(prepared, kind, monkeypatch):
    _, registry, _, entry = prepared
    order = dict(entry.params["training_order"])
    if kind == "expired":
        order["deadline_at"] = 1100
    elif kind == "wrong_order":
        order["order_sha256"] = "c" * 64
    elif kind == "missing_token_cap":
        order.pop("max_tokens")
    entry = replace(entry, params={**entry.params, "training_order": order},
                    admission="paused" if kind == "paused" else "open",
                    status="retired" if kind == "retired" else "active",
                    retired_at=1050 if kind == "retired" else None)
    private_json(registry, json.loads(render_registry({entry.task_id: entry})))
    monkeypatch.setattr(s, "verify_model", lambda *a: pytest.fail("must not hash an unauthorized model"))
    report = run(prepared)
    assert report["status"] == "blocked"
    assert {"name": "local_registry_order_and_limits", "status": "blocked"} in report["checks"]
    assert "limits" not in report


@pytest.mark.parametrize("name", ["journal.json", "controller", "trainer", "home", "hf-cache",
                                  "controller.log", "usage.json", "completion.json", "checkpoint"])
def test_old_state_cannot_pass_as_a_fresh_run(prepared, name):
    state = prepared[2].runtime_dir
    state.mkdir(mode=0o700)
    (state / name).write_text("retained ownership evidence")
    assert {"name": "private_fresh_state", "status": "blocked"} in run(prepared)["checks"]


def test_only_a_private_regular_supervisor_lock_may_already_exist(prepared):
    state = prepared[2].runtime_dir
    state.mkdir(mode=0o700)
    lock = state / "supervisor.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    assert run(prepared)["status"] == "local_prerequisites_checked"
    lock.chmod(0o644)
    assert {"name": "private_fresh_state", "status": "blocked"} in run(prepared)["checks"]


def test_model_change_and_incomplete_credentials_are_reported_without_sensitive_errors(prepared):
    config = prepared[2]
    (p.Path(config.model_dir) / "model.safetensors").write_bytes(b"changed")
    values = json.loads(p.Path(config.credentials_file).read_text())
    values["RELIQUARY_SIGNER_KEY"] = "/missing/private/key.pem"
    private_json(p.Path(config.credentials_file), values)
    report = run(prepared)
    assert report["status"] == "blocked"
    assert {"name": "materialized_model_bytes", "status": "blocked"} in report["checks"]
    assert {"name": "local_storage_and_signer_material", "status": "blocked"} in report["checks"]
    assert "key.pem" not in json.dumps(report)


def test_config_and_registry_must_be_private_and_unambiguous(prepared):
    config_file, registry, _, _ = prepared
    registry.write_text('{"tasks":{},"tasks":{}}')
    assert run(prepared)["status"] == "blocked"
    config_file.chmod(0o644)
    report = run(prepared)
    assert report["checks"] == [{"name": "private_config", "status": "blocked"}]
    assert "binding" not in report


def test_fifo_config_is_refused_without_waiting_for_a_writer(tmp_path):
    config = tmp_path / "config.json"
    os.mkfifo(config, mode=0o600)
    result = subprocess.run([sys.executable, "-B", "-m", "reliquary.rl_orders.preflight",
                             "--config", str(config), "--registry", str(tmp_path / "registry.json")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["checks"] == [{"name": "private_config", "status": "blocked"}]


@pytest.mark.parametrize("missing", ["max_price_per_token", "rl_planned_tokens_per_period"])
def test_missing_effective_auction_or_async_declaration_blocks(prepared, missing):
    _, registry, _, entry = prepared
    params = dict(entry.params)
    params.pop(missing)
    entry = replace(entry, params=params)
    private_json(registry, json.loads(render_registry({entry.task_id: entry})))
    report = run(prepared)
    assert report["status"] == "blocked"
    assert any(check["status"] == "blocked" and check["name"] in
               {"local_registry_order_and_limits", "managed_async_auction_settings"} for check in report["checks"])


def test_deadline_caps_actual_runtime_and_non_linux_remains_blocked(prepared, monkeypatch):
    config_file, registry, _, _ = prepared
    report = p.preflight(config_file, registry, now=1599)
    assert report["limits"]["effective_runtime_seconds"] == 1
    monkeypatch.setattr(s.sys, "platform", "darwin")
    assert {"name": "linux_cgroup_parent", "status": "blocked"} in run(prepared)["checks"]


def test_cli_failure_is_json_and_nonzero_without_launch(prepared, monkeypatch, capsys):
    monkeypatch.setattr(p, "preflight", lambda *a: {"schema": p.SCHEMA, "status": "blocked"})
    monkeypatch.setattr(p.s.sys, "argv", ["preflight", "--config", str(prepared[0]), "--registry", str(prepared[1])])
    with pytest.raises(SystemExit) as exit_info:
        p.main()
    assert exit_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


@pytest.mark.parametrize("literal,accepted", [("None", False), ("'13.0'", True), ("'unavailable'", False),
                                             ("None\ncuda = '13.0'", False)])
def test_cuda_build_is_checked_without_importing_or_executing_torch(tmp_path, monkeypatch, literal, accepted):
    from types import SimpleNamespace

    # Call the original helper; it is not replaced by the prepared fixture here.
    source = tmp_path / "version.py"
    source.write_text("raise AssertionError('must never execute torch')\ncuda: str | None = " + literal)
    monkeypatch.setattr(p.metadata, "distribution", lambda name: SimpleNamespace(
        version="fixture", locate_file=lambda file: source))
    monkeypatch.setattr(p.metadata, "version", lambda name: "fixture")
    if accepted:
        assert p._cuda_build()["torch_cuda_build"] == "13.0"
    else:
        with pytest.raises(ValueError):
            p._cuda_build()


def test_clean_interpreter_preflight_never_imports_torch_or_performs_io_effects(prepared):
    code = '''
import json, sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("preflight tried to import torch")
sys.meta_path.insert(0, NoTorch())
def forbid(event, args):
    if event in {"socket.connect", "subprocess.Popen", "os.mkdir", "os.kill", "os.system"}:
        raise AssertionError("preflight tried to perform an effect: " + event)
sys.addaudithook(forbid)
from reliquary.rl_orders.preflight import preflight
report = preflight(sys.argv[1], sys.argv[2], now=1100)
assert "torch" not in sys.modules
assert report["qualified"] is report["launch_authorized"] is report["paid_admission"] is False
print(json.dumps({"status": report["status"], "torch_imported": False}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(prepared[0]), str(prepared[1])],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["torch_imported"] is False
