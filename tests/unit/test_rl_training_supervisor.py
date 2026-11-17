"""Bounded orchestration with fake owned processes: no runtime, GPU, provider or transaction is started."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.rl_orders import supervisor as s

PROFILE = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"
HOTKEY = "5" + "A" * 47


def private_json(path, body):
    path.write_text(json.dumps(body))
    path.chmod(0o600)


@pytest.fixture
def prepared(tmp_path):
    from reliquary.protocol.profiles import PROFILES

    credentials = tmp_path / "credentials.json"
    private_json(credentials, {"RELIQUARY_SIGNER_URL": "https://signer.invalid", "RELIQUARY_SIGNER_CA": "/ca.pem",
                               "RELIQUARY_SIGNER_CERT": "/cert.pem", "RELIQUARY_SIGNER_KEY": "/key.pem",
                               "RELIQUARY_SIGNER_EXPECTED_HOTKEY": HOTKEY})
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    (model / "model.safetensors").write_bytes(b"frozen test weights")
    profile = PROFILES[PROFILE]
    manifest = tmp_path / "model-manifest.json"
    private_json(manifest, {"schema": "reliquary/local-model/v1", "profile_id": PROFILE,
                            "model_id": profile.model_id, "revision": profile.model_revision,
                            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()}})
    config = s.SupervisorConfig(task_id="order-train-test", order_sha256="a" * 64, profile_id=PROFILE,
                                run_id="order-train-test-0123456789abcdef",
                                state_dir=str(tmp_path / "runtime" / "order-train-test"),
                                credentials_file=str(credentials), model_dir=str(model), model_manifest_file=str(manifest),
                                model_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                cgroup_parent="/sys/fs/cgroup/allocated", validator_hotkey=HOTKEY,
                                controller_port=18080, trainer_port=18889, trainer_device=0, network="test", netuid=1,
                                max_runtime_seconds=5, start_seconds=4, drain_seconds=3, stop_seconds=2)
    entry = SimpleNamespace(task_id=config.task_id, profile_id=PROFILE, status="active",
                            params={"rl_batch_groups": 64, "training_order": {"order_sha256": config.order_sha256,
                                                                              "deadline_at": 1000, "quote_id": "quote-1",
                                                                              "quote_sha256": "b" * 64,
                                                                              "billing_basis": "matched_token_ceiling_and_trainer_time_v1"}})
    return config, entry


class Clock:
    t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class Process:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        assert self.returncode is not None
        return self.returncode


class Backend:
    def __init__(self, config, runtime_id, *, clock):
        self.runtime_id, self.clock = runtime_id, clock
        self.handles, self.env, self.stops = [], None, 0
        self.fail_stop = False

    def spawn(self, role, config, env):
        self.env = env
        handle = s._OwnedProcess(role, Process(101 + len(self.handles)), "12345", self.clock(),
                                 int(self.clock() * 1e9))
        self.handles.append(handle)
        return handle

    def stop(self, handles, seconds):
        self.stops += 1
        assert handles == self.handles and seconds <= 120
        if self.fail_stop:
            raise s.ReconciliationRequired("simulated unknown cleanup")
        out = []
        for handle in handles:
            handle.process.returncode = 0
            out.append(s.StoppedProcess(handle.role, handle.process.pid, handle.process.pid, handle.start_id,
                                        0, handle.started_at, self.clock()))
        return tuple(out), self.clock(), int(max(.1, self.clock() - 100) * 1e9)


class Client:
    def __init__(self, config, token, *, backend, clock):
        self.backend, self.clock, self.reads = backend, clock, 0
        self.draining = self.closed = self.wrong = self.stale = self.never_ready = self.never_drained = False
        self.admission_closed_on_start = False

    def read(self):
        self.reads += 1
        env = self.backend.env
        if self.draining and not self.never_drained:
            self.backend.handles[-1].process.returncode = 0
        return {"schema": s.RUNTIME_SCHEMA, "task_id": env["RELIQUARY_TASK_ID"],
                "order_sha256": env["RELIQUARY_RL_ORDER_SHA256"], "run_id": env["RELIQUARY_RL_RUN"],
                "runtime_id": "another-session" if self.wrong else env["RELIQUARY_RL_RUNTIME_ID"],
                "observed_at": 1 if self.stale else self.clock(), "controller_ready": not self.never_ready,
                "trainer_ready": len(self.backend.handles) == 2,
                "admission_closed": self.admission_closed_on_start or self.draining or self.reads >= 4,
                "drained": self.draining and not self.never_drained, "latest_checkpoint": 3,
                "unrequested_private_field": "must not enter the journal"}

    def drain(self, binding):
        assert binding["runtime_id"] == self.backend.runtime_id
        self.draining = True

    def close(self):
        self.closed = True


def harness(config, entry):
    clock, backend, client = Clock(), [], []

    def new_backend(config, runtime_id):
        backend.append(Backend(config, runtime_id, clock=clock))
        return backend[-1]

    def new_client(config, token):
        assert len(token) >= 32
        client.append(Client(config, token, backend=backend[-1], clock=clock))
        return client[-1]

    kwargs = {"load_entry": lambda task_id: entry, "backend_factory": new_backend, "client_factory": new_client,
              "namespace_check": lambda config: None, "claim": lambda config, binding: None,
              "sleep": clock.sleep, "clock": clock, "wall": clock}
    return clock, backend, client, kwargs


def test_owned_ready_drained_runtime_is_not_complete_without_usage_and_checkpoint(prepared):
    config, entry = prepared
    _, backend, client, kwargs = harness(config, entry)
    witness = s.run_supervisor(config, **kwargs)
    journal = json.loads((config.runtime_dir / "journal.json").read_text())
    assert journal["state"] == "execution_stopped" and "finalization" not in journal
    assert witness.task_id == config.task_id and witness.order_sha256 == config.order_sha256
    assert witness.process_scope == "local-cgroup-v2-v1" and witness.backend == "single"
    assert {p.role for p in witness.processes} == {"controller", "trainer"}
    assert witness.latest_checkpoint == 3 and witness.trainer_elapsed_ns > 0
    assert backend[0].stops == 1 and client[0].closed
    assert "unrequested_private_field" not in journal["observation"]
    assert "RELIQUARY_RL_TRAINER_TOKEN" not in (config.runtime_dir / "journal.json").read_text()


def test_completion_requires_both_verified_receipts(prepared):
    config, entry = prepared
    _, _, _, kwargs = harness(config, entry)
    seen = []

    def finalize(actual_entry, witness, actual_config):
        assert actual_entry is entry and actual_config is config
        seen.append(witness)
        return s.FinalizedRuntime("b" * 64, "c" * 64)

    s.run_supervisor(config, finalize=finalize, **kwargs)
    journal = json.loads((config.runtime_dir / "journal.json").read_text())
    assert journal["state"] == "completed" and len(seen) == 1
    assert journal["finalization"] == {"usage_sha256": "b" * 64, "delivery_sha256": "c" * 64}


@pytest.mark.parametrize("mode", ["wrong", "stale", "never_ready", "never_drained"])
def test_uncertain_identity_readiness_or_drain_is_bounded_and_never_finalized(prepared, mode):
    config, entry = prepared
    clock, backend, client, kwargs = harness(config, entry)
    create = kwargs["client_factory"]

    def bad_client(config, token):
        out = create(config, token)
        setattr(out, mode, True)
        return out

    kwargs["client_factory"] = bad_client
    with pytest.raises(s.ReconciliationRequired):
        s.run_supervisor(config, finalize=lambda *a: pytest.fail("must not finalize"), **kwargs)
    assert backend[0].stops == 1 and client[0].closed
    assert clock() <= 100 + config.start_seconds + config.max_runtime_seconds + config.drain_seconds
    assert json.loads((config.runtime_dir / "journal.json").read_text())["state"] == "reconciliation_required"


def test_uncertain_cleanup_is_not_retried_or_declared_stopped(prepared):
    config, entry = prepared
    _, backend, _, kwargs = harness(config, entry)
    create = kwargs["backend_factory"]

    def broken_backend(config, runtime_id):
        out = create(config, runtime_id)
        out.fail_stop = True
        return out

    kwargs["backend_factory"] = broken_backend
    with pytest.raises(s.ReconciliationRequired):
        s.run_supervisor(config, **kwargs)
    assert backend[0].stops == 1
    journal = json.loads((config.runtime_dir / "journal.json").read_text())
    assert journal["state"] == "reconciliation_required" and "stopped_runtime" not in journal


def test_restart_never_adopts_or_signals_a_recorded_pid(prepared):
    config, entry = prepared
    config.runtime_dir.mkdir(mode=0o700, parents=True)
    previous = {"state": "starting_trainer", "processes": [{"pid": 1}]}
    private_json(config.runtime_dir / "journal.json", previous)
    _, backend, client, kwargs = harness(config, entry)
    with pytest.raises(s.ReconciliationRequired, match="prior runtime journal"):
        s.run_supervisor(config, **kwargs)
    assert backend == client == []
    assert json.loads((config.runtime_dir / "journal.json").read_text()) == previous


@pytest.mark.parametrize("folder", ["controller", "trainer"])
def test_orphan_runtime_state_without_a_journal_cannot_be_rebound(prepared, folder):
    config, entry = prepared
    config.runtime_dir.mkdir(mode=0o700, parents=True)
    orphan = config.runtime_dir / folder
    orphan.mkdir(mode=0o700)
    (orphan / "rl-async.sqlite3").write_bytes(b"old ownership epoch")
    _, backend, _, kwargs = harness(config, entry)
    kwargs["namespace_check"] = lambda config: pytest.fail("must refuse before any remote preparation")
    kwargs["claim"] = lambda *a: pytest.fail("must not claim a new namespace")
    with pytest.raises(s.ReconciliationRequired, match="pre-existing runtime state"):
        s.run_supervisor(config, **kwargs)
    assert not (config.runtime_dir / "journal.json").exists() and backend == []


@pytest.mark.parametrize("change", [{"status": "retired"}, {"profile_id": "wrong"},
                                  {"params": {"training_order": {"order_sha256": "f" * 64, "deadline_at": 1000}}},
                                  {"params": {"training_order": {"order_sha256": "a" * 64, "deadline_at": 99}}}])
def test_registry_identity_or_deadline_refusal_never_launches(prepared, change):
    config, entry = prepared
    entry = SimpleNamespace(**{**vars(entry), **change})
    _, backend, _, kwargs = harness(config, entry)
    with pytest.raises(ValueError):
        s.run_supervisor(config, **kwargs)
    assert backend == []


def test_legacy_declaration_without_quote_binding_is_never_launched(prepared):
    config, entry = prepared
    entry.params["training_order"].pop("quote_id")
    _, backend, _, kwargs = harness(config, entry)
    with pytest.raises(ValueError, match="immutable quote"):
        s.run_supervisor(config, **kwargs)
    assert backend == []


def test_deadline_passing_during_namespace_reservation_never_launches(prepared):
    config, entry = prepared
    clock, backend, _, kwargs = harness(config, entry)
    kwargs["claim"] = lambda *a: clock.sleep(901)
    with pytest.raises(s.ReconciliationRequired, match="deadline passed before"):
        s.run_supervisor(config, **kwargs)
    assert backend == []


def test_controller_observing_closed_admission_does_not_launch_trainer(prepared):
    config, entry = prepared
    _, backend, _, kwargs = harness(config, entry)
    create = kwargs["client_factory"]

    def closed(config, token):
        client = create(config, token)
        client.admission_closed_on_start = True
        return client

    kwargs["client_factory"] = closed
    with pytest.raises(s.ReconciliationRequired, match="closed before trainer"):
        s.run_supervisor(config, **kwargs)
    assert [handle.role for handle in backend[0].handles] == ["controller"]


def test_credentials_and_namespace_are_explicit_not_inherited(prepared, monkeypatch):
    config, entry = prepared
    monkeypatch.setenv("RELIQUARY_RL_RUN", "old-production-run")
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted.so")
    monkeypatch.setenv("RELIQUARY_RL_PAY_OUTAGE_FILL", "on")
    env = s.child_environment(config, entry, runtime_id="d" * 32, run_id="new-run", token="secret")
    assert env["RELIQUARY_RL_RUN"] == "new-run" and env["RELIQUARY_RL_PAY_OUTAGE_FILL"] == "off"
    assert "PYTHONPATH" not in env and "LD_PRELOAD" not in env
    assert env["RELIQUARY_TRAINER_BACKEND"] == "single"
    assert env["RELIQUARY_RL_BASE_MODEL"] == config.model_dir
    assert env["RELIQUARY_STATE_DIR"] == str(config.runtime_dir / "controller")
    assert env["RELIQUARY_TRAINER_STATE_DIR"] == str(config.runtime_dir / "trainer")
    credentials = config.credentials()
    private_json(Path(config.credentials_file), {**credentials, "PYTHONPATH": "/injected"})
    with pytest.raises(ValueError, match="unsupported environment"):
        config.credentials()


@pytest.mark.parametrize("change", [{"trainer_port": 18080}, {"max_runtime_seconds": 0},
                                  {"trainer_device": True}, {"state_dir": "/tmp/another-task"}])
def test_invalid_operator_config_is_rejected(prepared, change):
    config, _ = prepared
    with pytest.raises(ValueError):
        replace(config, **change).validate()


def test_operator_config_rejects_extra_argv_and_nonprivate_files(prepared, tmp_path):
    config, _ = prepared
    path = tmp_path / "config.json"
    body = {"schema": s.SCHEMA, **asdict(config)}
    private_json(path, body)
    assert s.SupervisorConfig.read(path) == config
    private_json(path, {**body, "argv": ["sh", "-c", "unrequested"]})
    with pytest.raises(ValueError, match="exactly"):
        s.SupervisorConfig.read(path)
    private_json(path, body)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        s.SupervisorConfig.read(path)


@pytest.mark.parametrize("mode", ["changed_bytes", "extra_file", "symlink", "changed_manifest"])
def test_model_snapshot_is_immutable_and_materialized(prepared, mode):
    config, entry = prepared
    s.verify_model(config, entry)
    if mode == "changed_bytes":
        (Path(config.model_dir) / "model.safetensors").write_bytes(b"changed")
    elif mode == "extra_file":
        (Path(config.model_dir) / "new.safetensors").write_bytes(b"extra")
    elif mode == "symlink":
        (Path(config.model_dir) / "link").symlink_to(config.model_manifest_file)
    else:
        Path(config.model_manifest_file).write_text("{}")
    with pytest.raises(ValueError):
        s.verify_model(config, entry)


def test_public_runtime_view_is_scoped_and_expires(prepared):
    config, _ = prepared
    config.runtime_dir.mkdir(mode=0o700, parents=True)
    binding = {"task_id": config.task_id, "order_sha256": config.order_sha256, "run_id": "run", "runtime_id": "d" * 32}
    observation = {"schema": s.RUNTIME_SCHEMA, **binding, "observed_at": 100,
                   "controller_ready": True, "trainer_ready": True, "admission_closed": False,
                   "drained": False, "latest_checkpoint": 3}
    journal = {"schema": s.SCHEMA, **binding, "state": "running", "started_at": 90, "updated_at": 100,
               "observation": observation, "processes": [{"pid": 123}], "private_path": "/private"}
    private_json(config.runtime_dir / "journal.json", journal)
    view = s.read_runtime_view(config.runtime_dir.parent, config.task_id, now=101)
    assert view["managed"] and view["state"] == "running" and view["latest_checkpoint"] == 3
    assert "processes" not in view and "private_path" not in view
    assert not s.read_runtime_view(config.runtime_dir.parent, config.task_id, now=111)["managed"]
    assert s.read_runtime_view(config.runtime_dir.parent, "../another", now=101) == {"state": "unknown", "managed": False}


def test_missing_linux_cgroup_support_refuses_before_spawn(prepared, monkeypatch):
    config, _ = prepared
    monkeypatch.setattr(s.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        s.LocalBackend(config, "d" * 32)


def test_spawn_uses_only_fixed_bootstrap_and_a_new_session(prepared, monkeypatch):
    config, _ = prepared
    config.runtime_dir.mkdir(mode=0o700, parents=True)
    backend = object.__new__(s.LocalBackend)
    backend.scope = Path("/sys/fs/cgroup/allocated") / ("d" * 32)
    called = []

    def popen(argv, **kwargs):
        called.append((argv, kwargs))
        return Process(123)

    monkeypatch.setattr(s.subprocess, "Popen", popen)
    monkeypatch.setattr(s, "_process_start", lambda pid: "12345")
    handle = backend.spawn("trainer", config, {"SAFE": "1"})
    argv, kwargs = called[0]
    assert argv == [s.sys.executable, "-P", "-m", "reliquary.rl_orders.supervisor", "--child-role", "trainer"]
    assert kwargs["start_new_session"] is True and kwargs["close_fds"] is True and "shell" not in kwargs
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["RELIQUARY_SUPERVISOR_CGROUP"] == str(backend.scope)
    assert handle.start_id == "12345" and (config.runtime_dir / "trainer.log").stat().st_mode & 0o077 == 0


def test_reused_pid_never_receives_a_group_signal(prepared, monkeypatch):
    backend = object.__new__(s.LocalBackend)
    handle = s._OwnedProcess("controller", Process(123), "original", 90, 90_000_000_000)
    monkeypatch.setattr(s, "_process_start", lambda pid: "reused")
    monkeypatch.setattr(s.os, "killpg", lambda *a: pytest.fail("must never signal an unowned group"))
    with pytest.raises(s.ReconciliationRequired, match="identity changed"):
        backend.stop([handle], 2)


def test_a_group_that_still_exists_cannot_mint_a_stopped_witness(monkeypatch):
    backend = object.__new__(s.LocalBackend)
    backend.empty = lambda: True
    process = Process(123)
    process.returncode = 0
    handle = s._OwnedProcess("controller", process, "12345", 90, 90_000_000_000)
    probes = []
    monkeypatch.setattr(s.os, "killpg", lambda pid, sig: probes.append((pid, sig)))
    with pytest.raises(s.ReconciliationRequired, match="group absence"):
        backend.stop([handle], 2)
    assert probes == [(123, 0)]


def test_detached_descendants_require_owned_cgroup_cleanup_even_after_leader_exit(tmp_path, monkeypatch):
    backend = object.__new__(s.LocalBackend)
    backend.scope = tmp_path
    kill_file = tmp_path / "cgroup.kill"
    kill_file.write_text("")
    backend.empty = lambda: kill_file.read_text() == "1"
    clock = Clock()
    monkeypatch.setattr(s.time, "monotonic", clock)
    monkeypatch.setattr(s.time, "monotonic_ns", lambda: int(clock() * 1e9))
    monkeypatch.setattr(s.time, "sleep", clock.sleep)
    monkeypatch.setattr(s.time, "time", clock)

    def absent(pid, sig):
        assert sig == 0
        raise ProcessLookupError

    monkeypatch.setattr(s.os, "killpg", absent)
    process = Process(123)
    process.returncode = 0
    handle = s._OwnedProcess("trainer", process, "12345", 90, 90_000_000_000)
    with pytest.raises(s.ReconciliationRequired, match="forced termination"):
        backend.stop([handle], 2)
    assert kill_file.read_text() == "1" and clock() <= 102
    assert handle.exited_ns - handle.started_ns == 10_000_000_000


@pytest.mark.parametrize("role", ["controller", "trainer"])
def test_child_joins_its_owned_scope_before_fixed_cli_exec(prepared, monkeypatch, tmp_path, role):
    config, entry = prepared
    runtime_id = "d" * 32
    scope = tmp_path / runtime_id
    scope.mkdir()
    (scope / "cgroup.procs").write_text("")
    env = s.child_environment(config, entry, runtime_id=runtime_id, run_id="new-run", token="fresh")
    for key, value in {**env, "RELIQUARY_SUPERVISOR_CGROUP": str(scope)}.items():
        monkeypatch.setenv(key, value)
    calls = []

    def execute(executable, command, environment):
        assert (scope / "cgroup.procs").read_text() == str(s.os.getpid())
        calls.append(command)

    monkeypatch.setattr(s.os, "execve", execute)
    s._child(role)
    command = calls[0]
    assert command[:4] == [s.sys.executable, "-P", "-m", "reliquary.cli.main"]
    if role == "controller":
        assert command[4:6] == ["validate", "--no-set-weights"]
        assert "--external-ip" not in command
        assert command[command.index("--checkpoint") + 1] == config.model_dir
    else:
        assert command[4:] == ["train-async"]


@pytest.mark.parametrize("listing,valid", [
    ({"KeyCount": 0, "IsTruncated": False}, True),
    ({"KeyCount": 1, "IsTruncated": False, "Contents": [{"Key": "orphan"}]}, False),
    ({}, False), ({"KeyCount": 0, "IsTruncated": True}, False),
])
def test_remote_namespace_absence_is_bounded_and_fail_closed(prepared, monkeypatch, listing, valid):
    config, _ = prepared
    calls = []

    class Store:
        def list_objects_v2(self, **kwargs):
            calls.append(kwargs)
            return listing

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(s, "_checkpoint_client", lambda config: (Store(), "bucket"))
    if valid:
        s.verify_empty_namespace(config)
    else:
        with pytest.raises(ValueError):
            s.verify_empty_namespace(config)
    assert calls == [{"Bucket": "bucket", "Prefix": f"rl/{config.run_id}/", "MaxKeys": 1}, "closed"]


@pytest.mark.parametrize("mode", ["created", "lost_reply", "conflict"])
def test_cross_host_claim_is_conditional_and_ambiguous_writes_are_reconciled(prepared, monkeypatch, mode):
    import io

    config, _ = prepared
    binding = {"task_id": config.task_id, "order_sha256": config.order_sha256,
               "run_id": config.run_id, "runtime_id": "d" * 32}
    calls, bodies = [], []

    class Store:
        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*" and kwargs["Key"].endswith("/runtime-claim.json")
            calls.append("put")
            self.value = json.loads(kwargs["Body"])
            if mode != "created":
                raise TimeoutError("ambiguous write")

        def get_object(self, **kwargs):
            calls.append("get")
            value = self.value if mode == "lost_reply" else {**self.value, "runtime_id": "another-runtime"}
            body = io.BytesIO(json.dumps(value).encode())
            bodies.append(body)
            return {"Body": body}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(s, "_checkpoint_client", lambda config: (Store(), "bucket"))
    if mode == "conflict":
        with pytest.raises(s.ReconciliationRequired, match="claim"):
            s.reserve_namespace(config, binding)
    else:
        s.reserve_namespace(config, binding)
    assert calls[0] == "put" and calls[-1] == "closed" and calls.count("put") == 1
    assert all(body.closed for body in bodies)


def test_module_entrypoint_uses_canonical_receipt_classes(monkeypatch):
    import runpy

    called = []
    monkeypatch.setattr(s, "main", lambda: called.append(True))
    runpy.run_path(s.__file__, run_name="__main__")
    assert called == [True]
