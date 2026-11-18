"""One explicitly allocated Linux training runtime, with owned cgroup/processes and a durable journal.

This module never provisions hardware or submits weights. Run it only with an operator-owned private config.
An unfinished journal is a reconciliation boundary, never permission to adopt or kill recorded PIDs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from reliquary.shared.strict_json import strict_json_loads

SCHEMA = "reliquary/training-supervisor/v1"
RUNTIME_SCHEMA = "reliquary/training-runtime/v1"
PROCESS_SCOPE = "local-cgroup-v2-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_CREDENTIALS = frozenset({
    "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_ID", "R2_ENDPOINT_URL", "R2_REGION",
    "R2_PUBLIC_ENDPOINT", "R2_PUBLIC_ACCESS_KEY_ID", "R2_PUBLIC_SECRET_ACCESS_KEY", "R2_PUBLIC_BUCKET",
    "RELIQUARY_SIGNER_URL", "RELIQUARY_SIGNER_CA", "RELIQUARY_SIGNER_CERT", "RELIQUARY_SIGNER_KEY",
    "RELIQUARY_SIGNER_EXPECTED_HOTKEY", "HF_TOKEN",
})


class ReconciliationRequired(RuntimeError):
    """Ownership or finality is uncertain; new work must remain closed."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _private_bytes(path: Path, limit: int = 65536) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("operator files must be regular, owned by this user and private")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("operator file exceeds its bound")
        return data
    finally:
        os.close(fd)


def _atomic_private(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + "." + secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SupervisorConfig:
    task_id: str
    order_sha256: str
    profile_id: str
    run_id: str
    state_dir: str
    credentials_file: str
    model_dir: str
    model_manifest_file: str
    model_manifest_sha256: str
    cgroup_parent: str
    validator_hotkey: str
    controller_port: int
    trainer_port: int
    trainer_device: int
    network: str
    netuid: int
    max_runtime_seconds: int
    start_seconds: int
    drain_seconds: int
    stop_seconds: int

    @property
    def runtime_dir(self) -> Path:
        return Path(self.state_dir)

    @property
    def task_dir(self) -> Path:
        return self.runtime_dir

    def credentials(self) -> dict[str, str]:
        values = strict_json_loads(_private_bytes(Path(self.credentials_file)))
        if (not isinstance(values, dict) or not set(values) <= _CREDENTIALS
                or not all(isinstance(value, str) and "\0" not in value for value in values.values())):
            raise ValueError("credentials file contains unsupported environment settings")
        signer = {key for key in _CREDENTIALS if key.startswith("RELIQUARY_SIGNER_")}
        if not signer <= set(values) or values["RELIQUARY_SIGNER_EXPECTED_HOTKEY"] != self.validator_hotkey:
            raise ValueError("an explicitly bound remote signer is required")
        return values

    @classmethod
    def read(cls, path: str | Path) -> SupervisorConfig:
        body = strict_json_loads(_private_bytes(Path(path)))
        if not isinstance(body, dict) or body.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported supervisor config")
        if set(body) != set(cls.__dataclass_fields__):
            raise ValueError("supervisor config must contain exactly the documented fields")
        out = cls(**body)
        out.validate()
        return out

    def validate(self) -> None:
        if not isinstance(self.task_id, str) or not _ID.fullmatch(self.task_id):
            raise ValueError("invalid task id")
        if not isinstance(self.profile_id, str) or not re.fullmatch(r"[\w.-]{1,128}", self.profile_id):
            raise ValueError("invalid profile id")
        if (not isinstance(self.run_id, str) or not self.run_id.startswith(self.task_id + "-")
                or not re.fullmatch(r"[a-z0-9-]{1,128}", self.run_id)
                or len(self.run_id) < len(self.task_id) + 17):
            raise ValueError("run id must pin a unique task-prefixed namespace")
        for value in (self.order_sha256, self.model_manifest_sha256):
            if not isinstance(value, str) or not _HEX.fullmatch(value):
                raise ValueError("expected an immutable SHA-256 binding")
        for field in ("state_dir", "credentials_file", "model_dir", "model_manifest_file", "cgroup_parent"):
            path = getattr(self, field)
            if not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("operator paths must be absolute")
        if Path(self.state_dir).name != self.task_id:
            raise ValueError("runtime state must use its exact task directory")
        if not isinstance(self.validator_hotkey, str) or not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{40,64}", self.validator_hotkey):
            raise ValueError("invalid validator hotkey")
        for field, lower, upper in (("controller_port", 1024, 65535), ("trainer_port", 1024, 65535),
                                    ("trainer_device", 0, 255), ("netuid", 1, 65535),
                                    ("max_runtime_seconds", 1, 60 * 86400), ("start_seconds", 1, 3600),
                                    ("drain_seconds", 1, 1800), ("stop_seconds", 1, 120)):
            value = getattr(self, field)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"invalid bounded {field}")
        if self.controller_port == self.trainer_port or self.network not in {"finney", "test", "local"}:
            raise ValueError("invalid isolated listener/network configuration")


@dataclass(frozen=True)
class StoppedProcess:
    role: str
    pid: int
    pgid: int
    start_id: str
    returncode: int
    started_at: float
    stopped_at: float


@dataclass(frozen=True)
class StoppedRuntime:
    task_id: str
    order_sha256: str
    run_id: str
    runtime_id: str
    started_at: float
    stopped_at: float
    trainer_elapsed_ns: int
    latest_checkpoint: int | None
    processes: tuple[StoppedProcess, ...]
    backend: str = "single"
    process_scope: str = PROCESS_SCOPE
    cgroup_id: str = ""
    cgroup_empty_at: float = 0.0


@dataclass(frozen=True)
class FinalizedRuntime:
    usage_sha256: str
    delivery_sha256: str


def verify_model(config: SupervisorConfig, entry) -> None:
    from reliquary.protocol.profiles import PROFILES

    raw = _private_bytes(Path(config.model_manifest_file), 1 << 20)
    if hashlib.sha256(raw).hexdigest() != config.model_manifest_sha256:
        raise ValueError("local model manifest changed")
    manifest = strict_json_loads(raw)
    profile = PROFILES[entry.profile_id]
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "profile_id", "model_id", "revision", "files"}
            or manifest["schema"] != "reliquary/local-model/v1" or manifest["profile_id"] != entry.profile_id
            or manifest["model_id"] != profile.model_id or manifest["revision"] != profile.model_revision):
        raise ValueError("local model manifest does not bind the selected immutable profile")
    files = manifest["files"]
    if (not isinstance(files, dict) or not 1 <= len(files) <= 4096 or "config.json" not in files
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("local model manifest is incomplete")
    root = Path(config.model_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("local model must be a materialized directory")
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("materialized models cannot contain symlinks")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    if observed != set(files):
        raise ValueError("local model file inventory changed")
    for name, digest in files.items():
        if (not isinstance(name, str) or len(name) > 256 or Path(name).is_absolute() or ".." in Path(name).parts
                or not isinstance(digest, str) or not _HEX.fullmatch(digest)):
            raise ValueError("invalid model file binding")
        actual = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError("local model bytes changed")


def child_environment(config: SupervisorConfig, entry, *, runtime_id: str, run_id: str, token: str) -> dict[str, str]:
    credentials = config.credentials()
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE") if key in os.environ}
    env.update(credentials)
    state = Path(config.state_dir)
    env.update({
        "HOME": str(state / "home"), "HF_HOME": str(state / "hf-cache"), "WANDB_MODE": "disabled",
        "RELIQUARY_TASK_ID": config.task_id, "RELIQUARY_PROTOCOL_PROFILE": config.profile_id,
        "RELIQUARY_STATE_DIR": str(state / "controller"), "RELIQUARY_TRAINER_STATE_DIR": str(state / "trainer"),
        "RELIQUARY_RL_MODE": "async", "RELIQUARY_RL_RUN": run_id,
        "RELIQUARY_RL_RUNTIME_ID": runtime_id, "RELIQUARY_RL_ORDER_SHA256": config.order_sha256,
        "RELIQUARY_RL_TRAINER_BIND": f"127.0.0.1:{config.trainer_port}",
        "RELIQUARY_RL_TRAINER_URL": f"http://127.0.0.1:{config.trainer_port}",
        "RELIQUARY_RL_TRAINER_TOKEN": token, "RELIQUARY_RL_VALIDATOR_HOTKEY": config.validator_hotkey,
        "RELIQUARY_RL_BASE_MODEL": config.model_dir, "RELIQUARY_RL_BATCH_GROUPS": str(entry.params["rl_batch_groups"]),
        "RELIQUARY_TRAINER_BACKEND": "single", "RELIQUARY_RL_AUDIT_SOURCE": "trainer",
        "RELIQUARY_RL_PAY_OUTAGE_FILL": "off",
        "RELIQUARY_RL_HF_BACKUP_EVERY": "", "RELIQUARY_RL_HF_BACKUP_PREFIX": "",
        "RELIQUARY_SUPERVISOR_HTTP_PORT": str(config.controller_port),
        "RELIQUARY_SUPERVISOR_NETWORK": config.network, "RELIQUARY_SUPERVISOR_NETUID": str(config.netuid),
    })
    return env


def _checkpoint_client(config: SupervisorConfig):
    import boto3
    from botocore.config import Config

    credentials = config.credentials()
    return boto3.client("s3", endpoint_url=credentials["R2_PUBLIC_ENDPOINT"],
                        aws_access_key_id=credentials["R2_PUBLIC_ACCESS_KEY_ID"],
                        aws_secret_access_key=credentials["R2_PUBLIC_SECRET_ACCESS_KEY"],
                        region_name=credentials.get("R2_REGION", "auto"),
                        config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}, proxies={})), \
        credentials.get("R2_PUBLIC_BUCKET", "reliquary-public")


def verify_empty_namespace(config: SupervisorConfig) -> None:
    """A new task may not overwrite an existing checkpoint namespace, including orphaned uploads."""
    from reliquary.shared.ckpt_index import run_prefix

    client, bucket = _checkpoint_client(config)
    try:
        listing = client.list_objects_v2(Bucket=bucket, Prefix=run_prefix(config.run_id) + "/", MaxKeys=1)
        if (not isinstance(listing, dict) or type(listing.get("KeyCount")) is not int
                or not 0 <= listing["KeyCount"] <= 1 or type(listing.get("IsTruncated")) is not bool
                or not isinstance(listing.get("Contents", []), list)):
            raise ValueError("checkpoint namespace absence is unproven")
        if listing.get("Contents") or listing.get("KeyCount", 0) or listing.get("IsTruncated", False):
            raise ValueError("checkpoint run namespace already exists")
    finally:
        client.close()


def reserve_namespace(config: SupervisorConfig, binding: dict) -> None:
    """A create-only binding arbitrates identical run configurations started on different hosts."""
    from reliquary.shared.ckpt_index import run_prefix

    client, bucket = _checkpoint_client(config)
    key = run_prefix(config.run_id) + "/runtime-claim.json"
    document = {"schema": "reliquary/training-runtime-claim/v1", **binding}
    try:
        try:
            client.put_object(Bucket=bucket, Key=key, Body=json.dumps(document, sort_keys=True).encode(),
                              ContentType="application/json", IfNoneMatch="*")
        except Exception as exc:
            try:
                response = client.get_object(Bucket=bucket, Key=key)
                try:
                    retained = strict_json_loads(response["Body"].read(4097))
                finally:
                    response["Body"].close()
                if retained == document:
                    return                      # the original create succeeded but its reply was lost
            except Exception:
                pass
            raise ReconciliationRequired("run namespace claim is conflicting or uncertain") from exc
    finally:
        client.close()


def _process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


@dataclass
class _OwnedProcess:
    role: str
    process: object
    start_id: str
    started_at: float
    started_ns: int
    exited_at: float | None = None
    exited_ns: int | None = None

    def poll(self, *, now: float | None = None, now_ns: int | None = None):
        code = self.process.poll()
        if code is not None and self.exited_at is None:
            self.exited_at = time.time() if now is None else now
            self.exited_ns = time.monotonic_ns() if now_ns is None else now_ns
        return code


class LocalBackend:
    """Linux cgroup v2 is mandatory: detached descendants cannot escape the termination proof."""

    @staticmethod
    def preflight(config: SupervisorConfig) -> None:
        if sys.platform != "linux":
            raise ValueError("managed training requires Linux with a delegated cgroup v2 parent")
        parent = Path(config.cgroup_parent)
        if parent.is_symlink() or not (parent / "cgroup.controllers").is_file():
            raise ValueError("operator must supply a delegated cgroup v2 parent")

    def __init__(self, config: SupervisorConfig, runtime_id: str):
        self.preflight(config)
        parent = Path(config.cgroup_parent)
        self.scope = parent / runtime_id
        self.scope.mkdir()                      # never reuse or adopt an existing scope
        if not all((self.scope / name).is_file() for name in ("cgroup.procs", "cgroup.kill", "cgroup.events")):
            raise ValueError("delegated cgroup lacks bounded termination support")

    def spawn(self, role: str, config: SupervisorConfig, env: dict) -> _OwnedProcess:
        env = {**env, "RELIQUARY_SUPERVISOR_CGROUP": str(self.scope),
               "CUDA_VISIBLE_DEVICES": str(config.trainer_device) if role == "trainer" else ""}
        log_path = Path(config.state_dir) / (role + ".log")
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            process = subprocess.Popen([sys.executable, "-P", "-m", "reliquary.rl_orders.supervisor", "--child-role", role],
                                       env=env, stdin=subprocess.DEVNULL, stdout=descriptor, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        finally:
            os.close(descriptor)
        started = _process_start(process.pid)
        if started is None:
            # This live Popen child handle is ours; a journal PID is never used for this fallback.
            process.kill()
            try:
                process.wait(timeout=config.stop_seconds)
            except subprocess.TimeoutExpired as exc:
                raise ReconciliationRequired("new child identity and termination are uncertain") from exc
            raise ReconciliationRequired("spawned process identity could not be established")
        return _OwnedProcess(role, process, started, time.time(), time.monotonic_ns())

    def stop(self, handles: list[_OwnedProcess], seconds: float) -> tuple[tuple[StoppedProcess, ...], float, int]:
        deadline = time.monotonic() + seconds
        forced = False
        for handle in handles:
            process = handle.process
            if handle.poll() is None:
                if _process_start(process.pid) != handle.start_id or os.getpgid(process.pid) != process.pid:
                    raise ReconciliationRequired("owned process identity changed; no signal sent")
                os.killpg(process.pid, signal.SIGTERM)
        while time.monotonic() < deadline - seconds / 2:
            if all(handle.poll() is not None for handle in handles) and self.empty():
                break
            time.sleep(min(.1, max(0, deadline - time.monotonic())))
        if not self.empty():
            forced = True
            (self.scope / "cgroup.kill").write_text("1")  # only this invocation's newly created scope
        stopped = []
        trainer_ns = 0
        for handle in handles:
            process = handle.process
            try:
                code = process.wait(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise ReconciliationRequired("owned process did not exit within its stop budget") from exc
            handle.poll()
            if handle.role == "trainer":
                trainer_ns = handle.exited_ns - handle.started_ns
            stopped.append(StoppedProcess(handle.role, process.pid, process.pid, handle.start_id, code,
                                          handle.started_at, handle.exited_at))
        while not self.empty() and time.monotonic() < deadline:
            time.sleep(.05)
        if not self.empty():
            raise ReconciliationRequired("owned cgroup still contains processes")
        for handle in handles:
            try:
                os.killpg(handle.process.pid, 0)
            except ProcessLookupError:
                continue
            raise ReconciliationRequired("owned process group absence could not be established")
        if forced:
            raise ReconciliationRequired("owned scope required forced termination; no final witness")
        return tuple(stopped), time.time(), trainer_ns

    def empty(self) -> bool:
        values = dict(line.split() for line in (self.scope / "cgroup.events").read_text().splitlines())
        if values.get("populated") not in {"0", "1"}:
            raise ReconciliationRequired("cgroup state is unavailable")
        return values["populated"] == "0"


class RuntimeClient:
    def __init__(self, config: SupervisorConfig, token: str):
        import httpx

        self.client = httpx.Client(base_url=f"http://127.0.0.1:{config.trainer_port}/rl/trainer/",
                                   headers={"Authorization": f"Bearer {token}"}, timeout=2, trust_env=False)

    def read(self) -> dict | None:
        import httpx

        try:
            response = self.client.get("runtime")
            return strict_json_loads(response.content) if response.status_code == 200 else None
        except httpx.TransportError:
            return None

    def drain(self, binding: dict) -> None:
        response = self.client.post("drain", json=binding)
        if response.status_code != 200:
            raise ReconciliationRequired("runtime did not acknowledge the exact drain request")

    def close(self) -> None:
        self.client.close()


def _checked_snapshot(snapshot, binding: dict, started_at: float, now: float) -> dict | None:
    if snapshot is None:
        return None
    if (not isinstance(snapshot, dict) or snapshot.get("schema") != RUNTIME_SCHEMA
            or any(snapshot.get(key) != value for key, value in binding.items())):
        raise ReconciliationRequired("runtime evidence names another task/order/session")
    at = snapshot.get("observed_at")
    if type(at) not in (int, float) or not math.isfinite(at) or not started_at <= at <= now + 5 or now - at > 10:
        return None
    for field in ("controller_ready", "trainer_ready", "admission_closed", "drained"):
        if type(snapshot.get(field)) is not bool:
            raise ReconciliationRequired("runtime evidence has no strict readiness/drain state")
    checkpoint = snapshot.get("latest_checkpoint")
    if checkpoint is not None and (type(checkpoint) is not int or checkpoint < 0):
        raise ReconciliationRequired("runtime checkpoint identity is invalid")
    fields = ("schema", *binding, "observed_at", "controller_ready", "trainer_ready", "admission_closed",
              "drained", "latest_checkpoint")
    observed = {key: snapshot[key] for key in fields}
    for field in ("drain_requested", "trainer_quiesced"):
        if field in snapshot:
            if type(snapshot[field]) is not bool:
                raise ReconciliationRequired("runtime evidence has no strict stop state")
            observed[field] = snapshot[field]
    return observed


def read_runtime_view(root: str | Path, task_id: str, *, now: float | None = None) -> dict:
    """A bounded public projection; a stale journal never establishes a running runtime."""
    unknown = {"state": "unknown", "managed": False}
    if not isinstance(task_id, str) or not _ID.fullmatch(task_id):
        return unknown
    directory = Path(root) / task_id
    if directory.is_symlink():
        return unknown
    try:
        journal = strict_json_loads(_private_bytes(directory / "journal.json", 1 << 20))
        if (not isinstance(journal, dict) or journal.get("schema") != SCHEMA or journal.get("task_id") != task_id
                or not _HEX.fullmatch(journal.get("order_sha256", ""))):
            return unknown
        binding = {key: journal[key] for key in ("task_id", "order_sha256", "run_id", "runtime_id")}
        observed = _checked_snapshot(journal.get("observation"), binding, journal["started_at"],
                                     time.time() if now is None else now)
        result = {**binding, "state": "unknown", "managed": False, "started_at": journal["started_at"],
                  "updated_at": journal["updated_at"]}
        if observed is not None and journal["state"] in {"running", "draining"}:
            result.update(state=journal["state"], managed=True, observed_at=observed["observed_at"],
                          latest_checkpoint=observed["latest_checkpoint"])
            if journal["state"] == "running" and not (observed["controller_ready"] and observed["trainer_ready"]):
                result.update(state="unknown", managed=False)
        return result
    except (OSError, ValueError, KeyError, TypeError, ReconciliationRequired):
        return unknown


def run_supervisor(config: SupervisorConfig, *, load_entry, finalize=None, backend_factory=LocalBackend,
                   client_factory=RuntimeClient, verify=verify_model, namespace_check=verify_empty_namespace, sleep=time.sleep,
                   clock=time.monotonic, wall=time.time, claim=reserve_namespace) -> StoppedRuntime:
    """Run one locally allocated task. Only a verified usage+delivery callback can mark it completed."""
    config.validate()
    state = Path(config.state_dir)
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.is_symlink() or state.stat().st_uid != os.geteuid() or state.stat().st_mode & 0o077:
        raise ValueError("runtime state directory must be owned and private")
    lock_fd = os.open(state / "supervisor.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    client = None
    handles = []
    backend = None
    journal = None
    cleanup_attempted = False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / "journal.json"
        if journal_path.exists() or journal_path.is_symlink():
            raise ReconciliationRequired("a prior runtime journal exists; no processes adopted or signalled")
        if any(child.name != "supervisor.lock" for child in state.iterdir()):
            raise ReconciliationRequired("pre-existing runtime state has no safe new ownership epoch")
        if backend_factory is LocalBackend:
            LocalBackend.preflight(config)
        entry = load_entry(config.task_id)
        order = None if entry is None else entry.params.get("training_order")
        if (not isinstance(order, dict) or entry.status != "active" or entry.profile_id != config.profile_id
                or order.get("order_sha256") != config.order_sha256):
            raise ValueError("registry does not authorize this exact training order")
        from reliquary.validator.rl_training_usage import BILLING_BASIS

        if (not isinstance(order.get("quote_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}", order["quote_id"])
                or not isinstance(order.get("quote_sha256"), str) or not _HEX.fullmatch(order["quote_sha256"])
                or order.get("billing_basis") != BILLING_BASIS):
            raise ValueError("managed training requires the complete immutable quote and billing basis")
        now = wall()
        remaining = float(order["deadline_at"]) - now
        if not math.isfinite(remaining) or remaining <= 0:
            raise ValueError("training order deadline already passed")
        verify(config, entry)
        namespace_check(config)
        if float(order["deadline_at"]) <= wall():
            raise ValueError("training order deadline passed during preparation")
        runtime_id = secrets.token_hex(16)
        run_id = config.run_id
        binding = {"task_id": config.task_id, "order_sha256": config.order_sha256,
                   "run_id": run_id, "runtime_id": runtime_id}
        env = child_environment(config, entry, runtime_id=runtime_id, run_id=run_id, token=(token := secrets.token_urlsafe(48)))
        for folder in ("controller", "trainer", "home", "hf-cache"):
            (state / folder).mkdir(mode=0o700, exist_ok=False)
        journal = {"schema": SCHEMA, **binding, "config_sha256": _digest(asdict(config)), "state": "prepared",
                   "started_at": wall(), "processes": [], "paid_admission": False}

        def record(status: str, **fields) -> None:
            journal.update(state=status, updated_at=wall(), **fields)
            _atomic_private(journal_path, journal)

        record("prepared")
        claim(config, binding)
        if float(order["deadline_at"]) <= wall():
            raise ReconciliationRequired("training order deadline passed before owned process launch")
        backend = backend_factory(config, runtime_id)
        client = client_factory(config, token)
        started = clock()
        run_deadline = started + min(config.max_runtime_seconds, max(0, float(order["deadline_at"]) - wall()))

        def spawn(role: str) -> None:
            record("starting_" + role)
            handle = backend.spawn(role, config, env)
            handles.append(handle)
            record("starting_" + role, processes=[{"role": h.role, "pid": h.process.pid, "pgid": h.process.pid,
                                                  "start_id": h.start_id, "started_at": h.started_at} for h in handles])

        def wait_for(predicate, deadline: float, *, draining=False):
            while clock() < deadline:
                snapshot = _checked_snapshot(client.read(), binding, journal["started_at"], wall())
                for handle in handles:
                    code = handle.poll(now=wall(), now_ns=int(clock() * 1e9))
                    allowed_exit = handle.role == "trainer" and code == 0 and (
                        draining or (snapshot is not None and snapshot["admission_closed"]))
                    if code is not None and not allowed_exit:
                        raise ReconciliationRequired("runtime child exited before its task-bound acknowledgement")
                if snapshot is not None and wall() - journal.get("updated_at", 0) >= 1:
                    record(journal["state"], observation=snapshot)
                if snapshot is not None and predicate(snapshot):
                    return snapshot
                sleep(min(.25, max(0, deadline - clock())))
            raise ReconciliationRequired("runtime did not produce bounded task-bound readiness/drain evidence")

        spawn("controller")
        start_deadline = min(started + config.start_seconds, run_deadline)
        snapshot = wait_for(lambda s: s["controller_ready"], start_deadline)
        # A new controller fences miner leases while it waits for trainer readiness.
        awaiting_trainer = (not snapshot["trainer_ready"] and snapshot.get("drain_requested") is False
                            and snapshot.get("trainer_quiesced") is False)
        stopped = snapshot.get("drain_requested") is True or snapshot.get("trainer_quiesced") is True
        if stopped or (snapshot["admission_closed"] and not awaiting_trainer) or float(order["deadline_at"]) <= wall():
            raise ReconciliationRequired("training admission closed before trainer launch")
        spawn("trainer")
        snapshot = wait_for(lambda s: s["controller_ready"] and s["trainer_ready"], start_deadline)
        record("running", observation=snapshot)
        while clock() < run_deadline:
            snapshot = wait_for(lambda s: True, min(clock() + 5, run_deadline))
            if snapshot["admission_closed"]:
                break
            if not (snapshot["controller_ready"] and snapshot["trainer_ready"]):
                raise ReconciliationRequired("task-bound runtime readiness was lost")
            sleep(min(.25, max(0, run_deadline - clock())))
        record("draining")
        client.drain(binding)
        snapshot = wait_for(lambda s: s["admission_closed"] and s["drained"], clock() + config.drain_seconds, draining=True)
        record("stopping", latest_checkpoint=snapshot["latest_checkpoint"])
        stop_deadline = clock() + config.stop_seconds
        trainer = next(handle for handle in handles if handle.role == "trainer")
        while trainer.poll(now=wall(), now_ns=int(clock() * 1e9)) is None and clock() < stop_deadline:
            sleep(min(.1, max(0, stop_deadline - clock())))
        if trainer.poll(now=wall(), now_ns=int(clock() * 1e9)) != 0:
            raise ReconciliationRequired("trainer did not exit normally after its quiescence acknowledgement")
        cleanup_attempted = True
        processes, empty_at, trainer_ns = backend.stop(handles, max(.001, stop_deadline - clock()))
        if any(process.returncode != 0 for process in processes):
            raise ReconciliationRequired("runtime termination was not a clean, drained exit")
        witness = StoppedRuntime(**binding, started_at=journal["started_at"], stopped_at=wall(),
                                 trainer_elapsed_ns=trainer_ns, latest_checkpoint=snapshot["latest_checkpoint"],
                                 processes=processes, cgroup_id=runtime_id, cgroup_empty_at=empty_at)
        record("execution_stopped", stopped_runtime=asdict(witness))
        if finalize is not None:
            result = finalize(entry, witness, config)
            if (not isinstance(result, FinalizedRuntime) or not _HEX.fullmatch(result.usage_sha256)
                    or not _HEX.fullmatch(result.delivery_sha256)):
                raise ReconciliationRequired("usage and checkpoint delivery were not both finalized")
            record("completed", finalization=asdict(result))
        return witness
    except BaseException:
        if journal is not None:
            try:
                if backend is not None and not cleanup_attempted:
                    cleanup_attempted = True
                    backend.stop(handles, config.stop_seconds)
            except BaseException:
                pass
            try:
                journal.update(state="reconciliation_required", updated_at=wall())
                _atomic_private(state / "journal.json", journal)
            except BaseException:
                pass                           # retained launch intent still forbids adoption/relaunch
        raise
    finally:
        if client is not None:
            client.close()
        os.close(lock_fd)


def _child(role: str) -> None:
    if role not in {"controller", "trainer"}:
        raise ValueError("unsupported child role")
    runtime_id = os.environ["RELIQUARY_RL_RUNTIME_ID"]
    scope = Path(os.environ["RELIQUARY_SUPERVISOR_CGROUP"])
    if not re.fullmatch(r"[0-9a-f]{32}", runtime_id) or scope.name != runtime_id:
        raise ValueError("invalid owned process scope")
    (scope / "cgroup.procs").write_text(str(os.getpid()))
    command = [sys.executable, "-P", "-m", "reliquary.cli.main"]
    if role == "controller":
        command += ["validate", "--no-set-weights", "--http-host", "127.0.0.1", "--http-port",
                    os.environ["RELIQUARY_SUPERVISOR_HTTP_PORT"], "--checkpoint", os.environ["RELIQUARY_RL_BASE_MODEL"],
                    "--network", os.environ["RELIQUARY_SUPERVISOR_NETWORK"], "--netuid", os.environ["RELIQUARY_SUPERVISOR_NETUID"]]
    elif role == "trainer":
        command += ["train-async"]
    os.execve(sys.executable, command, os.environ)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--child-role", choices=("controller", "trainer"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_role:
        _child(args.child_role)
        return
    if args.config is None:
        parser.error("--config is required")
    config = SupervisorConfig.read(args.config)

    def load(task_id):
        import boto3
        from botocore.config import Config
        from reliquary.infrastructure.task_registry_store import REGISTRY_KEY
        from reliquary.shared.task_registry import parse_registry

        credentials = config.credentials()
        client = boto3.client("s3", endpoint_url=credentials.get("R2_ENDPOINT_URL") or
                              f"https://{credentials['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                              aws_access_key_id=credentials["R2_ACCESS_KEY_ID"],
                              aws_secret_access_key=credentials["R2_SECRET_ACCESS_KEY"],
                              region_name=credentials.get("R2_REGION", "auto"),
                              config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}, proxies={}))
        try:
            response = client.get_object(Bucket=credentials["R2_BUCKET_ID"], Key=REGISTRY_KEY)
            try:
                raw = response["Body"].read((1 << 20) + 1)
            finally:
                response["Body"].close()
            if len(raw) > 1 << 20:
                raise ValueError("registry exceeds its managed runtime bound")
            return parse_registry(raw, strict=True).get(task_id)
        finally:
            client.close()

    from reliquary.rl_orders.finalize import finalize_training

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        witness = run_supervisor(config, load_entry=load, finalize=finalize_training)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({"state": "completed", "runtime_id": witness.runtime_id, "paid_admission": False}))


if __name__ == "__main__":
    # Keep receipt classes identical when the CLI and finalization module import each other.
    from reliquary.rl_orders.supervisor import main as entrypoint

    entrypoint()
