"""Supervise a pinned upstream miner without importing its inference runtime.

The upstream signed-source bootstrap owns source admission, checkpoint loading,
proofs and uploads. A successful child exit is not an acceptance or training receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

from reliquary.shared.strict_json import strict_json_loads


class AffineRuntimeError(RuntimeError):
    """Operator-safe error: no credential, discovery URL or child output included."""


@dataclass(frozen=True)
class AffineConfig:
    upstream_checkout: Path = field(repr=False)
    upstream_revision: str
    python: Path = field(repr=False)
    authority: str
    current_url: str = field(repr=False)
    state_dir: Path = field(repr=False)
    source_cache_dir: Path = field(repr=False)
    key_file: Path | None = field(default=None, repr=False)
    cap_file: Path | None = field(default=None, repr=False)
    env_id: str | None = None
    indices: tuple[int, ...] | None = None
    search_budget: int = 32
    max_batches: int = 1
    process_timeout_seconds: float = 3600.0
    retry_attempts: int = 1
    retry_seconds: float = 10.0
    max_log_bytes: int = 8 * 1024 * 1024
    manifest_snapshot_file: Path | None = field(default=None, repr=False)


def _path(value, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise AffineRuntimeError(f"{name} requires an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AffineRuntimeError(f"{name} requires an absolute path without parent traversal")
    return path


def _private_path(path: Path) -> None:
    try:
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise AffineRuntimeError("private paths cannot have symlink components")
            if (parent / ".git").exists():
                raise AffineRuntimeError("private files and state must stay outside repositories")
    except OSError:
        raise AffineRuntimeError("unable to inspect a private runtime path") from None


def _private_file(path: Path) -> None:
    _private_path(path)
    try:
        info = path.stat()
    except OSError:
        raise AffineRuntimeError("an existing regular private file is required") from None
    if not stat.S_ISREG(info.st_mode):
        raise AffineRuntimeError("an existing regular private file is required")
    if info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise AffineRuntimeError("private files must be owner-only and owned by this account")


def _positive(value, name: str, maximum: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise AffineRuntimeError(f"{name} is outside its supported bounds")


def validate_config(config: AffineConfig) -> AffineConfig:
    """Validate operator input locally; never read a key or contact discovery."""
    if not isinstance(config.upstream_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", config.upstream_revision):
        raise AffineRuntimeError("upstream_revision must be an immutable 40-character revision")
    if not isinstance(config.authority, str) or not re.fullmatch(r"[0-9a-f]{64}", config.authority):
        raise AffineRuntimeError("authority must be the known Ed25519 public key")
    try:
        if not isinstance(config.current_url, str) or len(config.current_url) > 16384 \
                or any(ord(c) <= 32 for c in config.current_url):
            raise ValueError
        url = urlsplit(config.current_url)
        query = parse_qs(url.query)
        if (url.scheme != "https" or not (url.hostname or "").endswith(".r2.cloudflarestorage.com")
                or url.username or url.password or url.fragment or url.port not in (None, 443)
                or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
                or len(query.get("X-Amz-Signature", [])) != 1):
            raise ValueError
    except ValueError:
        raise AffineRuntimeError("current_url must be a signed direct R2 HTTPS discovery URL") from None
    for name in ("upstream_checkout", "python", "state_dir", "source_cache_dir"):
        _path(getattr(config, name), name)
    if not Path(config.upstream_checkout).is_dir() or not Path(config.python).is_file() \
            or not os.access(config.python, os.X_OK):
        raise AffineRuntimeError("upstream checkout and runtime interpreter must exist")
    if (config.key_file is None) == (config.cap_file is None):
        raise AffineRuntimeError("supply exactly one existing key_file or cap_file")
    _private_file(_path(config.key_file or config.cap_file, "credential file"))
    if config.manifest_snapshot_file is not None:
        _private_file(_path(config.manifest_snapshot_file, "manifest snapshot file"))
        if Path(config.manifest_snapshot_file).samefile(config.key_file or config.cap_file):
            raise AffineRuntimeError("manifest snapshot must be separate from the credential file")
        if Path(config.manifest_snapshot_file).stat().st_size > 64 * 1024 * 1024:
            raise AffineRuntimeError("manifest snapshot exceeds its supported bound")
    state, cache = Path(config.state_dir), Path(config.source_cache_dir)
    for path in (state, cache):
        _private_path(path)
        if path.exists() and (not path.is_dir() or path.stat().st_mode & 0o077
                              or path.stat().st_uid != os.getuid()):
            raise AffineRuntimeError("state and source cache must be owner-only directories")
    if state == cache or state in cache.parents or cache in state.parents:
        raise AffineRuntimeError("state and source cache must be separate directories")
    if config.env_id is not None and (not isinstance(config.env_id, str)
                                    or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", config.env_id)):
        raise AffineRuntimeError("env_id must be a bounded environment identifier")
    if config.indices is not None and (config.env_id is None or not isinstance(config.indices, (tuple, list))
            or not config.indices or len(config.indices) > 10000
            or any(type(i) is not int or i < 0 for i in config.indices)
            or len(set(config.indices)) != len(config.indices)):
        raise AffineRuntimeError("indices require an environment and unique non-negative integers")
    for name, maximum in (("search_budget", 128), ("max_batches", 10000), ("retry_attempts", 10),
                          ("max_log_bytes", 64 * 1024 * 1024)):
        if type(getattr(config, name)) is not int:
            raise AffineRuntimeError(f"{name} must be an integer")
        _positive(getattr(config, name), name, maximum)
    _positive(config.process_timeout_seconds, "process_timeout_seconds", 86400)
    _positive(config.retry_seconds, "retry_seconds", 300)
    return config


def load_config(path: str | Path) -> AffineConfig:
    path = _path(path, "config")
    _private_file(path)
    try:
        with path.open("rb") as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise ValueError
        document = strict_json_loads(raw)
        if not isinstance(document, dict) or set(document) - {f.name for f in fields(AffineConfig)}:
            raise ValueError
        for name in ("upstream_checkout", "python", "state_dir", "source_cache_dir", "key_file", "cap_file",
                     "manifest_snapshot_file"):
            if document.get(name) is not None:
                document[name] = _path(document[name], name)
        if "indices" in document and document["indices"] is not None:
            if not isinstance(document["indices"], list):
                raise ValueError
            document["indices"] = tuple(document["indices"])
        return validate_config(AffineConfig(**document))
    except (ValueError, TypeError, OSError, UnicodeError):
        raise AffineRuntimeError("private runtime config is malformed or unreadable") from None


def verify_checkout(config: AffineConfig) -> None:
    """Refuse a different or edited bootstrap before any upstream execution."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(config.upstream_checkout), *args],
                                           stderr=subprocess.DEVNULL, env=environment, timeout=15)
        if git("rev-parse", "HEAD").decode().strip() != config.upstream_revision \
                or git("status", "--porcelain", "--untracked-files=all"):
            raise AffineRuntimeError("upstream checkout must be clean at the pinned revision")
        bootstrap = Path(config.upstream_checkout) / "subnet/source_bootstrap.py"
        if bootstrap.is_symlink() or not bootstrap.is_file() \
                or bootstrap.read_bytes() != git("show", f"{config.upstream_revision}:subnet/source_bootstrap.py"):
            raise AffineRuntimeError("pinned upstream source bootstrap is missing or modified")
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise AffineRuntimeError("unable to verify the pinned upstream checkout") from None


def _directory(path: Path) -> None:
    try:
        missing = []
        parent = path
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for parent in reversed(missing):
            parent.mkdir(mode=0o700, exist_ok=True)
        _private_path(path)
        if path.stat().st_mode & 0o077 or path.stat().st_uid != os.getuid():
            raise AffineRuntimeError("runtime directories must remain owner-only")
    except OSError:
        raise AffineRuntimeError("unable to create private runtime directories") from None


def _private_open(path: Path, flags: int) -> int:
    try:
        fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError:
        raise AffineRuntimeError("unable to open a private runtime file") from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        os.close(fd)
        raise AffineRuntimeError("runtime files must be regular, owner-only and owned by this account")
    return fd


@contextmanager
def _lock(directory: Path):
    path = directory / "runner.lock"
    fd = _private_open(path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AffineRuntimeError("another runner owns this state directory") from None
        yield
    finally:
        os.close(fd)


def _terminate(process: subprocess.Popen) -> None:
    """Stop the complete session before a terminal journal entry is permitted."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _read_journal(state: Path) -> dict | None:
    path = state / "runner.json"
    if not path.exists() and not path.is_symlink():
        return None
    fd = _private_open(path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "rb") as handle:
            document = strict_json_loads(handle.read(16385))
        if not isinstance(document, dict) or document.get("schema") != "affine-runtime/v1" \
                or document.get("stage") not in ("running", "reconciled", "bootstrap_completed", "failed",
                                                 "cancelled", "budget_exhausted", "log_limit_exceeded"):
            raise ValueError
        return document
    except (ValueError, TypeError, OSError, UnicodeError):
        raise AffineRuntimeError("private runner journal needs explicit reconciliation") from None


def _write_journal(state: Path, document: dict) -> None:
    name = None
    try:
        fd, name = tempfile.mkstemp(dir=state, prefix=".runner-")
        with os.fdopen(fd, "w") as handle:
            json.dump(document, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, state / "runner.json")
    except OSError:
        raise AffineRuntimeError("unable to persist the private runner journal") from None
    finally:
        if name is not None:
            try:
                Path(name).unlink(missing_ok=True)
            except OSError:
                pass


def reconcile_state(config: AffineConfig) -> dict:
    """Explicitly clear a crashed runner only after its recorded group no longer exists.

    This never kills an unknown process. Missing group metadata or a live group
    requires operator inspection; a new runner cannot silently resume it.
    """
    validate_config(config)
    state = Path(config.state_dir)
    if not state.is_dir():
        raise AffineRuntimeError("no private runner state exists to reconcile")
    with _lock(state):
        document = _read_journal(state)
        if document is None or document.get("stage") != "running":
            raise AffineRuntimeError("no unresolved running journal exists to reconcile")
        group = document.get("process_group")
        if type(group) is not int or group <= 1:
            raise AffineRuntimeError("runner process group is unknown; operator inspection is required")
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            document["stage"] = "reconciled"
            _write_journal(state, document)
            return {"schema": "affine-runtime/v1", "stage": "reconciled"}
        except OSError:
            raise AffineRuntimeError("unable to confirm that the previous process group has stopped") from None
        raise AffineRuntimeError("previous process group still exists; reconciliation refused")


class AffineRunner:
    def __init__(self, config: AffineConfig):
        self.config = validate_config(config)

    def _command(self) -> list[str]:
        c = self.config
        if c.manifest_snapshot_file is None:
            command = [str(c.python), "-I", "-B", str(Path(c.upstream_checkout) / "subnet/source_bootstrap.py"),
                       "--current-url", c.current_url]
        else:
            fd = _private_open(Path(c.manifest_snapshot_file), os.O_RDONLY)
            with os.fdopen(fd, "rb") as snapshot:
                digest = hashlib.file_digest(snapshot, "sha256").hexdigest()
            command = [str(c.python), "-I", "-B", str(Path(__file__).with_name("affine_epoch.py")),
                       "--upstream-checkout", str(c.upstream_checkout),
                       "--snapshot-file", str(c.manifest_snapshot_file), "--snapshot-sha256", digest]
        command += ["--authority", c.authority,
                   "--source-cache", str(c.source_cache_dir), "--state", str(c.state_dir),
                   "--key" if c.key_file else "--cap-file", str(c.key_file or c.cap_file),
                   "--once", "--search-budget", str(c.search_budget), "--max-batches", str(c.max_batches)]
        if c.env_id is not None:
            command += ["--env-id", c.env_id]
        if c.indices is not None:
            command += ["--indices", *map(str, c.indices)]
        return command

    def _execute(self, environment: dict, deadline: float, stop: threading.Event, record) -> tuple[str | None, int]:
        c = self.config
        fd = _private_open(Path(c.state_dir) / "runtime.log", os.O_APPEND | os.O_CREAT | os.O_WRONLY)
        with os.fdopen(fd, "ab", buffering=0) as log:
            remaining = c.max_log_bytes - os.fstat(log.fileno()).st_size
            if remaining <= 0:
                return "log_limit_exceeded", 0
            try:
                process = subprocess.Popen(self._command(), cwd=c.state_dir, env=environment,
                                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except OSError:
                raise AffineRuntimeError("unable to start the pinned upstream runtime") from None
            reason = None
            try:
                record("running", process)
                os.set_blocking(process.stdout.fileno(), False)
                child_deadline = min(deadline, time.monotonic() + c.process_timeout_seconds)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    try:
                        while True:
                            if stop.is_set():
                                reason = "cancelled"
                                break
                            if time.monotonic() >= child_deadline:
                                reason = "budget_exhausted"
                                break
                            ready = selector.select(0.1)
                            for key, _ in ready:
                                data = os.read(key.fd, 65536)
                                if not data:
                                    selector.unregister(key.fileobj)
                                    continue
                                log.write(data[:remaining])
                                remaining -= min(len(data), remaining)
                                if remaining == 0:
                                    reason = "log_limit_exceeded"
                                    break
                            if reason is not None or (process.poll() is not None and not selector.get_map()):
                                break
                            # An exited leader may have left descendants holding the output pipe.
                            if process.poll() is not None and not ready:
                                break
                    except KeyboardInterrupt:
                        reason = "cancelled"
                    except OSError:
                        raise AffineRuntimeError("unable to capture private upstream diagnostics") from None
            finally:
                try:
                    _terminate(process)
                except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
                    raise AffineRuntimeError("upstream cleanup is unconfirmed; running state retained") from None
                finally:
                    process.stdout.close()
            return reason, process.returncode

    def run(self, *, once: bool = True, stop_event: threading.Event | None = None,
            max_seconds: float | None = None, max_cycles: int | None = None) -> dict:
        """Repeat native one-pass bootstraps within an explicit total time budget."""
        c = self.config
        max_seconds = c.process_timeout_seconds if max_seconds is None else max_seconds
        _positive(max_seconds, "max_seconds", 7 * 86400)
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise AffineRuntimeError("max_cycles must be a positive integer")
        stop = stop_event or threading.Event()
        validate_config(c)
        verify_checkout(c)
        _directory(Path(c.state_dir))
        _directory(Path(c.source_cache_dir))
        state = Path(c.state_dir)
        deadline = time.monotonic() + max_seconds
        cycles = attempts = 0
        exit_code = None

        def result(stage, process=None):
            value = {"schema": "affine-runtime/v1", "stage": stage, "cycles": cycles,
                     "attempts": attempts, "exit_code": exit_code,
                     "uploaded": None, "accepted": None, "trained": None}
            journal = dict(value)
            if stage == "running":
                journal.update(owner_pid=os.getpid(), child_pid=process.pid if process else None,
                               process_group=process.pid if process else None)
            _write_journal(state, journal)
            return value

        # Never inherit unrelated host tokens or Python import configuration.
        environment = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES",
                       "LD_LIBRARY_PATH", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE") if k in os.environ}
        environment.update(HF_HOME=str(state / "model-cache"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with _lock(state):
            previous = _read_journal(state)
            if previous is not None and previous["stage"] == "running":
                raise AffineRuntimeError("previous running state is unresolved; explicit reconciliation is required")
            while True:
                try:
                    if stop.is_set():
                        return result("cancelled")
                    if time.monotonic() >= deadline:
                        return result("budget_exhausted")
                    verify_checkout(c)
                    cycles += 1
                    for attempt in range(c.retry_attempts):
                        attempts += 1
                        result("running")
                        # Diagnostics stay in private state; never echo upstream tracebacks or presigned URLs.
                        reason, exit_code = self._execute(environment, deadline, stop, result)
                        if reason is not None:
                            return result(reason)
                        if exit_code == 0:
                            break
                        if attempt + 1 >= c.retry_attempts:
                            return result("failed")
                        if stop.wait(min(c.retry_seconds, max(0, deadline - time.monotonic()))):
                            return result("cancelled")
                        if time.monotonic() >= deadline:
                            return result("budget_exhausted")
                    if once or (max_cycles is not None and cycles >= max_cycles):
                        return result("bootstrap_completed")
                    if stop.wait(min(c.retry_seconds, max(0, deadline - time.monotonic()))):
                        return result("cancelled")
                except KeyboardInterrupt:
                    return result("cancelled")


__all__ = ["AffineConfig", "AffineRunner", "AffineRuntimeError", "load_config", "validate_config", "verify_checkout",
           "reconcile_state"]
