"""The Miles Tinker gateway the async trainer drives under ``RELIQUARY_TRAINER_BACKEND=miles``.

The trainer starts the gateway itself, on this box's GPUs, from the weights it resumes from (the bucket's latest
checkpoint, or the base model): full fine-tuning, training only (no SGLang: miners sample the published checkpoints),
with the trainer's health gate (``--tinker-grad-norm-skip``). The launcher is a Miles launcher script (the lab's
``serve_*_tinker.py`` shape: ``serve --mode full --no-inference --hf-checkpoint ... --extra-args ...``).

Settings: ``RELIQUARY_MILES_LAUNCHER``, ``RELIQUARY_MILES_MODEL_TYPE`` (Megatron model type, e.g. ``qwen3.5-9B``),
``RELIQUARY_MILES_OUTPUT_DIR`` (checkpoints and exports: the trainer reads the exports there), and
``RELIQUARY_MILES_TP`` (2), ``RELIQUARY_MILES_GPUS`` (8), ``RELIQUARY_MILES_PORT`` (10613),
``RELIQUARY_MILES_READY_S`` (1800), ``RELIQUARY_MILES_CHUNK`` (64: rollouts per ``forward_backward`` call, the
trainer's side), ``RELIQUARY_MILES_EXTRA_ARGS`` (more gateway flags), ``RELIQUARY_MILES_PYTHON``
(the gateway's interpreter: Miles, Megatron and the same ``reliquary`` as this trainer, checked at start).
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

READY_LINE = "tinker gateway serving"
API_KEY = "tml-reliquary-trainer"


def miles_settings(env: Mapping[str, str] = os.environ) -> dict:
    required = {}
    for name in ("RELIQUARY_MILES_LAUNCHER", "RELIQUARY_MILES_MODEL_TYPE", "RELIQUARY_MILES_OUTPUT_DIR"):
        value = str(env.get(name, "")).strip()
        if not value:
            raise ValueError(f"{name} is required by RELIQUARY_TRAINER_BACKEND=miles")
        required[name] = value

    def number(name, default, kind=int):
        raw = str(env.get(name, "")).strip()
        value = kind(raw) if raw else default
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    return {"launcher": required["RELIQUARY_MILES_LAUNCHER"], "model_type": required["RELIQUARY_MILES_MODEL_TYPE"],
            "output_dir": Path(required["RELIQUARY_MILES_OUTPUT_DIR"]),
            "tp": number("RELIQUARY_MILES_TP", 2), "gpus": number("RELIQUARY_MILES_GPUS", 8),
            "port": number("RELIQUARY_MILES_PORT", 10613), "ready_s": number("RELIQUARY_MILES_READY_S", 1800.0, float),
            "chunk": number("RELIQUARY_MILES_CHUNK", 64),
            "extra_args": str(env.get("RELIQUARY_MILES_EXTRA_ARGS", "")).strip(),
            "python": str(env.get("RELIQUARY_MILES_PYTHON", "")).strip() or sys.executable}


def gateway_command(model_path, settings: dict, *, base_model: str, grad_norm_skip: float) -> list[str]:
    extra = f"--tinker-base-model {base_model} --tinker-grad-norm-skip {float(grad_norm_skip)}"
    if settings["extra_args"]:
        extra += " " + settings["extra_args"]
    gpus = str(settings["gpus"])
    return [settings["python"], settings["launcher"], "serve", "--mode", "full", "--no-inference",
            "--hf-checkpoint", str(model_path), "--model-type", settings["model_type"], "--tp", str(settings["tp"]),
            "--actor-num-gpus", gpus, "--num-gpus-per-node", gpus, "--tinker-port", str(settings["port"]),
            "--output-dir", str(settings["output_dir"]), "--extra-args", extra]


def _ray_stop() -> None:
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True, timeout=120)


class GatewayProcess:
    """The gateway as a child process group, its output in ``log_path``."""

    def __init__(self, command: list[str], *, log_path: Path, env: dict, cleanup: Callable[[], None] = _ray_stop,
                 base_url: str = "", healthy: Callable[[], bool] | None = None) -> None:
        self.log_path, self.cleanup, self.base_url = Path(log_path), cleanup, base_url
        self.healthy = healthy or self._healthz
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():               # a previous run's ready line must never read as this run's
            self.log_path.replace(self.log_path.with_name(self.log_path.name + ".prev"))
        self._log = open(self.log_path, "wb")
        logger.info("starting the Miles gateway: %s", shlex.join(command))
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)

    def wait_ready(self, *, timeout_s: float, poll_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # the line is logged before the server binds its port: the health route decides
            if READY_LINE in self.log_path.read_text(errors="replace") and self.healthy():
                logger.info("Miles gateway ready")
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"the Miles gateway exited ({self.process.returncode}); see {self.log_path}")
            time.sleep(poll_s)
        self.stop()
        raise RuntimeError(f"the Miles gateway was not ready in {timeout_s:.0f}s; see {self.log_path}")

    def _healthz(self) -> bool:
        import httpx

        try:
            return httpx.get(f"{self.base_url}/api/v1/healthz", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=30)
        self._log.close()
        self.cleanup()                          # Ray workers outlive the launcher's process group


def _port_taken(port: int) -> bool:
    import socket

    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", int(port))) == 0


def launch_gateway(model_path, settings: dict, *, base_model: str, grad_norm_skip: float, env: dict) -> GatewayProcess:
    """A gateway nobody stopped (the trainer killed) holds the GPUs and the port: Ray is stopped first, and a port
    still answering refuses the launch (never train on a stale gateway's weights)."""
    _ray_stop()
    if _port_taken(settings["port"]):
        raise RuntimeError(f"port {settings['port']} is already in use: a previous gateway is still running")
    gateway = GatewayProcess(gateway_command(model_path, settings, base_model=base_model, grad_norm_skip=grad_norm_skip),
                             log_path=settings["output_dir"] / "gateway.log", env=env,
                             base_url=f"http://127.0.0.1:{settings['port']}")
    try:
        gateway.wait_ready(timeout_s=settings["ready_s"])
    except BaseException:
        gateway.stop()
        raise
    return gateway


# what the gateway's reliquary_gate reads from its own ``reliquary``: must be the trainer's
_GATE_CONSTANTS = ("PROTOCOL_VERSION", "RL_ROLLOUT_MATCH_FLOOR", "RL_GROUP_MATCH_FLOOR", "RL_FORGERY_MATCH_FLOOR",
                   "RL_GAP_TOLERANCE", "RL_SUBSTITUTE_GAP", "RL_TERMINAL_GAP", "FORCED_SEED_STOCHASTIC_MAXPROB",
                   "FORCED_SEED_ROLLOUT_MIN_STOCH")
_CONSTANTS_SCRIPT = """
import json, torch
from reliquary import constants as c
from reliquary.environment.forced_sampling import gumbel_noise
out = {name: getattr(c, name) for name in %r}
out["gumbel_noise"] = [round(float(v), 9) for v in gumbel_noise(torch.tensor([12345, -7]), torch.arange(4))[0]]
print(json.dumps(out))
"""


def gate_constants() -> dict:
    """This process's gate constants and a Gumbel-noise probe."""
    out = subprocess.run([sys.executable, "-c", _CONSTANTS_SCRIPT % (_GATE_CONSTANTS,)], capture_output=True,
                         text=True, check=True, timeout=300)
    return json.loads(out.stdout.strip().splitlines()[-1])


def check_gateway_constants(python: str, *, local: dict | None = None) -> None:
    """Refuse a gateway whose ``reliquary`` would judge differently from this trainer (another commit or profile)."""
    local = gate_constants() if local is None else local
    out = subprocess.run([python, "-c", _CONSTANTS_SCRIPT % (_GATE_CONSTANTS,)], capture_output=True, text=True,
                         timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"the gateway's python cannot import reliquary: {out.stderr[-500:]}")
    remote = json.loads(out.stdout.strip().splitlines()[-1])
    differ = sorted(k for k in local if remote.get(k) != local[k])
    if differ:
        raise RuntimeError(f"the gateway's reliquary differs from the trainer's: {differ}")


def create_full_training_client(base_url: str, base_model: str, api_key: str = API_KEY):
    """A full-model Tinker training client (the SDK has no full-model helper: ``create_model`` with the gateway's
    ``parameterization`` extension)."""
    import httpx
    import tinker

    service = tinker.ServiceClient(base_url=base_url, api_key=api_key)
    model_seq_id = service.holder.get_training_client_id()
    with httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=60) as http:
        response = http.post("/api/v1/create_model", json={
            "session_id": service.holder.get_session_id(), "model_seq_id": model_seq_id,
            "base_model": base_model, "parameterization": {"type": "full"}})
        response.raise_for_status()
        created = response.json()
        deadline = time.monotonic() + 900
        while True:
            poll = http.post("/api/v1/retrieve_future", json={"request_id": created["request_id"]})
            poll.raise_for_status()
            result = poll.json()
            if "error" in result:
                raise RuntimeError(f"create_model failed: {result}")
            if result.get("type") != "try_again":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("create_model timed out")
            time.sleep(1)
    return tinker.TrainingClient(service.holder, model_seq_id=model_seq_id, model_id=created["model_id"])


class _WatchedFuture:
    def __init__(self, future, alive: Callable[[], bool], poll_s: float) -> None:
        self._future, self._alive, self._poll_s = future, alive, poll_s

    def result(self, timeout: float | None = None):
        import concurrent.futures

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = self._poll_s if deadline is None else max(0.0, min(self._poll_s, deadline - time.monotonic()))
            try:
                return self._future.result(timeout=wait)
            except (concurrent.futures.TimeoutError, TimeoutError):
                if not self._alive():
                    raise RuntimeError("the Miles gateway exited while a request was pending") from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise


class WatchedClient:
    """The Tinker client with every wait bounded by the gateway's life: the SDK retries a refused connection without
    end, so a dead gateway (an OOM in a Ray actor) would hang the trainer instead of failing its step."""

    def __init__(self, client, *, alive: Callable[[], bool], poll_s: float = 15.0) -> None:
        self._client, self._alive, self._poll_s = client, alive, poll_s

    def __getattr__(self, name):
        value = getattr(self._client, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            out = value(*args, **kwargs)
            return _WatchedFuture(out, self._alive, self._poll_s) if hasattr(out, "result") else out

        return call


def checkpoint_deleter(base_url: str, api_key: str = API_KEY) -> Callable[[str], None]:
    """``delete(tinker_path)`` through the gateway's REST route; a failure is logged (the export is only disk)."""
    import httpx

    def delete(path: str) -> None:
        model_id, rest = path.removeprefix("tinker://").split("/", 1)
        try:
            httpx.delete(f"{base_url}/api/v1/training_runs/{model_id}/checkpoints/{rest}",
                         headers={"X-API-Key": api_key}, timeout=120).raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("could not delete the gateway export %s: %s", path, exc)

    return delete


__all__ = ["GatewayProcess", "WatchedClient", "check_gateway_constants", "checkpoint_deleter", "gate_constants", "create_full_training_client", "gateway_command",
           "launch_gateway", "miles_settings"]
