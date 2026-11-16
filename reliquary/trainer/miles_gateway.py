"""The Miles Tinker gateway the async trainer drives under ``RELIQUARY_TRAINER_BACKEND=miles``.

The trainer starts the gateway itself, on this box's GPUs, from the weights it resumes from (the bucket's latest
checkpoint, or the base model): full fine-tuning, training only (no SGLang: miners sample the published checkpoints),
with the trainer's health gate (``--tinker-grad-norm-skip``). The launcher is a Miles launcher script (the lab's
``serve_*_tinker.py`` shape: ``serve --mode full --no-inference --hf-checkpoint ... --extra-args ...``).

Settings: ``RELIQUARY_MILES_LAUNCHER``, ``RELIQUARY_MILES_MODEL_TYPE`` (Megatron model type, e.g. ``qwen3.5-9B``),
``RELIQUARY_MILES_OUTPUT_DIR`` (checkpoints and exports: the trainer reads the exports there), and
``RELIQUARY_MILES_TP`` (2), ``RELIQUARY_MILES_GPUS`` (8), ``RELIQUARY_MILES_PORT`` (10613),
``RELIQUARY_MILES_READY_S`` (1800), ``RELIQUARY_MILES_EXTRA_ARGS`` (more gateway flags).
"""
from __future__ import annotations

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
            "extra_args": str(env.get("RELIQUARY_MILES_EXTRA_ARGS", "")).strip()}


def gateway_command(model_path, settings: dict, *, base_model: str, grad_norm_skip: float) -> list[str]:
    extra = f"--tinker-base-model {base_model} --tinker-grad-norm-skip {float(grad_norm_skip)}"
    if settings["extra_args"]:
        extra += " " + settings["extra_args"]
    gpus = str(settings["gpus"])
    return [sys.executable, settings["launcher"], "serve", "--mode", "full", "--no-inference",
            "--hf-checkpoint", str(model_path), "--model-type", settings["model_type"], "--tp", str(settings["tp"]),
            "--actor-num-gpus", gpus, "--num-gpus-per-node", gpus, "--tinker-port", str(settings["port"]),
            "--output-dir", str(settings["output_dir"]), "--extra-args", extra]


def _ray_stop() -> None:
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True, timeout=120)


class GatewayProcess:
    """The gateway as a child process group, its output in ``log_path``."""

    def __init__(self, command: list[str], *, log_path: Path, env: dict, cleanup: Callable[[], None] = _ray_stop,
                 base_url: str = "") -> None:
        self.log_path, self.cleanup, self.base_url = Path(log_path), cleanup, base_url
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        logger.info("starting the Miles gateway: %s", shlex.join(command))
        self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)

    def wait_ready(self, *, timeout_s: float, poll_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if READY_LINE in self.log_path.read_text(errors="replace"):
                logger.info("Miles gateway ready")
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"the Miles gateway exited ({self.process.returncode}); see {self.log_path}")
            time.sleep(poll_s)
        self.stop()
        raise RuntimeError(f"the Miles gateway was not ready in {timeout_s:.0f}s; see {self.log_path}")

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


def launch_gateway(model_path, settings: dict, *, base_model: str, grad_norm_skip: float, env: dict) -> GatewayProcess:
    gateway = GatewayProcess(gateway_command(model_path, settings, base_model=base_model, grad_norm_skip=grad_norm_skip),
                             log_path=settings["output_dir"] / "gateway.log", env=env,
                             base_url=f"http://127.0.0.1:{settings['port']}")
    try:
        gateway.wait_ready(timeout_s=settings["ready_s"])
    except BaseException:
        gateway.stop()
        raise
    return gateway


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


__all__ = ["GatewayProcess", "checkpoint_deleter", "create_full_training_client", "gateway_command",
           "launch_gateway", "miles_settings"]
