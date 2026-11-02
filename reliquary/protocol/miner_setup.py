"""Opt-in, self-declared miner hardware (``POST /miner/setup``).

Nothing here is attested and nothing reads it for scoring or pay: it is a census the subnet owner asked miners to
share. The request is still signed by the hotkey for one validator and route, so nobody declares for someone else.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

from pydantic import BaseModel, ConfigDict, Field

MINER_SETUP_PATH = "/miner/setup"
MINER_SETUP_DOMAIN = b"reliquary/miner-setup/v1"
MAX_GPUS = 64


class GpuInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=160)
    memory_mib: int = Field(default=0, ge=0, le=10_000_000)
    # sha256(hotkey, GPU UUID), truncated: counts each card once across a hotkey's processes without revealing it.
    id_hash: str = Field(default="", pattern=r"^([0-9a-f]{16})?$")


class MinerSetupReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(..., pattern=r"^[0-9a-f]{16,64}$")
    gpus: list[GpuInfo] = Field(default_factory=list, max_length=MAX_GPUS)
    cpu_count: int | None = Field(default=None, ge=0, le=100_000)
    ram_gib: float | None = Field(default=None, ge=0, le=1_000_000)
    note: str = Field(default="", max_length=500)


class MinerSetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(..., min_length=1, max_length=64)
    at: int = Field(..., ge=0)
    report: MinerSetupReport
    signature: str = Field(default="", max_length=256)


def _report_digest(report: MinerSetupReport) -> bytes:
    body = json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).digest()


def build_miner_setup_binding(*, hotkey: str, at: int, report: MinerSetupReport, validator_hotkey: str,
                              path: str = MINER_SETUP_PATH) -> bytes:
    parts = [str(hotkey).encode("utf-8"), int(at).to_bytes(8, "big", signed=False), _report_digest(report),
             str(validator_hotkey).encode("utf-8"), str(path).encode("utf-8")]
    h = hashlib.sha256()
    h.update(MINER_SETUP_DOMAIN)
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def verify_miner_setup_signature(request: MinerSetupRequest, *, validator_hotkey: str,
                                 path: str = MINER_SETUP_PATH) -> bool:
    from reliquary.protocol.signatures import verify_hotkey_signature

    try:
        binding = build_miner_setup_binding(hotkey=request.miner_hotkey, at=request.at, report=request.report,
                                            validator_hotkey=validator_hotkey, path=path)
    except (TypeError, ValueError, OverflowError):
        return False
    return verify_hotkey_signature(request.miner_hotkey, binding, request.signature)


# --- collection on the miner -----------------------------------------------------------------------------------

def _visible(devices: list[dict], visible: str | None) -> list[dict]:
    """The GPUs this process may use: CUDA_VISIBLE_DEVICES by index or UUID (unset: all of them)."""
    if visible is None:
        return devices
    wanted = [item.strip() for item in visible.split(",") if item.strip()]
    picked = []
    for item in wanted:
        if item.isdigit():
            picked += [d for d in devices if d["index"] == int(item)]
        else:
            picked += [d for d in devices if d["uuid"].startswith(item)]
    return picked


def query_gpus(run=subprocess.run) -> list[dict]:
    """``nvidia-smi``'s GPUs (index, name, memory_mib, uuid); empty without it. Never initialises CUDA here, so
    vLLM still owns the device."""
    if shutil.which("nvidia-smi") is None and run is subprocess.run:
        return []
    try:
        out = run(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        try:
            memory = int(float(fields[2]))
        except ValueError:
            memory = 0
        devices.append({"index": int(fields[0]), "name": fields[1], "memory_mib": memory, "uuid": fields[3]})
    return devices


def _ram_gib() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def collect_miner_setup(*, hotkey: str, instance_id: str, note: str = "", devices: list[dict] | None = None,
                        visible: str | None = None) -> MinerSetupReport:
    if devices is None:
        devices = query_gpus()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = [GpuInfo(name=d["name"][:160] or "unknown", memory_mib=d["memory_mib"],
                    id_hash=hashlib.sha256(f"{hotkey}:{d['uuid']}".encode()).hexdigest()[:16] if d["uuid"] else "")
            for d in _visible(devices, visible)][:MAX_GPUS]
    return MinerSetupReport(instance_id=instance_id, gpus=gpus, cpu_count=os.cpu_count(), ram_gib=_ram_gib(),
                            note=note[:500])


def load_instance_id(path) -> str:
    """This miner process's id, kept under its work dir so a restart reports as the same instance."""
    from pathlib import Path

    path = Path(path)
    try:
        value = path.read_text().strip()
        if len(value) == 32 and all(c in "0123456789abcdef" for c in value):
            return value
    except OSError:
        pass
    value = os.urandom(16).hex()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")
    return value


__all__ = [
    "GpuInfo", "MINER_SETUP_PATH", "MinerSetupReport", "MinerSetupRequest", "build_miner_setup_binding",
    "collect_miner_setup", "load_instance_id", "query_gpus", "verify_miner_setup_signature",
]
