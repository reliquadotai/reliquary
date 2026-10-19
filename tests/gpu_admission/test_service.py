import builtins
import sqlite3

import pytest
from fastapi.testclient import TestClient

from reliquary.validator.gpu_registry import GpuRegistry, registry_from_env
from reliquary.validator.rl_async_service import mount_rl_async
from reliquary.validator.server import ValidatorServer
from tests.unit.test_rl_async_service import components
from .test_routes import MINER, RUN, VALIDATOR, attest


def test_real_service_mounts_api_and_closes_owned_registry(tmp_path, authority, now):
    rl = components(tmp_path)
    registry = GpuRegistry(tmp_path / "gpu.db", validator=VALIDATOR.ss58_address, run=RUN, mode="required",
                            verifier=authority.verifier, clock=lambda: now[0])
    rl.gpu_registry = registry
    server = ValidatorServer()
    try:
        mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address, clock=lambda: now[0])
        with TestClient(server.app) as client:
            assert client.get("/rl/gpu/fleet").status_code == 200
            assert attest(client, authority).status_code == 200
            assert client.get("/rl/quota", params={"hotkey": MINER.ss58_address}).json()["gpu_admission"]["attested"]
    finally:
        rl.close()
    with pytest.raises(sqlite3.ProgrammingError):
        registry.status(MINER.ss58_address)


def test_default_off_mounts_no_gpu_api_and_loads_no_optional_crypto(tmp_path, monkeypatch):
    actual_import = builtins.__import__
    def without_jwt(name, *args, **kwargs):
        if name == "jwt":
            raise ImportError("extra not installed")
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_jwt)
    assert registry_from_env(tmp_path / "unused", validator="v", run="r", env={}) is None
    with pytest.raises(RuntimeError, match="dependency extra"):
        registry_from_env(tmp_path / "unused", validator="v", run="r", env={"RELIQUARY_RL_GPU_ADMISSION": "required"})
    rl = components(tmp_path)
    try:
        server = ValidatorServer()
        mount_rl_async(server, rl, validator_hotkey=VALIDATOR.ss58_address)
        with TestClient(server.app) as client:
            assert client.get("/rl/gpu/fleet").status_code == 404
    finally:
        rl.close()


def test_state_file_is_private_and_symlinks_are_refused(tmp_path, authority):
    path = tmp_path / "private.db"
    registry = GpuRegistry(path, validator="v", run="r", mode="observe", verifier=authority.verifier)
    registry.close()
    assert path.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "link.db"
    link.symlink_to(path)
    with pytest.raises(OSError):
        GpuRegistry(link, validator="v", run="r", mode="observe", verifier=authority.verifier)
