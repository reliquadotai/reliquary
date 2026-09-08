from pathlib import Path

import pytest

from scripts import install_logic_runtime as installer
from scripts.qualify_verifiers_interop import LOGIC_WHEEL_SHA256
from reliquary.environment.registry import get_environment_spec


def test_logic_release_pin_matches_runtime_catalog_and_qualification():
    spec = get_environment_spec("reliquary_logic_v2")
    assert installer.PIN["artifact_sha256"] == spec.environment_manifest_sha256
    assert installer.PIN["wheel_sha256"] == LOGIC_WHEEL_SHA256
    assert installer.PIN["tag"] == "logic-v0.1.0a1"


def test_bad_candidate_wheel_never_reaches_pip(tmp_path, monkeypatch):
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"untrusted wheel bytes")
    commands = []
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: commands.append(args))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        installer.install(wheel)
    assert commands == []
    assert wheel.read_bytes() == b"untrusted wheel bytes"


def test_download_hash_is_checked_before_install(monkeypatch):
    import io

    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong release"))
    commands = []
    monkeypatch.setattr(installer.subprocess, "run", lambda *args, **kwargs: commands.append(args))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        installer.install(None)
    assert commands == []


def test_cpu_role_images_do_not_install_optional_logic_stack():
    root = Path(__file__).resolve().parents[2]
    for role in ("cpu-executor", "signer"):
        assert "install_logic_runtime" not in (root / "docker" / f"Dockerfile.{role}").read_text()
