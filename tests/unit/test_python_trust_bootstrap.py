import hashlib
import io
from pathlib import Path
import ssl
import subprocess
import zipfile

import pytest

from scripts import bootstrap_python_trust as trust


@pytest.fixture
def roots(monkeypatch):
    certificates = ssl.create_default_context().get_ca_certs(binary_form=True)
    blocked, allowed = certificates[:2]
    monkeypatch.setattr(trust, "DISTRUSTED_ROOTS", {hashlib.sha256(blocked).hexdigest()})
    return tuple(ssl.DER_cert_to_PEM_cert(cert).encode("ascii") for cert in (blocked, allowed))


def test_bundle_cleanup_preserves_other_roots_and_rejects_unsafe_store(tmp_path, roots):
    blocked, allowed = roots
    path = tmp_path / "cacert.pem"
    path.write_bytes(blocked + allowed)
    with pytest.raises(ValueError, match="Unsafe"):
        trust.check_pem(path.read_bytes(), "test")
    trust.clean_bundle(path)
    trust.check_pem(path.read_bytes(), "test")
    assert path.read_bytes() == b"\n" + allowed
    trust.clean_bundle(path)
    assert path.read_bytes() == b"\n" + allowed
    path.write_bytes(blocked)
    trust.clean_bundle(path)
    assert not path.exists()


def test_both_downloads_must_match_before_any_install(tmp_path, monkeypatch):
    good = b"pinned wheel"
    monkeypatch.setattr(trust, "WHEELS", (
        ("https://example.test/pip.whl", hashlib.sha256(good).hexdigest()),
        ("https://example.test/certifi.whl", hashlib.sha256(good).hexdigest()),
    ))
    monkeypatch.setattr(trust.urllib.request, "urlopen", lambda url, **kwargs: io.BytesIO(
        good if url.endswith("pip.whl") else b"tampered"))
    commands = []
    monkeypatch.setattr(trust.subprocess, "run", lambda *args, **kwargs: commands.append(args))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trust.bootstrap()
    assert commands == []


@pytest.mark.parametrize("platform,uid", [("darwin", 0), ("linux", 1000)])
def test_system_mode_requires_managed_linux_root_before_download(monkeypatch, platform, uid):
    monkeypatch.setattr(trust.sys, "platform", platform)
    monkeypatch.setattr(trust.os, "geteuid", lambda: uid)
    monkeypatch.setattr(trust.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("download attempted"))
    with pytest.raises(ValueError, match="root-owned managed Linux"):
        trust.bootstrap(system=True)


def test_bootstrap_install_is_offline_and_repairs_embedded_bundle(tmp_path, roots, monkeypatch):
    blocked, allowed = roots
    bundle = tmp_path / "embedded.pem"
    bundle.write_bytes(blocked + allowed)
    data = b"verified wheel"
    monkeypatch.setattr(trust, "WHEELS", (("https://example.test/fixed.whl", hashlib.sha256(data).hexdigest()),))
    monkeypatch.setattr(trust.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    monkeypatch.setattr(trust, "package_bundles", lambda: [bundle])
    commands = []
    monkeypatch.setattr(trust.subprocess, "run", lambda command, **kwargs: commands.append(command))
    trust.bootstrap()
    assert len(commands) == 1
    assert "--no-index" in commands[0] and "--no-deps" in commands[0]
    trust.check_pem(bundle.read_bytes(), "embedded")


def test_managed_system_repairs_anchors_and_future_venv_seed(tmp_path, roots, monkeypatch):
    import ensurepip

    blocked, allowed = roots
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    (anchors / "blocked.crt").write_bytes(blocked)
    (anchors / "allowed.crt").write_bytes(allowed)
    source = tmp_path / "ensurepip" / "__init__.py"
    source.parent.mkdir()
    source.write_text('_PIP_VERSION = "25.0"\n')
    bundled = source.parent / "_bundled"
    bundled.mkdir()
    (bundled / "pip-25.0-py3-none-any.whl").write_bytes(b"old")
    fixed = tmp_path / "pip-26.2.2-py3-none-any.whl"
    with zipfile.ZipFile(fixed, "w") as archive:
        archive.writestr("pip/_vendor/certifi/cacert.pem", allowed)
    monkeypatch.setattr(ensurepip, "__file__", str(source))
    monkeypatch.setattr(trust.sysconfig, "get_config_var", lambda name: None)
    monkeypatch.setattr(trust, "SYSTEM_CERT_DIRS", (str(anchors),))
    commands = []
    monkeypatch.setattr(trust.subprocess, "run", lambda command, **kwargs: commands.append(command))
    trust.repair_system(fixed)
    assert not (anchors / "blocked.crt").exists()
    assert (anchors / "allowed.crt").read_bytes() == allowed
    assert commands == [["update-ca-certificates", "--fresh"]]
    assert source.read_text() == '_PIP_VERSION = "26.2.2"\n'
    assert list(bundled.iterdir()) == [bundled / fixed.name]
    assert (bundled / fixed.name).read_bytes() == fixed.read_bytes()


def test_unsupported_ensurepip_leaves_system_untouched(tmp_path, monkeypatch):
    import ensurepip

    source = tmp_path / "__init__.py"
    source.write_text("unsupported layout")
    monkeypatch.setattr(ensurepip, "__file__", str(source))
    monkeypatch.setattr(trust, "clean_bundle", lambda *args: pytest.fail("system mutation attempted"))
    with pytest.raises(ValueError, match="Unsupported ensurepip"):
        trust.repair_system(tmp_path / "pip.whl")


def test_remote_setup_rejects_streamed_script_before_running_tools(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/setup_remote_box.sh"
    result = subprocess.run(["bash"], input=script.read_text(), text=True,
                            capture_output=True, cwd=tmp_path)
    assert result.returncode == 2
    assert "trusted local copy" in result.stderr
    assert "apt install" not in result.stdout


def test_remote_setup_authenticates_transport_before_git(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/setup_remote_box.sh"
    local_script = tmp_path / script.name
    local_script.write_text(script.read_text())
    (tmp_path / "bootstrap_python_trust.py").touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "trust-repaired"
    (tools / "apt").write_text("#!/bin/bash\nexit 0\n")
    (tools / "python3.12").write_text(
        '#!/bin/bash\nmkdir -p "$3/bin"\n'
        f"printf '#!/bin/bash\\ntouch {marker}\\n' > \"$3/bin/python\"\n"
        'chmod +x "$3/bin/python"\n')
    (tools / "git").write_text(f"#!/bin/bash\n[ -f '{marker}' ] || exit 98\nexit 99\n")
    for tool in tools.iterdir():
        tool.chmod(0o755)
    result = subprocess.run(["bash", str(local_script)], capture_output=True,
                            env={**trust.os.environ, "PATH": f"{tools}:/usr/bin:/bin",
                                 "RELIQUARY_INSTALL_DIR": str(tmp_path / "new-checkout")})
    assert result.returncode == 99, result.stderr.decode()
    assert marker.exists()
