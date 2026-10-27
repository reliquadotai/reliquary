import base64
import csv
import hashlib
import io
from pathlib import Path
import ssl
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from scripts import bootstrap_python_trust as trust


@pytest.fixture(autouse=True)
def isolate_installed_virtualenv(monkeypatch):
    original = trust.importlib.metadata.distribution
    def distribution(name):
        if name == "virtualenv":
            raise trust.importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(trust.importlib.metadata, "distribution", distribution)


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


@pytest.fixture
def virtualenv_seed(tmp_path, roots, monkeypatch):
    blocked, allowed = roots
    embed = tmp_path / "virtualenv/seed/wheels/embed"
    embed.mkdir(parents=True)
    wheels = []
    for version, bundle in [("26.0.1", blocked + allowed), ("26.2.2", allowed)]:
        wheel = embed / f"pip-{version}-py3-none-any.whl"
        record = f"pip-{version}.dist-info/RECORD"
        members = {"pip/__init__.py": b"# unchanged pip code\n", "pip/_vendor/certifi/cacert.pem": bundle}
        rows = io.StringIO(newline="")
        writer = csv.writer(rows)
        for name, data in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + digest, len(data)])
        writer.writerow([record, "", ""])
        members[record] = rows.getvalue().encode()
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.comment = b"preserved archive comment"
            for name, data in members.items():
                archive.writestr(name, data)
        wheel.chmod(0o640)
        wheels.append(wheel)
    selector = embed / "__init__.py"
    selector.write_text('BUNDLE_SUPPORT = {"3.9": "pip-26.0.1", "3.12": "pip-26.2.2"}\n'
                        'BUNDLE_SHA256 = {\n' + "".join(
                            f'    "{wheel.name}": "{hashlib.sha256(wheel.read_bytes()).hexdigest()}",\n'
                            for wheel in wheels) + '}\n')
    metadata = tmp_path / "virtualenv-21.14.13.dist-info"
    metadata.mkdir()
    record = metadata / "RECORD"
    rows = io.StringIO(newline="")
    for path in [*wheels, selector]:
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        csv.writer(rows).writerow([path.relative_to(tmp_path).as_posix(), "sha256=" + digest, len(data)])
    csv.writer(rows).writerow([record.relative_to(tmp_path).as_posix(), "", ""])
    record.write_bytes(rows.getvalue().encode())
    distribution = SimpleNamespace(locate_file=lambda path: tmp_path / path,
                                   files=[record.relative_to(tmp_path)])
    original = trust.importlib.metadata.distribution
    monkeypatch.setattr(trust.importlib.metadata, "distribution", lambda name:
                        distribution if name == "virtualenv" else original(name))
    return SimpleNamespace(root=tmp_path, embed=embed, wheels=wheels, selector=selector, record=record)


@pytest.mark.parametrize("system", [False, True])
def test_verify_rejects_unsafe_virtualenv_seed_without_writes(virtualenv_seed, monkeypatch, system):
    monkeypatch.setattr(trust, "package_bundles", lambda: [])
    before = {path: path.read_bytes() for path in virtualenv_seed.root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unsafe"):
        trust.verify(system=system)
    assert all(path.read_bytes() == data for path, data in before.items())


def assert_record_hashes(record, read):
    for name, digest, size in csv.reader(io.StringIO(record.decode())):
        if digest:
            data = read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + expected
            assert int(size) == len(data)


def test_virtualenv_repair_preserves_legacy_selector_and_valid_records(virtualenv_seed, roots):
    seed = virtualenv_seed
    old, clean = seed.wheels
    selector = seed.selector.read_text()
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    untouched = clean.read_bytes()
    with zipfile.ZipFile(old) as archive:
        code = archive.read("pip/__init__.py")
        info = archive.getinfo("pip/__init__.py")
    trust.repair_virtualenv()
    assert clean.read_bytes() == untouched
    assert old.stat().st_mode & 0o777 == 0o640
    assert seed.selector.read_text() == selector.replace(old_hash, hashlib.sha256(old.read_bytes()).hexdigest())
    with zipfile.ZipFile(old) as archive:
        assert archive.read("pip/__init__.py") == code
        changed_info = archive.getinfo("pip/__init__.py")
        for field in ["date_time", "compress_type", "external_attr", "extra", "comment"]:
            assert getattr(info, field) == getattr(changed_info, field)
        assert archive.comment == b"preserved archive comment"
        trust.check_pem(archive.read("pip/_vendor/certifi/cacert.pem"), "legacy seed")
        assert_record_hashes(archive.read("pip-26.0.1.dist-info/RECORD"), archive.read)
    assert_record_hashes(seed.record.read_bytes(), lambda name: (seed.root / name).read_bytes())
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in seed.root.rglob("*") if path.is_file()}
    trust.repair_virtualenv()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == data for path, data in before.items())


@pytest.mark.parametrize("failure", ["checksum", "record", "stage", "empty", "signed", "duplicate"])
def test_virtualenv_repair_failure_preserves_originals(virtualenv_seed, roots, monkeypatch, failure):
    seed = virtualenv_seed
    if failure == "checksum":
        seed.selector.write_text(seed.selector.read_text().replace(
            hashlib.sha256(seed.wheels[0].read_bytes()).hexdigest(), "0" * 64))
    elif failure == "record":
        seed.record.write_text("unrelated-file,,\n")
    elif failure == "stage":
        original = trust.tempfile.NamedTemporaryFile
        calls = []
        def fail_second(**kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise OSError("fixture staging failure")
            return original(**kwargs)
        monkeypatch.setattr(trust.tempfile, "NamedTemporaryFile", fail_second)
    else:
        wheel = seed.wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        if failure == "empty":
            members["pip/_vendor/certifi/cacert.pem"] = roots[0]
        elif failure == "signed":
            members["pip-26.0.1.dist-info/RECORD.jws"] = b"signature"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
            if failure == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr("pip/_vendor/certifi/cacert.pem", roots[1])
    before = {path: path.read_bytes() for path in seed.root.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, OSError)):
        trust.repair_virtualenv()
    after = {path: path.read_bytes() for path in seed.root.rglob("*") if path.is_file()}
    assert before == after


def test_non_system_bootstrap_repairs_virtualenv_and_checks_selector(virtualenv_seed, monkeypatch):
    data = b"verified wheel"
    monkeypatch.setattr(trust, "WHEELS", (("https://example.test/fixed.whl", hashlib.sha256(data).hexdigest()),))
    monkeypatch.setattr(trust.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    monkeypatch.setattr(trust, "package_bundles", lambda: [])
    commands = []
    monkeypatch.setattr(trust.subprocess, "run", lambda command, **kwargs: commands.append(command))
    trust.bootstrap()
    assert len(commands) == 2
    assert "--no-index" in commands[0]
    assert "BUNDLE_SUPPORT" in commands[1][-1]
    for wheel in virtualenv_seed.wheels:
        with zipfile.ZipFile(wheel) as archive:
            trust.check_pem(archive.read("pip/_vendor/certifi/cacert.pem"), "seed")


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
