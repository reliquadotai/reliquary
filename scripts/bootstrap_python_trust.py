#!/usr/bin/env python3
"""Bootstrap hash-pinned package tooling; repair managed Linux image trust.

Run before other package downloads. --system is for root-owned Linux build
environments; --verify-only checks the final installed bundles without writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile

# Independently published hashes, not checksums fetched beside the downloads.
WHEELS = (
    ("https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl",
     "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"),
    ("https://files.pythonhosted.org/packages/c4/51/29046e48f693c54e5b7412216d447a1d8362c9653148d98c02cdf65fa733/certifi-2026.10.19-py3-none-any.whl",
     "42d35340cc5b9ba78e818ddbff0c5b1c2b779de448a340fd1b989f5fa8670e1b"),
)
DISTRUSTED_ROOTS = {
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
    "7dd78d5f4f13459a83dff9abbba62edbaf6f2d102bf257fd712f4d9f2746ed8d",
    "f2edaaff3d9b701fb4d14c9f285f8c0bad90fc3bcc1fa07d47f4718edaba34a5",
}
CERTIFICATE = re.compile(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S)
SYSTEM_CERT_DIRS = ("/usr/share/ca-certificates", "/usr/local/share/ca-certificates", "/etc/ssl/certs")


def distrusted(pem: bytes) -> bool:
    der = ssl.PEM_cert_to_DER_cert(pem.decode("ascii"))
    return hashlib.sha256(der).hexdigest() in DISTRUSTED_ROOTS


def clean_bundle(path: Path) -> None:
    original = path.read_bytes()
    clean = CERTIFICATE.sub(lambda match: b"" if distrusted(match[0]) else match[0], original)
    if clean != original:
        if CERTIFICATE.search(clean):
            path.write_bytes(clean)
        else:
            path.unlink()


def package_bundles() -> list[Path]:
    bundles = []
    for package, relative in (("certifi", "certifi/cacert.pem"),
                              ("pip", "pip/_vendor/certifi/cacert.pem"),
                              ("botocore", "botocore/cacert.pem")):
        try:
            bundles.append(Path(importlib.metadata.distribution(package).locate_file(relative)))
        except importlib.metadata.PackageNotFoundError:
            if package != "botocore":
                raise
    return bundles


def check_pem(pem: bytes, label: str) -> None:
    certificates = CERTIFICATE.findall(pem)
    if not certificates or any(distrusted(cert) for cert in certificates):
        raise ValueError(f"Unsafe or empty certificate bundle: {label}")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cadata=pem.decode("ascii"))


def verify(*, system: bool = False) -> None:
    for bundle in package_bundles():
        check_pem(bundle.read_bytes(), str(bundle))
    if system:
        context = ssl.create_default_context()
        roots = context.get_ca_certs(binary_form=True)
        if not roots or any(hashlib.sha256(cert).hexdigest() in DISTRUSTED_ROOTS for cert in roots):
            raise ValueError("Unsafe or empty default system trust store")
        # OpenSSL also loads capath entries lazily, beyond get_ca_certs().
        paths = ssl.get_default_verify_paths()
        if paths.capath:
            for path in Path(paths.capath).iterdir():
                if path.is_file() and any(distrusted(cert) for cert in CERTIFICATE.findall(path.read_bytes())):
                    raise ValueError(f"Unsafe system certificate: {path}")
        for wheel in ensurepip_wheels():
            with zipfile.ZipFile(wheel) as archive:
                check_pem(archive.read("pip/_vendor/certifi/cacert.pem"), str(wheel))
    print("Installed certificate bundles verified")


def ensurepip_wheels() -> list[Path]:
    import ensurepip

    locations = [Path(ensurepip.__file__).parent / "_bundled"]
    wheel_dir = sysconfig.get_config_var("WHEEL_PKG_DIR")
    if wheel_dir:
        locations.append(Path(wheel_dir))
    return [wheel for location in locations for wheel in location.glob("pip-*.whl")]


def repair_system(pip_wheel: Path) -> None:
    import ensurepip

    source = Path(ensurepip.__file__)
    updated, count = re.subn(r'^_PIP_VERSION = "[^"]+"$', '_PIP_VERSION = "26.2.2"',
                            source.read_text(), flags=re.M)
    if count != 1:
        raise ValueError("Unsupported ensurepip layout; update the Python base image")
    # These are public CA stores only; application private mTLS CAs are separate.
    for directory in SYSTEM_CERT_DIRS:
        for path in Path(directory).rglob("*"):
            if path.is_file() and not path.is_symlink():
                clean_bundle(path)
    subprocess.run(["update-ca-certificates", "--fresh"], check=True)
    bundled = source.parent / "_bundled"
    bundled.mkdir(exist_ok=True)
    locations = {bundled, *(wheel.parent for wheel in ensurepip_wheels())}
    for location in locations:
        for wheel in location.glob("pip-*.whl"):
            wheel.unlink()
        shutil.copyfile(pip_wheel, location / pip_wheel.name)
    source.write_text(updated)


def bootstrap(*, system: bool = False) -> None:
    if system:
        if sys.platform != "linux" or os.geteuid() != 0:
            raise ValueError("--system requires a root-owned managed Linux build environment")
        if not shutil.which("update-ca-certificates"):
            raise ValueError("--system requires Debian/Ubuntu update-ca-certificates")
        import ensurepip

        if not re.search(r'^_PIP_VERSION = "[^"]+"$', Path(ensurepip.__file__).read_text(), re.M):
            raise ValueError("Unsupported ensurepip layout; update the Python base image")
    with tempfile.TemporaryDirectory(prefix="python-trust-") as directory:
        wheels = []
        for url, expected in WHEELS:
            wheel = Path(directory) / url.rsplit("/", 1)[1]
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(4 * 1024 * 1024 + 1)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Bootstrap wheel SHA-256 mismatch; nothing installed")
            wheel.write_bytes(data)
            wheels.append(wheel)
        subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check",
                        "install", "--no-index", "--no-deps", "--upgrade", *map(str, wheels)], check=True)
        if system:
            repair_system(wheels[0])
    for bundle in package_bundles():
        clean_bundle(bundle)
    verify(system=system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(system=args.system)
    else:
        bootstrap(system=args.system)
