"""Building a customer package into its image, on a builder that holds no secret.

The build context is the checked archive (unpacked with tarfile's ``data`` filter) plus a Dockerfile we write:
the customer never supplies a Dockerfile. Dependencies install from the hash-pinned lock with ``--no-deps``,
then the package itself, then the worker's manifest. The result is pushed and named by digest.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from reliquary.customer_env import EnvId
from reliquary.customer_env.package import Package

# Pushed from docker/customer-env/Dockerfile.base; set at the base image's first push.
BASE_IMAGE = "registry.reliqua.ai/customer-envs/base@sha256:" + "0" * 64
REPOSITORY = "registry.reliqua.ai/customer-envs"
BUILD_TIMEOUT_S = 1800
SCORE_TIMEOUT_S = 60.0
# The base provides these; a lock that pins them would fight the base's own.
PROVIDED = frozenset({"verifiers", "reliquary"})
_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}\Z")


class BuildFailed(RuntimeError):
    """The customer's package did not build (``build_failed``): its lock or its sources."""


def dockerfile(base_image: str) -> str:
    return "\n".join([
        f"FROM {base_image}",
        "COPY package /opt/customer/package",
        "COPY manifest.json /opt/reliquary/manifest.json",
        "RUN pip install --no-cache-dir --require-hashes --no-deps -r /opt/customer/package/requirements.lock"
        " && pip install --no-cache-dir --no-deps --no-build-isolation /opt/customer/package"
        " && python -I -c \"import json, verifiers.v1 as vf; m = json.load(open('/opt/reliquary/manifest.json'));"
        " vf.taskset_config_type(m['taskset'])\"",
        "USER 65534:65534",
        'ENTRYPOINT ["python", "-I", "-m", "reliquary.customer_env.worker"]',
        "",
    ])


def manifest(package: Package) -> dict:
    return {"taskset": package.taskset, "args": package.args, "kind": package.kind,
            "score_timeout_s": SCORE_TIMEOUT_S, "package_sha256": package.sha256}


def check_lock(package: Package) -> None:
    pinned = {line.split("==", 1)[0].split("[", 1)[0].lower().replace("_", "-") for line in package.requirements}
    clash = sorted(pinned & PROVIDED)
    if clash:
        from reliquary.customer_env.package import PackageInvalid

        raise PackageInvalid(f"requirements.lock pins {clash}: the base image provides them")


def write_context(directory: Path, data: bytes, package: Package, *, base_image: str = BASE_IMAGE) -> Path:
    """The build context in ``directory``: ``package/`` (the archive's root), ``manifest.json``, ``Dockerfile``."""
    check_lock(package)
    target = directory / "package"
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        prefix = f"{package.root}/" if package.root else ""
        members = []
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            if prefix and not name.startswith(prefix):
                continue
            member.name = name.removeprefix(prefix)
            if member.name:
                members.append(member)
        archive.extractall(target, members=members, filter="data")
    (directory / "manifest.json").write_text(json.dumps(manifest(package), sort_keys=True))
    (directory / "Dockerfile").write_text(dockerfile(base_image))
    return directory


def image_tag(env: EnvId) -> str:
    return f"{REPOSITORY}/{env.account.lower()}/{env.name}:{env.sha256[:32]}"


@dataclass
class DockerBuilder:
    """``docker build`` then ``docker push`` on the builder host; answers ``repository@sha256:...``."""

    run: Callable[[Sequence[str], float], subprocess.CompletedProcess] | None = None
    docker: str = "docker"

    def _run(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
        if self.run is not None:
            return self.run(argv, timeout)
        return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)

    def build(self, context: Path, env: EnvId) -> str:
        tag = image_tag(env)
        try:
            built = self._run([self.docker, "build", "--pull=false", "--no-cache", "-t", tag, str(context)],
                              BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise BuildFailed(f"the build took over {BUILD_TIMEOUT_S} s") from None
        if built.returncode != 0:
            raise BuildFailed((built.stderr or built.stdout or "")[-2000:])
        pushed = self._run([self.docker, "push", tag], 900)
        if pushed.returncode != 0:
            raise RuntimeError(f"push failed: {(pushed.stderr or '')[-500:]}")
        inspected = self._run([self.docker, "image", "inspect", "--format", "{{json .RepoDigests}}", tag], 60)
        digests = [d for d in json.loads(inspected.stdout or "[]") if d.startswith(tag.rsplit(":", 1)[0] + "@")]
        if inspected.returncode != 0 or not digests or not _DIGEST_RE.search(digests[0]):
            raise RuntimeError("the pushed image has no repository digest")
        return digests[0]


@dataclass
class LocalBuilder:
    """``build_archive(data, env)`` on this host: the build context in a temporary directory, then ``inner``."""

    inner: DockerBuilder

    def build_archive(self, data: bytes, env: EnvId) -> str:
        import tempfile

        from reliquary.customer_env.package import read_package

        package = read_package(data)
        with tempfile.TemporaryDirectory(prefix="customer-env-") as tmp:
            return self.inner.build(write_context(Path(tmp), data, package), env)


__all__ = ["BASE_IMAGE", "BuildFailed", "DockerBuilder", "LocalBuilder", "check_lock", "dockerfile", "image_tag", "manifest",
           "write_context"]
