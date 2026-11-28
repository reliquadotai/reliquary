"""Packing a customer env, on the customer's own machine: a source directory, or a Prime Environments Hub env.

    python -m reliquary.customer_env.pack ./my-env --examples examples.jsonl -o my-env.tar.gz
    python -m reliquary.customer_env.pack --from-prime owner/name@1.2.0 --taskset name \\
        --examples examples.jsonl -o name.tar.gz

It adds ``[tool.reliquary]`` when the directory has none (``--taskset``, ``--args``, ``--license``), records a
Prime Hub env's source (owner, name, version: shown wherever the env is listed), writes a hash-pinned
``requirements.lock`` with ``uv pip compile --generate-hashes`` when there is none (verifiers left out: our base
provides it), copies the examples, and writes a deterministic archive that ``package.read_package`` accepts.
Nothing here runs on our hosts.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from reliquary.customer_env.image import PROVIDED
from reliquary.customer_env.package import read_package

SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build", ".ruff_cache"}
Run = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, check=False)


def pull_from_prime(spec: str, target: Path, *, run: Run = _run) -> Path:
    """``prime env pull owner/name[@version] -t target`` (the Prime CLI, logged in as the customer)."""
    if "/" not in spec:
        raise ValueError("a Prime Hub env is owner/name[@version]")
    done = run(["prime", "env", "pull", spec, "-t", str(target)])
    if done.returncode != 0:
        raise RuntimeError(f"prime env pull failed: {(done.stderr or done.stdout)[-1000:]}")
    candidates = [p.parent for p in target.rglob("pyproject.toml")]
    if not candidates:
        raise RuntimeError("the pulled env has no pyproject.toml")
    return min(candidates, key=lambda p: len(p.parts))


def prime_source(spec: str, pulled: Path) -> dict:
    """The provenance of a Prime Hub env: its owner, name and the version pulled."""
    owner, _, rest = spec.partition("/")
    name, _, version = rest.partition("@")
    if not version:
        version = str((tomllib.loads((pulled / "pyproject.toml").read_text()).get("project") or {}).get("version", ""))
    source = {"hub": "prime_hub", "owner": owner, "name": name,
              "url": f"https://app.primeintellect.ai/dashboard/environments/{owner}/{name}"}
    return {**source, "revision": version} if version else source


def _toml_table(header: str, values: dict) -> list[str]:
    return ["", f"[{header}]", *(f"{k} = {json.dumps(v)}" for k, v in values.items())]


def _with_manifest(pyproject: str, taskset: str | None, args: dict | None, *, license: str | None = None,
                   source: dict | None = None) -> str:
    project = tomllib.loads(pyproject)
    tool = (project.get("tool") or {}).get("reliquary")
    lines = []
    if tool is None:
        if not taskset:
            raise ValueError("pyproject.toml has no [tool.reliquary]: pass --taskset (the verifiers taskset id)")
        lines = ["", "[tool.reliquary]", f"taskset = {json.dumps(taskset)}"]
        if license:
            lines.append(f"license = {json.dumps(license)}")
        if args:
            body = ", ".join(f"{k} = {json.dumps(v)}" for k, v in sorted(args.items()))
            lines.append(f"args = {{{body}}}")
    elif license and not tool.get("license"):
        if source is None:
            raise ValueError("pyproject.toml has a [tool.reliquary]: name the license there")
        source = {**source, "license": license}  # the table is closed: the upstream license goes with its source
    if source and not (tool or {}).get("source"):
        lines += _toml_table("tool.reliquary.source", source)
    if not lines:
        return pyproject
    return pyproject.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def lock(source: Path, *, run: Run = _run) -> str:
    excluded = [arg for name in sorted(PROVIDED) for arg in ("--no-emit-package", name)]
    done = run(["uv", "pip", "compile", str(source / "pyproject.toml"), "--generate-hashes", "--no-header",
                "--no-annotate", "--python-version", "3.12", *excluded])
    if done.returncode != 0:
        raise RuntimeError(f"uv pip compile failed: {(done.stderr or done.stdout)[-1000:]}")
    return done.stdout


def archive_bytes(directory: Path, root: str) -> bytes:
    """A reproducible .tar.gz: sorted entries, zero times, no owners, regular files and directories only."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if any(part in SKIP for part in relative.parts) or path.is_symlink():
                continue
            if not (path.is_file() or path.is_dir()):
                continue
            info = tar.gettarinfo(str(path), arcname=f"{root}/{relative.as_posix()}")
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def pack(source: Path, examples: Path, *, taskset: str | None = None, args: dict | None = None,
         license: str | None = None, provenance: dict | None = None, run: Run = _run) -> bytes:
    with tempfile.TemporaryDirectory(prefix="pack-") as tmp:
        work = Path(tmp) / source.name
        shutil.copytree(source, work, ignore=shutil.ignore_patterns(*SKIP), symlinks=False)
        pyproject = work / "pyproject.toml"
        if not pyproject.is_file():
            raise ValueError(f"{source} has no pyproject.toml")
        pyproject.write_text(_with_manifest(pyproject.read_text(), taskset, args, license=license,
                                            source=provenance))
        if not (work / "requirements.lock").is_file():
            (work / "requirements.lock").write_text(lock(work, run=run))
        shutil.copyfile(examples, work / "examples.jsonl")
        data = archive_bytes(work, work.name)
    read_package(data)  # refuse here what the platform would refuse
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reliquary.customer_env.pack", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path, help="the env's source directory")
    parser.add_argument("--from-prime", metavar="OWNER/NAME[@VERSION]", help="pull it from the Prime Hub instead")
    parser.add_argument("--taskset", help="the verifiers taskset id, when pyproject.toml has no [tool.reliquary]")
    parser.add_argument("--args", type=json.loads, default=None, help="the taskset's config fields, as JSON")
    parser.add_argument("--license", help="the env's license (SPDX id): a version without one cannot be public")
    parser.add_argument("--examples", type=Path, required=True, help="examples.jsonl: key, answer, expected_score")
    parser.add_argument("-o", "--output", type=Path, required=True)
    ns = parser.parse_args(argv)
    if (ns.source is None) == (ns.from_prime is None):
        parser.error("give a source directory or --from-prime, not both")
    with tempfile.TemporaryDirectory(prefix="prime-") as tmp:
        source = pull_from_prime(ns.from_prime, Path(tmp)) if ns.from_prime else ns.source
        provenance = prime_source(ns.from_prime, source) if ns.from_prime else None
        data = pack(source, ns.examples, taskset=ns.taskset, args=ns.args, license=ns.license,
                    provenance=provenance)
    ns.output.write_bytes(data)
    pkg = read_package(data)
    print(json.dumps({"output": str(ns.output), "sha256": pkg.sha256, "taskset": pkg.taskset,
                      "examples": len(pkg.examples), "requirements": len(pkg.requirements),
                      "license": pkg.license, "source": pkg.source}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
