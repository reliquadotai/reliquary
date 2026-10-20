"""Opt-in handoff to a pinned, separately installed authenticated trainer."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TEE_RUNTIME_COMMIT = "2203514f4cf049094a1f8a60edfc57cf8edfc0be"
TEE_RUNTIME_VERSION = "0.4.0rc1"
TEE_RUNTIME_URL = "https://github.com/reliquadotai/reliquary-TEE.git"

_PATH_OPTIONS = (
    "config",
    "model_dir",
    "assets",
    "current_policy",
    "trust_policy",
    "queue",
    "checkpoints",
)


class AttestedTrainingJob(BaseModel):
    """Operator-owned local configuration; never accepted from a miner."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Literal["reliquary/attested-training-job/v1"]
    runtime_python: str
    config: str
    model_dir: str
    assets: str
    current_policy: str
    trust_policy: str
    queue: str
    checkpoints: str
    device: Literal["cuda", "cpu-development"] = "cuda"
    max_steps: int = Field(default=1, ge=1, le=100000)
    idle_seconds: int = Field(default=0, ge=0, le=86400)
    lease_seconds: int = Field(default=300, ge=1, le=3600)

    @field_validator("runtime_python", *_PATH_OPTIONS)
    @classmethod
    def absolute_paths(cls, value: str) -> str:
        if "\0" in value or not Path(value).is_absolute():
            raise ValueError("job paths must be absolute")
        return value

    def trainer_args(self) -> list[str]:
        args = ["trainer"]
        for name in _PATH_OPTIONS:
            args.extend((f"--{name.replace('_', '-')}", getattr(self, name)))
        for name in ("device", "max_steps", "idle_seconds", "lease_seconds"):
            args.extend((f"--{name.replace('_', '-')}", str(getattr(self, name))))
        return args


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate job field")
        result[key] = value
    return result


def load_job(path: Path) -> AttestedTrainingJob:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o022
        ):
            raise ValueError("job must be an owner-controlled regular file")
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("job exceeds 16 KiB")
    document = json.loads(raw, object_pairs_hook=_unique_object)
    return AttestedTrainingJob.model_validate(document)


# Check the installed distribution and enter its CLI in the same interpreter.
# This is installation provenance, not hardware attestation or local tamper detection.
_BOOTSTRAP = """
import importlib.metadata
import json
import os
import sys

expected_url, expected_commit, expected_version, parent_prefix = sys.argv[1:5]
if sys.prefix == sys.base_prefix or os.path.realpath(sys.prefix) == os.path.realpath(parent_prefix):
    raise SystemExit("refused: attested trainer requires a separate virtual environment")
try:
    dist = importlib.metadata.distribution("reliquary-tee")
    source = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = source.get("vcs_info", {})
    valid = (
        dist.version == expected_version
        and source.get("url") == expected_url
        and vcs.get("vcs") == "git"
        and vcs.get("commit_id") == expected_commit
        and not source.get("dir_info", {}).get("editable", False)
    )
except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError, TypeError):
    valid = False
if not valid:
    raise SystemExit("refused: attested trainer installation does not match the pinned Git revision")
sys.argv = ["reliquary-tee", *sys.argv[5:]]
from reliquary_tee.cli import main
main()
"""


def run_attested_worker(path: Path) -> None:
    """Replace this process, preserving supervisor signals and child exit status."""
    job = load_job(path)
    args = [
        job.runtime_python,
        "-I",
        "-c",
        _BOOTSTRAP,
        TEE_RUNTIME_URL,
        TEE_RUNTIME_COMMIT,
        TEE_RUNTIME_VERSION,
        sys.prefix,
        *job.trainer_args(),
    ]
    os.execv(job.runtime_python, args)
