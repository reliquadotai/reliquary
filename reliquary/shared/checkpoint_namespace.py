"""Explicit checkpoint storage boundaries, with an unchanged legacy default."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

from reliquary.shared.task_id import TASK_ID_RE


LEGACY_CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"
LEGACY_CHECKPOINT_PREFIX = "reliquary/checkpoints"
SCOPED_CHECKPOINT_NAMESPACE = "task-scoped/v1"
_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class CheckpointNamespace:
    task_id: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.task_id is None and self.run_id is None:
            return
        if not isinstance(self.task_id, str) or not TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("scoped checkpoints require a canonical task id")
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("scoped checkpoints require a canonical run id")

    @property
    def scoped(self) -> bool:
        return self.task_id is not None

    @property
    def identity(self) -> dict[str, str]:
        if not self.scoped:
            return {}
        return {"checkpoint_namespace": SCOPED_CHECKPOINT_NAMESPACE,
                "task_id": self.task_id, "training_run_id": self.run_id}

    @property
    def prefix(self) -> str:
        return f"reliquary/tasks/{self.task_id}/runs/{self.run_id}"

    @property
    def candidate_manifest_key(self) -> str:
        if not self.scoped:
            return LEGACY_CANDIDATE_MANIFEST_KEY
        return f"{self.prefix}/training/candidate-manifest.json"

    @property
    def checkpoint_prefix(self) -> str:
        if not self.scoped:
            return LEGACY_CHECKPOINT_PREFIX
        return f"{self.prefix}/checkpoints"

    def local_path(self, root: str | Path) -> Path:
        root = Path(root)
        if not self.scoped:
            return root
        return self.child_path(root, "tasks", self.task_id, "runs", self.run_id)

    def child_path(self, root: str | Path, *parts: str) -> Path:
        """Reject aliases below the configured root in the requested mode."""
        path = Path(root)
        for part in parts:
            if self.scoped and (not isinstance(part, str) or part in {"", ".", ".."}
                                or "/" in part or "\\" in part):
                raise ValueError("invalid scoped checkpoint path component")
            path = path / part
            if self.scoped and path.is_symlink():
                raise ValueError("scoped checkpoint path cannot contain a symlink")
        return path

    def require_policy(self, policy: Mapping[str, object]) -> None:
        """A configured runtime must agree with the ordered checkpoint mode."""
        if policy.get("kind") != "trainer-driven/v1":
            raise ValueError("training checkpoint runtime requires trainer-driven/v1")
        requested = policy.get("task_scoped")
        if type(requested) is not int or requested not in {0, 1} or bool(requested) != self.scoped:
            raise ValueError("checkpoint runtime namespace differs from its ordered policy")

    def require_identity(self, value: Mapping[str, object]) -> None:
        if not self.scoped:
            if "checkpoint_namespace" in value:
                raise ValueError("scoped checkpoint requires its explicit namespace")
            return
        for key, expected in self.identity.items():
            if value.get(key) != expected:
                raise ValueError(f"checkpoint namespace identity mismatch for {key}")


def active_checkpoint_namespace(env: Mapping[str, str] | None = None) -> CheckpointNamespace:
    """Opt in only; an incomplete requested identity must never fall back."""
    env = os.environ if env is None else env
    enabled = env.get("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "0")
    if enabled == "0":
        return CheckpointNamespace()
    if enabled != "1":
        raise ValueError("RELIQUARY_TASK_SCOPED_CHECKPOINTS must be 0 or 1")
    return CheckpointNamespace(task_id=env.get("RELIQUARY_TASK_ID"),
                               run_id=env.get("RELIQUARY_TRAINING_RUN_ID"))
