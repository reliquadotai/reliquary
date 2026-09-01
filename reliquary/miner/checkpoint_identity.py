"""Restart-safe checkpoint identity for the reference miner."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from reliquary.shared.checkpoint_identity import (
    require_checkpoint_repository,
    require_immutable_checkpoint_revision,
)


_SCHEMA_VERSION = 1


class CheckpointIdentityError(RuntimeError):
    """A checkpoint identity is invalid, corrupt, rolled back, or rebound."""


@dataclass(frozen=True)
class ActivatedCheckpoint:
    checkpoint_n: int
    repo_id: str
    oid: str

    def __post_init__(self) -> None:
        if type(self.checkpoint_n) is not int or self.checkpoint_n < 0:
            raise ValueError("checkpoint number must be a non-negative integer")
        require_checkpoint_repository(self.repo_id)
        require_immutable_checkpoint_revision(self.oid)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_n": self.checkpoint_n,
            "repo_id": self.repo_id,
            "oid": self.oid,
        }

    @classmethod
    def from_record(cls, record: object) -> "ActivatedCheckpoint":
        if not isinstance(record, dict) or set(record) != {
            "schema_version",
            "checkpoint_n",
            "repo_id",
            "oid",
        }:
            raise ValueError("checkpoint identity record has an invalid shape")
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("checkpoint identity schema is unsupported")
        return cls(
            checkpoint_n=record["checkpoint_n"],
            repo_id=record["repo_id"],
            oid=record["oid"],
        )


def checkpoint_identity_from_state(state) -> ActivatedCheckpoint | None:
    """Parse the complete public identity from a validator state response."""

    checkpoint_n = getattr(state, "checkpoint_n", None)
    repo_id = getattr(state, "checkpoint_repo_id", None)
    revision = getattr(state, "checkpoint_revision", None)
    if checkpoint_n == 0 and repo_id is None and revision is None:
        return None
    try:
        return ActivatedCheckpoint(
            checkpoint_n=checkpoint_n,
            repo_id=repo_id,
            oid=revision,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIdentityError(
            "validator advertised an invalid checkpoint identity"
        ) from exc


def require_successor(
    current: ActivatedCheckpoint | None,
    candidate: ActivatedCheckpoint,
) -> None:
    """Reject rollback and same-number identity rebinding."""

    if current is None:
        return
    if candidate.checkpoint_n < current.checkpoint_n:
        raise CheckpointIdentityError("checkpoint number rolled back")
    if candidate.checkpoint_n == current.checkpoint_n and candidate != current:
        raise CheckpointIdentityError(
            "checkpoint number was rebound to a different repository or revision"
        )


def default_checkpoint_identity_path(wallet_address: str) -> Path:
    """Return a stable per-network, per-miner record path."""

    state_root = Path(
        os.environ.get("RELIQUARY_MINER_STATE_DIR")
        or os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    )
    netuid = os.environ.get("NETUID", "unknown")
    network = os.environ.get("BT_NETWORK", "unknown")
    scope = hashlib.sha256(
        f"{network}\0{netuid}\0{wallet_address}".encode("utf-8")
    ).hexdigest()[:24]
    return state_root / "miner" / f"checkpoint-{scope}.json"


class MinerCheckpointIdentityStore:
    """Atomically persist one monotonic activated checkpoint identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> ActivatedCheckpoint | None:
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
            return ActivatedCheckpoint.from_record(record)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise CheckpointIdentityError(
                "durable checkpoint identity is corrupt"
            ) from exc

    def assert_advertisement(self, candidate: ActivatedCheckpoint) -> None:
        require_successor(self.load(), candidate)

    def commit(self, candidate: ActivatedCheckpoint) -> None:
        """Create or advance the record after both miner models activate."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            current = self.load()
            require_successor(current, candidate)
            if current == candidate:
                return
            payload = json.dumps(
                candidate.to_record(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as file_handle:
                    file_handle.write(payload)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                os.replace(temporary, self.path)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                temporary.unlink(missing_ok=True)
                raise
