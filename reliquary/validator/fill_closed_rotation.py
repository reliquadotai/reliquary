"""Durable synchronization barrier between experimental fill windows.

The gate is intentionally small and dependency-free.  It records the last
locally committed trainer-journal key for one closed window and, when that
window produced a complete publication cadence, the parent checkpoint that a
successor must replace.  Absence is not success: the validator clears this
file only after the measured trainer cursor and (when required) an adopted
checkpoint both cover the gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from reliquary.shared.checkpoint_identity import (
    require_immutable_checkpoint_revision,
)
from reliquary.shared.strict_json import strict_json_loads


@dataclass(frozen=True)
class FillClosedRotationGate:
    source_window: int
    required_journal_key: int
    parent_checkpoint_n: int
    parent_revision: str
    durable_payload_count: int
    requires_successor: bool
    adopted_checkpoint_n: int | None = None
    adopted_revision: str | None = None
    adopted_trainer_cursor: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported fill-closed rotation gate schema")
        for field in (
            "source_window",
            "required_journal_key",
            "parent_checkpoint_n",
            "durable_payload_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        require_immutable_checkpoint_revision(
            self.parent_revision,
            field="fill-closed parent revision",
        )
        if not isinstance(self.requires_successor, bool):
            raise ValueError("requires_successor must be a boolean")
        adoption = (
            self.adopted_checkpoint_n,
            self.adopted_revision,
            self.adopted_trainer_cursor,
        )
        if any(value is not None for value in adoption) and not all(
            value is not None for value in adoption
        ):
            raise ValueError("checkpoint adoption fields must be all set or absent")
        if all(value is not None for value in adoption):
            if (
                isinstance(self.adopted_checkpoint_n, bool)
                or not isinstance(self.adopted_checkpoint_n, int)
                or self.adopted_checkpoint_n < 0
            ):
                raise ValueError("adopted_checkpoint_n must be non-negative")
            require_immutable_checkpoint_revision(
                self.adopted_revision,
                field="fill-closed adopted revision",
            )
            if (
                isinstance(self.adopted_trainer_cursor, bool)
                or not isinstance(self.adopted_trainer_cursor, int)
                or self.adopted_trainer_cursor < 0
            ):
                raise ValueError("adopted_trainer_cursor must be non-negative")

    def record_adoption(
        self,
        *,
        checkpoint_n: int,
        revision: str,
        trained_cursor: int,
    ) -> "FillClosedRotationGate":
        return replace(
            self,
            adopted_checkpoint_n=checkpoint_n,
            adopted_revision=revision,
            adopted_trainer_cursor=trained_cursor,
        )

    def adoption_covers(self, checkpoint: Any | None) -> bool:
        """Whether the active manifest is the recorded covering successor."""
        if not self.requires_successor:
            return True
        if checkpoint is None or self.adopted_checkpoint_n is None:
            return False
        checkpoint_n = getattr(checkpoint, "checkpoint_n", None)
        revision = getattr(checkpoint, "revision", None)
        return (
            type(checkpoint_n) is int
            and checkpoint_n == self.adopted_checkpoint_n
            and isinstance(revision, str)
            and revision == self.adopted_revision
            and self.adopted_checkpoint_n > self.parent_checkpoint_n
            and self.adopted_revision != self.parent_revision
            and self.adopted_trainer_cursor is not None
            and self.adopted_trainer_cursor >= self.required_journal_key
        )

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "FillClosedRotationGate":
        try:
            payload = strict_json_loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid fill-closed rotation gate JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("fill-closed rotation gate must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("fill-closed rotation gate fields differ")
        gate = cls(**payload)
        if raw != gate.to_bytes():
            raise ValueError("fill-closed rotation gate is not canonical")
        return gate


class FillClosedRotationStore:
    """Single-record, atomic local store for the active rotation barrier."""

    def __init__(self, state_dir: str | Path) -> None:
        self.path = Path(state_dir) / "fill_closed_rotation.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> FillClosedRotationGate | None:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return None
        return FillClosedRotationGate.from_bytes(raw)

    def save(self, gate: FillClosedRotationGate) -> None:
        raw = gate.to_bytes()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
