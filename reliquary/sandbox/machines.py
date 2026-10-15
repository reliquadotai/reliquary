"""The machine directory as the validator reads it: an immutable snapshot of the R2
documents that answers `attest.MachineDirectory.public_key` (for `verify_transcript`
and `heartbeat_valid`) and gives placement each machine's address, provider, capacity
and status. Addresses come from here only, never from a heartbeat or a request.
Machine documents are never deleted, so past transcripts keep their keys."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.attest import MachineKey, StaticDirectory

from reliquary.infrastructure.sandbox_store import (
    MACHINE_SCHEMA, MACHINE_STATUSES, unix_seconds, validated_address, validated_machine_id,
    validated_public_key,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MachineKeyEntry:
    key_id: str
    public_key_b64: str
    valid_from: int
    valid_until: int | None


@dataclass(frozen=True)
class MachineEntry:
    machine_id: str
    address: str
    provider: str
    capacity: int
    status: str
    keys: tuple[MachineKeyEntry, ...]

    @classmethod
    def from_document(cls, document: Any) -> MachineEntry:
        """Strict: a document the store could not have written is refused (ValueError),
        including one listing a key id or a public key twice."""
        if not isinstance(document, Mapping) or document.get("schema") != MACHINE_SCHEMA:
            raise ValueError("not a machine document")
        status, provider, capacity = (document.get(f) for f in ("status", "provider", "capacity"))
        if status not in MACHINE_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty name")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        raw_keys = document.get("keys")
        if not isinstance(raw_keys, list):
            raise ValueError("keys must be a list")
        keys = []
        for raw in raw_keys:
            if not isinstance(raw, Mapping):
                raise ValueError("a key must be an object")
            until = raw.get("valid_until")
            keys.append(MachineKeyEntry(
                validated_machine_id(raw.get("key_id")),
                validated_public_key(raw.get("public_key_b64")),
                unix_seconds(raw.get("valid_from"), "valid_from"),
                None if until is None else unix_seconds(until, "valid_until")))
        if len({k.key_id for k in keys}) != len(keys):
            raise ValueError("a key id is listed twice")
        if len({k.public_key_b64 for k in keys}) != len(keys):
            raise ValueError("a public key is listed twice")
        return cls(machine_id=validated_machine_id(document.get("machine_id")),
                   address=validated_address(document.get("address")),
                   provider=provider, capacity=capacity, status=status, keys=tuple(keys))


class DirectorySnapshot:
    """A key is valid at `at` inside its window, whatever its machine's status:
    revoking a machine stops new sessions, ending its key (backdated on compromise)
    is what refuses its signatures."""

    def __init__(self, entries: Iterable[MachineEntry] = ()) -> None:
        entries = list(entries)
        self._entries = {entry.machine_id: entry for entry in entries}
        if len(self._entries) != len(entries):
            raise ValueError("a machine id is listed twice")
        self._keys = StaticDirectory([
            MachineKey(entry.machine_id, key.key_id, key.public_key_b64, key.valid_from,
                       key.valid_until)
            for entry in self._entries.values() for key in entry.keys])

    def public_key(self, machine_id: str, key_id: str, at: int):
        try:
            return self._keys.public_key(machine_id, key_id, at)
        except ValueError:                   # a stored key that does not decode
            return None

    def entry(self, machine_id: str) -> MachineEntry | None:
        return self._entries.get(machine_id)

    def entries(self) -> tuple[MachineEntry, ...]:
        return tuple(self._entries[m] for m in sorted(self._entries))


def snapshot_from_documents(documents: Iterable[Any]) -> DirectorySnapshot:
    """Pure. A document that does not read is logged and left out; a machine id that
    two documents claim is left out entirely (neither is trusted) and logged."""
    entries: dict[str, list[MachineEntry]] = {}
    for document in documents:
        try:
            entry = MachineEntry.from_document(document)
        except (KeyError, TypeError, ValueError):
            logger.error("machine document %r is unreadable; left out of the directory",
                         str(document.get("machine_id") if isinstance(document, Mapping) else "?"))
            continue
        entries.setdefault(entry.machine_id, []).append(entry)
    for machine_id in sorted(m for m, found in entries.items() if len(found) > 1):
        logger.error("machine %r is listed twice; left out of the directory", machine_id)
    return DirectorySnapshot(found[0] for found in entries.values() if len(found) == 1)
