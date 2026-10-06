"""The machine directory as the validator reads it: an immutable snapshot of the R2
documents that answers `attest.MachineDirectory.public_key` (for `verify_transcript`
and `heartbeat_valid`) and gives placement each machine's address, provider, capacity
and status. Addresses come from here only, never from a heartbeat or a request."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.attest import MachineKey, StaticDirectory

from reliquary.infrastructure.sandbox_store import MACHINE_SCHEMA, MACHINE_STATUSES

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
    def from_document(cls, document: Mapping[str, Any]) -> MachineEntry:
        if document.get("schema") != MACHINE_SCHEMA:
            raise ValueError("not a machine document")
        if document["status"] not in MACHINE_STATUSES:
            raise ValueError(f"unknown status {document['status']!r}")
        keys = tuple(MachineKeyEntry(str(k["key_id"]), str(k["public_key_b64"]),
                                     int(k["valid_from"]),
                                     None if k.get("valid_until") is None else int(k["valid_until"]))
                     for k in document["keys"])
        return cls(machine_id=str(document["machine_id"]), address=str(document["address"]),
                   provider=str(document["provider"]), capacity=int(document["capacity"]),
                   status=str(document["status"]), keys=keys)


class DirectorySnapshot:
    """A key is valid at `at` inside its window, whatever its machine's status:
    revoking a machine stops new sessions, ending its key (backdated on compromise)
    is what refuses its signatures."""

    def __init__(self, entries: Iterable[MachineEntry] = ()) -> None:
        self._entries = {entry.machine_id: entry for entry in entries}
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


def snapshot_from_documents(documents: Iterable[Mapping[str, Any]]) -> DirectorySnapshot:
    entries = []
    for document in documents:
        try:
            entries.append(MachineEntry.from_document(document))
        except (KeyError, TypeError, ValueError):
            logger.error("machine document %r is unreadable; left out of the directory",
                         str(document.get("machine_id") if isinstance(document, Mapping) else "?"))
    return DirectorySnapshot(entries)
