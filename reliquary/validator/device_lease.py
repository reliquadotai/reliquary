"""One physical card, one task -- enforced across processes.

Capacity qualification binds cards by UUID, but the proof pool's spawn locks
are per process: nothing stopped two validators listing the same cuda:0.

The authority is the kernel, through ``fcntl.flock``, never the file's
contents.  A flock is tracked by open file description, so it is atomic (no
check-then-write window), it is released however the holder dies, and -- the
reason it is the only workable primitive here -- it means the same thing in
every pid namespace.  The lease directory is a shared Docker volume while
each container gets Docker's default private pid namespace, so a pid read out
of a lease file written by another container names a different process, or
one of the reader's own.  The JSON payload survives only as human-readable
metadata for the error message.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, NamedTuple

logger = logging.getLogger(__name__)


class DeviceLeaseError(RuntimeError):
    """A card is already held by a live process."""


class _Lease(NamedTuple):
    handle: IO[str]
    task_id: str


# A flock lives exactly as long as the open file description holding it: drop
# the handle and the kernel releases the lock silently, leaving the card
# unprotected with no error anywhere.  Handles therefore live here for the
# process lifetime and are never returned to a caller that could drop them.
_HELD: dict[Path, _Lease] = {}


def default_lease_directory() -> Path:
    """Beside the validator's other state, so no new mount is needed."""
    explicit = os.environ.get("RELIQUARY_DEVICE_LEASE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return Path(state_dir) / "device-leases"


def _describe_holder(handle: IO[str]) -> str:
    """Name the conflicting holder for the operator.  Never raises.

    Reading through our own handle does not disturb the holder's lock.  The
    payload is advisory: it can be empty (the file was just created by a
    racing acquirer), half-written, or from a task that wrote a different
    shape.  None of that may turn a clear refusal into a crash, so any
    failure degrades to a vaguer message.
    """
    try:
        handle.seek(0)
        record = json.loads(handle.read())
        return f"task {str(record['task_id'])!r} (pid {int(record['pid'])})"
    except Exception:  # noqa: BLE001 - a vaguer message beats an exception here
        return "another process"


def acquire_device_leases(
    device_uuids: Sequence[str],
    *,
    task_id: str,
    directory: str | os.PathLike[str],
) -> list[Path]:
    """Claim every card for ``task_id``, or claim none and raise.

    A directory that cannot be created or written is an environment fault,
    not a conflict: it is logged and treated as "no leases held" (cards
    unprotected, exactly as they are today) rather than refusing startup.
    A real conflict with a live holder still raises ``DeviceLeaseError``.
    """
    base = Path(directory)
    taken: list[Path] = []
    # Only what this call locked may be rolled back: releasing a card an
    # earlier call already holds would unprotect a card we are still using.
    fresh: list[Path] = []
    try:
        base.mkdir(parents=True, exist_ok=True)
        for device_uuid in device_uuids:
            path = base / f"{device_uuid}.lease"
            held = _HELD.get(path)
            if held is not None:
                # Ours already.  Re-acquiring for the same task is idempotent;
                # a second task in this process is the very collision this
                # module exists to refuse.
                if held.task_id != task_id:
                    raise DeviceLeaseError(
                        f"card {device_uuid} is held by task {held.task_id!r} "
                        f"(pid {os.getpid()}, this process)"
                    )
                taken.append(path)
                continue
            # Append mode creates the file without truncating a live holder's
            # metadata: the lock decides, and only a winner rewrites it.
            handle = open(path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:  # BlockingIOError is an OSError
                held_by = _describe_holder(handle)
                handle.close()
                raise DeviceLeaseError(
                    f"card {device_uuid} is held by {held_by}"
                )
            # Append-mode writes always land at end of file, which truncation
            # has just made position zero.
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {"task_id": task_id, "pid": os.getpid(), "ts": time.time()}
                )
            )
            handle.flush()
            _HELD[path] = _Lease(handle, task_id)
            taken.append(path)
            fresh.append(path)
    except DeviceLeaseError:
        release_device_leases(fresh)
        raise
    except OSError as exc:
        logger.warning(
            "device lease directory %s is unusable (%s); starting without card leases",
            base,
            exc,
        )
        release_device_leases(fresh)
        return []
    return taken


def release_device_leases(paths: Iterable[Path]) -> None:
    """Give the cards back.  A path we do not hold is ignored.

    Closing the handle is what releases the kernel lock.  The file itself is
    deliberately left in place: unlinking used to let a rollback delete a
    *live* holder's lease, and it races another acquirer recreating the
    inode.  What remains is stale metadata, overwritten by whoever takes the
    lock next.
    """
    for path in paths:
        lease = _HELD.pop(Path(path), None)
        if lease is None:
            continue
        try:
            lease.handle.close()
        except OSError:
            logger.warning(
                "could not close device lease %s", path, exc_info=True
            )
