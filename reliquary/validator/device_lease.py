"""One physical card, one task -- enforced across processes.

Capacity qualification binds cards by UUID, but the proof pool's spawn locks
are per process: nothing stopped two validators listing the same cuda:0.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceLeaseError(RuntimeError):
    """A card is already held by a live process."""


def default_lease_directory() -> Path:
    """Beside the validator's other state, so no new mount is needed."""
    explicit = os.environ.get("RELIQUARY_DEVICE_LEASE_DIR", "").strip()
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return Path(state_dir) / "device-leases"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder(path: Path) -> tuple[str, int] | None:
    """The task and pid holding this card, or None if the lease is stale."""
    try:
        record = json.loads(path.read_text())
        task_id, pid = str(record["task_id"]), int(record["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (task_id, pid) if _pid_alive(pid) else None


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
    try:
        base.mkdir(parents=True, exist_ok=True)
        for device_uuid in device_uuids:
            path = base / f"{device_uuid}.lease"
            holder = _holder(path) if path.exists() else None
            # Same task, another live pid: a rolling restart. Leases are never
            # released on shutdown, so refusing it would make every restart a
            # manual unlock.
            if holder is not None and holder[0] != task_id:
                raise DeviceLeaseError(
                    f"card {device_uuid} is held by task {holder[0]!r} (pid {holder[1]})"
                )
            payload = json.dumps(
                {"task_id": task_id, "pid": os.getpid(), "ts": time.time()}
            )
            staging = path.with_suffix(".lease.staging")
            staging.write_text(payload)
            os.replace(staging, path)
            taken.append(path)
    except DeviceLeaseError:
        release_device_leases(taken)
        raise
    except OSError as exc:
        logger.warning(
            "device lease directory %s is unusable (%s); starting without card leases",
            base,
            exc,
        )
        release_device_leases(taken)
        return []
    return taken


def release_device_leases(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
