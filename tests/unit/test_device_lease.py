"""A card belongs to one task at a time, across processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from reliquary.validator.device_lease import (
    DeviceLeaseError,
    acquire_device_leases,
    release_device_leases,
)

UUID_A = "gpu-0000-aaaa"
UUID_B = "gpu-1111-bbbb"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Acquires a card and holds it until killed. Note that it drops the value
# returned by ``acquire_device_leases``: if the lock lived on a handle handed
# back to the caller instead of the module's registry, it would be garbage
# collected here and the card would silently go free.
_HOLDER_SOURCE = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from reliquary.validator.device_lease import acquire_device_leases

    directory, device_uuid, ready = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    acquire_device_leases([device_uuid], task_id="logic-probe", directory=directory)
    ready.write_text("held")
    sys.stdin.read()  # block until the parent kills us
    """
)


def _spawn_holder(tmp_path: Path, device_uuid: str) -> subprocess.Popen:
    """A separate process holding ``device_uuid``, returned once it certainly has it."""
    ready = tmp_path / "holder-ready"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SOURCE, str(tmp_path), device_uuid, str(ready)],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 60.0
    while not ready.exists():
        if child.poll() is not None:
            raise AssertionError(f"holder exited early: {child.stderr.read()}")
        if time.monotonic() > deadline:
            child.kill()
            child.wait()
            raise AssertionError("holder never took its lease")
        time.sleep(0.05)
    return child


def test_a_free_card_is_leased(tmp_path):
    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["task_id"] == "default"


def test_a_card_held_by_another_process_is_refused(tmp_path):
    """The whole point of the module: the holder is in another process."""
    child = _spawn_holder(tmp_path, UUID_A)
    try:
        with pytest.raises(DeviceLeaseError, match="logic-probe"):
            acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)
    finally:
        child.kill()
        child.wait()


def test_a_card_freed_by_a_dead_process_is_reclaimed(tmp_path):
    """Death releases the lock, whatever kind of death it was."""
    child = _spawn_holder(tmp_path, UUID_A)
    child.kill()
    child.wait()

    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert json.loads(paths[0].read_text())["task_id"] == "default"


def test_a_card_held_by_a_live_process_is_refused(tmp_path):
    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    with pytest.raises(DeviceLeaseError, match="default"):
        acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)


def test_a_lease_left_by_a_dead_process_is_reclaimed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (tmp_path / f"{UUID_A}.lease").write_text(
        json.dumps({"task_id": "default", "pid": dead.pid, "ts": 0})
    )

    paths = acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)

    assert json.loads(paths[0].read_text())["task_id"] == "logic-probe"


def test_releasing_frees_the_card(tmp_path):
    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)
    release_device_leases(paths)

    retaken = acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)

    assert retaken == paths
    assert json.loads(retaken[0].read_text())["task_id"] == "logic-probe"


def test_releasing_a_card_we_never_took_is_tolerated(tmp_path):
    release_device_leases([tmp_path / f"{UUID_A}.lease"])

    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert json.loads(paths[0].read_text())["task_id"] == "default"


def test_a_partial_conflict_leaves_the_card_claimable(tmp_path):
    """The lease file survives -- the claim on it must not."""
    acquire_device_leases([UUID_B], task_id="default", directory=tmp_path)

    with pytest.raises(DeviceLeaseError):
        acquire_device_leases([UUID_A, UUID_B], task_id="logic-probe", directory=tmp_path)

    paths = acquire_device_leases([UUID_A], task_id="third-task", directory=tmp_path)

    assert json.loads(paths[0].read_text())["task_id"] == "third-task"


def test_the_same_task_can_retake_its_own_cards(tmp_path):
    first = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    second = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert second == first
    assert json.loads(second[0].read_text())["task_id"] == "default"
    # Re-acquiring must not have quietly dropped the claim.
    with pytest.raises(DeviceLeaseError, match="default"):
        acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)


def test_an_unusable_directory_does_not_refuse_startup(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory")

    assert acquire_device_leases([UUID_A], task_id="default", directory=blocked / "leases") == []
