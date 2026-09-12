"""A card belongs to one task at a time, across processes."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from reliquary.validator.device_lease import (
    DeviceLeaseError,
    acquire_device_leases,
    release_device_leases,
)

UUID_A = "gpu-0000-aaaa"
UUID_B = "gpu-1111-bbbb"


def test_a_free_card_is_leased(tmp_path):
    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert len(paths) == 1
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

    acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)


def test_a_partial_conflict_leaves_no_lease_behind(tmp_path):
    acquire_device_leases([UUID_B], task_id="default", directory=tmp_path)

    with pytest.raises(DeviceLeaseError):
        acquire_device_leases([UUID_A, UUID_B], task_id="logic-probe", directory=tmp_path)

    assert not (tmp_path / f"{UUID_A}.lease").exists()


def test_the_same_task_can_retake_its_own_cards(tmp_path):
    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)


def test_an_unusable_directory_does_not_refuse_startup(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory")

    assert acquire_device_leases([UUID_A], task_id="default", directory=blocked / "leases") == []
