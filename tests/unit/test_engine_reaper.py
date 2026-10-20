"""Engine children die with their parent, even one killed with SIGKILL (engine_reaper)."""
import os
import signal
import subprocess
import sys
import time

from reliquary.miner.engine_reaper import child_pids, process_start_time

PARENT = r"""
import subprocess, sys, time
from reliquary.miner.engine_reaper import child_pids, reap_engine_children_on_exit
before = set(child_pids())
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
reap_engine_children_on_exit(before)
print(child.pid, flush=True)
time.sleep(120)
"""


def _alive(pid):
    return process_start_time(pid) is not None and "Z" not in open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1][:3]


def test_a_registered_child_dies_when_its_parent_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True,
                              cwd=os.getcwd())
    child = int(parent.stdout.readline())
    assert _alive(child)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)


def test_child_pids_lists_this_process_children():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert proc.pid in child_pids()
    finally:
        proc.kill()
        proc.wait()
