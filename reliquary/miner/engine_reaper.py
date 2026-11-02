"""Engine processes die with the miner, even when the miner is killed with SIGKILL.

vLLM runs its engine core in a spawned child that holds the GPU; a miner killed without a chance to clean up
leaves that child running with its memory, and the next miner on the card cannot start. ``EngineReaper`` starts a
small watcher in a session of its own whose stdin is a pipe from the miner: the pipe closes when the miner dies,
however it dies, and the watcher then kills every process the miner registered. A process is named by its pid and
its start time, so a pid the kernel reused after the engine exited is never touched.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_WATCHER = r"""
import os, signal, sys
def start_time(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None
watched = []
for line in sys.stdin:
    pid, start = line.split()
    watched.append((int(pid), start))
for pid, start in watched:
    if start_time(pid) == start:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
"""


def process_start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks); None when the process is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


def child_pids(parent: int | None = None) -> list[int]:
    """The direct children of ``parent`` (default: this process), read from /proc."""
    parent = os.getpid() if parent is None else int(parent)
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == parent:
            out.append(int(entry))
    return sorted(out)


class EngineReaper:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._watched: set[int] = set()

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen([sys.executable, "-I", "-c", _WATCHER], stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                                          text=True)
            self._watched = set()
        return self._proc

    def watch(self, pid: int) -> bool:
        """Kill ``pid`` when this process dies; False when ``pid`` is already gone."""
        start = process_start_time(pid)
        if start is None:
            return False
        proc = self._ensure()
        if pid not in self._watched:
            proc.stdin.write(f"{int(pid)} {start}\n")
            proc.stdin.flush()
            self._watched.add(pid)
        return True

    def watch_children(self, exclude: set[int] | frozenset[int] = frozenset()) -> list[int]:
        """Watch every current child of this process but ``exclude`` and the watcher itself."""
        own = None if self._proc is None else self._proc.pid
        pids = [pid for pid in child_pids() if pid not in exclude and pid != own]
        try:
            return [pid for pid in pids if self.watch(pid)]
        except OSError:
            logger.warning("the engine reaper is not running: engine children may outlive this process")
            return []


_REAPER: EngineReaper | None = None


def reap_engine_children_on_exit(before: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """Children started since ``before`` (``set(child_pids())`` taken before building the engine) die with this
    process. Best effort: never fails the caller."""
    global _REAPER
    try:
        _REAPER = _REAPER or EngineReaper()
        watched = _REAPER.watch_children(exclude=set(before))
    except Exception:
        logger.warning("engine children could not be registered with the reaper", exc_info=True)
        return []
    if watched:
        logger.info("engine processes %s die with this miner", watched)
    return watched


__all__ = ["EngineReaper", "child_pids", "process_start_time", "reap_engine_children_on_exit"]
