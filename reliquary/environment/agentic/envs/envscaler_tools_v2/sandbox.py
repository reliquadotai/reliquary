"""The parent side of a world process: start it, talk to it, end it.

One process per episode, started at ``reset`` and killed at ``close``. A
process per episode rather than per call keeps the world's state where the
world put it (no re-execution of earlier calls), and rather than one per
miner or validator keeps two rollouts from ever sharing a uuid counter, a
random generator or a mutated class attribute — which ``envscaler_tools_v1``
did, through a cached class.

Why a local subprocess and not the gVisor grading service: that service runs
stateless programs (a source and its stdin, a verdict back) behind an HTTP
pool on validators only. An episode needs a stateful world answering a dozen
calls, on the miner and on the corpus generator as much as on the validator.
Reusing it would mean a new session protocol in a service miners do not run;
a subprocess under rlimits, a namespace and an audit hook (see ``worker``)
contains what this code can do with the arguments a policy chooses, and costs
one interpreter start per episode.

Every failure of the process — killed by its CPU limit, out of memory, past
the wall-clock deadline, a reply that is not the protocol — surfaces as one
``WorldProcessError`` with a fixed message, so the observation it becomes is
the same bytes on every machine. The only place timing can leak into a replay
is a call that sits on the CPU limit itself, ten seconds of CPU for a world
whose calls take milliseconds.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any
import weakref

WORKER_PATH = Path(__file__).with_name("worker.py")
PROTOCOL = "envscaler-world/v1"
# The child's whole environment. Nothing is inherited: no PYTHONPATH, no
# credentials, no proxy settings. PYTHONHASHSEED fixes set and dict-of-hash
# order; TZ fixes every local-time conversion; UTF-8 mode fixes encodings.
WORKER_ENVIRONMENT: Mapping[str, str] = {
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TZ": "UTC",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
}
# -S: no site, so nothing installed on the host is importable by a world.
# -P: the worker's own directory is not put on sys.path. -B: no bytecode.
# Not -I, which implies -E and would discard the two variables above.
WORKER_FLAGS = ("-S", "-P", "-B")
# Wall-clock bounds. The CPU limit inside the worker is the real bound on a
# runaway; these catch a world that blocks without spending CPU.
START_SECONDS = 30.0
REQUEST_SECONDS = 30.0
# A reply is at most the worker's MAX_RESULT_BYTES plus its framing.
MAX_REPLY_BYTES = (1 << 20) + 4096
TERMINATED = "world process terminated"


class WorldProcessError(RuntimeError):
    """The world process is gone. The message is fixed on purpose."""


def worker_argv() -> list[str]:
    return [sys.executable, *WORKER_FLAGS, str(WORKER_PATH)]


def _kill(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for stream in (process.stdin, process.stdout):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


class WorldProcess:
    """One running world. Not thread-safe; an episode is driven by one thread."""

    def __init__(
        self,
        *,
        start_seconds: float | None = None,
        request_seconds: float | None = None,
    ) -> None:
        # Read at call time, so a test can shorten them for every world the
        # environment starts.
        start_seconds = START_SECONDS if start_seconds is None else start_seconds
        self._request_seconds = float(
            REQUEST_SECONDS if request_seconds is None else request_seconds
        )
        self._buffer = b""
        self._process: subprocess.Popen | None = subprocess.Popen(
            worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=dict(WORKER_ENVIRONMENT),
            cwd="/",
            close_fds=True,
            start_new_session=True,
        )
        # Garbage collection kills a world nobody closed.
        self._finalizer = weakref.finalize(self, _kill, self._process)
        try:
            hello = self._read(start_seconds)
        except WorldProcessError:
            raise
        except BaseException:
            self.close()
            raise
        if hello.get("protocol") != PROTOCOL:
            self.close()
            raise WorldProcessError(TERMINATED)
        self.network = str(hello.get("network", ""))

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        if not self.alive:
            raise WorldProcessError(TERMINATED)
        payload = json.dumps(
            message, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except TimeoutError:
            # A caller's deadline (a replay worker's SIGALRM) is the caller's,
            # never a world failure: TimeoutError is an OSError, so it would
            # otherwise become one below and the deadline would be swallowed.
            self.close()
            raise
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            raise WorldProcessError(TERMINATED) from None
        try:
            return self._read(self._request_seconds)
        except WorldProcessError:
            raise
        except BaseException:
            # Interrupted mid-reply (a caller's deadline): the pipe may hold
            # half a line, so this world can never be spoken to again.
            self.close()
            raise

    def _read(self, seconds: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise WorldProcessError(TERMINATED)
        descriptor = process.stdout.fileno()
        deadline = time.monotonic() + seconds
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or len(self._buffer) > MAX_REPLY_BYTES:
                    self.close()
                    raise WorldProcessError(TERMINATED)
                if not selector.select(remaining):
                    continue
                chunk = os.read(descriptor, 1 << 16)
                if not chunk:
                    self.close()
                    raise WorldProcessError(TERMINATED)
                self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            reply = json.loads(line)
        except ValueError:
            reply = None
        if not isinstance(reply, dict):
            self.close()
            raise WorldProcessError(TERMINATED)
        return reply

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            self._finalizer.detach()
            _kill(process)


__all__ = [
    "PROTOCOL",
    "TERMINATED",
    "WORKER_ENVIRONMENT",
    "WORKER_FLAGS",
    "WorldProcess",
    "WorldProcessError",
    "worker_argv",
]
