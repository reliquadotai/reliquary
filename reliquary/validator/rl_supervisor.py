"""The controller supervisor: one async RL controller process per training task, and the map file the RL front
reads (``{"default": task | null, "tasks": {task: {"port", "run"}}}``).

A retired task keeps its controller until its bill is final (its last period settled); then it is stopped
(SIGTERM, SIGKILL after ``KILL_AFTER_S``). A controller that dies is restarted with a doubling delay."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TICK_S = 15.0
KILL_AFTER_S = 60.0
BACKOFF_START_S = 30.0
BACKOFF_MAX_S = 600.0
DEFAULT_TASK_ENV = "RELIQUARY_RL_DEFAULT_TASK"
TRAINING_PARAM = "training_order"


class _Running:
    def __init__(self, proc: Any, port: int, started: float) -> None:
        self.proc, self.port, self.started = proc, port, started


class Supervisor:
    """``registry_reader()``: the registry's entries by task id. ``spawn(task_id, port)``: a process handle
    (``poll``/``terminate``/``kill``). ``final_of(entry)``: whether a retired task's bill is final."""

    def __init__(self, registry_reader: Callable[[], Mapping[str, Any]], spawn: Callable[[str, int], Any], *,
                 ports: range, map_path: Path, clock: Callable[[], float] = time.time,
                 final_of: Callable[[Any], bool], default_task: str | None = None) -> None:
        self._registry_reader = registry_reader
        self._spawn = spawn
        self._ports = ports
        self._map_path = Path(map_path)
        self._clock = clock
        self._final_of = final_of
        self._default_task = default_task or None
        self._running: dict[str, _Running] = {}
        self._stopping: dict[str, tuple[_Running, float]] = {}     # the process, when to kill it
        self._backoff: dict[str, float] = {}
        self._restart_at: dict[str, float] = {}
        self._written: str | None = None

    def backoff_of(self, task_id: str) -> float:
        return self._backoff.get(task_id, BACKOFF_START_S)

    def _free_port(self) -> int | None:
        used = {r.port for r in self._running.values()} | {r.port for r, _ in self._stopping.values()}
        return next((p for p in self._ports if p not in used), None)

    def tick(self) -> None:
        now = self._clock()
        try:
            entries = self._registry_reader()
        except Exception as exc:
            logger.warning("rl supervisor: registry unreadable (%r); keeping what runs", exc)
            entries = None
        steps = [lambda: self._reap_stopping(now), lambda: self._revive_dead(now)]
        if entries is not None:
            steps = [steps[0], lambda: self._stop_finished(entries, now), steps[1],
                     lambda: self._start_missing(entries, now)]
        for step in [*steps, self._write_map]:
            try:
                step()
            except Exception:                  # one failed step must not take the controllers down with us
                logger.exception("rl supervisor: a tick step failed; continuing")

    def _reap_stopping(self, now: float) -> None:
        for task_id, (run, kill_at) in list(self._stopping.items()):
            if run.proc.poll() is not None:
                del self._stopping[task_id]
            elif now >= kill_at:
                run.proc.kill()

    def _stop_finished(self, entries: Mapping[str, Any], now: float) -> None:
        for task_id, run in list(self._running.items()):
            entry = entries.get(task_id)
            if entry is None or entry.status != "retired":
                continue
            if self._is_final(task_id, entry):
                logger.info("rl supervisor: stopping %s (final)", task_id)
                run.proc.terminate()
                del self._running[task_id]
                self._stopping[task_id] = (run, now + KILL_AFTER_S)

    def _is_final(self, task_id: str, entry: Any) -> bool:
        try:
            return bool(self._final_of(entry))
        except Exception as exc:
            logger.warning("rl supervisor: finality of %s unknown (%r)", task_id, exc)
            return False

    def _revive_dead(self, now: float) -> None:
        for task_id, run in list(self._running.items()):
            code = run.proc.poll()
            if code is None:
                if now - run.started >= BACKOFF_MAX_S:
                    self._backoff.pop(task_id, None)             # it held: the next death starts over
                continue
            wait = self._backoff.get(task_id, BACKOFF_START_S)
            logger.warning("rl supervisor: %s exited (%s); restart in %.0f s", task_id, code, wait)
            self._backoff[task_id] = min(wait * 2, BACKOFF_MAX_S)
            self._restart_at[task_id] = now + wait
            del self._running[task_id]

    def _start_missing(self, entries: Mapping[str, Any], now: float) -> None:
        for task_id, entry in sorted(entries.items()):
            if (entry.status not in ("active", "retired") or TRAINING_PARAM not in entry.params
                    or task_id in self._running or task_id in self._stopping
                    or self._restart_at.get(task_id, 0.0) > now):
                continue
            if entry.status == "retired" and self._is_final(task_id, entry):
                continue                       # a retired task runs on until its bill is final
            port = self._free_port()
            if port is None:
                logger.error("rl supervisor: no free port for %s", task_id)
                continue
            try:
                proc = self._spawn(task_id, port)
            except OSError as exc:
                logger.error("rl supervisor: cannot start %s (%r)", task_id, exc)
                self._restart_at[task_id] = now + self._backoff.get(task_id, BACKOFF_START_S)
                continue
            self._restart_at.pop(task_id, None)
            self._running[task_id] = _Running(proc, port, now)

    def _default(self) -> str | None:
        if self._default_task in self._running:
            return self._default_task
        return min(self._running, key=lambda t: (self._running[t].started, t), default=None)

    def _write_map(self) -> None:
        body = json.dumps({"default": self._default(),
                           "tasks": {t: {"port": r.port, "run": t} for t, r in sorted(self._running.items())}},
                          sort_keys=True)
        if body == self._written:
            return
        self._map_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._map_path.parent, prefix=".map-")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(body)
            os.replace(tmp, self._map_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._written = body


def subprocess_spawn(command: list[str], state_dir: Path, env: Mapping[str, str]) -> Callable[[str, int], Any]:
    """Start ``command`` (``reliquary validate`` and the host's wallet options) as an async RL controller on
    loopback: no ``--external-ip``, the front owns the axon."""
    def spawn(task_id: str, port: int) -> Any:
        state_dir.mkdir(parents=True, exist_ok=True)
        log = open(state_dir / f"{task_id}.log", "ab")
        try:
            return subprocess.Popen(
                [*command, "--http-host", "127.0.0.1", "--http-port", str(port)],
                env={**env, "RELIQUARY_RL_MODE": "async", "RELIQUARY_TASK_ID": task_id,
                     "RELIQUARY_RL_RUN": task_id},
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        finally:
            log.close()
    return spawn


class StateLocked(RuntimeError):
    """Another supervisor holds the state directory."""


def lock_state_dir(state_dir: Path) -> Any:
    """Hold ``<state_dir>/supervisor.lock`` for the process lifetime: close the returned file to release."""
    import fcntl

    state_dir.mkdir(parents=True, exist_ok=True)
    handle = open(state_dir / "supervisor.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise StateLocked(f"another supervisor holds {state_dir}") from exc
    return handle


def read_registry_entries() -> Mapping[str, Any]:
    import asyncio

    from reliquary.infrastructure import task_registry_store

    return asyncio.run(task_registry_store.read_registry(strict=False))[0]


def admin_final(entry: Any) -> bool:
    """The admin's own ``final`` for a training task (the spend route's rule, on the summary in the bucket)."""
    from reliquary.admin.service import _round_time
    from reliquary.admin.training import training_spend
    from reliquary.validator import rl_spend

    if entry.status != "retired" or entry.retired_at is None:
        return False
    summary = rl_spend.read_spend(entry.task_id)           # a training task's run is its task id
    if summary is None or summary.get("task_id") != entry.task_id:
        return False
    return bool(training_spend(entry, summary, retired_time=_round_time(int(entry.retired_at)))["final"])


__all__ = ["DEFAULT_TASK_ENV", "StateLocked", "Supervisor", "TICK_S", "lock_state_dir", "admin_final", "read_registry_entries", "subprocess_spawn"]
