"""The controller supervisor: one async RL controller process per training task, and the map file the RL front
reads (``{"default": task | null, "tasks": {task: {"port", "run", "model"?}}}``). ``model`` (``{"id", "revision"}``,
from the task's generation contract) lets miners keep to tasks their engine can play.

A retired task keeps its controller until its bill is final (its last period settled); then it is stopped
(SIGTERM, SIGKILL after ``KILL_AFTER_S``). A controller that dies is restarted with a doubling delay.

The default task (bare paths) is ``RELIQUARY_RL_DEFAULT_TASK`` when kept, else the one already chosen while it is
kept (running or restarting: the front answers 503 meanwhile), else the first task seen (``order_path`` keeps that
order across supervisor restarts)."""
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
NO_SPEND_ALERT_AFTER_S = 24 * 3600.0
NO_SPEND_ALERT_EVERY_S = 3600.0
DEFAULT_TASK_ENV = "RELIQUARY_RL_DEFAULT_TASK"
TRAINING_PARAM = "training_order"


class _Running:
    def __init__(self, proc: Any, port: int, started: float) -> None:
        self.proc, self.port, self.started = proc, port, started


class Supervisor:
    """``registry_reader()``: the registry's entries by task id. ``spawn(task_id, port, entry)``: a process handle
    (``poll``/``terminate``/``kill``). ``final_of(entry)``: whether a retired task's bill is final. ``has_spend(entry)``:
    whether its controller ever wrote a spend summary (a retired task without one for a day is logged hourly)."""

    def __init__(self, registry_reader: Callable[[], Mapping[str, Any]], spawn: Callable[[str, int, Any], Any], *,
                 ports: range, map_path: Path, clock: Callable[[], float] = time.time,
                 final_of: Callable[[Any], bool], default_task: str | None = None,
                 order_path: Path | None = None, has_spend: Callable[[Any], bool] | None = None) -> None:
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
        self._models: dict[str, dict] = {}
        self._finished: set[str] = set()                           # stopped for good (final)
        self._chosen: str | None = None                            # the current default
        self._order_path = None if order_path is None else Path(order_path)
        self._order: list[str] = self._load_order()
        self._has_spend = has_spend
        self._retired_seen: dict[str, float] = {}
        self._alerted_at: dict[str, float] = {}

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
                     lambda: self._start_missing(entries, now), lambda: self._read_models(entries),
                     lambda: self._note_order(entries), lambda: self._alert_unbilled(entries, now)]
        for step in [*steps, lambda: self._choose_default(entries), self._write_map]:
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
                self._finished.add(task_id)
                self._restart_at.pop(task_id, None)

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
                self._finished.add(task_id)
                self._restart_at.pop(task_id, None)
                continue                       # a retired task runs on until its bill is final
            port = self._free_port()
            if port is None:
                logger.error("rl supervisor: no free port for %s", task_id)
                continue
            try:
                proc = self._spawn(task_id, port, entry)
            except OSError as exc:
                logger.error("rl supervisor: cannot start %s (%r)", task_id, exc)
                self._restart_at[task_id] = now + self._backoff.get(task_id, BACKOFF_START_S)
                continue
            self._restart_at.pop(task_id, None)
            self._running[task_id] = _Running(proc, port, now)

    def _read_models(self, entries: Mapping[str, Any]) -> None:
        self._models = {t: m for t in self._running if (m := task_model(entries.get(t))) is not None}

    def _load_order(self) -> list[str]:
        if self._order_path is None:
            return []
        try:
            order = json.loads(self._order_path.read_text())
            return [str(t) for t in order] if isinstance(order, list) else []
        except (OSError, ValueError):
            return []

    def _note_order(self, entries: Mapping[str, Any]) -> None:
        known = [t for t in self._order if t in entries]
        new = sorted(t for t, e in entries.items() if TRAINING_PARAM in e.params and t not in known)
        if known + new == self._order:
            return
        self._order = known + new
        if self._order_path is not None:
            _write_atomic(self._order_path, json.dumps(self._order))

    def _kept(self, task_id: str | None, entries: Mapping[str, Any] | None) -> bool:
        if task_id is None or task_id in self._finished or (entries is not None and task_id not in entries):
            return False
        return task_id in self._running or task_id in self._restart_at

    def _choose_default(self, entries: Mapping[str, Any] | None) -> None:
        if self._kept(self._default_task, entries):
            self._chosen = self._default_task
        elif not self._kept(self._chosen, entries):
            rank = {t: i for i, t in enumerate(self._order)}
            self._chosen = min(self._running, key=lambda t: (rank.get(t, len(rank)), self._running[t].started, t),
                               default=None)

    def _alert_unbilled(self, entries: Mapping[str, Any], now: float) -> None:
        if self._has_spend is None:
            return
        retired = {t for t in self._running if getattr(entries.get(t), "status", None) == "retired"}
        for task_id in list(self._retired_seen):
            if task_id not in retired:
                self._retired_seen.pop(task_id)
                self._alerted_at.pop(task_id, None)
        for task_id in retired:
            seen = self._retired_seen.setdefault(task_id, now)
            last = self._alerted_at.get(task_id)
            if now - seen < NO_SPEND_ALERT_AFTER_S or (last is not None and now - last < NO_SPEND_ALERT_EVERY_S):
                continue
            try:
                billed = bool(self._has_spend(entries[task_id]))
            except Exception as exc:
                logger.warning("rl supervisor: spend summary of %s unreadable (%r)", task_id, exc)
                continue
            if billed:
                self._retired_seen[task_id] = float("inf")             # it wrote one: no alert from here on
                continue
            self._alerted_at[task_id] = now
            logger.error("rl supervisor: %s retired %.0f h ago and its controller never wrote a spend summary; "
                         "its bill cannot turn final", task_id, (now - seen) / 3600)

    def _write_map(self) -> None:
        body = json.dumps({"default": self._chosen,
                           "tasks": {t: {"port": r.port, "run": t,
                                         **({"model": self._models[t]} if t in self._models else {})}
                                     for t, r in sorted(self._running.items())}},
                          sort_keys=True)
        if body == self._written:
            return
        _write_atomic(self._map_path, body)
        self._written = body


def _write_atomic(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def task_model(entry: Any) -> dict | None:
    """The model a task trains (its carried contract's, else its profile's): ``{"id", "revision"}``, or None."""
    from reliquary.protocol.profiles import profile_from_contract, resolve_protocol_profile

    if entry is None:
        return None
    try:
        contract = getattr(entry, "contract", None)
        if contract:
            profile = profile_from_contract(contract)
        elif getattr(entry, "profile_id", None):
            profile = resolve_protocol_profile(entry.profile_id)
        else:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return {"id": str(profile.model_id), "revision": str(profile.model_revision)}


CHILD_DROPPED_ENV = ("RELIQUARY_RL_TRAINER_TOKEN", "RELIQUARY_RL_TRAINER_BIND", "RELIQUARY_PROTOCOL_PROFILE",
                     "RELIQUARY_TASK_CONTRACT")


def child_profile(entry: Any, task_dir: Path) -> tuple[dict[str, str], list[str]]:
    """The environment and options that make a controller run ``entry``'s profile and model: its carried contract
    (written to ``task_dir``) or its profile id, and ``--checkpoint`` its model."""
    if entry is None:
        return {}, []
    env: dict[str, str] = {}
    contract = getattr(entry, "contract", None)
    if contract:
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / "contract.json"
        _write_atomic(path, json.dumps(contract, sort_keys=True))
        env["RELIQUARY_TASK_CONTRACT"] = str(path)
    elif getattr(entry, "profile_id", None):
        env["RELIQUARY_PROTOCOL_PROFILE"] = str(entry.profile_id)
    model = task_model(entry)
    return env, (["--checkpoint", model["id"]] if model is not None else [])


def subprocess_spawn(command: list[str], state_dir: Path, env: Mapping[str, str]) -> Callable[..., Any]:
    """Start ``command`` (``reliquary validate`` and the host's wallet options) as an async RL controller on
    loopback: no ``--external-ip`` (the front owns the axon), no weights set (the host's own validator does), its
    own state directory (``<state_dir>/<task>``), its task's profile and model, and no trainer listener."""
    def spawn(task_id: str, port: int, entry: Any = None) -> Any:
        task_dir = state_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        profile_env, options = child_profile(entry, task_dir)
        child = {k: v for k, v in env.items() if k not in CHILD_DROPPED_ENV}
        log = open(state_dir / f"{task_id}.log", "ab")
        try:
            return subprocess.Popen(
                [*command, *options, "--no-set-weights", "--http-host", "127.0.0.1", "--http-port", str(port)],
                env={**child, **profile_env, "RELIQUARY_RL_MODE": "async", "RELIQUARY_TASK_ID": task_id,
                     "RELIQUARY_RL_RUN": task_id, "RELIQUARY_STATE_DIR": str(task_dir)},
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


def has_spend(entry: Any) -> bool:
    """Whether the task's controller ever wrote a spend summary."""
    from reliquary.validator import rl_spend

    return rl_spend.read_spend(entry.task_id) is not None


__all__ = ["DEFAULT_TASK_ENV", "StateLocked", "Supervisor", "TICK_S", "admin_final", "child_profile", "has_spend",
           "lock_state_dir", "read_registry_entries", "subprocess_spawn"]
