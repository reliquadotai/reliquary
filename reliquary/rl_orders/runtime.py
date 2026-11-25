"""Private, task-bound readiness and drainage for one supervised local training runtime."""
from __future__ import annotations

import os
import re
import signal
import threading
import time
from collections.abc import Mapping


def requires_owned_training_runtime(order: object) -> bool:
    """Quote-bound declarations belong exclusively to the operator-owned supervisor."""
    return isinstance(order, Mapping) and any(key in order for key in ("quote_id", "quote_sha256", "billing_basis"))


class TrainerRuntime:
    def __init__(self, entry, rl, *, runtime_id: str, clock=time.time, monotonic=time.monotonic):
        from reliquary.validator.rl_training_usage import bind_runtime

        order = entry.params.get("training_order") or {}
        self.identity = {"task_id": entry.task_id, "order_sha256": order.get("order_sha256"),
                         "run_id": rl.settings.run, "runtime_id": runtime_id}
        if not re.fullmatch(r"[0-9a-f]{64}", str(self.identity["order_sha256"])):
            raise ValueError("a supervised runtime needs its immutable training order digest")
        if not re.fullmatch(r"[0-9a-f]{32,64}", runtime_id):
            raise ValueError("invalid supervised runtime identity")
        self.rl, self.clock = rl, clock
        self.monotonic, self._drain_started = monotonic, None
        self.limits = order
        bind_runtime(rl.book, run_id=rl.settings.run, runtime_id=runtime_id)
        with rl.book.transaction():
            for key, value in (("training_task_id", entry.task_id),
                               ("training_order_sha256", order["order_sha256"])):
                old = self._meta(key)
                if old is not None and old != value:
                    raise ValueError("runtime database belongs to another task or order")
                rl.book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) VALUES(?,?)", (key, value))

    def _meta(self, key):
        row = self.rl.book.db.execute("SELECT value FROM rl_lease_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def _matches(self, body: dict, *, checkpoint=False):
        keys = set(self.identity) | ({"latest_checkpoint"} if checkpoint else set())
        if set(body) != keys or any(body.get(key) != value for key, value in self.identity.items()):
            raise ValueError("runtime identity does not match this task")

    def drain_requested(self):
        with self.rl.book.lock:
            return any(self._meta(key) is not None for key in ("training_registry_stop", "training_limit_stop"))

    def sealed(self):
        with self.rl.book.lock:
            return self._meta("training_trainer_quiesced") == self.identity["runtime_id"]

    def admission_reason(self):
        with self.rl.book.lock:
            if self.sealed():
                return "runtime_quiesced"
            if self.drain_requested():
                return "runtime_drain"
            if self._meta("training_trainer_ready") != self.identity["runtime_id"]:
                return "trainer_starting"
        return None

    def snapshot(self):
        from reliquary.validator.rl_training_usage import inspect_drain

        self.check_limits()
        with self.rl.book.lock:
            state = inspect_drain(self.rl.book.db)
            ready = self._meta("training_trainer_ready") == self.identity["runtime_id"]
            quiesced = self.sealed()
            draining = self.drain_requested()
            return {"schema": "reliquary/training-runtime/v1", **self.identity,
                    "controller_ready": True, "trainer_ready": ready,
                    "trainer_quiesced": quiesced, "drain_requested": draining,
                    "admission_closed": draining or quiesced or not ready,
                    "drained": bool(state["drained"]) and quiesced,
                    "latest_checkpoint": self.rl.book.latest_checkpoint,
                    "observed_at": self.clock()}

    def check_limits(self):
        """Enforce finite admission even if the supervisor stops polling or no miner asks again."""
        with self.rl.book.transaction():
            reason = None
            if self.clock() >= self.limits["deadline_at"]:
                reason = "deadline"
            elif (self.rl.book.latest_checkpoint is not None
                  and self.rl.book.latest_checkpoint >= self.limits["max_steps"]):
                reason = "max_steps"
            elif (self.limits.get("max_tokens") is not None
                  and self.rl.book.auctions.paid_tokens() >= self.limits["max_tokens"]):
                reason = "max_tokens"
            if reason is not None:
                self.rl.book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) "
                                        "VALUES('training_limit_stop',?)", (reason,))
        if self.drain_requested():
            if self._drain_started is None:
                self._drain_started = self.monotonic()
            if self.sealed():
                from reliquary.validator.rl_training_usage import settle_quiesced_work

                if self.monotonic() - self._drain_started >= 60:
                    self.rl.book.release_for_order_drain(runtime_id=self.identity["runtime_id"])
                settle_quiesced_work(self.rl)

    def ready(self, body):
        self._matches(body)
        with self.rl.book.transaction():
            if self.drain_requested() or self.sealed():
                raise ValueError("the runtime is already draining")
            if self.rl.book.latest_checkpoint is None:
                raise ValueError("the trainer has not published its initial checkpoint")
            self.rl.book.db.execute("INSERT OR IGNORE INTO rl_lease_meta(key,value) "
                                    "VALUES('training_trainer_ready',?)", (self.identity["runtime_id"],))
        return self.snapshot()

    def drain(self, body):
        from reliquary.validator.rl_training_usage import request_drain

        self._matches(body)
        request_drain(self.rl.book, runtime_id=self.identity["runtime_id"])
        return self.snapshot()

    def quiesce(self, body):
        from reliquary.validator.rl_training_usage import quiesce_trainer

        self._matches(body, checkpoint=True)
        latest = body["latest_checkpoint"]
        if isinstance(latest, bool) or not isinstance(latest, int) or latest < 0:
            raise ValueError("latest_checkpoint must be a published integer checkpoint")
        quiesce_trainer(self.rl, runtime_id=self.identity["runtime_id"], latest_checkpoint=latest)
        return self.snapshot()


class ManagedControllerShutdown:
    """Accept Uvicorn's replayed SIGTERM only after durable drainage and completed local cleanup."""

    def __init__(self, runtime):
        self.runtime, self.identity = runtime, dict(runtime.identity)
        self._drained = self._clean = False
        self._previous = None

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread() or self._previous is not None:
            raise RuntimeError("managed controller shutdown must own the main-thread signal scope")
        previous = signal.getsignal(signal.SIGTERM)
        if previous is None:
            raise RuntimeError("the previous SIGTERM handler is unknown")
        self._drained = self._clean = False
        self._previous = previous
        signal.signal(signal.SIGTERM, self._replayed_sigterm)
        return self

    def capture_drain(self) -> bool:
        self._drained = False
        try:
            snapshot = self.runtime.snapshot()
            self._drained = (isinstance(snapshot, Mapping)
                             and all(snapshot.get(key) == value for key, value in self.identity.items())
                             and snapshot.get("schema") == "reliquary/training-runtime/v1"
                             and all(snapshot.get(key) is True for key in (
                                 "controller_ready", "admission_closed", "drain_requested", "trainer_quiesced", "drained"))
                             and type(snapshot.get("latest_checkpoint")) is int
                             and snapshot["latest_checkpoint"] >= 0)
        except Exception:
            pass                               # cleanup still runs; an unproven stop cannot exit successfully
        return self._drained

    def cleanup_completed(self) -> None:
        self._clean = True

    def _replayed_sigterm(self, signum, frame) -> None:
        if not (self._drained and self._clean):
            raise SystemExit(128 + signum)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None and not (self._drained and self._clean):
                raise RuntimeError("managed controller exited without verified drainage and complete cleanup")
        finally:
            signal.signal(signal.SIGTERM, self._previous)
            self._previous = None
        return False


def supervised_runtime(entry, rl, *, environ: Mapping[str, str] | None = None):
    env = os.environ if environ is None else environ
    runtime_id = env.get("RELIQUARY_RL_RUNTIME_ID", "")
    order_sha = env.get("RELIQUARY_RL_ORDER_SHA256", "")
    if not runtime_id and not order_sha:
        return None
    if (not runtime_id or not order_sha or env.get("RELIQUARY_TASK_ID") != entry.task_id
            or env.get("RELIQUARY_RL_RUN") != rl.settings.run
            or order_sha != (entry.params.get("training_order") or {}).get("order_sha256")):
        raise ValueError("incomplete or mismatched supervised runtime binding")
    if rl.settings.pay_outage_fill:
        raise ValueError("supervised usage cannot include synthetic trainer-outage payments")
    return TrainerRuntime(entry, rl, runtime_id=runtime_id)
