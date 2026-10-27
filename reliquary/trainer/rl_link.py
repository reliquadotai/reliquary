"""The detached async trainer's link to the validator (``validator/rl_async_service.build_rl_trainer_router``).

The buffer, the submissions, the lease book and the prices live in the validator's process (one SQLite file on the
controller); the trainer runs on its own GPU box, as the window trainer does today. ``TrainerLink`` is the one
object ``AsyncTrainer`` takes as ``buffer``, ``submissions``, ``book`` and ``prices``: each call is one HTTPS
request with the bearer token.

Every call retries transport errors, 429 and 5xx with backoff for about ``patience_s`` (a validator restart must not
cost the trainer its optimizer state), then raises ``TrainerLinkError``. A report whose answer was lost is resent;
the validator then no longer holds the group (404), which counts as done.
"""
from __future__ import annotations

import base64
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_trainer_owner import (
    EPOCH_HEADER,
    HEARTBEAT_SECONDS,
    OWNER_HEADER,
)

logger = logging.getLogger(__name__)

BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
PATIENCE_S = 600.0


class TrainerLinkError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class TrainerLink:
    def __init__(self, base_url: str, token: str, *, client=None, prefix: str = "/rl/trainer",
                 patience_s: float = PATIENCE_S, backoff_s: Sequence[float] = BACKOFF_S,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        import httpx

        self._base = base_url.rstrip("/") + prefix
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0))
        self._headers = {"Authorization": f"Bearer {token}"}
        self._patience, self._backoff, self._sleep, self._clock = patience_s, tuple(backoff_s), sleep, clock
        self._owner = uuid.uuid4().hex
        self._epoch: int | None = None
        self._owner_deadline = 0.0
        self._owner_stopped = threading.Event()
        self._owner_thread: threading.Thread | None = None
        self._owner_lock = threading.Lock()

    def close(self) -> None:
        self._owner_stopped.set()
        if self._owner_thread is not None:
            self._owner_thread.join(timeout=125)
        try:
            if self._epoch is not None and self._clock() < self._owner_deadline:
                self._call("POST", "owner/release", {})
        finally:
            self._client.close()

    def _accept_owner(self, answer, started: float) -> None:
        seconds = answer.get("lease_seconds")
        if (answer.get("owner") != self._owner or type(answer.get("epoch")) is not int
                or answer["epoch"] < 1 or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or not 0 < seconds <= 180):
            raise TrainerLinkError("invalid trainer ownership response")
        self._epoch = answer["epoch"]
        self._headers.update({OWNER_HEADER: self._owner, EPOCH_HEADER: str(self._epoch)})
        self._owner_deadline = started + seconds

    def claim_ownership(self, *, start_heartbeat: bool = False) -> None:
        with self._owner_lock:
            if self._epoch is None:
                started = self._clock()
                self._accept_owner(self._call("POST", "owner/claim", {"owner": self._owner}), started)
            self.assert_ownership()
            if start_heartbeat and self._owner_thread is None:
                self._owner_thread = threading.Thread(target=self._renew_owner, daemon=True,
                                                      name="trainer-owner-heartbeat")
                self._owner_thread.start()

    def assert_ownership(self) -> None:
        if self._epoch is None or self._owner_stopped.is_set() or self._clock() >= self._owner_deadline:
            raise TrainerLinkError("trainer ownership lost", 409)

    @property
    def writer_id(self) -> str:
        self.assert_ownership()
        return f"{self._epoch}-{self._owner}"

    def _renew_owner(self) -> None:
        while not self._owner_stopped.wait(HEARTBEAT_SECONDS):
            try:
                self.assert_ownership()
                started = self._clock()
                self._accept_owner(self._call("POST", "owner/heartbeat", {}), started)
            except Exception:
                self._owner_stopped.set()
                logger.exception("trainer ownership renewal failed")
                return

    def publish_index(self, index: dict) -> None:
        self._call("POST", "publish_index", {"index": index})

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None, *, resend_ok: bool = False,
              unavailable_ok: bool = False, patience_s: float | None = None) -> Any:
        import httpx

        patience = self._patience if patience_s is None else patience_s
        owned = method == "POST" and not path.startswith("owner/")
        if owned and self._epoch is None:
            self.claim_ownership()
        started, attempt = self._clock(), 0
        while True:
            if owned or path == "owner/heartbeat":
                self.assert_ownership()
            try:
                response = self._client.request(method, f"{self._base}/{path}", headers=dict(self._headers),
                                                **({"json": dict(body)} if body is not None else {}))
                status = response.status_code
                if status == 200:
                    return response.json()
                if status == 503 and unavailable_ok:
                    return None
                if status == 404 and resend_ok and attempt > 0:
                    return None                     # the lost first answer had done it
                why = f"{path}: {status} {response.text[:200]}"
                if status != 429 and status < 500:
                    raise TrainerLinkError(why, status)
            except httpx.HTTPError as exc:
                why = f"{path}: {type(exc).__name__}"
            delay = self._backoff[min(attempt, len(self._backoff) - 1)]
            if (self._clock() - started + delay > patience
                    or (self._epoch is not None and self._clock() + delay >= self._owner_deadline)):
                raise TrainerLinkError(f"validator unreachable: {why}")
            logger.warning("trainer link %s; retrying in %.0fs", why, delay)
            self._sleep(delay)
            attempt += 1

    @staticmethod
    def _groups(answer: Mapping[str, Any]) -> list[SubmittedGroup]:
        return [SubmittedGroup.unpack(base64.b64decode(blob)) for blob in answer["groups"]]

    # --- the validator's task -----------------------------------------------------------------------------------
    def config(self) -> dict:
        """``task_id``, ``run`` (the checkpoint index's), ``targets`` (the env mix) and ``latest``."""
        return self._call("GET", "config")

    def checkpoint(self) -> dict:
        """The validator's signed checkpoint answer (``/rl/checkpoint``'s body, served on the link: the trainer
        reaches the trainer listener only)."""
        return self._call("GET", "checkpoint")

    def checkpoint_if_available(self) -> dict | None:
        return self._call("GET", "checkpoint", unavailable_ok=True)

    # --- the lease book -------------------------------------------------------------------------------------------
    @property
    def latest_checkpoint(self) -> int | None:
        return self._call("GET", "latest")["latest"]

    def adopt_checkpoint(self, k: int) -> None:
        self._call("POST", "adopt", {"k": int(k)})

    # --- the buffer -----------------------------------------------------------------------------------------------
    def take(self, limit: int) -> list[SubmittedGroup]:
        return self._groups(self._call("POST", "take", {"limit": int(limit)}))

    def take_older(self, below_k: int, limit: int) -> list[SubmittedGroup]:
        return self._groups(self._call("POST", "take_older", {"below_k": int(below_k), "limit": int(limit)}))

    def take_exhausted(self, max_attempts: int, limit: int) -> list[SubmittedGroup]:
        return self._groups(self._call("POST", "take_exhausted", {"max_attempts": int(max_attempts),
                                                                  "limit": int(limit)}))

    def release(self, lease_ids) -> None:
        self._call("POST", "release", {"lease_ids": [str(i) for i in lease_ids]})

    def requeue(self) -> int:
        """Every taken group back to the queue: the trainer holds none (start, or a take whose answer was lost)."""
        return int(self._call("POST", "requeue", {})["requeued"])

    def depth(self) -> dict[str, int]:
        return {str(env): int(n) for env, n in self._call("GET", "depth")["depth"].items()}

    # --- the submissions ------------------------------------------------------------------------------------------
    def admit_training(self, lease_id: str, substitutions) -> dict:
        return self._call("POST", "admit", {"lease_id": lease_id, "substitutions": substitutions})

    def report_training(self, lease_id: str, *, rollout_match, accepted: bool, failure_class: str | None = None,
                        matched_tokens=None, substitutions=None, judged: bool | None = None,
                        admission_id: str | None = None) -> None:
        body = {
            "lease_id": lease_id, "rollout_match": [float(m) for m in rollout_match], "accepted": bool(accepted),
            "failure_class": failure_class,
            "admission_id": admission_id,
            "matched_tokens": None if matched_tokens is None else [int(n) for n in matched_tokens],
            "substitutions": None if substitutions is None else [[[int(p), int(t)] for p, t in s]
                                                                 for s in substitutions]}
        if judged is not None:
            body["judged"] = judged
        self._call("POST", "report", body, resend_ok=True)


    def heartbeat(self, patience_s: float = 30.0) -> None:
        """Tell the validator this trainer is alive (it pays miners for the time the trainer is down)."""
        self._call("POST", "heartbeat", {}, patience_s=patience_s)

    # --- the audits (trainer.audits) ------------------------------------------------------------------------------
    def take_audits(self, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued out-of-zone audits (the validator drops the stale ones first)."""
        return self._groups(self._call("POST", "audits", {"limit": int(limit)}))

    def start_audit(self, lease_id: str, attempt_id: str) -> bool:
        return bool(self._call("POST", "audit_start", {"lease_id": lease_id,
                                                       "attempt_id": attempt_id})["started"])

    def report_audit_failure(self, lease_id: str, attempt_id: str) -> bool:
        answer = self._call("POST", "audit_failure", {"lease_id": lease_id, "attempt_id": attempt_id},
                            resend_ok=True)
        return answer is None or bool(answer["concluded"])

    def report_audit(self, lease_id: str, result) -> None:
        self._call("POST", "audit", {
            "lease_id": lease_id, "passed": result.passed, "failure_class": result.failure_class,
            "rollout_match": [float(m) for m in result.rollout_match], "suspect": bool(result.suspect)},
            resend_ok=True)


__all__ = ["TrainerLink", "TrainerLinkError"]


class TrainerHeartbeat:
    """A daemon thread calling ``link.heartbeat()`` every ``every_s`` seconds, steps included. It never turns itself
    off: a failure (a validator restarting, one without the route) is logged and retried at the next beat."""

    def __init__(self, link, every_s: float = 60.0) -> None:
        self.link, self.every_s = link, float(every_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trainer-heartbeat", daemon=True)

    def start(self) -> "TrainerHeartbeat":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.link.heartbeat()
            except TrainerLinkError as exc:
                logger.warning("trainer heartbeat failed: %s", exc)
            except Exception:
                logger.exception("trainer heartbeat failed")
            self._stop.wait(self.every_s)
