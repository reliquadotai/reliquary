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
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from reliquary.validator.rl_buffer import SubmittedGroup

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

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None, *, resend_ok: bool = False) -> Any:
        import httpx

        started, attempt = self._clock(), 0
        while True:
            try:
                response = self._client.request(method, f"{self._base}/{path}", headers=self._headers,
                                                **({"json": dict(body)} if body is not None else {}))
                status = response.status_code
                if status == 200:
                    return response.json()
                if status == 404 and resend_ok and attempt > 0:
                    return None                     # the lost first answer had done it
                why = f"{path}: {status} {response.text[:200]}"
                if status != 429 and status < 500:
                    raise TrainerLinkError(why, status)
            except httpx.HTTPError as exc:
                why = f"{path}: {type(exc).__name__}"
            delay = self._backoff[min(attempt, len(self._backoff) - 1)]
            if self._clock() - started + delay > self._patience:
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
    def report_training(self, lease_id: str, *, rollout_match, accepted: bool, failure_class: str | None = None,
                        matched_tokens=None, substitutions=None) -> None:
        self._call("POST", "report", {
            "lease_id": lease_id, "rollout_match": [float(m) for m in rollout_match], "accepted": bool(accepted),
            "failure_class": failure_class,
            "matched_tokens": None if matched_tokens is None else [int(n) for n in matched_tokens],
            "substitutions": None if substitutions is None else [[[int(p), int(t)] for p, t in s]
                                                                 for s in substitutions]}, resend_ok=True)


    # --- the audits (trainer.audits) ------------------------------------------------------------------------------
    def take_audits(self, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued out-of-zone audits (the validator drops the stale ones first)."""
        return self._groups(self._call("POST", "audits", {"limit": int(limit)}))

    def report_audit(self, lease_id: str, result) -> None:
        self._call("POST", "audit", {
            "lease_id": lease_id, "passed": result.passed, "failure_class": result.failure_class,
            "rollout_match": [float(m) for m in result.rollout_match], "suspect": bool(result.suspect)},
            resend_ok=True)


__all__ = ["TrainerLink", "TrainerLinkError"]
