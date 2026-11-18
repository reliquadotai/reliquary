"""The detached async trainer's link to the validator (``validator/rl_async_service.build_rl_trainer_router``).

The buffer, the submissions, the lease book and the prices live in the validator's process (one SQLite file on the
controller); the trainer runs on its own GPU box, as the window trainer does today. ``TrainerLink`` is the one
object ``AsyncTrainer`` takes as ``buffer``, ``submissions``, ``book`` and ``prices``: each call is one HTTPS
request with the bearer token (plain HTTP only on loopback or with ``RELIQUARY_RL_TRAINER_INSECURE=1``).

Every call retries transport errors, 429 and 5xx with backoff for about ``patience_s`` (a validator restart must not
cost the trainer its optimizer state), then raises ``TrainerLinkError``. A report whose answer was lost is resent;
the validator then no longer holds the group (404), which counts as done.
"""
from __future__ import annotations

import base64
import logging
import math
import re
import threading
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


INSECURE_ENV = "RELIQUARY_RL_TRAINER_INSECURE"


def check_link_url(base_url: str, environ: Mapping[str, str]) -> None:
    """ValueError unless ``base_url`` is HTTPS, plain HTTP on loopback, or plain HTTP with ``INSECURE_ENV=1``
    (a private network only)."""
    import ipaddress
    from urllib.parse import urlsplit

    parts = urlsplit(base_url)
    if parts.scheme == "https":
        return
    if parts.scheme != "http":
        raise ValueError(f"the trainer link URL must be https, got {base_url!r}")
    host = parts.hostname or ""
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and str(environ.get(INSECURE_ENV, "")).strip() != "1":
        raise ValueError(f"the trainer link URL {base_url!r} is plain HTTP off loopback: use https, or set "
                         f"{INSECURE_ENV}=1 on a private network")


class TrainerLink:
    def __init__(self, base_url: str, token: str, *, client=None, prefix: str = "/rl/trainer",
                 patience_s: float = PATIENCE_S, backoff_s: Sequence[float] = BACKOFF_S,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 timeout_s: float = 120.0, environ: Mapping[str, str] | None = None) -> None:
        import os

        import httpx

        check_link_url(base_url, os.environ if environ is None else environ)
        self._base = base_url.rstrip("/") + prefix
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=min(15.0, timeout_s)))
        self._headers = {"Authorization": f"Bearer {token}"}
        self._patience, self._backoff, self._sleep, self._clock = patience_s, tuple(backoff_s), sleep, clock

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None, *, resend_ok: bool = False,
              patience_s: float | None = None) -> Any:
        import httpx

        patience = self._patience if patience_s is None else patience_s
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
            if self._clock() - started + delay > patience:
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

    def runtime(self) -> dict:
        return self._call("GET", "runtime", patience_s=10.0)

    def runtime_ready(self, identity: Mapping[str, str]) -> dict:
        return self._call("POST", "runtime/ready", identity, patience_s=10.0)

    def runtime_quiesced(self, identity: Mapping[str, str], latest_checkpoint: int) -> dict:
        return self._call("POST", "runtime/quiesced", {**identity, "latest_checkpoint": latest_checkpoint},
                          patience_s=10.0)

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
                        matched_tokens=None, substitutions=None, judged: bool | None = None) -> None:
        body = {
            "lease_id": lease_id, "rollout_match": [float(m) for m in rollout_match], "accepted": bool(accepted),
            "failure_class": failure_class,
            "matched_tokens": None if matched_tokens is None else [int(n) for n in matched_tokens],
            "substitutions": None if substitutions is None else [[[int(p), int(t)] for p, t in s]
                                                                 for s in substitutions]}
        if judged is not None:
            body["judged"] = judged
        self._call("POST", "report", body, resend_ok=True)


    def heartbeat(self, patience_s: float = 30.0) -> None:
        """Tell the validator this trainer is alive (it pays miners for the time the trainer is down)."""
        self._call("POST", "heartbeat", {}, patience_s=patience_s)

    # --- the checkpoint bucket, through the validator's presigned URLs ---------------------------------------------
    def upload_urls(self, keys: Sequence[str]) -> dict[str, str]:
        return dict(self._call("POST", "uploads", {"keys": list(keys)})["urls"])

    def read_urls(self, keys: Sequence[str]) -> dict[str, str]:
        return dict(self._call("POST", "reads", {"keys": list(keys)})["urls"])

    def delete_keys(self, keys: Sequence[str]) -> None:
        self._call("POST", "deletes", {"keys": list(keys)}, resend_ok=True)

    # --- the audits (trainer.audits) ------------------------------------------------------------------------------
    def take_audits(self, limit: int) -> list[SubmittedGroup]:
        """Up to ``limit`` queued out-of-zone audits (the validator drops the stale ones first)."""
        return self._groups(self._call("POST", "audits", {"limit": int(limit)}))

    def report_audit(self, lease_id: str, result) -> None:
        self._call("POST", "audit", {
            "lease_id": lease_id, "passed": result.passed, "failure_class": result.failure_class,
            "rollout_match": [float(m) for m in result.rollout_match], "suspect": bool(result.suspect)},
            resend_ok=True)


class TrainerRuntime:
    """The managed worker trusts only fresh answers for its exact supervisor-owned runtime."""

    def __init__(self, link, identity: Mapping[str, str], *, clock=time.time):
        if set(identity) != {"task_id", "order_sha256", "run_id", "runtime_id"}:
            raise ValueError("managed training needs an exact runtime identity")
        if not isinstance(identity["order_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", identity["order_sha256"]):
            raise ValueError("managed training needs the pinned order digest")
        if not all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", v)
                   for k, v in identity.items() if k != "order_sha256"):
            raise ValueError("invalid managed runtime identity")
        self.link, self.identity, self.clock = link, dict(identity), clock

    def _accept(self, reply: Any) -> dict:
        if not isinstance(reply, dict) or reply.get("schema") != "reliquary/training-runtime/v1":
            raise TrainerLinkError("invalid managed runtime snapshot")
        if any(reply.get(k) != v for k, v in self.identity.items()):
            raise TrainerLinkError("managed runtime identity changed")
        if not all(isinstance(reply.get(k), bool) for k in (
                "controller_ready", "trainer_ready", "admission_closed", "drain_requested", "trainer_quiesced", "drained")):
            raise TrainerLinkError("invalid managed runtime state")
        if ((reply["drain_requested"] and not reply["admission_closed"])
                or (reply["trainer_quiesced"] and not reply["drain_requested"])
                or (reply["drained"] and not reply["trainer_quiesced"])):
            raise TrainerLinkError("inconsistent managed drain state")
        observed = reply.get("observed_at")
        if (isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed)
                or not -5 <= self.clock() - observed <= 10):
            raise TrainerLinkError("stale managed runtime snapshot")
        latest = reply.get("latest_checkpoint")
        if latest is not None and (isinstance(latest, bool) or not isinstance(latest, int) or latest < 0):
            raise TrainerLinkError("invalid managed checkpoint")
        return reply

    def snapshot(self) -> dict:
        return self._accept(self.link.runtime())

    def drain_requested(self) -> bool:
        return self.snapshot()["drain_requested"]

    def ready(self, latest_checkpoint: int) -> None:
        if isinstance(latest_checkpoint, bool) or not isinstance(latest_checkpoint, int) or latest_checkpoint < 0:
            raise TrainerLinkError("invalid initialized checkpoint")
        reply = self._accept(self.link.runtime_ready(self.identity))
        if (not reply["controller_ready"] or not reply["trainer_ready"] or reply["trainer_quiesced"]
                or reply["latest_checkpoint"] != latest_checkpoint):
            raise TrainerLinkError("controller did not acknowledge the initialized trainer")

    def quiesced(self, latest_checkpoint: int) -> None:
        if isinstance(latest_checkpoint, bool) or not isinstance(latest_checkpoint, int) or latest_checkpoint < 0:
            raise TrainerLinkError("invalid quiesced checkpoint")
        reply = self._accept(self.link.runtime_quiesced(self.identity, latest_checkpoint))
        if (not reply["drain_requested"] or not reply["admission_closed"] or not reply["trainer_quiesced"]
                or reply["latest_checkpoint"] != latest_checkpoint):
            raise TrainerLinkError("controller did not acknowledge trainer quiescence")


__all__ = ["TrainerLink", "TrainerLinkError", "TrainerRuntime"]


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

    def stop(self, timeout_s: float = 35.0) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TrainerLinkError("trainer heartbeat has not stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.link.heartbeat()
            except TrainerLinkError as exc:
                logger.warning("trainer heartbeat failed: %s", exc)
            except Exception:
                logger.exception("trainer heartbeat failed")
            self._stop.wait(self.every_s)
