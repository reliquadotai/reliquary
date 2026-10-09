"""``POST {prefix}/episodes/precommit`` (phase 2, plan 2C; spec §4.1.2).

A miner signs ``{order, window, environment, task_index, checkpoint, pool_sha256, hotkey}`` for this
validator and this route (the sandbox requests' audience rule) and posts it before opening any sandbox
session for that task; the runtime records it against the frozen window
(``ServiceRuntime.record_episode_precommit``, in a thread). Its digest is what every session of the
group names (``rl:{window}:{precommit}:{seed}``).

The body is read bounded (in size, and in time: ``body_timeout_s``) and parsed off the loop with the
sandbox routes' own helpers; every refusal is ``{"reason", "detail"}``. A window that is not open (no
announced pool, between windows) is a retryable 503; anything else the miner must change is a 4xx.

The body is read first (bounded in size and time, holding nothing). Then, at most ``max_preauth``
requests at once (one more is refused ``precommit_busy``, 503): the body is decoded, the hotkey's
registration checked (in memory: an unregistered hotkey costs no signature check and is never tracked),
the hotkey's rolling minute checked (``precommit_rate``, 429, before the verification), and the
signature (sr25519) verified. Only a verified request counts against its hotkey's minute (unsigned
requests naming a hotkey cannot spend its budget); at most ``max_tracked_hotkeys`` hotkeys are tracked,
the least recently seen forgotten first."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from reliquary.protocol.signatures import verify_episode_precommit_signature
from reliquary.sandbox.routes import (
    MAX_OPEN_BODY_BYTES, _answer, _decode, _read_within, _refuse, _Refused, ss58_address,
)
from reliquary.sandbox.sessions import SandboxPolicy
from reliquary.validator.corpus_registration import NOT_REGISTERED

logger = logging.getLogger(__name__)

PRECOMMIT_STATUS: dict[str, int] = {
    "precommit_invalid": 422, "hotkey_mismatch": 403, "precommit_stale": 409, "window_not_open": 503,
    "window_closed": 409, "order_mismatch": 409, "environment_not_episode": 409,
    "environment_not_active": 409, "task_out_of_range": 409, "checkpoint_mismatch": 409,
    "pool_mismatch": 409, "precommit_exists": 409, "task_in_cooldown": 409,
    "precommit_rate": 429, "precommit_busy": 503,
}
BODY_TIMEOUT_S = 10.0
MAX_PER_MINUTE = 30
MAX_PREAUTH = 8
MAX_TRACKED_HOTKEYS = 4096


def episode_precommit_path(prefix: str = "/rl") -> str:
    return f"{prefix}/episodes/precommit"


class EpisodePrecommitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    at: int = Field(ge=0, strict=True)
    precommit: dict[str, Any]
    signature: str = Field(min_length=1, max_length=256)


def _refusal(reason: str, detail: dict | None = None, retry_after: float | None = None) -> JSONResponse:
    status = PRECOMMIT_STATUS.get(reason)
    if status is None:
        return _refuse(reason, detail, retry_after)
    headers = {"Cache-Control": "no-store"}
    if status in (429, 503):
        headers["Retry-After"] = str(max(1, math.ceil(retry_after or 1)))
    return JSONResponse(status_code=status, content={"reason": reason, "detail": dict(detail or {})},
                        headers=headers)


def build_episode_precommit_router(
    *, record: Callable[[Any], tuple[bool, str]], current_window: Callable[[], int | None],
    validator_hotkey: str, policy: SandboxPolicy, prefix: str = "/rl",
    verify: Callable[..., bool] = verify_episode_precommit_signature,
    registration: Callable[[str], Awaitable[str | None]] | None = None,
    clock: Callable[[], float] = time.time,
    body_timeout_s: float = BODY_TIMEOUT_S, max_per_minute: int = MAX_PER_MINUTE,
    max_preauth: int = MAX_PREAUTH, max_tracked_hotkeys: int = MAX_TRACKED_HOTKEYS,
) -> APIRouter:
    """``record``: ``ServiceRuntime.record_episode_precommit`` (blocking, run in a thread);
    ``current_window``: the service window admissions are open for, or None between windows. It is
    called on the event loop: it must read memory only (never the runtime's SQLite)."""
    audience = ss58_address(validator_hotkey)
    if audience is None:
        raise ValueError("validator_hotkey must be an ss58 account address")
    if not (body_timeout_s > 0 and max_per_minute > 0 and max_preauth > 0 and max_tracked_hotkeys > 0):
        raise ValueError("the precommit route's bounds must be positive")
    router = APIRouter()
    path = episode_precommit_path(prefix)
    retry_s = policy.retry_after_s
    preauth = asyncio.Semaphore(max_preauth)
    recent: OrderedDict[str, deque] = OrderedDict()

    def over_rate(hotkey: str, now: float) -> JSONResponse | None:
        times = recent.get(hotkey)
        if times is None:
            return None
        while times and times[0] <= now - 60.0:
            times.popleft()
        if len(times) >= max_per_minute:
            return _refusal("precommit_rate", {"max_per_minute": max_per_minute}, times[0] + 60.0 - now)
        return None

    def count(hotkey: str, now: float) -> None:
        """A verified request of a registered hotkey (never anything else)."""
        recent.setdefault(hotkey, deque()).append(now)
        recent.move_to_end(hotkey)
        while len(recent) > max_tracked_hotkeys:
            recent.popitem(last=False)

    @router.post(path)
    async def precommit(http: Request) -> JSONResponse:
        try:
            raw = await _read_within(http, MAX_OPEN_BODY_BYTES, body_timeout_s)
            if preauth.locked():
                return _refusal("precommit_busy", {"max_concurrent": max_preauth}, retry_s)
            async with preauth:
                value, hotkey = await authenticate(raw)
            return await handle(value, hotkey)
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            logger.error("episode precommit failed (%s)", type(exc).__name__)
            return _refuse("internal_error")

    async def authenticate(raw: bytes):
        """Decode, registration (in memory), the hotkey's minute, then the signature: (precommit, hotkey)."""
        from reliquary.protocol.service_episode import EpisodePrecommit, EpisodeWireError

        request, hotkey = await asyncio.to_thread(_decode, raw, EpisodePrecommitRequest)
        if registration is not None:
            try:
                why = await registration(hotkey)
            except Exception as exc:
                logger.warning("episode precommit registration check failed (%s)", type(exc).__name__)
                why = "unavailable"
            if why == NOT_REGISTERED:
                raise _Refused(_refuse("hotkey_not_registered", {"why": why}))
            if why is not None:
                raise _Refused(_refuse("registration_unavailable", {"why": str(why)}, retry_s))
        now = float(clock())
        limited = over_rate(hotkey, now)
        if limited is not None:
            raise _Refused(limited)
        if abs(int(now) - request.at) > policy.request_skew_s:
            raise _Refused(_refuse("stale_request", {"max_skew_s": policy.request_skew_s, "now": int(now)}))
        try:
            value = EpisodePrecommit.from_dict(request.precommit)
        except (EpisodeWireError, TypeError):
            raise _Refused(_refusal("precommit_invalid")) from None
        if ss58_address(value.hotkey) != hotkey:
            raise _Refused(_refusal("hotkey_mismatch"))
        try:
            verified = bool(await asyncio.to_thread(verify, hotkey, request.precommit, at=request.at,
                                                    signature=request.signature,
                                                    validator_hotkey=audience, path=path))
        except Exception:
            verified = False
        if not verified:
            raise _Refused(_refuse("bad_signature"))
        count(hotkey, now)
        return value, hotkey

    async def handle(value, hotkey: str) -> JSONResponse:
        from reliquary.services.runtime import EpisodePrecommitRefused

        window = current_window()
        if window is None:
            return _refusal("window_not_open", {}, retry_s)
        if value.window != window:
            return _refusal("precommit_stale", {"window": window})
        try:
            created, digest = await asyncio.to_thread(record, value)
        except EpisodePrecommitRefused as refused:
            detail = {"precommit_sha256": refused.existing} if refused.existing else {}
            return _refusal(refused.reason, detail, retry_s)
        logger.info("episode precommit %s of %s: %s task %d, window %d%s", digest[:12], hotkey[:12],
                    value.environment, value.task_index, value.window, "" if created else " (resent)")
        return _answer({"precommit_sha256": digest, "created": created})

    return router


__all__ = ["BODY_TIMEOUT_S", "MAX_PER_MINUTE", "MAX_PREAUTH", "MAX_TRACKED_HOTKEYS", "EpisodePrecommitRequest",
           "PRECOMMIT_STATUS", "build_episode_precommit_router", "episode_precommit_path"]
