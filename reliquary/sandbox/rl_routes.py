"""``POST {prefix}/episodes/precommit``.

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

from reliquary.protocol.signatures import verify_episode_precommit_signature, verify_rl_allocation_signature
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
    # No machine serves the env now / the fleet view is not ready: the 2C miner sends the precommit again until
    # the window ends. No machine serving the env can take this task: the miner moves to another task (it may be
    # servable in a later window). None of them records anything.
    "environment_unserved": 503, "fleet_unavailable": 503, "task_unservable": 409,
    # the allocation route: an unsigned request, its hotkey's minute, a signature already used, too many at once
    "signature_required": 401, "allocation_rate": 429, "request_reused": 409, "allocation_busy": 503,
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
    admit: Callable[[Any], Awaitable[tuple[str, dict] | None]] | None = None,
    precheck: Callable[[Any], tuple[bool, str]] | None = None,
) -> APIRouter:
    """``record``: ``ServiceRuntime.record_episode_precommit`` (blocking, run in a thread);
    ``current_window``: the service window admissions are open for, or None between windows. It is
    called on the event loop: it must read memory only (never the runtime's SQLite). ``admit(value)``: the RL
    wiring's check that the env is served and the task servable, before anything is recorded; a
    ``(reason, detail)`` answer refuses the precommit. ``precheck``: ``record``'s own checks with nothing written
    (blocking, run in a thread), asked before ``admit``: its final refusals are answered first, and the resend of
    a precommit already recorded is answered at once, without the admission."""
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

    def _refused_by_record(refused) -> JSONResponse:
        detail = {"precommit_sha256": refused.existing} if refused.existing else {}
        return _refusal(refused.reason, detail, retry_s)

    async def handle(value, hotkey: str) -> JSONResponse:
        from reliquary.services.runtime import EpisodePrecommitRefused

        window = current_window()
        if window is None:
            return _refusal("window_not_open", {}, retry_s)
        if value.window != window:
            return _refusal("precommit_stale", {"window": window})
        if admit is not None:
            if precheck is not None:
                try:
                    created, digest = await asyncio.to_thread(precheck, value)
                except EpisodePrecommitRefused as refused:
                    return _refused_by_record(refused)
                if not created:                       # already recorded: its resend is answered as before
                    logger.info("episode precommit %s of %s: %s task %d, window %d (resent)", digest[:12],
                                hotkey[:12], value.environment, value.task_index, value.window)
                    return _answer({"precommit_sha256": digest, "created": False})
            refused = await admit(value)
            if refused is not None:
                return _refusal(refused[0], refused[1], retry_s)
        try:
            created, digest = await asyncio.to_thread(record, value)
        except EpisodePrecommitRefused as refused:
            return _refused_by_record(refused)
        logger.info("episode precommit %s of %s: %s task %d, window %d%s", digest[:12], hotkey[:12],
                    value.environment, value.task_index, value.window, "" if created else " (resent)")
        return _answer({"precommit_sha256": digest, "created": created})

    return router


def episode_allocation_path(prefix: str = "/rl") -> str:
    return f"{prefix}/sessions/allocation"


class AllocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    at: int = Field(ge=0, strict=True)
    signature: str = Field(default="", max_length=256)


ALLOCATION_MAX_PER_MINUTE = 60
"""Allocation reads per hotkey per rolling minute (the reference miner reads it at every 2 s poll)."""


def build_allocation_router(allocation_of: Callable[[str], dict], *, validator_hotkey: str, policy: SandboxPolicy,
                            prefix: str = "/rl", verify: Callable[..., bool] = verify_rl_allocation_signature,
                            registration: Callable[[str], Awaitable[str | None]] | None = None,
                            clock: Callable[[], float] = time.time, body_timeout_s: float = BODY_TIMEOUT_S,
                            max_per_minute: int = ALLOCATION_MAX_PER_MINUTE, max_preauth: int = MAX_PREAUTH,
                            max_tracked_hotkeys: int = MAX_TRACKED_HOTKEYS) -> APIRouter:
    """``POST {prefix}/sessions/allocation``: the CALLER's allocation (how many groups the reference miner runs), for
    its own hotkey only. The body ``{"miner_hotkey", "at", "signature"}`` is signed by that hotkey for this
    validator and route (``build_rl_allocation_binding``), within ``policy.request_skew_s`` of now; a signature is
    answered once (a replay within the skew is ``request_reused``). An unsigned request is 401, a signature that
    does not verify (another key, another validator or route) 403, an unregistered hotkey 403. The checks run as on
    the precommit route: the body bounded in size and time, at most ``max_preauth`` decoded and verified at once
    (the signature in a worker thread), the hotkey's minute counted after verification. ``allocation_of(hotkey)``
    reads memory only (a cached snapshot)."""
    from reliquary.sandbox.routes import ss58_address as _ss58

    audience = _ss58(validator_hotkey)
    if audience is None:
        raise ValueError("validator_hotkey must be an ss58 account address")
    router = APIRouter()
    path = episode_allocation_path(prefix)
    retry_s = policy.retry_after_s
    skew = policy.request_skew_s
    preauth = asyncio.Semaphore(max_preauth)
    recent: OrderedDict[str, deque] = OrderedDict()            # hotkey -> times of its verified reads
    used: OrderedDict[str, float] = OrderedDict()               # signature -> when it expires (replay guard)

    def remember(hotkey: str, signature: str, now: float) -> JSONResponse | None:
        while used and next(iter(used.values())) < now:
            used.popitem(last=False)
        if signature in used:
            return _refusal("request_reused", {})
        times = recent.setdefault(hotkey, deque())
        recent.move_to_end(hotkey)
        while times and times[0] <= now - 60.0:
            times.popleft()
        if len(times) >= max_per_minute:
            return _refusal("allocation_rate", {"max_per_minute": max_per_minute}, times[0] + 60.0 - now)
        times.append(now)
        used[signature] = now + 2 * skew + 1
        while len(recent) > max_tracked_hotkeys:
            recent.popitem(last=False)
        return None

    @router.post(path)
    async def allocation(http: Request) -> JSONResponse:
        try:
            raw = await _read_within(http, 4096, body_timeout_s)
            if preauth.locked():
                return _refusal("allocation_busy", {"max_concurrent": max_preauth}, retry_s)
            async with preauth:
                request, hotkey = await asyncio.to_thread(_decode, raw, AllocationRequest)
                if not request.signature:
                    return _refusal("signature_required", {})
                if registration is not None:
                    try:
                        why = await registration(hotkey)
                    except Exception as exc:
                        logger.warning("allocation registration check failed (%s)", type(exc).__name__)
                        why = "unavailable"
                    if why == NOT_REGISTERED:
                        return _refuse("hotkey_not_registered", {"why": why})
                    if why is not None:
                        return _refuse("registration_unavailable", {"why": str(why)}, retry_s)
                now = float(clock())
                if abs(int(now) - request.at) > skew:
                    return _refuse("stale_request", {"max_skew_s": skew, "now": int(now)})
                try:
                    verified = bool(await asyncio.to_thread(verify, hotkey, at=request.at,
                                                            signature=request.signature,
                                                            validator_hotkey=audience, path=path))
                except Exception:
                    verified = False
                if not verified:
                    return _refuse("bad_signature")
                refused = remember(hotkey, request.signature, now)
                if refused is not None:
                    return refused
            return _answer(allocation_of(hotkey))
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            logger.error("allocation read failed (%s)", type(exc).__name__)
            return _refuse("internal_error")

    return router


__all__ = [
    "ALLOCATION_MAX_PER_MINUTE",
    "BODY_TIMEOUT_S",
    "MAX_PER_MINUTE",
    "MAX_PREAUTH",
    "MAX_TRACKED_HOTKEYS",
    "PRECOMMIT_STATUS",
    "AllocationRequest",
    "EpisodePrecommitRequest",
    "build_allocation_router",
    "build_episode_precommit_router",
    "episode_allocation_path",
    "episode_precommit_path",
]
