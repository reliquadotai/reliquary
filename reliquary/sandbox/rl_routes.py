"""``POST {prefix}/episodes/precommit`` (phase 2, plan 2C; spec §4.1.2).

A miner signs ``{order, window, environment, task_index, checkpoint, pool_sha256, hotkey}`` for this
validator and this route (the sandbox requests' audience rule) and posts it before opening any sandbox
session for that task; the runtime records it against the frozen window
(``ServiceRuntime.record_episode_precommit``, in a thread). Its digest is what every session of the
group names (``rl:{window}:{precommit}:{seed}``).

The body is read bounded and parsed off the loop with the sandbox routes' own helpers; every refusal
is ``{"reason", "detail"}``. A window that is not open (no announced pool, between windows) is a
retryable 503; anything else the miner must change is a 4xx."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from reliquary.protocol.signatures import verify_episode_precommit_signature
from reliquary.sandbox.routes import MAX_OPEN_BODY_BYTES, _answer, _parse, _refuse, _Refused, ss58_address
from reliquary.sandbox.sessions import SandboxPolicy
from reliquary.validator.corpus_registration import NOT_REGISTERED

logger = logging.getLogger(__name__)

PRECOMMIT_STATUS: dict[str, int] = {
    "precommit_invalid": 422, "hotkey_mismatch": 403, "precommit_stale": 409, "window_not_open": 503,
    "window_closed": 409, "order_mismatch": 409, "environment_not_episode": 409,
    "environment_not_active": 409, "task_out_of_range": 409, "checkpoint_mismatch": 409,
    "pool_mismatch": 409, "precommit_exists": 409, "task_in_cooldown": 409,
}


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
    if status == 503:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after or 1)))
    return JSONResponse(status_code=status, content={"reason": reason, "detail": dict(detail or {})},
                        headers=headers)


def build_episode_precommit_router(
    *, record: Callable[[Any], tuple[bool, str]], current_window: Callable[[], int | None],
    validator_hotkey: str, policy: SandboxPolicy, prefix: str = "/rl",
    verify: Callable[..., bool] = verify_episode_precommit_signature,
    registration: Callable[[str], Awaitable[str | None]] | None = None,
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    """``record``: ``ServiceRuntime.record_episode_precommit`` (blocking, run in a thread);
    ``current_window``: the service window admissions are open for, or None between windows."""
    audience = ss58_address(validator_hotkey)
    if audience is None:
        raise ValueError("validator_hotkey must be an ss58 account address")
    router = APIRouter()
    path = episode_precommit_path(prefix)
    retry_s = policy.retry_after_s

    @router.post(path)
    async def precommit(http: Request) -> JSONResponse:
        from reliquary.protocol.service_episode import EpisodePrecommit, EpisodeWireError
        from reliquary.services.runtime import EpisodePrecommitRefused

        try:
            request, hotkey = await _parse(http, EpisodePrecommitRequest, MAX_OPEN_BODY_BYTES)
            now = int(clock())
            if abs(now - request.at) > policy.request_skew_s:
                return _refuse("stale_request", {"max_skew_s": policy.request_skew_s, "now": now})
            try:
                value = EpisodePrecommit.from_dict(request.precommit)
            except (EpisodeWireError, TypeError):
                return _refusal("precommit_invalid")
            if ss58_address(value.hotkey) != hotkey:
                return _refusal("hotkey_mismatch")
            try:
                verified = bool(await asyncio.to_thread(verify, hotkey, request.precommit, at=request.at,
                                                        signature=request.signature,
                                                        validator_hotkey=audience, path=path))
            except Exception:
                verified = False
            if not verified:
                return _refuse("bad_signature")
            if registration is not None:
                try:
                    why = await registration(hotkey)
                except Exception as exc:
                    logger.warning("episode precommit registration check failed (%s)", type(exc).__name__)
                    why = "unavailable"
                if why == NOT_REGISTERED:
                    return _refuse("hotkey_not_registered", {"why": why})
                if why is not None:
                    return _refuse("registration_unavailable", {"why": str(why)}, retry_s)
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
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            logger.error("episode precommit failed (%s)", type(exc).__name__)
            return _refuse("internal_error")

    return router


__all__ = ["EpisodePrecommitRequest", "PRECOMMIT_STATUS", "build_episode_precommit_router",
           "episode_precommit_path"]
