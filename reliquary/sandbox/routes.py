"""`POST {prefix}/sandbox/sessions` and `POST {prefix}/sandbox/sessions/{id}/close`
(plan 3 ruling 1). Mounted on the corpus app with prefix `/corpus` (the nginx
location and tunnel miners already reach); the RL server mounts the same router under
its own prefix. A request is fresh (±policy.request_skew_s of this validator's
clock), signed by its hotkey and, when a registration check is wired, registered,
before the issuer sees it.

The body is read here, not by FastAPI: it is bounded before it is parsed (by its
declared length and as it streams), and a malformed one is answered with the error
locations only, never FastAPI's default 422 that echoes the input (a close's transcript
carries the session token). Every refusal is `{"reason", "detail"}` with the status from
`REFUSAL_STATUS`, plus `Retry-After` when the refusal names a delay. An issuer failure
is a bare 500 and logs its exception type only. The grant's token is in the answer
body once and nowhere else; nothing here logs it."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES
from reliquary.protocol.sandbox_session import SandboxSessionCloseRequest, SandboxSessionOpenRequest
from reliquary.protocol.signatures import (
    verify_sandbox_close_signature, verify_sandbox_open_signature,
)
from reliquary.sandbox.sessions import Refusal, SandboxPolicy

logger = logging.getLogger(__name__)

NOT_REGISTERED = "not_registered"          # corpus_registration.NOT_REGISTERED
MAX_OPEN_BODY_BYTES = 16 * 1024
MAX_CLOSE_BODY_BYTES = MAX_TRANSCRIPT_BYTES + 64 * 1024
MAX_REPORTED_ERRORS = 8

# Every reason either side of the routes refuses with, and the status a miner acts on:
# 400/403/404/409/413/422 do not retry the same request; 429 waits for a session to
# end; 503 retries after `Retry-After`; 500 is this validator's fault.
REFUSAL_STATUS: dict[str, int] = {
    # the request itself
    "stale_request": 400,
    "body_too_large": 413,
    "malformed_request": 422,
    # who is asking
    "bad_signature": 403, "hotkey_not_registered": 403, "miner_banned": 403,
    # what is asked for
    "job_not_served": 404, "session_unknown": 404,
    "job_not_signed": 409, "prompt_mismatch": 409, "prompt_unavailable": 409,
    "job_complete": 409, "request_reused": 409, "request_conflict": 409,
    "engagement_kind_unsupported": 409, "transcript_invalid": 409,
    # per-hotkey caps
    "live_cap": 429, "prompt_live_cap": 429, "job_live_cap": 429, "open_rate_cap": 429,
    "aborted_cap": 429,
    # this validator cannot answer now: retry
    "sandbox_capacity": 503, "directory_unavailable": 503, "store_unavailable": 503,
    "ledger_unavailable": 503, "task_unavailable": 503, "registration_unavailable": 503,
    "internal_error": 500,
}
UNMAPPED_STATUS = 409


def _refuse(reason: str, detail: dict | None = None,
            retry_after: int | None = None) -> JSONResponse:
    status = REFUSAL_STATUS.get(reason)
    if status is None:
        logger.warning("sandbox session refusal %s has no status; answering %d", reason,
                       UNMAPPED_STATUS)
        status = UNMAPPED_STATUS
    headers = {"Cache-Control": "no-store"}
    if retry_after:
        headers["Retry-After"] = str(int(retry_after))
    return JSONResponse(status_code=status, content={"reason": reason, "detail": dict(detail or {})},
                        headers=headers)


def _answer(content: dict) -> JSONResponse:
    return JSONResponse(status_code=200, content=content, headers={"Cache-Control": "no-store"})


class _Refused(Exception):
    def __init__(self, response: JSONResponse) -> None:
        self.response = response


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


async def _read_bounded(request: Request, cap: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        raise _Refused(_refuse("body_too_large", {"max_bytes": cap}))
    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > cap:
            raise _Refused(_refuse("body_too_large", {"max_bytes": cap}))
        chunks.append(chunk)
    return b"".join(chunks)


async def _parse(request: Request, model: type[BaseModel], cap: int) -> Any:
    raw = await _read_bounded(request, cap)
    try:
        body = json.loads(raw.decode("utf-8"), parse_constant=_no_constants)
    except (UnicodeDecodeError, ValueError):
        raise _Refused(_refuse("malformed_request", {"why": "the body is not JSON"})) from None
    if not isinstance(body, dict):
        raise _Refused(_refuse("malformed_request", {"why": "the body is not a JSON object"}))
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        # Locations and error types only: a message or an input could carry the token.
        errors = [{"loc": [str(part) for part in error.get("loc", ())],
                   "type": str(error.get("type"))}
                  for error in exc.errors(include_url=False, include_input=False,
                                          include_context=False)[:MAX_REPORTED_ERRORS]]
        raise _Refused(_refuse("malformed_request", {"errors": errors})) from None


def build_sandbox_sessions_router(
    issuer, *, policy: SandboxPolicy, prefix: str = "/corpus",
    verify_open: Callable = verify_sandbox_open_signature,
    verify_close: Callable = verify_sandbox_close_signature,
    registration: Callable[[str], Awaitable[str | None]] | None = None,
    clock: Callable[[], float] = time.time,
) -> APIRouter:
    router = APIRouter()

    async def gate(request, verify) -> None:
        if abs(int(clock()) - request.at) > policy.request_skew_s:
            raise _Refused(_refuse("stale_request", {"max_skew_s": policy.request_skew_s}))
        try:
            verified = bool(verify(request))
        except Exception:
            verified = False
        if not verified:
            raise _Refused(_refuse("bad_signature"))
        if registration is not None:
            try:
                why = await registration(request.miner_hotkey)
            except Exception as exc:
                logger.warning("sandbox session registration check failed (%s)",
                               type(exc).__name__)
                why = "unavailable"
            if why == NOT_REGISTERED:
                raise _Refused(_refuse("hotkey_not_registered", {"why": why}))
            if why is not None:
                raise _Refused(_refuse("registration_unavailable", {"why": str(why)},
                                       policy.retry_after_s))

    def outcome_of(what: str, hotkey: str, outcome) -> JSONResponse | None:
        if isinstance(outcome, Refusal):
            logger.info("sandbox session %s refused for %s: %s", what, hotkey[:12], outcome.reason)
            return _refuse(outcome.reason, outcome.detail, outcome.retry_after)
        return None

    async def guarded(what: str, step: Callable[[], Awaitable[JSONResponse]]) -> JSONResponse:
        try:
            return await step()
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            # The exception type only: neither a traceback nor a message reaches a log
            # or the answer (either could carry a request's token).
            logger.error("sandbox session %s failed (%s)", what, type(exc).__name__)
            return _refuse("internal_error")

    @router.post(f"{prefix}/sandbox/sessions")
    async def open_session(http: Request) -> JSONResponse:
        async def step() -> JSONResponse:
            request = await _parse(http, SandboxSessionOpenRequest, MAX_OPEN_BODY_BYTES)
            await gate(request, verify_open)
            outcome = await issuer.open(
                hotkey=request.miner_hotkey, request_id=request.request_id,
                engagement=request.engagement.model_dump(exclude_none=True))
            refused = outcome_of("open", request.miner_hotkey, outcome)
            if refused is not None:
                return refused
            logger.info("sandbox session %s granted to %s", outcome.session_id,
                        request.miner_hotkey[:12])
            return _answer({"session_id": outcome.session_id, "token": outcome.token,
                            "gateway_url": outcome.gateway_url, "expires_at": outcome.expires_at})

        return await guarded("open", step)

    @router.post(f"{prefix}/sandbox/sessions/{{session_id}}/close")
    async def close_session(session_id: str, http: Request) -> JSONResponse:
        async def step() -> JSONResponse:
            request = await _parse(http, SandboxSessionCloseRequest, MAX_CLOSE_BODY_BYTES)
            if request.session_id != session_id:
                return _refuse("session_unknown", {"session_id": request.session_id})
            await gate(request, verify_close)
            outcome = await issuer.close(hotkey=request.miner_hotkey, session_id=session_id,
                                         reason=request.reason, transcript=request.transcript)
            refused = outcome_of("close", request.miner_hotkey, outcome)
            if refused is not None:
                return refused
            return _answer(dict(outcome))

        return await guarded("close", step)

    return router


__all__ = ["MAX_CLOSE_BODY_BYTES", "MAX_OPEN_BODY_BYTES", "REFUSAL_STATUS",
           "build_sandbox_sessions_router"]
