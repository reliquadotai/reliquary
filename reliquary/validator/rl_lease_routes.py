"""The RL lease routes (rl-async-design.md §4.2-4.3, P2). Not wired into the service yet (P3 does it): the caller
builds a ``LeaseBook`` and a ``LeaseCollector`` and mounts ``build_rl_lease_router`` (``ValidatorServer
.mount_rl_lease_router``).

* ``POST {prefix}/lease`` ``{miner_hotkey, at, environment, checkpoint, prompt_ids, signature}``, signed by the
  hotkey for this validator and route (``build_rl_lease_binding``), within ``request_skew_s`` of now. The
  checks follow the allocation route (``sandbox/rl_routes.py``): the body bounded in size and time, at most
  ``MAX_PREAUTH`` decoded and verified at once (the signature in a worker thread), the registration checked, a
  signature answered once (``request_reused``), the hotkey's minute counted after verification. The request then
  waits for its interval and is resolved with the others (``LeaseCollector``). The answer lists the caller's leases
  and, per refused prompt, only ``taken`` / ``cooling`` / ``blacklisted`` / ``quota`` / ``stale`` / ``banned``.
* ``GET {prefix}/quota?hotkey=``: the hotkey's explainable quota (public, like ``/rl/pay``; it lists no lease).
* ``GET {prefix}/lease/commitments``: the commitments of the leasable checkpoints and the revealed secrets.
* ``GET {prefix}/prompts/state?since=&environment=``: the prompt lifecycle (``LeaseBook.prompt_state``): the scans
  after cursor ``since`` and the state (cooldown until k, blacklisted) of the prompts they moved. Public, no hotkey.
* ``POST {prefix}/submit`` (mounted when ``submissions`` is given) ``{miner_hotkey, at, lease_id, rollouts:
  [{index, tokens, force_span}], signature}``: a completed lease (``protocol/rl_submit.py``: every index, completion
  tokens only, compressed), signed over ``rollouts_digest`` of the rollouts as sent (``build_rl_submit_binding``),
  authenticated like a lease request with its own per-minute bound, inflated only after. Graded off the event loop (``RlSubmissions.submit``); the answer is
  the verdict and the validator's rewards. A malformed group is refused before the lease is touched (it stays
  open); a grader outage answers 503 (resend).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.protocol.rl_submit import GROUP_ROLLOUTS, decode_tokens, rollouts_digest
from reliquary.protocol.signatures import (
    hotkey_signature_bytes, verify_rl_lease_signature, verify_rl_submit_signature,
)
from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
from reliquary.validator.corpus_registration import NOT_REGISTERED
from reliquary.validator.rl_leases import MAX_PROMPTS_PER_REQUEST, LeaseBook, LeaseCollector, LeaseRequest

logger = logging.getLogger(__name__)

# The statuses follow the sandbox routes' (``sandbox/routes.py``): 4xx the miner must change, 429/503 retry after
# ``Retry-After``. This module does not import them: the validator serves leases without the sandbox package.
LEASE_STATUS: dict[str, int] = {
    "stale_request": 400, "body_too_large": 413, "malformed_request": 422, "body_timeout": 408,
    "signature_required": 401, "bad_signature": 403, "hotkey_not_registered": 403,
    "lease_invalid": 409, "request_reused": 409, "lease_rate": 429,
    "lease_busy": 503, "registration_unavailable": 503, "internal_error": 500,
    "lease_unknown": 404, "lease_closed": 409, "lease_not_yours": 403, "grading_unavailable": 503,
}
REQUEST_SKEW_S = 120
RETRY_AFTER_S = 10
SS58_FORMAT = 42
LEASE_MAX_PER_MINUTE = 120
MAX_BODY_BYTES = 32 * 1024
BODY_TIMEOUT_S = 10.0
MAX_PREAUTH = 8
MAX_TRACKED_HOTKEYS = 4096
SUBMIT_MAX_PER_MINUTE = 600          # a submission ends a lease: the quota bounds them, this bounds the verify cost
MAX_SUBMIT_BODY_BYTES = 8 * 1024 * 1024
SUBMIT_BODY_TIMEOUT_S = 30.0
MAX_GRADING = 4                       # groups graded at once (CPU); the others wait their turn


def rl_lease_path(prefix: str = "/rl") -> str:
    return f"{prefix}/lease"


class RlLeaseRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    at: int = Field(ge=0, strict=True)
    environment: str = Field(min_length=1, max_length=128)
    checkpoint: int = Field(ge=0, strict=True)
    prompt_ids: list[int] = Field(min_length=1, max_length=MAX_PROMPTS_PER_REQUEST)
    signature: str = Field(default="", max_length=256)


class RlSubmitRollout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, strict=True)
    tokens: str = Field(min_length=1, max_length=MAX_SUBMIT_BODY_BYTES)
    force_span: tuple[int, int] | None = None


class RlSubmitRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    at: int = Field(ge=0, strict=True)
    lease_id: str = Field(min_length=1, max_length=64)
    rollouts: list[RlSubmitRollout] = Field(min_length=1, max_length=GROUP_ROLLOUTS)
    signature: str = Field(default="", max_length=256)


def submitted_rollouts(body: RlSubmitRequestBody, *, max_tokens: int, rollouts: int = GROUP_ROLLOUTS):
    """The group in index order as ``rl_submit.Rollout``; ValueError unless indices are exactly 0..rollouts-1 and
    every rollout decodes (``decode_tokens``) with a well-formed span inside its completion."""
    from reliquary.validator.rl_submit import Rollout

    if sorted(r.index for r in body.rollouts) != list(range(rollouts)):
        raise ValueError(f"rollout indices must be exactly 0..{rollouts - 1}")
    group = []
    for item in sorted(body.rollouts, key=lambda r: r.index):
        tokens = decode_tokens(item.tokens, max_tokens=max_tokens)
        span = item.force_span
        if span is not None and not (0 <= span[0] < span[1] <= len(tokens)):
            raise ValueError("force span must lie inside the completion")
        group.append(Rollout(tokens, span))
    return group


def ss58_address(address) -> str | None:
    """``address`` re-encoded as an ss58 format-42 account address, or None (fail closed without the decoder)."""
    if not isinstance(address, str) or not address:
        return None
    try:
        from scalecodec.utils.ss58 import ss58_decode, ss58_encode

        public_key = ss58_decode(address)
        if not isinstance(public_key, str) or len(public_key) != 64:
            return None
        return ss58_encode(public_key, SS58_FORMAT)
    except Exception:
        return None


class _Refused(Exception):
    def __init__(self, response: JSONResponse) -> None:
        self.response = response


def _answer(content: dict) -> JSONResponse:
    return JSONResponse(status_code=200, content=content, headers={"Cache-Control": "no-store"})


def _refusal(reason: str, detail: dict | None = None, retry_after: float | None = None) -> JSONResponse:
    status = LEASE_STATUS[reason]
    headers = {"Cache-Control": "no-store"}
    if status in (429, 503):
        headers["Retry-After"] = str(max(1, math.ceil(retry_after or 1)))
    return JSONResponse(status_code=status, content={"reason": reason, "detail": dict(detail or {})},
                        headers=headers)




async def _read_within(request: Request, cap: int, deadline_s: float) -> bytes:
    async def read() -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > cap:
            raise _Refused(_refusal("body_too_large", {"max_bytes": cap}))
        chunks, received = [], 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > cap:
                raise _Refused(_refusal("body_too_large", {"max_bytes": cap}))
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        return await asyncio.wait_for(read(), deadline_s)
    except TimeoutError:
        raise _Refused(_refusal("body_timeout", {"max_seconds": deadline_s})) from None


def _no_constants(name: str):
    raise ValueError(f"{name} is not JSON")


def _decode(raw: bytes, model: type[BaseModel] = RlLeaseRequestBody) -> tuple[BaseModel, str]:
    try:
        body = json.loads(raw.decode("utf-8"), parse_constant=_no_constants)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _Refused(_refusal("malformed_request", {"why": "the body is not JSON"})) from None
    if not isinstance(body, dict):
        raise _Refused(_refusal("malformed_request", {"why": "the body is not a JSON object"}))
    try:
        parsed = model.model_validate(body)
    except ValidationError as exc:
        errors = [{"loc": [str(part) for part in error.get("loc", ())], "type": str(error.get("type"))}
                  for error in exc.errors(include_url=False, include_input=False, include_context=False)[:8]]
        raise _Refused(_refusal("malformed_request", {"errors": errors})) from None
    hotkey = ss58_address(parsed.miner_hotkey)
    if hotkey is None:
        raise _Refused(_refusal("malformed_request", {"errors": [{"loc": ["miner_hotkey"], "type": "ss58_address"}]}))
    return parsed, hotkey


def build_rl_lease_router(*, book: LeaseBook, collector: LeaseCollector, validator_hotkey: str,
                          prefix: str = "/rl", request_skew_s: int = REQUEST_SKEW_S,
                          retry_after_s: int = RETRY_AFTER_S,
                          verify: Callable[..., bool] = verify_rl_lease_signature,
                          registration: Callable[[str], Awaitable[str | None]] | None = None,
                          clock: Callable[[], float] = time.time, submissions=None,
                          verify_submit: Callable[..., bool] = verify_rl_submit_signature) -> APIRouter:
    audience = ss58_address(validator_hotkey)
    if audience is None:
        raise ValueError("validator_hotkey must be an ss58 account address")
    router = APIRouter()
    path = rl_lease_path(prefix)
    retry_s = retry_after_s
    skew = request_skew_s
    preauth = asyncio.Semaphore(MAX_PREAUTH)
    recent: OrderedDict[str, deque] = OrderedDict()
    recent_submits: OrderedDict[str, deque] = OrderedDict()
    used: OrderedDict[str, float] = OrderedDict()
    grading = asyncio.Semaphore(MAX_GRADING)

    def remember(hotkey: str, signature: str, now: float, *, minutes: OrderedDict = recent,
                 per_minute: int = LEASE_MAX_PER_MINUTE) -> JSONResponse | None:
        while used and next(iter(used.values())) < now:
            used.popitem(last=False)
        if signature in used:
            return _refusal("request_reused", {})
        times = minutes.setdefault(hotkey, deque())
        minutes.move_to_end(hotkey)
        while times and times[0] <= now - 60.0:
            times.popleft()
        if len(times) >= per_minute:
            return _refusal("lease_rate", {"max_per_minute": per_minute}, times[0] + 60.0 - now)
        times.append(now)
        used[signature] = now + 2 * skew + 1
        while len(minutes) > MAX_TRACKED_HOTKEYS:
            minutes.popitem(last=False)
        return None

    async def authenticate(raw: bytes, model: type[BaseModel] = RlLeaseRequestBody,
                           check: Callable[[BaseModel, str], bool] | None = None,
                           minutes: OrderedDict = recent,
                           per_minute: int = LEASE_MAX_PER_MINUTE) -> tuple[BaseModel, str]:
        request, hotkey = await asyncio.to_thread(_decode, raw, model)
        if not request.signature:
            raise _Refused(_refusal("signature_required", {}))
        if registration is not None:
            try:
                why = await registration(hotkey)
            except Exception as exc:
                logger.warning("lease registration check failed (%s)", type(exc).__name__)
                why = "unavailable"
            if why == NOT_REGISTERED:
                raise _Refused(_refusal("hotkey_not_registered", {"why": why}))
            if why is not None:
                raise _Refused(_refusal("registration_unavailable", {"why": str(why)}, retry_s))
        now = float(clock())
        if abs(int(now) - request.at) > skew:
            raise _Refused(_refusal("stale_request", {"max_skew_s": skew, "now": int(now)}))
        if check is None:
            def check(request: BaseModel, hotkey: str) -> bool:
                return verify(hotkey, at=request.at, environment=request.environment, checkpoint=request.checkpoint,
                              prompt_ids=list(request.prompt_ids), signature=request.signature,
                              validator_hotkey=audience, path=path)
        try:
            verified = bool(await asyncio.to_thread(check, request, hotkey))
        except _Refused:
            raise
        except Exception:
            verified = False
        if not verified:
            raise _Refused(_refusal("bad_signature"))
        signature = hotkey_signature_bytes(request.signature)
        refused = remember(hotkey, request.signature if signature is None else signature.hex(), now,
                           minutes=minutes, per_minute=per_minute)
        if refused is not None:
            raise _Refused(refused)
        return request, hotkey

    @router.post(path)
    async def lease(http: Request) -> JSONResponse:
        try:
            raw = await _read_within(http, MAX_BODY_BYTES, BODY_TIMEOUT_S)
            if preauth.locked():
                return _refusal("lease_busy", {"max_concurrent": MAX_PREAUTH}, retry_s)
            async with preauth:
                request, hotkey = await authenticate(raw)
            asked = LeaseRequest(hotkey=hotkey, environment=request.environment,
                                 prompt_ids=tuple(request.prompt_ids), checkpoint=request.checkpoint)
            try:
                answer = await collector.request(asked)
            except ValueError as exc:
                return _refusal("lease_invalid", {"why": str(exc)})
            now = float(clock())
            content = answer.to_dict()
            # SQLite reads under the book's lock: off the event loop, like every other book call
            latest, granted, in_use = await asyncio.to_thread(
                lambda: (book.latest_checkpoint, book.quota.quota(hotkey, now), book.in_use(hotkey)))
            content.update(latest=latest, quota=granted, in_use=in_use)
            if answer.leases:
                logger.info("rl leases: %d to %s on %s k=%d (%d refused)", len(answer.leases), hotkey[:12],
                            request.environment, request.checkpoint, len(answer.refused))
            return _answer(content)
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            logger.error("rl lease failed (%s)", type(exc).__name__)
            return _refusal("internal_error")

    @router.get(f"{prefix}/quota")
    async def quota(hotkey: str = "") -> JSONResponse:
        address = ss58_address(hotkey)
        if address is None:
            return _refusal("malformed_request", {"errors": [{"loc": ["hotkey"], "type": "ss58_address"}]})
        now = float(clock())

        def read() -> dict:
            state = book.quota.state(address, now)
            state.update(in_use=book.in_use(address), max_lag=book.config.max_lag, latest=book.latest_checkpoint)
            return state

        return _answer(await asyncio.to_thread(read))

    @router.get(f"{prefix}/lease/commitments")
    async def commitments() -> JSONResponse:
        return _answer(await asyncio.to_thread(book.commitments))

    @router.get(f"{prefix}/prompts/state")
    async def prompt_state(since: str = "0", environment: str | None = None) -> JSONResponse:
        if not (since.isascii() and since.isdigit()) or len(since) > 19:
            return _refusal("malformed_request", {"errors": [{"loc": ["since"], "type": "cursor"}]})
        return _answer(await asyncio.to_thread(book.prompt_state, int(since), environment=environment))

    if submissions is None:
        return router
    submit_path = f"{prefix}/submit"

    def check_submit(request: RlSubmitRequestBody, hotkey: str) -> bool:
        if sorted(r.index for r in request.rollouts) != list(range(submissions.config.rollouts)):
            raise _Refused(_refusal("malformed_request", {
                "why": f"rollout indices must be exactly 0..{submissions.config.rollouts - 1}"}))
        digest = rollouts_digest([(r.index, r.tokens, r.force_span)
                                  for r in sorted(request.rollouts, key=lambda r: r.index)])
        return verify_submit(hotkey, at=request.at, lease_id=request.lease_id, rollouts_digest=digest,
                             signature=request.signature, validator_hotkey=audience, path=submit_path)

    @router.post(submit_path)
    async def submit(http: Request) -> JSONResponse:
        try:
            raw = await _read_within(http, MAX_SUBMIT_BODY_BYTES, SUBMIT_BODY_TIMEOUT_S)
            arrived_at = float(clock())                       # the lease is judged at arrival, not after grading
            if preauth.locked():
                return _refusal("lease_busy", {"max_concurrent": MAX_PREAUTH}, retry_s)
            async with preauth:
                request, hotkey = await authenticate(raw, RlSubmitRequestBody, check_submit, recent_submits,
                                                     SUBMIT_MAX_PER_MINUTE)
            try:
                group = await asyncio.to_thread(submitted_rollouts, request, max_tokens=MAX_NEW_TOKENS_PROTOCOL_CAP,
                                                rollouts=submissions.config.rollouts)
            except ValueError as exc:
                return _refusal("malformed_request", {"why": str(exc)})
            async with grading:
                try:
                    answer = await asyncio.to_thread(submissions.submit, request.lease_id, hotkey, group,
                                                     arrived_at=arrived_at)
                except GraderInfrastructureError:
                    return _refusal("grading_unavailable", {}, retry_s)
                except PermissionError as exc:
                    return _refusal("lease_not_yours", {"why": str(exc)})
                except LookupError as exc:
                    reason = "lease_unknown" if "unknown" in str(exc) else "lease_closed"
                    return _refusal(reason, {"why": str(exc)})
                except ValueError as exc:
                    return _refusal("malformed_request", {"why": str(exc)})
            logger.info("rl submit: %s from %s: %s", request.lease_id[:12], hotkey[:12], answer["verdict"])
            return _answer(answer)
        except _Refused as refused:
            return refused.response
        except Exception as exc:
            logger.error("rl submit failed (%s)", type(exc).__name__)
            return _refusal("internal_error")

    return router


__all__ = [
    "LEASE_MAX_PER_MINUTE", "LEASE_STATUS", "SUBMIT_MAX_PER_MINUTE", "RlLeaseRequestBody", "RlSubmitRequestBody",
    "build_rl_lease_router", "rl_lease_path", "submitted_rollouts",
]
