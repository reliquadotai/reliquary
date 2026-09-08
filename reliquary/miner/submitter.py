"""HTTP client used by miners to push GRPO submissions to the validator.

V1 assumption: a single validator. Discovery returns the first axon advertised
by a hotkey holding `validator_permit`. Multi-validator routing is intentionally
out of scope here — see the GRPO refactor plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from contextlib import asynccontextmanager
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.signatures import (
    sign_envelope,
    sign_precommit,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    MinerState,
    RejectReason,
    RuntimeContract,
    VerdictsPage,
    SubmissionPrecommitRequest,
    SubmissionPrecommitResponse,
)
from reliquary.shared.runtime_fingerprint import bind_runtime_profile_nonce

logger = logging.getLogger(__name__)

# Retry configuration: 3 attempts, exponential backoff 1s / 2s / 4s.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
# Default timeout is generous: the validator may need several seconds to verify
# a submission even in the async-queue path (the queue can back up under load).
# Miners running against slow links (Targon port-forward etc.) benefit further.
_DEFAULT_TIMEOUT = 60.0
_CONTROL_TIMEOUT = 3.0
_OPTIONAL_TIMEOUT = 2.0
_CONTROL_RETRY_DELAYS = (0.2, 0.5, 1.0)
_PRECOMMIT_HEADER = "X-Reliquary-Precommit"
_DRAND_BOUNDARY_SAFETY_SECONDS = 1.0
_DRAND_BOUNDARY_SETTLE_SECONDS = 0.05


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted."""


class EndpointNotFoundError(SubmissionError):
    """An optional or versioned validator capability is unavailable."""




def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, min(seconds, 60.0)) if math.isfinite(seconds) else None
    except ValueError:
        return None

async def _sleep_retry(delay: float, *, jitter: bool) -> None:
    if jitter:
        delay *= random.uniform(0.8, 1.2)
    await asyncio.sleep(max(0.0, delay))


def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Return the HTTP URL of the first validator advertised on the metagraph.

    Picks the first uid with validator_permit=True and an axon IP that isn't
    the unset placeholder. Multi-validator coordination is out of scope; this
    deliberately picks ONE validator.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        # Use the validator's own port if it's set; fall back to the protocol default.
        axon_port = getattr(axon, "port", None) or port
        return f"http://{ip}:{axon_port}"
    raise NoValidatorFoundError("no validator with permit and routable axon")


async def _post_with_retry(
    full_url: str,
    payload_factory: Callable[[int], bytes],
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            payload = payload_factory(attempt)
            try:
                resp = await cli.post(
                    full_url,
                    content=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "submit attempt %d to %s failed: %r (type=%s)",
                    attempt, full_url, e, type(e).__name__,
                )
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            # 503 "no active window" is informational for BatchSubmissionResponse —
            # don't retry, surface as a structured reject.
            if resp.status_code == 503 and response_model is BatchSubmissionResponse:
                result = BatchSubmissionResponse(accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE)
                result._retry_after_seconds = _retry_after_seconds(resp)
                return result
            # 4xx means the request is malformed or the validator rejected it
            # for a deterministic reason — retrying is pointless. Parse and return.
            if 400 <= resp.status_code < 500:
                detail = _safe_detail(resp)
                if response_model is BatchSubmissionResponse:
                    if resp.status_code == 409:
                        reason = RejectReason.WINDOW_MISMATCH
                    else:
                        reason = RejectReason.BAD_PROMPT_IDX
                    return BatchSubmissionResponse(accepted=False, reason=reason)
                raise SubmissionError(f"HTTP {resp.status_code}: {detail}")
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def _get_with_retry(
    full_url: str,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
    retry_delays: tuple[float, ...] = _RETRY_DELAYS,
    jitter: bool = False,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(retry_delays, start=1):
            try:
                resp = await cli.get(full_url, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < len(retry_delays):
                    await _sleep_retry(delay, jitter=jitter)
                continue
            if resp.status_code == 503:
                # No active window yet — caller's job to handle.
                retry_after = _retry_after_seconds(resp)
                suffix = (
                    f" retry_after={retry_after}"
                    if retry_after is not None else ""
                )
                raise SubmissionError(
                    f"no active window at {full_url}{suffix}"
                )
            if resp.status_code == 404:
                raise EndpointNotFoundError(
                    f"endpoint not found: {full_url}"
                )
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(retry_delays):
                    await _sleep_retry(delay, jitter=jitter)
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


def _safe_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]






async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    wallet: Any | None = None,
    randomness: str = "",
    drand_round_fn: Callable[[], int] | None = None,
) -> BatchSubmissionResponse:
    """POST a v2 batch submission, refreshing signed freshness per attempt.

    When ``wallet`` is provided, the miner-finalized envelope fields are rebuilt
    immediately before every network attempt. The large rollout body is encoded
    exactly once with Pydantic's native JSON serializer, avoiding the old
    ``model_dump`` plus httpx JSON double pass. A retry never reuses a stale
    drand round or nonce. Callers that already finalized an envelope can omit
    ``wallet`` and retain the legacy pre-signed behavior.
    """

    drand_chain_info: dict[str, Any] | None = None
    if wallet is not None and drand_round_fn is None:
        from reliquary.infrastructure.chain import compute_current_drand_round
        from reliquary.infrastructure.drand import get_current_chain

        drand_chain_info = get_current_chain()

        def drand_round_fn() -> int:
            return compute_current_drand_round(
                time.time(),
                drand_chain_info["genesis_time"],
                drand_chain_info["period"],
            )

    async def _wait_for_safe_drand_round() -> None:
        if drand_chain_info is None:
            return
        from reliquary.infrastructure.chain import (
            seconds_until_next_drand_boundary,
        )

        remaining = seconds_until_next_drand_boundary(
            time.time(),
            drand_chain_info["genesis_time"],
            drand_chain_info["period"],
        )
        if 0.0 < remaining < _DRAND_BOUNDARY_SAFETY_SECONDS:
            await asyncio.sleep(remaining + _DRAND_BOUNDARY_SETTLE_SECONDS)

    wire_exclude = (
        {"generation_profile_id"}
        if not request.generation_profile_id
        else None
    )
    static_payload = (
        request.model_dump_json(exclude=wire_exclude).encode("utf-8")
        if wallet is None
        else None
    )

    def _finalize_attempt(
        attempt: int,
    ) -> tuple[bytes, SubmissionPrecommitRequest]:
        assert wallet is not None
        assert drand_round_fn is not None
        environments = {rollout.env_name for rollout in request.rollouts}
        if len(environments) != 1:
            raise SubmissionError("submission must contain exactly one environment")
        environment = next(iter(environments))
        drand_round = int(drand_round_fn())
        nonce = os.urandom(16).hex()
        if request.runtime_fingerprint is not None:
            nonce = bind_runtime_profile_nonce(
                nonce, request.runtime_fingerprint.profile_hash,
            )
        signature = sign_envelope(
            wallet=wallet,
            miner_hotkey=request.miner_hotkey,
            window_start=request.window_start,
            prompt_idx=request.prompt_idx,
            merkle_root=request.merkle_root,
            checkpoint_hash=request.checkpoint_hash,
            drand_round=drand_round,
            randomness=randomness,
            nonce=nonce,
            protocol_version=request.protocol_version,
            generation_profile_id=request.generation_profile_id,
        ).hex()
        finalized = request.model_copy(
            update={
                "drand_round": drand_round,
                "nonce": nonce,
                "envelope_signature": signature,
            }
        )
        started = time.perf_counter()
        payload = finalized.model_dump_json(exclude=wire_exclude).encode("utf-8")
        precommit_fields = {
            "miner_hotkey": request.miner_hotkey,
            "window_start": request.window_start,
            "prompt_idx": request.prompt_idx,
            "merkle_root": request.merkle_root,
            "checkpoint_hash": request.checkpoint_hash,
            "environment": environment,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "drand_round": drand_round,
            "randomness": randomness,
            "protocol_version": request.protocol_version,
            "generation_profile_id": request.generation_profile_id,
            "nonce": nonce,
        }
        precommit_signature = sign_precommit(
            wallet=wallet,
            **precommit_fields,
        ).hex()
        precommit = SubmissionPrecommitRequest(
            **{
                key: value
                for key, value in precommit_fields.items()
                if key != "randomness"
            },
            precommit_signature=precommit_signature,
        )
        logger.info(
            "submission_payload_finalized window=%d prompt=%d attempt=%d "
            "drand_round=%d payload_bytes=%d serialization_ms=%.3f",
            request.window_start,
            request.prompt_idx,
            attempt,
            drand_round,
            len(payload),
            (time.perf_counter() - started) * 1000.0,
        )
        return payload, precommit

    def _payload_for_attempt(attempt: int) -> bytes:
        if wallet is None:
            assert static_payload is not None
            return static_payload
        return _finalize_attempt(attempt)[0]

    if wallet is None:
        return await _post_with_retry(
            f"{url}/submit",
            _payload_for_attempt,
            BatchSubmissionResponse,
            client=client,
            timeout=timeout,
        )

    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    await _wait_for_safe_drand_round()
    payload, precommit = _finalize_attempt(1)
    receipt_id: str | None = None
    last_exc: Exception | None = None
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            if receipt_id is None:
                try:
                    precommit_response = await cli.post(
                        f"{url}/submit/precommit",
                        content=precommit.model_dump_json(
                            exclude=(
                                {"generation_profile_id"}
                                if not precommit.generation_profile_id
                                else None
                            )
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        timeout=timeout,
                    )
                except (httpx.RequestError, httpx.TimeoutException) as exc:
                    last_exc = exc
                    if attempt < len(_RETRY_DELAYS):
                        await asyncio.sleep(delay)
                    continue
                if precommit_response.status_code == 404:
                    logger.warning(
                        "validator has no upload-precommit endpoint; using "
                        "deadline-sensitive direct submission"
                    )
                    return await _post_with_retry(
                        f"{url}/submit",
                        _payload_for_attempt,
                        BatchSubmissionResponse,
                        client=cli,
                        timeout=timeout,
                    )
                if precommit_response.status_code >= 500:
                    last_exc = SubmissionError(
                        f"HTTP {precommit_response.status_code} from precommit"
                    )
                    if attempt < len(_RETRY_DELAYS):
                        await asyncio.sleep(delay)
                    continue
                if precommit_response.status_code >= 400:
                    raise SubmissionError(
                        f"precommit HTTP {precommit_response.status_code}: "
                        f"{_safe_detail(precommit_response)}"
                    )
                precommit_verdict = SubmissionPrecommitResponse.model_validate(
                    precommit_response.json()
                )
                if not precommit_verdict.accepted:
                    if (
                        precommit_verdict.reason is RejectReason.BATCH_FILLED
                        and attempt < len(_RETRY_DELAYS)
                    ):
                        # Upload capacity is a live reservation pool: cheap
                        # terminal rejects may return a slot immediately.  Keep
                        # the exact signed precommit and retry with the existing
                        # bounded backoff rather than turning a transient burst
                        # into a permanent missed window.
                        await asyncio.sleep(delay)
                        continue
                    if (
                        precommit_verdict.reason is RejectReason.STALE_ROUND
                        and attempt < len(_RETRY_DELAYS)
                    ):
                        await asyncio.sleep(delay)
                        await _wait_for_safe_drand_round()
                        payload, precommit = _finalize_attempt(attempt + 1)
                        continue
                    return BatchSubmissionResponse(
                        accepted=False,
                        reason=precommit_verdict.reason,
                    )
                receipt_id = precommit_verdict.receipt_id
                if not receipt_id:
                    raise SubmissionError("accepted precommit omitted receipt_id")

            try:
                response = await cli.post(
                    f"{url}/submit",
                    content=payload,
                    headers={
                        "Content-Type": "application/json",
                        _PRECOMMIT_HEADER: receipt_id,
                    },
                    timeout=timeout,
                )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if response.status_code == 503:
                result = BatchSubmissionResponse(accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE)
                result._retry_after_seconds = _retry_after_seconds(response)
                return result
            if 400 <= response.status_code < 500:
                reason = (
                    RejectReason.WINDOW_MISMATCH
                    if response.status_code == 409
                    else RejectReason.BAD_PROMPT_IDX
                )
                return BatchSubmissionResponse(accepted=False, reason=reason)
            if response.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {response.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return BatchSubmissionResponse.model_validate(response.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def get_window_state_v2(
    url: str,
    *,
    env: str | None = None,
    window: int | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = _CONTROL_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's current v2 GrpoBatchState.

    ``cooldown_prompts`` is per-env; pass ``env`` to read a specific env's
    cooldown set. Omitting it preserves legacy behavior.
    """
    if window is not None and env is None:
        raise ValueError("window requires env")
    state_url = f"{url}/state"
    query: list[str] = []
    if env is not None:
        query.append(f"env={quote(env, safe='')}")
    if window is not None:
        query.append(f"window={int(window)}")
    if query:
        state_url = f"{state_url}?{'&'.join(query)}"
    return await _get_with_retry(
        state_url, GrpoBatchState,
        client=client, timeout=timeout,
        retry_delays=_CONTROL_RETRY_DELAYS, jitter=True,
    )


async def get_miner_state_v1(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _CONTROL_TIMEOUT,
    etag: str | None = None,
) -> tuple[MinerState | None, str | None]:
    """Fetch bounded state, returning ``None`` for a conditional 304."""
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    last_exc: Exception | None = None
    try:
        for attempt, delay in enumerate(_CONTROL_RETRY_DELAYS, start=1):
            headers = {"Accept-Encoding": "gzip"}
            if etag:
                headers["If-None-Match"] = etag
            try:
                response = await cli.get(
                    f"{url}/miner-state",
                    headers=headers,
                    timeout=timeout,
                )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < len(_CONTROL_RETRY_DELAYS):
                    await _sleep_retry(delay, jitter=True)
                continue
            if response.status_code == 304:
                return None, response.headers.get("ETag") or etag
            if response.status_code in {404, 409}:
                raise EndpointNotFoundError(
                    f"endpoint unavailable: {url}/miner-state"
                )
            if response.status_code == 503:
                raise SubmissionError(
                    f"no active window at {url}/miner-state"
                )
            if response.status_code >= 400:
                last_exc = SubmissionError(
                    f"HTTP {response.status_code}: {_safe_detail(response)}"
                )
                if response.status_code < 500:
                    raise last_exc
                if attempt < len(_CONTROL_RETRY_DELAYS):
                    await _sleep_retry(delay, jitter=True)
                continue
            return (
                MinerState.model_validate(response.json()),
                response.headers.get("ETag"),
            )
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()




















async def get_runtime_contract_v1(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _OPTIONAL_TIMEOUT,
) -> RuntimeContract:
    """Discover optional runtime telemetry without changing legacy `/state`."""
    return await _get_with_retry(
        f"{url}/runtime-contract",
        RuntimeContract,
        client=client,
        timeout=timeout,
        retry_delays=(0.0,),
        jitter=False,
    )




async def get_verdicts_page_v1(
    url: str,
    hotkey: str,
    *,
    after: int = 0,
    stream_id: str | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = _OPTIONAL_TIMEOUT,
) -> VerdictsPage:
    """Poll final verdicts without delaying state acquisition on failure."""
    verdict_url = (
        f"{url}/miner-verdicts/{quote(hotkey, safe='')}"
        f"?after={max(0, int(after))}"
    )
    if stream_id is not None:
        verdict_url += f"&stream_id={quote(stream_id, safe='')}"
    return await _get_with_retry(
        verdict_url,
        VerdictsPage,
        client=client,
        timeout=timeout,
        retry_delays=(0.0,),
        jitter=False,
    )


@asynccontextmanager
async def monitor_submission_verdicts(url, hotkey, client, submitted):
    """Poll after the first accepted submission; own and cancel the HTTP task."""
    async def poll():
        cursor, stream_id = 0, None
        await submitted.wait()
        while True:
            try:
                page = await get_verdicts_page_v1(
                    url, hotkey, after=cursor, stream_id=stream_id, client=client,
                )
                if page.truncated:
                    logger.warning("verdict history gap or validator restart; resuming available feed")
                for verdict in page.verdicts:
                    logger.info("verdict window=%s root=%s accepted=%s reason=%s selected=%s",
                                verdict.window_n, verdict.merkle_root, verdict.accepted,
                                verdict.reason, verdict.selected_for_batch)
                cursor, stream_id = page.next_cursor, page.stream_id
            except EndpointNotFoundError:
                return
            except (SubmissionError, ValueError):
                logger.debug("optional verdict poll unavailable")
            await asyncio.sleep(2.0)

    task = asyncio.create_task(poll(), name="miner-verdicts")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
