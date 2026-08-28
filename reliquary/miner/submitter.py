"""HTTP client used by miners to push GRPO submissions to the validator.

V1 assumption: a single validator. Discovery returns the first axon advertised
by a hotkey holding `validator_permit`. Multi-validator routing is intentionally
out of scope here — see the GRPO refactor plan.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.signatures import (
    sign_envelope,
    sign_epoch_generation_intent,
    sign_precommit,
)
from reliquary.protocol.signatures import (
    verify_epoch_commitment_set_signature,
    verify_epoch_generation_intent_set_signature,
    verify_epoch_intent_signature,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    EpochCommitmentStatus,
    EpochGenerationIntentRequest,
    EpochGenerationIntentResponse,
    EpochGenerationIntentStatus,
    GrpoBatchState,
    RejectReason,
    RuntimeContract,
    SubmissionPrecommitRequest,
    SubmissionPrecommitResponse,
)
from reliquary.shared.checkpoint_epoch_market import (
    SignedGenerationIntentSet,
    canonical_signed_generation_intent_set_bytes,
    parse_signed_generation_intent_set,
)
from reliquary.shared.checkpoint_epoch import (
    CHECKPOINT_EPOCH_CAPABILITY_ID,
    EpochPlan,
    SignedEpochCommitmentSet,
    canonical_signed_commitment_set_bytes,
    commitment_set_sha256,
    generation_contract_sha256,
    manifest_sha256,
    parse_epoch_plan,
    parse_signed_commitment_set,
    validate_commitment_set_for_plan,
    validate_epoch_plan,
)
from reliquary.validator.checkpoint_epoch_runtime import (
    SignedEpochIntent,
    canonical_signed_intent_bytes,
    plan_from_intent,
    parse_signed_epoch_intent,
)
from reliquary.shared.runtime_fingerprint import bind_runtime_profile_nonce

logger = logging.getLogger(__name__)

# Retry configuration: 3 attempts, exponential backoff 1s / 2s / 4s.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_CHECKPOINT_EPOCH_BEACON_VERIFY_TIMEOUT_SECONDS = 12.0
# Default timeout is generous: the validator may need several seconds to verify
# a submission even in the async-queue path (the queue can back up under load).
# Miners running against slow links (Targon port-forward etc.) benefit further.
_DEFAULT_TIMEOUT = 60.0
_PRECOMMIT_HEADER = "X-Reliquary-Precommit"
_EPOCH_INTENT_HEADER = "X-Reliquary-Epoch-Intent"
_DRAND_BOUNDARY_SAFETY_SECONDS = 1.0
_DRAND_BOUNDARY_SETTLE_SECONDS = 0.05


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted."""


@dataclass(frozen=True, slots=True)
class FinalizedEpochCommitment:
    """Exact reveal bytes plus the compact commitment that binds them."""

    payload: bytes
    precommit: SubmissionPrecommitRequest


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
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE
                )
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
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.get(full_url, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if resp.status_code == 503:
                # No active window yet — caller's job to handle.
                raise SubmissionError(f"no active window at {full_url}")
            if resp.status_code == 404:
                raise SubmissionError(f"endpoint not found: {full_url}")
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
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


def _safe_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]


async def _verify_checkpoint_epoch_public_beacon(plan: EpochPlan) -> None:
    """Independently verify the exact public drand output in a plan."""
    from reliquary.infrastructure.drand import (
        get_beacon,
        get_current_chain,
        verify_beacon_signature,
    )

    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + _CHECKPOINT_EPOCH_BEACON_VERIFY_TIMEOUT_SECONDS
    )

    async def bounded(callback, *args, **kwargs):
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("checkpoint epoch beacon verification timed out")
        return await asyncio.wait_for(
            asyncio.to_thread(callback, *args, **kwargs),
            timeout=remaining,
        )

    try:
        chain_info = await bounded(get_current_chain)
        expected = plan.epoch_beacon
        if (
            str(chain_info["name"]) != expected.chain
            or str(chain_info["hash"]) != expected.chain_hash
        ):
            raise ValueError("configured drand chain differs from manifest")
        beacon = await bounded(
            get_beacon,
            round_id=str(expected.round),
            use_drand=True,
            use_fallback=False,
        )
        if (
            str(beacon.get("source")) != "drand"
            or str(beacon.get("chain")) != expected.chain
            or str(beacon.get("chain_hash")) != expected.chain_hash
            or int(beacon.get("round", -1)) != expected.round
            or str(beacon.get("randomness")) != expected.randomness
            or not beacon.get("signature")
        ):
            raise ValueError("public drand output differs from manifest")
        verified = await bounded(
            verify_beacon_signature,
            expected.chain_hash,
            expected.round,
            expected.randomness,
            str(beacon["signature"]),
        )
        if verified is not True:
            raise ValueError("public drand signature verification failed")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise SubmissionError(
            "checkpoint epoch public beacon validation failed"
        ) from exc


def finalize_checkpoint_epoch_commitment_v1(
    request: BatchSubmissionRequest,
    *,
    plan: EpochPlan,
    wallet: Any,
    drand_round: int,
    nonce: str | None = None,
) -> FinalizedEpochCommitment:
    """Bind prepared generation to fresh transport fields at commitment OPEN.

    The returned payload must be durably retained before its compact precommit
    is posted. Re-finalizing after an accepted precommit creates different
    reveal bytes and is therefore intentionally not a retry mechanism.
    """
    validate_epoch_plan(plan)
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("wallet must provide hotkey.sign()")
    wallet_hotkey = str(getattr(wallet.hotkey, "ss58_address", ""))
    if not wallet_hotkey or wallet_hotkey != request.miner_hotkey:
        raise SubmissionError("wallet hotkey differs from prepared payload")
    if (
        isinstance(drand_round, bool)
        or not isinstance(drand_round, int)
        or drand_round < 1
    ):
        raise ValueError("drand_round must be positive")
    if nonce is not None and (
        not isinstance(nonce, str) or not nonce or len(nonce) > 128
    ):
        raise ValueError("nonce must be a non-empty bounded string")
    environments = {rollout.env_name for rollout in request.rollouts}
    if len(environments) != 1:
        raise SubmissionError("submission must contain exactly one environment")
    environment = next(iter(environments))
    offset = request.window_start - plan.first_window
    if offset < 0 or offset >= plan.window_count:
        raise SubmissionError("prepared window is outside checkpoint epoch")
    epoch_window = plan.windows[offset]
    prompt_slices = {item.environment: item for item in epoch_window.prompt_slices}
    prompt_slice = prompt_slices.get(environment)
    if (
        epoch_window.window_number != request.window_start
        or request.checkpoint_hash != plan.checkpoint.revision
        or request.protocol_version != plan.protocol.protocol_version
        or request.generation_profile_id != plan.protocol.profile_id
        or prompt_slice is None
        or not prompt_slice.start <= request.prompt_idx < prompt_slice.stop
    ):
        raise SubmissionError("prepared payload differs from checkpoint epoch")
    randomness = epoch_window.generation_randomness
    for rollout in request.rollouts:
        beacon = rollout.commit.get("beacon")
        if not isinstance(beacon, dict) or beacon.get("randomness") != randomness:
            raise SubmissionError("prepared rollout has wrong epoch randomness")

    fresh_nonce = nonce if nonce is not None else os.urandom(16).hex()
    if request.runtime_fingerprint is not None:
        fresh_nonce = bind_runtime_profile_nonce(
            fresh_nonce,
            request.runtime_fingerprint.profile_hash,
        )
    if len(fresh_nonce) > 128:
        raise ValueError("runtime-bound nonce is too long")
    envelope_signature = sign_envelope(
        wallet=wallet,
        miner_hotkey=request.miner_hotkey,
        window_start=request.window_start,
        prompt_idx=request.prompt_idx,
        merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash,
        drand_round=int(drand_round),
        randomness=randomness,
        nonce=fresh_nonce,
        protocol_version=request.protocol_version,
        generation_profile_id=request.generation_profile_id,
    ).hex()
    finalized = request.model_copy(
        update={
            "drand_round": int(drand_round),
            "nonce": fresh_nonce,
            "envelope_signature": envelope_signature,
        }
    )
    wire_exclude = (
        {"generation_profile_id"} if not finalized.generation_profile_id else None
    )
    payload = finalized.model_dump_json(exclude=wire_exclude).encode("utf-8")
    precommit_fields = {
        "miner_hotkey": finalized.miner_hotkey,
        "window_start": finalized.window_start,
        "prompt_idx": finalized.prompt_idx,
        "merkle_root": finalized.merkle_root,
        "checkpoint_hash": finalized.checkpoint_hash,
        "environment": environment,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "drand_round": int(drand_round),
        "randomness": randomness,
        "protocol_version": finalized.protocol_version,
        "generation_profile_id": finalized.generation_profile_id,
        "nonce": fresh_nonce,
    }
    precommit = SubmissionPrecommitRequest(
        **{
            key: value for key, value in precommit_fields.items() if key != "randomness"
        },
        precommit_signature=sign_precommit(
            wallet=wallet,
            **precommit_fields,
        ).hex(),
    )
    return FinalizedEpochCommitment(payload=payload, precommit=precommit)


async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    wallet: Any | None = None,
    randomness: str = "",
    drand_round_fn: Callable[[], int] | None = None,
    epoch_generation_intent_id: str | None = None,
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
                        headers={
                            "Content-Type": "application/json",
                            **(
                                {_EPOCH_INTENT_HEADER: epoch_generation_intent_id}
                                if epoch_generation_intent_id is not None
                                else {}
                            ),
                        },
                        timeout=timeout,
                    )
                except (httpx.RequestError, httpx.TimeoutException) as exc:
                    last_exc = exc
                    if attempt < len(_RETRY_DELAYS):
                        await asyncio.sleep(delay)
                    continue
                if precommit_response.status_code == 404:
                    if epoch_generation_intent_id is not None:
                        raise SubmissionError(
                            "validator has no generation-intent submission path"
                        )
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
                return BatchSubmissionResponse(
                    accepted=False,
                    reason=RejectReason.WINDOW_NOT_ACTIVE,
                )
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
    timeout: float = _DEFAULT_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's current v2 GrpoBatchState.

    ``cooldown_prompts`` is per-env; pass ``env`` to read a specific env's
    cooldown set. Experimental concurrent epochs also require ``window`` for
    exact lane revalidation. Omitting both preserves legacy behavior.
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
    )


def finalize_checkpoint_epoch_generation_intent_v1(
    *,
    wallet: Any,
    operator_id: str,
    plan: EpochPlan,
    window_start: int,
    environment: str,
    prompt_idx: int,
    prompt_content_sha256: str,
    nonce: str | None = None,
) -> EpochGenerationIntentRequest:
    """Sign one cheap prompt claim before generation ticket selection."""
    epoch_window = next(
        (item for item in plan.windows if item.window_number == window_start),
        None,
    )
    if epoch_window is None or environment not in {
        item.environment for item in epoch_window.prompt_slices
    }:
        raise ValueError("generation intent lane is outside the epoch plan")
    miner_hotkey = str(wallet.hotkey.ss58_address)
    generation_nonce = nonce or os.urandom(16).hex()
    fields = {
        "miner_hotkey": miner_hotkey,
        "operator_id": operator_id,
        "epoch_id": plan.epoch_id,
        "manifest_sha256": manifest_sha256(plan),
        "window_start": window_start,
        "environment": environment,
        "prompt_idx": prompt_idx,
        "prompt_content_sha256": prompt_content_sha256,
        "checkpoint_hash": plan.checkpoint.revision,
        "generation_randomness": epoch_window.generation_randomness,
        "protocol_version": plan.protocol.protocol_version,
        "generation_profile_id": plan.protocol.profile_id,
        "nonce": generation_nonce,
    }
    return EpochGenerationIntentRequest(
        **fields,
        intent_signature=sign_epoch_generation_intent(
            wallet=wallet, **fields
        ).hex(),
    )


async def post_checkpoint_epoch_generation_intent_v1(
    url: str,
    intent: EpochGenerationIntentRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> EpochGenerationIntentResponse:
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.post(
            f"{url}/checkpoint-epoch/generation-intents",
            content=intent.model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"generation intent HTTP {response.status_code}: "
                f"{_safe_detail(response)}"
            )
        result = EpochGenerationIntentResponse.model_validate(response.json())
        if result.accepted and result.intent_id is None:
            raise SubmissionError("accepted generation intent omitted intent_id")
        return result
    finally:
        if own_client:
            await http.aclose()


async def get_checkpoint_epoch_generation_intent_status_v1(
    url: str,
    intent_id: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> EpochGenerationIntentStatus:
    return await _get_with_retry(
        f"{url}/checkpoint-epoch/generation-intents/{quote(intent_id, safe='')}",
        EpochGenerationIntentStatus,
        client=client,
        timeout=timeout,
    )


async def get_checkpoint_epoch_generation_intent_set_v1(
    url: str,
    *,
    expected_validator_hotkey: str,
    expected_intent_set_sha256: str,
    expected_plan: EpochPlan,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SignedGenerationIntentSet:
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(
            f"{url}/checkpoint-epoch/generation-intent-set", timeout=timeout
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"generation intent set HTTP {response.status_code}: "
                f"{_safe_detail(response)}"
            )
        publication = parse_signed_generation_intent_set(response.content)
        if (
            publication.intent_set_sha256 != expected_intent_set_sha256
            or publication.intent_set.epoch_id != expected_plan.epoch_id
            or publication.intent_set.manifest_sha256
            != manifest_sha256(expected_plan)
        ):
            raise SubmissionError("generation intent set differs from live epoch")
        if not verify_epoch_generation_intent_set_signature(
            publication,
            expected_validator_hotkey=expected_validator_hotkey,
        ):
            raise SubmissionError("generation intent set signature is invalid")
        if response.content != canonical_signed_generation_intent_set_bytes(
            publication
        ):
            raise SubmissionError("generation intent set is not canonical")
        return publication
    finally:
        if own_client:
            await http.aclose()


async def post_checkpoint_epoch_commitment_v1(
    url: str,
    commitment: SubmissionPrecommitRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SubmissionPrecommitResponse:
    """Post only the compact signed commitment; never upload its payload."""
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.post(
            f"{url}/submit/precommit",
            content=commitment.model_dump_json().encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"commitment HTTP {response.status_code}: {_safe_detail(response)}"
            )
        result = SubmissionPrecommitResponse.model_validate(response.json())
        if result.accepted and not result.receipt_id:
            raise SubmissionError("accepted commitment omitted receipt_id")
        return result
    finally:
        if own_client:
            await http.aclose()


async def get_checkpoint_epoch_commitment_status_v1(
    url: str,
    receipt_id: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> EpochCommitmentStatus:
    return await _get_with_retry(
        f"{url}/checkpoint-epoch/commitments/{quote(receipt_id, safe='')}",
        EpochCommitmentStatus,
        client=client,
        timeout=timeout,
    )


async def get_checkpoint_epoch_commitment_set_v1(
    url: str,
    *,
    expected_validator_hotkey: str,
    expected_commitment_set_sha256: str | None = None,
    expected_plan: EpochPlan | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SignedEpochCommitmentSet:
    """Fetch and verify the exact set signed before admission beacon A."""
    if not expected_validator_hotkey:
        raise ValueError("expected validator hotkey is required")
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(
            f"{url}/checkpoint-epoch/commitment-set",
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"commitment-set HTTP {response.status_code}: {_safe_detail(response)}"
            )
        try:
            publication = parse_signed_commitment_set(response.content)
        except (TypeError, ValueError) as exc:
            raise SubmissionError("invalid checkpoint epoch commitment set") from exc
        digest = publication.commitment_set_sha256
        if (
            expected_commitment_set_sha256 is not None
            and digest != expected_commitment_set_sha256
        ):
            raise SubmissionError("checkpoint epoch commitment set changed")
        if digest != commitment_set_sha256(publication.commitment_set):
            raise SubmissionError("checkpoint epoch commitment-set hash differs")
        etag = response.headers.get("ETag", "")
        if etag != f'"{digest}"':
            raise SubmissionError("checkpoint epoch commitment-set ETag differs")
        if not verify_epoch_commitment_set_signature(
            publication,
            expected_validator_hotkey=expected_validator_hotkey,
        ):
            raise SubmissionError("checkpoint epoch commitment-set signature failed")
        if expected_plan is not None:
            try:
                validate_commitment_set_for_plan(
                    publication.commitment_set,
                    expected_plan,
                )
            except (TypeError, ValueError) as exc:
                raise SubmissionError(
                    "checkpoint epoch commitment set differs from plan"
                ) from exc
        if response.content != canonical_signed_commitment_set_bytes(publication):
            raise SubmissionError("checkpoint epoch commitment set is not canonical")
        return publication
    finally:
        if own_client:
            await http.aclose()


async def get_checkpoint_epoch_intent_v1(
    url: str,
    *,
    expected_validator_hotkey: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> SignedEpochIntent:
    """Fetch the validator-signed intent that existed before epoch beacon."""
    if not expected_validator_hotkey:
        raise ValueError("expected validator hotkey is required")
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(
            f"{url}/checkpoint-epoch/intent",
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"epoch-intent HTTP {response.status_code}: {_safe_detail(response)}"
            )
        try:
            publication = parse_signed_epoch_intent(response.content)
        except (TypeError, ValueError) as exc:
            raise SubmissionError("invalid signed checkpoint epoch intent") from exc
        if response.headers.get("ETag", "") != (f'"{publication.intent_sha256}"'):
            raise SubmissionError("checkpoint epoch intent ETag differs")
        if not verify_epoch_intent_signature(
            publication,
            expected_validator_hotkey=expected_validator_hotkey,
        ):
            raise SubmissionError("checkpoint epoch intent signature failed")
        if response.content != canonical_signed_intent_bytes(publication):
            raise SubmissionError("checkpoint epoch intent is not canonical")
        return publication
    finally:
        if own_client:
            await http.aclose()


async def reveal_checkpoint_epoch_payload_v1(
    url: str,
    *,
    receipt_id: str,
    payload: bytes,
    plan: EpochPlan,
    expected_validator_hotkey: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BatchSubmissionResponse:
    """Reveal one locally retained payload after selection authorizes it."""
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        request = BatchSubmissionRequest.model_validate_json(payload)
        environments = {rollout.env_name for rollout in request.rollouts}
        if len(environments) != 1:
            raise SubmissionError("epoch reveal payload has mixed environments")
        environment = next(iter(environments))
        state = await get_window_state_v2(
            url,
            env=environment,
            window=request.window_start,
            client=http,
            timeout=timeout,
        )
        offset = request.window_start - plan.first_window
        if offset < 0 or offset >= plan.window_count:
            raise SubmissionError("epoch reveal window is outside the plan")
        epoch_window = plan.windows[offset]
        slices = {
            item.environment: item for item in epoch_window.prompt_slices
        }
        prompt_slice = slices.get(environment)
        expected_manifest = manifest_sha256(plan)

        def exact_live_state(candidate) -> bool:
            candidate_contract = candidate.generation_contract or {}
            return bool(
                candidate.state.value == "open"
                and candidate.checkpoint_epoch_phase == "reveal"
                and candidate.checkpoint_epoch_id == plan.epoch_id
                and candidate.checkpoint_epoch_manifest_sha256 == expected_manifest
                and candidate.window_n == request.window_start
                and candidate.checkpoint_n == plan.checkpoint.number
                and candidate.checkpoint_repo_id == plan.checkpoint.repo_id
                and candidate.checkpoint_revision
                == request.checkpoint_hash
                == plan.checkpoint.revision
                and candidate.protocol_version == plan.protocol.protocol_version
                and candidate.generation_profile_id == plan.protocol.profile_id
                and generation_contract_sha256(candidate_contract)
                == plan.protocol.generation_contract_sha256
                and candidate.randomness == epoch_window.generation_randomness
                and candidate.checkpoint_epoch_commitment_set_sha256 is not None
                and candidate.checkpoint_epoch_commitment_root is not None
                and request.prompt_idx not in candidate.cooldown_prompts
                and prompt_slice is not None
                and prompt_slice.start <= request.prompt_idx < prompt_slice.stop
            )

        if not exact_live_state(state):
            raise SubmissionError("live state no longer matches epoch reveal binding")
        status = await get_checkpoint_epoch_commitment_status_v1(
            url,
            receipt_id,
            client=http,
            timeout=timeout,
        )
        if status.status != "selected":
            return BatchSubmissionResponse(
                accepted=False,
                reason=RejectReason.REVEAL_NOT_SELECTED,
            )
        if (
            status.commitment_set_sha256 != state.checkpoint_epoch_commitment_set_sha256
            or status.commitment_root != state.checkpoint_epoch_commitment_root
        ):
            raise SubmissionError("commitment status differs from live frozen set")
        publication = await get_checkpoint_epoch_commitment_set_v1(
            url,
            expected_validator_hotkey=expected_validator_hotkey,
            expected_commitment_set_sha256=(
                state.checkpoint_epoch_commitment_set_sha256
            ),
            expected_plan=plan,
            client=http,
            timeout=timeout,
        )
        commitment_set = publication.commitment_set
        if (
            commitment_set.epoch_id != plan.epoch_id
            or commitment_set.manifest_sha256 != manifest_sha256(plan)
            or commitment_set.commitment_root != state.checkpoint_epoch_commitment_root
            or status.admission_beacon_round is None
            or state.checkpoint_epoch_admission_beacon_round is None
            or status.admission_beacon_round
            != state.checkpoint_epoch_admission_beacon_round
            or state.checkpoint_epoch_admission_beacon_round
            <= commitment_set.commitment_close_round
        ):
            raise SubmissionError("signed commitment set differs from epoch state")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        matching = [
            record
            for record in commitment_set.commitments
            if record.receipt_id == receipt_id
        ]
        if len(matching) != 1:
            raise SubmissionError("commitment receipt is absent or duplicated")
        record = matching[0]
        if (
            record.miner_hotkey != request.miner_hotkey
            or record.window_number != request.window_start
            or record.environment != environment
            or record.prompt_idx != request.prompt_idx
            or record.payload_sha256 != payload_sha256
        ):
            raise SubmissionError("payload differs from signed commitment record")
        release_state = await get_window_state_v2(
            url,
            env=environment,
            window=request.window_start,
            client=http,
            timeout=timeout,
        )
        if (
            not exact_live_state(release_state)
            or release_state.checkpoint_epoch_commitment_set_sha256
            != publication.commitment_set_sha256
            or release_state.checkpoint_epoch_commitment_root
            != commitment_set.commitment_root
            or release_state.checkpoint_epoch_admission_beacon_round
            != status.admission_beacon_round
        ):
            raise SubmissionError("epoch binding changed before payload release")
        response = await http.post(
            f"{url}/submit",
            content=payload,
            headers={
                "Content-Type": "application/json",
                _PRECOMMIT_HEADER: receipt_id,
            },
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise SubmissionError(
                f"reveal HTTP {response.status_code}: {_safe_detail(response)}"
            )
        return BatchSubmissionResponse.model_validate(response.json())
    finally:
        if own_client:
            await http.aclose()


async def get_runtime_contract_v1(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> RuntimeContract:
    """Discover optional runtime telemetry without changing legacy `/state`."""
    return await _get_with_retry(
        f"{url}/runtime-contract",
        RuntimeContract,
        client=client,
        timeout=timeout,
    )


async def get_checkpoint_epoch_plan_v1(
    url: str,
    *,
    expected_validator_hotkey: str,
    expected_manifest_sha256: str | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> EpochPlan:
    """Fetch canonical plan bytes, using live state or the endpoint ETag.

    Omitting ``expected_manifest_sha256`` is useful during the advertised
    warm-up, before an ordinary window exists. Local release must still match
    the same digest in the later exact-OPEN state.
    """
    if not expected_validator_hotkey:
        raise ValueError("expected validator hotkey is required")
    if expected_manifest_sha256 is not None and (
        len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        raise ValueError("expected manifest hash must be lowercase SHA-256")

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    last_error: Exception | None = None
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                response = await http.get(
                    f"{url}/checkpoint-epoch",
                    timeout=timeout,
                )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if response.status_code == 404:
                raise SubmissionError("checkpoint epoch capability is disabled")
            if 400 <= response.status_code < 500:
                raise SubmissionError(
                    f"HTTP {response.status_code}: {_safe_detail(response)}"
                )
            if response.status_code >= 500:
                last_error = SubmissionError(f"HTTP {response.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue

            etag = response.headers.get("ETag", "")
            advertised_digest = etag[1:-1] if (
                len(etag) == 66 and etag.startswith('"') and etag.endswith('"')
            ) else ""
            if expected_manifest_sha256 is None:
                if (
                    len(advertised_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in advertised_digest
                    )
                ):
                    raise SubmissionError(
                        "checkpoint epoch endpoint omitted a valid ETag"
                    )
                expected_digest = advertised_digest
            else:
                expected_digest = expected_manifest_sha256
            if advertised_digest != expected_digest:
                raise SubmissionError(
                    "checkpoint epoch ETag does not match live state"
                )
            try:
                plan = parse_epoch_plan(
                    response.content,
                    expected_manifest_sha256=expected_digest,
                )
            except (TypeError, ValueError) as exc:
                raise SubmissionError(
                    "checkpoint epoch manifest validation failed"
                ) from exc
            if plan.experimental_capability_id != CHECKPOINT_EPOCH_CAPABILITY_ID:
                raise SubmissionError("unsupported checkpoint epoch capability")
            publication = await get_checkpoint_epoch_intent_v1(
                url,
                expected_validator_hotkey=expected_validator_hotkey,
                client=http,
                timeout=timeout,
            )
            try:
                expected_plan = plan_from_intent(
                    publication.intent,
                    beacon=plan.epoch_beacon,
                )
            except (TypeError, ValueError) as exc:
                raise SubmissionError(
                    "checkpoint epoch plan differs from signed intent"
                ) from exc
            if expected_plan != plan:
                raise SubmissionError(
                    "checkpoint epoch plan differs from signed intent"
                )
            await _verify_checkpoint_epoch_public_beacon(plan)
            return plan
        raise SubmissionError(f"all retries failed: {last_error}")
    finally:
        if own_client:
            await http.aclose()
