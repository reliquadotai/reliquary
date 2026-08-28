"""mTLS-only semantic signer API. No endpoint accepts arbitrary bytes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException

from reliquary.signer.backend import SignerBackend
from reliquary.signer.journal import (
    JournalConflict,
    JournalReplay,
    JournalUncertain,
    SignerJournal,
)
from reliquary.signer.protocol import (
    CheckpointSignRequest,
    CheckpointSignResponse,
    ServeAxonRequest,
    ServeAxonResponse,
    SetWeightsRequest,
    SetWeightsResponse,
    SignerHealth,
    checkpoint_payload,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignerPolicy:
    network: str
    netuid: int
    repo_id: str
    axon_ip: str | None = None
    axon_port: int | None = None


def _reserve_or_http(journal: SignerJournal, **kwargs):
    try:
        return journal.reserve(**kwargs)
    except JournalReplay as exc:
        raise HTTPException(status_code=409, detail="monotonic_replay_denied") from exc
    except JournalUncertain as exc:
        raise HTTPException(
            status_code=409, detail="operation_outcome_uncertain"
        ) from exc
    except JournalConflict as exc:
        raise HTTPException(status_code=409, detail="operation_conflict") from exc


def create_signer_app(
    *, policy: SignerPolicy, backend: SignerBackend, journal: SignerJournal
) -> FastAPI:
    app = FastAPI(
        title="Reliquary semantic signer",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    chain_lock = asyncio.Lock()

    def enforce_common(netuid: int) -> None:
        if int(netuid) != int(policy.netuid):
            raise HTTPException(status_code=403, detail="netuid_denied")

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> SignerHealth:
        return SignerHealth(
            signer_hotkey=backend.hotkey_address,
            network=policy.network,
            netuid=policy.netuid,
            repo_id=policy.repo_id,
            axon_ip=policy.axon_ip,
            axon_port=policy.axon_port,
        )

    @app.post("/v1/checkpoints/sign")
    async def sign_checkpoint(request: CheckpointSignRequest) -> CheckpointSignResponse:
        enforce_common(request.netuid)
        if request.repo_id != policy.repo_id:
            raise HTTPException(status_code=403, detail="repository_denied")
        reservation = _reserve_or_http(
            journal,
            operation_id=request.operation_id,
            kind="checkpoint-sign",
            payload_digest=request.operation_id,
            cursor_name="checkpoint_n",
            cursor_value=request.checkpoint_n,
        )
        if reservation.cached_response is not None:
            cached = dict(reservation.cached_response)
            cached["cached"] = True
            return CheckpointSignResponse.model_validate(cached)
        try:
            signature = await asyncio.to_thread(
                backend.sign_checkpoint,
                checkpoint_payload(request.checkpoint_n, request.revision),
            )
            response = CheckpointSignResponse(
                operation_id=request.operation_id,
                signer_hotkey=backend.hotkey_address,
                signature_hex=signature.hex(),
            )
            journal.complete(request.operation_id, response.model_dump())
            logger.info(
                "checkpoint signed operation=%s checkpoint=%d revision=%s",
                request.operation_id[:12],
                request.checkpoint_n,
                request.revision[:12],
            )
            return response
        except HTTPException:
            raise
        except Exception as exc:
            journal.fail_definite(request.operation_id, type(exc).__name__)
            logger.exception(
                "checkpoint signing failed operation=%s", request.operation_id[:12]
            )
            raise HTTPException(
                status_code=503, detail="checkpoint_signing_failed"
            ) from exc

    @app.post("/v1/weights/set")
    async def set_weights(request: SetWeightsRequest) -> SetWeightsResponse:
        enforce_common(request.netuid)
        reservation = _reserve_or_http(
            journal,
            operation_id=request.operation_id,
            kind="set-weights",
            payload_digest=request.operation_id,
            cursor_name="weight_epoch",
            cursor_value=request.epoch_id,
        )
        if reservation.cached_response is not None:
            cached = dict(reservation.cached_response)
            cached["cached"] = True
            return SetWeightsResponse.model_validate(cached)
        try:
            async with chain_lock:
                result = await backend.set_weights(
                    netuid=request.netuid,
                    uids=request.uids,
                    weights=request.weights,
                )
            response = SetWeightsResponse(
                operation_id=request.operation_id,
                signer_hotkey=backend.hotkey_address,
                accepted=result.accepted,
                message=result.message,
            )
            journal.complete(request.operation_id, response.model_dump())
            logger.info(
                "weights attempted operation=%s epoch=%d uids=%d accepted=%s",
                request.operation_id[:12],
                request.epoch_id,
                len(request.uids),
                result.accepted,
            )
            return response
        except Exception as exc:
            # A timeout may happen after chain acceptance. Never replay this epoch.
            journal.mark_uncertain(request.operation_id, type(exc).__name__)
            logger.exception(
                "weight outcome uncertain operation=%s", request.operation_id[:12]
            )
            raise HTTPException(
                status_code=503, detail="weight_outcome_uncertain"
            ) from exc

    @app.post("/v1/axon/serve")
    async def serve_axon(request: ServeAxonRequest) -> ServeAxonResponse:
        enforce_common(request.netuid)
        if policy.axon_ip is None or policy.axon_port is None:
            raise HTTPException(status_code=403, detail="axon_operation_disabled")
        if request.ip != policy.axon_ip or request.port != policy.axon_port:
            raise HTTPException(status_code=403, detail="axon_endpoint_denied")
        reservation = _reserve_or_http(
            journal,
            operation_id=request.operation_id,
            kind="serve-axon",
            payload_digest=request.operation_id,
        )
        if reservation.cached_response is not None:
            cached = dict(reservation.cached_response)
            cached["cached"] = True
            return ServeAxonResponse.model_validate(cached)
        try:
            async with chain_lock:
                result = await backend.serve_axon(
                    netuid=request.netuid,
                    ip=request.ip,
                    port=request.port,
                )
            response = ServeAxonResponse(
                operation_id=request.operation_id,
                signer_hotkey=backend.hotkey_address,
                accepted=result.accepted,
                message=result.message,
            )
            journal.complete(request.operation_id, response.model_dump())
            logger.info(
                "axon attempted operation=%s endpoint=%s:%d accepted=%s",
                request.operation_id[:12],
                request.ip,
                request.port,
                result.accepted,
            )
            return response
        except Exception as exc:
            journal.mark_uncertain(request.operation_id, type(exc).__name__)
            logger.exception(
                "axon outcome uncertain operation=%s", request.operation_id[:12]
            )
            raise HTTPException(
                status_code=503, detail="axon_outcome_uncertain"
            ) from exc

    return app
