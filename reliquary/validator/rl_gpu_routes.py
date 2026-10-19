"""Authenticated, bounded RL GPU admission API; no wallet or CUDA on validator CPU."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reliquary.protocol.gpu_attestation import gpu_request_binding
from reliquary.protocol.signatures import verify_hotkey_signature
from reliquary.validator.corpus_registration import NOT_REGISTERED
from reliquary.validator.gpu_attestation import AttestationError, VerifierUnavailable, strict_json
from reliquary.validator.gpu_registry import AdmissionError
from reliquary.validator.rl_lease_routes import _Refused, _read_within, ss58_address

logger = logging.getLogger(__name__)
HEADER = "X-Reliquary-Hotkey"
MAX_REQUEST_BYTES = 160 * 1024


class SignedGpuRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    miner_hotkey: str = Field(min_length=1, max_length=64)
    at: int = Field(ge=0)
    signature: str = Field(min_length=1, max_length=256)


class GpuDeclarations(BaseModel):
    """Signed by the hotkey, never promoted to NVIDIA-attested telemetry."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    gpu_name: str | None = Field(default=None, min_length=1, max_length=128)
    vram_bytes: int | None = Field(default=None, ge=1, le=2**50)
    compute_capability: str | None = Field(default=None, pattern=r"^[0-9]{1,2}\.[0-9]$", max_length=4)
    utilization_percent: float | None = Field(default=None, ge=0, le=100)
    temperature_c: float | None = Field(default=None, ge=-40, le=150)
    power_w: float | None = Field(default=None, ge=0, le=2000)


class GpuAttestRequest(SignedGpuRequest):
    challenge_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    bundle: list
    declared: GpuDeclarations = Field(default_factory=GpuDeclarations)


def build_gpu_router(registry, *, registration=None, prefix="/rl", verify=verify_hotkey_signature):
    router = APIRouter()
    workers = asyncio.Semaphore(4)

    def answer(content, status=200):
        return JSONResponse(content=content, status_code=status, headers={"Cache-Control": "no-store"})

    async def handle(http, model, action):
        # Limit reading/cryptographic verification together; do not queue unlimited EAT bodies.
        if workers.locked():
            return answer({"reason": "gpu_verifier_busy"}, 503)
        async with workers:
            try:
                hotkey = ss58_address(http.headers.get(HEADER, ""))
                if hotkey is None:
                    return answer({"reason": "hotkey_required"}, 401)
                if registration is not None:
                    try:
                        why = await registration(hotkey)
                    except Exception:
                        why = "unavailable"
                    if why is not None:
                        return answer({"reason": "hotkey_not_registered" if why == NOT_REGISTERED else
                                       "registration_unavailable"}, 403 if why == NOT_REGISTERED else 503)
                raw = await _read_within(http, MAX_REQUEST_BYTES if model is GpuAttestRequest else 4096, 10)
                payload = strict_json(raw)
                request = model.model_validate(payload)
                if request.miner_hotkey != hotkey:
                    return answer({"reason": "hotkey_mismatch"}, 403)
                if abs(registry.clock() - request.at) > 120:
                    return answer({"reason": "stale_request"}, 400)
                binding = gpu_request_binding(payload, validator_hotkey=registry.validator,
                                              run=registry.run, path=http.url.path)
                if not await asyncio.to_thread(verify, hotkey, binding, request.signature):
                    return answer({"reason": "bad_signature"}, 403)
                return answer(await asyncio.to_thread(action, request))
            except _Refused as exc:
                return exc.response
            except (ValidationError, AttestationError, TypeError, ValueError) as exc:
                if isinstance(exc, AdmissionError):
                    return answer({"reason": exc.reason}, exc.status)
                return answer({"reason": "malformed_gpu_request"}, 422)
            except VerifierUnavailable:
                return answer({"reason": "gpu_verifier_unavailable"}, 503)
            except Exception as exc:
                # No tokens, UEIDs, signatures, or attacker-controlled messages in logs.
                logger.error("GPU admission failed (%s)", type(exc).__name__)
                return answer({"reason": "gpu_admission_unavailable"}, 503)

    @router.post(f"{prefix}/gpu/challenge")
    async def challenge(http: Request):
        return await handle(http, SignedGpuRequest, lambda r: registry.challenge(r.miner_hotkey))

    @router.post(f"{prefix}/gpu/attest")
    async def attest(http: Request):
        return await handle(http, GpuAttestRequest, lambda r: registry.attest(
            r.miner_hotkey, r.challenge_id, r.bundle, r.declared.model_dump(exclude_none=True)))

    @router.post(f"{prefix}/gpu/status")
    async def status(http: Request):
        return await handle(http, SignedGpuRequest, lambda r: registry.status(r.miner_hotkey))

    @router.get(f"{prefix}/gpu/fleet")
    async def fleet():
        return answer(await asyncio.to_thread(registry.fleet))

    return router
