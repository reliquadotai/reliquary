"""Out-of-zone audits on the remote proof worker (rl-async-design.md §4.4; implementation map §9).

The V1 controller has no GPU. The proof worker keeps a second, in-process verify model at the latest published
async checkpoint and runs ``rl_audit``'s forward on it; the controller drains its audit queue through it.

* ``POST /v1/rl/checkpoint`` (``RlAuditCheckpoint``): the controller forwards the board's answer (the signed index
  and its presigned URLs, as ``GET /rl/checkpoint`` serves it). The worker checks the index signature against the
  validator hotkey it is configured with, rebuilds the newest version with ``CheckpointSync`` (deltas from its
  own version, else an anchor; every object hash-checked) and loads it. Answers the version it holds.
* ``POST /v1/rl/audit`` (``RlAuditRequest``): one ``SubmittedGroup``'s tokens and lease draw params (nonce,
  prompt, k): the worker recomputes the uniforms (``protocol.rl_draw``), nothing else is trusted for the draw. 409
  when the worker does not hold the version the controller expects (it restarted): the controller re-adopts once
  and retries.

Same listener, mTLS and request bounds as the proof routes (``remote_proof_server``); the controller reuses the
proof executor's client configuration (``RELIQUARY_PROOF_EXECUTOR_URL``, ``RELIQUARY_PROOF_TLS_*``,
``RELIQUARY_PROOF_EXPECTED_WORKER_ID``). Worker side: ``RELIQUARY_PROOF_RL_AUDIT_DEVICE`` (e.g. ``cuda:0``; unset:
the routes are not served) and ``RELIQUARY_PROOF_RL_VALIDATOR_HOTKEY`` (the hotkey that signs the index).
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import os
import ssl
import threading
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response
from pydantic import Field, model_validator

from reliquary.validator.remote_proof_protocol import (
    GZIP_LEVEL, GZIP_MIN_BYTES, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_TOKENS, Count, Name, Probability,
    WireModel, canonical_bytes, digest,
)
from reliquary.validator.rl_buffer import SubmittedGroup
from reliquary.validator.rl_submit import AuditResult

logger = logging.getLogger(__name__)

AUDIT_DEVICE_ENV = "RELIQUARY_PROOF_RL_AUDIT_DEVICE"
VALIDATOR_HOTKEY_ENV = "RELIQUARY_PROOF_RL_VALIDATOR_HOTKEY"
WORK_DIR_ENV = "RELIQUARY_PROOF_RL_AUDIT_DIR"
MAX_AUDIT_ROLLOUTS = 64
TokenId = Annotated[int, Field(ge=0, lt=2**31)]
Hex = Annotated[str, Field(min_length=2, max_length=256, pattern=r"^[0-9a-f]+$")]
Span = Annotated[list[Count], Field(min_length=2, max_length=2)]   # [start, end), completion-relative


class RlAuditCheckpoint(WireModel):
    worker_id: Name
    answer: dict          # the board's answer: index, signature, validator_hotkey, urls, urls_expire_at


class RlAuditState(WireModel):
    worker_id: Name
    version: Count | None = None


class RlAuditRequest(WireModel):
    worker_id: Name
    version: Count                        # the checkpoint the controller expects the audit to run on (the latest)
    lease_id: Name
    environment: Name
    prompt_idx: Count
    checkpoint: Count                     # the lease's k: keys the draw
    nonce: Hex
    prompt_tokens: list[TokenId] = Field(min_length=1, max_length=MAX_TOKENS)
    completions: list[list[TokenId]] = Field(min_length=1, max_length=MAX_AUDIT_ROLLOUTS)
    force_spans: list[Span | None]

    @model_validator(mode="after")
    def bounded(self):
        if len(self.force_spans) != len(self.completions):
            raise ValueError("one force span (or null) per completion")
        if any(len(self.prompt_tokens) + len(c) > MAX_TOKENS for c in self.completions):
            raise ValueError("a rollout exceeds the token bound")
        if any(s is not None and not s[0] <= s[1] <= len(c) for s, c in zip(self.force_spans, self.completions)):
            raise ValueError("a force span lies outside its completion")
        return self

    @classmethod
    def of(cls, group: SubmittedGroup, *, worker_id: str, version: int) -> "RlAuditRequest":
        return cls(worker_id=worker_id, version=int(version), lease_id=group.lease_id,
                   environment=group.environment, prompt_idx=int(group.prompt_idx), checkpoint=int(group.checkpoint),
                   nonce=group.nonce, prompt_tokens=list(group.prompt_tokens),
                   completions=[list(c) for c in group.completions],
                   force_spans=[None if s is None else [int(s[0]), int(s[1])] for s in group.force_spans])

    def group(self) -> SubmittedGroup:
        """The group the audit forward reads (pay and grading fields play no part in it)."""
        n = len(self.completions)
        return SubmittedGroup(lease_id=self.lease_id, hotkey="", environment=self.environment,
                              prompt_idx=self.prompt_idx, checkpoint=self.checkpoint, nonce=self.nonce,
                              arrived_at=0.0, rewards=(0.0,) * n, truncated=(False,) * n,
                              prompt_tokens=tuple(self.prompt_tokens),
                              completions=tuple(tuple(c) for c in self.completions),
                              force_spans=tuple(None if s is None else tuple(s) for s in self.force_spans))


class RlAuditResponse(WireModel):
    worker_id: Name
    version: Count
    lease_id: Name
    request_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    passed: bool | None
    failure_class: Annotated[str, Field(max_length=64)] | None = None
    rollout_match: list[Probability] = Field(max_length=MAX_AUDIT_ROLLOUTS)
    suspect: bool = False                 # inconclusive because of the group (``AuditResult.suspect``)


class CheckpointMismatch(RuntimeError):
    """The worker does not hold the checkpoint the audit was asked to run on."""


# --- worker side --------------------------------------------------------------------------------------------------
class RlAuditBackend:
    """The worker's verify model at the latest async checkpoint (the controller's ``AuditModel``, remotely)."""

    def __init__(self, model: Any, sync: Any, *, forward=None):
        from reliquary.validator.rl_audit import audit_forward

        self.model, self.sync = model, sync
        self.version: int | None = None
        self._state = None
        self._forward = (forward or audit_forward)(model)
        self._model_lock = threading.Lock()
        self._sync_lock = threading.Lock()

    def adopt(self, answer: dict) -> int | None:
        """Bring the model to the answer's latest version (a no-op when it holds it)."""
        if not self._sync_lock.acquire(blocking=False):
            raise BlockingIOError("a checkpoint sync is already running")
        try:
            latest = int(answer["index"]["latest"]["version"])
            if self.version is not None and latest <= self.version:
                return self.version
            urls = answer.get("urls")
            if not isinstance(urls, dict) or any(urlsplit(str(u)).scheme != "https" for u in urls.values()):
                raise ValueError("checkpoint object URLs must be https")
            state, version, _ = self.sync.sync(answer, self._state, self.version)
            with self._model_lock:
                self.model.load_state_dict(state)
                self._state, self.version = state, int(version)
            logger.info("rl audit model at checkpoint %d", version)
            return self.version
        finally:
            self._sync_lock.release()

    def audit(self, request: RlAuditRequest) -> AuditResult:
        with self._model_lock:
            if self.version != request.version:
                raise CheckpointMismatch(f"holds {self.version}, asked {request.version}")
            return self._forward(request.group())


def mount_rl_audit(app, backend: RlAuditBackend, *, worker_id: str, read) -> None:
    """Serve the audit routes on the proof app (``read``: its bounded JSON body reader)."""
    def check(value) -> None:
        if value.worker_id != worker_id:
            raise HTTPException(409, "worker identity mismatch")

    def respond(value) -> Response:
        return Response(canonical_bytes(value.model_dump()), media_type="application/json")

    @app.post("/v1/rl/checkpoint")
    async def rl_checkpoint(request: Request):
        value = await read(request, RlAuditCheckpoint)
        check(value)
        try:
            version = await asyncio.to_thread(backend.adopt, value.answer)
        except BlockingIOError as exc:
            raise HTTPException(503, "checkpoint sync already running") from exc
        except Exception as exc:
            logger.warning("rl audit checkpoint sync failed: %s: %s", type(exc).__name__, exc)
            raise HTTPException(503, "checkpoint sync failed") from exc
        return respond(RlAuditState(worker_id=worker_id, version=version))

    @app.post("/v1/rl/audit")
    async def rl_audit(request: Request):
        value = await read(request, RlAuditRequest)
        check(value)
        try:
            result = await asyncio.to_thread(backend.audit, value)
        except CheckpointMismatch as exc:
            raise HTTPException(409, "rl checkpoint mismatch") from exc
        except Exception as exc:
            logger.warning("rl audit failed lease=%s: %s", value.lease_id[:12], type(exc).__name__)
            raise HTTPException(503, "rl audit failed") from exc
        return respond(RlAuditResponse(
            worker_id=worker_id, version=value.version, lease_id=value.lease_id,
            request_sha256=digest(value.model_dump()), passed=result.passed, failure_class=result.failure_class,
            rollout_match=[float(m) for m in result.rollout_match], suspect=bool(result.suspect)))


def rl_audit_backend_from_environment(*, checkpoint: str, load_kwargs: dict) -> RlAuditBackend | None:
    """The worker's backend when ``RELIQUARY_PROOF_RL_AUDIT_DEVICE`` is set (None otherwise)."""
    device = os.environ.get(AUDIT_DEVICE_ENV, "").strip()
    if not device:
        return None
    hotkey = os.environ.get(VALIDATOR_HOTKEY_ENV, "").strip()
    if not hotkey:
        raise RuntimeError(f"{VALIDATOR_HOTKEY_ENV} is required with {AUDIT_DEVICE_ENV}")
    import tempfile

    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.miner.ckpt_sync import CheckpointSync
    from reliquary.shared.modeling import load_text_generation_model

    work_dir = Path(os.environ.get(WORK_DIR_ENV, "").strip() or Path(tempfile.gettempdir()) / "reliquary-rl-audit")
    model = load_text_generation_model(checkpoint, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
                                       **load_kwargs).to(device).eval()
    return RlAuditBackend(model, CheckpointSync(validator_hotkey=hotkey, work_dir=work_dir))


# --- controller side ----------------------------------------------------------------------------------------------
class RemoteAuditModel:
    """The controller's ``AuditModel`` over the proof worker: same ``follow_once`` / ``audit`` interface."""

    def __init__(self, *, base_url: str, worker_id: str, tls: ssl.SSLContext | None = None, client=None,
                 request_timeout: float = 300.0, sync_timeout: float = 900.0):
        import httpx

        url = urlsplit(base_url)
        if client is None and (url.scheme != "https" or not url.hostname or url.path not in {"", "/"}):
            raise ValueError("the proof endpoint requires a bare HTTPS origin")
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), verify=tls, trust_env=False,
                                              follow_redirects=False)
        self.worker_id = worker_id
        self.version: int | None = None
        self._answer: dict | None = None
        self._timeouts = {"/v1/rl/checkpoint": sync_timeout, "/v1/rl/audit": request_timeout}

    @classmethod
    def from_environment(cls) -> "RemoteAuditModel":
        from reliquary import constants as c

        values = {}
        for name in ("RELIQUARY_PROOF_EXECUTOR_URL", "RELIQUARY_PROOF_TLS_CA", "RELIQUARY_PROOF_TLS_CERT",
                     "RELIQUARY_PROOF_TLS_KEY", "RELIQUARY_PROOF_EXPECTED_WORKER_ID"):
            values[name] = os.environ.get(name, "").strip()
            if not values[name]:
                raise RuntimeError(f"{name} is required for remote audits")
        tls = ssl.create_default_context(cafile=values["RELIQUARY_PROOF_TLS_CA"])
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(values["RELIQUARY_PROOF_TLS_CERT"], values["RELIQUARY_PROOF_TLS_KEY"])
        return cls(base_url=values["RELIQUARY_PROOF_EXECUTOR_URL"],
                   worker_id=values["RELIQUARY_PROOF_EXPECTED_WORKER_ID"], tls=tls,
                   request_timeout=c.PROOF_WORKER_REQUEST_TIMEOUT_SECONDS,
                   sync_timeout=c.PROOF_WORKER_RELOAD_TIMEOUT_SECONDS)

    def _post(self, path: str, value, response_cls):
        raw = canonical_bytes(value.model_dump())
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("rl audit request exceeds the transport bound")
        headers = {"content-type": "application/json"}
        if len(raw) >= GZIP_MIN_BYTES:
            raw, headers["content-encoding"] = gzip.compress(raw, GZIP_LEVEL), "gzip"
        response = self._client.post(path, content=raw, headers=headers, timeout=self._timeouts[path])
        if response.status_code == 409 and path == "/v1/rl/audit":
            raise CheckpointMismatch(response.text[:200])
        response.raise_for_status()
        result = response_cls.read(response.content, limit=MAX_RESPONSE_BYTES)
        if result.worker_id != self.worker_id:
            raise RuntimeError("unexpected proof worker identity")
        return result

    def _adopt(self, answer: dict) -> int | None:
        state = self._post("/v1/rl/checkpoint", RlAuditCheckpoint(worker_id=self.worker_id, answer=answer),
                           RlAuditState)
        self.version = state.version
        return self.version

    async def follow_once(self, board) -> int | None:
        answer = None if board is None else await board.answer()
        if not answer:
            return None
        self._answer = answer                     # fresh URLs, for a re-adoption after a worker restart
        latest = int(answer["index"]["latest"]["version"])
        if self.version is not None and latest <= self.version:
            return None
        version = await asyncio.to_thread(self._adopt, answer)
        logger.info("remote audit model at checkpoint %s", version)
        return version

    def forward(self, group: SubmittedGroup) -> AuditResult:
        request = RlAuditRequest.of(group, worker_id=self.worker_id, version=int(self.version))
        try:
            result = self._post("/v1/rl/audit", request, RlAuditResponse)
        except CheckpointMismatch:
            if self._answer is None or self._adopt(self._answer) != request.version:
                raise
            result = self._post("/v1/rl/audit", request, RlAuditResponse)
        if (result.lease_id, result.version, result.request_sha256) != (
                request.lease_id, request.version, digest(request.model_dump())):
            raise RuntimeError("rl audit response is not bound to the request")
        if len(result.rollout_match) not in (0, len(group.completions)):
            raise RuntimeError("rl audit response has the wrong number of rollouts")
        return AuditResult(result.passed, result.failure_class, tuple(result.rollout_match),
                           suspect=result.passed is None and result.suspect)

    def audit(self, latest: int | None, submissions, limit: int) -> int | None:
        """Drain audits when the worker holds ``latest``; None (nothing run) while it lags."""
        if latest is None or self.version != latest:
            return None
        return submissions.run_audits(self.forward, limit)


__all__ = [
    "AUDIT_DEVICE_ENV", "CheckpointMismatch", "RemoteAuditModel", "RlAuditBackend", "RlAuditCheckpoint",
    "RlAuditRequest", "RlAuditResponse", "RlAuditState", "VALIDATOR_HOTKEY_ENV", "mount_rl_audit",
    "rl_audit_backend_from_environment",
]
