"""mTLS GPU endpoint over the existing local ProofWorkerPool and kernels."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import os
import ssl
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response

from reliquary.validator.proof_worker import ProofWorkerUnavailable
from reliquary.validator.remote_proof_protocol import (
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, AdoptionRequest, CheckpointBinding,
    MAX_PROOF_PIPELINE_DEPTH, ProofHealth, ProofRequest, ProofResponse, ProofValues, SlotState,
    canonical_bytes, digest, transport_hash, ProofBatchRequest, ProofBatchResponse,
    MAX_BATCH_REQUEST_BYTES, MAX_BATCH_RESPONSE_BYTES,
)

logger = logging.getLogger(__name__)


def prune_checkpoint_cache(repo_id, protected):
    """Bound public weight downloads; never touch another repo or shared blobs."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE
    from pathlib import Path

    if not Path(HF_HUB_CACHE).is_dir():
        return
    cache = scan_cache_dir(HF_HUB_CACHE)
    for repo in cache.repos:
        if repo.repo_id != repo_id or repo.repo_type != "model":
            continue
        recent = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)[:2]
        keep = set(protected) | {r.commit_hash for r in recent}
        obsolete = [r.commit_hash for r in repo.revisions if r.commit_hash not in keep]
        if obsolete:
            strategy = cache.delete_revisions(*obsolete)
            logger.info("proof_cache_prune revisions=%d bytes=%d", len(obsolete),
                        strategy.expected_freed_size)
            strategy.execute()


class ProofBackend:
    """The only production backend: reuse the deployed batch=1 proof kernels."""

    def __init__(self, pool):
        self.pool = pool
        self.devices = pool.devices
        self._descriptions = {}
        self._anchor_revision = None

    def describe(self, device):
        description = self.pool.describe(device)
        self._descriptions[device] = description
        return description

    def health_descriptions(self):
        # Never queue a health call behind a running proof. Liveness comes
        # from the actual process; checkpoint metadata was read from that
        # process at adoption and before every proof.
        if any(not self.pool.is_alive(device) for device in self.devices):
            raise ProofWorkerUnavailable("a proof process exited")
        return [self._descriptions.get(d) or self.describe(d) for d in self.devices]

    def adopt(self, checkpoint: CheckpointBinding):
        from huggingface_hub import HfApi, hf_hub_download
        from pathlib import Path
        from reliquary.validator.resume import checkpoint_n_from_commit_title
        from reliquary.validator.checkpoint_profile import (
            CHECKPOINT_PROFILE_NAME, validate_checkpoint_profile,
        )

        # Metadata and weights are resolved only from this immutable public
        # repo/OID. A controller path is never interpreted on the GPU host.
        commits = HfApi(token=False).list_repo_commits(repo_id=checkpoint.repo_id)
        matches = [c for c in commits if c.commit_id == checkpoint.revision]
        if len(matches) != 1 or checkpoint_n_from_commit_title(matches[0].title) != checkpoint.checkpoint_n:
            raise ValueError("checkpoint number does not match published revision")
        if sum(checkpoint_n_from_commit_title(c.title) == checkpoint.checkpoint_n for c in commits) != 1:
            raise ValueError("published checkpoint number is ambiguous")
        path = hf_hub_download(checkpoint.repo_id, CHECKPOINT_PROFILE_NAME,
                               revision=checkpoint.revision, token=False)
        profile = validate_checkpoint_profile(Path(path).parent, required=True)
        for key in ("profile_id", "generation_contract_sha256", "training_run_id"):
            if profile.get(key) != getattr(checkpoint, key):
                raise ValueError(f"checkpoint profile mismatch: {key}")
        if self._anchor_revision is None:
            self._anchor_revision = checkpoint.revision
        prune_checkpoint_cache(checkpoint.repo_id, {
            self._anchor_revision, checkpoint.revision,
            *(self.pool.revision(device) for device in self.devices),
        })
        for device in self.devices:
            self.pool.reload(device, None, checkpoint.revision, checkpoint.repo_id)
            # The local pool's parent cache is not evidence of loaded weights.
            if self.pool.describe(device)["revision"] != checkpoint.revision:
                raise ProofWorkerUnavailable("worker did not acknowledge installed OID")

    def prove(self, request: ProofRequest):
        if self.describe(request.device_id)["revision"] != request.checkpoint.revision:
            raise ProofWorkerUnavailable("proof worker lost its checkpoint")
        return self.pool.call(request.device_id, request.payload.commit(),
                              request.payload.randomness, request.payload.seed_u_values)


def create_proof_app(*, backend, worker_id: str, profile_id: str,
                     generation_contract_sha256: str, training_run_id: str,
                     repo_id: str, software_revision: str, proof_path_hash: str,
                     environments: tuple[str, ...] | None = None, clock=time.time) -> FastAPI:
    """App factory for a dedicated mTLS listener (never mount on public HTTP)."""
    session_id = uuid.uuid4().hex
    if environments is None:
        from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV
        environments = tuple(MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)
    locks = {device: threading.Lock() for device in backend.devices}
    if not locks or len(locks) != len(backend.devices):
        raise ValueError("proof slots must be nonempty and distinct")
    # Each accepted proof holds a thread while it waits for its slot; the spare two
    # keep health and adoption answering.
    executor = ThreadPoolExecutor(max_workers=len(locks) * MAX_PROOF_PIPELINE_DEPTH + 2,
                                  thread_name_prefix="remote-proof")
    state_lock = threading.Lock()
    # Proofs accepted per slot and not yet finished: the one on the GPU plus those queued.
    pending = {device: 0 for device in locks}
    checkpoint = None
    highest_requested = None
    cache: OrderedDict[tuple[str, int], tuple[str, bytes | None]] = OrderedDict()
    cache_bytes = 0
    cache_limit = 64 * 1024 * 1024

    def describe():
        from reliquary.validator.utility_telemetry import utility_telemetry_enabled
        descriptions = (backend.health_descriptions() if hasattr(backend, "health_descriptions")
                        else [backend.describe(device) for device in locks])
        slots = [SlotState.model_validate({k: d[k] for k in SlotState.model_fields})
                 for d in descriptions]
        if checkpoint is not None and any(s.revision != checkpoint.revision for s in slots):
            raise ProofWorkerUnavailable("a GPU slot lost the installed checkpoint")
        return ProofHealth(
            utility_telemetry_enabled=utility_telemetry_enabled(),
            worker_id=worker_id, session_id=session_id, profile_id=profile_id,
            generation_contract_sha256=generation_contract_sha256,
            training_run_id=training_run_id, repo_id=repo_id,
            software_revision=software_revision, proof_path_hash=proof_path_hash,
            transport_sha256=transport_hash(),
            checkpoint=checkpoint, slots=slots,
            # HF dictionaries retain integer id2label keys and tuple settings.
            # Validate their actual JSON representation at this wire boundary.
            config=json.loads(canonical_bytes(descriptions[0]["config"])),
            generation_config=json.loads(canonical_bytes(descriptions[0]["generation_config"])),
        )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            # A disconnected HTTP request does not cancel its GPU work.
            executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    async def read(request, cls, limit=MAX_REQUEST_BYTES):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise HTTPException(415, "application/json required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise HTTPException(413, "request too large")
        try:
            return cls.read(bytes(body), limit=limit)
        except (ValueError, TypeError, RecursionError) as exc:
            raise HTTPException(422, "invalid proof protocol message") from exc

    def response(value):
        body = canonical_bytes(value.model_dump())
        if len(body) > MAX_RESPONSE_BYTES:
            raise ProofWorkerUnavailable("proof response exceeded bound")
        return Response(body, media_type="application/json")

    def check_identity(value):
        if value.worker_id != worker_id or value.session_id != session_id:
            raise HTTPException(409, "worker session mismatch")
        cp = value.checkpoint
        if (cp.profile_id, cp.generation_contract_sha256, cp.training_run_id, cp.repo_id) != (
            profile_id, generation_contract_sha256, training_run_id, repo_id
        ):
            raise HTTPException(409, "proof checkpoint lineage mismatch")

    @app.get("/v1/health")
    async def health():
        try:
            return response(await asyncio.get_running_loop().run_in_executor(executor, describe))
        except Exception as exc:
            raise HTTPException(503, "proof worker unavailable") from exc

    @app.post("/v1/adopt")
    async def adopt(request: Request):
        nonlocal checkpoint, highest_requested
        value = await read(request, AdoptionRequest)
        check_identity(value)
        acquired = []
        with state_lock:
            if highest_requested is not None and (
                value.checkpoint.checkpoint_n < highest_requested.checkpoint_n
                or (value.checkpoint.checkpoint_n == highest_requested.checkpoint_n
                    and value.checkpoint != highest_requested)
            ):
                raise HTTPException(409, "checkpoint identity cannot regress or rebind")
            for lock in locks.values():
                if not lock.acquire(blocking=False):
                    for held in acquired:
                        held.release()
                    raise HTTPException(503, "proofs have not drained")
                acquired.append(lock)
            if checkpoint == value.checkpoint:
                try:
                    return response(describe())
                except Exception as exc:
                    raise HTTPException(503, "installed proof checkpoint unavailable") from exc
                finally:
                    for held in acquired:
                        held.release()
            checkpoint = None
            highest_requested = value.checkpoint

        def install():
            nonlocal checkpoint
            backend.adopt(value.checkpoint)
            slots = [backend.describe(d) for d in locks]
            if any(s["revision"] != value.checkpoint.revision for s in slots):
                raise ProofWorkerUnavailable("partial checkpoint adoption")
            checkpoint = value.checkpoint
            return describe()

        try:
            future = executor.submit(install)
        except Exception:
            for held in acquired:
                held.release()
            raise HTTPException(503, "proof executor stopped")
        future.add_done_callback(lambda _f: [lock.release() for lock in acquired])
        try:
            return response(await asyncio.wrap_future(future))
        except Exception as exc:
            logger.warning("proof adoption failed: %s", type(exc).__name__)
            raise HTTPException(503, "checkpoint adoption failed") from exc

    async def prove_values(values, *, batch):
        nonlocal cache_bytes
        hashes = []
        now_ms = int(clock() * 1000)
        for value in values:
            check_identity(value)
            if value.environment not in environments:
                raise HTTPException(409, "environment is not in the proof profile")
            if not now_ms < value.expires_at_ms <= now_ms + 3_600_000:
                raise HTTPException(409, "proof deadline expired or unbounded")
            raw = canonical_bytes(value.model_dump())
            if len(raw) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "proof item too large")
            hashes.append(digest(value.model_dump()))
        first = values[0]
        # Ordered batch cache plus per-item receipts: retry either endpoint without
        # executing completed work again. A running/failed attempt stays fenced.
        key = ("batch:" + digest(hashes), 0) if batch else (first.job_id, first.attempt)
        request_hash = digest(hashes) if batch else hashes[0]
        limit = MAX_BATCH_RESPONSE_BYTES if batch else MAX_RESPONSE_BYTES
        with state_lock:
            if any(value.checkpoint != checkpoint for value in values):
                raise HTTPException(409, "checkpoint not adopted")
            for value, bound_hash in zip(values, hashes):
                previous = cache.get((value.job_id, value.attempt))
                if previous is not None and previous[0] != bound_hash:
                    raise HTTPException(409, "proof attempt was rebound")
            previous = cache.get(key)
            if previous is not None:
                if previous[0] != request_hash:
                    raise HTTPException(409, "proof attempt was rebound")
                if not previous[1]:
                    raise HTTPException(503, "proof attempt is running or failed")
                return Response(previous[1], media_type="application/json")
            lock = locks.get(first.device_id)
            if lock is None:
                raise HTTPException(409, "unknown proof slot")
            if pending[first.device_id] >= MAX_PROOF_PIPELINE_DEPTH:
                raise HTTPException(503, "proof slot queue full")
            reserved = {key} | {(value.job_id, value.attempt) for value in values}
            # Reserve room for both the batch envelope and per-item receipts.
            while len(cache) + len(reserved) > 128 or cache_bytes + 2 * limit > cache_limit:
                old_key = next((k for k, (_, body) in cache.items()
                                if body is not None and k not in reserved), None)
                if old_key is None:
                    raise HTTPException(503, "proof retry cache full")
                _, old_body = cache.pop(old_key)
                cache_bytes -= len(old_body)
            cache[key] = (request_hash, None)
            pending[first.device_id] += 1

        def execute():
            nonlocal cache_bytes
            results = []
            acquired = False
            try:
                # A queued proof waits for the GPU no longer than its own deadline.
                deadline = min(value.expires_at_ms for value in values) / 1000
                acquired = lock.acquire(timeout=max(0.0, deadline - clock()))
                if not acquired:
                    raise ProofWorkerUnavailable("proof deadline expired waiting for its slot")
                for value, bound_hash in zip(values, hashes):
                    item_key = (value.job_id, value.attempt)
                    if int(clock() * 1000) >= value.expires_at_ms:
                        raise ProofWorkerUnavailable("proof deadline expired before execution")
                    with state_lock:
                        old = cache.get(item_key) if batch else None
                        if old is not None and not old[1]:
                            raise ProofWorkerUnavailable("proof attempt is running or failed")
                        if old is None:
                            cache[item_key] = (bound_hash, None)
                    if old is not None:
                        result = ProofResponse.read(old[1], limit=MAX_RESPONSE_BYTES)
                    else:
                        started = time.perf_counter()
                        description = backend.describe(value.device_id)
                        if description["revision"] != value.checkpoint.revision or description["runtime"]["profile_hash"] != value.runtime_hash:
                            raise ProofWorkerUnavailable("loaded proof identity changed")
                        backend_started = time.perf_counter()
                        proof = ProofValues.from_kernel(backend.prove(value))
                        backend_ms = (time.perf_counter() - backend_started) * 1000
                        proof.validate_input_coverage(value.payload)
                        if int(clock() * 1000) >= value.expires_at_ms:
                            raise ProofWorkerUnavailable("proof completed after deadline")
                        result = ProofResponse(request_sha256=bound_hash,
                            **{k: getattr(value, k) for k in (
                                "job_id", "attempt", "worker_id", "session_id", "device_id",
                                "runtime_hash", "checkpoint", "window", "environment", "content_sha256",
                            )}, result=proof)
                        raw = canonical_bytes(result.model_dump())
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raise ProofWorkerUnavailable("proof response exceeded bound")
                        with state_lock:
                            cache[item_key] = (bound_hash, raw)
                            cache_bytes += len(raw)
                        logger.info("proof_backend job=%s window=%d env=%s identity_ms=%.3f backend_ms=%.3f total_ms=%.3f",
                                    value.job_id, value.window, value.environment,
                                    (backend_started - started) * 1000, backend_ms,
                                    (time.perf_counter() - started) * 1000)
                    results.append(result)
                    if not result.result.all_passed:
                        break
                raw = canonical_bytes((ProofBatchResponse(items=results) if batch else results[0]).model_dump())
                if len(raw) > limit:
                    raise ProofWorkerUnavailable("proof response exceeded bound")
                if batch:
                    with state_lock:
                        cache[key] = (request_hash, raw)
                        cache_bytes += len(raw)
                return raw
            except Exception:
                with state_lock:
                    for failed_key, (bound_hash, body) in list(cache.items()):
                        if failed_key in reserved and body is None:
                            cache[failed_key] = (bound_hash, b"")
                raise
            finally:
                if acquired:
                    lock.release()

        try:
            future = executor.submit(execute)
        except Exception:
            with state_lock:
                cache[key] = (request_hash, b"")
                pending[first.device_id] -= 1
            raise HTTPException(503, "proof executor stopped")
        def completed(f):
            with state_lock:
                if f.cancelled():
                    cache[key] = (request_hash, b"")
                pending[first.device_id] -= 1
        future.add_done_callback(completed)
        try:
            raw = await asyncio.wrap_future(future)
            return Response(raw, media_type="application/json")
        except Exception as exc:
            logger.warning("proof failed job=%s cause=%s", first.job_id, type(exc).__name__)
            raise HTTPException(503, "proof infrastructure failure") from exc

    @app.post("/v1/prove")
    async def prove(request: Request):
        return await prove_values([await read(request, ProofRequest)], batch=False)

    @app.post("/v1/prove-batch")
    async def prove_batch(request: Request):
        value = await read(request, ProofBatchRequest, MAX_BATCH_REQUEST_BYTES)
        return await prove_values(value.items, batch=True)

    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from reliquary import constants as c
    from reliquary.validator.observability import immutable_build_revision
    from reliquary.validator.proof_capacity import (
        compute_proof_path_hash, expand_proof_slots, resolve_cuda_proof_devices,
    )
    from reliquary.validator.proof_worker import (
        assert_proof_slots_supported, build_isolated_proof_plane,
    )
    import torch

    if c.PROTOCOL_VERSION < 5:
        raise RuntimeError("remote proof requires a generation-contract-stamped profile (V5+)")

    def required(key):
        value = os.environ.get(key, "").strip()
        if not value:
            raise RuntimeError(f"{key} is required")
        return value

    worker_id = required("RELIQUARY_PROOF_WORKER_ID")
    host = required("RELIQUARY_PROOF_HOST")
    repo_id = required("RELIQUARY_HF_REPO_ID")
    revision = immutable_build_revision()
    if not isinstance(revision, str) or len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
        raise RuntimeError("proof worker requires an immutable build revision")
    identities = resolve_cuda_proof_devices(
        required("RELIQUARY_PROOF_DEVICES").split(","), cuda=torch.cuda,
    )
    devices = tuple(d.device_id for d in identities)
    assert_proof_slots_supported(slots_per_device=c.PROOF_SLOTS_PER_DEVICE,
                                 isolation=True, proof_devices=devices)
    pool, _ = build_isolated_proof_plane(
        devices=expand_proof_slots(devices, c.PROOF_SLOTS_PER_DEVICE),
        checkpoint=c.DEFAULT_BASE_MODEL,
        load_kwargs={"revision": c.DEFAULT_BASE_MODEL_REVISION},
    )
    # Validate mandatory TLS before allocating any GPU replicas.
    tls = {"ssl_certfile": required("RELIQUARY_PROOF_TLS_CERT"),
           "ssl_keyfile": required("RELIQUARY_PROOF_TLS_KEY"),
           "ssl_ca_certs": required("RELIQUARY_PROOF_CLIENT_CA"),
           "ssl_cert_reqs": ssl.CERT_REQUIRED}
    try:
        pool.start()
        app = create_proof_app(
            backend=ProofBackend(pool), worker_id=worker_id,
            profile_id=c.PROTOCOL_PROFILE_ID,
            generation_contract_sha256=digest(c.PROTOCOL_GENERATION_CONTRACT),
            training_run_id=c.TRAINING_RUN_ID, repo_id=repo_id,
            software_revision=revision, proof_path_hash=compute_proof_path_hash(),
        )
        uvicorn.run(app, host=host,
                    port=int(os.environ.get("RELIQUARY_PROOF_PORT", "8445")),
                    access_log=False, server_header=False, timeout_keep_alive=120,
                    limit_concurrency=len(pool.devices) * MAX_PROOF_PIPELINE_DEPTH + 4, **tls)
    finally:
        pool.close(force=True)


if __name__ == "__main__":
    main()
