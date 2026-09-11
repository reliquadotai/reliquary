"""Fail-closed client and pool adapter for the typed GPU proof endpoint."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import os
import ssl
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
import uuid

import httpx

from reliquary.shared.runtime_fingerprint import runtime_profile_hash
from reliquary.validator.proof_worker import (
    ProofModelProxy, ProofWorkerUnavailable, remote_commitment_verifier,
)
from reliquary.validator.remote_proof_protocol import (
    MAX_PROOF_PIPELINE_DEPTH,
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, AdoptionRequest, CheckpointBinding,
    ProofHealth, ProofInput, ProofRequest, ProofResponse, ProofValues,
    RemoteProofMeasurement, canonical_bytes, digest, transport_hash,
    ProofBatchRequest, ProofBatchResponse, MAX_PROOF_BATCH,
    MAX_BATCH_REQUEST_BYTES, MAX_BATCH_RESPONSE_BYTES,
)

logger = logging.getLogger(__name__)


def executor_mode() -> str:
    mode = os.environ.get("RELIQUARY_PROOF_EXECUTOR_MODE", "local").strip().lower()
    if mode not in {"local", "shadow", "remote"}:
        raise ValueError("RELIQUARY_PROOF_EXECUTOR_MODE must be local, shadow or remote")
    if mode == "local" and os.environ.get("RELIQUARY_PROOF_EXECUTOR_URL", "").strip():
        raise ValueError("proof executor URL requires explicit shadow or remote mode")
    return mode


# Scheduler lanes beyond the first on one worker slot: ``cuda:0`` then ``cuda:0~1``.
_LANE = "~"


class RemoteProofPool:
    is_remote = True

    def __init__(self, *, base_url: str, ca_path: str, cert_path: str,
                 key_path: str, expected_worker_id: str, profile_id: str,
                 generation_contract_sha256: str, training_run_id: str,
                 repo_id: str, request_timeout: float, reload_timeout: float,
                 pipeline_depth: int = 1):
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("proof endpoint requires a bare HTTPS origin")
        if not 0 < request_timeout <= 3600 or not 0 < reload_timeout <= 3600:
            raise ValueError("proof timeouts must be positive and bounded")
        if not 1 <= pipeline_depth <= MAX_PROOF_PIPELINE_DEPTH:
            raise ValueError(f"proof pipeline depth must be within 1..{MAX_PROOF_PIPELINE_DEPTH}")
        tls = ssl.create_default_context(cafile=ca_path)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(cert_path, key_path)
        # Keep one connection per dispatch lane plus health: a reconnect is a TLS handshake.
        self._client = httpx.Client(base_url=base_url.rstrip("/"), verify=tls,
                                    trust_env=False, follow_redirects=False,
                                    limits=httpx.Limits(max_connections=66, max_keepalive_connections=pipeline_depth + 1,
                                                        keepalive_expiry=90))
        self.worker_id = expected_worker_id
        self.pipeline_depth = pipeline_depth
        self._identity = dict(profile_id=profile_id,
                              generation_contract_sha256=generation_contract_sha256,
                              training_run_id=training_run_id, repo_id=repo_id)
        self.request_timeout = request_timeout
        self.reload_timeout = reload_timeout
        self.health: ProofHealth | None = None
        self._checkpoint: CheckpointBinding | None = None
        self._adopted: CheckpointBinding | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._health_checked_at = 0.0
        self._health_probe = None
        self._measurements = threading.local()
        self._rpc_lock = threading.Lock()
        self._rpc_stats = dict(calls=0, reconnects=0, seconds=0.0,
                               request_bytes=0, response_bytes=0, rollouts=0)

    @contextmanager
    def measure_group(self):
        """Capture authenticated wire completions on this scheduler thread only."""
        if getattr(self._measurements, "receipts", None) is not None:
            raise ProofWorkerUnavailable("nested proof measurement")
        receipts = []
        self._measurements.receipts = receipts
        try:
            yield receipts
        finally:
            self._measurements.receipts = None

    @classmethod
    def from_environment(cls, *, repo_id):
        from reliquary import constants as c
        if c.PROTOCOL_VERSION < 5:
            raise RuntimeError("remote proof requires a generation-contract-stamped profile (V5+)")
        values = {}
        for field, name in {
            "base_url": "RELIQUARY_PROOF_EXECUTOR_URL",
            "ca_path": "RELIQUARY_PROOF_TLS_CA",
            "cert_path": "RELIQUARY_PROOF_TLS_CERT",
            "key_path": "RELIQUARY_PROOF_TLS_KEY",
            "expected_worker_id": "RELIQUARY_PROOF_EXPECTED_WORKER_ID",
        }.items():
            values[field] = os.environ.get(name, "").strip()
            if not values[field]:
                raise RuntimeError(f"{name} is required")
        return cls(**values, repo_id=repo_id, profile_id=c.PROTOCOL_PROFILE_ID,
                   generation_contract_sha256=digest(c.PROTOCOL_GENERATION_CONTRACT),
                   training_run_id=c.TRAINING_RUN_ID,
                   request_timeout=c.PROOF_WORKER_REQUEST_TIMEOUT_SECONDS,
                   reload_timeout=c.PROOF_WORKER_RELOAD_TIMEOUT_SECONDS,
                   pipeline_depth=c.PROOF_PIPELINE_DEPTH)

    def _request(self, method, path, body=None, *, timeout, retry=False,
                 request_limit=MAX_REQUEST_BYTES, response_limit=MAX_RESPONSE_BYTES):
        started = time.perf_counter()
        reconnects = 0
        def trace(event, _info):
            nonlocal reconnects
            if event == "connection.connect_tcp.started":
                reconnects += 1
        raw = None if body is None else canonical_bytes(body.model_dump())
        if raw is not None and len(raw) > request_limit:
            raise ProofWorkerUnavailable("proof request exceeds transport bound")
        retry = retry or method == "GET"  # A stale keep-alive must not fence a healthy worker.
        for attempt in range(2 if retry else 1):
            try:
                with self._client.stream(method, path, content=raw,
                        headers={"content-type": "application/json"}, timeout=timeout,
                        extensions={"trace": trace}) as response:
                    response.raise_for_status()
                    result = bytearray()
                    for chunk in response.iter_bytes():
                        result.extend(chunk)
                        if len(result) > response_limit:
                            raise ProofWorkerUnavailable("proof response exceeds transport bound")
                    if path in {"/v1/prove", "/v1/prove-batch"}:
                        with self._rpc_lock:
                            self._rpc_stats["calls"] += 1
                            self._rpc_stats["reconnects"] += reconnects
                            self._rpc_stats["seconds"] += time.perf_counter() - started
                            self._rpc_stats["request_bytes"] += len(raw or b"")
                            self._rpc_stats["response_bytes"] += len(result)
                    logger.info("proof_rpc path=%s request_bytes=%d response_bytes=%d elapsed_ms=%.3f reconnects=%d retry=%d",
                                path, len(raw or b""), len(result), (time.perf_counter() - started) * 1000, reconnects, attempt)
                    return bytes(result)
            except httpx.TransportError as exc:
                if retry and attempt == 0:
                    continue  # Idempotent GET or the same cached proof job/attempt/body.
                raise ProofWorkerUnavailable("proof transport unavailable",
                                              remote_error_type=type(exc).__name__) from exc
            except httpx.HTTPStatusError as exc:
                raise ProofWorkerUnavailable(f"proof endpoint refused HTTP {exc.response.status_code}") from exc

    def _validate_health(self, health):
        from reliquary.validator.proof_capacity import compute_proof_path_hash
        from reliquary.validator.utility_telemetry import utility_telemetry_enabled
        if health.utility_telemetry_enabled != utility_telemetry_enabled():
            raise ProofWorkerUnavailable("remote proof utility telemetry setting differs")
        if health.worker_id != self.worker_id:
            raise ProofWorkerUnavailable("unexpected proof worker identity")
        if any(getattr(health, k) != v for k, v in self._identity.items()):
            raise ProofWorkerUnavailable("remote proof profile/run/repository mismatch")
        if health.proof_path_hash != compute_proof_path_hash():
            raise ProofWorkerUnavailable("remote proof kernel path differs")
        if health.transport_sha256 != transport_hash():
            raise ProofWorkerUnavailable("remote proof transport implementation differs")
        if self.health is not None and health.session_id != self.health.session_id:
            raise ProofWorkerUnavailable("remote proof worker restarted; requalification required")
        devices = [s.device_id for s in health.slots]
        if len(devices) != len(set(devices)):
            raise ProofWorkerUnavailable("remote proof slots are not unique")
        hashes = set()
        for slot in health.slots:
            profile = slot.runtime
            if profile.get("cuda_available") is not True or profile.get("profile_hash") != runtime_profile_hash(profile):
                raise ProofWorkerUnavailable("invalid remote GPU runtime fingerprint")
            hashes.add(profile["profile_hash"])
        if len(hashes) != 1:
            raise ProofWorkerUnavailable("proof workers have different numerical runtimes")
        if self.health is not None:
            before = [(s.device_id, s.physical_device, s.device_uuid, s.hardware_class, s.runtime) for s in self.health.slots]
            after = [(s.device_id, s.physical_device, s.device_uuid, s.hardware_class, s.runtime) for s in health.slots]
            if before != after:
                raise ProofWorkerUnavailable("remote proof hardware/runtime changed")
            for name in ("config", "generation_config"):
                old, new = getattr(self.health, name), getattr(health, name)
                if any(old.get(k) != new.get(k) for k in (
                    "eos_token_id", "pad_token_id", "bos_token_id", "vocab_size", "model_type", "hidden_size",
                )):
                    raise ProofWorkerUnavailable("remote model/tokenizer metadata changed")
        return health

    def start(self):
        try:
            health = ProofHealth.read(self._request("GET", "/v1/health", timeout=self.reload_timeout))
            self.health = self._validate_health(health)
        except (ValueError, TypeError) as exc:
            raise ProofWorkerUnavailable("invalid proof worker health") from exc

    @property
    def devices(self):
        return tuple(s.device_id for s in self.health.slots) if self.health else ()

    @property
    def dispatch_devices(self):
        """Scheduler lanes: each slot keeps its id, plus one lane per extra in-flight request."""
        return tuple(lane for slot in self.devices for lane in self._lanes(slot))

    def _lanes(self, slot):
        return (slot, *(f"{slot}{_LANE}{n}" for n in range(1, self.pipeline_depth)))

    def slot_for(self, device_id):
        """The worker slot a lane dispatches to, or None if the id is not a lane of this pool."""
        return next((slot for slot in self.devices if device_id in self._lanes(slot)), None)

    @property
    def runtime_fingerprint(self):
        if not self.health:
            raise ProofWorkerUnavailable("remote proof pool has not started")
        return dict(self.health.slots[0].runtime)

    def proxies(self):
        if not self.health:
            raise ProofWorkerUnavailable("remote proof pool has not started")
        return {d: ProofModelProxy(d, SimpleNamespace(**self.health.config),
                                   SimpleNamespace(**self.health.generation_config))
                for d in self.dispatch_devices}

    def bind_checkpoint(self, checkpoint_n, repo_id, revision):
        cp = CheckpointBinding(**{**self._identity, "repo_id": repo_id},
                               checkpoint_n=checkpoint_n, revision=revision)
        if repo_id != self._identity["repo_id"]:
            raise ProofWorkerUnavailable("proof checkpoint repository changed")
        if self._checkpoint is not None and (
            cp.checkpoint_n < self._checkpoint.checkpoint_n or
            (cp.checkpoint_n == self._checkpoint.checkpoint_n and cp != self._checkpoint)
        ):
            raise ProofWorkerUnavailable("proof checkpoint cannot regress or rebind")
        self._checkpoint = cp

    def reload(self, device_id, snapshot_dir, checkpoint_revision, repo_id=None):
        del snapshot_dir  # Controller paths are never sent to another host.
        with self._lock:
            cp = self._checkpoint
            if self._closed or not self.health or cp is None or cp.revision != checkpoint_revision or cp.repo_id != repo_id or self.slot_for(device_id) is None:
                raise ProofWorkerUnavailable("remote proof adoption requires an exact bound checkpoint")
            if self._adopted == cp:
                return
            self._adopted = None
            try:
                value = AdoptionRequest(worker_id=self.worker_id,
                    session_id=self.health.session_id, checkpoint=cp)
                health = ProofHealth.read(self._request("POST", "/v1/adopt", value,
                                                        timeout=self.reload_timeout))
                self._validate_health(health)
                if health.checkpoint != cp or any(s.revision != cp.revision for s in health.slots):
                    raise ProofWorkerUnavailable("remote GPU adoption was not acknowledged")
                self.health = health
                self._adopted = cp
                self._health_checked_at = time.monotonic()
            except (ValueError, TypeError) as exc:
                raise ProofWorkerUnavailable("invalid proof adoption acknowledgement") from exc

    def revision(self, device_id):
        if self._adopted is None or self.slot_for(device_id) is None:
            return None
        return self._adopted.revision

    def assert_ready(self):
        # A health reply from before an adoption must not clear the new ack.
        with self._lock:
            if self._closed or self._adopted is None:
                raise ProofWorkerUnavailable("remote proof checkpoint is not adopted")
            try:
                health = self._validate_health(ProofHealth.read(
                    self._request("GET", "/v1/health", timeout=self.request_timeout)))
                if health.checkpoint != self._adopted or any(s.revision != self._adopted.revision for s in health.slots):
                    raise ProofWorkerUnavailable("remote proof lost its adopted checkpoint")
                self._health_checked_at = time.monotonic()
                return health
            except Exception:
                self._adopted = None
                raise

    def readiness_snapshot(self):
        if (self._adopted is not None and time.monotonic() - self._health_checked_at > 2
                and (self._health_probe is None or not self._health_probe.is_alive())):
            def probe():
                try:
                    self.assert_ready()
                except Exception:
                    logger.warning("remote proof health check failed", exc_info=True)
            self._health_probe = threading.Thread(target=probe, daemon=True,
                                                  name="remote-proof-health")
            self._health_probe.start()
        with self._rpc_lock:
            transport = dict(self._rpc_stats)
        return {"mode": "remote", "worker_id": self.worker_id, "transport": transport,
                "ready": self._adopted is not None and not self._closed
                    and time.monotonic() - self._health_checked_at <= 10,
                "revision": self._adopted.revision if self._adopted else None}

    def qualify(self, activation_revision):
        """Use the same pinned capacity contract with measured remote GPUs."""
        import math
        from reliquary import constants as c
        from reliquary.validator.proof_capacity import capacity_budget, load_proof_capacity_qualification
        path = os.environ.get("RELIQUARY_PROOF_CAPACITY_MANIFEST", "").strip()
        sha = os.environ.get("RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256", "").strip()
        if not path or not sha or self.health is None:
            raise ProofWorkerUnavailable("remote proof requires a pinned capacity manifest")
        qualification = load_proof_capacity_qualification(path, expected_sha256=sha)
        if qualification.schema_version == 4:
            from reliquary.validator.observability import immutable_build_revision
            if immutable_build_revision() != self.health.software_revision:
                raise ProofWorkerUnavailable("combined capacity requires the measured controller image too")
        from pathlib import Path
        import hashlib
        from reliquary.shared.strict_json import strict_json_loads
        raw = Path(path).read_bytes()
        # Read the very same pinned file, including the network evidence.
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ProofWorkerUnavailable("remote proof capacity manifest changed")
        measurement = RemoteProofMeasurement.model_validate(
            strict_json_loads(raw).get("remote_proof"))
        if measurement != RemoteProofMeasurement(
            worker_id=self.worker_id, transport_sha256=self.health.transport_sha256,
        ):
            raise ProofWorkerUnavailable("capacity was not measured through this remote proof plane")
        physical = {}
        for slot in self.health.slots:
            value = (slot.hardware_class, slot.device_uuid)
            if physical.setdefault(slot.physical_device, value) != value:
                raise ProofWorkerUnavailable("remote physical GPU identity is ambiguous")
        from reliquary.shared.hf_compat import resolve_max_context_length
        budget = capacity_budget()
        report = qualification.validate(
            profile_id=c.PROTOCOL_PROFILE_ID, model_revision=c.PROTOCOL_MODEL_REVISION,
            software_revision=self.health.software_revision,
            checkpoint_revision=activation_revision or "",
            runtime_fingerprint_hash=self.runtime_fingerprint["profile_hash"],
            proof_path_hash=self.health.proof_path_hash,
            configured_devices=tuple(physical),
            configured_hardware=tuple(v[0] for v in physical.values()),
            configured_device_uuids=tuple(v[1] for v in physical.values()),
            proof_wall_seconds=budget["wall_seconds"],
            minimum_proofs_per_environment=budget["proofs_per_environment"],
            minimum_completion_tokens_per_environment={e: math.ceil(cap * .9)
                for e, cap in c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.items()},
            configured_slots={s.device_id: s.device_uuid.casefold() for s in self.health.slots},
            maximum_context_tokens=(
                resolve_max_context_length(SimpleNamespace(**self.health.config))
                if qualification.schema_version == 4 else None),
            full_completion_tokens_per_environment=c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV,
        )
        report["remote_proof"] = measurement.model_dump()
        return report

    def verifier_for_window(self, window, environment, revision):
        binding = self._checkpoint
        if binding is None or binding.revision != revision:
            raise ProofWorkerUnavailable("window proof binding is not current")

        def verify(commit, model, window_randomness, *, tokenizer=None, seed_u_values=None):
            del tokenizer
            if not isinstance(model, ProofModelProxy):
                raise ProofWorkerUnavailable("remote proof requires a model metadata proxy")
            return self.prove(model.device_id, commit, window_randomness, seed_u_values,
                              window=window, environment=environment, checkpoint=binding)
        def verify_batch(inputs, model, window_randomness):
            if not isinstance(model, ProofModelProxy):
                raise ProofWorkerUnavailable("remote proof requires a model metadata proxy")
            return self.prove_many(model.device_id, inputs, window_randomness,
                                   window=window, environment=environment, checkpoint=binding)
        verify.batch = verify_batch
        return verify

    def prove(self, device_id, commit, randomness, seed_u_values, *, window, environment, checkpoint):
        return self.prove_many(device_id, [(commit, seed_u_values)], randomness,
                               window=window, environment=environment, checkpoint=checkpoint)[0]

    def prove_many(self, device_id, inputs, randomness, *, window, environment, checkpoint):
        slot = self.slot_for(device_id)
        if self._closed or checkpoint != self._adopted or slot is None:
            raise ProofWorkerUnavailable("remote proof device/checkpoint is not ready")
        try:
            if not 1 <= len(inputs) <= MAX_PROOF_BATCH:
                raise ValueError("proof batch size exceeded")
            requests = []
            for commit, seed_u_values in inputs:
                payload = ProofInput(tokens=commit["tokens"], commitments=commit["commitments"],
                                     rollout=commit.get("rollout") or {}, randomness=randomness,
                                     seed_u_values=seed_u_values)
                requests.append(ProofRequest(job_id=uuid.uuid4().hex, attempt=0,
                    worker_id=self.worker_id, session_id=self.health.session_id,
                    device_id=slot, runtime_hash=self.runtime_fingerprint["profile_hash"],
                    checkpoint=checkpoint, window=window, environment=environment,
                    expires_at_ms=int((time.time() + self.request_timeout) * 1000),
                    content_sha256=digest(payload.model_dump()), payload=payload))
            if len(requests) == 1:
                raw = self._request("POST", "/v1/prove", requests[0], timeout=self.request_timeout, retry=True)
                results = [ProofResponse.read(raw, limit=MAX_RESPONSE_BYTES)]
            else:
                raw = self._request("POST", "/v1/prove-batch", ProofBatchRequest(items=requests),
                                    timeout=self.request_timeout, retry=True,
                                    request_limit=MAX_BATCH_REQUEST_BYTES, response_limit=MAX_BATCH_RESPONSE_BYTES)
                results = ProofBatchResponse.read(raw, limit=MAX_BATCH_RESPONSE_BYTES).items
            if (len(results) > len(requests) or
                (len(results) < len(requests) and results[-1].result.all_passed)):
                raise ProofWorkerUnavailable("remote proof batch coverage mismatch")
            receipts = getattr(self._measurements, "receipts", None)
            for request, result in zip(requests, results):
                if time.time() * 1000 >= request.expires_at_ms:
                    raise ProofWorkerUnavailable("remote proof response expired")
                if result.request_sha256 != digest(request.model_dump()) or any(
                    getattr(result, key) != getattr(request, key) for key in (
                        "job_id", "attempt", "worker_id", "session_id", "device_id", "runtime_hash",
                        "checkpoint", "window", "environment", "content_sha256",
                    )
                ):
                    raise ProofWorkerUnavailable("remote proof response binding mismatch")
                result.result.validate_input_coverage(request.payload)
            # Record only after every returned receipt is validated.
            for result in results:
                if receipts is not None:
                    receipts.append({key: getattr(result, key) for key in (
                        "job_id", "attempt", "device_id", "window", "environment", "content_sha256",
                    )} | {"checkpoint": result.checkpoint.model_dump(),
                         "policy_tokens": len(result.result.completion_chosen_probs)})
            with self._rpc_lock:
                self._rpc_stats["rollouts"] += len(results)
            return [result.result.to_kernel() for result in results]
        except ProofWorkerUnavailable:
            self._adopted = None
            raise
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self._adopted = None
            raise ProofWorkerUnavailable("invalid remote proof payload/result") from exc

    def call(self, *_args, **_kwargs):
        raise ProofWorkerUnavailable("remote proof calls require the trusted window/environment binding")

    def close(self, force=False):
        self._closed = True
        self._adopted = None
        self._client.close()


class ShadowProofPool:
    """Local remains authoritative; bounded shadow work never delays it."""
    is_remote = False
    is_shadow = True

    def __init__(self, local, remote):
        self.local, self.remote = local, remote
        self._executor = ThreadPoolExecutor(max_workers=len(remote.devices))
        self._slots = threading.BoundedSemaphore(len(remote.devices))
        self.compared = self.diverged = self.unavailable = self.dropped = 0

    def __getattr__(self, key):
        return getattr(self.local, key)

    def bind_checkpoint(self, checkpoint_n, repo_id, revision):
        self.remote.bind_checkpoint(checkpoint_n, repo_id, revision)

    def shadow_snapshot(self):
        return {"mode": "shadow", "worker_id": self.remote.worker_id,
                "compared": self.compared, "diverged": self.diverged,
                "unavailable": self.unavailable, "dropped": self.dropped}

    def reload(self, device_id, snapshot_dir, checkpoint_revision, repo_id=None):
        self.local.reload(device_id, snapshot_dir, checkpoint_revision, repo_id)
        try:
            self.remote.reload(self.remote.devices[0], None, checkpoint_revision, repo_id)
        except Exception:
            self.unavailable += 1
            logger.warning("shadow proof checkpoint unavailable", exc_info=True)

    def verifier_for_window(self, window, environment, revision):
        local_verify = remote_commitment_verifier(self.local)

        def verify(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
            result = local_verify(commit, model, randomness, tokenizer=tokenizer,
                                  seed_u_values=seed_u_values)
            if not self._slots.acquire(blocking=False):
                self.dropped += 1
                return result
            try:
                remote_verify = self.remote.verifier_for_window(window, environment, revision)
                device = self.remote.devices[self.local.devices.index(model.device_id) % len(self.remote.devices)]
                proxy = self.remote.proxies()[device]
                # Freeze bytes before the admission thread annotates the rollout.
                frozen = ProofInput(tokens=commit["tokens"], commitments=commit["commitments"],
                                    rollout=commit.get("rollout") or {}, randomness=randomness,
                                    seed_u_values=seed_u_values).model_copy(deep=True)

                def compare():
                    try:
                        candidate = remote_verify(frozen.commit(), proxy, frozen.randomness,
                                                   seed_u_values=frozen.seed_u_values)
                        self.compared += 1
                        if ProofValues.from_kernel(candidate) != ProofValues.from_kernel(result):
                            self.diverged += 1
                            logger.warning("shadow proof diverged window=%s environment=%s", window, environment)
                    except Exception:
                        self.unavailable += 1
                        logger.warning("shadow GPU proof unavailable", exc_info=True)
                future = self._executor.submit(compare)
            except Exception:
                self._slots.release()
                self.unavailable += 1
                logger.warning("shadow GPU proof not scheduled", exc_info=True)
            else:
                future.add_done_callback(lambda _f: self._slots.release())
            return result
        return verify

    def close(self, force=False):
        self._executor.shutdown(wait=not force, cancel_futures=True)
        self.remote.close(force=force)
        self.local.close(force=force)
