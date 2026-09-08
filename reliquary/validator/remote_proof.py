"""Fail-closed client and pool adapter for the typed GPU proof endpoint."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, AdoptionRequest, CheckpointBinding,
    ProofHealth, ProofInput, ProofRequest, ProofResponse, ProofValues,
    RemoteProofMeasurement, canonical_bytes, digest, transport_hash,
)

logger = logging.getLogger(__name__)


def executor_mode() -> str:
    mode = os.environ.get("RELIQUARY_PROOF_EXECUTOR_MODE", "local").strip().lower()
    if mode not in {"local", "shadow", "remote"}:
        raise ValueError("RELIQUARY_PROOF_EXECUTOR_MODE must be local, shadow or remote")
    if mode == "local" and os.environ.get("RELIQUARY_PROOF_EXECUTOR_URL", "").strip():
        raise ValueError("proof executor URL requires explicit shadow or remote mode")
    return mode


class RemoteProofPool:
    is_remote = True

    def __init__(self, *, base_url: str, ca_path: str, cert_path: str,
                 key_path: str, expected_worker_id: str, profile_id: str,
                 generation_contract_sha256: str, training_run_id: str,
                 repo_id: str, request_timeout: float, reload_timeout: float):
        url = urlsplit(base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("proof endpoint requires a bare HTTPS origin")
        if not 0 < request_timeout <= 3600 or not 0 < reload_timeout <= 3600:
            raise ValueError("proof timeouts must be positive and bounded")
        tls = ssl.create_default_context(cafile=ca_path)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(cert_path, key_path)
        self._client = httpx.Client(base_url=base_url.rstrip("/"), verify=tls,
                                    trust_env=False, follow_redirects=False)
        self.worker_id = expected_worker_id
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
                   reload_timeout=c.PROOF_WORKER_RELOAD_TIMEOUT_SECONDS)

    def _request(self, method, path, body=None, *, timeout, retry=False):
        raw = None if body is None else canonical_bytes(body.model_dump())
        if raw is not None and len(raw) > MAX_REQUEST_BYTES:
            raise ProofWorkerUnavailable("proof request exceeds transport bound")
        for attempt in range(2 if retry else 1):
            try:
                with self._client.stream(method, path, content=raw,
                        headers={"content-type": "application/json"}, timeout=timeout) as response:
                    response.raise_for_status()
                    result = bytearray()
                    for chunk in response.iter_bytes():
                        result.extend(chunk)
                        if len(result) > MAX_RESPONSE_BYTES:
                            raise ProofWorkerUnavailable("proof response exceeds transport bound")
                    return bytes(result)
            except httpx.TransportError as exc:
                if retry and attempt == 0:
                    continue  # Same job/attempt/body; server returns a cached result.
                raise ProofWorkerUnavailable("proof transport unavailable",
                                              remote_error_type=type(exc).__name__) from exc
            except httpx.HTTPStatusError as exc:
                raise ProofWorkerUnavailable(f"proof endpoint refused HTTP {exc.response.status_code}") from exc

    def _validate_health(self, health):
        from reliquary.validator.proof_capacity import compute_proof_path_hash
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
    def runtime_fingerprint(self):
        if not self.health:
            raise ProofWorkerUnavailable("remote proof pool has not started")
        return dict(self.health.slots[0].runtime)

    def proxies(self):
        if not self.health:
            raise ProofWorkerUnavailable("remote proof pool has not started")
        return {d: ProofModelProxy(d, SimpleNamespace(**self.health.config),
                                   SimpleNamespace(**self.health.generation_config))
                for d in self.devices}

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
            if not self.health or cp is None or cp.revision != checkpoint_revision or cp.repo_id != repo_id or device_id not in self.devices:
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
        if self._adopted is None or device_id not in self.devices:
            return None
        return self._adopted.revision

    def assert_ready(self):
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
        return {"mode": "remote", "worker_id": self.worker_id,
                "ready": self._adopted is not None and not self._closed
                    and time.monotonic() - self._health_checked_at <= 10,
                "revision": self._adopted.revision if self._adopted else None}

    def qualify(self, activation_revision):
        """Use the same pinned capacity contract with measured remote GPUs."""
        import math
        from reliquary import constants as c
        from reliquary.validator.proof_capacity import load_proof_capacity_qualification
        path = os.environ.get("RELIQUARY_PROOF_CAPACITY_MANIFEST", "").strip()
        sha = os.environ.get("RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256", "").strip()
        if not path or not sha or self.health is None:
            raise ProofWorkerUnavailable("remote proof requires a pinned capacity manifest")
        qualification = load_proof_capacity_qualification(path, expected_sha256=sha)
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
        report = qualification.validate(
            profile_id=c.PROTOCOL_PROFILE_ID, model_revision=c.PROTOCOL_MODEL_REVISION,
            software_revision=self.health.software_revision,
            checkpoint_revision=activation_revision or "",
            runtime_fingerprint_hash=self.runtime_fingerprint["profile_hash"],
            proof_path_hash=self.health.proof_path_hash,
            configured_devices=tuple(physical),
            configured_hardware=tuple(v[0] for v in physical.values()),
            configured_device_uuids=tuple(v[1] for v in physical.values()),
            proof_wall_seconds=c.MAX_PROOF_WALL_SECONDS,
            minimum_proofs_per_environment=c.MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW + c.FORENSIC_SAMPLE_PER_WINDOW,
            minimum_completion_tokens_per_environment={e: math.ceil(cap * .9)
                for e, cap in c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV.items()},
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
        return verify

    def prove(self, device_id, commit, randomness, seed_u_values, *, window, environment, checkpoint):
        if self._closed or checkpoint != self._adopted or device_id not in self.devices:
            raise ProofWorkerUnavailable("remote proof device/checkpoint is not ready")
        try:
            payload = ProofInput(tokens=commit["tokens"], commitments=commit["commitments"],
                                 rollout=commit.get("rollout") or {}, randomness=randomness,
                                 seed_u_values=seed_u_values)
            request = ProofRequest(job_id=uuid.uuid4().hex, attempt=0,
                worker_id=self.worker_id, session_id=self.health.session_id,
                device_id=device_id, runtime_hash=self.runtime_fingerprint["profile_hash"],
                checkpoint=checkpoint, window=window, environment=environment,
                expires_at_ms=int((time.time() + self.request_timeout) * 1000),
                content_sha256=digest(payload.model_dump()), payload=payload)
            raw = self._request("POST", "/v1/prove", request,
                                timeout=self.request_timeout, retry=True)
            result = ProofResponse.read(raw, limit=MAX_RESPONSE_BYTES)
            if time.time() * 1000 >= request.expires_at_ms:
                raise ProofWorkerUnavailable("remote proof response expired")
            if result.request_sha256 != digest(request.model_dump()) or any(
                getattr(result, key) != getattr(request, key) for key in (
                    "job_id", "attempt", "worker_id", "session_id", "device_id", "runtime_hash",
                    "checkpoint", "window", "environment", "content_sha256",
                )
            ):
                raise ProofWorkerUnavailable("remote proof response binding mismatch")
            result.result.validate_input_coverage(payload)
            return result.result.to_kernel()
        except ProofWorkerUnavailable:
            self._adopted = None
            raise
        except (ValueError, TypeError, KeyError) as exc:
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

    def __init__(self, local, remote):
        self.local, self.remote = local, remote
        self._executor = ThreadPoolExecutor(max_workers=len(remote.devices))
        self._slots = threading.BoundedSemaphore(len(remote.devices))
        self.compared = self.diverged = self.unavailable = self.dropped = 0

    def __getattr__(self, key):
        return getattr(self.local, key)

    def bind_checkpoint(self, checkpoint_n, repo_id, revision):
        self.remote.bind_checkpoint(checkpoint_n, repo_id, revision)

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
                        self.diverged += ProofValues.from_kernel(candidate) != ProofValues.from_kernel(result)
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
