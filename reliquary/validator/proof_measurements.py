"""Private opt-in timings of complete scheduler-owned remote proof groups."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time

from reliquary.validator.proof_worker import ProofWorkerUnavailable
from reliquary.validator.remote_proof_protocol import RemoteProofMeasurement, canonical_bytes


class ProofMeasurements:
    def __init__(self, path, pool):
        from reliquary.validator.remote_proof import RemoteProofPool
        if not isinstance(pool, RemoteProofPool):
            raise ValueError("end-to-end measurements require the authoritative remote proof pool")
        self.pool = pool
        self.path = Path(path)
        if not self.path.is_absolute():
            raise ValueError("proof measurement path must be absolute")
        self._lock = threading.Lock()
        # New evidence per process; never append to a previous run or follow a symlink.
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self._identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        os.close(fd)

    @classmethod
    def from_environment(cls, pool):
        path = os.environ.get("RELIQUARY_PRIVATE_REMOTE_PROOF_MEASUREMENTS", "").strip()
        return cls(path, pool) if path else None

    def execute(self, invocation, model, execute):
        from reliquary.constants import M_ROLLOUTS, PROTOCOL_MODEL_REVISION, PROTOCOL_PROFILE_ID
        from reliquary.validator.batcher import _ScheduledProofPayload, ValidSubmission
        payload = invocation.candidate.payload
        if not isinstance(payload, _ScheduledProofPayload):
            raise ProofWorkerUnavailable("measurement requires the complete batcher proof payload")
        health, checkpoint = self.pool.health, self.pool._adopted
        if (health is None or checkpoint is None or invocation.checkpoint_revision != checkpoint.revision
                or health.profile_id != PROTOCOL_PROFILE_ID
                or payload.batcher.window_start != payload.pending.request.window_start
                or invocation.environment != payload.batcher.env.name):
            raise ProofWorkerUnavailable("measurement checkpoint/window/profile is not ready")
        slot_id = self.pool.slot_for(invocation.device_id)
        slot = next((s for s in health.slots if s.device_id == slot_id), None)
        if slot is None or slot.revision != checkpoint.revision:
            raise ProofWorkerUnavailable("measurement GPU slot is not adopted")
        started = time.perf_counter()
        submission, failure = None, None
        with self.pool.measure_group() as receipts:
            try:
                submission = execute(model)
                return submission
            except BaseException as exc:
                failure = type(exc).__name__
                raise
            finally:
                seconds = time.perf_counter() - started
                rollouts = payload.pending.request.rollouts
                complete = (len(rollouts) == len(receipts) == M_ROLLOUTS
                    and len({r["job_id"] for r in receipts}) == M_ROLLOUTS
                    and self.pool._adopted == checkpoint
                    and all(r["device_id"] == slot.device_id
                        and r["window"] == payload.batcher.window_start
                        and r["environment"] == invocation.environment
                        and r["checkpoint"] == checkpoint.model_dump() for r in receipts))
                row = {"schema_version": 1, "environment": invocation.environment,
                    "seconds": seconds, "proof_passed": isinstance(submission, ValidSubmission)
                        and len(submission.rollouts) == M_ROLLOUTS and complete and failure is None,
                    "infrastructure_error_type": failure, "complete_remote_group": complete,
                    "profile_id": health.profile_id, "model_revision": PROTOCOL_MODEL_REVISION,
                    "software_revision": health.software_revision, "checkpoint_revision": checkpoint.revision,
                    "session_id": health.session_id,
                    "checkpoint_n": checkpoint.checkpoint_n, "repo_id": checkpoint.repo_id,
                    "training_run_id": checkpoint.training_run_id,
                    "runtime_fingerprint_hash": slot.runtime["profile_hash"],
                    "hardware_class": slot.hardware_class, "device_uuid": slot.device_uuid,
                    "device_id": slot.device_id, "rollout_count": len(rollouts),
                    # Authenticated sparse-output coverage, never miner-claimed lengths.
                    "completion_token_lengths": [r["policy_tokens"] for r in receipts],
                    "plan_id": invocation.plan_id, "job_id": invocation.candidate.job_id,
                    "window": payload.batcher.window_start, "wire_receipts": receipts,
                    "remote_proof": RemoteProofMeasurement(worker_id=health.worker_id,
                        transport_sha256=health.transport_sha256).model_dump()}
                try:
                    with self._lock:
                        fd = os.open(self.path, os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, "ab") as handle:
                            stat = os.fstat(handle.fileno())
                            if (stat.st_dev, stat.st_ino) != self._identity:
                                raise OSError("measurement file identity changed")
                            handle.write(canonical_bytes(row) + b"\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                except OSError as exc:
                    raise ProofWorkerUnavailable("private proof measurement could not be persisted") from exc
