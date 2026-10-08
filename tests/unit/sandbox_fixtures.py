"""Keys, machine documents, signed transcripts and capacity reports for the
signed-sandbox tests (built with reliquary_sandbox.attest, as a machine would)."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from reliquary_sandbox.attest import (
    HEARTBEAT_DOMAIN, Budgets, CallBody, FinalBody, OpenBody, SessionClaims, Signer, chain_hash,
    generate_private_key_pem, issue_session_token, load_private_key, sign_record, token_sha256,
)

from reliquary.corpus.signed_reasons import corpus_engagement
from reliquary.sandbox.machines import DirectorySnapshot, MachineEntry

IMAGE = "registry.example/swe@sha256:" + "a" * 64
MACHINE = "machine-1"
ADDRESS = "http://10.0.0.5:8080"
ENV = "reliquary-swe"
ENV_PACKAGE = "reliquary-swe==0.1.0a1"
JOB_ID = "swe-agentic-v1"
CHECKPOINT = "c" * 64
NOW = 1_800_000_000
GIB = 1024**3
BUDGETS = dict(max_calls=64, per_call_timeout_s=600, cpu_s=3600, wall_s=3600,
               memory_bytes=4 * GIB, pids=1024, disk_bytes=10 * GIB)


def signer(tmp_path: Path, name: str, key_id: str) -> Signer:
    path = tmp_path / f"{name}.pem"
    path.write_bytes(generate_private_key_pem())
    path.chmod(0o600)
    return Signer(key_id, load_private_key(path))


def machine_document(machine: Signer, *, machine_id=MACHINE, address=ADDRESS, valid_from=0,
                     valid_until=None, status="active", capacity=4, provider="hetzner") -> dict:
    return {"schema": "reliquary/sandbox-machine/v1", "machine_id": machine_id,
            "address": address, "provider": provider, "capacity": capacity, "status": status,
            "keys": [{"key_id": machine.key_id, "public_key_b64": machine.public_key_b64,
                      "valid_from": valid_from, "valid_until": valid_until}],
            "registered_at": 0.0, "last_heartbeat": None}


def directory(machine: Signer, **kw) -> DirectorySnapshot:
    return DirectorySnapshot([MachineEntry.from_document(machine_document(machine, **kw))])


def claims(**overrides) -> SessionClaims:
    values = dict(session_id="s-1", hotkey="5Hot", engagement=corpus_engagement(JOB_ID, 0),
                  env=ENV, split="train:20", index=0, image=IMAGE, checkpoint=CHECKPOINT,
                  machine_id=MACHINE, issued_at=NOW, expires_at=NOW + 4500,
                  budgets=Budgets(**BUDGETS))
    values.update(overrides)
    return SessionClaims(**values)


def transcript(validator: Signer, machine: Signer, session: SessionClaims, *, calls=(),
               status="graded", reward=1.0, state: bytes | None = b"d\n",
               tools=("bash", "edit"), env_package=ENV_PACKAGE, reason=None, at=None) -> dict:
    """A transcript as the gateway signs it. `calls`: dicts with `turn`, `k`,
    `arguments`, and optionally `tool`, `output`, `truncated`, `timed_out`."""
    token = issue_session_token(session, validator)
    at = session.issued_at + 10 if at is None else at
    bodies = [OpenBody(i=0, session_id=session.session_id, machine_id=session.machine_id,
                       token_sha256=token_sha256(token), image=session.image, env=session.env,
                       env_package=env_package, tools_version="reliquary-tools/1",
                       tools=tuple(tools), setup_output_sha256="0" * 64,
                       budgets=session.budgets, at=at)]
    for call in calls:
        bodies.append(CallBody(
            i=len(bodies), session_id=session.session_id, turn=call["turn"], k=call["k"],
            tool=call.get("tool", "bash"), arguments=call["arguments"],
            output=call.get("output", ""), exit_code=0,
            truncated=call.get("truncated", False), timed_out=call.get("timed_out", False),
            cpu_ms=1, wall_ms=1, at=at, prev=chain_hash(bodies[-1])))
    graded = status == "graded"
    bodies.append(FinalBody(
        i=len(bodies), session_id=session.session_id, status=status,
        reward=float(reward) if graded else None,
        grading={"tests_passed": reward == 1.0} if graded else None,
        state_sha256=hashlib.sha256(state).hexdigest() if graded and state is not None else None,
        cpu_total_ms=2, reason=reason, at=at, prev=chain_hash(bodies[-1])))
    return {"token": token,
            "records": [sign_record(machine, session.machine_id, body) for body in bodies]}


def capacity_report(machine: Signer, *, at: int, machine_id=MACHINE, address=ADDRESS,
                    free=4, capacity=4, images=(IMAGE,), env_packages=None, caps=None,
                    tools_version="reliquary-tools/1") -> dict:
    document = {
        "machine_id": machine_id, "public_base_url": address, "key_id": machine.key_id,
        "at": at, "capacity": capacity, "active": capacity - free, "free": free,
        "load": [0.1, 0.1, 0.1], "cpu_count": 16, "images": sorted(images),
        "env_packages": dict(env_packages or {ENV: ENV_PACKAGE}),
        "tools_version": tools_version, "runsc_version": "release-20260928.0",
        "busybox_version": "1.37.0", "helper_busybox_sha256": "0" * 64,
        "caps": dict(caps or {"max_calls": 512, "max_cpu_s": 3600, "max_call_timeout_s": 600,
                              "max_wall_s": 14400, "max_memory_bytes": 8 * GIB,
                              "max_pids": 1024, "max_disk_bytes": 10 * GIB,
                              "max_argument_bytes": 122880, "output_cap_bytes": 65536,
                              "max_token_validity_s": 7 * 86400,
                              "max_transcript_bytes": 8 * 1024**2}),
    }
    return {"document": document, "signature": machine.sign(HEARTBEAT_DOMAIN, document)}


def public_key_b64(seed: int) -> str:
    """A canonical base64 32-byte value (not a real key: shape checks only)."""
    return base64.b64encode(bytes([seed]) * 32).decode("ascii")
