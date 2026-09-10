"""An operator authorization never becomes a synthetic qualification."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reliquary.validator import observed_proof_rollout as observed


@pytest.fixture
def rollout(tmp_path, monkeypatch):
    from reliquary import constants as c
    from reliquary.validator import observability
    from reliquary.validator.proof_capacity import capacity_budget
    from reliquary.validator.remote_proof_protocol import CheckpointBinding

    monkeypatch.setenv("RELIQUARY_PROOF_CAPACITY_MODE", "observed_live")
    for key, value in {"FILL_CLOSED_ENABLED": True, "FILL_CLOSED_BOUNDED_PROOFS": True,
            "FILL_CLOSED_MAX_SECONDS": 1800.0, "FILL_CLOSED_PROOF_DRAIN_SECONDS": 360.0,
            "FILL_CLOSED_PROOF_DISPATCH_SECONDS": 1440.0,
            "MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV": {"math": 8192, "code": 8192, "logic": 8192}}.items():
        monkeypatch.setattr(c, key, value)
    source, revision = "a" * 40, "b" * 40
    monkeypatch.setattr(observability, "immutable_build_revision", lambda: source)
    cp = CheckpointBinding(checkpoint_n=1812, revision=revision, profile_id=c.PROTOCOL_PROFILE_ID,
        generation_contract_sha256="c" * 64, training_run_id="new-v1", repo_id="test/model")
    slot = SimpleNamespace(device_id="cuda:0", physical_device="cuda:0", device_uuid="gpu-1",
        hardware_class="H100", revision=revision)
    health = SimpleNamespace(software_revision=source, worker_id="proof-1", session_id="session-1",
        checkpoint=cp, slots=[slot], transport_sha256="d" * 64, proof_path_hash="e" * 64,
        **{k: getattr(cp, k) for k in ("generation_contract_sha256", "training_run_id", "repo_id")})
    pool = SimpleNamespace(is_remote=True, health=health,
        runtime_fingerprint={"profile_hash": "f" * 64}, _validate_health=Mock(side_effect=lambda h: h))
    identity = {"controller_software_revision": source, "worker_software_revision": source,
        "worker_id": health.worker_id, "session_id": health.session_id,
        "runtime_fingerprint_hash": "f" * 64, "transport_sha256": health.transport_sha256,
        "proof_path_hash": health.proof_path_hash, "profile_id": c.PROTOCOL_PROFILE_ID,
        "model_revision": c.PROTOCOL_MODEL_REVISION, "generation_contract_sha256": cp.generation_contract_sha256,
        "training_run_id": cp.training_run_id, "repo_id": cp.repo_id,
        "checkpoint_n": cp.checkpoint_n, "checkpoint_revision": revision,
        "device_id": slot.device_id, "physical_device": slot.physical_device,
        "device_uuid": slot.device_uuid, "hardware_class": slot.hardware_class,
        "environments": ["code", "logic", "math"]}
    events = [{"event": "run_started", "checkpoint": cp.model_dump()}]
    for i, env in enumerate(["code", "logic", "math", "code"]):
        common = {"job_id": f"job:{i}", "environment": env, "input_sha256": str(i) * 64}
        events.append({"event": "admission", "outcome": "accepted", **common})
        events.append({"event": "proof", "outcome": "passed" if i < 3 else "rejected",
            "reason": None if i < 3 else "logprob_mismatch", **common})
    events.append({"event": "run_complete", "admitted_groups": 4, "passing_groups": 3,
        "admission_rejections_by_reason": {}, "proof_rejections_by_reason": {"logprob_mismatch": 1},
        "numerical_rejections_requires_review": {"logprob_mismatch": 1}})
    def pin(name, raw):
        path = tmp_path / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    natural = pin("natural.jsonl", b"\n".join(json.dumps(row).encode() for row in events))
    stress = pin("stress.jsonl", json.dumps({"software_revision": "9" * 40,
        "checkpoint_revision": revision, "device_uuid": slot.device_uuid, "environment": "math"}).encode())
    manifest = {"schema_version": 1, "mode": "observed_live", "qualified": False,
        "identity": identity, "budget": capacity_budget(), "evidence": {"natural_attempts": natural,
        "historical_stress": {**stress, "software_revision": "9" * 40, "qualified_for_current_runtime": False}}}
    def save():
        pinned = pin("operator.json", json.dumps(manifest).encode())
        monkeypatch.setenv("RELIQUARY_PROOF_OBSERVED_MANIFEST", pinned["path"])
        monkeypatch.setenv("RELIQUARY_PROOF_OBSERVED_MANIFEST_SHA256", pinned["sha256"])
    save()
    return SimpleNamespace(pool=pool, manifest=manifest, save=save, revision=revision, events=events)


def test_live_authorization_retains_real_numeric_rejection_and_never_qualifies(rollout):
    report = observed.authorize_observed_live(rollout.pool, rollout.revision)
    assert report["qualified"] is False and report["mode"] == "observed_live"
    assert report["natural_observations"]["passing_groups"] == 3
    assert report["natural_observations"]["numerical_rejections_requires_review"] == {"logprob_mismatch": 1}
    assert report["historical_stress_observations"]["qualified_for_current_runtime"] is False
    observed.assert_proof_start_authorized(report, rollout.pool, rollout.revision)
    assert rollout.pool._validate_health.call_count == 2
    report["natural_observations"]["numerical_rejections_requires_review"] = {}
    with pytest.raises(ValueError, match="differs"):
        observed.assert_proof_start_authorized(report, rollout.pool, rollout.revision)


@pytest.mark.parametrize("field", ["controller_software_revision", "worker_software_revision",
    "worker_id", "session_id", "runtime_fingerprint_hash", "transport_sha256", "proof_path_hash",
    "checkpoint_revision", "device_uuid", "environments"])
def test_wrong_pinned_identity_fails(rollout, field):
    rollout.manifest["identity"][field] = "wrong"
    rollout.save()
    with pytest.raises(ValueError, match="identity mismatch"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)


def test_missing_operator_mode_and_unqualified_default_are_refused(rollout, monkeypatch):
    monkeypatch.delenv("RELIQUARY_PROOF_CAPACITY_MODE")
    with pytest.raises(ValueError, match="explicit operator"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)
    with pytest.raises(RuntimeError, match="not qualified"):
        observed.assert_proof_start_authorized({}, None, "")
    observed.assert_proof_start_authorized({"qualified": True}, None, "")


def test_no_extra_slots_or_unadopted_checkpoint(rollout):
    rollout.pool.health.slots *= 2
    with pytest.raises(ValueError, match="one GPU"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)
    rollout.pool.health.slots = rollout.pool.health.slots[:1]
    rollout.pool.health.checkpoint = None
    with pytest.raises(ValueError, match="adopted"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)


def test_evidence_bytes_and_window_budget_cannot_change(rollout):
    from pathlib import Path
    rollout.manifest["budget"]["drain_seconds"] = 1
    rollout.save()
    with pytest.raises(ValueError, match="bounded"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)
    rollout.manifest["budget"]["drain_seconds"] = 360
    rollout.save()
    Path(rollout.manifest["evidence"]["natural_attempts"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        observed.authorize_observed_live(rollout.pool, rollout.revision)


@pytest.mark.parametrize("fault", ["missing", "failed", "hidden_numeric_rejection"])
def test_incomplete_or_rewritten_native_ledger_rejected(rollout, fault):
    events = rollout.events
    if fault == "missing":
        events.pop(-2)
    elif fault == "failed":
        events[-2]["outcome"] = "timed_out"
    else:
        events[-1]["numerical_rejections_requires_review"] = {}
    with pytest.raises(ValueError):
        observed._natural_summary(b"\n".join(json.dumps(e).encode() for e in events),
            rollout.pool.health.checkpoint.model_dump(), ["math", "code", "logic"])
