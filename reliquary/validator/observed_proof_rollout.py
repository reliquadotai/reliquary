"""Explicit operator-authorized live observation; never a capacity qualification.

This CPU startup policy leaves all proof verdicts, transport and scheduler
deadlines intact. Its pinned evidence may include numerical rejections and an
older stress run: those remain visible, and are not promoted to qualification.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import re

from reliquary.shared.strict_json import strict_json_loads


def observed_live_requested() -> bool:
    mode = os.environ.get("RELIQUARY_PROOF_CAPACITY_MODE", "qualified").strip()
    if mode not in {"qualified", "observed_live"}:
        raise ValueError("proof capacity mode must be qualified or observed_live")
    return mode == "observed_live"


def _read_pinned(path, sha):
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("observed rollout evidence needs an absolute path")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("observed rollout requires a lowercase SHA256 pin")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("observed rollout file digest mismatch")
    return raw


def _natural_summary(raw, checkpoint, environments):
    events = [strict_json_loads(line) for line in raw.splitlines() if line.strip()]
    if (not events or events[0].get("event") != "run_started"
            or events[-1].get("event") != "run_complete"
            or any(row.get("event") not in {"run_started", "admission", "proof", "run_complete"}
                   for row in events)
            or sum(row.get("event") == "run_started" for row in events) != 1
            or sum(row.get("event") == "run_complete" for row in events) != 1
            or events[0].get("checkpoint") != checkpoint):
        raise ValueError("observed natural evidence is incomplete or checkpoint-mismatched")
    admitted, decided = {}, set()
    admission_rejects, proof_rejects, passes = Counter(), Counter(), Counter()
    for row in events[1:-1]:
        env = row.get("environment")
        if env not in environments:
            raise ValueError("observed natural evidence environment mismatch")
        if row["event"] == "admission":
            if row.get("outcome") == "rejected" and row.get("reason"):
                admission_rejects[row["reason"]] += 1
            elif (row.get("outcome") == "accepted" and isinstance(row.get("job_id"), str)
                    and row["job_id"] and row["job_id"] not in admitted
                    and re.fullmatch(r"[0-9a-f]{64}", row.get("input_sha256", ""))):
                admitted[row["job_id"]] = (env, row.get("input_sha256"))
            else:
                raise ValueError("observed natural admission failed or duplicated")
        elif row["event"] == "proof":
            job = row.get("job_id")
            if (job not in admitted or job in decided
                    or admitted[job] != (env, row.get("input_sha256"))):
                raise ValueError("observed proof decision is missing, duplicated or misbound")
            decided.add(job)
            if row.get("outcome") == "passed":
                passes[env] += 1
            elif row.get("outcome") == "rejected" and row.get("reason"):
                proof_rejects[row["reason"]] += 1
            else:
                raise ValueError("observed natural proof has an infrastructure failure")
        else:
            raise ValueError("observed natural evidence has an unexpected event")
    if set(admitted) != decided or any(passes[env] == 0 for env in environments):
        raise ValueError("observed rollout requires complete attempts and real passes for each ENV")
    summary = {"admitted_groups": len(admitted), "passing_groups": sum(passes.values()),
        "admission_rejections_by_reason": dict(admission_rejects),
        "proof_rejections_by_reason": dict(proof_rejects),
        "numerical_rejections_requires_review": {key: value for key, value in proof_rejects.items()
            if key in {"grail_fail", "logprob_mismatch", "seed_mismatch"}}}
    if any(events[-1].get(key) != value for key, value in summary.items()):
        raise ValueError("observed natural evidence summary does not conserve attempts")
    return {**summary, "passing_groups_by_environment": dict(passes)}


def authorize_observed_live(pool, activation_revision):
    """Validate the separate rollout authorization against the actual live pool.

    Called at CLI startup and again at the service authority boundary. No
    proof/adoption requests are made here; RemoteProofPool.start validates the
    authenticated health before this function, and normal adoption remains
    mandatory in ValidationService.
    """
    from reliquary import constants as c
    from reliquary.validator.observability import immutable_build_revision
    from reliquary.validator.proof_capacity import capacity_budget

    if not observed_live_requested():
        raise ValueError("observed rollout requires explicit operator mode")
    path = os.environ.get("RELIQUARY_PROOF_OBSERVED_MANIFEST", "")
    sha = os.environ.get("RELIQUARY_PROOF_OBSERVED_MANIFEST_SHA256", "")
    manifest = strict_json_loads(_read_pinned(path, sha))
    if (manifest.get("schema_version") != 1 or manifest.get("mode") != "observed_live"
            or manifest.get("qualified") is not False):
        raise ValueError("observed rollout is not a qualified capacity manifest")
    if not getattr(pool, "is_remote", False) or pool.health is None:
        raise ValueError("observed rollout requires authenticated remote proof health")
    health = pool._validate_health(pool.health)
    if len(health.slots) != 1:
        raise ValueError("observed rollout is limited to one GPU and one proof slot")
    if health.checkpoint is None or health.checkpoint.revision != activation_revision:
        raise ValueError("observed rollout requires the exact adopted activation checkpoint")
    slot = health.slots[0]
    if slot.revision != activation_revision:
        raise ValueError("observed rollout slot checkpoint mismatch")
    budget = capacity_budget()
    environments = sorted(c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)
    if (budget.get("mode") != "fill_closed_bounded" or len(environments) != 3
            or budget.get("drain_seconds") != 360
            or budget.get("wall_seconds") != 1800 or manifest.get("budget") != budget):
        raise ValueError("observed rollout requires the pinned bounded 1800s window / 360s drain")
    identity = {
        "controller_software_revision": immutable_build_revision(),
        "worker_software_revision": health.software_revision,
        "worker_id": health.worker_id, "session_id": health.session_id,
        "proof_pipeline_depth": pool.pipeline_depth,
        "runtime_fingerprint_hash": pool.runtime_fingerprint["profile_hash"],
        "transport_sha256": health.transport_sha256, "proof_path_hash": health.proof_path_hash,
        "profile_id": c.PROTOCOL_PROFILE_ID, "model_revision": c.PROTOCOL_MODEL_REVISION,
        "generation_contract_sha256": health.generation_contract_sha256,
        "training_run_id": health.training_run_id, "repo_id": health.repo_id,
        "checkpoint_n": health.checkpoint.checkpoint_n, "checkpoint_revision": activation_revision,
        "device_id": slot.device_id, "physical_device": slot.physical_device,
        "device_uuid": slot.device_uuid, "hardware_class": slot.hardware_class,
        "environments": environments,
    }
    if (not re.fullmatch(r"[0-9a-f]{40}", identity["controller_software_revision"] or "")
            or manifest.get("identity") != identity):
        raise ValueError("observed rollout controller/worker/runtime/checkpoint identity mismatch")
    evidence = manifest.get("evidence", {})
    natural = evidence.get("natural_attempts", {})
    summary = _natural_summary(_read_pinned(natural.get("path"), natural.get("sha256")),
        health.checkpoint.model_dump(), environments)
    historical = evidence.get("historical_stress", {})
    if (historical.get("qualified_for_current_runtime") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", historical.get("software_revision", ""))):
        raise ValueError("historical stress must remain explicitly unqualified for this runtime")
    stress = [strict_json_loads(line) for line in
        _read_pinned(historical.get("path"), historical.get("sha256")).splitlines() if line.strip()]
    if not stress or any(
            row.get("software_revision") != historical["software_revision"]
            or row.get("checkpoint_revision") != activation_revision
            or row.get("device_uuid") != slot.device_uuid
            or row.get("environment") not in environments for row in stress):
        raise ValueError("historical stress source/checkpoint/GPU evidence mismatch")
    return {"mode": "observed_live", "qualified": False,
        "operator_authorized": True, "authorization_sha256": sha,
        "identity": identity, "budget": budget, "benchmark_evidence": evidence,
        "natural_observations": summary,
        "historical_stress_observations": {"groups": len(stress),
            "source_revision": historical["software_revision"], "qualified_for_current_runtime": False},
        "limitations": ["capacity_not_fully_qualified", "historical_stress_not_current_qualification",
                        "numerical_rejections_retained", "live_miner_training_adoption_pending"]}


def assert_proof_start_authorized(report, pool, activation_revision):
    """Qualified startup is unchanged; live observation must revalidate its pin."""
    if report.get("qualified") is True:
        return
    if not observed_live_requested():
        raise RuntimeError("auction-v3 proof capacity is not qualified")
    if report != authorize_observed_live(pool, activation_revision):
        raise ValueError("observed rollout startup authorization differs from the pinned manifest")
