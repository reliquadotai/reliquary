#!/usr/bin/env python3
"""Measure real full proof groups on an isolated remote GPU worker.

No validator HTTP server, wallet, checkpoint publication, training or chain
client is created. This command does not call or weaken production capacity
qualification. Its JSONL output is an input to qualify_proof_capacity.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AdmissionRejected(ValueError):
    """A normal admission verdict, distinct from malformed input/infra errors."""

    def __init__(self, response):
        self.reason = response.reason.value
        super().__init__(f"corpus candidate failed normal admission: {self.reason}")


def prepare_candidate(row, *, pool, environments, tokenizer, index):
    from reliquary import constants as c
    from reliquary.protocol.signatures import verify_envelope_signature
    from reliquary.protocol.submission import BatchSubmissionRequest
    from reliquary.validator.batcher import _ScheduledProofPayload
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.validator.dedup import RolloutHashSet
    from reliquary.validator.proof_scheduler import RankedProof
    from reliquary.validator.service import open_grpo_window

    if not isinstance(row, dict) or set(row) != {"environment", "randomness", "request"}:
        raise ValueError("corpus row requires environment, randomness and complete signed request")
    env_name, randomness = row["environment"], row["randomness"]
    if env_name not in environments or not isinstance(randomness, str) or not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{2,256}", randomness):
        raise ValueError("invalid corpus environment/randomness")
    request = BatchSubmissionRequest.model_validate(row["request"])
    if (len(request.rollouts) != c.M_ROLLOUTS or request.checkpoint_hash != pool._adopted.revision
            or request.generation_profile_id != c.PROTOCOL_PROFILE_ID):
        raise ValueError("corpus does not bind the adopted profile/checkpoint and complete group")
    if not verify_envelope_signature(randomness=randomness, **{
        key: getattr(request, key) for key in ("miner_hotkey", "window_start", "prompt_idx", "merkle_root",
            "checkpoint_hash", "drand_round", "nonce", "protocol_version", "generation_profile_id", "envelope_signature")
    }):
        raise ValueError("corpus envelope signature is invalid")
    # Fresh isolated state per group avoids paying/retaining any benchmark
    # candidate. All signature, token, prompt, grade and proof-dependent gates
    # remain the ordinary batcher's code. No HTTP arrival-time claim is made.
    batcher = open_grpo_window(request.window_start, environments[env_name],
        next(iter(pool.proxies().values())), tokenizer=tokenizer,
        cooldown_map=CooldownMap(c.BATCH_PROMPT_COOLDOWN_WINDOWS),
        hash_set=RolloutHashSet(c.HASH_DEDUP_RETENTION_WINDOWS),
        verify_commitment_proofs_fn=pool.verifier_for_window(request.window_start, env_name, request.checkpoint_hash))
    if not batcher.difficulty_auction_enabled:
        raise ValueError("isolated benchmark requires the deferred-proof auction path")
    batcher.current_checkpoint_hash = request.checkpoint_hash
    batcher.randomness = randomness
    batcher.set_prompt_range()
    response = batcher.accept_submission(request)
    pending = batcher.pending_submissions()
    if not response.accepted:
        raise AdmissionRejected(response)
    if len(pending) != 1:
        raise ValueError("accepted corpus candidate did not enter exactly one pending slot")
    return env_name, RankedProof(job_id=f"benchmark:{index}", rank=index,
        prompt_key=(index, request.prompt_idx), payload=_ScheduledProofPayload(batcher, pending[0]))


def measure(corpus, *, output, pool, tokenizer, environments, timeout, combined_natural=False):
    if combined_natural:
        return _measure_combined_natural(corpus, output=output, pool=pool,
            tokenizer=tokenizer, environments=environments, timeout=timeout)
    from reliquary.shared.strict_json import strict_json_loads
    from reliquary.validator.proof_measurements import ProofMeasurements
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler,
        ProofPlan,
        ProofPlanOutcome,
    )
    from reliquary.validator.remote_proof_protocol import canonical_bytes
    from reliquary.validator.service import ValidationService

    pool.assert_ready()
    revision = pool._adopted.revision
    candidates = {name: [] for name in environments}
    seen = set()
    corpus_hash = hashlib.sha256()
    with Path(corpus).open("rb") as source:
        for index, raw in enumerate(source):
            corpus_hash.update(raw)
            if not raw.strip():
                continue
            if len(raw) > 256 * 1024 * 1024:
                raise ValueError("corpus group exceeds the bounded input size")
            row = strict_json_loads(raw)
            digest = hashlib.sha256(canonical_bytes(row)).hexdigest()
            if digest in seen:
                raise ValueError("duplicate corpus group cannot count as independent evidence")
            seen.add(digest)
            env, candidate = prepare_candidate(row, pool=pool,
                environments=environments, tokenizer=tokenizer, index=index)
            candidates[env].append(candidate)
    if any(not values for values in candidates.values()):
        raise ValueError("corpus must exercise every active environment")
    recorder = ProofMeasurements(output, pool)
    context = SimpleNamespace(_proof_models=pool.proxies(), _proof_measurements=recorder)
    with GlobalProofScheduler(devices=pool.devices, environments=tuple(environments),
        checkpoint_revision=revision,
        proof_callable=lambda invocation: ValidationService._execute_scheduled_proof(context, invocation)) as scheduler:
        plans = [ProofPlan(plan_id=f"benchmark:{env}", environment=env,
            checkpoint_revision=revision, candidates=values, required_passes=0,
            max_attempts=len(values), complete_all=True, deadline_at=time.monotonic() + timeout)
            for env, values in candidates.items()]
        results = [handle.result(timeout=timeout + 5) for handle in scheduler.submit_many(plans)]
        if any(r.outcome is not ProofPlanOutcome.COMPLETED or len(r.winner_job_ids) != len(candidates[p.environment])
                for p, r in zip(plans, results, strict=True)):
            raise RuntimeError("benchmark contains rejected/failed/timed-out groups; retain evidence and diagnose")
    pool.assert_ready()
    if corpus_hash.hexdigest() != hashlib.sha256(Path(corpus).read_bytes()).hexdigest():
        raise ValueError("benchmark corpus changed during measurement")
    return {"scope": "isolated-scheduler-proof-groups-no-public-admission", "qualified": False,
        "samples": str(output), "samples_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        "corpus_sha256": corpus_hash.hexdigest(),
        "groups": sum(map(len, candidates.values())), "worker_id": pool.worker_id,
        "checkpoint": pool._adopted.model_dump(), "software_revision": pool.health.software_revision,
        "runtime_fingerprint_hash": pool.runtime_fingerprint["profile_hash"]}


def _new_evidence(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("evidence path must be absolute")
    return os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb")


def _file_sha256(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _measure_combined_natural(corpus, *, output, pool, tokenizer, environments, timeout):
    """Retain every attempt; export only authentic complete passing groups.

    OUT_OF_ZONE is normal base-model reward supply, so it may be skipped.
    Proof rejects remain explicit evidence; errors/timeouts invalidate the run.
    This does not qualify capacity or waive review of numerical proof failures.
    """
    from reliquary.constants import M_ROLLOUTS
    from reliquary.shared.strict_json import strict_json_loads
    from reliquary.validator.batcher import ValidSubmission
    from reliquary.validator.proof_measurements import ProofMeasurements
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler,
        ProofDecisionStatus,
        ProofPlan,
        ProofPlanOutcome,
    )
    from reliquary.validator.remote_proof_protocol import canonical_bytes
    from reliquary.validator.service import ValidationService

    output = Path(output)
    attempts_path = output.with_name(output.name + ".attempts.jsonl")
    raw_path = output.with_name(output.name + ".proof-attempts.jsonl")
    if any(path.exists() or path.is_symlink() for path in (output, attempts_path, raw_path)):
        raise FileExistsError("combined evidence requires three new output paths")
    pool.assert_ready()
    checkpoint = pool._adopted
    corpus_sha = _file_sha256(corpus)
    candidates = {name: [] for name in environments}
    inputs, seen = {}, set()
    admission_reasons, proof_reasons = Counter(), Counter()
    passed_jobs, rejected_jobs = set(), set()
    with _new_evidence(attempts_path) as ledger:
        def record(row):
            ledger.write(canonical_bytes(row) + b"\n")
            ledger.flush()
            os.fsync(ledger.fileno())

        record({"event": "run_started", "corpus_sha256": corpus_sha,
                "scope": "combined-natural-all-attempts/v1", "checkpoint": checkpoint.model_dump(),
                "worker_id": pool.worker_id, "session_id": pool.health.session_id})
        try:
            with Path(corpus).open("rb") as source:
                for index, raw in enumerate(source):
                    if not raw.strip():
                        continue
                    event = {"event": "admission", "row_index": index,
                             "raw_input_sha256": hashlib.sha256(raw).hexdigest()}
                    admission_started = None
                    try:
                        if len(raw) > 256 * 1024 * 1024:
                            raise ValueError("corpus group exceeds the bounded input size")
                        row = strict_json_loads(raw)
                        digest = hashlib.sha256(canonical_bytes(row)).hexdigest()
                        event.update(input_sha256=digest, environment=row.get("environment"))
                        if digest in seen:
                            raise ValueError("duplicate corpus group cannot count as independent evidence")
                        seen.add(digest)
                        admission_started = time.perf_counter()
                        env, candidate = prepare_candidate(row, pool=pool,
                            environments=environments, tokenizer=tokenizer, index=index)
                    except AdmissionRejected as exc:
                        event.update(outcome="rejected", reason=exc.reason,
                                     admission_seconds=time.perf_counter() - admission_started)
                        record(event)
                        admission_reasons[exc.reason] += 1
                        if exc.reason != "out_of_zone":
                            raise
                        continue
                    except Exception as exc:
                        if admission_started is not None:
                            event["admission_seconds"] = time.perf_counter() - admission_started
                        record({**event, "outcome": "error", "error_type": type(exc).__name__})
                        raise
                    event.update(outcome="accepted", job_id=candidate.job_id,
                                 admission_seconds=time.perf_counter() - admission_started)
                    record(event)
                    inputs[candidate.job_id] = digest
                    candidates[env].append(candidate)
            if any(not values for values in candidates.values()):
                raise ValueError("corpus has no admitted candidate for an active environment")
            recorder = ProofMeasurements(raw_path, pool)
            context = SimpleNamespace(_proof_models=pool.proxies(), _proof_measurements=recorder)
            by_job = {candidate.job_id: candidate for values in candidates.values() for candidate in values}
            with GlobalProofScheduler(devices=pool.devices, environments=tuple(environments),
                    checkpoint_revision=checkpoint.revision,
                    proof_callable=lambda invocation: ValidationService._execute_scheduled_proof(context, invocation)) as scheduler:
                plans = [ProofPlan(plan_id=f"benchmark:{env}", environment=env,
                    checkpoint_revision=checkpoint.revision, candidates=values, required_passes=0,
                    max_attempts=len(values), complete_all=True, deadline_at=time.monotonic() + timeout)
                    for env, values in candidates.items()]
                results = [handle.result(timeout=timeout + 5) for handle in scheduler.submit_many(plans)]
                failed = False
                for result in results:
                    failed |= result.outcome is not ProofPlanOutcome.COMPLETED
                    for decision in result.decisions:
                        pending = by_job[decision.job_id].payload.pending
                        rejected = getattr(pending, "reject_response", None)
                        reason = rejected.reason.value if rejected is not None else decision.reason
                        event = {"event": "proof", "job_id": decision.job_id,
                            "input_sha256": inputs[decision.job_id], "environment": result.environment,
                            "outcome": decision.status.value, "reason": reason,
                            "device_id": decision.device_id}
                        record(event)
                        if decision.status is ProofDecisionStatus.PASSED:
                            if not isinstance(decision.value, ValidSubmission) or len(decision.value.rollouts) != M_ROLLOUTS:
                                raise ValueError("scheduler passing value is not a complete ValidSubmission")
                            passed_jobs.add(decision.job_id)
                        elif decision.status is ProofDecisionStatus.REJECTED:
                            rejected_jobs.add(decision.job_id)
                            proof_reasons[reason or "proof_rejected"] += 1
                        else:
                            failed = True
                if failed or passed_jobs | rejected_jobs != set(inputs):
                    raise RuntimeError("combined natural run contains infrastructure failure/timeout/incomplete decisions")
            pool.assert_ready()
            if pool._adopted != checkpoint or _file_sha256(corpus) != corpus_sha:
                raise ValueError("checkpoint or corpus changed during measurement")
            measured = [strict_json_loads(raw) for raw in raw_path.read_bytes().splitlines() if raw.strip()]
            if (len(measured) != len(inputs) or {row["job_id"] for row in measured} != set(inputs)
                    or any(row.get("infrastructure_error_type") is not None for row in measured)):
                raise ValueError("raw proof attempts missing, duplicated or contain infrastructure failures")
            passing = [row for row in measured if row["job_id"] in passed_jobs]
            if any(row.get("proof_passed") is not True or row.get("complete_remote_group") is not True
                    or len(row.get("wire_receipts", [])) != M_ROLLOUTS for row in passing):
                raise ValueError("passing group has no complete authenticated proof receipt")
            if any(row.get("proof_passed") is not False for row in measured if row["job_id"] in rejected_jobs):
                raise ValueError("rejected proof was incorrectly marked passing")
            summary = {"admitted_groups": len(inputs), "passing_groups": len(passing),
                "admission_rejections_by_reason": dict(admission_reasons),
                "proof_rejections_by_reason": dict(proof_reasons),
                "numerical_rejections_requires_review": {key: value for key, value in proof_reasons.items()
                    if key in {"grail_fail", "logprob_mismatch", "seed_mismatch"}}}
            record({"event": "run_complete", **summary})
        except BaseException as exc:
            record({"event": "run_failed", "error_type": type(exc).__name__})
            raise
    bindings = {"attempt_ledger_sha256": _file_sha256(attempts_path),
                "proof_attempts_sha256": _file_sha256(raw_path), "corpus_sha256": corpus_sha}
    with _new_evidence(output) as target:
        for row in passing:
            target.write(canonical_bytes({**row, **bindings, "input_sha256": inputs[row["job_id"]]}) + b"\n")
        target.flush()
        os.fsync(target.fileno())
    return {"scope": "combined-natural-authentic-passes-with-complete-attempt-ledger/v1", "qualified": False,
        "proof_timing_scope": "post-admission scheduler execution including mTLS GPU and postproof CPU",
        "admission_timing_scope": "separate actual prepare_candidate/accept_submission elapsed seconds in attempt ledger",
        "full_http_or_grader_capacity_qualified": False,
        "samples": str(output), "samples_sha256": _file_sha256(output), **bindings, **summary,
        "attempt_ledger": str(attempts_path), "proof_attempts": str(raw_path),
        "corpus": str(Path(corpus).resolve()), "checkpoint": checkpoint.model_dump(),
        "worker_id": pool.worker_id, "software_revision": pool.health.software_revision,
        "runtime_fingerprint_hash": pool.runtime_fingerprint["profile_hash"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--checkpoint-n", type=int, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
    parser.add_argument("--combined-natural", action="store_true",
        help="retain every attempt and export only complete authentic passing groups for capacity v4")
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 86400 or args.output.exists() or not args.output.is_absolute():
        parser.error("use a new absolute output path and timeout in (0,86400]")
    from reliquary import constants as c
    from reliquary.environment import load_environments
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.observability import immutable_build_revision
    from reliquary.validator.remote_proof import RemoteProofPool, executor_mode
    if executor_mode() != "remote":
        parser.error("select explicit remote proof mode and a dedicated isolated worker")
    pool = RemoteProofPool.from_environment(repo_id=args.hf_repo_id)
    try:
        pool.start()
        if immutable_build_revision() != pool.health.software_revision:
            raise ValueError("benchmark controller and GPU worker require the same immutable image revision")
        pool.bind_checkpoint(args.checkpoint_n, args.hf_repo_id, args.checkpoint_revision)
        pool.reload(pool.devices[0], None, args.checkpoint_revision, args.hf_repo_id)
        tokenizer = load_tokenizer(args.hf_repo_id, revision=args.checkpoint_revision, token=False)
        report = measure(args.corpus, output=args.output, pool=pool, tokenizer=tokenizer,
            environments=load_environments(list(c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)), timeout=args.timeout_seconds,
            combined_natural=args.combined_natural)
        print(json.dumps(report, sort_keys=True))
    finally:
        pool.close(force=True)


if __name__ == "__main__":
    main()
