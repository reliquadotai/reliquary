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
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prepare_candidate(row, *, pool, environments, tokenizer, index):
    from reliquary import constants as c
    from reliquary.protocol.submission import BatchSubmissionRequest
    from reliquary.protocol.signatures import verify_envelope_signature
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
    if not response.accepted or len(pending) != 1:
        raise ValueError(f"corpus candidate failed normal admission: {response.reason}")
    return env_name, RankedProof(job_id=f"benchmark:{index}", rank=index,
        prompt_key=(index, request.prompt_idx), payload=_ScheduledProofPayload(batcher, pending[0]))


def measure(corpus, *, output, pool, tokenizer, environments, timeout):
    from reliquary.validator.proof_measurements import ProofMeasurements
    from reliquary.validator.proof_scheduler import GlobalProofScheduler, ProofPlan, ProofPlanOutcome
    from reliquary.validator.service import ValidationService
    from reliquary.shared.strict_json import strict_json_loads
    from reliquary.validator.remote_proof_protocol import canonical_bytes

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--checkpoint-n", type=int, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
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
            environments=load_environments(list(c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)), timeout=args.timeout_seconds)
        print(json.dumps(report, sort_keys=True))
    finally:
        pool.close(force=True)


if __name__ == "__main__":
    main()
