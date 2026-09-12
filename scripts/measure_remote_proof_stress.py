#!/usr/bin/env python3
"""Measure NONADMISSIBLE full-context proof and CPU workloads on the real mTLS plane.

Run inside the same CPU-limited controller image/container as production,
against an isolated adopted GPU worker. No wallet, admission, training, storage
or chain client is created. All stress verdicts remain invalid submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_group(*, tokenizer, model, environment, group_id):
    from reliquary import constants as c
    from reliquary.shared.hf_compat import resolve_max_context_length
    from reliquary.shared.modeling import resolve_eos_token_ids
    context = resolve_max_context_length(model.config)
    cap = c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[environment]
    # This measured envelope is deliberately scoped to the release contract.
    if context != 32768 or cap != 8192 or c.M_ROLLOUTS != 16:
        raise ValueError('stress implementation requires V1 32768/8192/M16')
    vocab = int(model.config.vocab_size)
    eos = sorted(resolve_eos_token_ids(model, tokenizer))
    if not eos:
        raise ValueError('stress requires the real tokenizer/model EOS identity')
    rng = random.Random(group_id)
    commits = []
    for index in range(c.M_ROLLOUTS):
        tokens = [rng.randrange(vocab) for _ in range(context)]
        tokens[-1] = eos[0]  # exercise terminal forced-CDF diagnostics too
        commits.append({'tokens': tokens,
            'commitments': [{'sketch': rng.randrange(2**63)} for _ in tokens],
            'rollout': {'prompt_length': context-cap, 'completion_length': cap,
                'token_logprobs': [-1.] * cap, 'rollout_index': index,
                'forced': False, 'eos_token_id': eos[0]}})
    return commits


def measure_group(invocation, *, pool, tokenizer, environments, controller, forensic_directory=None):
    from reliquary import constants as c
    from reliquary.environment.forced_sampling import u_at
    from reliquary.validator.proof_capacity_combined import SCOPE
    from reliquary.validator.proof_stress import (
        measure_postproof_stress,
        validate_postproof_measurement,
    )
    from reliquary.validator.remote_proof_protocol import RemoteProofMeasurement
    checkpoint, health = pool._adopted, pool.health
    model = pool.proxies()[invocation.device_id]
    slot = next(s for s in health.slots if s.device_id == invocation.device_id)
    env, group_id = invocation.environment, invocation.candidate.job_id
    randomness = hashlib.sha256(group_id.encode()).hexdigest()
    # Input preparation is outside the GPU timer. Natural E2E timing is added
    # separately by the qualifier to cover ordinary fixed/group overhead.
    commits = make_group(tokenizer=tokenizer, model=model, environment=env, group_id=group_id)
    cap, window, prompt_idx = c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV[env], 1, 0
    uniforms = [[u_at(randomness, prompt_idx, checkpoint.revision, i, j)
                 for j in range(cap)] for i in range(c.M_ROLLOUTS)]
    started = time.perf_counter()
    with pool.measure_group() as receipts:
        proofs = [pool.prove(invocation.device_id, commit, randomness, seed,
                    window=window, environment=env, checkpoint=checkpoint)
                  for commit, seed in zip(commits, uniforms, strict=True)]
    remote_seconds = time.perf_counter() - started
    cpu = measure_postproof_stress(commits=commits, proofs=proofs, tokenizer=tokenizer,
        environment=environments[env], randomness=randomness, prompt_idx=prompt_idx,
        checkpoint_revision=checkpoint.revision, proof_model=model, forensic_directory=forensic_directory)
    validate_postproof_measurement(cpu, rollout_count=c.M_ROLLOUTS, completion_tokens=cap)
    seconds = time.perf_counter() - started
    entropy_count = min(cap, 64) if health.utility_telemetry_enabled else 0
    if (pool._adopted != checkpoint or len(receipts) != c.M_ROLLOUTS
            or any(p.seed_n_positions != cap or p.checked != c.CHALLENGE_K
                   or len(p.completion_entropies) != entropy_count for p in proofs)):
        raise ValueError('stress did not execute every seed position/challenge on the adopted checkpoint')
    return {'scope': SCOPE, 'synthetic': True, 'valid_submission': False,
        'complete_remote_group': True, 'environment': env, 'seconds': seconds, 'remote_seconds': remote_seconds,
        'profile_id': health.profile_id, 'model_revision': c.PROTOCOL_MODEL_REVISION,
        'software_revision': health.software_revision, 'checkpoint_revision': checkpoint.revision,
        'checkpoint_n': checkpoint.checkpoint_n, 'repo_id': checkpoint.repo_id,
        'training_run_id': checkpoint.training_run_id, 'session_id': health.session_id,
        'configured_slots': {s.device_id: s.device_uuid.casefold() for s in health.slots},
        'runtime_fingerprint_hash': slot.runtime['profile_hash'],
        'hardware_class': slot.hardware_class, 'device_uuid': slot.device_uuid.casefold(),
        'device_id': slot.device_id, 'rollout_count': len(commits), 'window': window,
        'total_token_lengths': [len(commit['tokens']) for commit in commits],
        'completion_token_lengths': [len(p.completion_chosen_probs) for p in proofs],
        'seed_position_counts': [p.seed_n_positions for p in proofs],
        'challenge_counts': [p.checked for p in proofs],
        'entropy_sample_counts': [len(p.completion_entropies) for p in proofs],
        'group_id': group_id, 'controller': controller, 'wire_receipts': receipts,
        'postproof_cpu': cpu,
        'kernel_verdicts': [{'all_passed': p.all_passed, 'passed': p.passed,
                            'checked': p.checked, 'seed_n_hard_mismatch': p.seed_n_hard_mismatch}
                           for p in proofs],
        'remote_proof': RemoteProofMeasurement(worker_id=health.worker_id,
                            transport_sha256=health.transport_sha256,
                            pipeline_depth=pool.pipeline_depth).model_dump()}


def measure(*, pool, tokenizer, environments, output, groups, timeout):
    from reliquary.validator.proof_capacity_combined import controller_identity
    from reliquary.validator.proof_scheduler import (
        GlobalProofScheduler,
        ProofExecution,
        ProofPlan,
        ProofPlanOutcome,
        RankedProof,
    )
    from reliquary.validator.remote_proof_protocol import canonical_bytes
    pool.assert_ready()
    controller = controller_identity()
    output = Path(output)
    if os.stat(output.parent).st_dev != controller["state_filesystem_device"]:
        raise ValueError("stress output must use the controller state filesystem")
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    lock = threading.Lock()
    prefix = uuid.uuid4().hex
    # Exercise every configured slot concurrently, including shared-GPU
    # contention. The qualifier aggregates by physical UUID, never slot count.
    devices = tuple(pool.devices)
    count = 0
    with os.fdopen(fd, 'wb') as handle:
        def execute(invocation):
            nonlocal count
            failure = None
            try:
                row = measure_group(invocation, pool=pool, tokenizer=tokenizer,
                                    environments=environments, controller=controller, forensic_directory=output.parent)
            except Exception as exc:  # noqa: BLE001 - persist the failed attempt, then re-raise
                failure = exc
                row = {'scope': 'full-envelope-stress-failure', 'synthetic': True,
                    'valid_submission': False, 'complete_remote_group': False,
                    'environment': invocation.environment, 'device_id': invocation.device_id,
                    'group_id': invocation.candidate.job_id, 'error_type': type(exc).__name__}
            with lock:
                handle.write(canonical_bytes(row) + b'\n')
                handle.flush()
                os.fsync(handle.fileno())
                count += 1
            if failure is not None:
                raise failure
            # Rejected by design. complete_all counts executions, not passes.
            return ProofExecution(passed=False, reason='nonadmissible capacity stress')

        def on_device(device):
            with GlobalProofScheduler(devices=(device,), environments=tuple(environments),
                    checkpoint_revision=pool._adopted.revision, proof_callable=execute) as scheduler:
                plans = [ProofPlan(plan_id=f'{prefix}:{device}:{env}', environment=env,
                    checkpoint_revision=pool._adopted.revision, required_passes=0,
                    max_attempts=groups, complete_all=True, deadline_at=time.monotonic()+timeout,
                    candidates=[RankedProof(job_id=f'{prefix}:{device}:{env}:{i}', rank=i,
                        prompt_key=(env, i), payload=None) for i in range(groups)]) for env in environments]
                results = [h.result(timeout=timeout+5) for h in scheduler.submit_many(plans)]
                if any(r.outcome is not ProofPlanOutcome.COMPLETED for r in results):
                    raise RuntimeError('stress scheduler aborted; retain incomplete evidence, no qualification')
        with ThreadPoolExecutor(max_workers=len(devices)) as workers:
            list(workers.map(on_device, devices))
    pool.assert_ready()
    if count != groups*len(environments)*len(devices):
        raise ValueError('stress workload incomplete')
    return {'qualified': False, 'valid_submission': False, 'groups': count,
        'samples_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'path': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hf-repo-id', required=True)
    parser.add_argument('--checkpoint-n', type=int, required=True)
    parser.add_argument('--checkpoint-revision', required=True)
    parser.add_argument('--groups-per-environment-and-device', type=int, default=20)
    parser.add_argument('--timeout-seconds', type=float, default=86400.)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.exists()
            or not 20 <= args.groups_per_environment_and_device <= 100
            or not 0 < args.timeout_seconds <= 86400):
        parser.error('require new absolute output, 20..100 groups and timeout (0,86400]')
    from reliquary import constants as c
    from reliquary.environment import load_environments
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.observability import immutable_build_revision
    from reliquary.validator.remote_proof import RemoteProofPool, executor_mode
    if executor_mode() != 'remote':
        parser.error('select explicit remote proof mode and isolated worker')
    pool = RemoteProofPool.from_environment(repo_id=args.hf_repo_id)
    try:
        pool.start()
        if immutable_build_revision() != pool.health.software_revision:
            raise ValueError('stress controller and GPU require the same immutable revision')
        pool.bind_checkpoint(args.checkpoint_n, args.hf_repo_id, args.checkpoint_revision)
        pool.reload(pool.devices[0], None, args.checkpoint_revision, args.hf_repo_id)
        tokenizer = load_tokenizer(args.hf_repo_id, revision=args.checkpoint_revision, token=False)
        print(json.dumps(measure(pool=pool, tokenizer=tokenizer,
            environments=load_environments(list(c.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV)),
            output=args.output, groups=args.groups_per_environment_and_device,
            timeout=args.timeout_seconds), sort_keys=True))
    finally:
        pool.close(force=True)


if __name__ == '__main__':
    main()
