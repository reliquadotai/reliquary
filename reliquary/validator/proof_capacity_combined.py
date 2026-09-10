"""Explicit capacity v4: genuine E2E correctness plus full-envelope cost.

Stress rows are deliberately invalid submissions. Their timings can bound
resource use, never establish acceptance or replace the real passing groups.
The proof-wall budget begins after admission/reward grading. It does not
qualify public HTTP, admission sandbox, storage or concurrent ingress capacity.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

SCOPE = "full-envelope-mtls-plus-postproof-cpu/v1"
BOUND_METHOD = "max-stress-plus-max-authentic-e2e/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def controller_identity():
    """Require the measured Linux controller CPU and container resource limits."""
    cpu = Path('/proc/cpuinfo').read_text()
    model = next((line.split(':', 1)[1].strip() for line in cpu.splitlines()
                  if line.startswith('model name')), '')
    if not model:
        raise ValueError('controller CPU identity unavailable')
    limits = {name: (Path('/sys/fs/cgroup') / name).read_text().strip()
              for name in ('cpu.max', 'memory.max')}
    # Measure in the actual bounded production controller container, not an
    # unbounded host process that could understate post-proof CPU latency.
    if limits['cpu.max'].split()[0] == 'max' or limits['memory.max'] == 'max':
        raise ValueError('combined benchmark requires bounded controller CPU and memory')
    return {'cpu_model': model, **limits,
        'state_filesystem_device': os.stat(os.environ.get('RELIQUARY_STATE_DIR', '/state')).st_dev}


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


def validate_combined_evidence(evidence, *, device_uuids, environments,
        maximum_context_tokens, completion_caps, minimum_samples, natural_p95,
        natural_samples_sha256, controller=None, configured_slots=None):
    if (not isinstance(evidence, dict) or evidence.get('method') != BOUND_METHOD
            or evidence.get('budget_scope') != 'post-admission-proof-wall-only'):
        raise ValueError('missing explicit combined evidence')
    if (not isinstance(maximum_context_tokens, int) or maximum_context_tokens <= 0
            or not completion_caps or set(completion_caps) != set(environments)):
        raise ValueError('runtime context and full completion caps are required')
    if (evidence.get('natural_samples_sha256') != natural_samples_sha256
            or not SHA256.fullmatch(str(evidence.get('stress_samples_sha256', '')))):
        raise ValueError('combined sample digest mismatch')
    if evidence.get('controller') != (controller if controller is not None else controller_identity()):
        raise ValueError('controller hardware/resource limits differ from stress benchmark')
    from reliquary.validator.proof_capacity import capacity_budget
    from reliquary.validator.proof_stress import runtime_settings
    if evidence.get("window_budget") != capacity_budget():
        raise ValueError("active scheduler budget/deadline differs from benchmark")
    if evidence.get('cpu_settings') != runtime_settings():
        raise ValueError('postproof CPU settings differ from benchmark')
    if not configured_slots or evidence.get('configured_slots') != configured_slots:
        raise ValueError('runtime slots differ from measured contention topology')
    selection = evidence.get('natural_selection')
    if (not isinstance(selection, dict) or selection.get('numerical_rejections_requires_review') != {}
            or any(not SHA256.fullmatch(str(selection.get(key, ''))) for key in (
                'attempt_ledger_sha256', 'proof_attempts_sha256', 'corpus_sha256'))):
        raise ValueError('missing complete natural selection provenance or numerical failures unresolved')
    rows = evidence.get('by_environment_and_device', {})
    if set(rows) != set(environments):
        raise ValueError('combined evidence must cover every environment')
    bounds = {}
    for env, devices in rows.items():
        if set(devices) != set(device_uuids):
            raise ValueError('combined evidence must cover every physical GPU')
        bounds[env] = {}
        for device, row in devices.items():
            if (row.get('scope') != SCOPE or row.get('context_tokens') != maximum_context_tokens
                    or row.get('completion_tokens') != completion_caps[env]
                    or type(row.get('stress_sample_count')) is not int
                    or row['stress_sample_count'] < max(20, minimum_samples)):
                raise ValueError('incomplete full-envelope stress coverage')
            natural = positive(row.get('max_authentic_e2e_seconds'), 'authentic maximum')
            stress = positive(row.get('max_stress_seconds'), 'stress maximum')
            bound = positive(row.get('seconds_per_proof_bound'), 'combined bound')
            if natural < natural_p95[env][device] or not math.isclose(bound, natural + stress, rel_tol=0, abs_tol=1e-9):
                raise ValueError('combined bound understates measured work')
            bounds[env][device] = bound
    return bounds


def load_stress_samples(path, *, natural_path, corpus_path, expected_identity, device_uuids, completion_caps,
                        maximum_context_tokens, remote_proof, natural_samples,
                        natural_samples_sha256):
    """Validate complete typed-wire evidence; no synthetic validity assertion."""
    from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
    from reliquary.shared.strict_json import strict_json_loads
    from reliquary.validator.proof_capacity import capacity_budget
    from reliquary.validator.proof_stress import validate_postproof_measurement
    raw = Path(path).read_bytes()
    samples = {env: {device: [] for device in device_uuids} for env in completion_caps}
    controller = None
    cpu_settings = None
    configured_slots = None
    slot_counts = {}
    jobs, groups = set(), set()
    selection = validate_natural_selection(natural_path, corpus_path=corpus_path)
    checkpoint_identity = validate_natural_receipts(natural_path)
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = strict_json_loads(line)
        env, device = row.get('environment'), row.get('device_uuid')
        if env not in samples or device not in samples[env]:
            raise ValueError(f'unknown stress environment/GPU at row {number}')
        if (row.get('scope') != SCOPE or row.get('synthetic') is not True
                or row.get('valid_submission') is not False or 'proof_passed' in row
                or row.get('complete_remote_group') is not True):
            raise ValueError('stress cannot assert valid submission or omit full execution')
        for key, value in expected_identity.items():
            if row.get(key) != value:
                raise ValueError(f'stress {key} mismatch')
        if row.get('remote_proof') != remote_proof:
            raise ValueError('stress remote plane mismatch')
        identity = {key: row.get(key) for key in ('checkpoint_n', 'repo_id', 'training_run_id', 'session_id')}
        if any(value is None or value == '' for value in identity.values()):
            raise ValueError('stress checkpoint/session identity missing')
        if identity != checkpoint_identity:
            raise ValueError('stress checkpoint/session changed during measurement')
        group = row.get('group_id')
        if not isinstance(group, str) or not group or group in groups:
            raise ValueError('duplicate/missing stress group')
        groups.add(group)
        if controller is None:
            controller = row.get('controller')
        if not isinstance(controller, dict) or row.get('controller') != controller:
            raise ValueError('stress controller identity changed')
        slots = row.get('configured_slots')
        if (not isinstance(slots, dict) or not slots or set(slots.values()) != set(device_uuids)
                or slots.get(row.get('device_id')) != device):
            raise ValueError('stress slot topology is missing or invalid')
        if configured_slots is None:
            configured_slots = slots
        if slots != configured_slots:
            raise ValueError('stress contention topology changed')
        key = (env, row['device_id'])
        slot_counts[key] = slot_counts.get(key, 0) + 1
        cap = completion_caps[env]
        if (row.get('rollout_count') != M_ROLLOUTS
                or row.get('total_token_lengths') != [maximum_context_tokens] * M_ROLLOUTS
                or row.get('completion_token_lengths') != [cap] * M_ROLLOUTS
                or row.get('seed_position_counts') != [cap] * M_ROLLOUTS
                or row.get('challenge_counts') != [CHALLENGE_K] * M_ROLLOUTS):
            raise ValueError('stress must execute every context/policy/seed/challenge position')
        receipts = row.get('wire_receipts')
        if not isinstance(receipts, list) or len(receipts) != M_ROLLOUTS:
            raise ValueError('stress needs one authenticated wire receipt per rollout')
        for receipt in receipts:
            job = receipt.get('job_id')
            checkpoint = receipt.get('checkpoint', {})
            if (not isinstance(job, str) or not job or job in jobs
                    or not SHA256.fullmatch(str(receipt.get('content_sha256', '')))
                    or receipt.get('device_id') != row.get('device_id')
                    or receipt.get('window') != row.get('window')
                    or receipt.get('environment') != env or receipt.get('policy_tokens') != cap
                    or checkpoint.get('revision') != expected_identity['checkpoint_revision']
                    or any(checkpoint.get(k) != identity[k] for k in ('checkpoint_n', 'repo_id', 'training_run_id'))):
                raise ValueError('stress wire receipt is incomplete, duplicated or misbound')
            jobs.add(job)
        cpu = row.get('postproof_cpu')
        validate_postproof_measurement(cpu, rollout_count=M_ROLLOUTS, completion_tokens=cap)
        if cpu['settings'].get('auth_forensics_enabled') and cpu.get('forensic_io', {}).get('filesystem_device') != controller.get('state_filesystem_device'):
            raise ValueError('forensic append filesystem differs from production controller state')
        entropy_count = min(cap, 64) if cpu['settings']['utility_telemetry_enabled'] else 0
        if row.get('entropy_sample_counts') != [entropy_count] * M_ROLLOUTS:
            raise ValueError('stress must cover the effective worker entropy workload')
        if cpu_settings is None:
            cpu_settings = cpu['settings']
        if cpu['settings'] != cpu_settings:
            raise ValueError('CPU settings changed during measurement')
        seconds = positive(row.get('seconds'), 'stress duration')
        if seconds < positive(row.get('remote_seconds'), 'remote duration') + positive(cpu.get('seconds'), 'CPU duration'):
            raise ValueError('stress timer does not include all measured work')
        samples[env][device].append(seconds)
    if not configured_slots or any(slot_counts.get((env, slot), 0) < 20
            for env in completion_caps for slot in configured_slots):
        raise ValueError('at least 20 stress groups must exercise each configured slot')
    combined = {}
    for env, devices in samples.items():
        combined[env] = {}
        for device, durations in devices.items():
            if len(durations) < 20:
                raise ValueError('at least 20 stress groups per environment and GPU are required')
            natural_max = max(natural_samples[env][device])
            stress_max = max(durations)
            combined[env][device] = {'scope': SCOPE, 'context_tokens': maximum_context_tokens,
                'completion_tokens': completion_caps[env], 'stress_sample_count': len(durations),
                'max_authentic_e2e_seconds': natural_max, 'max_stress_seconds': stress_max,
                'seconds_per_proof_bound': natural_max + stress_max}
    return {'method': BOUND_METHOD, 'budget_scope': 'post-admission-proof-wall-only',
        'window_budget': capacity_budget(), 'natural_selection': selection,
        'natural_samples_sha256': natural_samples_sha256,
        'stress_samples_sha256': hashlib.sha256(raw).hexdigest(), 'controller': controller,
        'cpu_settings': cpu_settings, 'configured_slots': configured_slots,
        'by_environment_and_device': combined}


def validate_natural_receipts(path):
    from reliquary.constants import M_ROLLOUTS
    from reliquary.shared.strict_json import strict_json_loads
    identity, jobs = None, set()
    for line in Path(path).read_bytes().splitlines():
        if not line.strip():
            continue
        row = strict_json_loads(line)
        current = {key: row.get(key) for key in ('checkpoint_n', 'repo_id', 'training_run_id', 'session_id')}
        if any(value is None or value == '' for value in current.values()):
            raise ValueError('authentic checkpoint/session identity missing')
        if identity is None:
            identity = current
        if current != identity or row.get('proof_passed') is not True or row.get('complete_remote_group') is not True:
            raise ValueError('authentic evidence requires complete real passing groups on one checkpoint/session')
        receipts = row.get('wire_receipts')
        if not isinstance(receipts, list) or len(receipts) != M_ROLLOUTS:
            raise ValueError('authentic evidence needs every wire receipt')
        for index, receipt in enumerate(receipts):
            job, checkpoint = receipt.get('job_id'), receipt.get('checkpoint', {})
            if (not job or job in jobs or not SHA256.fullmatch(str(receipt.get('content_sha256', '')))
                    or receipt.get('environment') != row.get('environment')
                    or receipt.get('device_id') != row.get('device_id')
                    or receipt.get('window') != row.get('window')
                    or receipt.get('policy_tokens') != row.get('completion_token_lengths', [])[index]
                    or checkpoint.get('revision') != row.get('checkpoint_revision')
                    or any(checkpoint.get(k) != current[k] for k in ('checkpoint_n', 'repo_id', 'training_run_id'))):
                raise ValueError('authentic wire evidence is duplicated or misbound')
            jobs.add(job)
    if identity is None:
        raise ValueError('empty authentic evidence')
    return identity


def validate_natural_selection(path, *, corpus_path):
    """Bind each selected row to the complete immutable attempt/source ledger."""
    from collections import Counter

    from reliquary.shared.strict_json import strict_json_loads
    from reliquary.validator.remote_proof_protocol import canonical_bytes
    path = Path(path)
    ledger_raw = path.with_name(path.name + '.attempts.jsonl').read_bytes()
    proofs_raw = path.with_name(path.name + '.proof-attempts.jsonl').read_bytes()
    ledger = [strict_json_loads(line) for line in ledger_raw.splitlines() if line.strip()]
    proofs = [strict_json_loads(line) for line in proofs_raw.splitlines() if line.strip()]
    selected = [strict_json_loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    if (not ledger or ledger[0].get('event') != 'run_started'
            or ledger[0].get('scope') != 'combined-natural-all-attempts/v1'
            or ledger[-1].get('event') != 'run_complete'
            or any(row.get('event') not in {'run_started','admission','proof','run_complete'} for row in ledger)
            or sum(row.get('event') == 'run_started' for row in ledger) != 1
            or sum(row.get('event') == 'run_complete' for row in ledger) != 1):
        raise ValueError('natural selection requires a complete successful attempt ledger')
    corpus_hash = hashlib.sha256()
    source_inputs = {}
    with Path(corpus_path).open('rb') as source:
        for index, raw in enumerate(source):
            corpus_hash.update(raw)
            if raw.strip():
                source_inputs[index] = (hashlib.sha256(raw).hexdigest(),
                    hashlib.sha256(canonical_bytes(strict_json_loads(raw))).hexdigest())
    binding = {'attempt_ledger_sha256':hashlib.sha256(ledger_raw).hexdigest(),
        'proof_attempts_sha256':hashlib.sha256(proofs_raw).hexdigest(),
        'corpus_sha256':corpus_hash.hexdigest()}
    if ledger[0].get('corpus_sha256') != binding['corpus_sha256']:
        raise ValueError('natural corpus digest mismatch')
    admissions = [row for row in ledger if row['event']=='admission']
    if len(admissions)!=len(source_inputs) or {r.get('row_index') for r in admissions}!=set(source_inputs):
        raise ValueError('natural ledger omitted or duplicated a corpus admission')
    inputs, admission_rejections = {}, Counter()
    for row in admissions:
        if (source_inputs[row['row_index']] != (row.get('raw_input_sha256'),row.get('input_sha256'))
                or row.get('outcome') not in {'accepted','rejected'}):
            raise ValueError('natural admission does not match original corpus')
        positive(row.get('admission_seconds'), 'actual admission duration')
        if row['outcome']=='rejected':
            if row.get('reason') != 'out_of_zone':
                raise ValueError('unexpected admission failure cannot be selected away')
            admission_rejections[row['reason']] += 1
        else:
            job=row.get('job_id')
            if not job or job in inputs:
                raise ValueError('duplicate/missing admitted job')
            inputs[job]=row['input_sha256']
    decisions=[row for row in ledger if row['event']=='proof']
    if len(decisions)!=len(inputs) or {row.get('job_id') for row in decisions}!=set(inputs):
        raise ValueError('natural ledger omitted or duplicated a proof decision')
    if len(proofs)!=len(inputs) or {row.get('job_id') for row in proofs}!=set(inputs):
        raise ValueError('natural raw proof attempts incomplete')
    raw_by_job={row['job_id']:row for row in proofs}
    passing, proof_rejections = set(), Counter()
    for row in decisions:
        raw=raw_by_job[row['job_id']]
        if (row.get('input_sha256')!=inputs[row['job_id']] or row.get('outcome') not in {'passed','rejected'}
                or raw.get('infrastructure_error_type') is not None
                or raw.get('proof_passed') is not (row['outcome']=='passed')
                or row.get('environment')!=raw.get('environment') or row.get('device_id')!=raw.get('device_id')):
            raise ValueError('failed or misbound numerical execution cannot be selected away')
        if row['outcome']=='passed':
            passing.add(row['job_id'])
        else:
            proof_rejections[row.get('reason') or 'proof_rejected'] += 1
    if len(selected)!=len(passing) or {row.get('job_id') for row in selected}!=passing:
        raise ValueError('natural output omitted or added passing measurements')
    for row in selected:
        if (any(row.get(k)!=v for k,v in binding.items()) or row.get('input_sha256')!=inputs[row['job_id']]
                or {k:v for k,v in row.items() if k not in {*binding,'input_sha256'}} != raw_by_job[row['job_id']]):
            raise ValueError('selected natural row differs from original proof measurement')
    summary={'admitted_groups':len(inputs),'passing_groups':len(passing),
        'admission_rejections_by_reason':dict(admission_rejections),
        'proof_rejections_by_reason':dict(proof_rejections),
        'numerical_rejections_requires_review':{k:v for k,v in proof_rejections.items()
            if k in {'grail_fail','logprob_mismatch','seed_mismatch'}}}
    if any(ledger[-1].get(k)!=v for k,v in summary.items()):
        raise ValueError('natural selection summary does not conserve actual attempts')
    if summary['numerical_rejections_requires_review']:
        raise ValueError('canonical numerical proof rejections require diagnosis; no qualification override')
    return {**binding, **summary, 'full_http_or_grader_capacity_qualified':False}
