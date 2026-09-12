"""Combined qualification must keep real correctness separate from resource stress."""
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reliquary import constants as c
from reliquary.validator import proof_capacity_combined as combined
from reliquary.validator import proof_stress
from reliquary.validator.proof_capacity import (
    ProofCapacityQualification,
    ProofCapacityQualificationError,
)

ENV = ('openmathinstruct', 'opencodeinstruct', 'reliquary_logic_v2')
CONTROLLER = {'cpu_model': 'test CPU', 'cpu.max': '300000 100000', 'memory.max': '25769803776'}
IDENTITY = {'profile_id': 'test-profile', 'model_revision': 'a'*40,
            'software_revision': 'b'*40, 'checkpoint_revision': 'c'*40,
            'runtime_fingerprint_hash': 'e'*64, 'hardware_class': 'H100'}
REMOTE = {'worker_id': 'test-worker', 'transport_sha256': 'f'*64,
          'pipeline_depth': 1, 'protocol': 'reliquary.remote-proof/v1',
          'measurement_scope': 'validator-end-to-end-mtls'}
BINDING = {'checkpoint_n': 1821, 'repo_id': 'test/model', 'training_run_id': 'new-v1',
           'revision': 'c'*40, 'profile_id': 'test-profile', 'generation_contract_sha256': 'a'*64}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'FILL_CLOSED_ENABLED', False)
    monkeypatch.setattr(c, 'MAX_PROOF_WALL_SECONDS', 1000.)
    monkeypatch.setattr(c, 'MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW', 32)
    monkeypatch.setattr(c, 'FORENSIC_SAMPLE_PER_WINDOW', 2)
    monkeypatch.setattr(c, 'M_ROLLOUTS', 16)
    monkeypatch.setattr(c, 'CHALLENGE_K', 32)
    monkeypatch.setattr(combined, 'controller_identity', lambda: CONTROLLER)
    monkeypatch.setattr(proof_stress, 'runtime_settings', lambda: {'counterfactual': False, 'utility_telemetry_enabled':False})
    # Helper semantics and real CPU branches are covered in test_proof_stress;
    # this fixture isolates transport/aggregation/activation evidence validation.
    def validate_cpu(report, *, rollout_count, completion_tokens):
        if report != {'seconds': .5, 'settings': {'counterfactual': False, 'utility_telemetry_enabled':False}, 'complete': True}:
            raise ValueError('CPU coverage missing')
        assert rollout_count == 16 and completion_tokens == 8192
    monkeypatch.setattr(proof_stress, 'validate_postproof_measurement', validate_cpu)
    natural, stress = [], []
    for env in ENV:
        for i in range(20):
            base = {**IDENTITY, 'environment': env, 'device_uuid': 'gpu-0', 'device_id': 'cuda:0',
                'window': 1, 'checkpoint_n': 1821, 'repo_id': 'test/model', 'training_run_id': 'new-v1',
                'session_id': 'session1', 'rollout_count': 16, 'remote_proof': REMOTE,
                'complete_remote_group': True, 'configured_slots': {'cuda:0':'gpu-0'}}
            def receipts(kind, length, env=env, i=i):
                return [{'job_id': f'{kind}-{env}-{i}-{j}', 'device_id': 'cuda:0', 'window': 1,
                    'environment': env, 'content_sha256': hashlib.sha256(f'{kind}-{env}-{i}-{j}'.encode()).hexdigest(),
                    'checkpoint': BINDING, 'policy_tokens': length} for j in range(16)]
            natural.append({**base, 'job_id':f'group-{env}-{i}', 'seconds': 1., 'proof_passed': True,
                            'completion_token_lengths': [64]*16, 'wire_receipts': receipts('natural', 64)})
            stress.append({**base, 'scope': combined.SCOPE, 'synthetic': True, 'valid_submission': False,
                'seconds': 2.5, 'remote_seconds': 2., 'completion_token_lengths': [8192]*16,
                'total_token_lengths': [32768]*16, 'seed_position_counts': [8192]*16,
                'challenge_counts': [32]*16, 'entropy_sample_counts':[0]*16, 'group_id': f'{env}-{i}', 'controller': CONTROLLER,
                'postproof_cpu': {'seconds': .5, 'settings': {'counterfactual': False, 'utility_telemetry_enabled':False}, 'complete': True},
                'wire_receipts': receipts('stress', 8192)})
    def load(*, stress_rows=None, natural_rows=None):
        from reliquary.validator.remote_proof_protocol import canonical_bytes
        natural_path, stress_path = tmp_path/'natural.jsonl', tmp_path/'stress.jsonl'
        corpus_path=tmp_path/'corpus.jsonl'
        rows=natural if natural_rows is None else natural_rows
        corpus_lines=[canonical_bytes({'environment':row['environment'],'index':i})+b'\n' for i,row in enumerate(rows)]
        corpus_path.write_bytes(b''.join(corpus_lines))
        corpus_sha=hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        ledger=[{'event':'run_started','scope':'combined-natural-all-attempts/v1','corpus_sha256':corpus_sha,
                 'checkpoint':BINDING,'worker_id':'test-worker','session_id':'session1'}]
        inputs={}
        for i,(row,raw) in enumerate(zip(rows,corpus_lines,strict=True)):
            input_sha=hashlib.sha256(raw.rstrip(b'\n')).hexdigest()
            inputs[row['job_id']]=input_sha
            ledger.append({'event':'admission','row_index':i,'raw_input_sha256':hashlib.sha256(raw).hexdigest(),
                'input_sha256':input_sha,'outcome':'accepted','job_id':row['job_id'],'admission_seconds':.1})
            ledger.append({'event':'proof','job_id':row['job_id'],'input_sha256':input_sha,
                'environment':row['environment'],'device_id':row['device_id'],'outcome':'passed'})
        ledger.append({'event':'run_complete','admitted_groups':len(rows),'passing_groups':len(rows),
            'admission_rejections_by_reason':{},'proof_rejections_by_reason':{},'numerical_rejections_requires_review':{}})
        ledger_raw=b''.join(canonical_bytes(row)+b'\n' for row in ledger)
        proofs_raw=b''.join(canonical_bytes(row)+b'\n' for row in rows)
        natural_path.with_name(natural_path.name+'.attempts.jsonl').write_bytes(ledger_raw)
        natural_path.with_name(natural_path.name+'.proof-attempts.jsonl').write_bytes(proofs_raw)
        bindings={'attempt_ledger_sha256':hashlib.sha256(ledger_raw).hexdigest(),
                  'proof_attempts_sha256':hashlib.sha256(proofs_raw).hexdigest(),'corpus_sha256':corpus_sha}
        natural_path.write_bytes(b''.join(canonical_bytes({**row,**bindings,'input_sha256':inputs[row['job_id']]})+b'\n' for row in rows))
        stress_path.write_text(''.join(json.dumps(r)+'\n' for r in (stress if stress_rows is None else stress_rows)))
        return combined.load_stress_samples(stress_path, natural_path=natural_path,corpus_path=corpus_path,
            expected_identity=IDENTITY, device_uuids=('gpu-0',), completion_caps={e:8192 for e in ENV},
            maximum_context_tokens=32768, remote_proof=REMOTE,
            natural_samples={e:{'gpu-0':[1.]*20} for e in ENV},
            natural_samples_sha256=hashlib.sha256(natural_path.read_bytes()).hexdigest())
    return natural, stress, load


def manifest(e):
    return {**IDENTITY, 'schema_version':4, 'samples_sha256':e['natural_samples_sha256'],
        'benchmark_device_count':1, 'benchmark_device_uuids':['gpu-0'],
        'proof_wall_seconds':1000., 'headroom_fraction':.2,
        'proofs_per_environment':{env:34 for env in ENV},
        'p95_seconds_per_proof':{env:1. for env in ENV},
        'p95_seconds_per_proof_by_environment_and_device':{env:{'gpu-0':1.} for env in ENV},
        'sample_count_by_environment':{env:20 for env in ENV},
        'sample_count_by_environment_and_device':{env:{'gpu-0':20} for env in ENV},
        'minimum_samples_per_device_per_environment':20,
        'minimum_completion_tokens_by_environment':{env:64 for env in ENV},
        'measured_at':'test', 'qualified':True, 'combined_evidence':e}


def activate(value, **overrides):
    args = {key:IDENTITY[key] for key in ('profile_id','model_revision','software_revision',
                                        'checkpoint_revision','runtime_fingerprint_hash')}
    args.update(configured_devices=('cuda:0',), configured_hardware=('H100',),
        configured_device_uuids=('gpu-0',), proof_wall_seconds=1000., minimum_proofs_per_environment=34,
        minimum_completion_tokens_per_environment={env:7373 for env in ENV},
        configured_slots={'cuda:0':'gpu-0'}, maximum_context_tokens=32768, full_completion_tokens_per_environment={env:8192 for env in ENV})
    args.update(overrides)
    return ProofCapacityQualification.from_mapping(value).validate(**args)


def test_combined_preserves_short_real_groups_and_uses_actual_worse_cost(evidence):
    _, _, load = evidence
    value = manifest(load())
    result = activate(value)
    assert result['required_device_seconds'] == 34*3*3.5
    assert result['minimum_device_count'] == 1
    assert value['p95_seconds_per_proof'] == {env:1. for env in ENV}
    assert value['minimum_completion_tokens_by_environment'] == {env:64 for env in ENV}
    assert result['combined_seconds_per_proof_bound'][ENV[0]]['gpu-0'] == 3.5


@pytest.mark.parametrize('tamper', [
    lambda rows: rows[0].__setitem__('proof_passed', True),
    lambda rows: rows[0].__setitem__('valid_submission', True),
    lambda rows: rows[0].__setitem__('synthetic', False),
    lambda rows: rows[0].__setitem__('complete_remote_group', False),
    lambda rows: rows[0]['total_token_lengths'].__setitem__(0, 32767),
    lambda rows: rows[0]['completion_token_lengths'].__setitem__(0, 8191),
    lambda rows: rows[0]['seed_position_counts'].__setitem__(0, 8191),
    lambda rows: rows[0]['challenge_counts'].__setitem__(0, 31),
    lambda rows: rows[0].__setitem__('device_uuid', 'another-gpu'),
    lambda rows: rows[0].__setitem__('session_id', 'old-session'),
    lambda rows: rows[0].__setitem__('training_run_id', 'v5'),
    lambda rows: rows[0].__setitem__('software_revision', 'd'*40),
    lambda rows: rows[0].__setitem__('runtime_fingerprint_hash', 'd'*64),
    lambda rows: rows[0]['wire_receipts'][0]['checkpoint'].__setitem__('revision', 'd'*40),
    lambda rows: rows[0]['wire_receipts'][0].__setitem__('policy_tokens', 8191),
    lambda rows: rows[0]['wire_receipts'][0].__setitem__('content_sha256', 'bad'),
    lambda rows: rows[0]['wire_receipts'].pop(),
    lambda rows: rows[1].__setitem__('wire_receipts', rows[0]['wire_receipts']),
    lambda rows: rows[1].__setitem__('group_id', rows[0]['group_id']),
    lambda rows: rows[0].__setitem__('seconds', 2.),
    lambda rows: rows[0].__setitem__('remote_seconds', float('nan')),
    lambda rows: rows[0]['postproof_cpu'].__setitem__('complete', False),
    lambda rows: rows.pop(),
])
def test_partial_misbound_or_misrepresented_stress_is_refused(evidence, tamper):
    _, rows, load = evidence
    rows = deepcopy(rows)
    tamper(rows)
    with pytest.raises(ValueError):
        load(stress_rows=rows)


@pytest.mark.parametrize('tamper', [
    lambda rows: rows[0].__setitem__('proof_passed', False),
    lambda rows: rows[0].__setitem__('complete_remote_group', False),
    lambda rows: rows[0].pop('session_id'),
    lambda rows: rows[0]['wire_receipts'].pop(),
    lambda rows: rows[1].__setitem__('wire_receipts', rows[0]['wire_receipts']),
])
def test_stress_never_replaces_authentic_passing_groups(evidence, tamper):
    rows, _, load = evidence
    rows = deepcopy(rows)
    tamper(rows)
    with pytest.raises(ValueError):
        load(natural_rows=rows)


@pytest.mark.parametrize('tamper', [
    lambda v: v.__setitem__('combined_evidence', None),
    lambda v: v['combined_evidence'].__setitem__('stress_samples_sha256', 'bad'),
    lambda v: v['combined_evidence']['controller'].__setitem__('cpu.max', '100000 100000'),
    lambda v: v['combined_evidence']['cpu_settings'].__setitem__('counterfactual', True),
    lambda v: v['combined_evidence']['by_environment_and_device'][ENV[0]]['gpu-0'].__setitem__('stress_sample_count',19),
    lambda v: v['combined_evidence']['by_environment_and_device'][ENV[0]]['gpu-0'].__setitem__('context_tokens',8192),
    lambda v: v['combined_evidence']['by_environment_and_device'][ENV[0]]['gpu-0'].__setitem__('seconds_per_proof_bound',1.),
    lambda v: v.__setitem__('headroom_fraction',.19),
    lambda v: v['proofs_per_environment'].__setitem__(ENV[0],33),
    lambda v: v['sample_count_by_environment_and_device'][ENV[0]].__setitem__('gpu-0',19),
])
def test_runtime_rejects_incomplete_or_understated_combined_manifest(evidence, tamper):
    value = deepcopy(manifest(evidence[2]()))
    tamper(value)
    with pytest.raises(ProofCapacityQualificationError):
        activate(value)


def test_runtime_cap_context_image_and_capacity_remain_strict(evidence):
    value = manifest(evidence[2]())
    for overrides in ({'maximum_context_tokens':65536}, {'software_revision':'d'*40},
                      {'runtime_fingerprint_hash':'d'*64}, {'maximum_context_tokens':None}):
        with pytest.raises(ProofCapacityQualificationError):
            activate(value, **overrides)
    for env in ENV:
        row = value['combined_evidence']['by_environment_and_device'][env]['gpu-0']
        row.update(max_stress_seconds=10., seconds_per_proof_bound=11.)
    with pytest.raises(ProofCapacityQualificationError, match='requires 2'):
        activate(value)


def test_new_measurement_uses_exact_envelope_without_claiming_validity(monkeypatch):
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[2]/'scripts/measure_remote_proof_stress.py'
    spec = importlib.util.spec_from_file_location('stress_script', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(c, 'M_ROLLOUTS',16)
    monkeypatch.setattr(c, 'MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV',{ENV[0]:8192})
    model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768,vocab_size=100,eos_token_id=99),
                          generation_config=SimpleNamespace(eos_token_id=99))
    commits=script.make_group(tokenizer=SimpleNamespace(eos_token_id=99), model=model,
                             environment=ENV[0],group_id='test-full-envelope')
    assert len(commits)==16
    assert all(len(v['tokens'])==len(v['commitments'])==32768 for v in commits)
    assert all(v['rollout']['prompt_length']==24576 and v['rollout']['completion_length']==8192 for v in commits)
    assert len({hashlib.sha256(json.dumps(v).encode()).hexdigest() for v in commits})==16
    assert all(v['tokens'][-1]==99 for v in commits)


def stress_script():
    path = Path(__file__).resolve().parents[2]/'scripts/measure_remote_proof_stress.py'
    spec = importlib.util.spec_from_file_location('stress_measure_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_group_serializer_retains_environment_identity_and_false_verdict(monkeypatch):
    """Unit transport stub; this does not claim GPU timing or physical qualification."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from reliquary.validator.remote_proof_protocol import canonical_bytes
    from reliquary.validator.verifier import ProofResult
    script = stress_script()
    monkeypatch.setattr(c, 'M_ROLLOUTS', 16)
    monkeypatch.setattr(c, 'CHALLENGE_K', 32)
    monkeypatch.setattr(c, 'MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV', {ENV[0]:8192})
    monkeypatch.setattr(script, 'make_group', lambda **kw: [
        {'tokens':[1]*32768, 'commitments':[], 'rollout':{}} for _ in range(16)])
    from reliquary.environment import forced_sampling
    monkeypatch.setattr(forced_sampling, 'u_at', lambda *args: .5)
    monkeypatch.setattr(proof_stress, 'measure_postproof_stress', lambda **kw: {'seconds':1e-9})
    monkeypatch.setattr(proof_stress, 'validate_postproof_measurement', lambda *a, **kw: None)
    binding = SimpleNamespace(**BINDING)
    slot = SimpleNamespace(device_id='cuda:0',device_uuid='GPU-0',runtime={'profile_hash':'e'*64},hardware_class='H100')
    health = SimpleNamespace(slots=[slot], profile_id='test-profile',software_revision='b'*40,
        session_id='session1', worker_id='test-worker',transport_sha256='f'*64,utility_telemetry_enabled=False)
    receipts = []
    @contextmanager
    def measure_group():
        yield receipts
    def prove(*args, **kwargs):
        receipts.append({'job_id':str(len(receipts))})
        return ProofResult(all_passed=False, passed=0, checked=32, sketch_diff_max=999,
            seed_n_positions=8192, seed_n_hard_mismatch=8192, completion_chosen_probs=[0.]*8192)
    pool = SimpleNamespace(_adopted=binding, health=health, pipeline_depth=1,
                           proxies=lambda:{'cuda:0':object()},
                           measure_group=measure_group, prove=prove)
    invocation = SimpleNamespace(device_id='cuda:0',environment=ENV[0],candidate=SimpleNamespace(job_id='unit-group'))
    row=script.measure_group(invocation,pool=pool,tokenizer=object(),
                             environments={ENV[0]:object()},controller=CONTROLLER)
    serialized=json.loads(canonical_bytes(row))
    assert serialized['environment']==ENV[0]
    assert serialized['configured_slots']=={'cuda:0':'gpu-0'}
    assert serialized['synthetic'] is True and serialized['valid_submission'] is False
    assert 'proof_passed' not in serialized
    assert all(v['all_passed'] is False for v in serialized['kernel_verdicts'])
    assert len(serialized['wire_receipts'])==16
    assert serialized['seconds'] >= serialized['remote_seconds']+serialized['postproof_cpu']['seconds']


def test_stress_scheduler_exercises_all_slots_and_retains_failed_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    script=stress_script()
    import os
    monkeypatch.setattr(combined,'controller_identity',lambda:{**CONTROLLER,'state_filesystem_device':os.stat(tmp_path).st_dev})
    pool=SimpleNamespace(devices=('cuda:0#0','cuda:0#1'), _adopted=SimpleNamespace(revision='c'*40),
                         assert_ready=lambda:None)
    monkeypatch.setattr(script,'measure_group',lambda invocation, **kw:
                        {'device_id':invocation.device_id,'environment':invocation.environment})
    output=tmp_path/'slots.jsonl'
    report=script.measure(pool=pool,tokenizer=None,environments={ENV[0]:None},output=output,groups=20,timeout=5)
    assert report['groups']==40 and report['qualified'] is False
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert all(sum(row['device_id']==device for row in rows)==20 for device in pool.devices)
    def fail(*args,**kwargs):
        raise ValueError('injected execution failure')
    monkeypatch.setattr(script,'measure_group',fail)
    failed=tmp_path/'failed.jsonl'
    with pytest.raises(RuntimeError,match='aborted'):
        script.measure(pool=pool,tokenizer=None,environments={ENV[0]:None},output=failed,groups=20,timeout=5)
    failures=[json.loads(line) for line in failed.read_text().splitlines()]
    assert failures
    assert all(row['complete_remote_group'] is False and row['error_type']=='ValueError' for row in failures)


def test_qualifier_natural_mode_is_explicit_and_retains_true_lengths(evidence, monkeypatch, tmp_path):
    natural, _, _ = evidence
    path = Path(__file__).resolve().parents[2]/'scripts/qualify_proof_capacity.py'
    spec = importlib.util.spec_from_file_location('capacity_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ENVIRONMENTS',ENV)
    monkeypatch.setattr(module,'ROLLOUTS_PER_PROOF',16)
    monkeypatch.setattr(module,'MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV',{e:8192 for e in ENV})
    monkeypatch.setattr(module,'PROTOCOL_PROFILE_ID',IDENTITY['profile_id'])
    monkeypatch.setattr(module,'PROTOCOL_MODEL_REVISION',IDENTITY['model_revision'])
    source=tmp_path/'raw.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in natural))
    kwargs={key:IDENTITY[key] for key in ('software_revision','checkpoint_revision','runtime_fingerprint_hash','hardware_class')}
    kwargs.update(benchmark_device_count=1,remote_proof=REMOTE)
    with pytest.raises(ValueError,match='not representative'):
        module._load_samples(source,**kwargs)
    samples,devices,lengths=module._load_samples(source,**kwargs,natural_completions=True)
    assert devices==('gpu-0',) and lengths=={env:64 for env in ENV}
    assert all(len(samples[env]['gpu-0'])==20 for env in ENV)


def test_capacity_budget_matches_actual_fill_state_not_legacy_proof_knob(monkeypatch):
    from reliquary.validator.proof_capacity import capacity_budget
    monkeypatch.setattr(c,'FILL_CLOSED_ENABLED',True)
    monkeypatch.setattr(c,'FILL_CLOSED_ADMISSION_BUDGET_PER_ENV',512)
    monkeypatch.setattr(c,'FILL_CLOSED_TARGET_GROUPS_PER_ENV',256)
    monkeypatch.setattr(c,'FILL_CLOSED_EMISSIONS_PER_WINDOW',16)
    monkeypatch.setattr(c,'FILL_CLOSED_PICKS_PER_WINDOW',16)
    monkeypatch.setattr(c,'FILL_CLOSED_MAX_SECONDS',1800.)
    monkeypatch.setattr(c,'MAX_PROOF_WALL_SECONDS',999.)
    assert capacity_budget()=={'mode':'fill_closed','proofs_per_environment':512,'wall_seconds':1800.,
                              'target_groups_per_environment':256,'picks_per_window':16}
    monkeypatch.setattr(c,'MAX_PROOF_WALL_SECONDS',999999.)
    assert capacity_budget()['wall_seconds']==1800.
    monkeypatch.setattr(c,'FILL_CLOSED_MAX_SECONDS',3600.)
    assert capacity_budget()['wall_seconds']==3600.
    monkeypatch.setattr(c,'FILL_CLOSED_PICKS_PER_WINDOW',10)
    monkeypatch.setattr(c,'FILL_CLOSED_TARGET_GROUPS_PER_ENV',160)
    assert capacity_budget()['picks_per_window']==10
    assert capacity_budget()['target_groups_per_environment']==160
    assert c.FILL_CLOSED_EMISSIONS_PER_WINDOW==16
    monkeypatch.setattr(c,'FILL_CLOSED_ENABLED',False)
    monkeypatch.setattr(c,'MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW',32)
    monkeypatch.setattr(c,'FORENSIC_SAMPLE_PER_WINDOW',2)
    assert capacity_budget()=={'mode':'seal_time_auction','proofs_per_environment':34,'wall_seconds':999999.}


def test_numerical_reject_cannot_be_removed_by_natural_selection(evidence, tmp_path):
    evidence[2]()
    path=tmp_path/'natural.jsonl'
    ledger_path=path.with_name(path.name+'.attempts.jsonl')
    proofs_path=path.with_name(path.name+'.proof-attempts.jsonl')
    ledger=[json.loads(line) for line in ledger_path.read_text().splitlines()]
    proofs=[json.loads(line) for line in proofs_path.read_text().splitlines()]
    selected=[json.loads(line) for line in path.read_text().splitlines()]
    job=proofs[0]['job_id']
    proofs[0]['proof_passed']=False
    for row in ledger:
        if row['event']=='proof' and row['job_id']==job:
            row.update(outcome='rejected',reason='grail_fail')
    ledger[-1].update(passing_groups=59,proof_rejections_by_reason={'grail_fail':1},
                      numerical_rejections_requires_review={'grail_fail':1})
    selected=[row for row in selected if row['job_id']!=job]
    for target,rows in ((ledger_path,ledger),(proofs_path,proofs)):
        target.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for row in selected:
        row.update(attempt_ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                   proof_attempts_sha256=hashlib.sha256(proofs_path.read_bytes()).hexdigest())
    path.write_text(''.join(json.dumps(row)+'\n' for row in selected))
    with pytest.raises(ValueError,match='canonical numerical proof rejections'):
        combined.validate_natural_selection(path,corpus_path=tmp_path/'corpus.jsonl')


@pytest.mark.parametrize('target', ['corpus.jsonl','natural.jsonl.attempts.jsonl','natural.jsonl.proof-attempts.jsonl'])
def test_original_corpus_and_both_attempt_ledgers_are_digest_bound(evidence,tmp_path,target):
    evidence[2]()
    path=tmp_path/target
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='digest|differs'):
        combined.validate_natural_selection(tmp_path/'natural.jsonl',corpus_path=tmp_path/'corpus.jsonl')
