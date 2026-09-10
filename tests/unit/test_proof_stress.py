"""Stress evidence must exercise full CPU paths without creating acceptance."""
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary import constants as c
from reliquary.validator import proof_stress as stress
from reliquary.validator.verifier import ProofResult


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(token) for token in tokens)

    def get_vocab(self):
        return {chr(i): i for i in range(32, 127)}

    def convert_tokens_to_ids(self, token):
        return 127


def inputs(monkeypatch, *, full=False, code=False):
    if not full:
        monkeypatch.setattr(stress, 'STRESS_PROMPT_TOKENS', 64)
        monkeypatch.setattr(stress, 'STRESS_POLICY_TOKENS', 128)
        monkeypatch.setattr(stress, 'STRESS_ROLLOUTS', 2)
    monkeypatch.setattr(c, 'T_PROTO', 1.)
    monkeypatch.setattr(c, 'BFT_ENABLED', False)
    monkeypatch.setenv('RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_ENABLED', '0')
    monkeypatch.setenv('RELIQUARY_AUTH_FORENSICS_ENABLED', '1')
    monkeypatch.setenv('RELIQUARY_AUTH_FORENSICS_MAX_FINDINGS_PER_ROLLOUT', '3')
    p, n, m = stress.STRESS_PROMPT_TOKENS, stress.STRESS_POLICY_TOKENS, stress.STRESS_ROLLOUTS
    commit = {'tokens': [ord('x')] * (p + n),
              'commitments': [{'sketch': i % 19} for i in range(p + n)],
              'rollout': {'prompt_length': p, 'completion_length': n,
                          'token_logprobs': [0.] * n, 'total_reward': 0., 'forced': False}}
    proof = ProofResult(all_passed=False, passed=0, checked=c.CHALLENGE_K, sketch_diff_max=999,
                        has_sparse_outputs=True, p_stop=.01,
                        challenge_lp_indices=list(range(p, p + c.CHALLENGE_K)),
                        challenge_lp_values=[-2.] * c.CHALLENGE_K,
                        completion_chosen_probs=[0.] * n, completion_argmax_probs=[1.] * n,
                        completion_argmax_ids=[ord('y')] * n, completion_entropies=[1.] * 64,
                        seed_n_stochastic=n, seed_n_match=0, seed_n_positions=n,
                        seed_n_hard_mismatch=n)
    environment = SimpleNamespace(name='opencodeinstruct' if code else 'openmathinstruct',
                                  get_problem=lambda _index: {'prompt': 'A public test prompt'},
                                  compute_reward=lambda *_: pytest.fail('CPU stress must not invoke sandbox/network reward'))
    return {"commits": [deepcopy(commit) for _ in range(m)],
                "proofs": [replace(proof) for _ in range(m)], "tokenizer": CharacterTokenizer(),
                "environment": environment, "randomness": 'ab' * 32, "prompt_idx": 11,
                "checkpoint_revision": 'c' * 40}


def test_real_rejections_do_not_short_circuit_later_rollouts_or_helpers(monkeypatch):
    arguments = inputs(monkeypatch)
    before = deepcopy(arguments['commits'])
    probabilities = [list(proof.completion_chosen_probs) for proof in arguments['proofs']]
    report = stress.measure_postproof_stress(**arguments)
    assert report['complete'], report['errors']
    assert report['admissible'] is False
    assert report['scope'] == 'NONADMISSIBLE_POSTPROOF_CPU_STRESS'
    for row in report['actual_results']:
        assert row['kernel_verdict']['all_passed'] is False
        assert row['logprob'][0] is False
        assert row['token_auth'][0] is False
        assert row['entropy']['mean'] == 1.
    for helper in ('rollout_hash', 'seed_u', 'logprob', 'distribution', 'boxed',
                   'token_auth', 'all_token_auth', 'utility_nll', 'utility_entropy'):
        assert report['helper_counts'][helper] == stress.STRESS_ROLLOUTS
    assert arguments['commits'] == before
    assert [proof.completion_chosen_probs for proof in arguments['proofs']] == probabilities
    assert all(not proof.all_passed for proof in arguments['proofs'])
    assert report['seconds'] == report['real_result_seconds'] + report['synthetic_cpu_fixture_seconds']
    json.dumps(report, allow_nan=False)


def test_cpu_fixtures_reach_scans_and_dense_code_after_real_early_reject(monkeypatch):
    report = stress.measure_postproof_stress(**inputs(monkeypatch, code=True))
    assert report['complete'], report['errors']
    for actual, fixture in zip(report['actual_results'], report['cpu_only_fixtures'], strict=True):
        assert actual['pi_old_count'] is None  # True zero-probability early return retained.
        assert fixture['scope'] == 'NONADMISSIBLE_SYNTHETIC_CPU_ONLY'
        assert fixture['full_scan_pi_old_count'] == stress.STRESS_POLICY_TOKENS
        assert fixture['full_scan_auth'][0] is True
        assert fixture['max_findings_auth'][1]['findings'] == stress.STRESS_POLICY_TOKENS
        assert len(fixture['max_findings_auth'][1]['finding_details']) == 3
        assert fixture['full_boxed'][1]['n_tokens'] > stress.STRESS_POLICY_TOKENS // 2
        assert fixture['dense_code_semantic'][1]['n_spans'] > 0
        assert fixture['dense_code_semantic'][1]['findings'] > 0
    assert report['synthetic_reward_shape']['suspicious'] is False  # Only two test rollouts.
    stress.validate_postproof_measurement(report, rollout_count=2, completion_tokens=128)


@pytest.mark.parametrize('tamper', [
    lambda report: report['helper_counts'].pop('utility_nll'),
    lambda report: report['helper_counts'].__setitem__('seed_u', 1),
    lambda report: report.__setitem__('seconds', report['real_result_seconds']),
    lambda report: report.__setitem__('seconds', float('nan')),
    lambda report: report['settings'].__setitem__('auth_forensics_context_chars', 999),
    lambda report: report['cpu_only_fixtures'][0]['full_boxed'][1].__setitem__('n_tokens', 0),
    lambda report: report['cpu_only_fixtures'][0]['max_findings_auth'][1].__setitem__('finding_details', []),
    lambda report: report['cpu_only_fixtures'][0]['dense_code_semantic'][1].__setitem__('n_spans', 0),
    lambda report: report['actual_results'].pop(),
    lambda report: report.__setitem__('admissible', True),
])
def test_incomplete_tampered_or_differently_configured_receipt_refused(monkeypatch, tamper):
    report = stress.measure_postproof_stress(**inputs(monkeypatch, code=True))
    assert report['complete'], report['errors']
    tamper(report)
    with pytest.raises(ValueError):
        stress.validate_postproof_measurement(report, rollout_count=2, completion_tokens=128)


def test_unmeasured_counterfactual_mode_cannot_be_qualified(monkeypatch):
    arguments = inputs(monkeypatch, code=True)
    monkeypatch.setattr(stress.b, 'code_semantic_counterfactual_enabled', lambda: True)
    report = stress.measure_postproof_stress(**arguments)
    assert report['complete'] is False
    assert report['errors'][0]['helper'] == 'coverage'
    assert 'counterfactual' in report['errors'][0]['message']


def test_changed_utility_telemetry_cannot_reuse_measurement(monkeypatch):
    report = stress.measure_postproof_stress(**inputs(monkeypatch))
    assert report['complete'], report['errors']
    monkeypatch.setattr(stress, 'utility_telemetry_enabled',
                        lambda: not report['settings']['utility_telemetry_enabled'])
    with pytest.raises(ValueError, match='runtime settings mismatch'):
        stress.validate_postproof_measurement(report, rollout_count=2, completion_tokens=128)


def test_exception_retained_and_other_helpers_still_executed(monkeypatch):
    arguments = inputs(monkeypatch)
    def broken(*_args, **_kwargs):
        raise RuntimeError('deliberate helper failure')
    monkeypatch.setattr(stress.b, 'compute_rollout_hash', broken)
    report = stress.measure_postproof_stress(**arguments)
    assert report['complete'] is False
    assert report['errors'] == [{'helper': 'rollout_hash', 'error_type': 'RuntimeError'}] * 2
    assert report['helper_counts']['utility_entropy'] == 2
    assert report['helper_counts']['fixture_full_boxed'] == 2


@pytest.mark.parametrize('field', ['completion_chosen_probs', 'completion_argmax_probs', 'completion_argmax_ids'])
def test_incomplete_sparse_coverage_cannot_count_as_stress(monkeypatch, field):
    arguments = inputs(monkeypatch)
    arguments['proofs'][0] = replace(arguments['proofs'][0], **{field: []})
    with pytest.raises(ValueError, match='cover all policy tokens'):
        stress.measure_postproof_stress(**arguments)


def test_short_commit_or_sparse_stub_refused(monkeypatch):
    arguments = inputs(monkeypatch)
    arguments['commits'][0]['tokens'].pop()
    with pytest.raises(ValueError, match='complete unforced'):
        stress.measure_postproof_stress(**arguments)
    arguments = inputs(monkeypatch)
    arguments['proofs'][0] = replace(arguments['proofs'][0], has_sparse_outputs=False)
    with pytest.raises(ValueError, match='sparse ProofResults'):
        stress.measure_postproof_stress(**arguments)


def test_full_protocol_size_processes_all_16_times_32768_tokens(monkeypatch):
    report = stress.measure_postproof_stress(**inputs(monkeypatch, full=True))
    assert report['complete'], report['errors']
    assert report['rollout_count'] == 16
    assert report['prompt_tokens'] == 24576 and report['policy_tokens'] == 8192
    assert report['helper_counts']['sketch_metrics'] == 16
    assert all(row['sketch_metrics']['sketch_count'] == 32768 for row in report['actual_results'])
    assert all(row['token_metrics']['token_count'] == 8192 for row in report['actual_results'])
    assert report['synthetic_reward_shape']['suspicious'] is True


def test_native_forensic_io_uses_requested_filesystem_and_cleans_temporary_files(monkeypatch,tmp_path):
    import os
    report=stress.measure_postproof_stress(**inputs(monkeypatch,code=True),forensic_directory=tmp_path)
    assert report['complete'],report['errors']
    assert report['forensic_io']['filesystem_device']==os.stat(tmp_path).st_dev
    assert report['forensic_io']['rows']==12  # two rollouts, three findings, two native writers
    assert report['forensic_io']['seconds']>0
    assert list(tmp_path.iterdir())==[]


def test_swallowed_native_forensic_write_failure_cannot_qualify(monkeypatch,tmp_path):
    from reliquary.validator import auth_forensics
    arguments=inputs(monkeypatch)
    monkeypatch.setattr(auth_forensics,'record_all_token_auth_findings',lambda **kwargs:None)
    report=stress.measure_postproof_stress(**arguments,forensic_directory=tmp_path)
    assert report['complete'] is False
    assert any(error['helper']=='fixture_forensic_appends' for error in report['errors'])
    with pytest.raises(ValueError):
        stress.validate_postproof_measurement(report,rollout_count=2,completion_tokens=128)
