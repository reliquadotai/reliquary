import copy
import hashlib
import json
from pathlib import Path

import pytest

from reliquary.vision import retail_benchmark as b


@pytest.fixture
def manifest(tmp_path):
    data = b'bounded synthetic media fixture'
    (tmp_path / 'clip.bin').write_bytes(data)
    return {'schema': 'score-retail-benchmark/v1', 'mode': 'offline_video',
            'classes': ['entry', 'exit'], 'thresholds': {'score_threshold': .5, 'tiou_threshold': .5},
            'cases': [{'id': 'case1', 'source_group': 'camera-day1', 'split': 'test', 'domain': 'proxy',
                       'source_start_us': 0, 'duration_us': 10_000_000, 'media': {'path': 'clip.bin', 'sha256': hashlib.sha256(data).hexdigest()},
                       'annotation_complete': True,
                       'rights': {'license': 'CC-BY-4.0', 'source_url': 'https://example.org/source',
                                  'attribution': 'Synthetic fixture author', 'commercial_evaluation': True, 'commercial_training': False},
                       'events': [{'label': 'entry', 'start_us': 1_000_000, 'end_us': 3_000_000}]}]}


def predictions(manifest, status='ok', values=None):
    return {'schema': 'score-retail-predictions/v1', 'benchmark_sha256': 'a' * 64,
            'model': {'id': 'external-fixture', 'revision': 'test-only'},
            'predictions': [{'case_id': 'case1', 'status': status, 'raw_output': json.dumps({'events': [] if values is None else values}),
                             'events': [] if values is None else values}]}


def event(start, end, label='entry'):
    return {'label': label, 'start_us': start, 'end_us': end, 'score': .9}


def test_matching_finds_cardinality_that_greedy_misses():
    # First prediction prefers first truth but can take second; second can only take first.
    truths = [event(0, 10), event(5, 15)]
    predicted = [event(0, 12), event(0, 5)]
    assert b.match(predicted, truths, .4) == [(0, 1), (1, 0)]


def test_duplicate_wrong_class_and_low_score(manifest):
    p = predictions(manifest, values=[event(1_000_000, 3_000_000), event(1_000_000, 3_000_000),
                                     event(1_000_000, 3_000_000, 'exit'),
                                     {**event(1_000_000, 3_000_000), 'score': .1}])
    result = b.score(manifest, p)
    assert result['micro'] == {'tp': 1, 'fp': 2, 'fn': 0, 'precision': 1/3, 'recall': 1., 'f1': .5}
    assert result['measured_cost_usd'] is None
    assert not result['customer_acceptance']


@pytest.mark.parametrize('status', ['missing', 'invalid', 'unobservable', 'ok'])
def test_failure_is_not_verified_silence(manifest, status):
    p = predictions(manifest, status if status != 'missing' else 'ok')
    if status == 'missing':
        p['predictions'] = []
    result = b.score(manifest, p)
    assert result['micro']['fn'] == 1
    assert result['case_status_counts'][status] == 1
    assert result['processed_duration_fraction'] == (1 if status == 'ok' else 0)
    assert (result['false_alerts_per_processed_camera_hour'] is None) == (status != 'ok')


def test_empty_reviewed_class_is_not_perfect(manifest):
    manifest['cases'][0]['events'] = []
    result = b.score(manifest, predictions(manifest))
    assert result['micro']['f1'] is None
    assert result['micro']['recall'] is None
    assert result['reviewed_normal_camera_hours'] == 10 / 3600
    assert result['zero_fp_poisson_upper95_per_hour'] > 1000


def test_incomplete_cases_prepare_but_cannot_score(manifest, tmp_path):
    manifest['cases'][0]['annotation_complete'] = False
    b.validate_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match='fully reviewed'):
        b.score(manifest, predictions(manifest))


def test_source_and_hash_leakage(manifest, tmp_path):
    for same_group in (True, False):
        value = copy.deepcopy(manifest)
        duplicate = copy.deepcopy(value['cases'][0])
        duplicate.update(id='case2', split='train')
        if not same_group:
            duplicate['source_group'] = 'other-camera'
        value['cases'].append(duplicate)
        with pytest.raises(ValueError, match='crosses splits'):
            b.validate_manifest(value, tmp_path)


def test_media_hash_and_path_are_checked(manifest, tmp_path):
    b.validate_manifest(manifest, tmp_path)
    (tmp_path / 'clip.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest mismatch'):
        b.validate_manifest(manifest, tmp_path)
    for path in ('../clip.bin', '/tmp/clip.bin'):
        with pytest.raises(ValueError):
            b.media_path(tmp_path, path)
    (tmp_path / 'link.bin').symlink_to(tmp_path / 'clip.bin')
    with pytest.raises(ValueError, match='symlink'):
        b.media_path(tmp_path, 'link.bin')


@pytest.mark.parametrize('value', [True, -1, 10_000_001, float('nan'), 1.5])
def test_invalid_timestamps_fail(manifest, tmp_path, value):
    manifest['cases'][0]['events'][0]['start_us'] = value
    with pytest.raises(ValueError):
        b.validate_manifest(manifest, tmp_path)


def test_rights_must_be_explicit(manifest, tmp_path):
    manifest['cases'][0]['rights']['commercial_evaluation'] = False
    with pytest.raises(ValueError, match='permission'):
        b.validate_manifest(manifest, tmp_path)


def test_prediction_binding_and_status(manifest):
    p = predictions(manifest)
    b.validate_predictions(p, manifest, 'a' * 64)
    with pytest.raises(ValueError, match='binding'):
        b.validate_predictions(p, manifest, 'b' * 64)
    p['predictions'][0].update(status='unobservable', events=[event(1, 2)])
    with pytest.raises(ValueError, match='non-ok'):
        b.validate_predictions(p, manifest, 'a' * 64)


def test_output_is_private_and_preserves_existing_bytes(tmp_path):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    out = private / 'report.json'
    b.write_private(out, {'ok': True})
    assert out.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        b.write_private(out, {})
    assert json.loads(out.read_text()) == {'ok': True}
    (private / '.git').mkdir()
    with pytest.raises(ValueError, match='outside repositories'):
        b.write_private(private / 'other.json', {})


def test_cli_roundtrip_and_nonfinite_json(manifest, tmp_path, capsys):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    m = private / 'manifest.json'
    m.write_text(json.dumps(manifest))
    p = predictions(manifest)
    p['benchmark_sha256'] = b.digest(m)
    pred = private / 'predictions.json'
    pred.write_text(json.dumps(p))
    assert b.main(['validate', str(m), '--media-root', str(tmp_path)]) == 0
    assert b.main(['score', str(m), str(pred), '--media-root', str(tmp_path), '--output', str(private / 'report.json')]) == 0
    assert json.loads((private / 'report.json').read_text())['micro']['fn'] == 1
    pred.write_text('{"bad":NaN}')
    assert b.main(['score', str(m), str(pred), '--media-root', str(tmp_path), '--output', str(private / 'other.json')]) == 1


def test_duplicate_media_cannot_inflate_exposure(manifest, tmp_path):
    duplicate = copy.deepcopy(manifest['cases'][0])
    duplicate['id'] = 'case2'
    manifest['cases'].append(duplicate)
    with pytest.raises(ValueError, match='double-count'):
        b.validate_manifest(manifest, tmp_path)


def test_ok_prediction_requires_matching_raw_json(manifest):
    p = predictions(manifest)
    p['predictions'][0]['raw_output'] = '```json {"events":[]} ```'
    with pytest.raises(ValueError):
        b.validate_predictions(p, manifest, 'a' * 64)
    p['predictions'][0]['raw_output'] = json.dumps({'events': [event(1, 2)]})
    with pytest.raises(ValueError, match='differ'):
        b.validate_predictions(p, manifest, 'a' * 64)


def test_reencoded_overlapping_source_intervals_are_not_double_exposure(manifest, tmp_path):
    other = b'reencoded synthetic bytes'
    (tmp_path / 'other.bin').write_bytes(other)
    duplicate = copy.deepcopy(manifest['cases'][0])
    duplicate.update(id='case2', media={'path': 'other.bin', 'sha256': hashlib.sha256(other).hexdigest()})
    manifest['cases'].append(duplicate)
    with pytest.raises(ValueError, match='overlapping'):
        b.validate_manifest(manifest, tmp_path)
    duplicate['source_start_us'] = 10_000_000
    b.validate_manifest(manifest, tmp_path)


def test_normal_exposure_subtracts_union_in_positive_clips(manifest):
    manifest['cases'][0]['events'] = [
        {'label': 'entry', 'start_us': 1_000_000, 'end_us': 5_000_000},
        {'label': 'exit', 'start_us': 3_000_000, 'end_us': 7_000_000},
        {'label': 'entry', 'start_us': 8_000_000, 'end_us': 9_000_000},
    ]
    result = b.score(manifest, predictions(manifest))
    assert result['reviewed_normal_camera_hours'] == 3 / 3600
    assert result['processed_camera_hours'] == 10 / 3600


@pytest.mark.parametrize('bind_original', [True, False])
def test_cli_uses_one_manifest_snapshot_during_mutation(manifest, tmp_path, monkeypatch, bind_original):
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    m = private / 'manifest.json'
    original_bytes = json.dumps(manifest).encode()
    changed = copy.deepcopy(manifest)
    changed['cases'][0]['events'] = []
    changed_bytes = json.dumps(changed).encode()
    m.write_bytes(original_bytes)
    p = predictions(manifest)
    p['benchmark_sha256'] = hashlib.sha256(original_bytes if bind_original else changed_bytes).hexdigest()
    pred = private / 'predictions.json'
    pred.write_text(json.dumps(p))
    original_validate = b.validate_manifest

    def mutate(value, root):
        m.write_bytes(changed_bytes)
        return original_validate(value, root)

    monkeypatch.setattr(b, 'validate_manifest', mutate)
    out = private / 'report.json'
    result = b.main(['score', str(m), str(pred), '--media-root', str(tmp_path), '--output', str(out)])
    assert result == (0 if bind_original else 1)
    if bind_original:
        report = json.loads(out.read_text())
        assert report['benchmark_sha256'] == hashlib.sha256(original_bytes).hexdigest()
        assert report['micro']['fn'] == 1
    else:
        assert not out.exists()
