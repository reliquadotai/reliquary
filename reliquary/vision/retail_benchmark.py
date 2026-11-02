"""Offline, hash-bound interval evaluation; never executes a model or a worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

from reliquary.shared.strict_json import strict_json_loads

SHA = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError("unexpected object fields")
    return value


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("nonempty bounded text required")
    return value


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid identifier")
    return value


def integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("integer outside bound")
    return value


def fraction(value, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1 or (positive and not value):
        raise ValueError("invalid fraction")
    return value


def load(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("JSON exceeds byte limit")
    return strict_json_loads(raw)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def media_path(root, relative):
    text(relative)
    rel = PurePosixPath(relative)
    if rel.is_absolute() or str(rel) != relative or '..' in rel.parts or '\\' in relative:
        raise ValueError("media must be a canonical relative path")
    path = Path(root).resolve()
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("media symlink forbidden")
    if not path.is_file():
        raise ValueError("media must be a regular file")
    return path


def events(values, classes, duration, predicted=False):
    if not isinstance(values, list) or len(values) > 512:
        raise ValueError("invalid event list")
    for event in values:
        keys(event, 'label start_us end_us score' if predicted else 'label start_us end_us')
        if not isinstance(event['label'], str) or event['label'] not in classes:
            raise ValueError("unknown class")
        integer(event['start_us'], 0, duration - 1)
        integer(event['end_us'], event['start_us'] + 1, duration)
        if predicted:
            fraction(event['score'])
    return values


def validate_manifest(value, root):
    keys(value, 'schema mode classes thresholds cases')
    if value['schema'] != 'score-retail-benchmark/v1' or value['mode'] != 'offline_video':
        raise ValueError("unsupported benchmark")
    classes = value['classes']
    if not isinstance(classes, list) or not 1 <= len(classes) <= 32:
        raise ValueError("invalid classes")
    for label in classes:
        identifier(label)
    if len(set(classes)) != len(classes):
        raise ValueError("duplicate classes")
    thresholds = keys(value['thresholds'], 'score_threshold tiou_threshold')
    fraction(thresholds['score_threshold'])
    fraction(thresholds['tiou_threshold'], positive=True)
    if not isinstance(value['cases'], list) or not 1 <= len(value['cases']) <= 256:
        raise ValueError("invalid cases")
    seen, groups, hashes, intervals = set(), {}, {}, {}
    for case in value['cases']:
        keys(case, 'id source_group split domain source_start_us duration_us media annotation_complete rights events')
        identifier(case['id']); identifier(case['source_group'])
        if case['id'] in seen:
            raise ValueError("duplicate case")
        seen.add(case['id'])
        if case['split'] not in ('train', 'validation', 'test') or case['domain'] not in ('proxy', 'score_station'):
            raise ValueError("unsupported split or domain")
        integer(case['duration_us'], 1, 3_600_000_000)
        integer(case['source_start_us'], 0, 10**18)
        if type(case['annotation_complete']) is not bool:
            raise ValueError("annotation completeness must be explicit")
        media = keys(case['media'], 'path sha256')
        if not isinstance(media['sha256'], str) or not SHA.fullmatch(media['sha256']):
            raise ValueError("invalid media digest")
        if digest(media_path(root, media['path'])) != media['sha256']:
            raise ValueError("media digest mismatch")
        duplicate_media = media['sha256'] in hashes
        for mapping, key in ((groups, case['source_group']), (hashes, media['sha256'])):
            if key in mapping and mapping[key] != case['split']:
                raise ValueError("source or media crosses splits")
            mapping[key] = case['split']
        if duplicate_media:
            raise ValueError("duplicate media would double-count exposure")
        start, end = case['source_start_us'], case['source_start_us'] + case['duration_us']
        previous = intervals.setdefault(case['source_group'], [])
        if any(start < old_end and old_start < end for old_start, old_end in previous):
            raise ValueError("overlapping source intervals would double-count exposure")
        previous.append((start, end))
        rights = keys(case['rights'], 'license source_url attribution commercial_evaluation commercial_training')
        if rights['license'] not in ('CC-BY-4.0', 'owner-permission'):
            raise ValueError("unsupported or unknown rights")
        text(rights['source_url']); text(rights['attribution'])
        if type(rights['commercial_training']) is not bool or rights['commercial_evaluation'] is not True:
            raise ValueError("commercial evaluation permission required")
        events(case['events'], classes, case['duration_us'])
    return value


def validate_predictions(value, manifest, benchmark_sha256):
    keys(value, 'schema benchmark_sha256 model predictions')
    if value['schema'] != 'score-retail-predictions/v1' or value['benchmark_sha256'] != benchmark_sha256:
        raise ValueError("prediction benchmark binding mismatch")
    model = keys(value['model'], 'id revision')
    text(model['id']); text(model['revision'])
    cases = {c['id']: c for c in manifest['cases']}
    rows = value['predictions']
    if not isinstance(rows, list) or len(rows) > len(cases):
        raise ValueError("invalid predictions")
    seen = set()
    for row in rows:
        keys(row, 'case_id status raw_output events')
        if not isinstance(row['case_id'], str) or row['case_id'] not in cases or row['case_id'] in seen:
            raise ValueError("unknown or duplicate prediction case")
        seen.add(row['case_id'])
        if row['status'] not in ('ok', 'unobservable', 'invalid'):
            raise ValueError("invalid prediction status")
        if not isinstance(row['raw_output'], str) or len(row['raw_output'].encode()) > 65536:
            raise ValueError("raw output exceeds bound")
        events(row['events'], manifest['classes'], cases[row['case_id']]['duration_us'], True)
        if row['status'] != 'ok' and row['events']:
            raise ValueError("non-ok result cannot contain events")
        if row['status'] == 'ok':
            raw = keys(strict_json_loads(row['raw_output']), 'events')
            events(raw['events'], manifest['classes'], cases[row['case_id']]['duration_us'], True)
            if raw['events'] != row['events']:
                raise ValueError("raw output and imported events differ")
    return value


def temporal_iou(left, right):
    intersection = max(0, min(left['end_us'], right['end_us']) - max(left['start_us'], right['start_us']))
    union = max(left['end_us'], right['end_us']) - min(left['start_us'], right['start_us'])
    return intersection / union


def match(predictions, targets, threshold):
    """Maximum-cardinality matching with deterministic overlap-first tie ordering."""
    edges = [sorted((j for j, truth in enumerate(targets)
                     if p['label'] == truth['label'] and temporal_iou(p, truth) >= threshold),
                    key=lambda j: (-temporal_iou(p, targets[j]), j)) for p in predictions]
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j not in visited:
                visited.add(j)
                if j not in owners or augment(owners[j], visited):
                    owners[j] = i
                    return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return sorted((i, j) for j, i in owners.items())


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def score(manifest, predictions, split='test'):
    selected = [c for c in manifest['cases'] if c['split'] == split]
    if not selected or any(not c['annotation_complete'] for c in selected):
        raise ValueError("scoring needs a nonempty fully reviewed split")
    rows = {r['case_id']: r for r in predictions['predictions']}
    counts = {label: [0, 0, 0] for label in manifest['classes']}
    states = {s: 0 for s in ('ok', 'unobservable', 'invalid', 'missing')}
    processed = normal = total = 0
    overlaps, details = [], []
    for case in selected:
        total += case['duration_us']
        row = rows.get(case['id'])
        status = row['status'] if row else 'missing'
        states[status] += 1
        truths = case['events']
        predicted = [p for p in row['events'] if p['score'] >= manifest['thresholds']['score_threshold']] if status == 'ok' else []
        if status == 'ok':
            processed += case['duration_us']
            occupied, cursor = 0, 0
            for truth in sorted(truths, key=lambda event: (event['start_us'], event['end_us'])):
                occupied += max(0, truth['end_us'] - max(truth['start_us'], cursor))
                cursor = max(cursor, truth['end_us'])
            normal += case['duration_us'] - occupied
        pairs = match(predicted, truths, manifest['thresholds']['tiou_threshold'])
        used_p, used_t = {i for i, _ in pairs}, {j for _, j in pairs}
        overlaps.extend(temporal_iou(predicted[i], truths[j]) for i, j in pairs)
        for i, p in enumerate(predicted):
            counts[p['label']][0 if i in used_p else 1] += 1
        for j, truth in enumerate(truths):
            if j not in used_t:
                counts[truth['label']][2] += 1
        details.append({'case_id': case['id'], 'status': status, 'tp': len(pairs),
                        'fp': len(predicted) - len(pairs), 'fn': len(truths) - len(pairs)})
    sums = [sum(c[i] for c in counts.values()) for i in range(3)]
    hours = processed / 3_600_000_000
    return {'schema': 'score-retail-report/v1', 'mode': 'offline_video', 'split': split,
            'model': predictions['model'], 'benchmark_sha256': predictions['benchmark_sha256'],
            'thresholds': manifest['thresholds'], 'micro': metrics(*sums),
            'per_class': {k: metrics(*v) for k, v in counts.items()},
            'matched_mean_tiou': sum(overlaps) / len(overlaps) if overlaps else None,
            'false_alerts_per_processed_camera_hour': sums[1] / hours if hours else None,
            'zero_fp_poisson_upper95_per_hour': -math.log(.05) / hours if hours and sums[1] == 0 else None,
            'processed_camera_hours': hours, 'reviewed_normal_camera_hours': normal / 3_600_000_000,
            'total_camera_hours': total / 3_600_000_000,
            'processed_duration_fraction': processed / total, 'case_status_counts': states,
            'cases': details, 'model_execution_verified': False, 'gpu_execution_verified': False,
            'live_qualification': False, 'customer_acceptance': False,
            'station_domain_qualification': False, 'measured_cost_usd': None,
            'inference_latency_ms': None, 'causal_detection_delay_ms': None}


def write_private(path, value):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("output symlink forbidden")
    if any((p / '.git').exists() for p in (path.parent, *path.parent.parents)):
        raise ValueError("output must be outside repositories")
    parent = path.parent
    if not parent.is_dir() or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077:
        raise ValueError("output directory must be owner private")
    with open(path, 'x', opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'score'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('predictions', type=Path, nargs='?')
    parser.add_argument('--media-root', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        with args.manifest.open('rb') as stream:
            manifest_bytes = stream.read(16 * 1024 * 1024 + 1)
        if len(manifest_bytes) > 16 * 1024 * 1024:
            raise ValueError("JSON exceeds byte limit")
        benchmark_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = validate_manifest(strict_json_loads(manifest_bytes), args.media_root)
        result = {'schema': 'score-retail-validation/v1', 'cases': len(manifest['cases']),
                  'benchmark_sha256': benchmark_sha256, 'annotation_complete': all(c['annotation_complete'] for c in manifest['cases']),
                  'commercial_training_authorized': all(c['rights']['commercial_training'] for c in manifest['cases']),
                  'live_qualification': False, 'customer_acceptance': False}
        if args.command == 'score':
            if args.predictions is None or args.output is None:
                raise ValueError("score requires predictions and private output")
            with args.predictions.open('rb') as stream:
                prediction_bytes = stream.read(16 * 1024 * 1024 + 1)
            if len(prediction_bytes) > 16 * 1024 * 1024:
                raise ValueError("JSON exceeds byte limit")
            predictions = validate_predictions(strict_json_loads(prediction_bytes), manifest, benchmark_sha256)
            result = score(manifest, predictions, args.split)
            write_private(args.output, result)
        elif args.predictions is not None or args.output is not None:
            raise ValueError("validate takes no predictions or output")
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RecursionError) as error:
        print(json.dumps({'error': str(error), 'qualification': 'not_run'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
