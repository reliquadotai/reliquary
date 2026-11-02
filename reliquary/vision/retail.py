"""Prepare causal retail frame windows and evaluate frozen saved predictions."""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from reliquary.shared.strict_json import strict_json_loads
from reliquary.vision.data import (
    MAX_INPUT_BYTES, _verify_png, load_verified_media_bytes, prepare_jsonl, validate_manifest,
)
from reliquary.vision.rewards import RETAIL_LABEL, temporal_iou, validate_target

BENCHMARK_SCHEMA = "retail-video-benchmark/v1"
PREDICTION_SCHEMA = "retail-video-predictions/v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _keys(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{name} has an unsupported schema")


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be integer microseconds in [{lower},{upper}]")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid {name}")


def _read(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds 5 MiB")
    return raw, strict_json_loads(raw)


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _jsonl(rows):
    parts, size = [], 0
    for row in rows:
        part = _json(row)
        size += len(part)
        if size > MAX_INPUT_BYTES:
            raise ValueError("export exceeds 5 MiB; use fewer windows or smaller frames")
        parts.append(part)
    return b"".join(parts)


def _private_parent(path):
    path = Path(path).absolute()
    if path.is_symlink() or any((p / ".git").exists() for p in (path.resolve(), *path.resolve().parents)):
        raise ValueError("outputs must be outside repositories and without symlinks")
    parent = path.parent.resolve()
    if not parent.is_dir() or parent.stat().st_mode & 0o077 or parent.stat().st_uid != os.getuid():
        raise ValueError("output parent must already be owner-private")
    return parent / path.name


def _write(path, raw):
    with open(path, "xb", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(raw)


def _prompt(labels, cutoff):
    return ("Identify only fully observed retail events in the supplied frames. "
            f"Allowed labels: {json.dumps(labels)}. Frame timestamps are clip-relative microseconds. "
            f"The observation ends at {cutoff} microseconds; do not predict future events. "
            'Return only JSON {"type":"retail_events","items":'
            '[{"label":"LABEL","start_us":0,"end_us":1,"confidence":1.0}]}; '
            'use an empty items list when no event is observed.')


def _event_items(recording, cutoff):
    return [{k: event[k] for k in ("label", "start_us", "end_us")} | {"confidence": 1.0}
            for event in recording["events"] if event["end_us"] <= cutoff]


def _example(recording, cutoff, frames, labels):
    items = _event_items(recording, cutoff)
    return {"id": f"{recording['id']}.{cutoff}", "source_id": recording["station_id"],
            "split": recording["split"], "task": "retail_events", "prompt": _prompt(labels, cutoff),
            "annotation_complete": True, "is_verified_negative": not items,
            "target": {"type": "retail_events", "items": items},
            "media": {"kind": "frames", "clip_start_us": recording["media"]["clip_start_us"],
                      "duration_us": cutoff, "frames": frames}}


def _normalize(value, base):
    """Reuse the vision parser and PNG verifier before exporting any window."""
    _keys(value, ("schema_version", "labels", "recordings"), "recording manifest")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported recording manifest version")
    labels = value["labels"]
    if (not isinstance(labels, list) or not 1 <= len(labels) <= 64
            or any(not isinstance(x, str) or not RETAIL_LABEL.fullmatch(x) for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("labels must be distinct ASCII event identifiers")
    recordings = copy.deepcopy(value["recordings"])
    if not isinstance(recordings, list) or not 1 <= len(recordings) <= 256:
        raise ValueError("recordings must contain 1 to 256 entries")
    rows, ids, stations, sources, camera_days = [], set(), {}, {}, {}
    frame_bytes = 0
    for recording in recordings:
        _keys(recording, ("id", "station_id", "camera_id", "day", "split", "rights",
                          "annotation_complete", "source_sha256", "media", "events", "cutoffs_us"), "recording")
        for field in ("id", "station_id", "camera_id"):
            _identifier(recording[field], field)
        if recording["id"] in ids:
            raise ValueError("duplicate recording id")
        ids.add(recording["id"])
        if not isinstance(recording["day"], str) or date.fromisoformat(recording["day"]).isoformat() != recording["day"]:
            raise ValueError("day must be canonical ISO date")
        split = recording["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("unsupported split")
        if stations.setdefault(recording["station_id"], split) != split:
            raise ValueError("station crosses dataset splits")
        camera_day = (recording["station_id"], recording["camera_id"], recording["day"])
        source = recording["source_sha256"]
        if not isinstance(source, str) or not _HASH.fullmatch(source):
            raise ValueError("invalid declared source SHA256")
        if sources.setdefault(source, split) != split:
            raise ValueError("declared source crosses dataset splits")
        rights = recording["rights"]
        _keys(rights, ("basis", "reference", "approved"), "rights declaration")
        if (rights["basis"] not in ("owned", "licensed", "synthetic") or rights["approved"] is not True
                or not isinstance(rights["reference"], str) or not 1 <= len(rights["reference"].strip()) <= 1024):
            raise ValueError("an approved rights self-declaration is required")
        if recording["annotation_complete"] is not True:
            raise ValueError("complete reviewed annotations are required")
        media = recording["media"]
        _keys(media, ("kind", "clip_start_us", "duration_us", "frames"), "media")
        duration = _integer(media["duration_us"], 1, 60_000_000, "duration_us")
        start = _integer(media["clip_start_us"], 0, 86_400_000_000 - duration, "clip_start_us")
        intervals = camera_days.setdefault(camera_day, [])
        if any(start < end and previous < start + duration for previous, end in intervals):
            raise ValueError("recording intervals overlap on the same camera-day")
        intervals.append((start, start + duration))
        if not isinstance(recording["events"], list) or len(recording["events"]) > 64:
            raise ValueError("too many events")
        event_ids = set()
        for event in recording["events"]:
            _keys(event, ("id", "label", "start_us", "end_us"), "event")
            _identifier(event["id"], "event id")
            if event["id"] in event_ids or event["label"] not in labels:
                raise ValueError("duplicate event id or unknown event label")
            event_ids.add(event["id"])
            for field in ("start_us", "end_us"):
                _integer(event[field], 0, duration, field)
            if event["start_us"] >= event["end_us"]:
                raise ValueError("event interval must be nonempty")
        cutoffs = recording["cutoffs_us"]
        if not isinstance(cutoffs, list) or not 1 <= len(cutoffs) <= 64:
            raise ValueError("cutoffs must contain 1 to 64 times")
        for cutoff in cutoffs:
            _integer(cutoff, 1, duration, "cutoff_us")
        if cutoffs != sorted(set(cutoffs)) or cutoffs[-1] != duration:
            raise ValueError("cutoffs must increase and include the recording end")
        if not isinstance(media["frames"], list) or not 1 <= len(media["frames"]) <= 64:
            raise ValueError("recording requires 1 to 64 frames")
        raw_frames, local_frames = [], []
        for frame in media["frames"]:
            if not isinstance(frame, dict):
                raise ValueError("frame must be an object")
            inline = "data_base64" in frame
            _keys(frame, ("data_base64" if inline else "path", "sha256", "width", "height", "pts_us"), "frame")
            _integer(frame["pts_us"], 0, duration - 1, "pts_us")
            _integer(frame["width"], 1, 2048, "width")
            _integer(frame["height"], 1, 2048, "height")
            local = {k: v for k, v in frame.items() if k != "data_base64"}
            if inline:
                local["path"] = f"frames/{frame['sha256']}.png"
                if not isinstance(frame["data_base64"], str):
                    raise ValueError("inline PNG must be canonical base64")
                try:
                    raw = base64.b64decode(frame["data_base64"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise ValueError("inline PNG must be canonical base64") from exc
                if base64.b64encode(raw).decode() != frame["data_base64"] or hashlib.sha256(raw).hexdigest() != frame["sha256"]:
                    raise ValueError("inline PNG SHA256 or base64 mismatch")
                _verify_png(raw, frame)
            else:
                raw = load_verified_media_bytes({"media": {"frames": [local]}}, base)[0]
            frame_bytes += len(raw)
            if frame_bytes > MAX_INPUT_BYTES:
                raise ValueError("recording media exceeds 5 MiB")
            local_frames.append(local)
            raw_frames.append({k: frame[k] for k in ("sha256", "width", "height", "pts_us")} |
                              {"data_base64": base64.b64encode(raw).decode()})
        rows.append(_example(recording, duration, local_frames, labels))
        media["frames"] = raw_frames
        if not any(frame["pts_us"] < cutoffs[0] for frame in raw_frames):
            raise ValueError("every cutoff requires at least one observed frame")
    validate_manifest({"schema_version": 1, "examples": rows})
    if sum(len(r["cutoffs_us"]) for r in recordings) > 256:
        raise ValueError("more than 256 windows")
    return {"schema_version": 1, "labels": labels, "recordings": recordings}


def _windows(value, max_frames):
    windows = []
    for recording in value["recordings"]:
        for cutoff in recording["cutoffs_us"]:
            observed = [f for f in recording["media"]["frames"] if f["pts_us"] < cutoff]
            count = min(max_frames, len(observed))
            indices = ([len(observed) - 1] if count == 1 else
                       [i * (len(observed) - 1) // (count - 1) for i in range(count)])
            row = _example(recording, cutoff, [observed[i] for i in indices], value["labels"])
            windows.append((recording, cutoff, row))
    return windows


def prepare(source, output, *, max_frames=4, evaluation_only=False):
    _integer(max_frames, 1, 64, "max_frames")
    raw, value = _read(source)
    value = _normalize(value, Path(source).parent)
    windows = _windows(value, max_frames)
    training = [row for recording, cutoff, row in windows if recording["split"] != "test"
                and not any(e["start_us"] < cutoff < e["end_us"] for e in recording["events"])]
    if evaluation_only and any(r["split"] != "test" for r in value["recordings"]):
        raise ValueError("evaluation-only preparation requires test recordings only")
    if not evaluation_only and {row["split"] for row in training} != {"train", "validation"}:
        raise ValueError("closed-event training windows need train and validation stations")
    heldout = [(recording, cutoff, row) for recording, cutoff, row in windows if recording["split"] == "test"]
    if not heldout:
        raise ValueError("a separate test station is required")
    benchmark = {"schema": BENCHMARK_SCHEMA, "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                 "max_frames": max_frames, "labels": value["labels"],
                 "recordings": [r for r in value["recordings"] if r["split"] == "test"],
                 "mode": "timestamped_frames_v1", "rights_assurance": "caller-self-declared"}
    benchmark_raw = _json(benchmark)
    if len(benchmark_raw) > MAX_INPUT_BYTES:
        raise ValueError("benchmark exceeds 5 MiB")
    training_raw = _jsonl(training)
    requests_raw = _jsonl({k: row[k] for k in ("id", "task", "prompt", "media")} for _, _, row in heldout)
    output = _private_parent(output)
    if os.path.lexists(output):
        raise ValueError("output exists; use a new directory")
    output.mkdir(mode=0o700)
    if not evaluation_only:
        _write(output / "training.jsonl", training_raw)
        prepare_jsonl(output / "training.jsonl", output / "training")
    _write(output / "benchmark.private.json", benchmark_raw)
    _write(output / "requests.private.jsonl", requests_raw)
    result = {"schema": "retail-video-preparation/v1", "benchmark_sha256": hashlib.sha256(benchmark_raw).hexdigest(),
              "training_examples": len(training), "test_windows": len(heldout),
              "test_in_training": False, "media_mode": "timestamped_frames_v1",
              "rights_assurance": "caller-self-declared", "gpu_qualification": "not_run",
              "synthetic_data": all(r["rights"]["basis"] == "synthetic" for r in value["recordings"]),
              "customer_acceptance": False}
    _write(output / "preparation.private.json", _json(result))
    for item in output.rglob("*"):
        item.chmod(0o700 if item.is_dir() else 0o600)
    return result


def _percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def _deduplicate(alerts, threshold):
    result = []
    for alert in sorted(alerts, key=lambda a: (a["available_us"], a["window_id"], a["start_us"], a["end_us"])):
        previous = next((old for old in result if old["recording_id"] == alert["recording_id"]
                         and old["label"] == alert["label"] and alert["window_id"] not in old["windows"]
                         and temporal_iou(old, alert) >= threshold), None)
        if previous is None:
            result.append(alert | {"windows": {alert["window_id"]}})
        else:
            previous["windows"].add(alert["window_id"])
    return result


def _match(alerts, truth, threshold):
    """Maximum-cardinality class/interval matching; deterministic IoU tie order."""
    edges = [sorted((j for j, event in enumerate(truth) if event["recording_id"] == alert["recording_id"]
                     and event["label"] == alert["label"] and temporal_iou(event, alert) >= threshold),
                    key=lambda j: (-temporal_iou(truth[j], alert), truth[j]["id"])) for alert in alerts]
    assigned = {}
    def augment(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target], seen):
                assigned[target] = index
                return True
        return False
    for index in range(len(alerts)):
        augment(index, set())
    return assigned


def evaluate(benchmark_path, prediction_path, output, *, confidence=0.5, temporal_threshold=0.5):
    if (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1
            or type(temporal_threshold) not in (int, float) or not math.isfinite(temporal_threshold) or not 0 < temporal_threshold <= 1):
        raise ValueError("invalid confidence or temporal IoU threshold")
    raw, benchmark = _read(benchmark_path)
    _keys(benchmark, ("schema", "source_manifest_sha256", "max_frames", "labels", "recordings", "mode", "rights_assurance"), "benchmark")
    if (benchmark["schema"] != BENCHMARK_SCHEMA or benchmark["mode"] != "timestamped_frames_v1"
            or benchmark["rights_assurance"] != "caller-self-declared"
            or not isinstance(benchmark["source_manifest_sha256"], str) or not _HASH.fullmatch(benchmark["source_manifest_sha256"])):
        raise ValueError("unsupported benchmark")
    _integer(benchmark["max_frames"], 1, 64, "max_frames")
    value = _normalize({"schema_version": 1, "labels": benchmark["labels"], "recordings": benchmark["recordings"]}, Path(benchmark_path).parent)
    if any(r["split"] != "test" for r in value["recordings"]):
        raise ValueError("evaluation requires frozen test recordings only")
    prediction_raw, predictions = _read(prediction_path)
    _keys(predictions, ("schema", "benchmark_sha256", "model", "predictions"), "predictions")
    if predictions["schema"] != PREDICTION_SCHEMA or predictions["benchmark_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("predictions do not bind this benchmark")
    model = predictions["model"]
    _keys(model, ("id", "revision", "processor_revision"), "model")
    if (not isinstance(model["id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["id"])
            or any(not isinstance(model[k], str) or not _REVISION.fullmatch(model[k]) for k in ("revision", "processor_revision"))):
        raise ValueError("model and processor require immutable revisions")
    windows = {row["id"]: (r, cutoff) for r, cutoff, row in _windows(value, benchmark["max_frames"])}
    items = predictions["predictions"]
    if not isinstance(items, list) or len(items) > len(windows):
        raise ValueError("invalid prediction count")
    seen, successful, alerts, latencies, failures = set(), {}, [], [], []
    for item in items:
        _keys(item, ("window_id", "status", "completion", "latency_ms"), "prediction")
        window_id = item["window_id"]
        if not isinstance(window_id, str) or window_id not in windows or window_id in seen:
            raise ValueError("unknown or duplicate prediction window")
        seen.add(window_id)
        latency = item["latency_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or not 0 <= latency <= 3_600_000:
            raise ValueError("latency_ms must be finite and between zero and one hour")
        if item["status"] not in ("succeeded", "failed") or not isinstance(item["completion"], str):
            raise ValueError("invalid prediction status or completion")
        recording, cutoff = windows[window_id]
        if item["status"] == "failed":
            failures.append({"window_id": window_id, "reason": "inference_failed"})
            continue
        try:
            target = validate_target(strict_json_loads(item["completion"]), "retail_events", cutoff)
        except (ValueError, TypeError, RecursionError):
            failures.append({"window_id": window_id, "reason": "invalid_completion"})
            continue
        successful[recording["id"]] = max(successful.get(recording["id"], 0), cutoff)
        latencies.append(latency)
        for event in target["items"]:
            if event["confidence"] >= confidence:
                alerts.append(event | {"recording_id": recording["id"], "window_id": window_id,
                                       "available_us": cutoff + latency * 1000})
    failures.extend({"window_id": window_id, "reason": "missing_prediction"} for window_id in sorted(set(windows) - seen))
    alerts = _deduplicate(alerts, temporal_threshold)
    truth = [event | {"recording_id": r["id"]} for r in value["recordings"] for event in r["events"]]
    assigned = _match(alerts, truth, temporal_threshold)
    tp, fp, fn = len(assigned), len(alerts) - len(assigned), len(truth) - len(assigned)
    delays = [(alerts[prediction]["available_us"] - truth[event]["start_us"]) / 1_000_000
              for event, prediction in assigned.items()]
    total_us = sum(r["media"]["duration_us"] for r in value["recordings"])
    processed_us = sum(successful.values())
    result = {"schema": "retail-video-evaluation/v1", "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
              "predictions_sha256": hashlib.sha256(prediction_raw).hexdigest(),
              "model": model, "confidence_threshold": confidence, "temporal_iou_threshold": temporal_threshold,
              "true_positives": tp, "false_positives": fp, "false_negatives": fn,
              "precision": tp / (tp + fp) if tp + fp else None,
              "recall": tp / (tp + fn) if tp + fn else None,
              "total_camera_hours": total_us / 3_600_000_000,
              "processed_camera_hours": processed_us / 3_600_000_000,
              "coverage": processed_us / total_us,
              "false_alerts_per_camera_hour": fp * 3_600_000_000 / processed_us if processed_us else None,
              "detection_delay_seconds": {"p50": _percentile(delays, 0.5), "p95": _percentile(delays, 0.95)},
              "inference_latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
              "window_count": len(windows), "successful_windows": len(windows) - len(failures),
              "successful_window_fraction": (len(windows) - len(failures)) / len(windows),
              "failures": failures, "complete": not failures, "matching": "maximum-cardinality-class-interval",
              "coverage_basis": "union-of-successful-prefix-windows-per-recording",
              "model_identity_assurance": "caller-declared-saved-predictions",
              "synthetic_data": all(r["rights"]["basis"] == "synthetic" for r in value["recordings"]),
              "rights_assurance": "caller-self-declared", "gpu_qualification": "not_run", "customer_acceptance": False}
    output = _private_parent(output)
    _write(output, _json(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("input", type=Path)
    preparation.add_argument("output", type=Path)
    preparation.add_argument("--max-frames", type=int, default=4)
    preparation.add_argument("--evaluation-only", action="store_true")
    inference = commands.add_parser("infer")
    inference.add_argument("benchmark", type=Path)
    inference.add_argument("model", type=Path)
    inference.add_argument("output", type=Path)
    inference.add_argument("--device", choices=("cpu", "cuda"), required=True)
    inference.add_argument("--max-pixels", type=int, default=200704)
    inference.add_argument("--max-context-tokens", type=int, default=8192)
    inference.add_argument("--max-new-tokens", type=int, default=128)
    inference.add_argument("--seed", type=int, default=17)
    inference.add_argument("--max-seconds", type=int, default=300)
    inference.add_argument("--checkpoint", type=Path)
    inference.add_argument("--checkpoint-sha256")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("benchmark", type=Path)
    evaluation.add_argument("predictions", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--confidence", type=float, default=0.5)
    evaluation.add_argument("--temporal-iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.input, args.output, max_frames=args.max_frames,
                             evaluation_only=args.evaluation_only)
        elif args.command == "infer":
            from reliquary.vision.retail_inference import infer
            result = infer(args.benchmark, args.model, args.output, device=args.device,
                           max_pixels=args.max_pixels, max_context_tokens=args.max_context_tokens,
                           max_new_tokens=args.max_new_tokens, seed=args.seed, max_seconds=args.max_seconds,
                           checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256)
        else:
            result = evaluate(args.benchmark, args.predictions, args.output, confidence=args.confidence,
                              temporal_threshold=args.temporal_iou)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"error": str(error), "gpu_qualification": "not_run", "customer_acceptance": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
