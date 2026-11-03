from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from reliquary.vision import retail


def png(color):
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * 2) * 2)) + chunk(b"IEND", b""))


def recording(index, split):
    frames = []
    for i, pts in enumerate((0, 2_000_000, 4_000_000, 6_000_000, 8_000_000)):
        raw = png((index * 20, i * 20, 100))
        frames.append({"data_base64": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest(),
                       "width": 2, "height": 2, "pts_us": pts})
    return {"id": f"clip-{index}", "station_id": f"station-{index}", "camera_id": "camera-1", "day": "2026-11-02",
            "split": split, "rights": {"basis": "synthetic", "reference": "generated test fixture", "approved": True},
            "annotation_complete": True, "source_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "media": {"kind": "frames", "clip_start_us": 0, "duration_us": 10_000_000, "frames": frames},
            "events": [{"id": "event-1", "label": "zone_entry", "start_us": 1_000_000, "end_us": 4_000_000}],
            "cutoffs_us": [2_000_000, 5_000_000, 10_000_000]}


def manifest():
    return {"schema_version": 1, "labels": ["zone_entry"],
            "recordings": [recording(1, "train"), recording(2, "validation"), recording(3, "test")]}


def write(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def prepared(tmp_path):
    tmp_path.chmod(0o700)
    source = write(tmp_path / "source.private.json", manifest())
    output = tmp_path / "prepared"
    report = retail.prepare(source, output, max_frames=2)
    return output, report


def predictions(output, completion=None):
    raw = (output / "benchmark.private.json").read_bytes()
    values = []
    for cutoff in (2_000_000, 5_000_000, 10_000_000):
        items = ([] if cutoff == 2_000_000 else
                 [{"label": "zone_entry", "start_us": 1_000_000, "end_us": 4_000_000, "confidence": 0.9}])
        values.append({"window_id": f"clip-3.{cutoff}", "status": "succeeded", "latency_ms": 200,
                       "completion": json.dumps({"type": "retail_events", "items": items})})
    if completion is not None:
        for value in values:
            value["completion"] = completion
    return {"schema": retail.PREDICTION_SCHEMA, "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
            "model": {"id": "fixture/tiny", "revision": "a" * 40, "processor_revision": "b" * 40},
            "predictions": values}


def evaluate(output, value, name="evaluation.private.json"):
    source = write(output.parent / "predictions.private.json", value)
    return retail.evaluate(output / "benchmark.private.json", source, output.parent / name)


def test_prepare_uses_causal_prefixes_and_keeps_truth_out_of_requests(prepared):
    output, report = prepared
    requests = [json.loads(line) for line in (output / "requests.private.jsonl").read_text().splitlines()]
    assert report["training_examples"] == 4
    assert report["test_windows"] == 3
    assert set(requests[0]) == {"id", "task", "prompt", "media"}
    assert "event-1" not in json.dumps(requests)
    assert "is_verified_negative" not in json.dumps(requests)
    for request in requests:
        assert len(request["media"]["frames"]) <= 2
        assert all(frame["pts_us"] < request["media"]["duration_us"] for frame in request["media"]["frames"])
    assert [f["pts_us"] for f in requests[-1]["media"]["frames"]] == [0, 8_000_000]
    training = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert {row["split"] for row in training} == {"train", "validation"}
    assert all(row["media"]["duration_us"] != 2_000_000 for row in training)
    assert (output / "training/manifest.json").is_file()
    assert all(path.stat().st_mode & 0o077 == 0 for path in output.rglob("*"))


def test_evaluation_only_preserves_test_and_never_exports_training(tmp_path):
    tmp_path.chmod(0o700)
    value = manifest()
    value["recordings"] = [value["recordings"][-1]]
    source = write(tmp_path / "source.private.json", value)
    with pytest.raises(ValueError, match="train and validation"):
        retail.prepare(source, tmp_path / "refused")
    output = tmp_path / "evaluation"
    assert retail.main(["prepare", str(source), str(output), "--evaluation-only"]) == 0
    assert not (output / "training.jsonl").exists()
    assert not (output / "training").exists()
    assert evaluate(output, predictions(output))["recall"] == 1
    write(source, manifest())
    with pytest.raises(ValueError, match="test recordings only"):
        retail.prepare(source, tmp_path / "mixed", evaluation_only=True)


def test_prefix_union_and_repeated_alerts_do_not_inflate_camera_hours(prepared):
    output, _ = prepared
    report = evaluate(output, predictions(output))
    assert (report["true_positives"], report["false_positives"], report["false_negatives"]) == (1, 0, 0)
    assert report["recall"] == report["precision"] == 1
    assert report["processed_camera_hours"] == report["total_camera_hours"] == 10 / 3600
    assert report["detection_delay_seconds"]["p50"] == pytest.approx(4.2)
    assert report["inference_latency_ms"]["p95"] == 200
    assert report["complete"] is True
    assert report["customer_acceptance"] is False
    assert report["synthetic_data"] is True
    assert report["model_identity_assurance"] == "caller-declared-saved-predictions"


def test_duplicate_alert_in_one_window_is_still_a_false_positive(prepared):
    output, _ = prepared
    value = predictions(output)
    completion = json.loads(value["predictions"][1]["completion"])
    completion["items"] *= 2
    value["predictions"][1]["completion"] = json.dumps(completion)
    report = evaluate(output, value)
    assert (report["true_positives"], report["false_positives"]) == (1, 1)
    assert report["false_alerts_per_camera_hour"] == 360


def test_matching_finds_maximum_cardinality_instead_of_greedy_iou():
    truth = [{"id": "a", "recording_id": "r", "label": "x", "start_us": 0, "end_us": 4},
             {"id": "b", "recording_id": "r", "label": "x", "start_us": 2, "end_us": 6}]
    alerts = [{"recording_id": "r", "label": "x", "start_us": 0, "end_us": 5},
              {"recording_id": "r", "label": "x", "start_us": 0, "end_us": 2}]
    assert len(retail._match(alerts, truth, 0.3)) == 2


@pytest.mark.parametrize("failure", ["missing", "failed", "invalid"])
def test_missing_failed_and_invalid_inference_reduce_coverage(prepared, failure):
    output, _ = prepared
    value = predictions(output)
    if failure == "missing":
        value["predictions"].pop()
    elif failure == "failed":
        value["predictions"][-1]["status"] = "failed"
    else:
        value["predictions"][-1]["completion"] = "not json"
    report = evaluate(output, value)
    assert report["complete"] is False
    assert report["successful_windows"] == 2
    assert report["coverage"] == 0.5
    assert len(report["failures"]) == 1


def test_all_failures_are_not_successful_negatives(prepared):
    output, _ = prepared
    report = evaluate(output, predictions(output, "not JSON"))
    assert report["coverage"] == 0
    assert report["processed_camera_hours"] == 0
    assert report["false_alerts_per_camera_hour"] is None
    assert report["recall"] == 0
    assert report["complete"] is False


def test_later_prefix_coverage_does_not_hide_a_failed_earlier_decision(prepared):
    output, _ = prepared
    value = predictions(output)
    value["predictions"][0]["status"] = "failed"
    report = evaluate(output, value)
    assert report["coverage"] == 1
    assert report["successful_window_fraction"] == pytest.approx(2 / 3)
    assert report["complete"] is False


def test_future_event_predictions_fail_instead_of_becoming_silent_negatives(prepared):
    output, _ = prepared
    value = predictions(output)
    value["predictions"][0]["completion"] = json.dumps({"type": "retail_events", "items": [
        {"label": "zone_entry", "start_us": 1_000_000, "end_us": 3_000_000, "confidence": 1}]})
    report = evaluate(output, value)
    assert report["complete"] is False
    assert report["failures"][0]["reason"] == "invalid_completion"


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(benchmark_sha256="f" * 64),
    lambda v: v["model"].update(revision="main"),
    lambda v: v["predictions"].append(copy.deepcopy(v["predictions"][0])),
    lambda v: v["predictions"][0].update(window_id="unknown"),
    lambda v: v["predictions"][0].update(latency_ms=float("nan")),
])
def test_rejects_unbound_duplicate_unknown_or_nonfinite_predictions(prepared, mutation):
    output, _ = prepared
    value = predictions(output)
    mutation(value)
    with pytest.raises(ValueError):
        evaluate(output, value)
    assert not (output.parent / "evaluation.private.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda v: v["recordings"][1].update(station_id="station-1"),
    lambda v: v["recordings"][1].update(source_sha256=v["recordings"][0]["source_sha256"]),
    lambda v: v["recordings"][0]["rights"].update(approved=False),
    lambda v: v["recordings"][0].update(annotation_complete=False),
    lambda v: v["recordings"][0]["media"]["frames"][0].update(pts_us=10_000_000),
])
def test_rejects_unreviewed_data_split_leaks_and_invalid_pts(tmp_path, mutation):
    tmp_path.chmod(0o700)
    value = manifest()
    mutation(value)
    source = write(tmp_path / "input.private.json", value)
    with pytest.raises(ValueError):
        retail.prepare(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_disjoint_same_camera_recordings_accumulate_exposure_and_overlaps_fail(tmp_path):
    tmp_path.chmod(0o700)
    value = manifest()
    second = recording(4, "test")
    second.update(station_id="station-3", source_sha256="d" * 64)
    second["media"]["clip_start_us"] = 10_000_000
    value["recordings"].append(second)
    source = write(tmp_path / "source.private.json", value)
    retail.prepare(source, tmp_path / "good")
    second["media"]["clip_start_us"] = 9_999_999
    write(source, value)
    with pytest.raises(ValueError, match="overlap"):
        retail.prepare(source, tmp_path / "bad")


def test_verified_normal_footage_has_null_recall_and_can_count_false_alerts(tmp_path):
    tmp_path.chmod(0o700)
    value = manifest()
    value["recordings"][-1]["events"] = []
    source = write(tmp_path / "source.private.json", value)
    output = tmp_path / "prepared"
    retail.prepare(source, output)
    value = predictions(output, '{"type":"retail_events","items":[]}')
    report = evaluate(output, value)
    assert report["recall"] is None and report["precision"] is None
    assert report["false_alerts_per_camera_hour"] == 0
    value["predictions"][-1]["completion"] = json.dumps({"type": "retail_events", "items": [
        {"label": "zone_entry", "start_us": 1, "end_us": 2, "confidence": 1}]})
    report = evaluate(output, value, "false-positive.private.json")
    assert report["false_positives"] == 1
    assert report["false_alerts_per_camera_hour"] == 360


def test_prepare_and_evaluate_refuse_overwrite_and_repository_outputs(prepared, tmp_path):
    output, _ = prepared
    source = output.parent / "source.private.json"
    old = (output / "benchmark.private.json").read_bytes()
    with pytest.raises(ValueError, match="exists"):
        retail.prepare(source, output)
    assert (output / "benchmark.private.json").read_bytes() == old
    value = predictions(output)
    evaluate(output, value)
    with pytest.raises(FileExistsError):
        evaluate(output, value)
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside repositories"):
        retail.prepare(source, repository / "output")


def test_local_png_paths_are_verified_before_preparation(tmp_path):
    tmp_path.chmod(0o700)
    value = manifest()
    for recording_value in value["recordings"]:
        for i, frame in enumerate(recording_value["media"]["frames"]):
            name = f"{recording_value['id']}-{i}.png"
            (tmp_path / name).write_bytes(base64.b64decode(frame.pop("data_base64")))
            frame["path"] = name
    source = write(tmp_path / "source.private.json", value)
    retail.prepare(source, tmp_path / "good")
    (tmp_path / value["recordings"][0]["media"]["frames"][0]["path"]).write_bytes(png((1, 1, 1)))
    with pytest.raises(ValueError, match="SHA256"):
        retail.prepare(source, tmp_path / "changed")


def test_repeated_window_exports_are_bounded_before_joining(monkeypatch):
    monkeypatch.setattr(retail, "MAX_INPUT_BYTES", 40)
    with pytest.raises(ValueError, match="export exceeds"):
        retail._jsonl([{"x": "0" * 20}, {"x": "1" * 20}])
