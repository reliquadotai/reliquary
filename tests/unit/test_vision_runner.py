import base64
import io
import json
from pathlib import Path
import tarfile
import time
from unittest.mock import Mock

import pytest

from reliquary.vision import runner


@pytest.fixture
def allocation(tmp_path):
    from PIL import Image
    model, job, output = (tmp_path / name for name in ("model", "job", "output"))
    (model / "weights").mkdir(parents=True)
    job.mkdir()
    (model / "weights" / "config.json").write_text('{"model_type":"smolvlm"}')
    (model / "weights" / "model.safetensors").write_bytes(b"fixture, not model weights")
    (model / "weights" / "tokenizer_config.json").write_text('{"chat_template":"template"}')
    (model / "training.py").write_text("# runtime fixture\n")
    revision = "a" * 40
    lock = runner.lock_model(model, "test/model", revision, revision)
    rows = []
    for split, color in (("train", "red"), ("validation", "blue")):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        data = buffer.getvalue()
        rows.append({
            "id": split, "source_id": split, "split": split, "task": "image_grounding",
            "prompt": "Return JSON grounding.", "annotation_complete": True,
            "is_verified_negative": False,
            "target": {"type": "grounding", "items": [{"label": "ball", "box": [0, 0, 1000, 1000]}]},
            "media": {"kind": "image", "clip_start_us": 0, "duration_us": 0, "frames": [{
                "data_base64": base64.b64encode(data).decode(),
                "sha256": runner.hashlib.sha256(data).hexdigest(),
                "width": 2, "height": 2, "pts_us": 0,
            }]},
        })
    (job / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    profile = {
        "digest": "b" * 64, "imageDigest": "sha256:" + "c" * 64,
        "conformanceDigest": "d" * 64, "environmentId": "vision-test",
        "environmentVersion": "1", "offerId": "cpu-test",
        "modelId": "test/model", "modelRevision": revision, "datasetFormat": "vision",
        "method": "sft", "environmentDigest": runner.checksum(model / "training.py"),
        "gpuCount": 0, "maxSteps": 10, "maxHours": 1,
        "vision": {**lock, "schemaVersion": 1, "mode": "timestamped_frames_v1",
                   "maxFrames": 4, "maxPixels": 1000, "maxSequenceLength": 1024,
                   "task": "image_grounding"},
    }
    configuration = {
        "schemaVersion": 2, "profileDigest": profile["digest"],
        "model": {"id": profile["modelId"]},
        "environment": {"id": profile["environmentId"], "version": profile["environmentVersion"]},
        "dataset": {"sha256": runner.checksum(job / "dataset.jsonl")},
        "optimization": {"maxSteps": 1, "method": "sft"},
        "compute": {"maxHours": 1, "offerId": "cpu-test", "gpuCount": 0, "providerPolicy": "private"},
    }
    value = {"profile": profile, "configuration": configuration,
             "deadlineEpoch": time.time() + 3500, "finalizationReserveSeconds": 60}
    (job / "job.json").write_text(json.dumps(value))
    return model, job, output, value


def mock_training(request, dataset, media_root, weights, output, adapter, *, check_deadline):
    check_deadline()
    output.mkdir()
    (output / "adapter").mkdir()
    (output / "adapter" / "adapter_model.safetensors").write_bytes(b"test adapter")
    (output / "adapter" / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": request["model_id"],
        "revision": request["model_revision"],
    }))
    assert len(dataset["examples"]) == 2
    return {"reloadVerified": True, "steps": 1, "baseline": 0.25, "postRun": 0.5,
            "samples": 1, "evaluationSetDigest": "f" * 64, "device": "cpu", "metric": "mean_task_reward"}


def test_adapter_delivery_and_parent_continuation(allocation, monkeypatch):
    model, job, output, value = allocation
    from reliquary.vision import runtime
    monkeypatch.setattr(runtime, "run_job", mock_training)
    report = runner.execute(job, model, output)
    assert report == runner.read_json(output / "result.json")
    assert report["vision"] == value["profile"]["vision"]
    checkpoint = output / report["checkpoint"]
    assert runner.checksum(checkpoint) == report["checkpointSha256"]
    with tarfile.open(checkpoint) as archive:
        assert set(archive.getnames()) == runner.ADAPTER_FILES
        assert all(entry.mtime == 0 for entry in archive.getmembers())
    (job / "parent.checkpoint").write_bytes(checkpoint.read_bytes())
    value["configuration"]["parent"] = {"sha256": report["checkpointSha256"]}
    (job / "job.json").write_text(json.dumps(value))
    continued = output.with_name("continued")
    runner.execute(job, model, continued)
    assert (continued / "parent" / "binding.json").is_file()
    value["profile"]["vision"]["task"] = "football_events"
    parent_output = output.with_name("wrong-parent")
    parent_output.mkdir()
    with pytest.raises(ValueError, match="different model or visual contract"):
        runner.load_parent(job, parent_output, value["configuration"], value["profile"])


@pytest.mark.parametrize("mutation", [
    lambda v: v["profile"].update(datasetFormat="instruction"),
    lambda v: v["profile"].update(gpuCount=2),
    lambda v: v["profile"].update(gpuCount=True),
    lambda v: v["profile"]["vision"].update(processorRevision="main"),
    lambda v: v["profile"]["vision"].update(mode="native_video"),
    lambda v: v["profile"]["vision"].update(extra="unbound"),
    lambda v: v["configuration"]["optimization"].update(maxSteps=11),
    lambda v: v["configuration"]["compute"].update(maxHours=2),
    lambda v: v["configuration"]["model"].update(id="wrong/model"),
    lambda v: v["configuration"]["optimization"].update(method="grpo"),
    lambda v: v["configuration"]["compute"].update(providerPolicy="public"),
    lambda v: v.update(deadlineEpoch=time.time() + 30),
    lambda v: v.update(finalizationReserveSeconds=0),
])
def test_invalid_requests_fail_before_runtime(allocation, mutation):
    *_, value = allocation
    mutation(value)
    with pytest.raises(ValueError):
        runner.validate_job(value)


@pytest.mark.parametrize("filename", ["config.json", "tokenizer_config.json", "model.safetensors"])
def test_snapshot_mutation_rejected(allocation, filename):
    model, _, _, value = allocation
    (model / "weights" / filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        runner.verify_model(model, value["profile"])


def test_extra_model_file_and_symlink_rejected(allocation):
    model, _, _, value = allocation
    unexpected = model / "weights" / "new_config.json"
    unexpected.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        runner.verify_model(model, value["profile"])
    unexpected.unlink()
    unexpected.symlink_to(model / "weights" / "config.json")
    with pytest.raises(ValueError, match="symlinks"):
        runner.verify_model(model, value["profile"])


def test_runtime_and_dataset_binding_fail_closed(allocation, monkeypatch):
    model, job, output, value = allocation
    from reliquary.vision import runtime
    called = Mock(side_effect=AssertionError("must not call runtime"))
    monkeypatch.setattr(runtime, "run_job", called)
    (job / "dataset.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="Dataset integrity"):
        runner.execute(job, model, output)
    called.assert_not_called()
    (model / "training.py").write_text("changed")
    with pytest.raises(ValueError, match="entrypoint"):
        runner.verify_model(model, value["profile"])


def test_failed_training_never_publishes_success(allocation, monkeypatch):
    model, job, output, _ = allocation
    from reliquary.vision import runtime
    def no_updates(*args, **kwargs):
        result = mock_training(*args, **kwargs)
        result["steps"] = 0
        return result
    monkeypatch.setattr(runtime, "run_job", no_updates)
    with pytest.raises(ValueError, match="complete an update"):
        runner.execute(job, model, output)
    assert not (output / "result.json").exists()


def test_parent_archive_rejects_links_and_unknown_members(allocation):
    _, job, output, value = allocation
    output.mkdir()
    parent = job / "parent.checkpoint"
    with tarfile.open(parent, "w") as archive:
        entry = tarfile.TarInfo("../adapter_config.json")
        entry.size = 2
        archive.addfile(entry, io.BytesIO(b"{}"))
    value["configuration"]["parent"] = {"sha256": runner.checksum(parent)}
    with pytest.raises(ValueError, match="bounded adapter"):
        runner.load_parent(job, output, value["configuration"], value["profile"])


def test_supervisor_kills_process_group_on_deadline(allocation, monkeypatch):
    model, job, output, _ = allocation
    child = Mock(pid=123)
    child.wait.side_effect = [runner.subprocess.TimeoutExpired("worker", 1), 0]
    spawn = Mock(return_value=child)
    kill = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    monkeypatch.setattr(runner.os, "killpg", kill)
    with pytest.raises(runner.subprocess.TimeoutExpired):
        runner.supervise(job, model, output)
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert spawn.call_args.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    kill.assert_called_once_with(123, runner.signal.SIGKILL)
    assert not (output / "result.json").exists()
