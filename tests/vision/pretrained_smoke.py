"""Opt-in, offline CPU execution of the immutable small-model test fixture.

Set VISION_TEST_MODEL_DIR to the downloaded SmolVLM snapshot and
VISION_TEST_OUTPUT_DIR to an empty writable directory. All profile and hardware
identities here are synthetic test fixtures, never production qualification.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
REVISION = "067788b187b95ebe7b2e040b3e4299e342e5b8fd"
WEIGHTS_SHA256 = "dcc8f4c00d30d70bb3028a1bfc373e1b1972e25919f8788183cae4ae72431ce2"


def main() -> None:
    if not os.environ.get("VISION_TEST_MODEL_DIR"):
        print("Skipped: VISION_TEST_MODEL_DIR is not set.")
        return
    from PIL import Image
    from reliquary.vision import runner

    source = Path(os.environ["VISION_TEST_MODEL_DIR"])
    output = Path(os.environ["VISION_TEST_OUTPUT_DIR"])
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("Smoke output must be an empty regular directory")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "model"
    model.mkdir()
    shutil.copytree(source, model / "weights",
                    ignore=shutil.ignore_patterns(".cache", ".git", ".gitattributes"))
    if runner.checksum(model / "weights" / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("Test model weights differ from the pinned public fixture")
    runtime_root = Path(runner.__file__).resolve().parents[2]
    entrypoints = (Path(os.environ.get("VISION_TEST_ENTRYPOINT", "/opt/vision-runtime/training.py")),
                   runtime_root / "training.py",
                   runtime_root / "scripts" / "vision-training-runtime.py")
    entrypoint = next((path for path in entrypoints if path.is_file()), None)
    if entrypoint is None:
        raise ValueError("The installed runtime entrypoint is missing")
    shutil.copyfile(entrypoint, model / "training.py")
    lock = runner.lock_model(model, MODEL_ID, REVISION, REVISION)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    reports = []
    parent = None
    for name in ("sft", "continued", "frames"):
        frame_task = name == "frames"
        task = "football_events" if frame_task else "image_grounding"
        rows = []
        for split, colors in (("train", ("red", "blue")), ("test", ("yellow", "green"))):
            frames = []
            for number, color in enumerate(colors if frame_task else colors[:1]):
                buffer = io.BytesIO()
                Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
                content = buffer.getvalue()
                frames.append({
                    "data_base64": base64.b64encode(content).decode(),
                    "sha256": runner.hashlib.sha256(content).hexdigest(),
                    "width": 64, "height": 64, "pts_us": number * 1_000_000,
                })
            media = {"kind": "frames" if frame_task else "image", "frames": frames,
                     "clip_start_us": 0, "duration_us": 2_000_000 if frame_task else 0}
            if frame_task:
                media["score_fps"] = 25
            target = (
                {"type": "soccer_action", "items": [{"frame": 25, "action": "pass", "confidence": 1.0}]}
                if frame_task else
                {"type": "grounding", "items": [{"label": "square", "box": [0, 0, 1000, 1000]}]}
            )
            rows.append({
                "id": split, "source_id": split, "split": split, "task": task,
                "prompt": "Return the annotated objects or events as JSON.",
                "annotation_complete": True, "is_verified_negative": False,
                "media": media, "target": target,
            })
        job_dir = output / f"{name}-job"
        job_dir.mkdir()
        dataset = job_dir / "dataset.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        profile = {
            "digest": "b" * 64, "imageDigest": "sha256:" + "c" * 64,
            "conformanceDigest": "d" * 64, "environmentId": "cpu-test-fixture",
            "environmentVersion": "1", "offerId": "cpu-test-fixture",
            "modelId": MODEL_ID, "modelRevision": REVISION, "datasetFormat": "vision",
            "method": "sft", "environmentDigest": runner.checksum(model / "training.py"),
            "gpuCount": 0, "maxSteps": 1, "maxHours": 1,
            "vision": {**lock, "schemaVersion": 1, "mode": "timestamped_frames_v1",
                       "maxFrames": 2 if frame_task else 1, "maxPixels": 200704,
                       "maxSequenceLength": 1024, "task": task},
        }
        config = {
            "schemaVersion": 2, "profileDigest": profile["digest"], "model": {"id": MODEL_ID},
            "environment": {"id": profile["environmentId"], "version": "1"},
            "dataset": {"sha256": runner.checksum(dataset)},
            "optimization": {"maxSteps": 1, "method": "sft"},
            "compute": {"maxHours": 1, "offerId": profile["offerId"], "gpuCount": 0,
                        "providerPolicy": "private"},
        }
        if name == "continued":
            shutil.copyfile(parent, job_dir / "parent.checkpoint")
            config["parent"] = {"sha256": runner.checksum(parent)}
        runner.write_json(job_dir / "job.json", {
            "profile": profile, "configuration": config,
            "deadlineEpoch": time.time() + 900, "finalizationReserveSeconds": 60,
        })
        destination = output / f"{name}-output"
        runner.supervise(job_dir, model, destination)
        report = runner.read_json(destination / "result.json")
        assert report["reloadVerified"] is True and report["steps"] == 1
        assert report["device"] == "cpu" and report["samples"] == 1
        assert report["modelRevision"] == REVISION
        assert 0 <= report["baseline"] <= 1 and 0 <= report["postRun"] <= 1
        predictions = json.loads((destination / "run" / "predictions.json").read_text())
        assert len(predictions) == 1 and predictions[0]["id"] == "test"
        checkpoint = destination / "checkpoint.tar"
        assert runner.checksum(checkpoint) == report["checkpointSha256"]
        with tarfile.open(checkpoint) as archive:
            assert set(archive.getnames()) == runner.ADAPTER_FILES
            adapter = json.load(archive.extractfile("adapter_config.json"))
            assert adapter["base_model_name_or_path"] == MODEL_ID
            assert adapter["revision"] == REVISION
        if name == "continued":
            assert runner.checksum(parent) != report["checkpointSha256"]
        parent = checkpoint
        reports.append({"case": name, "steps": report["steps"], "reloadVerified": True})
    runner.write_json(output / "smoke-result.json", {
        "scope": "Synthetic inputs; CPU execution and adapter portability only.",
        "qualifiedGpu": False, "customerQualityMeasured": False,
        "profileIdentities": "Explicit synthetic test fixtures, not qualified capacity.",
        "cases": reports,
    })
    print(json.dumps({"status": "passed", "cases": reports}))


if __name__ == "__main__":
    main()
