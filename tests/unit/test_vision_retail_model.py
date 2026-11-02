"""Offline CPU smoke: random tiny Qwen3-VL, not pretrained retail quality."""

import base64
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

import pytest
import torch

pytestmark = pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn.t accept a copy keyword:DeprecationWarning"
)
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from reliquary.vision import runner


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_retail_actual_multiframe_lora_delivery(tmp_path, cpu_threads):
    torch.manual_seed(17)
    model = tmp_path / "model"
    weights = model / "weights"
    weights.mkdir(parents=True)
    special = [
        "[UNK]",
        "[PAD]",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    backend = Tokenizer(
        WordLevel({v: i for i, v in enumerate(special)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=special[2:],
    )
    template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }} {% endif %}{% endfor %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=1024,
            max_pixels=1024,
        ),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    processor.save_pretrained(weights)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 8,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 1024,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
            },
            "eos_token_id": 3,
            "pad_token_id": 1,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=6,
        video_token_id=7,
        vision_start_token_id=4,
        vision_end_token_id=5,
    )
    Qwen3VLForConditionalGeneration(config).save_pretrained(weights)
    shutil.copyfile(
        Path(runner.__file__).resolve().parents[2]
        / "scripts/vision-training-runtime.py",
        model / "training.py",
    )
    lock = runner.lock_model(model, "fixture/random-tiny-qwen3-vl", "a" * 40, "a" * 40)
    job = tmp_path / "job"
    job.mkdir()
    rows = []
    for split, colors in [
        ("train", ("red", "blue")),
        ("validation", ("green", "yellow")),
    ]:
        frames = []
        for i, color in enumerate(colors):
            buffer = io.BytesIO()
            Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
            raw = buffer.getvalue()
            frames.append(
                {
                    "data_base64": base64.b64encode(raw).decode(),
                    "sha256": runner.hashlib.sha256(raw).hexdigest(),
                    "width": 32,
                    "height": 32,
                    "pts_us": i * 1_000_000,
                }
            )
        rows.append(
            {
                "id": split,
                "source_id": split,
                "split": split,
                "task": "retail_events",
                "prompt": "Return retail events as JSON.",
                "annotation_complete": True,
                "is_verified_negative": False,
                "target": {
                    "type": "retail_events",
                    "items": [
                        {
                            "label": "zone_entry",
                            "start_us": 0,
                            "end_us": 1_000_000,
                            "confidence": 1,
                        }
                    ],
                },
                "media": {
                    "kind": "frames",
                    "clip_start_us": 0,
                    "duration_us": 2_000_000,
                    "frames": frames,
                },
            }
        )
    dataset = job / "dataset.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    profile = {
        "digest": "b" * 64,
        "imageDigest": "sha256:" + "c" * 64,
        "conformanceDigest": "d" * 64,
        "environmentId": "cpu-fixture",
        "environmentVersion": "1",
        "offerId": "cpu-fixture",
        "modelId": "fixture/random-tiny-qwen3-vl",
        "modelRevision": "a" * 40,
        "datasetFormat": "vision",
        "method": "sft",
        "environmentDigest": runner.checksum(model / "training.py"),
        "gpuCount": 0,
        "maxSteps": 1,
        "maxHours": 1,
        "vision": {
            **lock,
            "schemaVersion": 1,
            "mode": "timestamped_frames_v1",
            "maxFrames": 2,
            "maxPixels": 1024,
            "maxSequenceLength": 1024,
            "task": "retail_events",
        },
    }
    configuration = {
        "schemaVersion": 2,
        "profileDigest": profile["digest"],
        "model": {"id": profile["modelId"]},
        "environment": {"id": "cpu-fixture", "version": "1"},
        "dataset": {"sha256": runner.checksum(dataset)},
        "optimization": {"maxSteps": 1, "method": "sft"},
        "compute": {
            "maxHours": 1,
            "offerId": "cpu-fixture",
            "gpuCount": 0,
            "providerPolicy": "private",
        },
    }
    runner.write_json(
        job / "job.json",
        {
            "profile": profile,
            "configuration": configuration,
            "deadlineEpoch": time.time() + 300,
            "finalizationReserveSeconds": 60,
        },
    )
    output = tmp_path / "output"
    report = runner.execute(job, model, output)
    assert report["steps"] == 1 and report["reloadVerified"] is True
    assert report["device"] == "cpu" and report["vision"]["task"] == "retail_events"
    assert runner.checksum(output / "checkpoint.tar") == report["checkpointSha256"]
    with tarfile.open(output / "checkpoint.tar") as archive:
        from safetensors.torch import load

        tensors = load(archive.extractfile("adapter_model.safetensors").read())
        assert all(torch.isfinite(t).all() for t in tensors.values())
        assert any(
            torch.count_nonzero(t) for name, t in tensors.items() if "lora_B" in name
        )
        binding = json.load(archive.extractfile("binding.json"))
        assert binding["vision"]["task"] == "retail_events"
    assert {row["split"] for row in rows} == {"train", "validation"}

    from reliquary.vision import retail
    from reliquary.vision.retail_inference import infer

    records = []
    for index, row in enumerate([*rows, copy.deepcopy(rows[0])]):
        # Different source pixels keep the heldout station independent.
        if index == 2:
            for frame_index, frame in enumerate(row["media"]["frames"]):
                buffer = io.BytesIO()
                Image.new("RGB", (32, 32), (45 + frame_index, 23, 170)).save(
                    buffer, format="PNG"
                )
                raw = buffer.getvalue()
                frame.update(
                    data_base64=base64.b64encode(raw).decode(),
                    sha256=runner.hashlib.sha256(raw).hexdigest(),
                )
        records.append(
            {
                "id": f"clip{index}",
                "station_id": f"station{index}",
                "camera_id": "camera",
                "day": "2026-11-02",
                "split": ["train", "validation", "test"][index],
                "rights": {
                    "basis": "synthetic",
                    "reference": "generated fixture",
                    "approved": True,
                },
                "annotation_complete": True,
                "source_sha256": str(index) * 64,
                "media": row["media"],
                "events": [
                    {
                        "id": "entry",
                        "label": "zone_entry",
                        "start_us": 0,
                        "end_us": 1_000_000,
                    }
                ],
                "cutoffs_us": [2_000_000],
            }
        )
    recording_manifest = tmp_path / "recordings.private.json"
    recording_manifest.write_text(
        json.dumps(
            {"schema_version": 1, "labels": ["zone_entry"], "recordings": records}
        )
    )
    prepared = tmp_path / "prepared"
    retail.prepare(recording_manifest, prepared, max_frames=2)
    benchmark = prepared / "benchmark.private.json"
    frozen = benchmark.read_bytes()
    for case, archive in [("base", None), ("checkpoint", output / "checkpoint.tar")]:
        destination = tmp_path / (case + "-inference")
        evidence = infer(
            benchmark,
            model,
            destination,
            device="cpu",
            max_pixels=1024,
            max_context_tokens=1024,
            max_new_tokens=128,
            checkpoint=archive,
            checkpoint_sha256=runner.checksum(archive) if archive else None,
        )
        assert evidence["windows"] == 1
        assert evidence["source_sha256"] == runner.checksum(model / "source.json")
        assert evidence["checkpoint_sha256"] == (
            report["checkpointSha256"] if archive else None
        )
        envelope = json.loads((destination / "predictions.private.json").read_text())
        assert len(envelope["predictions"]) == 1
        assert envelope["predictions"][0]["latency_ms"] >= 0
        retail.evaluate(
            benchmark,
            destination / "predictions.private.json",
            tmp_path / (case + "-evaluation"),
        )
        assert benchmark.read_bytes() == frozen
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        infer(
            benchmark,
            model,
            tmp_path / "wrong-checksum",
            device="cpu",
            checkpoint=output / "checkpoint.tar",
            checkpoint_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="different model or visual contract"):
        infer(
            benchmark,
            model,
            tmp_path / "wrong-contract",
            device="cpu",
            max_pixels=1024,
            max_context_tokens=1023,
            checkpoint=output / "checkpoint.tar",
            checkpoint_sha256=report["checkpointSha256"],
        )

    shutil.copyfile(output / "checkpoint.tar", job / "parent.checkpoint")
    configuration["parent"] = {"sha256": report["checkpointSha256"]}
    destination = tmp_path / "parent-check"
    destination.mkdir()
    wrong_profile = copy.deepcopy(profile)
    wrong_profile["vision"]["task"] = "football_events"
    with pytest.raises(ValueError, match="different model or visual contract"):
        runner.load_parent(job, destination, configuration, wrong_profile)
