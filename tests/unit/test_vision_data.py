from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from reliquary.vision.data import (
    MAX_INPUT_BYTES,
    load_manifest,
    load_verified_media_bytes,
    prepare_jsonl,
    validate_manifest,
)


def png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    def chunk(kind: bytes, content: bytes) -> bytes:
        return (struct.pack(">I", len(content)) + kind + content
                + struct.pack(">I", zlib.crc32(kind + content)))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * 2) * 2))
            + chunk(b"IEND", b""))


def example(*, football: bool = True, inline: bool = False) -> dict:
    image = png_bytes()
    frame = {"sha256": hashlib.sha256(image).hexdigest(), "width": 2, "height": 2, "pts_us": 0}
    frame.update({"data_base64": base64.b64encode(image).decode()} if inline else {"path": "a.png"})
    return {
        "id": "clip-1", "source_id": "match-1", "split": "train",
        "task": "football_events" if football else "image_grounding",
        "prompt": "Return the requested annotations as JSON.",
        "annotation_complete": True, "is_verified_negative": False,
        "target": ({"type": "soccer_action", "items": [
            {"frame": 25, "action": "pass", "confidence": 1.0},
        ]} if football else {"type": "grounding", "items": [
            {"label": "ball", "box": [0, 0, 1000, 1000]},
        ]}),
        "media": {
            "kind": "frames" if football else "image", "frames": [frame],
            "clip_start_us": 0, "duration_us": 2_000_000 if football else 0,
            **({"score_fps": 25} if football else {}),
        },
    }


def manifest(row: dict | None = None) -> dict:
    return {"schema_version": 1, "examples": [row if row is not None else example()]}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("football", [True, False])
def test_inline_png_materializes_and_reloads_idempotently(tmp_path: Path, football: bool):
    source = write_jsonl(tmp_path / "input.jsonl", [example(football=football, inline=True)])
    target = prepare_jsonl(source, tmp_path / "dataset")
    loaded = load_manifest(target)
    assert loaded["examples"][0]["task"] == ("football_events" if football else "image_grounding")
    assert load_verified_media_bytes(loaded["examples"][0], target.parent) == [png_bytes()]
    assert prepare_jsonl(source, target.parent) == target
    assert "data_base64" not in target.read_text()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("examples", 0, "annotation_complete"), False),
        (("examples", 0, "is_verified_negative"), 1),
        (("examples", 0, "media", "score_fps"), 30),
        (("examples", 0, "media", "score_fps"), True),
        (("examples", 0, "media", "duration_us"), 60_000_001),
        (("examples", 0, "media", "clip_start_us"), -1),
        (("examples", 0, "media", "frames", 0, "pts_us"), 2_000_000),
        (("examples", 0, "media", "frames", 0, "pts_us"), 0.5),
        (("examples", 0, "media", "frames", 0, "width"), 2049),
        (("examples", 0, "media", "frames", 0, "width"), True),
        (("examples", 0, "media", "frames", 0, "sha256"), "A" * 64),
        (("examples", 0, "media", "frames", 0, "path"), "../a.png"),
        (("examples", 0, "media", "frames", 0, "path"), "/a.png"),
        (("examples", 0, "media", "frames", 0, "path"), "a//b.png"),
        (("examples", 0, "media", "frames", 0, "path"), "a/./b.png"),
        (("examples", 0, "media", "frames", 0, "path"), "https://site/a.png"),
        (("examples", 0, "media", "frames", 0, "path"), "a\\b.png"),
        (("examples", 0, "target", "items", 0, "frame"), 50),
        (("examples", 0, "target", "items", 0, "frame"), True),
        (("examples", 0, "target", "items", 0, "confidence"), float("nan")),
        (("examples", 0, "target", "items", 0, "confidence"), float("inf")),
        (("examples", 0, "target", "items", 0, "confidence"), 10**1000),
        (("examples", 0, "target", "items", 0, "action"), "unknown"),
    ],
)
def test_rejects_noncanonical_contract(path, value):
    data = manifest()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        validate_manifest(data)


def test_rejects_unknown_fields_and_duplicate_ids():
    data = manifest()
    data["examples"][0]["media"]["url"] = "https://example.com/video.mp4"
    with pytest.raises(ValueError, match="exactly"):
        validate_manifest(data)
    row = example()
    with pytest.raises(ValueError, match="duplicate example"):
        validate_manifest({"schema_version": 1, "examples": [row, copy.deepcopy(row)]})


def test_source_and_identical_media_cannot_cross_splits():
    train, validation = example(), example()
    validation.update(id="clip-2", split="validation")
    with pytest.raises(ValueError, match="source_id crosses"):
        validate_manifest({"schema_version": 1, "examples": [train, validation]})
    validation["source_id"] = "match-2"
    with pytest.raises(ValueError, match="identical media"):
        validate_manifest({"schema_version": 1, "examples": [train, validation]})
    validation["media"]["frames"][0]["sha256"] = "a" * 64
    validate_manifest({"schema_version": 1, "examples": [train, validation]})


def test_timestamps_must_increase_and_pixel_budget_is_bounded():
    row = example()
    row["media"]["frames"] *= 2
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_manifest(manifest(row))
    row["media"]["frames"] = [
        dict(example()["media"]["frames"][0], pts_us=i, width=2048, height=2048)
        for i in range(16)
    ]
    with pytest.raises(ValueError, match="pixel bound"):
        validate_manifest(manifest(row))


def test_empty_labels_require_verified_negatives():
    row = example()
    row["target"]["items"] = []
    with pytest.raises(ValueError, match="verified negative"):
        validate_manifest(manifest(row))
    row["is_verified_negative"] = True
    validate_manifest(manifest(row))


def test_json_integral_numbers_are_canonicalized_without_accepting_booleans():
    row = example()
    row["media"]["duration_us"] = 2_000_000.0
    row["media"]["frames"][0]["width"] = 2.0
    row["target"]["items"][0]["frame"] = 25.0
    validated = validate_manifest({"schema_version": 1.0, "examples": [row]})
    assert type(validated["schema_version"]) is int
    assert type(row["media"]["duration_us"]) is int
    assert type(row["media"]["frames"][0]["width"]) is int
    assert type(row["target"]["items"][0]["frame"]) is int


def test_changed_bytes_and_symlink_are_rejected(tmp_path: Path):
    (tmp_path / "a.png").write_bytes(png_bytes())
    data = manifest()
    validate_manifest(data, tmp_path, verify_files=True)
    (tmp_path / "a.png").write_bytes(png_bytes((0, 0, 255)))
    with pytest.raises(ValueError, match="SHA256"):
        load_verified_media_bytes(data["examples"][0], tmp_path)
    (tmp_path / "a.png").unlink()
    (tmp_path / "target.png").write_bytes(png_bytes())
    (tmp_path / "a.png").symlink_to(tmp_path / "target.png")
    with pytest.raises(ValueError, match="without symlinks"):
        load_verified_media_bytes(data["examples"][0], tmp_path)


@pytest.mark.parametrize("mutator", [
    lambda value: value[:-8],
    lambda value: value + b"unexpected",
    lambda value: value[:40] + bytes([value[40] ^ 1]) + value[41:],
])
def test_bad_png_is_rejected_even_with_matching_digest(tmp_path: Path, mutator):
    row = example(inline=True)
    content = mutator(png_bytes())
    row["media"]["frames"][0].update(
        data_base64=base64.b64encode(content).decode(),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    source = write_jsonl(tmp_path / "input.jsonl", [row])
    with pytest.raises(ValueError):
        prepare_jsonl(source, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_declared_dimensions_checked_and_existing_dataset_not_overwritten(tmp_path: Path):
    row = example(inline=True)
    source = write_jsonl(tmp_path / "input.jsonl", [row])
    output = prepare_jsonl(source, tmp_path / "dataset")
    original = output.read_bytes()
    row["media"]["frames"][0]["width"] = 3
    write_jsonl(source, [row])
    with pytest.raises(ValueError, match="declared size"):
        prepare_jsonl(source, output.parent)
    assert output.read_bytes() == original
    row["media"]["frames"][0]["width"] = 2
    row["prompt"] = "Changed prompt"
    write_jsonl(source, [row])
    with pytest.raises(ValueError, match="different dataset"):
        prepare_jsonl(source, output.parent)
    assert output.read_bytes() == original


def test_rejects_oversized_duplicate_json_keys_and_blank_rows(tmp_path: Path):
    source = tmp_path / "input.jsonl"
    source.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="5 MiB"):
        prepare_jsonl(source, tmp_path / "dataset")
    source.write_text('{"media":{},"media":{}}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare_jsonl(source, tmp_path / "dataset")
    source.write_text("\n")
    with pytest.raises(ValueError, match="blank"):
        prepare_jsonl(source, tmp_path / "dataset")
