"""Small, fully reviewed VLM datasets with immutable local PNG frames."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import zlib

from reliquary.shared.strict_json import strict_json_loads
from reliquary.vision.rewards import validate_target

MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_TOTAL_PIXELS = 64_000_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    return value


def _integer(value: object, lower: int, upper: int, name: str) -> int:
    if (type(value) not in (int, float) or not lower <= value <= upper
            or int(value) != value):
        raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
    return int(value)


def _relative_path(value: object) -> PurePosixPath:
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or "\\" in value or "\x00" in value):
        raise ValueError("frame path must be a canonical relative PNG path")
    path = PurePosixPath(value)
    if (path.is_absolute() or value != str(path) or ".." in path.parts
            or path.suffix != ".png"):
        raise ValueError("frame path must be a canonical relative PNG path")
    return path


def resolve_media_path(root: str | Path, relative_path: str) -> Path:
    """Resolve only regular files under the supplied root, without symlinks."""
    path = Path(root).resolve()
    for part in _relative_path(relative_path).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("media symlinks are not allowed")
    return path


def _read_frame(root: str | Path, frame: dict) -> bytes:
    # Traverse directory descriptors so replacing a path with a symlink cannot
    # redirect a verified read outside the dataset root.
    parts = _relative_path(frame["path"]).parts
    descriptor = os.open(Path(root).resolve(), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        child = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=descriptor)
        with os.fdopen(child, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_INPUT_BYTES:
                raise ValueError("frame must be a bounded regular file")
            content = stream.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ValueError("frame must be an accessible local file without symlinks") from exc
    finally:
        os.close(descriptor)
    if len(content) > MAX_INPUT_BYTES:
        raise ValueError("frame exceeds byte bound")
    if hashlib.sha256(content).hexdigest() != frame["sha256"]:
        raise ValueError("frame SHA256 mismatch")
    return content


def _verify_png(content: bytes, frame: dict) -> None:
    from PIL import Image

    _integer(frame["width"], 1, 2048, "width")
    _integer(frame["height"], 1, 2048, "height")
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("frame must contain PNG bytes")
    offset, compressed, chunks = 8, bytearray(), []
    while offset < len(content):
        if len(content) - offset < 12:
            raise ValueError("truncated PNG chunk")
        length, kind = struct.unpack(">I4s", content[offset:offset + 8])
        end = offset + 12 + length
        if end > len(content):
            raise ValueError("truncated PNG chunk")
        body = content[offset + 8:end - 4]
        if zlib.crc32(kind + body) != struct.unpack(">I", content[end - 4:end])[0]:
            raise ValueError("PNG checksum mismatch")
        if not chunks:
            if kind != b"IHDR" or length != 13:
                raise ValueError("PNG requires an initial IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body)
            if ((width, height) != (frame["width"], frame["height"]) or depth != 8
                    or color not in (2, 6) or compression or filtering or interlace):
                raise ValueError("PNG must be RGB/RGBA 8-bit, noninterlaced, at declared size")
            channels = 3 if color == 2 else 4
        elif kind not in (b"IDAT", b"IEND"):
            if kind in (b"IHDR", b"acTL", b"fcTL", b"fdAT") or not re.fullmatch(
                    b"[a-z][A-Za-z]{3}", kind):
                raise ValueError("PNG must contain one static RGB/RGBA image")
        if kind == b"IDAT":
            if b"IDAT" in chunks and chunks[-1] != b"IDAT":
                raise ValueError("PNG IDAT chunks must be consecutive")
            compressed.extend(body)
        if kind == b"IEND":
            if length or end != len(content):
                raise ValueError("PNG must end immediately after IEND")
        chunks.append(kind)
        offset = end
    if not chunks or chunks[-1] != b"IEND" or not compressed:
        raise ValueError("incomplete PNG")
    expected_size = frame["height"] * (1 + frame["width"] * channels)
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(compressed, expected_size + 1)
    except zlib.error as exc:
        raise ValueError("invalid PNG compressed pixels") from exc
    if (len(pixels) != expected_size or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("PNG decompressed pixels do not match dimensions")
    if any(pixels[i] > 4 for i in range(0, len(pixels), 1 + frame["width"] * channels)):
        raise ValueError("invalid PNG scanline filter")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if (image.format != "PNG" or image.size != (frame["width"], frame["height"])
                    or getattr(image, "is_animated", False)):
                raise ValueError("frame must be a static PNG matching declared dimensions")
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("frame is not a complete PNG") from exc


def load_verified_media_bytes(example: dict, root: str | Path) -> list[bytes]:
    """Verify immediately before decoding; manifest validation alone is not a lease."""
    contents = []
    for frame in example["media"]["frames"]:
        content = _read_frame(root, frame)
        _verify_png(content, frame)
        contents.append(content)
    return contents


def validate_manifest(
    manifest: object, media_root: str | Path | None = None, *, verify_files: bool = False,
) -> dict:
    """Validate strict schema, time bases, reviewed targets and split isolation."""
    _keys(manifest, {"schema_version", "examples"}, "manifest")
    manifest["schema_version"] = _integer(manifest["schema_version"], 1, 1, "schema_version")
    examples = manifest["examples"]
    if not isinstance(examples, list) or not 1 <= len(examples) <= 256:
        raise ValueError("manifest requires between 1 and 256 examples")
    identifiers, sources, hashes = set(), {}, {}
    total_pixels = 0
    for example in examples:
        _keys(example, {"id", "source_id", "split", "task", "prompt",
                        "annotation_complete", "is_verified_negative", "target", "media"},
              "example")
        for name in ("id", "source_id"):
            if not isinstance(example[name], str) or not _IDENTIFIER.fullmatch(example[name]):
                raise ValueError(f"{name} must be a canonical identifier")
        if example["id"] in identifiers:
            raise ValueError("duplicate example id")
        identifiers.add(example["id"])
        split = example["split"]
        if not isinstance(split, str) or split not in {"train", "validation", "test"}:
            raise ValueError("unknown dataset split")
        if sources.setdefault(example["source_id"], split) != split:
            raise ValueError("source_id crosses dataset splits")
        if example["task"] not in ("image_grounding", "football_events"):
            raise ValueError("unknown vision task")
        prompt = example["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise ValueError("prompt must contain 1 to 8192 characters")
        if example["annotation_complete"] is not True:
            raise ValueError("partial or unreviewed annotations are not trainable")
        if type(example["is_verified_negative"]) is not bool:
            raise ValueError("is_verified_negative must be a boolean")
        football = example["task"] == "football_events"
        media_keys = {"kind", "frames", "clip_start_us", "duration_us"}
        if football:
            media_keys.add("score_fps")
        media = _keys(example["media"], media_keys, "media")
        if media["kind"] != ("frames" if football else "image"):
            raise ValueError("media kind does not match task")
        media["clip_start_us"] = _integer(
            media["clip_start_us"], 0, 86_400_000_000 if football else 0, "clip_start_us")
        media["duration_us"] = _integer(
            media["duration_us"], 1 if football else 0, 60_000_000 if football else 0, "duration_us")
        if football:
            media["score_fps"] = _integer(media["score_fps"], 25, 25, "score_fps")
        frames = media["frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= (64 if football else 1):
            raise ValueError("invalid number of frames")
        previous_pts = -1
        for frame in frames:
            _keys(frame, {"path", "sha256", "width", "height", "pts_us"}, "frame")
            _relative_path(frame["path"])
            if not isinstance(frame["sha256"], str) or not _SHA256.fullmatch(frame["sha256"]):
                raise ValueError("frame sha256 must be lowercase hexadecimal")
            if hashes.setdefault(frame["sha256"], split) != split:
                raise ValueError("identical media crosses dataset splits")
            width = frame["width"] = _integer(frame["width"], 1, 2048, "width")
            height = frame["height"] = _integer(frame["height"], 1, 2048, "height")
            total_pixels += width * height
            if total_pixels > MAX_TOTAL_PIXELS:
                raise ValueError("dataset exceeds decoded pixel bound")
            pts = frame["pts_us"] = _integer(
                frame["pts_us"], 0, media["duration_us"] - 1 if football else 0, "pts_us")
            if pts <= previous_pts:
                raise ValueError("frame timestamps must be strictly increasing")
            previous_pts = pts
        target = validate_target(example["target"], example["task"], media["duration_us"])
        if example["is_verified_negative"] != (not target["items"]):
            raise ValueError("empty labels require a verified negative, and vice versa")
    if verify_files:
        if media_root is None:
            raise ValueError("media_root is required to verify files")
        for example in examples:
            load_verified_media_bytes(example, media_root)
    return manifest


def _read_bounded(path: str | Path) -> bytes:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("dataset exceeds 5 MiB")
    return raw


def load_manifest(path: str | Path, media_root: str | Path | None = None) -> dict:
    path = Path(path)
    return validate_manifest(strict_json_loads(_read_bounded(path)),
                             media_root if media_root is not None else path.parent,
                             verify_files=True)


def prepare_jsonl(path: str | Path, output_root: str | Path) -> Path:
    """Materialize bounded inline PNG rows for the network-free training worker."""
    rows, contents = [], {}
    for line in _read_bounded(path).splitlines():
        if not line.strip():
            raise ValueError("dataset cannot contain blank JSONL rows")
        if len(rows) >= 256:
            raise ValueError("dataset exceeds 256 rows")
        row = strict_json_loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("media"), dict):
            raise ValueError("invalid inline example")
        frames = row["media"].get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 64:
            raise ValueError("invalid inline frames")
        for frame in frames:
            _keys(frame, {"data_base64", "sha256", "width", "height", "pts_us"}, "inline frame")
            encoded = frame.pop("data_base64")
            if not isinstance(encoded, str):
                raise ValueError("data_base64 must be a base64 string")
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid PNG base64") from exc
            if base64.b64encode(content).decode("ascii") != encoded:
                raise ValueError("data_base64 must use canonical base64")
            digest = hashlib.sha256(content).hexdigest()
            if digest != frame["sha256"]:
                raise ValueError("inline frame SHA256 mismatch")
            frame["path"] = f"frames/{digest}.png"
            contents[frame["path"]] = content
        rows.append(row)
    manifest = validate_manifest({"schema_version": 1, "examples": rows})
    for row in rows:
        for frame in row["media"]["frames"]:
            _verify_png(contents[frame["path"]], frame)
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
               + "\n").encode()
    output_root = Path(output_root)
    if output_root.is_symlink():
        raise ValueError("output directory cannot be a symlink")
    if output_root.exists():
        existing = load_manifest(output_root / "manifest.json")
        if existing != manifest:
            raise ValueError("output directory already contains a different dataset")
        return output_root / "manifest.json"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vision-dataset-", dir=output_root.parent) as temp:
        staging = Path(temp)
        (staging / "frames").mkdir()
        for relative, content in contents.items():
            (staging / relative).write_bytes(content)
        (staging / "manifest.json").write_bytes(payload)
        staging.rename(output_root)
    return output_root / "manifest.json"
