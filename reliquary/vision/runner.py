"""Offline platform adapter: bind inputs, enforce bounds, deliver a portable LoRA."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.shared.strict_json import strict_json_loads

SHA256 = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
PROCESSOR_SUFFIXES = {".json", ".txt", ".jinja", ".model"}
ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors", "binding.json"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, limit: int = 65536) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Expected a bounded regular JSON file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON byte limit exceeded")
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
    os.replace(pending, path)


def _integer(value, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, low: float, high: float, name: str) -> float:
    if (type(value) not in {int, float} or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} is outside its finite bounds")
    return float(value)


def _digest(value, pattern, name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be an immutable digest")
    return value


def snapshot_files(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Model snapshot must be a regular local directory")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Model snapshot cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in PROCESSOR_SUFFIXES | {".safetensors", ".md"}:
            raise ValueError("Model snapshot contains an unsupported file")
        files[path.relative_to(directory).as_posix()] = checksum(path)
    if (not files or len(files) > 1024 or "config.json" not in files
            or not any(name.endswith(".safetensors") for name in files)):
        raise ValueError("Model snapshot requires config and safetensors weights")
    return files


def lock_model(directory: Path, model_id: str, revision: str,
               processor_revision: str) -> dict:
    """Lock an already downloaded snapshot; this does not certify its provenance."""
    _digest(revision, REVISION, "model revision")
    _digest(processor_revision, REVISION, "processor revision")
    if not isinstance(model_id, str) or re.fullmatch(r"[\w.-]+/[\w.-]+", model_id) is None:
        raise ValueError("Expected a portable organization/model identifier")
    files = snapshot_files(directory / "weights")
    processor = {name: digest for name, digest in files.items()
                 if PurePosixPath(name).suffix in PROCESSOR_SUFFIXES}
    value = {"modelId": model_id, "revision": revision, "files": files,
             "processorRevision": processor_revision, "processorFiles": processor}
    write_json(directory / "source.json", value)
    return {"processorDigest": canonical_sha256(processor),
            "processorRevision": processor_revision}


def validate_job(job: dict, now: float | None = None) -> dict:
    profile, config = job["profile"], job["configuration"]
    if profile["datasetFormat"] != "vision" or profile["method"] not in {"sft", "grpo"}:
        raise ValueError("This adapter requires a vision SFT or GRPO profile")
    _digest(profile["modelRevision"], REVISION, "model revision")
    _digest(profile["environmentDigest"], SHA256, "environment digest")
    _digest(profile["digest"], SHA256, "profile digest")
    _digest(profile["conformanceDigest"], SHA256, "conformance digest")
    _digest(profile["imageDigest"], re.compile(r"sha256:[0-9a-f]{64}"), "image digest")
    if (config.get("schemaVersion") != 2 or config.get("profileDigest") != profile["digest"]
            or config["model"]["id"] != profile["modelId"]
            or config["environment"] != {"id": profile["environmentId"], "version": profile["environmentVersion"]}
            or config["optimization"]["method"] != profile["method"]
            or config["compute"]["offerId"] != profile["offerId"]
            or type(config["compute"]["gpuCount"]) is not int
            or config["compute"]["gpuCount"] != profile["gpuCount"]
            or config["compute"]["providerPolicy"] != "private"):
        raise ValueError("Configuration differs from its approved execution profile")
    vision = profile["vision"]
    if set(vision) != {"schemaVersion", "processorRevision", "processorDigest", "mode",
                       "maxFrames", "maxPixels", "maxSequenceLength", "task"}:
        raise ValueError("Unexpected vision profile fields")
    if type(vision["schemaVersion"]) is not int or vision["schemaVersion"] != 1:
        raise ValueError("Unsupported vision contract version")
    if vision["mode"] != "timestamped_frames_v1" or vision["task"] not in {
        "football_events", "image_grounding",
    }:
        raise ValueError("Unsupported vision task or processor mode")
    _digest(vision["processorRevision"], REVISION, "processor revision")
    _digest(vision["processorDigest"], SHA256, "processor digest")
    _integer(vision["maxFrames"], 1, 64, "frame limit")
    _integer(vision["maxPixels"], 1, 4194304, "pixel limit")
    _integer(vision["maxSequenceLength"], 32, 32768, "context limit")
    gpus = _integer(profile["gpuCount"], 0, 1, "GPU count")
    steps = _integer(config["optimization"]["maxSteps"], 1, 10000, "step limit")
    if steps > _integer(profile["maxSteps"], 1, 10000, "profile step limit"):
        raise ValueError("Requested steps exceed the qualified profile")
    hours = _number(config["compute"]["maxHours"], 1 / 3600, 168, "allocation hours")
    if hours > _number(profile["maxHours"], 1 / 3600, 168, "profile hours"):
        raise ValueError("Requested duration exceeds the qualified profile")
    reserve = _integer(job["finalizationReserveSeconds"], 60, 3600, "finalization reserve")
    now = time.time() if now is None else now
    deadline = _number(job["deadlineEpoch"], now + reserve + 1, now + hours * 3600 + 5,
                       "allocation deadline")
    _digest(config["dataset"]["sha256"], SHA256, "dataset digest")
    return {
        "method": profile["method"] + "_lora", "max_steps": steps,
        "max_seconds": deadline - now - reserve, "max_context_tokens": vision["maxSequenceLength"],
        "max_new_tokens": min(128, vision["maxSequenceLength"] // 2),
        "group_size": 4, "lora_rank": 8, "learning_rate": 1e-4, "seed": 17,
        "device": "cuda" if gpus else "cpu", "precision": "bf16" if gpus else "float32",
        "max_pixels": vision["maxPixels"], "max_eval_examples": 256,
        "processor_mode": vision["mode"], "model_id": profile["modelId"],
        "model_revision": profile["modelRevision"],
    }


def verify_model(directory: Path, profile: dict) -> Path:
    source = read_json(directory / "source.json", 256 * 1024)
    if (source["modelId"] != profile["modelId"] or source["revision"] != profile["modelRevision"]
            or source["processorRevision"] != profile["vision"]["processorRevision"]):
        raise ValueError("Cached model or processor revision differs from the approved profile")
    files = snapshot_files(directory / "weights")
    processor = {name: digest for name, digest in files.items()
                 if PurePosixPath(name).suffix in PROCESSOR_SUFFIXES}
    if (files != source["files"] or processor != source["processorFiles"]
            or canonical_sha256(processor) != profile["vision"]["processorDigest"]):
        raise ValueError("Cached model or processor integrity check failed")
    entrypoint = directory / "training.py"
    if entrypoint.is_symlink() or not entrypoint.is_file() or checksum(entrypoint) != profile["environmentDigest"]:
        raise ValueError("Runtime entrypoint differs from the approved environment")
    return directory / "weights"


def checkpoint_binding(profile: dict) -> dict:
    return {"schemaVersion": 1, "modelId": profile["modelId"],
            "modelRevision": profile["modelRevision"], "vision": profile["vision"]}


def load_parent(job_dir: Path, output: Path, config: dict, profile: dict) -> Path | None:
    parent = job_dir / "parent.checkpoint"
    if not config.get("parent"):
        if parent.exists():
            raise ValueError("Unexpected parent checkpoint")
        return None
    expected = _digest(config["parent"]["sha256"], SHA256, "parent digest")
    if (parent.is_symlink() or not parent.is_file() or parent.stat().st_size > 1024**3 + 65536
            or checksum(parent) != expected):
        raise ValueError("Parent checkpoint integrity check failed")
    destination = output / "parent"
    destination.mkdir()
    with tarfile.open(parent, mode="r:") as archive:
        entries = archive.getmembers()
        if (len(entries) != len(ADAPTER_FILES) or {entry.name for entry in entries} != ADAPTER_FILES
                or any(not entry.isfile() or entry.size < 0 for entry in entries)
                or sum(entry.size for entry in entries) > 1024**3):
            raise ValueError("Parent checkpoint must contain only bounded adapter files")
        for entry in entries:
            with archive.extractfile(entry) as source, (destination / entry.name).open("xb") as target:
                shutil.copyfileobj(source, target)
    if read_json(destination / "binding.json") != checkpoint_binding(profile):
        raise ValueError("Parent checkpoint belongs to a different model or visual contract")
    adapter = read_json(destination / "adapter_config.json")
    if (adapter.get("peft_type") != "LORA" or adapter.get("base_model_name_or_path") != profile["modelId"]
            or adapter.get("revision") != profile["modelRevision"]):
        raise ValueError("Parent is not a portable LoRA for the approved base model")
    return destination


def execute(job_dir: Path, model_dir: Path, output: Path) -> dict:
    from reliquary.vision.data import load_manifest, prepare_jsonl
    from reliquary.vision.runtime import run_job

    job = read_json(job_dir / "job.json")
    request = validate_job(job)
    profile, config = job["profile"], job["configuration"]
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("Execution requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    weights = verify_model(model_dir, profile)
    dataset_file = job_dir / "dataset.jsonl"
    if dataset_file.is_symlink() or not dataset_file.is_file() or checksum(dataset_file) != config["dataset"]["sha256"]:
        raise ValueError("Dataset integrity check failed")
    manifest_path = prepare_jsonl(dataset_file, output / "input")
    dataset = load_manifest(manifest_path)
    for example in dataset["examples"]:
        if (example["task"] != profile["vision"]["task"]
                or len(example["media"]["frames"]) > profile["vision"]["maxFrames"]
                or any(frame["width"] * frame["height"] > request["max_pixels"]
                       for frame in example["media"]["frames"])):
            raise ValueError("Dataset exceeds the approved visual task bounds")
    adapter = load_parent(job_dir, output, config, profile)
    compute_deadline = job["deadlineEpoch"] - job["finalizationReserveSeconds"]

    def check_deadline():
        if time.time() >= compute_deadline:
            raise TimeoutError("Training/evaluation exceeded its execution deadline")

    result = run_job(request, dataset, manifest_path.parent, weights, output / "run",
                     adapter, check_deadline=check_deadline)
    if result["reloadVerified"] is not True or not 1 <= result["steps"] <= request["max_steps"]:
        raise ValueError("Training must complete an update and pass adapter reload verification")
    saved = output / "run" / "adapter"
    write_json(saved / "binding.json", checkpoint_binding(profile))
    checkpoint = output / "checkpoint.tar"
    with tarfile.open(checkpoint, "x") as archive:
        for name in sorted(ADAPTER_FILES):
            path = saved / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing portable adapter file")
            info = archive.gettarinfo(path, arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with path.open("rb") as contents:
                archive.addfile(info, contents)
    report = {key: result[key] for key in (
        "baseline", "postRun", "samples", "evaluationSetDigest", "steps", "device", "metric",
    )}
    for key in ("baseline", "postRun"):
        _number(report[key], 0, 1, key)
    _integer(report["samples"], 1, 256, "evaluation samples")
    _digest(report["evaluationSetDigest"], SHA256, "evaluation set digest")
    report.update(checkpoint=checkpoint.name, checkpointSha256=checksum(checkpoint),
                  reloadVerified=True, modelRevision=profile["modelRevision"],
                  environmentDigest=profile["environmentDigest"], vision=profile["vision"])
    if time.time() >= job["deadlineEpoch"]:
        raise TimeoutError("Checkpoint finalization exceeded the allocation deadline")
    write_json(output / "progress.json", {"step": result["steps"]})
    write_json(output / "result.json", report)
    return report


def supervise(job_dir: Path, model_dir: Path, output: Path) -> None:
    """Hard process-group deadline, including model load, backward and finalization."""
    job = read_json(job_dir / "job.json")
    validate_job(job)
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    process = subprocess.Popen(
        [sys.executable, "-m", "reliquary.vision", "_execute", "--job", str(job_dir),
         "--model", str(model_dir), "--output", str(output)],
        start_new_session=True, env=environment,
    )
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        code = process.wait(timeout=max(0, job["deadlineEpoch"] - time.time()))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    if code:
        raise RuntimeError(f"Vision worker exited with status {code}")
