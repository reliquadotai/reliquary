"""Bounded, prompt-free training measurements and checkpoint download manifests."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping

from reliquary.shared.ckpt_index import (
    INDEX_FORMAT,
    checkpoint_key_version,
    index_key,
    run_prefix,
)

SCHEMA = "reliquary/training-outputs/v1"
MAX_POINTS = 2000
MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 512
_METRICS = frozenset(
    {
        "train/entropy",
        "train/kl",
        "train/kl_term_computed_ratio",
        "train/ppo_loss",
        "train/loss",
        "train/grad_norm",
        "train/lr",
        "train/ppo_clip_active_ratio",
        "train/ppo_ratio_outside_clip_ratio",
        "batch/degenerate_ratio",
        "async/completion_tokens",
        "async/completion_tokens_per_s",
        "async/step_s",
        "async/train_s",
        "async/wait_s",
        "async/lag_mean",
        "async/steps_skipped_total",
        "async/groups_accepted",
        "async/groups_rejected",
        "async/stale_dropped",
        "async/exhausted_dropped",
    }
)
_ENV = re.compile(
    r"train/env/[a-zA-Z0-9_.-]{1,96}/(?:reward_mean|reward_nonzero_ratio|raw_completion_tokens|rollouts)$"
)
_SHA = re.compile(r"[0-9a-f]{64}$")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def metric_values(values: Mapping) -> dict:
    return {
        k: float(v)
        for k, v in values.items()
        if isinstance(k, str) and (k in _METRICS or _ENV.fullmatch(k)) and finite(v)
    }


def metrics_key(run: str) -> str:
    return f"{run_prefix(run)}/metrics.json"


def append_metrics(
    previous, run: str, checkpoint: int, observed_at: float, values: Mapping
) -> dict:
    if (
        isinstance(checkpoint, bool)
        or not isinstance(checkpoint, int)
        or checkpoint < 0
        or not finite(observed_at)
    ):
        raise ValueError("invalid checkpoint measurement")
    rows = {}
    truncated = False
    if previous is not None:
        if previous.get("schema") != SCHEMA or previous.get("task_id") != run:
            raise ValueError("metrics of another task or schema")
        truncated = (
            bool(previous.get("metrics_truncated"))
            or len(previous.get("metrics", [])) > MAX_POINTS
        )
        for row in previous.get("metrics", [])[-MAX_POINTS:]:
            k, at = row["checkpoint"], row["observed_at"]
            if isinstance(k, int) and not isinstance(k, bool) and k >= 0 and finite(at):
                rows[k] = {
                    "checkpoint": k,
                    "observed_at": at,
                    "values": metric_values(row["values"]),
                }
    rows[checkpoint] = {
        "checkpoint": checkpoint,
        "observed_at": observed_at,
        "values": metric_values(values),
    }
    keys = sorted(rows)
    truncated |= len(keys) > MAX_POINTS
    out = {
        "schema": SCHEMA,
        "task_id": run,
        "metrics": [rows[k] for k in keys[-MAX_POINTS:]],
        "metrics_truncated": truncated,
    }
    # Leave room for the checkpoint index in the combined admin response.
    while len(json.dumps(out).encode()) > MAX_BYTES // 2:
        out["metrics"].pop(0)
        out["metrics_truncated"] = True
    return out


def checkpoint_entries(index, run: str) -> tuple[list[dict], bool]:
    if index is None:
        return [], False
    if index.get("format") != INDEX_FORMAT or index.get("run") != run:
        raise ValueError("checkpoint index of another task or format")
    latest = index.get("latest", {})
    version = latest.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 0
        or not _SHA.fullmatch(str(latest.get("sha256")))
    ):
        raise ValueError("invalid latest checkpoint identity")
    prefix = run_prefix(run) + "/"
    by_version = {}
    total_files = 0
    truncated = False
    for kind, entries in (
        ("delta", index.get("deltas", [])),
        ("full", index.get("anchors", [])),
    ):
        for entry in entries:
            version = entry.get("version") if kind == "full" else entry.get("to")
            sha = entry.get("sha256") if kind == "full" else entry.get("target_sha256")
            if (
                not isinstance(version, int)
                or isinstance(version, bool)
                or version < 0
                or not _SHA.fullmatch(str(sha))
            ):
                raise ValueError("invalid checkpoint identity")
            objects = (
                [*entry["files"], *entry.get("model_files", [])]
                if kind == "full"
                else [entry]
            )
            files = []
            for obj in objects:
                key, size, digest = obj.get("key"), obj.get("size"), obj.get("sha256")
                name = key.rsplit("/", 1)[-1] if isinstance(key, str) else ""
                if (
                    not isinstance(key, str)
                    or not key.startswith(prefix)
                    or checkpoint_key_version(run, key) != version
                    or ".." in key.split("/")
                    or not _NAME.fullmatch(name)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 0
                    or not _SHA.fullmatch(str(digest))
                ):
                    raise ValueError("invalid checkpoint file")
                files.append({"name": name, "key": key, "size": size, "sha256": digest})
            names = {f["name"] for f in files}
            if len(names) != len(files):
                raise ValueError("duplicate checkpoint filename")
            loadable = (
                kind == "full"
                and any(name.endswith(".safetensors") for name in names)
                and "config.json" in names
                and "model.safetensors.index.json" in names
                and bool(names & {"tokenizer.json", "tokenizer.model", "vocab.json"})
            )
            # A cumulative delta may be shorter than a per-step delta for the same target.
            current = by_version.get(version)
            if current and (
                kind != "full"
                and sum(f["size"] for f in current["files"])
                <= sum(f["size"] for f in files)
            ):
                continue
            at = entry.get("published_at")
            by_version[version] = {
                "version": version,
                "published_at": at if finite(at) else None,
                "sha256": sha,
                "format": "full" if loadable else "delta",
                "files": files,
            }
    result = []
    for version in sorted(by_version, reverse=True):
        entry = by_version[version]
        if total_files + len(entry["files"]) > MAX_FILES:
            truncated = True
            continue
        result.append(entry)
        total_files += len(entry["files"])
    return sorted(result, key=lambda e: e["version"]), truncated


class TrainingOutputsStore:
    """Subnet-owned reads and short-lived URLs; no R2 credential crosses the admin boundary."""

    def __init__(self, client=None, bucket=None):
        if client is None:
            from reliquary.infrastructure.storage import (
                public_checkpoint_bucket,
                public_checkpoint_client,
            )

            client, bucket = public_checkpoint_client(), public_checkpoint_bucket()
        self.client, self.bucket = client, bucket

    def read(self, key):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        stream = response["Body"]
        try:
            raw = stream.read(MAX_BYTES + 1)
        finally:
            stream.close()
        if len(raw) > MAX_BYTES:
            raise ValueError("training output document too large")
        return json.loads(raw)

    def outputs(self, task_id):
        # Checkpoints are committed before metrics. Reading metrics first avoids a future row
        # paired with an older index when an optimizer step publishes between these reads.
        metrics, index = self.read(metrics_key(task_id)), self.read(index_key(task_id))
        clean = None
        if metrics is not None:
            if metrics.get("schema") != SCHEMA or metrics.get("task_id") != task_id:
                raise ValueError("metrics of another task")
            source = metrics.get("metrics", [])
            if not isinstance(source, list):
                raise ValueError("invalid metrics history")
            clean = {
                "metrics": [],
                "metrics_truncated": bool(metrics.get("metrics_truncated"))
                or len(source) > MAX_POINTS,
            }
            for row in source[-MAX_POINTS:]:
                checkpoint, at = row["checkpoint"], row["observed_at"]
                if (
                    isinstance(checkpoint, bool)
                    or not isinstance(checkpoint, int)
                    or checkpoint < 0
                    or not finite(at)
                ):
                    raise ValueError("invalid checkpoint measurement")
                clean["metrics"].append(
                    {
                        "checkpoint": checkpoint,
                        "observed_at": at,
                        "values": metric_values(row["values"]),
                    }
                )
        rows = [] if clean is None else clean["metrics"]
        checkpoints, truncated = checkpoint_entries(index, task_id)
        latest = None if index is None else index.get("latest", {}).get("version")
        times = [r["observed_at"] for r in rows]
        if index is not None and finite(index.get("published_at")):
            times.append(index["published_at"])
        result = {
            "schema": SCHEMA,
            "task_id": task_id,
            "observed_at": max(times) if times else None,
            "latest_checkpoint": latest,
            "metrics": rows,
            "checkpoints": checkpoints,
            "metrics_truncated": bool(clean and clean["metrics_truncated"]),
            "checkpoints_truncated": truncated,
        }
        if len(json.dumps(result).encode()) > MAX_BYTES:
            raise ValueError("training outputs too large")
        return result

    def download(self, task_id, version, filename, *, now):
        if not _NAME.fullmatch(filename):
            raise KeyError(filename)
        entries, _ = checkpoint_entries(self.read(index_key(task_id)), task_id)
        for checkpoint in entries:
            if checkpoint["version"] == version:
                for file in checkpoint["files"]:
                    if file["name"] == filename:
                        url = self.client.generate_presigned_url(
                            "get_object",
                            Params={
                                "Bucket": self.bucket,
                                "Key": file["key"],
                                "ResponseContentDisposition": f'attachment; filename="{filename}"',
                            },
                            ExpiresIn=900,
                        )
                        return {
                            "task_id": task_id,
                            "checkpoint": version,
                            "filename": filename,
                            "size": file["size"],
                            "sha256": file["sha256"],
                            "url": url,
                            "expires_at": int(now) + 900,
                        }
        raise KeyError(filename)
