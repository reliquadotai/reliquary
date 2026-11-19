"""Publish final usage and owned checkpoint bytes only after a stopped runtime is verified."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.rl_orders.checkpoint_delivery import FREEZE_SECONDS, R2CheckpointSource, TrainingCheckpointStore
from reliquary.rl_orders.supervisor import FinalizedRuntime, StoppedRuntime, _private_bytes, read_runtime_view
from reliquary.shared.strict_json import strict_json_loads
from reliquary.validator.rl_training_usage import (
    finalize_usage, stopped_document, validate_usage_envelope, write_final_usage,
)

COMPLETION_SCHEMA = "reliquary/training-completion/v1"
MAX_RECEIPT_BYTES = 64 * 1024


class FinalityUnavailable(ValueError):
    pass


def _immutable_json(path: Path, document: dict) -> None:
    raw = canonical_json_bytes(document)
    descriptor, temporary = tempfile.mkstemp(prefix=".completion-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _private_bytes(path, MAX_RECEIPT_BYTES) != raw:
                raise FinalityUnavailable("immutable completion replay changed")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _source(config):
    import boto3
    from botocore.config import Config

    env = config.credentials()
    endpoint = env.get("R2_PUBLIC_ENDPOINT", "")
    url = urlsplit(endpoint)
    if (url.scheme != "https" or not (url.hostname or "").endswith(".r2.cloudflarestorage.com")
            or url.username or url.password or url.query or url.fragment or url.port not in (None, 443)
            or url.path not in ("", "/")):
        raise FinalityUnavailable("a fixed HTTPS R2 checkpoint endpoint is required")
    bucket = env.get("R2_PUBLIC_BUCKET", "reliquary-public")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,62}", bucket):
        raise FinalityUnavailable("invalid checkpoint bucket")
    access, secret = env.get("R2_PUBLIC_ACCESS_KEY_ID"), env.get("R2_PUBLIC_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise FinalityUnavailable("checkpoint read credentials are unavailable")
    client = boto3.client("s3", endpoint_url=endpoint, region_name=env.get("R2_REGION", "auto"),
                          aws_access_key_id=access, aws_secret_access_key=secret,
                          config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30,
                                        retries={"max_attempts": 2, "mode": "standard"}))
    return R2CheckpointSource(client, bucket)


def finalize_training(entry, stopped, config, *, checkpoint_store=None):
    """The fixed supervisor callback. It performs no ledger, chain, rental or registry write."""
    state = config.runtime_dir
    if state.name != entry.task_id or state.is_symlink():
        raise FinalityUnavailable("runtime directory does not match this task")
    if not isinstance(stopped, StoppedRuntime) or any(process.returncode != 0 for process in stopped.processes):
        raise FinalityUnavailable("terminal delivery requires a clean exit")
    store = checkpoint_store or TrainingCheckpointStore(state.parent)
    if store.root.resolve() != state.parent.resolve():
        raise FinalityUnavailable("checkpoint cache belongs to another runtime root")
    runtime_store = TrainingRuntimeStore(state.parent, checkpoint_store=store)
    existing = runtime_store.final(entry, verify_objects_until=store.clock() + FREEZE_SECONDS)
    if existing is not None:
        receipt = existing["usage"]["receipt"]
        # Process IDs can be reused after completion; compare the original committed witness.
        if (canonical_sha256(stopped_document(stopped, verify_process_groups=False)) != receipt["termination_sha256"]
                or stopped.latest_checkpoint != receipt["checkpoint"]["version"]):
            raise FinalityUnavailable("completed runtime replay changed")
        return FinalizedRuntime(existing["usage"]["sha256"], existing["checkpoint"]["manifest_sha256"])
    cached = store.load_manifest(entry.task_id)
    if cached is not None:
        if (any(cached[key] != getattr(stopped, key) for key in ("task_id", "order_sha256", "run_id", "runtime_id"))
                or cached["checkpoint"]["version"] != stopped.latest_checkpoint):
            raise FinalityUnavailable("cached checkpoint belongs to another runtime")
        reference = {"version": cached["checkpoint"]["version"], "index_sha256": cached["index_sha256"]}
    else:
        if store.source is None:
            store.source = _source(config)
        reference = store.checkpoint_reference(stopped)
    envelope = finalize_usage(entry, state / "controller" / "rl-async.sqlite3", stopped, checkpoint=reference)
    manifest = store.freeze(stopped, envelope)
    write_final_usage(state / "usage.json", envelope)
    receipt = envelope["receipt"]
    completion = {"schema": COMPLETION_SCHEMA,
                  **{key: receipt[key] for key in ("task_id", "order_sha256", "quote_id", "quote_sha256",
                                                  "run_id", "runtime_id")},
                  "usage_sha256": envelope["sha256"], "checkpoint_manifest_sha256": manifest["manifest_sha256"]}
    _immutable_json(state / "completion.json", completion)
    runtime_store.final(entry)
    return FinalizedRuntime(envelope["sha256"], manifest["manifest_sha256"])


class TrainingRuntimeStore:
    """Admin reads are scoped to registry tasks, never caller-supplied paths or R2 keys."""
    def __init__(self, root, *, checkpoint_store=None):
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.is_symlink():
            raise ValueError("training runtime root must be an absolute private directory")
        self.checkpoints = checkpoint_store or TrainingCheckpointStore(self.root)

    def final(self, entry, *, verify_objects_until=None):
        order = entry.params.get("training_order") or {}
        folder = self.root / entry.task_id
        if folder.is_symlink():
            raise FinalityUnavailable("runtime task directory changed")
        try:
            completion = strict_json_loads(_private_bytes(folder / "completion.json", MAX_RECEIPT_BYTES))
        except FileNotFoundError:
            return None
        expected = {"schema", "task_id", "order_sha256", "quote_id", "quote_sha256", "run_id", "runtime_id",
                    "usage_sha256", "checkpoint_manifest_sha256"}
        if not isinstance(completion, dict) or set(completion) != expected or completion["schema"] != COMPLETION_SCHEMA:
            raise FinalityUnavailable("invalid training completion")
        try:
            usage = validate_usage_envelope(strict_json_loads(_private_bytes(folder / "usage.json", MAX_RECEIPT_BYTES)))
            manifest = self.checkpoints.load_manifest(entry.task_id, verify_objects_until=verify_objects_until)
            receipt = usage["receipt"]
            binding = {"task_id": entry.task_id,
                       **{key: order.get(key) for key in ("order_sha256", "quote_id", "quote_sha256")}}
            if (any(completion.get(key) != value or receipt.get(key) != value for key, value in binding.items())
                    or completion["usage_sha256"] != usage["sha256"] or manifest is None
                    or completion["checkpoint_manifest_sha256"] != manifest["manifest_sha256"]
                    or any(completion[key] != receipt[key] or manifest[key] != receipt[key]
                           for key in ("run_id", "runtime_id", "task_id", "order_sha256"))
                    or manifest["training_id"] != order.get("training_id")
                    or manifest["usage_sha256"] != usage["sha256"]
                    or receipt["checkpoint"] != {"version": manifest["checkpoint"]["version"],
                                                  "index_sha256": manifest["index_sha256"]}):
                raise FinalityUnavailable("final usage and checkpoint bindings differ")
            return {"usage": usage, "checkpoint": manifest}
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise FinalityUnavailable("training finality could not be verified") from exc

    def view(self, entry):
        final = self.final(entry)
        if final is not None:
            receipt = final["usage"]["receipt"]
            return {"runtime": {"state": "completed", "managed": True,
                                **{key: receipt[key] for key in ("task_id", "order_sha256", "run_id", "runtime_id")}},
                    "final_usage": final["usage"]}
        runtime = read_runtime_view(self.root, entry.task_id)
        if runtime.get("order_sha256") != (entry.params.get("training_order") or {}).get("order_sha256"):
            runtime = {"state": "unknown", "managed": False}
        return {"runtime": runtime, "final_usage": None}
