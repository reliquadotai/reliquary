"""Checkpoint publication helpers: HF upload and verification, R2 checkpoint keys, publication receipts."""

from __future__ import annotations

import asyncio
from fnmatch import fnmatchcase
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from reliquary.shared.checkpoint_identity import (
    require_immutable_checkpoint_revision,
)
from reliquary.shared.checkpoint_namespace import (
    CheckpointNamespace, active_checkpoint_namespace,
    LEGACY_CANDIDATE_MANIFEST_KEY, LEGACY_CHECKPOINT_PREFIX,
)

logger = logging.getLogger(__name__)
CANDIDATE_MANIFEST_KEY = LEGACY_CANDIDATE_MANIFEST_KEY
R2_CHECKPOINT_PREFIX = LEGACY_CHECKPOINT_PREFIX
PUBLICATION_RECEIPT = "reliquary_publication.json"
PENDING_PUBLICATION = "publication.json"
WEIGHT_PATTERNS = (
    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
    "pytorch_model.bin", "pytorch_model.bin.index.json", "pytorch_model-*.bin",
)
# Weight blobs are mirrored AFTER the candidate manifest: the controller stages
# only metadata, so the multi-gigabyte upload stays off the rotation path.
# MIRROR_COMPLETE is written last; any reader that needs the weights from the
# mirror must see it first.
MIRROR_DEFERRED_SUFFIXES = (".safetensors", ".bin")
MIRROR_COMPLETE = "reliquary_mirror_complete.json"
# The deferred mirror and its marker are the new trainer's protocol: used for a
# task-scoped namespace, or when this flag is "1". Otherwise (the legacy
# namespace, the live run) every file is mirrored before the manifest and no
# marker is written or required, exactly as before, so a controller and a
# trainer of either release adopt each other's checkpoints.
MIRROR_MARKER_ENV = "RELIQUARY_CHECKPOINT_MIRROR_MARKER"


def mirror_marker_required(namespace: CheckpointNamespace | None = None,
                           environ: Any = None) -> bool:
    namespace = namespace or active_checkpoint_namespace()
    flag = (os.environ if environ is None else environ).get(MIRROR_MARKER_ENV, "").strip()
    return bool(namespace.scoped) or flag == "1"


class PublicationConflict(RuntimeError):
    """Local or remote state cannot be attributed to this publication."""


def checkpoint_key(revision: str, filename: str, *, namespace: CheckpointNamespace | None = None) -> str:
    revision = require_immutable_checkpoint_revision(revision)
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("checkpoint filename must be a single path component")
    namespace = namespace or active_checkpoint_namespace()
    return f"{namespace.checkpoint_prefix}/{revision}/{filename}"


def _multipart_transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=32 * 1024 * 1024,
        max_concurrency=16,
    )


def _default_hf_head(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, timeout=30).sha


async def _default_hf_upload(**kwargs) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    # HF combines deletions and additions in one commit, retaining files being
    # replaced and preserving history and the caller's parent-commit check.
    revision = (await asyncio.to_thread(
        api.upload_folder, **kwargs, delete_patterns=list(WEIGHT_PATTERNS)
    )).oid
    await asyncio.to_thread(
        _verify_weight_file_set, api, kwargs["repo_id"], revision,
        [path.name for path in Path(kwargs["folder_path"]).iterdir() if path.is_file()],
    )
    return revision


def _verify_weight_file_set(api, repo_id, revision, files) -> None:
    def weights(names):
        return {name for name in names if any(fnmatchcase(name, p) for p in WEIGHT_PATTERNS)}

    if weights(api.list_repo_files(repo_id, revision=revision)) != weights(files):
        raise PublicationConflict("HF publication has an unexpected weight file set")


def _default_hf_verify(
    *, repo_id, revision, parent_revision, commit_message, files
) -> None:
    """Validate commit provenance and content without downloading model weights."""
    from huggingface_hub import HfApi

    api = HfApi()
    _verify_weight_file_set(api, repo_id, revision, files)
    commits = api.list_repo_commits(repo_id, revision=revision)
    if (
        len(commits) < 2
        or commits[0].commit_id != revision
        or commits[0].title != commit_message
        or commits[1].commit_id != parent_revision
    ):
        raise PublicationConflict("HF HEAD is not the exact pending publication commit")
    paths = {
        item.path: item
        for item in api.get_paths_info(repo_id, list(files), revision=revision)
    }
    if set(paths) != set(files):
        raise PublicationConflict("HF publication file set is incomplete")
    for name, expected in files.items():
        item = paths[name]
        lfs = getattr(item, "lfs", None)
        digest = lfs.sha256 if lfs is not None else getattr(item, "blob_id", None)
        expected_digest = expected["sha256"] if lfs is not None else expected["blob_id"]
        if item.size != expected["size"] or digest != expected_digest:
            raise PublicationConflict(f"HF publication content mismatch: {name}")


def _file_identity(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise PublicationConflict("snapshot requires regular files without symlinks")
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            blob.update(chunk)
        os.fsync(handle.fileno())
    return {"size": size, "sha256": sha.hexdigest(), "blob_id": blob.hexdigest()}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


