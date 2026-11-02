"""The model an order runs on, pinned from its own files before any job exists.

An order names a public Hugging Face repo at a full revision. Its pin is what the job's manifest and contract
carry: the architecture (one of ``SUPPORTED_ARCHITECTURES``, the preset), the eos, and the checkpoint fingerprint
miners and validators check their loaded weights against (``corpus.encoding.checkpoint_fingerprint``). The
fingerprint is computed from the hub's per-file sha256 of the ``*.safetensors`` shards, so no weight is downloaded:
the same value a GPU computes over the downloaded shards.

Forced-draw orders need nothing else: their thresholds are job parameters, so there is no per-model calibration
(the TOPLOC qualification is only for proof-checked orders).
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

REVISION_RE = re.compile(r"\A[0-9a-f]{40}\Z")
# Weights an order's miners must hold: two 80 GB cards in bf16.
MAX_CHECKPOINT_BYTES = 160 * 10**9
# Everything a miner downloads with them (the whole repo: other formats, originals, tokenizer).
MAX_REPO_BYTES = 200 * 10**9
PIN_SCHEMA = "reliquary/model-pin/v1"


@dataclass(frozen=True, slots=True)
class ModelPin:
    model: str
    revision: str
    architecture: str
    eos_token_id: int
    checkpoint_sha256: str
    checkpoint_bytes: int
    # The longest sequence the model's config declares (None: not declared).
    max_position_embeddings: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PIN_SCHEMA, **asdict(self)}


def fingerprint_of(shards: Iterable[tuple[str, str]]) -> str:
    """``checkpoint_fingerprint`` from ``(file name, file sha256)`` of the top-level shards."""
    outer = hashlib.sha256()
    for name, sha256 in sorted(shards):
        outer.update(name.encode() + b"\0" + sha256.encode() + b"\n")
    return outer.hexdigest()


def hub_shards(repo: str, revision: str) -> list[tuple[str, str | None, int]]:
    """``(name, sha256, size)`` of every file of the revision, from the hub's metadata (sha256 None for a file
    that is not a top-level ``*.safetensors`` shard). ValueError: not a public model at that revision, or no
    shard; any other error is transient."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
    except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise ValueError(f"{repo}@{revision} is not a public model at that revision: "
                         f"{type(exc).__name__}") from exc
    shards = []
    for sibling in info.siblings or ():
        name = sibling.rfilename
        lfs = getattr(sibling, "lfs", None)
        if not _is_shard(name):
            shards.append((name, None, int(getattr(lfs, "size", None) or sibling.size or 0)))
            continue
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if not sha256:
            raise ValueError(f"{repo}@{revision}: shard {name} has no sha256 on the hub")
        shards.append((name, str(sha256), int(getattr(lfs, "size", None) or sibling.size or 0)))
    if not any(_is_shard(f[0]) for f in shards):
        raise ValueError(f"{repo}@{revision} has no .safetensors shards")
    return shards


def _is_shard(name: str) -> bool:
    """A file ``checkpoint_fingerprint`` hashes: a top-level ``*.safetensors``."""
    return "/" not in name and name.endswith(".safetensors")


def _max_positions(config: Mapping) -> int | None:
    for source in (config, config.get("text_config") or {}):
        value = source.get("max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
    return None


def pin_model(repo: str, revision: str, *, facts: Callable[[str, str], Mapping] | None = None,
              shards: Callable[[str, str], list] = hub_shards,
              config: Callable[[str, str], Mapping] | None = None) -> ModelPin:
    """The order's model pin, refused (ValueError) when the preset does not run it."""
    from reliquary.constants import SUPPORTED_ARCHITECTURES

    if not isinstance(revision, str) or not REVISION_RE.match(revision):
        raise ValueError("an order pins its model at a full 40-hex revision")
    if facts is None or config is None:
        from reliquary.validator import eval_control

        facts = facts or eval_control.model_facts
        config = config or (lambda r, v: eval_control._model_config(eval_control._model_files(r, v), r, v))
    found = facts(repo, revision)
    architecture = str(found["architecture"])
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"architecture_unsupported: {architecture!r}; supported: "
                         f"{sorted(SUPPORTED_ARCHITECTURES)}")
    every = shards(repo, revision)
    files = [f for f in every if _is_shard(f[0])]
    size = sum(int(f[2]) for f in files)
    if size > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"model_too_large: {size} bytes of weights, at most {MAX_CHECKPOINT_BYTES}")
    total = sum(int(f[2]) for f in every)
    if total > MAX_REPO_BYTES:
        raise ValueError(f"repo_too_large: miners download {total} bytes, at most {MAX_REPO_BYTES}")
    return ModelPin(model=repo, revision=revision, architecture=architecture,
                    eos_token_id=int(found["eos_token_id"]),
                    checkpoint_sha256=fingerprint_of((f[0], f[1]) for f in files),
                    checkpoint_bytes=size, max_position_embeddings=_max_positions(config(repo, revision)))


__all__ = ["MAX_CHECKPOINT_BYTES", "MAX_REPO_BYTES", "ModelPin", "fingerprint_of", "hub_shards", "pin_model"]
