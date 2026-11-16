"""Materialize only an authenticated native job's public HF checkpoint files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from .affine import _directory, _private_open, _private_path
from .affine_checkpoint import _stamp
from .affine_native_evaluation import _load_intake
from .affine_scoring import require, write_json
from reliquary.protocol.release_contract import canonical_sha256


def _existing(path):
    while not path.exists():
        path = path.parent
    return path


def materialize(config_path, *, repository, revision, max_bytes):
    """Copy a pinned public snapshot from the shared HF cache into private files."""
    require(
        isinstance(repository, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", repository)
        and ".." not in repository
        and "--" not in repository
        and not repository.endswith(".git")
        and isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision),
        "public HF repository and full immutable revision required",
    )
    require(type(max_bytes) is int and 0 < max_bytes <= 1024**4, "explicit checkpoint byte ceiling required")
    config, envelope, _, manifest = _load_intake(config_path)
    expected = manifest["checkpoint"]["files"]
    destination = Path(config["checkpointDirectory"])
    provenance = destination.with_name(destination.name + ".hf-source.private.json")
    _private_path(destination)
    _private_path(provenance)
    _directory(destination.parent)
    require(not destination.exists() and not provenance.exists(), "new checkpoint and provenance paths required")

    from huggingface_hub import HfApi, constants, hf_hub_download

    info = HfApi(token=False).model_info(repository, revision=revision, files_metadata=True, timeout=30)
    require(
        info.sha == revision and info.private is False and info.gated is False,
        "exact public ungated HF revision required",
    )
    siblings = {item.rfilename: item for item in info.siblings or []}
    require(set(expected) <= set(siblings), "signed checkpoint files missing from HF revision")
    sizes = {name: siblings[name].size for name in expected}
    require(
        all(type(size) is int and size >= 0 for size in sizes.values())
        and 0 < sum(sizes.values()) <= max_bytes,
        "HF checkpoint metadata exceeds the byte ceiling",
    )
    total = sum(sizes.values())
    cache_root = Path(constants.HF_HUB_CACHE).expanduser().resolve()
    _private_path(cache_root)
    cache_parent = _existing(cache_root)
    shared_disk = cache_parent.stat().st_dev == destination.parent.stat().st_dev
    require(
        shutil.disk_usage(destination.parent).free >= total * (2 if shared_disk else 1) + 1024**2
        and (shared_disk or shutil.disk_usage(cache_parent).free >= total + 1024**2),
        "insufficient disk for HF cache and private checkpoint export",
    )
    temporary = Path(tempfile.mkdtemp(prefix=".hf-checkpoint-", dir=destination.parent))
    created = complete = False
    try:
        inventory = {}
        for name in sorted(expected):
            cached = Path(hf_hub_download(repository, name, revision=revision, token=False)).resolve(strict=True)
            fd = os.open(cached, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as source:
                before = os.fstat(source.fileno())
                require(stat.S_ISREG(before.st_mode) and before.st_size == sizes[name], "HF cache file size differs")
                digest, size = hashlib.sha256(), 0
                with os.fdopen(_private_open(temporary / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "wb") as output:
                    for block in iter(lambda: source.read(8 * 1024**2), b""):
                        size += len(block)
                        require(size <= sizes[name], "HF cache file grew while copying")
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                require(
                    _stamp(before) == _stamp(os.fstat(source.fileno())) == _stamp(cached.stat())
                    and size == sizes[name] and digest.hexdigest() == expected[name],
                    "HF checkpoint bytes differ from the signed inventory",
                )
            inventory[name] = {"sha256": digest.hexdigest(), "bytes": size}
        current, current_envelope, *_ = _load_intake(config_path)
        require(current == config and current_envelope == envelope, "native intake changed during HF download")
        result = {
            "schema": "hf-checkpoint-source/v1",
            "repository": repository,
            "revision": revision,
            "checkpointId": manifest["checkpoint"]["id"],
            "checkpointInventoryDigest": canonical_sha256(inventory),
        }
        write_json(temporary / "source.private.json", result)
        destination.mkdir(mode=0o700)
        created = True
        for name in sorted(expected):
            os.link(temporary / name, destination / name)
        os.link(temporary / "source.private.json", provenance)
        complete = True
        return result
    finally:
        if created and not complete:
            shutil.rmtree(destination)
        shutil.rmtree(temporary)
