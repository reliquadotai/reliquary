"""A checkpoint written the way a streamed traversal reads it.

A traversal wants one decoder layer at a time, with that layer's experts already stacked the way
the module holds them. A published checkpoint stores experts one tensor each, so a traversal that
reads it directly re-does the same stacking on every pass — measured at four fifths of the pass.
Written once when a revision is installed, every later traversal reads a file and loads it.

The store also outlives the staged download: intake deletes the staged directory as soon as the
swap completes, and a streamed replica needs its weights on disk for as long as it is the one
proving. The store is what it reads from, and it is kept until a newer revision replaces it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from reliquary.shared.layer_source import LAYER_FILE, LAYER_PREFIX, _ShardedTensors, fuse_layer_state

logger = logging.getLogger(__name__)

BACKBONE_FILE = "model.safetensors"
STAMP_FILE = "fused-layers.json"
# One generation of slack: a slot that has not rotated yet is still reading the previous store.
_KEEP_STORES = 2


def layer_file(index: int) -> str:
    return LAYER_FILE.format(index)


def fused_store_root() -> Path:
    """Where this validator keeps the stores, beside its other durable state."""
    state = os.environ.get("RELIQUARY_STATE_DIR", "/root/reliquary/state")
    return Path(state) / "proof_layers"


def is_fused_store(path: str | Path) -> bool:
    return (Path(path) / STAMP_FILE).exists()


def build_fused_store(source: str | Path, dest: str | Path, *, revision: str | None = None) -> Path:
    """Write ``source`` into ``dest`` as one file per layer, plus the fixed parts.

    The store appears whole or not at all: it is written beside its final name and renamed into
    place, so a crash mid-write leaves a directory a later run discards rather than a half store a
    traversal would read as if it were complete.
    """
    from safetensors.torch import save_file

    source, dest = Path(source), Path(dest)
    reader = _ShardedTensors(source)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    n_layers = int(text["num_hidden_layers"])
    names = reader.names()

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        save_file(
            reader.read([n for n in names if not n.startswith(LAYER_PREFIX)]),
            str(staging / BACKBONE_FILE),
        )
        for index in range(n_layers):
            prefix = f"{LAYER_PREFIX}{index}."
            raw = reader.read([n for n in names if n.startswith(prefix)])
            if not raw:
                raise ValueError(f"checkpoint {source} has no tensors for layer {index}")
            state = fuse_layer_state({name[len(prefix):]: tensor for name, tensor in raw.items()})
            del raw
            save_file(state, str(staging / layer_file(index)))
            del state
        for name in ("config.json", "generation_config.json"):
            if (source / name).exists():
                shutil.copy2(source / name, staging / name)
        (staging / STAMP_FILE).write_text(
            json.dumps({"layers": n_layers, "revision": revision}), encoding="utf-8",
        )
        try:
            staging.rename(dest)
        except OSError:
            # Another slot finished the same store first. Its copy is as good as this one.
            if not is_fused_store(dest):
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dest


def install_fused_store(source: str | Path, root: str | Path, revision: str) -> Path:
    """Return the store for ``revision``, building it from ``source`` the first time."""
    root = Path(root)
    dest = root / revision
    if is_fused_store(dest):
        return dest
    logger.info("fusing checkpoint %s for streamed proofs", revision[:12])
    build_fused_store(source, dest, revision=revision)
    prune_fused_stores(root, keep=revision)
    return dest


def prune_fused_stores(root: str | Path, keep: str) -> None:
    """Drop the stores this validator no longer proves against.

    The newest one behind the current is spared: rotation walks the slots one at a time, and the
    ones that have not rotated yet are still reading the store the current one just left.
    """
    root = Path(root)
    if not root.is_dir():
        return
    stores = [p for p in root.iterdir() if p.is_dir() and is_fused_store(p)]
    survivors = sorted(stores, key=lambda p: p.stat().st_mtime, reverse=True)[:_KEEP_STORES]
    for store in stores:
        if store.name != keep and store not in survivors:
            shutil.rmtree(store, ignore_errors=True)
