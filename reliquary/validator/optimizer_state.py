"""The optimizer's state on local disk, so a same-run restart resumes Adam instead of re-warming the LR.

After every optimizer step each process (one per FSDP2 rank) writes what it steps: the fp32 master shards and the
Adam moments, tagged with the scheduler position. At boot, the snapshot whose position is the restored schedule
step is loaded only if its masters round to exactly the bf16 weights just resumed (so it belongs to that
checkpoint), and only if every rank found one; otherwise the restart re-warms as before.

Enabled by RELIQUARY_OPTIMIZER_STATE_DIR (unset: off). ~12 bytes per parameter, split across the ranks.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

STATE_DIR_ENV = "RELIQUARY_OPTIMIZER_STATE_DIR"
FORMAT = 1
KEEP = 2                        # snapshots per rank: the one being written never replaces the last complete one
_NAME = re.compile(r"rank(\d+)-of(\d+)-step(\d+)\.pt")


def state_dir(environ=os.environ) -> Path | None:
    raw = environ.get(STATE_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _host(value):
    return _local(value).detach().to("cpu", copy=True) if torch.is_tensor(value) else value


def snapshot(params, optimizer, step: int) -> dict:
    """Host copies of the stepped tensors (local shards) and their optimizer state."""
    return {
        "format": FORMAT,
        "step": int(step),
        "params": [_host(p) for p in params],
        "state": [{key: _host(value) for key, value in optimizer.state.get(p, {}).items()} for p in params],
    }


def check(params, snap: dict, step: int) -> str | None:
    """Why *snap* cannot be restored onto *params* at schedule *step*, or None. Mutates nothing."""
    if snap.get("format") != FORMAT:
        return f"format {snap.get('format')}"
    if snap.get("step") != step:
        return f"snapshot at step {snap.get('step')}, schedule at {step}"
    saved = snap["params"]
    if len(saved) != len(params):
        return f"{len(saved)} tensors saved, {len(params)} stepped"
    for index, (param, master) in enumerate(zip(params, saved)):
        local = _local(param).detach()
        if tuple(master.shape) != tuple(local.shape):
            return f"tensor {index}: shape {tuple(master.shape)} saved, {tuple(local.shape)} now"
        if not torch.equal(master.to(torch.bfloat16), local.cpu().to(torch.bfloat16)):
            return f"tensor {index}: masters do not round to the resumed weights"
    return None


@torch.no_grad()
def restore(params, optimizer, snap: dict) -> None:
    """Write the saved masters into *params* and the saved state into *optimizer* (after ``check``)."""
    for param, master, saved in zip(params, snap["params"], snap["state"]):
        local = _local(param)
        local.copy_(master.to(local.device))
        state = {}
        for key, value in saved.items():
            if torch.is_tensor(value) and value.dim() > 0 and tuple(value.shape) == tuple(local.shape):
                buffer = torch.zeros_like(param, dtype=value.dtype)    # a DTensor for an FSDP2 shard
                _local(buffer).copy_(value.to(local.device))
                state[key] = buffer
            elif torch.is_tensor(value) and value.dim() > 0:
                state[key] = value.to(local.device)
            else:
                state[key] = value                              # AdamW's step count stays a host scalar
        if state:
            optimizer.state[param] = state


def _path(root: Path, rank: int, world: int, step: int) -> Path:
    return root / f"rank{rank:02d}-of{world:02d}-step{step:08d}.pt"


def load(root: Path, rank: int, world: int, step: int) -> dict | None:
    path = _path(root, rank, world, step)
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


class SnapshotWriter:
    """Writes one rank's snapshots in the background (tmp file, then rename), keeping the last ``KEEP``."""

    def __init__(self, root: Path, rank: int, world: int) -> None:
        self.root, self.rank, self.world = root, rank, world
        self._thread: threading.Thread | None = None

    def submit(self, snap: dict) -> None:
        self.wait()
        self._thread = threading.Thread(target=self._write, args=(snap,), name="optimizer-state", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _write(self, snap: dict) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            final = _path(self.root, self.rank, self.world, snap["step"])
            tmp = final.with_suffix(".tmp")
            torch.save(snap, tmp)
            os.replace(tmp, final)
            self._prune()
            logger.info("optimizer state saved: %s", final.name)
        except Exception:                               # a lost snapshot only costs a re-warmup at the next restart
            logger.exception("optimizer state not saved (step %s)", snap.get("step"))

    def _prune(self) -> None:
        mine = []
        for path in self.root.iterdir():
            found = _NAME.fullmatch(path.name)
            if found and int(found[1]) == self.rank and int(found[2]) == self.world:
                mine.append((int(found[3]), path))
        for _, path in sorted(mine)[:-KEEP]:
            path.unlink(missing_ok=True)


__all__ = ["KEEP", "STATE_DIR_ENV", "SnapshotWriter", "check", "load", "restore", "snapshot", "state_dir"]
