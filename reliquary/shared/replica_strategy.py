"""Which replica a proof slot builds, decided from the model and the card.

A resident replica is what the validator has always used: the whole model on the device. It stops
being possible when the model outgrows the card, which is where a streamed replica takes over. The
choice is derived rather than configured, because an operator who ticks the wrong box would either
waste a card or leave rollouts unverifiable, and neither failure announces itself.
"""

from __future__ import annotations

import json
from pathlib import Path

RESIDENT = "resident"
STREAMED = "streamed"

# What a pass needs on the card beyond the weights: activations for the batch in flight, plus the
# output head's projection. Measured at 13.5 GB for 65k tokens on a 2048-wide model, so a fifth of
# an 80 GB card is the room a resident replica has to leave to be worth choosing.
_HEADROOM = 0.20


def choose_replica(
    *, weights_bytes: int, free_bytes: int, override: str | None = None, headroom: float = _HEADROOM
) -> str:
    """RESIDENT when the model leaves room to run, STREAMED otherwise."""
    if override is not None:
        if override not in (RESIDENT, STREAMED):
            raise ValueError(f"replica override must be {RESIDENT!r} or {STREAMED!r}, got {override!r}")
        return override
    return RESIDENT if weights_bytes <= free_bytes * (1.0 - headroom) else STREAMED


def checkpoint_weight_bytes(path: str | Path) -> int:
    """How much the checkpoint's tensors weigh, read from the files rather than loaded.

    The decision is taken before anything reaches the card, so it counts bytes on disk: the shard
    index when there is one, the single file otherwise.
    """
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text())
        total = payload.get("metadata", {}).get("total_size")
        if total:
            return int(total)
        shards = {path / shard for shard in payload["weight_map"].values()}
        return sum(shard.stat().st_size for shard in shards)
    single = path / "model.safetensors"
    if single.exists():
        return single.stat().st_size
    raise FileNotFoundError(f"no safetensors checkpoint in {path}")
