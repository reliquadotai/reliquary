"""Which replica a proof slot builds, decided from the model and the card.

A resident replica is what the validator has always used: the whole model on the device. It stops
being possible when the model outgrows the card, which is where a streamed replica takes over.

The choice is derived by default, because an operator who ticks the wrong box would either waste a
card or leave rollouts unverifiable, and neither failure announces itself. A task may pin one all
the same: a fleet whose cards differ would otherwise split between the two paths, and a task that
wants to know which one it is paying for says so. What a task cannot do is pin a path a card
cannot build — that is refused where it is noticed, at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

RESIDENT = "resident"
STREAMED = "streamed"


class ReplicaUnavailable(RuntimeError):
    """This slot cannot build the replica it was told to build."""

# What a pass needs on the card beyond the weights: activations for the batch in flight, plus the
# output head's projection. Measured at 13.5 GB for 65k tokens on a 2048-wide model, so a fifth of
# an 80 GB card is the room a resident replica has to leave to be worth choosing.
_HEADROOM = 0.20


def choose_replica(
    *,
    weights_bytes: int,
    free_bytes: int,
    override: str | None = None,
    declared: str | None = None,
    headroom: float = _HEADROOM,
) -> str:
    """RESIDENT when the model leaves room to run, STREAMED otherwise.

    ``declared`` is what the task pinned, and it wins over the derivation: a task that wants every
    validator on one path gets it, whatever each card could have held. It does not win over
    physics — a card that cannot hold the model refuses here, at startup, rather than running out
    of memory mid-window and blaming the miner whose rollout was in flight.

    ``override`` is a single validator forcing a path for a test or an incident, and it is taken as
    given: that is what forcing means.
    """
    for name, value in (("override", override), ("declared", declared)):
        if value is not None and value not in (RESIDENT, STREAMED):
            raise ValueError(f"replica {name} must be {RESIDENT!r} or {STREAMED!r}, got {value!r}")
    if override is not None:
        return override
    fits = weights_bytes <= free_bytes * (1.0 - headroom)
    if declared is not None:
        if declared == RESIDENT and not fits:
            raise ReplicaUnavailable(
                f"the task pins a resident replica, and {weights_bytes / 1e9:.1f} GB of weights "
                f"do not fit in {free_bytes / 1e9:.1f} GB of free device memory with room to run"
            )
        return declared
    return RESIDENT if fits else STREAMED


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
