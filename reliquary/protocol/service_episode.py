"""Signed episodes in the v2 service RL path (phase 2, plan 2C): the miner's precommit, the session
engagement it opens, and what a rollout commit binds for its episode. Pure: no I/O, no GPU."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA

PRECOMMIT_SCHEMA = "reliquary/episode-precommit/v1"
RL_ENGAGEMENT_PREFIX = "rl"
MAX_SAFE_INTEGER = 2**53 - 1
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ENVIRONMENT = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
_DIGITS = re.compile(r"(0|[1-9][0-9]{0,15})\Z")
_SS58 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{1,64}\Z")   # the base58 alphabet
_PRECOMMIT_FIELDS = frozenset({"schema", "order", "window", "environment", "task_index", "checkpoint",
                               "pool_sha256", "hotkey"})


class EpisodeWireError(ValueError):
    """A precommit, an engagement or an episode that is not well formed."""


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise EpisodeWireError(f"{name}: lowercase SHA-256 required")
    return value


def _uint(value: Any, name: str, high: int = MAX_SAFE_INTEGER) -> int:
    if type(value) is not int or not 0 <= value <= high:
        raise EpisodeWireError(f"{name}: integer in [0, {high}] required")
    return value


# A pool has at most 128 seeds (service_contract pool_seeds); SignedEpisodeMetadata.seed_index is le=127.
MAX_ENGAGEMENT_SEED = 127


@dataclass(frozen=True, slots=True)
class EpisodePrecommit:
    """What a miner commits to before opening sessions (spec §4.1.2): the order, the window, the env
    and task, the checkpoint and the task's public seed pool. Its digest names every session it opens."""

    order: str
    window: int
    environment: str
    task_index: int
    checkpoint: str
    pool_sha256: str
    hotkey: str

    def __post_init__(self) -> None:
        _sha(self.order, "order")
        _uint(self.window, "window")
        if not isinstance(self.environment, str) or not _ENVIRONMENT.fullmatch(self.environment):
            raise EpisodeWireError("environment: canonical environment id required")
        _uint(self.task_index, "task_index")
        if not isinstance(self.checkpoint, str) or not _REVISION.fullmatch(self.checkpoint):
            raise EpisodeWireError("checkpoint: the window checkpoint's 40-hex revision required")
        _sha(self.pool_sha256, "pool_sha256")
        if not isinstance(self.hotkey, str) or not _SS58.fullmatch(self.hotkey):
            raise EpisodeWireError("hotkey: a bounded ss58 address required")

    def to_dict(self) -> dict:
        return {"schema": PRECOMMIT_SCHEMA, "order": self.order, "window": self.window,
                "environment": self.environment, "task_index": self.task_index,
                "checkpoint": self.checkpoint, "pool_sha256": self.pool_sha256, "hotkey": self.hotkey}

    @classmethod
    def from_dict(cls, value: Any) -> EpisodePrecommit:
        if (not isinstance(value, Mapping) or set(value) != _PRECOMMIT_FIELDS
                or value["schema"] != PRECOMMIT_SCHEMA):
            raise EpisodeWireError("unknown episode precommit")
        return cls(order=value["order"], window=value["window"], environment=value["environment"],
                   task_index=value["task_index"], checkpoint=value["checkpoint"],
                   pool_sha256=value["pool_sha256"], hotkey=value["hotkey"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def rl_engagement(window: int, precommit_sha256: str, seed_index: int) -> str:
    """The engagement a session token names: ``rl:{window}:{precommit}:{seed}`` (spec §4.1.3)."""
    _uint(window, "window")
    _sha(precommit_sha256, "precommit_sha256")
    _uint(seed_index, "seed_index", MAX_ENGAGEMENT_SEED)
    return f"{RL_ENGAGEMENT_PREFIX}:{window}:{precommit_sha256}:{seed_index}"


def parse_rl_engagement(text: Any) -> tuple[int, str, int]:
    """``(window, precommit_sha256, seed_index)`` of an RL engagement; EpisodeWireError otherwise."""
    parts = text.split(":") if isinstance(text, str) else []
    if (len(parts) != 4 or parts[0] != RL_ENGAGEMENT_PREFIX or not _DIGITS.fullmatch(parts[1])
            or not _DIGITS.fullmatch(parts[3])):
        raise EpisodeWireError("not an RL engagement")
    return (_uint(int(parts[1]), "window"), _sha(parts[2], "precommit_sha256"),
            _uint(int(parts[3]), "seed_index", MAX_ENGAGEMENT_SEED))


def is_signed_episode(meta: Any) -> bool:
    """Whether rollout metadata carries a signed episode (``rollout.episode`` of schema signed-episode/v1)."""
    episode = meta.get("episode") if isinstance(meta, Mapping) else None
    return isinstance(episode, Mapping) and episode.get("schema_version") == SIGNED_EPISODE_SCHEMA


def episode_commit_material(episode: Any) -> bytes:
    """The bytes a commit signature binds for its episode: every field but the transcript in canonical
    JSON, then the transcript's digest (the corpus binding's digest, so both sides hash one way)."""
    from reliquary.protocol.signatures import transcript_digest

    if not isinstance(episode, Mapping) or "transcript" not in episode:
        raise EpisodeWireError("a signed episode carries its transcript")
    if episode.get("schema_version") != SIGNED_EPISODE_SCHEMA or not isinstance(episode["transcript"], Mapping):
        raise EpisodeWireError("not a signed episode")
    rest = {key: value for key, value in episode.items() if key != "transcript"}
    # Canonical JSON then a fixed 32-byte digest: the split point is unambiguous.
    return canonical_json_bytes(rest) + transcript_digest(episode["transcript"])


def episode_without_transcript(meta: Mapping) -> dict:
    """A copy of rollout metadata whose signed episode no longer carries its transcript (what the proof
    worker, the training payload and every log receive)."""
    out = dict(meta)
    episode = out.get("episode")
    if isinstance(episode, Mapping) and episode.get("schema_version") == SIGNED_EPISODE_SCHEMA:
        out["episode"] = {key: value for key, value in episode.items() if key != "transcript"}
    return out
