"""Where a published set lives, and `publish-set`.

The platform bucket holds what the platform and the pod may read:
``eval-sets/{set_id}/prompts.jsonl`` and ``set.json``. The subnet bucket holds
``reliquary/eval-sets/{set_id}/grading.jsonl`` (read by the admin host only),
``prompts.jsonl`` (for the validators of an eval job) and its own copy of
``set.json``: the grader trusts that copy, never the
platform's. A set is frozen, so publishing is create-only: the same bytes again
are a no-op, other bytes are refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reliquary.eval.sets import validated_set_id

PLATFORM_PREFIX = "eval-sets"
SUBNET_PREFIX = "reliquary/eval-sets"


def platform_key(set_id: str, name: str) -> str:
    return f"{PLATFORM_PREFIX}/{validated_set_id(set_id)}/{name}"


def subnet_key(set_id: str, name: str) -> str:
    return f"{SUBNET_PREFIX}/{validated_set_id(set_id)}/{name}"


class SetConflict(Exception):
    """The bucket already holds other bytes under a frozen set's key."""


class SubnetEvalStore:
    """The subnet bucket (``R2_*`` credentials, as the admin host's record store)."""

    def __init__(self, **client_kwargs) -> None:
        self._client_kwargs = client_kwargs

    async def get_bytes(self, key: str) -> bytes | None:
        from reliquary.infrastructure.corpus_job_store import _get

        body, _ = await _get(key, **dict(self._client_kwargs))
        return body

    async def put_bytes(self, key: str, body: bytes) -> None:
        """Create-only: a key already written raises ``SetConflict`` unless
        it holds these very bytes."""
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict, _put

        try:
            await _put(key, body, None, **dict(self._client_kwargs))
        except CorpusStoreConflict:
            if await self.get_bytes(key) != body:
                raise SetConflict(f"{key} already holds other bytes") from None


async def _create(store, key: str, body: bytes) -> bool:
    """Write unless present; True when written. Other bytes are a conflict."""
    existing = await store.get_bytes(key)
    if existing is not None:
        if existing != body:
            raise SetConflict(f"{key} already holds other bytes: a set is frozen")
        return False
    await store.put_bytes(key, body)
    return True


def _verified(directory: Path) -> tuple[dict, bytes, bytes, bytes]:
    card_body = (directory / "set.json").read_bytes()
    card = json.loads(card_body)
    prompts = (directory / "prompts.jsonl").read_bytes()
    grading = (directory / "grading.jsonl").read_bytes()
    for name, body in (("prompts", prompts), ("grading", grading)):
        if hashlib.sha256(body).hexdigest() != card.get(f"{name}_sha256"):
            raise ValueError(f"{name}.jsonl does not match the sha256 set.json records")
    validated_set_id(card.get("set_id"))
    return card, card_body, prompts, grading


async def publish_set(directory: str | Path, *, platform, subnet) -> dict:
    """Upload a built set. The platform's ``set.json`` goes last: its presence
    is what makes a set orderable."""
    card, card_body, prompts, grading = _verified(Path(directory))
    set_id = card["set_id"]
    written = []
    for store, key, body in (
        (subnet, subnet_key(set_id, "grading.jsonl"), grading),
        # Validators of an eval job read its prompts here (no platform credential).
        (subnet, subnet_key(set_id, "prompts.jsonl"), prompts),
        (subnet, subnet_key(set_id, "set.json"), card_body),
        (platform, platform_key(set_id, "prompts.jsonl"), prompts),
        (platform, platform_key(set_id, "set.json"), card_body),
    ):
        if await _create(store, key, body):
            written.append(key)
    return {"set_id": set_id, "written": written, "count": card["count"],
            "env": card["env"]}


__all__ = [
    "PLATFORM_PREFIX",
    "SUBNET_PREFIX",
    "SetConflict",
    "SubnetEvalStore",
    "platform_key",
    "publish_set",
    "subnet_key",
]
