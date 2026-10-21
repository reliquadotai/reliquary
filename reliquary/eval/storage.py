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
import asyncio
import json
from pathlib import Path

from reliquary.eval.sets import validated_set_id
from reliquary.corpus.delivery import LocalDirectorySink, R2DeliverySink

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

    async def create_bytes(self, key: str, body: bytes) -> bool:
        """Create-only: a key already written raises ``SetConflict`` unless
        it holds these very bytes."""
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict, _put

        try:
            await _put(key, body, None, **dict(self._client_kwargs))
        except CorpusStoreConflict:
            if await self.get_bytes(key) != body:
                raise SetConflict(f"{key} already holds other bytes") from None
            return False
        return True

    async def put_bytes(self, key: str, body: bytes) -> None:
        await self.create_bytes(key, body)


class PlatformEvalStore(R2DeliverySink):
    """Set publication uses a conditional create, never the delivery overwrite API."""

    async def create_bytes(self, key: str, body: bytes) -> bool:
        from botocore.exceptions import ClientError

        try:
            await asyncio.to_thread(self._client.put_object, Bucket=self._bucket,
                                    Key=key, Body=body, IfNoneMatch="*")
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
                raise
            if await self.get_bytes(key) != body:
                raise SetConflict(f"{key} already holds other bytes") from None
            return False
        return True


async def _create(store, key: str, body: bytes) -> bool:
    """Write unless present; True when written. Other bytes are a conflict."""
    if isinstance(store, LocalDirectorySink):
        path = store._path(key)
        try:
            with path.open("xb") as handle:
                handle.write(body)
        except FileExistsError:
            if path.read_bytes() != body:
                raise SetConflict(f"{key} already holds other bytes: a set is frozen") from None
            return False
        return True
    create = getattr(store, "create_bytes", None)
    if create is None:
        raise TypeError("set publication requires an atomic create_bytes store")
    return await create(key, body)


def _verified(directory: Path) -> tuple[dict, bytes, bytes, bytes]:
    card_body = (directory / "set.json").read_bytes()
    card = json.loads(card_body)
    prompts = (directory / "prompts.jsonl").read_bytes()
    grading = (directory / "grading.jsonl").read_bytes()
    for name, body in (("prompts", prompts), ("grading", grading)):
        if hashlib.sha256(body).hexdigest() != card.get(f"{name}_sha256"):
            raise ValueError(f"{name}.jsonl does not match the sha256 set.json records")
    validated_set_id(card.get("set_id"))
    from reliquary.eval.protocol import validate_set

    validate_set(card)
    return card, card_body, prompts, grading


def is_operator_set(card: dict) -> bool:
    """A set `build_source_set` built (any source, any range): the operator's
    own, never orderable by the platform's customers."""
    return "source_kind" in card


def read_published_set(set_id: str) -> tuple[dict, bytes]:
    """A published set's card and prompt lines: from ``RELIQUARY_EVAL_SETS_DIR``
    when it holds the set, else from the subnet bucket."""
    import asyncio
    import os

    from reliquary.eval.prompt_source import SETS_DIR_ENV

    validated_set_id(set_id)
    root = os.environ.get(SETS_DIR_ENV, "").strip()
    if root:
        directory = Path(root) / set_id
        if (directory / "set.json").exists() and (directory / "prompts.jsonl").exists():
            return (json.loads((directory / "set.json").read_text()),
                    (directory / "prompts.jsonl").read_bytes())

    async def read():
        store = SubnetEvalStore()
        return (await store.get_bytes(subnet_key(set_id, "set.json")),
                await store.get_bytes(subnet_key(set_id, "prompts.jsonl")))

    card, prompts = asyncio.run(read())
    if card is None or prompts is None:
        raise ValueError(f"set {set_id!r} is not published (reliquary eval publish-set)")
    return json.loads(card), prompts


async def publish_set(directory: str | Path, *, platform, subnet, publish_reviewed: bool = False) -> dict:
    """Create immutable set objects, with public metadata last.

    Operator sets stay subnet-only unless explicitly published for review;
    publication never qualifies or registers a set in the platform catalog.
    """
    card, card_body, prompts, grading = _verified(Path(directory))
    set_id = card["set_id"]
    if publish_reviewed:
        from reliquary.eval.protocol import validate_set
        validate_set(card)
        source = card.get("protocol", {}).get("source", {})
        evidence = card.get("protocol", {}).get("disjointness", {})
        if (not source.get("data") or not source.get("licenses")
                or type(evidence.get("disjoint")) is not bool or not evidence.get("evidence_sha256")):
            raise ValueError("reviewed publication requires frozen data, rights and explicit overlap evidence")
    written = []
    uploads = [
        (subnet, subnet_key(set_id, "grading.jsonl"), grading),
        # Validators of an eval job read its prompts here (no platform credential).
        (subnet, subnet_key(set_id, "prompts.jsonl"), prompts),
        (subnet, subnet_key(set_id, "set.json"), card_body),
    ]
    if not is_operator_set(card) or publish_reviewed:
        if platform is None:
            raise ValueError(f"set {set_id} is a platform preset: it needs the platform bucket")
        from reliquary.eval.protocol import canonical, public_taskset

        public_card = card
        if card.get("protocol"):
            public_card = {key: card[key] for key in (
                "schema", "set_id", "env", "source", "source_kind", "interaction", "split",
                "index_range", "source_length", "count", "seed", "order", "created_at",
                "prompt_template_id", "default_max_new_tokens", "prompts_sha256", "grading_sha256",
                "environment_manifest_sha256", "disjointness", "episode", "protocol", "protocol_json",
                "protocol_sha256") if key in card}
            if card.get("taskset"):
                public_card["taskset"] = public_taskset(card["taskset"])
        order = [json.loads(line)["problem_id"] for line in prompts.decode().splitlines() if line.strip()]
        if (len(order) != card["count"] or any(not isinstance(item, str) or not item for item in order)
                or len(set(order)) != len(order)):
            raise ValueError("public set order requires each frozen problem identity exactly once")
        public_card = {**public_card, "order": order}
        public_body = canonical(public_card) if public_card != card else card_body
        uploads += [(platform, platform_key(set_id, "prompts.jsonl"), prompts),
                    (platform, platform_key(set_id, "set.json"), public_body)]
    for store, key, body in uploads:
        if await _create(store, key, body):
            written.append(key)
    return {"set_id": set_id, "written": written, "count": card["count"], "env": card["env"],
            **({"registration": {"set_id": set_id, "env": card["env"], "name": set_id,
                "description": "Reviewed frozen evaluation set; runtime qualification pending",
                "problem_count": card["count"], "default_max_new_tokens": card.get("default_max_new_tokens"),
                "avg_tokens": None, "avg_tokens_thinking": None, "prompts_sha256": card["prompts_sha256"],
                "created_at": card["created_at"], "availability": "withdrawn",
                "protocol_json": card["protocol_json"], "protocol_sha256": card["protocol_sha256"]}}
               if publish_reviewed else {})}


__all__ = [
    "PLATFORM_PREFIX",
    "SUBNET_PREFIX",
    "SetConflict",
    "SubnetEvalStore",
    "PlatformEvalStore",
    "is_operator_set",
    "platform_key",
    "publish_set",
    "read_published_set",
    "subnet_key",
]
