"""Scoring a customer env's rows from a host that must not run them (admin host, validators).

The row's reference names the frozen task (``{"key", "prompt_sha256"}``); the version's image comes from its
qualified record; the score comes from the reward host. Nothing here imports customer code.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any

from reliquary.customer_env import is_customer_env, parse_env_id
from reliquary.customer_env.runner import RewardInfrastructureError, ScoreResult

MAX_REFERENCE_CHARS = 512


def reference(key: str, prompt_sha256: str) -> str:
    return json.dumps({"key": key, "prompt_sha256": prompt_sha256}, sort_keys=True, separators=(",", ":"))


def check_reference(grader: str, text: Any) -> dict:
    parse_env_id(grader)
    if not isinstance(text, str) or len(text) > MAX_REFERENCE_CHARS:
        raise ValueError("a customer env reference is {key, prompt_sha256}")
    try:
        ref = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("a customer env reference is {key, prompt_sha256}") from None
    if not isinstance(ref, dict) or set(ref) != {"key", "prompt_sha256"} or not isinstance(ref["key"], str) \
            or not isinstance(ref["prompt_sha256"], str) or len(ref["prompt_sha256"]) != 64:
        raise ValueError("a customer env reference is {key, prompt_sha256}")
    return ref


class CustomerScorer:
    """``image_for(env_id) -> image digest`` (a qualified version's); ``open_client(image) -> client``
    (``RemoteRewardClient`` on a reward host). Clients are kept per image."""

    def __init__(self, image_for: Callable[[str], str], open_client: Callable[[str], Any]) -> None:
        self._image_for, self._open = image_for, open_client
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _client(self, env_id: str):
        image = self._image_for(env_id)
        with self._lock:
            if image not in self._clients:
                self._clients[image] = self._open(image)
            return self._clients[image]

    def score_many(self, env_id: str, refs: list[dict], answers: list[str]) -> list[ScoreResult]:
        items = [{"key": r["key"], "prompt_sha256": r["prompt_sha256"], "answer": a} for r, a in zip(refs, answers)]
        return self._client(env_id).score(items)

    def score(self, env_id: str, ref_text: str, answer: str) -> tuple[float | None, str]:
        """``(score, detail)``: a score, or None with why (the customer's code on this item, or ours)."""
        ref = check_reference(env_id, ref_text)
        try:
            (result,) = self.score_many(env_id, [ref], [answer])
        except RewardInfrastructureError as exc:
            return None, f"grader_infrastructure: {exc}"[:500]
        except LookupError as exc:
            return None, f"customer_env_unavailable: {exc}"[:500]
        if result.error is not None:
            return None, f"customer_grader: {result.error}"[:500]
        return result.score, ""

    def close(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()


def from_environment() -> CustomerScorer | None:
    """The scorer this host is configured for, or None: ``RELIQUARY_REWARD_HOST_URL``, the client certificate
    (``RELIQUARY_REWARD_HOST_CLIENT_CERT`` / ``_CLIENT_KEY``), the host's CA (``RELIQUARY_REWARD_HOST_CA``) and the
    version index (``RELIQUARY_CUSTOMER_ENV_DB``)."""
    url, db = os.environ.get("RELIQUARY_REWARD_HOST_URL"), os.environ.get("RELIQUARY_CUSTOMER_ENV_DB")
    if not url or not db:
        return None
    from reliquary.customer_env.host import RemoteRewardClient
    from reliquary.customer_env.registry import Registry

    registry = Registry(db)
    cert = (os.environ.get("RELIQUARY_REWARD_HOST_CLIENT_CERT"), os.environ.get("RELIQUARY_REWARD_HOST_CLIENT_KEY"))
    ca = os.environ.get("RELIQUARY_REWARD_HOST_CA")

    def image_for(env_id: str) -> str:
        record = registry.get(env_id)
        if record is None or record.status != "qualified" or not record.image_digest:
            raise LookupError(f"{env_id} is not a qualified version")
        return record.image_digest

    return CustomerScorer(image_for, lambda image: RemoteRewardClient(
        url, image, cert=cert if all(cert) else None, ca=ca))


__all__ = ["CustomerScorer", "check_reference", "from_environment", "is_customer_env", "reference"]
