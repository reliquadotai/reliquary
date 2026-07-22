"""Validator-authoritative prompt and target content identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_PROMPT_DOMAIN = b"reliquary/prompt-content/v1\0"
_TARGET_DOMAIN = b"reliquary/target-content/v1\0"


def _content_digest(domain: bytes, environment: str, payload: bytes) -> str:
    env = str(environment).strip()
    if not env:
        raise ValueError("environment must be non-empty")
    hasher = hashlib.sha256()
    hasher.update(domain)
    hasher.update(env.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(payload)
    return hasher.hexdigest()


def prompt_content_sha256(environment: str, rendered_prompt: str) -> str:
    """Hash the exact validator-rendered prompt bytes for one environment."""
    if not isinstance(rendered_prompt, str):
        raise TypeError("rendered_prompt must be a string")
    return _content_digest(
        _PROMPT_DOMAIN,
        environment,
        rendered_prompt.encode("utf-8"),
    )


def target_content_sha256(
    environment: str,
    problem: dict[str, Any],
    *,
    code_cases: list[dict[str, Any]] | None = None,
) -> str:
    """Hash trusted answer material for diagnostics, never eligibility.

    Math keeps the source answer bytes intact. Code uses canonical JSON for
    structured grader cases so dictionary insertion order cannot change the
    identity.
    """
    if environment == "opencodeinstruct" and code_cases is not None:
        payload = json.dumps(
            code_cases,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    else:
        target = problem.get("ground_truth", problem.get("answer", ""))
        if isinstance(target, str):
            payload = target.encode("utf-8")
        else:
            payload = json.dumps(
                target,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
    return _content_digest(_TARGET_DOMAIN, environment, payload)
