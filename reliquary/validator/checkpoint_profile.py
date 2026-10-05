"""Protocol-lineage metadata embedded in every validator checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from reliquary.constants import (
    PROTOCOL_MODEL_ID,
    PROTOCOL_MODEL_REVISION,
    PROTOCOL_GENERATION_CONTRACT,
    PROTOCOL_PROFILE_ID,
    PROTOCOL_VERSION,
    TRAINING_RUN_ID,
)
from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.checkpoint_namespace import CheckpointNamespace, active_checkpoint_namespace


CHECKPOINT_PROFILE_NAME = "reliquary_protocol_profile.json"


class CheckpointProfileMismatch(RuntimeError):
    pass


def active_checkpoint_profile(*, namespace: CheckpointNamespace | None = None) -> dict[str, Any]:
    namespace = namespace or active_checkpoint_namespace()
    profile = {
        "schema_version": 2 if PROTOCOL_VERSION >= 5 else 1,
        "profile_id": PROTOCOL_PROFILE_ID,
        "protocol_version": PROTOCOL_VERSION,
        "base_model_id": PROTOCOL_MODEL_ID,
        "base_model_revision": PROTOCOL_MODEL_REVISION,
        # Legacy run identity is informational. Scoped profiles bind the run
        # exactly before any cursor or LR schedule state is restored.
        "training_run_id": TRAINING_RUN_ID,
    }
    if PROTOCOL_VERSION >= 5 or namespace.scoped:
        # Profile IDs are immutable by convention; the canonical contract hash
        # makes that convention fail-closed for v5 prompt text and every other
        # generation field even if an ID were accidentally reused.
        profile["generation_contract_sha256"] = hashlib.sha256(
            json.dumps(
                PROTOCOL_GENERATION_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    if namespace.scoped:
        profile.update(namespace.identity)
        profile["schema_version"] = 3
    return profile


def write_checkpoint_profile(
    path: str | Path, extra: Mapping[str, Any] | None = None,
    *, namespace: CheckpointNamespace | None = None,
) -> Path:
    """Write the lineage profile, optionally with run-state fields.

    ``extra`` keys (e.g. ``lr_schedule_step``) cannot replace lineage fields.
    Legacy profiles retain their original bytes; scoped profiles use a new
    schema so a legacy reader cannot silently accept a task-bound snapshot.
    """
    destination = Path(path) / CHECKPOINT_PROFILE_NAME
    payload = active_checkpoint_profile(namespace=namespace)
    if extra:
        collisions = set(payload).intersection(extra)
        if collisions:
            raise ValueError(
                "checkpoint run-state fields cannot replace lineage fields: "
                + ", ".join(sorted(collisions))
            )
        payload.update(dict(extra))
    destination.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def validate_checkpoint_profile(
    path: str | Path,
    *,
    required: bool,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    profile_path = Path(path) / CHECKPOINT_PROFILE_NAME
    if not profile_path.exists():
        if required:
            raise CheckpointProfileMismatch(
                "checkpoint has no protocol-lineage metadata"
            )
        return None
    try:
        value = strict_json_loads(profile_path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise CheckpointProfileMismatch(
            "checkpoint protocol-lineage metadata is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointProfileMismatch(
            "checkpoint protocol-lineage metadata must be an object"
        )
    expected_value = dict(expected or active_checkpoint_profile())
    lineage_keys = [
        "schema_version",
        "profile_id",
        "protocol_version",
        "base_model_id",
        "base_model_revision",
    ]
    if int(expected_value.get("schema_version", 1)) >= 2:
        lineage_keys.append("generation_contract_sha256")
    if "checkpoint_namespace" in expected_value or "checkpoint_namespace" in value:
        lineage_keys.extend(["checkpoint_namespace", "task_id", "training_run_id"])
    for key in lineage_keys:
        if value.get(key) != expected_value.get(key):
            raise CheckpointProfileMismatch(
                f"checkpoint protocol-lineage mismatch for {key}"
            )
    return value


__all__ = [
    "CHECKPOINT_PROFILE_NAME",
    "CheckpointProfileMismatch",
    "active_checkpoint_profile",
    "validate_checkpoint_profile",
    "write_checkpoint_profile",
]
