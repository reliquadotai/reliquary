"""Dependency-free identity contract for one private training order and task."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

PRIVATE_TRAINING_WANDB_SCHEMA = "reliquary/private-training-wandb/v1"
_ORDER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TASK = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def private_wandb_run_id(order_id: str, task_id: str) -> str:
    if not isinstance(order_id, str) or not _ORDER.fullmatch(order_id):
        raise ValueError("invalid private training order identity")
    if not isinstance(task_id, str) or not _TASK.fullmatch(task_id):
        raise ValueError("invalid private training task identity")
    body = f"{PRIVATE_TRAINING_WANDB_SCHEMA}\n{order_id}\n{task_id}".encode("ascii")
    return "rl-" + hashlib.sha256(body).hexdigest()[:40]


def validate_private_wandb_binding(order_id: str, task_id: str, binding: Mapping) -> dict[str, str]:
    run_id = private_wandb_run_id(order_id, task_id)
    if not isinstance(binding, Mapping) or set(binding) != {"entity", "project", "run_id"}:
        raise ValueError("invalid private training W&B binding")
    if any(not isinstance(binding[k], str) or not _SLUG.fullmatch(binding[k])
           for k in ("entity", "project")) or binding["run_id"] != run_id:
        raise ValueError("invalid private training W&B binding")
    return {key: binding[key] for key in ("entity", "project", "run_id")}


def private_wandb_from_environment(config: Mapping, role: str | None,
                                  environ: Mapping[str, str]) -> dict[str, str]:
    """Require the authenticated task config and pod environment to identify the same run."""
    if environ.get("RELIQUARY_TRAINING_WANDB") != "on" or role != "trainer":
        raise ValueError("private training W&B is not configured for this role")
    if (environ.get("WANDB_MODE", "online") != "online"
            or environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/") != "https://api.wandb.ai"):
        raise ValueError("private training W&B requires the qualified online source")
    order_id, task_id = environ.get("RELIQUARY_ORDER_ID"), environ.get("RELIQUARY_TASK_ID")
    run_id = private_wandb_run_id(order_id, task_id)
    if not isinstance(config, Mapping) or config.get("order_id") != order_id or config.get("task_id") != task_id:
        raise ValueError("private training identity does not match task config")
    binding = validate_private_wandb_binding(order_id, task_id, config.get("wandb"))
    if (environ.get("WANDB_ENTITY") != binding["entity"]
            or environ.get("WANDB_PROJECT") != binding["project"]
            or environ.get("WANDB_RUN_ID", run_id) != run_id):
        raise ValueError("private training W&B binding does not match task config")
    return {"order_id": order_id, "task_id": task_id, **binding}
