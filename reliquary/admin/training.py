"""Training orders: an async RL task created from the admin API (docs/design/2026-11-16-env-catalog.md, phase 3).

A training order names a generation profile (the base model, its sampling and its envs), the envs it trains on and
their shares, the trainer's batch, its limits and a price ceiling. The admin service turns it into an async RL task
priced by the RL auction (docs/design/2026-11-16-rl-auction.md): one market per env, leases kept ``lag_target``
checkpoints ahead of the trainer.

Off unless ``RELIQUARY_TRAINING_ORDERS=on``: the routes are not served, and a request answers 404. The controller
and the trainer are still started by an operator for each task (one controller per container).

The task is an ``rl-discovered-price`` async task, not a ``service-rl`` one: the async controller reads its
``env_split`` and ``rl_*`` params, and nothing async consumes a service contract yet.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

TRAINING_ORDERS_ENV = "RELIQUARY_TRAINING_ORDERS"
# The task ids a training order may create, after the admin task prefix (``order-train-...``).
TRAINING_ID_INFIX = "train-"
# An RL task's cap is at least its incentive floor (``min_incentive_share``, 0.05 by default).
DEFAULT_TRAINING_CAP = 0.05
DEFAULT_BATCH_GROUPS = 64
MAX_BATCH_GROUPS = 4096
MAX_ENVIRONMENTS = 16
# What an order's period plan assumes before the trainer reports its speed (``rl_async_params``).
DEFAULT_PLANNED_TOKENS_PER_PERIOD = 2e8
# The limits the order carries: the RL auction leases nothing once one is reached (``rl_auction.RlAuctions.stopped``).
TRAINING_PARAM = "training_order"


def training_enabled(env: Mapping[str, str] | None = None) -> bool:
    """``RELIQUARY_TRAINING_ORDERS``: ``on`` serves the training routes, empty leaves them out; anything else
    is a configuration error."""
    chosen = (os.environ if env is None else env).get(TRAINING_ORDERS_ENV, "").strip()
    if chosen not in ("", "on"):
        raise ValueError(f"{TRAINING_ORDERS_ENV} must be 'on' or empty, got {chosen!r}")
    return chosen == "on"


class TrainingEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    share_bps: int = Field(ge=1, le=10000)


class TrainingLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(ge=1, le=100_000)
    deadline_seconds: int = Field(ge=600, le=60 * 86400)
    max_tokens: int | None = Field(default=None, ge=1)


class CreateTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_id: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9][a-z0-9-]*$")
    profile_id: str = Field(min_length=1, max_length=128)
    environments: list[TrainingEnvironment] = Field(min_length=1, max_length=MAX_ENVIRONMENTS)
    limits: TrainingLimits
    max_price_usd_per_mtoken: float = Field(gt=0, le=1000)
    batch_groups: int = Field(default=DEFAULT_BATCH_GROUPS, ge=1, le=MAX_BATCH_GROUPS)
    lag_target: int = Field(default=2, ge=1, le=16)
    cap: float | None = Field(default=None, gt=0, le=1)
    # Optional private telemetry identity; the platform supplies both together.
    order_id: str | None = Field(default=None, min_length=1, max_length=128)
    wandb: dict[str, str] | None = None

    @model_validator(mode="after")
    def _shares(self) -> CreateTraining:
        names = [e.name for e in self.environments]
        if len(set(names)) != len(names):
            raise ValueError("an environment is named twice")
        if sum(e.share_bps for e in self.environments) != 10000:
            raise ValueError("environment shares must sum to 10000 bps")
        if (self.order_id is None) != (self.wandb is None):
            raise ValueError("private training telemetry requires both order_id and wandb")
        return self


def training_refusal(body: CreateTraining) -> str | None:
    """Why the order cannot train (None: it can): a profile that is not an async RL one, an env outside it, or
    an env without live RL rows in the catalog."""
    from reliquary.env_catalog import entry as catalog_entry
    from reliquary.protocol.profiles import DRAW_GUMBEL_MAX, PROFILES

    profile = PROFILES.get(body.profile_id)
    if profile is None:
        return f"unknown profile {body.profile_id!r}"
    if profile.sampling.draw != DRAW_GUMBEL_MAX:
        return f"profile {body.profile_id!r} is not drawn by Gumbel-max: async RL cannot audit it"
    for env in body.environments:
        if env.name not in profile.environments:
            return f"environment {env.name!r} is not in profile {body.profile_id!r}"
        catalog = catalog_entry(env.name)
        if catalog is None or not any(r.live for r in catalog.rows_of("rl")):
            return f"environment {env.name!r} has no live RL rows in the catalog"
    return None


def order_digest(body: CreateTraining) -> str:
    """The order as the platform sent it: a replay under the same id must carry the same order."""
    from reliquary.protocol.release_contract import canonical_sha256

    values = body.model_dump(mode="json")
    # Keep pre-telemetry requests byte-equivalent for idempotent retries of old orders.
    for name in ("order_id", "wandb"):
        if values[name] is None:
            values.pop(name)
    return canonical_sha256(values)


def training_task_id(body: CreateTraining, task_prefix: str) -> str:
    return f"{task_prefix}{TRAINING_ID_INFIX}{body.training_id}"


def training_entry(body: CreateTraining, *, task_prefix: str, price_params: Mapping[str, Any],
                   created_at: float):
    """The registry entry of a training order (``price_params``: the ceiling in the task's alpha, as an order's,
    with the USD rate it was converted at). ValueError on an order the registry would refuse."""
    from dataclasses import replace

    from reliquary.cli.main import build_task_entry, with_rl_async_params
    from reliquary.corpus.token_auction import PRICING_AUCTION, PRICING_PARAM
    from reliquary.validator.rl_async_service import BATCH_GROUPS_PARAM
    from reliquary.validator.rl_auction import LAG_TARGET_PARAM

    refusal = training_refusal(body)
    if refusal is not None:
        raise ValueError(refusal)
    from reliquary.protocol.profiles import PROFILES

    shares = {e.name: e.share_bps / 10000 for e in body.environments}
    split = {name: shares.get(name, 0.0) for name in PROFILES[body.profile_id].environments}
    cap = DEFAULT_TRAINING_CAP if body.cap is None else float(body.cap)
    entry = build_task_entry(task_id=training_task_id(body, task_prefix), profile_id=body.profile_id, cap=cap,
                             overrides={}, env_split=split)
    entry = with_rl_async_params(entry, planned_tokens_per_period=DEFAULT_PLANNED_TOKENS_PER_PERIOD)
    from reliquary.validator.corpus_periods import MAX_PRICE_ALPHA_USD_PARAM

    if price_params.get(MAX_PRICE_ALPHA_USD_PARAM) is None:
        raise ValueError("the order's price was not converted from USD")
    # The alpha/USD rate its ceiling was converted at: the bill converts the miners' alpha back at it.
    order = {"training_id": body.training_id, "order_sha256": order_digest(body), "max_steps": body.limits.max_steps,
             "deadline_at": float(created_at) + body.limits.deadline_seconds,
             "max_price_usd_per_mtoken": float(body.max_price_usd_per_mtoken),
             "usd_per_alpha": float(price_params[MAX_PRICE_ALPHA_USD_PARAM])}
    if body.limits.max_tokens is not None:
        order["max_tokens"] = body.limits.max_tokens
    if body.order_id is not None:
        from reliquary.shared.private_training_wandb import validate_private_wandb_binding

        binding = validate_private_wandb_binding(body.order_id, entry.task_id, body.wandb)
        order.update(order_id=body.order_id, wandb=binding)
    return replace(entry, params={**entry.params, **price_params, PRICING_PARAM: PRICING_AUCTION,
                                  LAG_TARGET_PARAM: body.lag_target, BATCH_GROUPS_PARAM: body.batch_groups,
                                  TRAINING_PARAM: order})


def training_spend(entry, summary: Mapping[str, Any], *, retired_time: float | None) -> dict:
    """The spend route's answer: the controller's summary, the order's frozen rate, and ``final``: the task is
    retired and its controller has stopped the order, nothing pending can still pay, every period is settled, and
    the summary was written after the period holding the later of the retire (``retired_time``) and the order's
    stop had closed and settled (``settle_margin_s`` past its end, and the ledger's settle had closed it:
    ``settled_through``), so no later accrual or fill can appear."""
    from reliquary.validator import corpus_periods as cp

    order = entry.params.get(TRAINING_PARAM) or {}
    final = False
    stopped_at, settled_through = summary.get("stopped_at"), summary.get("settled_through")
    if (entry.status == "retired" and retired_time is not None and summary.get("stopped")
            and stopped_at is not None and settled_through is not None):
        ended = max(float(retired_time), float(stopped_at))
        closed = cp.period_end(cp.period_of(ended)) + float(summary.get("settle_margin_s", 0.0))
        pending = summary.get("pending")
        final = (isinstance(pending, Mapping) and not any(int(v) for v in pending.values())
                 and all(p.get("settled") is True for p in summary.get("periods", []))
                 and int(settled_through) >= cp.period_of(ended)
                 and float(summary.get("written_at", 0.0)) > closed)
    return {**summary, "usd_per_alpha": order.get("usd_per_alpha"), "final": final}


def training_view(entry) -> dict:
    """What the platform reads of a training task."""
    params = entry.params
    return {"task_id": entry.task_id, "status": entry.status, "profile_id": entry.profile_id,
            "cap": float(params.get("cap", 0.0)), "retired_at": entry.retired_at,
            "environments": {k: v for k, v in (entry.env_split or {}).items() if v > 0},
            "batch_groups": params.get("rl_batch_groups"), "lag_target": params.get("rl_auction_lag_target"),
            "order": params.get(TRAINING_PARAM)}


__all__ = [
    "TRAINING_ORDERS_ENV",
    "CreateTraining",
    "order_digest",
    "training_enabled",
    "training_entry",
    "training_profiles",
    "training_refusal",
    "training_spend",
    "training_task_id",
    "training_view",
]


def training_profiles(*, task_prefix: str | None = None) -> dict:
    """Exact pinned async RL profiles and their currently live training environments."""
    from reliquary.env_catalog import entry as catalog_entry
    from reliquary.protocol.profiles import DRAW_GUMBEL_MAX, PROFILES
    from reliquary.shared.private_training_wandb import PRIVATE_TRAINING_WANDB_SCHEMA

    profiles = []
    for profile_id, profile in PROFILES.items():
        if profile.sampling.draw != DRAW_GUMBEL_MAX:
            continue
        environments = [name for name in profile.environments
                        if (entry := catalog_entry(name)) is not None and any(r.live for r in entry.rows_of("rl"))]
        if environments:
            profiles.append({"id": profile_id, "model": profile.model_id, "revision": profile.model_revision,
                             "environments": environments})
    return {"schema": "reliquary/training-profiles/v1", "profiles": profiles,
            "capabilities": {"private_training_wandb": PRIVATE_TRAINING_WANDB_SCHEMA},
            **({"training_task_prefix": task_prefix + TRAINING_ID_INFIX} if task_prefix is not None else {})}
