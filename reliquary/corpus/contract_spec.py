"""Frozen declaration for a new order. Observation uses the ordinary auction, unchanged."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from fractions import Fraction

from reliquary.corpus.token_auction import market_key
from reliquary.validator import corpus_periods as cp

MARKETS_ENV = "RELIQUARY_ORDER_CONTRACT_MARKETS"
PARAM = "order_contract"
PILOT_UNAVAILABLE = "live grants and native settlement must enforce the same token reservation and frozen lease price"
OPEN_FIELDS = (
    "opened_at",
    "duration_s",
    "target_rate",
    "max_tokens",
    "budget_rao",
    "ceiling_micro",
    "new_rate_cap",
    "audit_cost_micro",
    "evaluation",
    "q_min",
    "trust_tokens",
)
_CONFIG_FIELDS = {"mode", "duration_s", "target_rate", "new_rate_cap", "audit_cost_micro", "q_min", "trust_tokens"}


def _positive(value, name, minimum=1):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63)")
    return value


def _ratio(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative price or cap")
    return Fraction(value)


def period_envelope(params):
    """Whole-order rao that fit even if every delivery falls in one unchanged work period.

    Each order task owns a cap in the checked registry. The settler also applies
    an output-token ceiling; round the bid ceiling down using both of its stored
    representations. This values the accounting obligation at the frozen rate,
    not a guarantee of a realized transfer on a particular rail.
    """
    emission = _ratio(params[cp.MAX_PRICE_EMISSION_PARAM])
    cap = _ratio(params["cap"])
    share_price = _ratio(params[cp.MAX_PRICE_PARAM])
    alpha_price = _ratio(params[cp.MAX_PRICE_ALPHA_PARAM])
    return (int(cap * emission * 10**9), int(min(alpha_price * 10**6, share_price * emission * 10**12)))


def open_kwargs(spec):
    """Arguments shared by a historical replay and the opening worker."""
    return {key: spec[key] for key in OPEN_FIELDS}


def validate_spec(spec):
    required = {"version", "mode", "market", *OPEN_FIELDS}
    if not isinstance(spec, Mapping) or set(spec) not in (required, required | {"opening_valuation"}):
        raise ValueError("invalid fields in order_contract")
    if "opening_valuation" in spec:
        valuation = spec["opening_valuation"]
        if not isinstance(valuation, Mapping) or set(valuation) != {cp.MAX_PRICE_PARAM, cp.MAX_PRICE_EMISSION_PARAM}:
            raise ValueError("invalid opening valuation")
        for value in valuation.values():
            _ratio(value)
    if type(spec["version"]) is not int or spec["version"] != 1:
        raise ValueError("unknown order_contract version")
    if spec["mode"] not in ("observation", "pilot"):
        raise ValueError("unknown order_contract mode")
    if not isinstance(spec["market"], str) or not spec["market"].endswith("|turn") or not spec["market"][:-5]:
        raise ValueError("order_contract requires a single-turn model market")
    if type(spec["evaluation"]) is not bool:
        raise ValueError("evaluation must be boolean")
    q = spec["q_min"]
    if type(q) not in (float, int) or not 0 < q <= 1:
        raise ValueError("q_min must lie in (0,1]")
    for field in ("opened_at", "budget_rao", "ceiling_micro", "audit_cost_micro"):
        _positive(spec[field], field, 0)
    for field in ("duration_s", "target_rate", "new_rate_cap", "max_tokens", "trust_tokens"):
        _positive(spec[field], field)
    _positive(spec["opened_at"] + spec["duration_s"], "deadline")


def declaration(config, job, params, *, opened_at, evaluation):
    """Called only when a new, priced registry task is created. Never changes a quote."""
    if not config:
        return None
    markets = json.loads(config)
    if not isinstance(markets, dict):
        raise TypeError("order contract markets must be an object")
    market = market_key(job)
    if market not in markets:
        return None
    settings = markets[market]
    if (
        not isinstance(settings, dict)
        or set(settings) - _CONFIG_FIELDS
        or not {"mode", "duration_s", "target_rate", "new_rate_cap"} <= settings.keys()
    ):
        raise ValueError("invalid order contract market settings")
    if params.get("pricing") != "auction" or job.sampling.draw is None or job.episode is not None:
        raise ValueError("order contract requires a single-turn forced-draw auction")
    budget, ceiling = period_envelope(params)
    max_tokens = job.prompt_count * job.slots_per_prompt * job.sampling.n * job.sampling.max_new_tokens
    spec = {
        "version": 1,
        "market": market,
        "opened_at": opened_at,
        "max_tokens": max_tokens,
        "budget_rao": budget,
        "opening_valuation": {key: params[key] for key in (cp.MAX_PRICE_PARAM, cp.MAX_PRICE_EMISSION_PARAM)},
        "ceiling_micro": ceiling,
        "evaluation": evaluation,
        "audit_cost_micro": 0,
        "q_min": 0.01,
        "trust_tokens": 10_000_000,
        **settings,
    }
    validate_spec(spec)
    if spec["mode"] == "pilot":
        raise ValueError(PILOT_UNAVAILABLE)
    return spec
