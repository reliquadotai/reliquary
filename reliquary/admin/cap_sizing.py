"""Order caps sized to what an order can pay (docs/design/2026-11-11-order-cap-sizing-design.md).

An order job pays at most its price ceiling times the tokens its audit cards let through: a card audits so
many tokens a second, and only the audited share ``audit_q`` of the paid records uses it. Its cap is that
bound with headroom, under the cap the platform asked for (``cap_max``), so the registry's sum of caps
counts what orders can pay, not a fixed share each.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from reliquary.validator import corpus_periods as cp

# ``on``: order jobs with a price ceiling get a sized cap at creation, resized every RESIZE_SECONDS.
SIZING_ENV = "RELIQUARY_ORDER_CAP_SIZING"
# The cap the order asked for: the most a resize may give it.
CAP_MAX_PARAM = "cap_max"
HEADROOM = 1.5
CAP_FLOOR = 0.002
# Audit tokens per second one card scores, for a model not measured yet: high on purpose, so an unmeasured
# order gets a large cap (at a Batch audit share it reaches the requested cap, as before sizing).
DEFAULT_COMPLETION_TOKENS_PER_SECOND = 2000.0
DEFAULT_PROMPT_TOKENS_PER_SECOND = 2000.0
# The control's status is older than this: no resize on it.
STATUS_MAX_AGE_SECONDS = 600.0
RESIZE_SECONDS = 300.0
# A resize smaller than this share of the current cap is not written.
RESIZE_MIN_CHANGE = 0.1


def sizing_enabled(env: Mapping[str, str] | None = None) -> bool:
    chosen = (os.environ if env is None else env).get(SIZING_ENV, "").strip()
    if chosen not in ("", "on"):
        raise ValueError(f"{SIZING_ENV} must be 'on' or unset, got {chosen!r}")
    return chosen == "on"


def model_rates(status: Mapping | None, model: str) -> tuple[float, float]:
    """Completion and prompt tokens one card of ``model`` (``repo@revision``) audits per second, from the
    control's status; the defaults while unmeasured."""
    entry = ((status or {}).get("models") or {}).get(model) or {}
    completion = entry.get("audit_tokens_per_second")
    prompt = entry.get("audit_prompt_tokens_per_second")
    if completion is None or prompt is None:
        return DEFAULT_COMPLETION_TOKENS_PER_SECOND, DEFAULT_PROMPT_TOKENS_PER_SECOND
    return float(completion), float(prompt)


def spend_estimate(params: Mapping, *, completion_tps: float, prompt_tps: float, cards: int) -> float | None:
    """The most an order job may pay a period, in pool share: its price ceiling on what ``cards`` cards
    let through at the audited share. None without a ceiling."""
    price, input_price = cp.max_price(params), cp.max_input_price(params)
    if price is None:
        return None
    audit_q = float(params.get("audit_q") or 1.0)
    per_second = completion_tps * price + prompt_tps * (input_price or 0.0)
    return per_second * cp.PERIOD_SECONDS * max(1, int(cards)) / audit_q


def sized_cap(cap_max: float, *, estimate: float, recent_paid: Sequence[float] = (),
              open_value: float | None = None) -> float:
    """What the order may pay a period, with headroom: never under what it paid lately or has accrued in
    a period still open, never over ``cap_max``."""
    need = max(estimate, max(recent_paid, default=0.0), open_value or 0.0) * HEADROOM
    return min(float(cap_max), max(CAP_FLOOR, need))


def resized(params: Mapping, sizing: Mapping, status: Mapping) -> float | None:
    """A job's new cap from the control's ``sizing`` entry, or None when it should stay (no ceiling, no
    requested cap, or a change under RESIZE_MIN_CHANGE)."""
    cap_max = params.get(CAP_MAX_PARAM)
    if cap_max is None:
        return None
    completion, prompt = model_rates(status, str(sizing.get("model")))
    estimate = spend_estimate(params, completion_tps=completion, prompt_tps=prompt,
                              cards=int(sizing.get("audit_executors") or 1))
    if estimate is None:
        return None
    cap = sized_cap(float(cap_max), estimate=estimate, recent_paid=sizing.get("recent_paid") or (),
                    open_value=sizing.get("open_value"))
    current = float(params["cap"])
    if current > 0 and abs(cap - current) < RESIZE_MIN_CHANGE * current:
        return None
    return cap
