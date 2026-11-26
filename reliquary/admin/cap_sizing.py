"""Order caps that follow what an order really pays (docs/design/2026-11-11-order-cap-sizing-design.md).

The settler pays a period with the cap a task holds when that period settles, so a cap need not be reserved
ahead: it follows the order's pay pace (its open period's accrued value projected over the whole period, and
what its last periods paid) with headroom, resized every few minutes. Two bounds sit above it: what its audit
cards let through (a card audits so many tokens a second; only the audited share ``audit_q`` of the paid
records uses it) and the cap the platform asked for (``cap_max``). A new order starts small (``START_CAP``,
or less when its whole order is worth less) and rises with its pay.
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
# A new order's cap (or one paying nothing yet): its pay raises it within a resize or two.
START_CAP = 0.02
# Kept free by every raise (``admin.service.bounded_raise``): room for a new order's start, a corpus order's
# (``START_CAP``) or a training order's (``admin.training.DEFAULT_TRAINING_CAP``, 0.05; a test pins it).
NEW_ORDER_RESERVE = 0.05


def raise_room(room: float) -> float:
    """What a raise may take of the free ``room``: all but ``NEW_ORDER_RESERVE``, and in any case up to
    ``START_CAP``: an order sized down early in its first period climbs back to its own start."""
    return max(room - NEW_ORDER_RESERVE, min(START_CAP, room))
# Early in a period, its accrued value is projected over at least this share of the period: at most x4.
MIN_ELAPSED_SHARE = 0.25


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


def period_share(now: float, genesis: float = cp.PERIOD_EPOCH) -> float:
    """How much of the current period has passed, in [0, 1)."""
    return ((float(now) - float(genesis)) % cp.PERIOD_SECONDS) / cp.PERIOD_SECONDS


def order_value(params: Mapping, tokens: int) -> float | None:
    """The most a whole order can pay, in pool share: ``tokens`` at its price ceiling; None without one."""
    price = cp.max_price(params)
    return None if price is None else float(tokens) * price


def sized_cap(cap_max: float, *, estimate: float, recent_paid: Sequence[float] = (),
              open_value: float | None = None, elapsed: float = 1.0, start: float = START_CAP) -> float:
    """What the order may pay a period, never over ``cap_max``: headroom over what it paid in its last
    periods and has accrued in an open one (the settler pays that period with this cap), and over its open
    period's accrued value projected over the whole period (``elapsed`` of it has passed), or ``start``
    while it pays nothing. Only the projection and the start are bounded by what its audit lets through
    (``estimate``): what it really paid always counts."""
    accrued = open_value or 0.0
    paid = max(max(recent_paid, default=0.0), accrued)
    projected = accrued / max(MIN_ELAPSED_SHARE, elapsed)
    ahead = projected * HEADROOM if projected > 0 else (start if paid == 0 else 0.0)
    return min(float(cap_max), max(CAP_FLOOR, paid * HEADROOM, min(ahead, estimate * HEADROOM)))


def resized(params: Mapping, sizing: Mapping, status: Mapping, now: float) -> float | None:
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
                    open_value=sizing.get("open_value"), elapsed=period_share(now))
    current = float(params["cap"])
    if current > 0 and abs(cap - current) < RESIZE_MIN_CHANGE * current:
        return None
    return cap
