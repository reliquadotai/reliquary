"""Corpus tasks paid on their own clock (design 2026-10-03-sft-period-clock).

A task declared with ``params["settlement"] == "period-ema-v1"`` is paid by
period of drand time, not by RL window:

- a token belongs to the period its submission was received in;
- a period is settled once nothing received in it is still undecided, and its
  pay enters the weights from the first period after its settlement with room
  for it (its entry period): at most ``CATCHUP_ENTRIES`` archives enter in one
  period, so a backlog settled at once is paid back within a few periods
  instead of queueing one per period behind every later one;
- the weights replay an EMA that decays every period, a period with no pay
  counting as zero, so each hotkey is paid in total what its tokens earned and
  a finished task stops paying on its own.

Everything here is pure: the clock is passed in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

SETTLEMENT_PERIOD_EMA = "period-ema-v1"
# Where period 0 starts: drand quicknet's genesis. A protocol constant rather
# than a fetched value, so the settler and every weight setter count the same
# periods with no network (any fixed origin would do; it must only be shared).
PERIOD_EPOCH = 1692803367.0
# One Bittensor epoch: the rate weights are set at.
PERIOD_SECONDS = 4320
PERIOD_EMA_N = 6
PERIOD_ALPHA = 2.0 / (PERIOD_EMA_N + 1)
# Periods replayed back from the current one: (1 - alpha)^24 ~ 0.03 % never paid.
REPLAY_DEPTH = 24
# Archives that may enter in one period. One per period is the steady state;
# more is a backlog being paid back. The weight setter bounds a task's replay at
# this many caps (each archive holding at most one cap), so it must not change
# without every weight setter first.
CATCHUP_ENTRIES = 4


# The highest cap a period task was paid at, written to its entry's params when
# its cap is lowered (``task_registry.set_cap``). Its archives keep paying up
# to that bound after the cap falls, so a cap only governs work still to come.
TAIL_CAP_PARAM = "tail_cap"


def pay_ceiling(params) -> float | None:
    """The most one of a period task's archives may pay: its cap, or the
    highest cap it was paid at if that is higher. None without a cap."""
    if not isinstance(params, Mapping) or params.get("cap") is None:
        return None
    ceiling = float(params["cap"])
    tail = params.get(TAIL_CAP_PARAM)
    if tail is not None:
        ceiling = max(ceiling, float(tail))
    return ceiling


def archive_bound(recorded, ceiling: float | None) -> float | None:
    """What one archive may pay: the cap it was settled under (``recorded``,
    absent from archives written before it was kept), never past the task's
    ``pay_ceiling``."""
    if recorded is None:
        return ceiling
    if ceiling is None:
        return float(recorded)
    return min(float(recorded), ceiling)


def is_period_task(entry) -> bool:
    params = getattr(entry, "params", None)
    return isinstance(params, Mapping) and params.get("settlement") == SETTLEMENT_PERIOD_EMA


def period_of(t: float, genesis: float = PERIOD_EPOCH) -> int:
    """The period holding instant ``t`` (seconds), counted from drand genesis."""
    return math.floor((float(t) - float(genesis)) / PERIOD_SECONDS)


def period_end(period: int, genesis: float = PERIOD_EPOCH) -> float:
    return float(genesis) + (int(period) + 1) * PERIOD_SECONDS


def closed_through(*, now: float, oldest_pending: float | None, genesis: float,
                   slack: float) -> int:
    """The last period nothing can still be added to.

    A period is closed when it ended before the oldest submission still
    undecided was received, and long enough ago (``slack``) that a submission
    received at its very end has reached the store."""
    bound = float(now) - float(slack)
    if oldest_pending is not None:
        bound = min(bound, float(oldest_pending))
    # The period holding `bound` may still gain work; the one before cannot.
    return period_of(bound, genesis) - 1


def entry_for(due: int, entered: Iterable[int], *, used: Iterable[int] = (),
              per_period: int = CATCHUP_ENTRIES) -> int:
    """The entry period of a new archive: the first from ``due`` on that holds
    fewer than ``per_period`` of the task's archives (``entered``, the entry
    periods already taken) and is not in ``used`` (its own work period's
    entries: one archive per work and entry period)."""
    taken: dict[int, int] = {}
    for entry in entered:
        if int(entry) >= due:
            taken[int(entry)] = taken.get(int(entry), 0) + 1
    used = {int(entry) for entry in used}
    entry = int(due)
    while taken.get(entry, 0) >= per_period or entry in used:
        entry += 1
    return entry


def replay(archives: Iterable[Mapping], current_period: int, *,
           alpha: float = PERIOD_ALPHA, depth: int = REPLAY_DEPTH) -> dict[str, float]:
    """One task's weights at ``current_period``: every archive's rewards, from
    its entry period on, decayed once per period whether or not anything was paid
    since. Archives that have not entered yet, or entered more than ``depth``
    periods ago, add nothing."""
    weights: dict[str, float] = {}
    for archive in archives:
        age = int(current_period) - int(archive["entry_period"])
        if age < 0 or age > depth:
            continue
        factor = alpha * (1.0 - alpha) ** age
        for hotkey, reward in (archive.get("rewards_by_hotkey") or {}).items():
            weights[hotkey] = weights.get(hotkey, 0.0) + factor * float(reward)
    return {hk: v for hk, v in weights.items() if v > 1e-9}


__all__ = [
    "CATCHUP_ENTRIES",
    "PERIOD_ALPHA",
    "PERIOD_EPOCH",
    "PERIOD_EMA_N",
    "PERIOD_SECONDS",
    "REPLAY_DEPTH",
    "SETTLEMENT_PERIOD_EMA",
    "TAIL_CAP_PARAM",
    "archive_bound",
    "closed_through",
    "entry_for",
    "is_period_task",
    "pay_ceiling",
    "period_end",
    "period_of",
    "replay",
]
