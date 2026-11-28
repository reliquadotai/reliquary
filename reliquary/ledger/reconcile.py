"""Settled shares against the miner incentive the chain paid.

The native rail pays an archive through the weights: from its entry period
each weight-set replays it at alpha x (1 - alpha)^age of its reward
(``cp.replay``: an EMA over ``PERIOD_EMA_N`` periods, ``REPLAY_DEPTH`` deep),
and each epoch the chain pays the miners' emission by the validators' weights
(Yuma consensus). A share reaches its hotkey spread over about a day from its
entry period, not on one date.

``reconcile`` lays every archive's settled shares (``Ledger.shares``) out on
that schedule over the epochs of a window, values each epoch's at that
epoch's own miner emission (``EpochPay.pot``), and compares per hotkey with
what the chain paid in those epochs. No price or rate enters: what is checked
is that the chain paid what the archives settled.

- each epoch pays the weights set ``lag_epochs`` epochs before it
  (commit-reveal), ``lead_seconds`` before an epoch boundary
  (``EPOCH_SUBMIT_LEAD_BLOCKS``): they replayed the period holding
  ``at - lag_epochs x epoch_seconds - lead_seconds``;
- ``expected_alpha``: the sum over those epochs of each share times its EMA
  factor there, times the epoch's pot;
- ``onchain_alpha``: what the hotkey received as miner incentive in them.

Tasks the weights do not pay (the collector rail) are left out, and so are
hotkeys that are not miners (the collector). The gap is per hotkey, summed
over every task it worked on: a pass says the chain paid each hotkey what its
archives settled, not that each order did. Known sources of a gap besides a
fault: an epoch falling twice in one period (or skipping one), Yuma clipping
while validators disagree, u16 weight rounding, a task's incentive floor, a
deregistered hotkey (its share burns), pay from window tasks no archive
records, and a lag other than the configured one (``lag_fit`` measures lags
up to ``MAX_LAG_FIT``).

Pure: the epochs are read by ``sources.read_epoch_pays``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from reliquary.constants import BLOCK_TIME_SECONDS, EPOCH_SUBMIT_LEAD_BLOCKS
from reliquary.validator import corpus_periods as cp

# Weights are computed this long before the epoch boundary they are submitted for.
LEAD_SECONDS = float(EPOCH_SUBMIT_LEAD_BLOCKS * BLOCK_TIME_SECONDS)
# The reconciliation criterion: every hotkey within 1 % over the window.
TOLERANCE = 0.01
MAX_LAG_FIT = 3


@dataclass(frozen=True, slots=True)
class EpochPay:
    """One epoch of the subnet as the chain recorded it: its block and time,
    the alpha each hotkey received as miner incentive (owner hotkeys left out),
    the incentive the owner hotkeys took (burned or recycled), and the miners'
    whole emission that epoch when known."""

    block: int
    at: float
    pay: Mapping[str, float]
    burned: float = 0.0
    pot: float | None = None


def ema_factor(age: int, *, alpha: float = cp.PERIOD_ALPHA, depth: int = cp.REPLAY_DEPTH) -> float:
    """The share of an archive's reward a weight-set ``age`` periods after its
    entry pays (``cp.replay``)."""
    if age < 0 or age > depth:
        return 0.0
    return alpha * (1.0 - alpha) ** age


def weight_period(at: float, *, lag_epochs: int, epoch_seconds: float = cp.PERIOD_SECONDS,
                  lead_seconds: float = LEAD_SECONDS, genesis: float = cp.PERIOD_EPOCH) -> int:
    """The period the weights paid at an epoch at ``at`` were replayed for."""
    return cp.period_of(float(at) - lag_epochs * float(epoch_seconds) - float(lead_seconds), genesis)


def expected_alpha(shares: Iterable[tuple[str, int, str, float]], epochs: Sequence[EpochPay],
                   periods: Sequence[int], *, exclude_tasks: Iterable[str] = ()) -> dict[str, float]:
    """Per hotkey, what the weights replaying ``shares`` pay over ``epochs``
    (each at its own pot), ``periods[k]`` being the period epoch k's weights
    replayed (a period counted twice pays twice)."""
    skip = {str(task) for task in exclude_tasks}
    by_entry: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for task, entry, hotkey, share in shares:
        if str(task) not in skip:
            by_entry[int(entry)][str(hotkey)] += float(share)
    parts: dict[str, list[float]] = defaultdict(list)
    for epoch, period in zip(epochs, periods):
        if epoch.pot is None:
            continue
        for age in range(cp.REPLAY_DEPTH + 1):
            held = by_entry.get(period - age)
            if not held:
                continue
            factor = ema_factor(age) * float(epoch.pot)
            for hotkey, share in held.items():
                parts[hotkey].append(share * factor)
    return {hotkey: math.fsum(values) for hotkey, values in parts.items()}


def onchain_alpha(epochs: Iterable[EpochPay]) -> dict[str, float]:
    parts: dict[str, list[float]] = defaultdict(list)
    for epoch in epochs:
        for hotkey, alpha in epoch.pay.items():
            parts[str(hotkey)].append(float(alpha))
    return {hotkey: math.fsum(values) for hotkey, values in parts.items()}


def reconcile(shares: Iterable[tuple[str, int, str, float]], epochs: Iterable[EpochPay], *, start: float,
              end: float, lag_epochs: int = 1, lead_seconds: float = LEAD_SECONDS,
              epoch_seconds: float = cp.PERIOD_SECONDS, tolerance: float = TOLERANCE, min_alpha: float = 0.0,
              genesis: float = cp.PERIOD_EPOCH, earliest_entry: int | None = None,
              exclude_tasks: Iterable[str] = (), exclude_hotkeys: Iterable[str] = ()) -> dict:
    """Per hotkey, what the weights replaying ``shares`` should have paid in the
    epochs of [start, end) against what the chain paid in them.

    ``min_alpha``: a hotkey expected less than this is listed but not checked.
    ``earliest_entry``: the first entry period the shares cover; when the
    window reaches back past ``REPLAY_DEPTH`` periods after it, the report says
    the history is too short. ``exclude_tasks``: tasks the weights do not pay;
    ``exclude_hotkeys``: hotkeys that are not miners (their pay is reported
    apart)."""
    if lag_epochs < 0:
        raise ValueError("lag_epochs must be >= 0")
    if not 0 <= tolerance:
        raise ValueError("tolerance must be >= 0")
    shares = list(shares)
    skip_tasks = sorted({str(task) for task in exclude_tasks})
    others = {str(hotkey) for hotkey in exclude_hotkeys}
    inside = sorted((e for e in epochs if start <= float(e.at) < end), key=lambda e: (e.at, e.block))

    def periods_at(lag: int) -> list[int]:
        return [weight_period(e.at, lag_epochs=lag, epoch_seconds=epoch_seconds, lead_seconds=lead_seconds,
                              genesis=genesis) for e in inside]

    periods = periods_at(lag_epochs)
    expected = expected_alpha(shares, inside, periods, exclude_tasks=skip_tasks)
    paid = onchain_alpha(inside)
    rows = []
    for hotkey in sorted((set(expected) | set(paid)) - others):
        owed, got = expected.get(hotkey, 0.0), paid.get(hotkey, 0.0)
        gap = None if owed <= 0 else got / owed - 1.0
        checked = owed > 0 and owed >= min_alpha
        rows.append({"hotkey": hotkey, "expected_alpha": owed, "onchain_alpha": got, "gap_alpha": got - owed,
                     "gap": gap, "checked": checked, "within": (abs(gap) <= tolerance) if checked else None})
    rows.sort(key=lambda row: (-max(row["expected_alpha"], row["onchain_alpha"]), row["hotkey"]))
    checked = [row for row in rows if row["checked"]]
    outside = [row for row in checked if not row["within"]]
    total_expected = math.fsum(row["expected_alpha"] for row in rows)
    total_paid = math.fsum(row["onchain_alpha"] for row in rows)
    unscheduled = math.fsum(row["onchain_alpha"] for row in rows if row["expected_alpha"] <= 0)
    pots = [float(e.pot) for e in inside if e.pot is not None]
    burned = math.fsum(float(e.burned) for e in inside)
    history_short = (earliest_entry is not None and bool(periods)
                     and min(periods) - cp.REPLAY_DEPTH < int(earliest_entry))
    return {
        "window": {"start": float(start), "end": float(end), "epochs": len(inside),
                   "epochs_without_pot": len(inside) - len(pots),
                   "first_block": inside[0].block if inside else None,
                   "last_block": inside[-1].block if inside else None,
                   "weight_periods": [min(periods), max(periods)] if periods else None,
                   "lag_epochs": lag_epochs, "lead_seconds": float(lead_seconds),
                   "history_too_short": history_short, "excluded_tasks": skip_tasks},
        "criterion": {"tolerance": tolerance, "min_alpha": min_alpha, "hotkeys_checked": len(checked),
                      "hotkeys_within": len(checked) - len(outside),
                      "pass": bool(checked) and not outside and not history_short,
                      "scope": "per hotkey, summed over every task it worked on"},
        "totals": {"expected_alpha": total_expected, "onchain_alpha": total_paid,
                   "gap": None if total_expected <= 0 else total_paid / total_expected - 1.0,
                   "onchain_unscheduled_alpha": unscheduled,
                   "owner_burned_alpha": burned,
                   "excluded_hotkeys_alpha": {hotkey: paid.get(hotkey, 0.0) for hotkey in sorted(others)},
                   "miner_pot_alpha": math.fsum(pots) if len(pots) == len(inside) and pots else None},
        "lag_fit": lag_fit(shares, inside, paid, epoch_seconds=epoch_seconds, lead_seconds=lead_seconds,
                           genesis=genesis, exclude_tasks=skip_tasks, exclude_hotkeys=others),
        "hotkeys": rows,
    }


def lag_fit(shares: Sequence, epochs: Sequence[EpochPay], paid: Mapping[str, float] | None = None, *,
            epoch_seconds: float = cp.PERIOD_SECONDS, lead_seconds: float = LEAD_SECONDS,
            genesis: float = cp.PERIOD_EPOCH, exclude_tasks: Iterable[str] = (),
            exclude_hotkeys: Iterable[str] = ()) -> dict[str, float | None]:
    """For each lag from 0 to ``MAX_LAG_FIT`` epochs, the summed absolute gap of
    every hotkey over the expected total: the lag that fits the chain best has
    the smallest."""
    paid = onchain_alpha(epochs) if paid is None else paid
    others = {str(hotkey) for hotkey in exclude_hotkeys}
    fit: dict[str, float | None] = {}
    for lag in range(MAX_LAG_FIT + 1):
        periods = [weight_period(e.at, lag_epochs=lag, epoch_seconds=epoch_seconds, lead_seconds=lead_seconds,
                                 genesis=genesis) for e in epochs]
        expected = {hotkey: owed for hotkey, owed in
                    expected_alpha(shares, epochs, periods, exclude_tasks=exclude_tasks).items()
                    if hotkey not in others}
        total = math.fsum(expected.values())
        miss = math.fsum(abs(paid.get(hotkey, 0.0) - owed) for hotkey, owed in expected.items())
        fit[str(lag)] = None if total <= 0 else miss / total
    return fit


__all__ = [
    "EpochPay",
    "LEAD_SECONDS",
    "MAX_LAG_FIT",
    "TOLERANCE",
    "ema_factor",
    "expected_alpha",
    "lag_fit",
    "onchain_alpha",
    "reconcile",
    "weight_period",
]
