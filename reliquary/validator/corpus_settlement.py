"""What every corpus settler shares: how a period's cap is split by verified
tokens, how the auditor feeds the settler, and the guard that keeps a
validator from archiving pay for a task it does not serve.

Corpus tasks are paid by period only (``corpus_period_settlement``, design
2026-10-03). The window settler that paid them by RL window index is gone: the
window archives it wrote stay in the bucket, and the weight setter no longer
reads them for a corpus task.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

# A fed settler learns new verdicts from the auditor; the store is listed at
# boot and this often, as the net for any verdict the feed did not report.
SETTLE_FULL_LIST_SECONDS = float(os.environ.get("RELIQUARY_CORPUS_SETTLE_FULL_LIST_SECONDS", "1800"))
# Verdict reads in flight at once for the verdicts the auditor did not feed.
VERDICT_READ_CONCURRENCY = 16


def period_price(verdicts: Iterable[Mapping], earlier: float | None = None) -> float | None:
    """The one price a period priced by auction pays every token at: the highest lease price delivered in it
    (``earlier``: the highest of the period's verdicts already settled)."""
    prices = [float(v["lease_price"]) for v in verdicts if v.get("passed") and v.get("lease_price") is not None]
    if earlier is not None:
        prices.append(float(earlier))
    return max(prices) if prices else None


def rewards_for(verdicts: Iterable[Mapping], cap: float, *, price: float | None = None,
                input_price: float | None = None, uniform_price: float | None = None) -> dict[str, float]:
    """What each hotkey earns of ``cap`` (the period's pool): its share of the passed tokens, or, on a job priced
    by auction (verdicts carrying ``lease_price``), its tokens at the period's price (``uniform_price``, else the
    highest lease price among them) less the debt they carried, scaled down only if that passes the pool (the
    rest burns). With an ``input_price`` (and the output ``price``), the share is of the passed work's value:
    output tokens x price + prompt tokens x input_price."""
    verdicts = list(verdicts)
    if any(v.get("lease_price") is not None for v in verdicts if v.get("passed")):
        return _priced_rewards(verdicts, cap, period_price(verdicts, uniform_price))
    if input_price is not None and price is not None:
        from reliquary.validator.corpus_periods import work_value

        values: dict[str, float] = {}
        for verdict in verdicts:
            if verdict.get("passed"):
                values[verdict["hotkey"]] = (values.get(verdict["hotkey"], 0.0)
                                             + work_value(verdict, price, input_price))
        whole = sum(values.values())
        if whole <= 0:
            return {}
        return {hotkey: cap * value / whole for hotkey, value in values.items()}
    tokens: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.get("passed"):
            tokens[verdict["hotkey"]] = tokens.get(verdict["hotkey"], 0) + int(verdict["token_count"])
    total = sum(tokens.values())
    if total <= 0:
        return {}
    return {hotkey: cap * count / total for hotkey, count in tokens.items()}


def _priced_rewards(verdicts: list[Mapping], cap: float, price: float) -> dict[str, float]:
    earned: dict[str, float] = {}
    for verdict in verdicts:
        if not verdict.get("passed") or verdict.get("lease_price") is None:
            continue
        value = int(verdict["token_count"]) * float(price) - float(verdict.get("lease_debit") or 0.0)
        earned[verdict["hotkey"]] = earned.get(verdict["hotkey"], 0.0) + value
    earned = {hotkey: value for hotkey, value in earned.items() if value > 0}
    total = sum(earned.values())
    if total <= 0:
        return {}
    scale = min(1.0, float(cap) / total) if cap > 0 else 0.0
    return {hotkey: value * scale for hotkey, value in earned.items() if value * scale > 0}


def _union(settled, ids) -> list[str]:
    return sorted(set(settled or []) | set(ids))


def settler_fed(settler, on_verdict=None):
    """The auditor's ``on_verdict``: the settler hears of the verdict first,
    so a failing status hook can never keep it from being paid on time."""

    def report(submission_id: str, verdict) -> None:
        settler.observe(submission_id, verdict)
        if on_verdict is not None:
            on_verdict(submission_id, verdict)

    return report


class R2Archives:
    """The guard on a corpus validator's period archives
    (``R2PeriodArchives(guard=...)``): it refuses a task this process does not
    serve.

    ``served`` names the tasks this process wired after boot, which
    ``RELIQUARY_TASK_ID`` cannot list. ``served_only`` uses that reader as the
    complete ownership set, including tasks wired at boot, without inheriting
    another control's environment task list.
    """

    def __init__(self, *, served=None, served_only: bool = False) -> None:
        if served_only and served is None:
            raise ValueError("explicit archive ownership requires a served task reader")
        self._served = served
        self._served_only = served_only

    def refuse_unserved(self, task_id: str) -> None:
        """The corpus validator runs under its own task id(s), so it refuses to
        write under any task RELIQUARY_TASK_ID (or its hot set) does not name.
        Unset is refused as before, never read as the legacy task."""
        from reliquary.shared.task_id import parse_task_ids

        served = os.getenv("RELIQUARY_TASK_ID")
        hot = set(self._served()) if self._served is not None else set()
        if self._served_only:
            if task_id not in hot:
                raise RuntimeError(f"this process does not serve {task_id!r}; refusing to archive")
            return
        if not served or (task_id not in parse_task_ids(served) and task_id not in hot):
            raise RuntimeError(f"RELIQUARY_TASK_ID does not name {task_id!r}; refusing to archive")


__all__ = [
    "R2Archives",
    "SETTLE_FULL_LIST_SECONDS",
    "VERDICT_READ_CONCURRENCY",
    "rewards_for",
    "settler_fed",
]
