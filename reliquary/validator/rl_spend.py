"""What a training task paid its miners, for the platform's bill (RL training orders, settlement).

``spend_summary`` reads the task's ledger (``rl_ledger``): per period, whether it is settled, its pay before the
cap (``accrued_alpha``: the archive's ``accrued_total`` once settled, the sum of its live accruals before) and after
it (``paid_alpha``, None until settled), and its accruals grouped by lease price per env (``buckets``: price in alpha
per token, token equivalent = amount / price, so a first scan counts at its share and a debit is subtracted at the
accrual's price). Forfeited accruals are left out. Then the trainer's starved spans (``rl_buffer.starved_intervals``),
the order's stop (``RlAuctions.stopped``, and when: ``RlPayLedger.stopped_at``), the latest checkpoint, and what
could still change the pay (``pending``): leases still open, groups buffered for the trainer and audits queued
(both left out once the order stopped and the trainer is silent past the ledger's outage grace, counted from the
stop when it never called: nothing will take them), payable events the ledger has not ingested.
``settle_margin_s``: how long after a period's end it settles; ``settled_through``: the last period the ledger's
latest settle closed (None before one). Accruals are never negative (a debit takes at most the accrual), so the archive's ``scale`` is exactly
``paid_alpha / accrued_alpha``, both summed over the same hotkeys. No hotkey is written.

Amounts convert from pool share to alpha at the miner emission per period the task's ceiling was declared with
(``max_price_miner_alpha_per_period``, frozen at creation). The controller writes it to ``rl/<run>/spend.json``
in the public bucket (``SpendPublisher``); the admin serves it (``GET /admin/v1/training/{task_id}/spend``).
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

SPEND_SCHEMA = "reliquary/rl-spend/v1"
SPEND_EVERY_S = 300.0            # without a new settle, the summary is rewritten at most this often
SPEND_CACHE = "no-store"


def spend_key(run: str) -> str:
    from reliquary.shared.ckpt_index import run_prefix

    return f"{run_prefix(run)}/spend.json"


def _buckets(rows, emission: float) -> dict[str, list[list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = {}
    for environment, price, tokens, amount in rows:
        price = float(price)
        equivalent = float(amount) / price if price > 0 else float(tokens)
        grouped.setdefault(environment, {}).setdefault(price, []).append(equivalent)
    return {env: [[price * emission, math.fsum(parts)] for price, parts in sorted(by_price.items())]
            for env, by_price in sorted(grouped.items())}


def _count(db, sql: str, args=()) -> int:
    import sqlite3

    try:
        return int(db.execute(sql, args).fetchone()[0])
    except sqlite3.OperationalError:              # a table its component never created
        return 0


def spend_summary(rl, *, now: float) -> dict:
    """The task's spend (module docstring). ValueError on a task not priced by auction (no alpha rate)."""
    from reliquary.validator.rl_buffer import dead_intervals, starved_intervals
    from reliquary.validator.rl_ledger import FORFEITED
    from reliquary.validator.rl_submit import FIRST_SCAN, TRAINED

    book, ledger = rl.book, rl.ledger
    auctions = getattr(book, "auctions", None)
    if auctions is None:
        raise ValueError("a spend summary needs a task priced by auction")
    emission = float(auctions.params.miner_alpha_per_period)
    with book.lock:
        db = book.db
        rows = db.execute("SELECT period, environment, token_price, matched_tokens, amount FROM rl_accruals "
                          "WHERE status<>? ORDER BY seq", (FORFEITED,)).fetchall()
        archives = {int(p): json.loads(d) for p, d in db.execute("SELECT work_period, doc FROM rl_pay_periods")}
        starved = starved_intervals(db, now=now)
        pending = {"open_leases": _count(db, "SELECT COUNT(*) FROM rl_leases WHERE state='open'"),
                   "buffered": _count(db, "SELECT COUNT(*) FROM rl_buffer"),
                   "audits": _count(db, "SELECT COUNT(*) FROM rl_audits WHERE body IS NOT NULL"),
                   "unledgered": _count(db, "SELECT COUNT(*) FROM rl_events WHERE seq > (SELECT seq FROM "
                                            "rl_pay_cursor WHERE id=0) AND kind IN (?, ?)", (TRAINED, FIRST_SCAN))}
        beat = db.execute("SELECT last_at FROM rl_trainer_beat WHERE id=0").fetchone()
    stopped_at = ledger.stopped_at(now)
    with book.lock:
        dead = dead_intervals(book.db, now=now if stopped_at is None else min(now, float(stopped_at)))
    config = ledger.config
    silent_since = None if stopped_at is None else (float(stopped_at) if beat is None else float(beat[0]))
    if silent_since is not None and now - silent_since > config.outage_grace_s:
        pending["buffered"] = pending["audits"] = 0     # the order stopped and its trainer is gone
    by_period: dict[int, list] = {}
    for period, environment, price, tokens, amount in rows:
        by_period.setdefault(int(period), []).append((environment, price, tokens, amount))
    periods = []
    for period in sorted(set(by_period) | set(archives)):
        own = by_period.get(period, [])
        doc = archives.get(period)
        if doc is None:
            accrued, paid = math.fsum(float(r[3]) for r in own), None
        else:
            accrued = math.fsum(float(v) for v in doc["accrued_by_hotkey"].values() if float(v) > 0)
            paid = math.fsum(float(v) for v in doc["rewards_by_hotkey"].values()) * emission
        periods.append({"period": period, "settled": doc is not None, "accrued_alpha": accrued * emission,
                        "paid_alpha": paid, "buckets": _buckets(own, emission)})
    return {"schema": SPEND_SCHEMA, "task_id": rl.settings.task_id, "run": rl.settings.run, "periods": periods,
            "starved": [[start, end] for start, end in starved],
            "dead": [[start, end] for start, end in dead], "stopped": auctions.stopped(),
            "stopped_at": stopped_at, "latest_checkpoint": book.latest_checkpoint, "pending": pending,
            "settled_through": getattr(ledger, "settled_through", None),
            "settle_margin_s": (max(config.slack_s, config.outage_grace_s) if config.outage_fill
                                else config.slack_s),
            "miner_alpha_per_period": emission, "written_at": float(now)}


def read_spend(run: str) -> dict | None:
    """The summary the controller last wrote for ``run`` (None: none yet), read with the public bucket's keys."""
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client

    client = public_checkpoint_client()
    try:
        body = client.get_object(Bucket=public_checkpoint_bucket(), Key=spend_key(run))["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(body)


class SpendPublisher:
    """Writes the summary after each settle, and at most every ``SPEND_EVERY_S`` otherwise.
    ``put(key, body, content_type, cache_control)``: the public bucket's (``rl_observations.bucket_io``)."""

    def __init__(self, rl, put: Callable[..., Awaitable[None]], *, clock: Callable[[], float] = time.time,
                 every_s: float = SPEND_EVERY_S) -> None:
        self.rl, self.put, self.clock, self.every_s = rl, put, clock, every_s
        self._written_at: float | None = None

    async def write(self, *, settled: bool) -> bool:
        import asyncio

        now = float(self.clock())
        if not settled and self._written_at is not None and now - self._written_at < self.every_s:
            return False
        summary = await asyncio.to_thread(spend_summary, self.rl, now=now)
        await self.put(spend_key(self.rl.settings.run), json.dumps(summary, sort_keys=True).encode(),
                       "application/json", SPEND_CACHE)
        self._written_at = now
        return True


__all__ = ["SPEND_SCHEMA", "SpendPublisher", "read_spend", "spend_key", "spend_summary"]
