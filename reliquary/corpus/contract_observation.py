"""Local comparison data for an explicitly marked order. Never used to set live pay."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import closing
from functools import cached_property

from reliquary.corpus.contract_spec import (
    MARKETS_ENV,
    PARAM,
    open_kwargs,
    validate_spec,
)
from reliquary.corpus.order_contracts import ContractBook
from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM, MAX_PRICE_PARAM

DB_ENV = "RELIQUARY_ORDER_OBSERVATION_DB"


class ContractObservation:
    def __init__(self, path, markets):
        self._path = path
        self._close_cursor = ""
        self.markets = frozenset(markets)

    @cached_property
    def book(self):
        # Called from the observation worker threads. A busy local book must never stall
        # validator startup or its registry refresh; the next call can retry initialization.
        book = ContractBook(self._path, timeout=0)
        with closing(sqlite3.connect(book.path, timeout=0)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS observed_periods (
                order_id TEXT NOT NULL, work_period INTEGER NOT NULL, entry_period INTEGER NOT NULL,
                document TEXT NOT NULL, PRIMARY KEY (order_id, work_period, entry_period))""")
            db.execute("""CREATE TABLE IF NOT EXISTS observed_closures (
                order_id TEXT PRIMARY KEY, verified_tokens INTEGER, settled INTEGER)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS missed_snapshots (order_id TEXT PRIMARY KEY, reason TEXT NOT NULL)"
            )

        return book

    @classmethod
    def from_env(cls):
        path = os.environ.get(DB_ENV)
        if not path:
            return None
        config = json.loads(os.environ.get(MARKETS_ENV) or "{}")
        return cls(
            path,
            [
                market
                for market, settings in config.items()
                if settings["mode"] == "observation"
            ],
        )

    def open(self, entry, owner_snapshot):
        spec = entry.params.get(PARAM)
        if not spec:
            return
        validate_spec(spec)
        if spec["mode"] != "observation":
            raise ValueError("not an observation order")
        if self.book.order(entry.job_id) is not None:
            return
        owners = {}
        observed_at = None
        if owner_snapshot is not None and owner_snapshot[0] <= spec["opened_at"]:
            observed_at, owners = owner_snapshot
        if observed_at is None:
            with closing(sqlite3.connect(self.book.path, timeout=0)) as db, db:
                db.execute(
                    "INSERT OR IGNORE INTO missed_snapshots VALUES (?,?)",
                    (entry.job_id, "registration_unavailable"),
                )
        self.book.open_order(
            entry.job_id, spec["market"], eligible_owners=owners, owners_observed_at=observed_at, **open_kwargs(spec)
        )

    def close(self, job_id, at, final):
        if self.book.order(job_id) is not None:
            self.book.close(job_id, at=at, observed_only=True)
            counts = final if final.get("counts_complete") is True else {}
            with closing(sqlite3.connect(self.book.path, timeout=0)) as db, db:
                db.execute(
                    "INSERT OR IGNORE INTO observed_closures VALUES (?,?,?)",
                    (job_id, counts.get("verified_tokens"), counts.get("settled")),
                )

    def _unclosed_page(self):
        with closing(sqlite3.connect(self.book.path, timeout=0)) as db:
            page = db.execute("""SELECT o.order_id FROM orders o
                LEFT JOIN observed_closures c ON c.order_id=o.order_id
                WHERE c.order_id IS NULL AND o.order_id>? ORDER BY o.order_id LIMIT 8""",
                (self._close_cursor,)).fetchall()
        self._close_cursor = page[-1][0] if len(page) == 8 else ""
        return [row[0] for row in page]

    async def recover_closures(self, records, active=()):
        """Retry local copies of durable finals independently of the live jobs and fleet."""
        for job_id in await asyncio.to_thread(self._unclosed_page):
            if job_id in active:
                continue
            try:
                final = await records.read_final_status(job_id)
                if final is not None and final.get("observation_closed_at") is not None:
                    await asyncio.to_thread(self.close, job_id, final["observation_closed_at"], final)
            except Exception:
                logging.getLogger(__name__).exception("could not reconcile a local observation closure")

    def recorder(self, entry):
        """A compact copy of the authoritative archive and its matching verified work, kept locally."""
        valuation = entry.params.get(PARAM, {}).get("opening_valuation", {})
        initial_price = valuation.get(MAX_PRICE_PARAM)
        initial_rate = valuation.get(MAX_PRICE_EMISSION_PARAM)

        def record(archive, ids, batch):
            rows = None
            if batch is not None and len(batch) == len(ids):
                rows = [
                    {
                        "id": sid,
                        "hotkey": v["hotkey"],
                        "tokens": int(v["token_count"]),
                        "audited": v.get("audited") is True,
                        "received_at": v.get("received_at"),
                        "lease_price": v.get("lease_price"),
                    }
                    for sid, v in zip(ids, batch)
                    if v.get("passed")
                ]
                if sum(row["tokens"] for row in rows) != archive["tokens"]:
                    rows = None
            rate = (
                initial_rate
                if initial_price is not None and initial_price == archive.get("max_price_per_token")
                else None
            )
            document = json.dumps(
                {
                    "tokens": archive["tokens"],
                    "rewards_by_hotkey": archive["rewards_by_hotkey"],
                    "miner_alpha_per_period": rate,
                    "valuation_source": "opening_declaration" if rate is not None else None,
                    "deliveries": rows,
                },
                allow_nan=False,
            )
            with closing(sqlite3.connect(self.book.path, timeout=0)) as db, db:
                db.execute(
                    "INSERT OR IGNORE INTO observed_periods VALUES (?,?,?,?)",
                    (
                        entry.job_id,
                        archive["work_period"],
                        archive["entry_period"],
                        document,
                    ),
                )

        return record
