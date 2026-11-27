"""Local comparison data for an explicitly marked order. Never used to set live pay."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing

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
        self.book = ContractBook(path)
        self.markets = frozenset(markets)
        with closing(sqlite3.connect(self.book.path)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS observed_periods (
                order_id TEXT NOT NULL, work_period INTEGER NOT NULL, entry_period INTEGER NOT NULL,
                document TEXT NOT NULL, PRIMARY KEY (order_id, work_period, entry_period))""")
            db.execute("""CREATE TABLE IF NOT EXISTS observed_closures (
                order_id TEXT PRIMARY KEY, verified_tokens INTEGER, settled INTEGER)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS missed_snapshots (order_id TEXT PRIMARY KEY, reason TEXT NOT NULL)"
            )

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

    def open(self, entry, owners):
        spec = entry.params.get(PARAM)
        if not spec:
            return
        validate_spec(spec)
        if spec["mode"] != "observation":
            raise ValueError("not an observation order")
        if self.book.order(entry.job_id) is not None:
            return
        if owners is None:
            with closing(sqlite3.connect(self.book.path, timeout=1)) as db, db:
                db.execute(
                    "INSERT OR IGNORE INTO missed_snapshots VALUES (?,?)",
                    (entry.job_id, "registration_unavailable"),
                )
            owners = {}
        self.book.open_order(
            entry.job_id, spec["market"], eligible_owners=owners, **open_kwargs(spec)
        )

    def close(self, job_id, at, final):
        if self.book.order(job_id) is not None:
            self.book.close(job_id, at=at, observed_only=True)
            counts = final if final.get("counts_complete") is True else {}
            with closing(sqlite3.connect(self.book.path, timeout=1)) as db, db:
                db.execute(
                    "INSERT OR REPLACE INTO observed_closures VALUES (?,?,?)",
                    (job_id, counts.get("verified_tokens"), counts.get("settled")),
                )

    def recorder(self, entry):
        """A compact copy of the authoritative archive and its matching verified work, kept locally."""
        initial_price = entry.params.get(MAX_PRICE_PARAM)
        initial_rate = entry.params[MAX_PRICE_EMISSION_PARAM]

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
                if initial_price == archive.get("max_price_per_token")
                else None
            )
            document = json.dumps(
                {
                    "tokens": archive["tokens"],
                    "rewards_by_hotkey": archive["rewards_by_hotkey"],
                    "miner_alpha_per_period": rate,
                    "deliveries": rows,
                },
                allow_nan=False,
            )
            with closing(sqlite3.connect(self.book.path, timeout=1)) as db, db:
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
