"""Private order contracts, shared by the orders of one authoritative allocator.

Offer prices are integer micro-alpha per million verified tokens. A token at a
price of 1 costs 1/1000 rao. Reservations use these thousandths exactly. The
caller supplies a funded order envelope; this book does not establish that a
settlement period or payment rail can fund it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from reliquary.corpus.contract_pacing import audit_rate, decay_counts, reliability

_WEEK = 7 * 86400
_INT_MAX = (1 << 63) - 1


def _int(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= _INT_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT_MAX}]")
    return value


def _key(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS offer_versions (
 id INTEGER PRIMARY KEY, market TEXT NOT NULL, hotkey TEXT NOT NULL, effective_at INTEGER NOT NULL,
 coldkey TEXT NOT NULL, price_micro INTEGER NOT NULL, rate INTEGER NOT NULL, slots INTEGER NOT NULL, sequence INTEGER);
CREATE INDEX IF NOT EXISTS historic_offers ON offer_versions (market,hotkey,effective_at,id);
CREATE TABLE IF NOT EXISTS orders (
 order_id TEXT PRIMARY KEY, market TEXT NOT NULL, spec TEXT NOT NULL, closed_at INTEGER);
CREATE TABLE IF NOT EXISTS opening_owner_reads (
 order_id TEXT PRIMARY KEY REFERENCES orders, observed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS snapshot (
 order_id TEXT NOT NULL REFERENCES orders, hotkey TEXT NOT NULL, coldkey TEXT NOT NULL,
 price_micro INTEGER NOT NULL, offer_rate INTEGER NOT NULL, offer_slots INTEGER NOT NULL,
 rate_cap INTEGER NOT NULL, q REAL NOT NULL, adjusted REAL NOT NULL, rank INTEGER NOT NULL,
 PRIMARY KEY (order_id, hotkey));
CREATE TABLE IF NOT EXISTS awards (
 id INTEGER PRIMARY KEY, order_id TEXT NOT NULL, hotkey TEXT NOT NULL, rate INTEGER NOT NULL, slots INTEGER NOT NULL,
 tokens INTEGER NOT NULL, started_at INTEGER NOT NULL, ended_at INTEGER, reason TEXT,
 coldkey TEXT NOT NULL, price_micro INTEGER NOT NULL, offer_rate INTEGER NOT NULL, offer_slots INTEGER NOT NULL,
 q REAL NOT NULL, FOREIGN KEY (order_id) REFERENCES orders);
CREATE INDEX IF NOT EXISTS active_awards ON awards (order_id, ended_at, hotkey);
CREATE TABLE IF NOT EXISTS grants (
 order_id TEXT NOT NULL, lease_id TEXT NOT NULL, award_id INTEGER NOT NULL REFERENCES awards,
 max_tokens INTEGER NOT NULL, at INTEGER NOT NULL, state TEXT NOT NULL,
 PRIMARY KEY (order_id, lease_id));
CREATE TABLE IF NOT EXISTS deliveries (
 order_id TEXT NOT NULL, delivery_id TEXT NOT NULL, award_id INTEGER NOT NULL REFERENCES awards,
 tokens INTEGER NOT NULL, audited_tokens INTEGER NOT NULL, at INTEGER NOT NULL, lease_id TEXT,
 PRIMARY KEY (order_id, delivery_id), UNIQUE (order_id, lease_id));
CREATE INDEX IF NOT EXISTS deliveries_award ON deliveries (award_id);
CREATE INDEX IF NOT EXISTS open_grants ON grants (award_id, state);
CREATE TABLE IF NOT EXISTS exits (
 order_id TEXT NOT NULL REFERENCES orders, event_id TEXT NOT NULL, request TEXT NOT NULL,
 result TEXT NOT NULL, PRIMARY KEY (order_id, event_id));
CREATE TABLE IF NOT EXISTS gaps (
 id INTEGER PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders, source TEXT, at INTEGER NOT NULL,
 hotkey TEXT, coldkey TEXT, price_micro INTEGER, rate INTEGER NOT NULL, tokens INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pending_replacements (
 award_id INTEGER PRIMARY KEY REFERENCES awards, source TEXT NOT NULL, eligible INTEGER NOT NULL,
 credited INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS extra_rounds (
 order_id TEXT NOT NULL REFERENCES orders, event_id TEXT NOT NULL, at INTEGER NOT NULL,
 snapshot TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY (order_id,event_id));
CREATE TABLE IF NOT EXISTS pauses (
 order_id TEXT NOT NULL REFERENCES orders, start INTEGER NOT NULL, end INTEGER NOT NULL,
 PRIMARY KEY (order_id,start,end));
CREATE TABLE IF NOT EXISTS performance (
 award_id INTEGER PRIMARY KEY REFERENCES awards, coldkey TEXT NOT NULL,
 at INTEGER NOT NULL, committed INTEGER NOT NULL, delivered INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS performance_owner ON performance (coldkey, at);
"""


class ContractBook:
    """A durable book. Each mutation is a single SQLite write transaction."""

    def __init__(self, path):
        self.path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("a durable database path is required")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def post_offer(self, market, hotkey, coldkey, *, price_micro, rate, slots, sequence=None, at=None):
        """coldkey must be authoritatively bound to the authenticated hotkey by the caller."""
        for value, name in ((market, "market"), (hotkey, "hotkey"), (coldkey, "coldkey")):
            _key(value, name)
        _int(price_micro, "price_micro")
        _int(rate, "rate")
        _int(slots, "slots")
        if (rate == 0) != (slots == 0):
            raise ValueError("withdraw an offer with both rate and slots set to zero")
        if at is None:
            at = int(time.time())
        _int(at, "at")
        if sequence is not None:
            _int(sequence, "sequence")
        with self._db(write=True) as db:
            old = db.execute(
                """SELECT * FROM offer_versions WHERE market=? AND hotkey=?
                ORDER BY effective_at DESC,id DESC LIMIT 1""",
                (market, hotkey),
            ).fetchone()
            if old and old["sequence"] is not None:
                if sequence is None or sequence < old["sequence"]:
                    raise ValueError("offer sequence has already advanced")
                if sequence == old["sequence"]:
                    if (coldkey, price_micro, rate, slots) != (
                        old["coldkey"],
                        old["price_micro"],
                        old["rate"],
                        old["slots"],
                    ):
                        raise ValueError("conflicting offer sequence")
                    return
            if old and at < old["effective_at"]:
                raise ValueError("offer clock moved backwards")
            db.execute(
                """INSERT INTO offer_versions (market,hotkey,effective_at,coldkey,price_micro,rate,slots,sequence)
                VALUES (?,?,?,?,?,?,?,?)""",
                (market, hotkey, at, coldkey, price_micro, rate, slots, sequence),
            )

    @staticmethod
    def _spec(db, order_id):
        row = db.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if row is None:
            raise KeyError(order_id)
        return {**json.loads(row["spec"]), "closed_at": row["closed_at"]}

    def order(self, order_id):
        with self._db() as db:
            try:
                return self._spec(db, order_id)
            except KeyError:
                return None

    @staticmethod
    def _history(db, market, coldkey, now):
        committed = delivered = 0.0
        for row in db.execute("SELECT * FROM performance WHERE coldkey=? AND at<=? ORDER BY award_id", (coldkey, now)):
            dec_delivered, dec_committed = decay_counts(row["delivered"], row["committed"], now - row["at"])
            delivered += dec_delivered
            committed += dec_committed
        history = db.execute(
            """SELECT d.tokens,d.audited_tokens,d.at,o.market FROM deliveries d
            JOIN awards a ON d.award_id=a.id
            JOIN orders o ON o.order_id=a.order_id
            WHERE a.coldkey=? AND d.at<=?""",
            (coldkey, now),
        )
        clean = recent = 0
        for row in history:
            clean += row["audited_tokens"]
            if row["market"] == market and row["at"] > now - _WEEK:
                recent += row["tokens"]
        return reliability(delivered, committed), clean, recent

    def open_order(
        self,
        order_id,
        market,
        *,
        opened_at,
        duration_s,
        target_rate,
        max_tokens,
        budget_rao,
        ceiling_micro,
        new_rate_cap,
        audit_cost_micro=0,
        evaluation=False,
        q_min=0.01,
        trust_tokens=10_000_000,
        eligible_owners=None,
        owners_observed_at=None,
    ):
        _key(order_id, "order_id")
        _key(market, "market")
        for name, value in (
            ("duration_s", duration_s),
            ("target_rate", target_rate),
            ("max_tokens", max_tokens),
            ("new_rate_cap", new_rate_cap),
            ("trust_tokens", trust_tokens),
        ):
            _int(value, name, 1)
        for name, value in (
            ("opened_at", opened_at),
            ("budget_rao", budget_rao),
            ("ceiling_micro", ceiling_micro),
            ("audit_cost_micro", audit_cost_micro),
        ):
            _int(value, name)
        _int(opened_at + duration_s, "deadline", 1)
        if type(evaluation) is not bool or type(q_min) not in (float, int) or not 0 < q_min <= 1:
            raise ValueError("invalid audit policy")
        spec = {
            "order_id": order_id,
            "market": market,
            "opened_at": opened_at,
            "duration_s": duration_s,
            "target_rate": target_rate,
            "max_tokens": max_tokens,
            "budget_rao": budget_rao,
            "ceiling_micro": ceiling_micro,
            "new_rate_cap": new_rate_cap,
            "audit_cost_micro": audit_cost_micro,
            "evaluation": evaluation,
            "q_min": float(q_min),
            "trust_tokens": trust_tokens,
        }
        with self._db(write=True) as db:
            stored = db.execute("SELECT spec FROM orders WHERE order_id=?", (order_id,)).fetchone()
            if stored:
                if json.loads(stored["spec"]) != spec:
                    raise ValueError("conflicting order opening")
                return self._contracts(db, order_id)
            db.execute("INSERT INTO orders VALUES (?,?,?,NULL)", (order_id, market, _json(spec)))
            if owners_observed_at is not None:
                if eligible_owners is None or not 0 <= owners_observed_at <= opened_at:
                    raise ValueError("ownership must have been observed by the opening second")
                db.execute("INSERT INTO opening_owner_reads VALUES (?,?)", (order_id, owners_observed_at))
            candidates = self._candidates(db, spec, opened_at, eligible_owners)
            for c in candidates:
                db.execute(
                    """INSERT INTO snapshot VALUES (:order_id,:hotkey,:coldkey,:price_micro,
                    :offer_rate,:offer_slots,:rate_cap,:q,:adjusted,:rank)""",
                    c,
                )
            reserved = self._allocate(db, spec, opened_at, target_rate, max_tokens, candidates)
            self._gap(db, order_id, opened_at, target_rate, min(max_tokens, duration_s * target_rate), reserved)
            return self._contracts(db, order_id)

    def _candidates(self, db, spec, at, owners):
        # Exclude the entire opening second: an accepted change in that second can be later
        # than the registry opening, even if the control process reads both some time later.
        offers = db.execute(
            """SELECT v.* FROM offer_versions v WHERE v.market=? AND v.rate>0
            AND v.price_micro<=? AND v.id=(SELECT later.id FROM offer_versions later
            WHERE later.market=v.market AND later.hotkey=v.hotkey AND later.effective_at<?
            ORDER BY later.effective_at DESC,later.id DESC LIMIT 1)""",
            (spec["market"], spec["ceiling_micro"], at),
        )
        candidates, histories = [], {}
        for offer in offers:
            owner = offer["coldkey"]
            if owners is not None and owners.get(offer["hotkey"]) != owner:
                continue
            if owner not in histories:
                histories[owner] = self._history(db, spec["market"], owner, at)
            r, clean, recent = histories[owner]
            q = audit_rate(clean, q_min=spec["q_min"], trust_tokens=spec["trust_tokens"], evaluation=spec["evaluation"])
            candidates.append(
                {
                    "order_id": spec["order_id"],
                    "hotkey": offer["hotkey"],
                    "coldkey": owner,
                    "price_micro": offer["price_micro"],
                    "offer_rate": offer["rate"],
                    "offer_slots": offer["slots"],
                    "rate_cap": max(spec["new_rate_cap"], 3 * recent // (2 * _WEEK)),
                    "q": q,
                    "adjusted": offer["price_micro"] * (2 - r) + q * spec["audit_cost_micro"],
                    "effective_at": offer["effective_at"],
                }
            )
        candidates.sort(key=lambda c: (c["adjusted"], c["coldkey"], c["hotkey"]))
        for rank, c in enumerate(candidates, 1):
            c["rank"] = rank
        return candidates

    @staticmethod
    def _gap(db, order_id, at, rate, tokens, awards, *, source=None, previous=None):
        rate -= sum(a["rate"] for a in awards)
        tokens -= sum(a["tokens"] for a in awards)
        if rate > 0 and (tokens > 0 or source is not None):
            db.execute(
                """INSERT INTO gaps (order_id,source,at,hotkey,coldkey,price_micro,rate,tokens)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    order_id,
                    source,
                    at,
                    previous["hotkey"] if previous else None,
                    previous["coldkey"] if previous else None,
                    previous["price_micro"] if previous else None,
                    rate,
                    tokens,
                ),
            )

    @staticmethod
    def _usage(db, award_id):
        delivered = db.execute(
            "SELECT coalesce(sum(tokens),0) FROM deliveries WHERE award_id=?", (award_id,)
        ).fetchone()[0]
        outstanding = db.execute(
            """SELECT coalesce(sum(max_tokens),0) FROM grants
            WHERE award_id=? AND state='open'""",
            (award_id,),
        ).fetchone()[0]
        return delivered, outstanding

    @staticmethod
    def _awards(db, order_id):
        return db.execute("SELECT * FROM awards WHERE order_id=? ORDER BY id", (order_id,)).fetchall()

    @staticmethod
    def _award(db, job, c, at, rate, slots, tokens):
        db.execute(
            """INSERT INTO awards (order_id,hotkey,coldkey,price_micro,offer_rate,offer_slots,q,
            rate,slots,tokens,started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job,
                c["hotkey"],
                c["coldkey"],
                c["price_micro"],
                c["offer_rate"],
                c["offer_slots"],
                c["q"],
                rate,
                slots,
                tokens,
                at,
            ),
        )

    def _allocate(self, db, spec, at, needed_rate, needed_tokens, candidates=None):
        job = spec["order_id"]
        deadline = spec["opened_at"] + spec["duration_s"]
        seconds = self._active_seconds(db, job, at, deadline)
        if seconds <= 0 or not needed_rate or not needed_tokens:
            return []
        volume = spec["max_tokens"]
        funds = spec["budget_rao"] * 1000
        target_free = spec["target_rate"]
        for a in self._awards(db, job):
            if a["ended_at"] is None:
                target_free -= a["rate"]
            exposure = a["tokens"] if a["ended_at"] is None else sum(self._usage(db, a["id"]))
            volume -= exposure
            funds -= exposure * a["price_micro"]
        needed_rate = min(needed_rate, target_free)
        result = []
        # First call candidates who have never held an award, then top up existing award holders.
        if candidates is None:
            candidates = db.execute(
                """SELECT s.*, EXISTS(SELECT 1 FROM awards a
                WHERE a.order_id=s.order_id AND a.hotkey=s.hotkey) AS previous
                FROM snapshot s WHERE s.order_id=?
                AND NOT EXISTS(SELECT 1 FROM awards a WHERE a.order_id=s.order_id AND a.hotkey=s.hotkey
                               AND a.ended_at IS NOT NULL)
                ORDER BY previous,rank""",
                (job,),
            ).fetchall()
        for c in candidates:
            if min(volume, needed_rate, needed_tokens) <= 0:
                break
            # One active hotkey has one price and one owner in this order, even if a
            # supplemental round sees a new signed price or an ownership change.
            if db.execute(
                """SELECT 1 FROM awards WHERE order_id=? AND hotkey=? AND ended_at IS NULL
                AND (price_micro!=? OR coldkey!=?) LIMIT 1""",
                (job, c["hotkey"], c["price_micro"], c["coldkey"]),
            ).fetchone():
                continue
            used_cold, used_hot, used_slots, job_rate, job_slots = db.execute(
                """SELECT
                coalesce(sum(CASE WHEN a.coldkey=:owner THEN a.rate ELSE 0 END),0),
                coalesce(sum(CASE WHEN a.hotkey=:hot AND o.market=:market THEN a.rate ELSE 0 END),0),
                coalesce(sum(CASE WHEN a.hotkey=:hot AND o.market=:market THEN a.slots ELSE 0 END),0),
                coalesce(sum(CASE WHEN a.hotkey=:hot AND a.order_id=:job THEN a.rate ELSE 0 END),0),
                coalesce(sum(CASE WHEN a.hotkey=:hot AND a.order_id=:job THEN a.slots ELSE 0 END),0)
                FROM awards a JOIN orders o USING(order_id)
                WHERE (a.coldkey=:owner OR a.hotkey=:hot) AND a.started_at<:deadline
                AND (a.ended_at IS NULL OR a.ended_at>:now)
                AND json_extract(o.spec,'$.opened_at')+json_extract(o.spec,'$.duration_s')>:now""",
                {
                    "hot": c["hotkey"],
                    "market": spec["market"],
                    "job": job,
                    "owner": c["coldkey"],
                    "now": at,
                    "deadline": deadline,
                },
            ).fetchone()
            price = c["price_micro"]
            # Already-reserved slots in this same order can carry an incremental
            # rate without reserving a second copy of a partially used slot.
            available_slots = c["offer_slots"] - used_slots + job_slots
            rate = min(
                needed_rate,
                c["rate_cap"] - used_cold,
                c["offer_rate"] - used_hot,
                available_slots * c["offer_rate"] // c["offer_slots"] - job_rate,
            )
            affordable = funds // price if price else volume
            tokens = min(volume, needed_tokens, affordable, rate * seconds)
            if tokens <= 0:
                continue
            rate = min(rate, (tokens + seconds - 1) // seconds)
            slots = max(0, ((job_rate + rate) * c["offer_slots"] + c["offer_rate"] - 1) // c["offer_rate"] - job_slots)
            self._award(db, job, c, at, rate, slots, tokens)
            result.append({"hotkey": c["hotkey"], "price_micro": price, "tokens": tokens, "rate": rate})
            needed_tokens -= tokens
            needed_rate -= rate
            volume -= tokens
            funds -= tokens * price
        return result

    def _contracts(self, db, order_id):
        result = {}
        for a in self._awards(db, order_id):
            if a["ended_at"] is not None:
                continue
            hotkey = a["hotkey"]
            c = result.setdefault(
                hotkey,
                {
                    "hotkey": hotkey,
                    "coldkey": a["coldkey"],
                    "price_micro": a["price_micro"],
                    "q": a["q"],
                    "rate": 0,
                    "slots": 0,
                    "tokens": 0,
                    "remaining_tokens": 0,
                },
            )
            c["rate"] += a["rate"]
            c["tokens"] += a["tokens"]
            c["remaining_tokens"] += a["tokens"] - sum(self._usage(db, a["id"]))
            c["slots"] += a["slots"]
        return list(result.values())

    def contracts(self, order_id):
        with self._db() as db:
            return self._contracts(db, order_id)

    def contract(self, order_id, hotkey):
        return next((c for c in self.contracts(order_id) if c["hotkey"] == hotkey), None)

    def waitlist(self, order_id):
        with self._db() as db:
            rows = db.execute(
                """SELECT s.* FROM snapshot s WHERE order_id=? AND NOT EXISTS
                (SELECT 1 FROM awards a WHERE a.order_id=s.order_id AND a.hotkey=s.hotkey) ORDER BY rank""",
                (order_id,),
            )
            return [dict(row) for row in rows]

    @staticmethod
    def _working_at(db, order_id, at):
        return (
            db.execute(
                "SELECT 1 FROM pauses WHERE order_id=? AND start<=? AND end>? LIMIT 1", (order_id, at, at)
            ).fetchone()
            is None
        )

    @staticmethod
    def _active_seconds(db, order_id, start, end):
        # Outages may overlap or coincide. Exclude each second at most once.
        total, cursor = 0, start
        for left, right in db.execute(
            """SELECT start,end FROM pauses WHERE order_id=?
                AND start<? AND end>? ORDER BY start,end""",
            (order_id, end, start),
        ):
            total += max(0, left - cursor)
            cursor = max(cursor, right)
        return total + max(0, end - cursor)

    def service_interval(self, order_id, *, start, end):
        """Record a verified service outage before making an attribution covering that interval."""
        _int(start, "start")
        _int(end, "end")
        if end <= start:
            raise ValueError("outage end must follow start")
        with self._db(write=True) as db:
            self._spec(db, order_id)
            if db.execute(
                "SELECT 1 FROM pauses WHERE order_id=? AND start=? AND end=?", (order_id, start, end)
            ).fetchone():
                return
            if db.execute(
                """SELECT 1 FROM awards WHERE order_id=? AND ended_at>?
                AND started_at<? LIMIT 1""",
                (order_id, start, end),
            ).fetchone():
                raise ValueError("outage overlaps a finalized performance attribution")
            db.execute("INSERT INTO pauses VALUES (?,?,?)", (order_id, start, end))

    def grant(self, order_id, hotkey, *, lease_id, max_tokens, at):
        """Reserve a known maximum payable token count before issuing a lease."""
        _int(max_tokens, "max_tokens", 1)
        _int(at, "at")
        _key(lease_id, "lease_id")
        with self._db(write=True) as db:
            spec = self._spec(db, order_id)
            old = db.execute(
                """SELECT g.*,a.hotkey FROM grants g JOIN awards a ON a.id=g.award_id
                WHERE g.order_id=? AND lease_id=?""",
                (order_id, lease_id),
            ).fetchone()
            if old and (old["max_tokens"] != max_tokens or old["hotkey"] != hotkey):
                raise ValueError("conflicting lease reservation")
            if spec["closed_at"] is not None or not spec["opened_at"] <= at < spec["opened_at"] + spec["duration_s"]:
                return False
            if not self._working_at(db, order_id, at):
                return False
            if old:
                return old["state"] == "open"
            awards = db.execute(
                """SELECT * FROM awards WHERE order_id=? AND hotkey=? AND
                ended_at IS NULL AND started_at<=? ORDER BY id""",
                (order_id, hotkey, at),
            ).fetchall()
            holds = db.execute(
                """SELECT count(*) FROM grants g JOIN awards a ON a.id=g.award_id
                WHERE g.order_id=? AND a.hotkey=? AND g.state='open'""",
                (order_id, hotkey),
            ).fetchone()[0]
            if holds >= sum(a["slots"] for a in awards):
                return False
            for a in awards:
                if a["tokens"] - sum(self._usage(db, a["id"])) >= max_tokens:
                    db.execute(
                        "INSERT INTO grants VALUES (?,?,?,?,?, 'open')", (order_id, lease_id, a["id"], max_tokens, at)
                    )
                    return True
            return False

    def _release_pending(self, db, order_id, award_id, at):
        """Only newly unspent exposure from before a departure can increase that departure's gap."""
        pending = db.execute(
            """SELECT p.*,a.tokens,a.ended_at FROM pending_replacements p
            JOIN awards a ON a.id=p.award_id WHERE p.award_id=?""",
            (award_id,),
        ).fetchone()
        if pending is None:
            return
        delivered, held = self._usage(db, award_id)
        free = min(pending["eligible"], pending["tokens"] - delivered - held)
        delta = free - pending["credited"]
        if held:
            db.execute("UPDATE pending_replacements SET credited=? WHERE award_id=?", (free, award_id))
        else:
            db.execute("DELETE FROM pending_replacements WHERE award_id=?", (award_id,))
        spec = self._spec(db, order_id)
        if delta <= 0 or spec["closed_at"] is not None:
            return
        deadline = spec["opened_at"] + spec["duration_s"]
        # With a legacy unclocked cancellation the actual supplemental round applies its
        # own time bound. A timestamped verdict also bounds the restored gap at release.
        when = max(pending["ended_at"], at if at is not None else pending["ended_at"])
        seconds = self._active_seconds(db, order_id, when, deadline)
        gap = db.execute(
            "SELECT id,rate,tokens FROM gaps WHERE order_id=? AND source=?", (order_id, pending["source"])
        ).fetchone()
        if gap and seconds:
            restored = min(delta, max(0, gap["rate"] * seconds - gap["tokens"]))
            db.execute("UPDATE gaps SET tokens=tokens+? WHERE id=?", (restored, gap["id"]))

    def cancel_grant(self, order_id, *, lease_id, at=None):
        """Release unspent exposure only after the lease can no longer be paid."""
        if at is not None:
            _int(at, "at")
        with self._db(write=True) as db:
            grant = db.execute(
                """SELECT award_id,at FROM grants WHERE order_id=? AND lease_id=?
                AND state='open'""",
                (order_id, lease_id),
            ).fetchone()
            if grant is None:
                return False
            if at is not None and at < grant["at"]:
                raise ValueError("cancellation before grant")
            db.execute("UPDATE grants SET state='cancelled' WHERE order_id=? AND lease_id=?", (order_id, lease_id))
            self._release_pending(db, order_id, grant["award_id"], at)
            return True

    def record_delivery(
        self, order_id, hotkey, *, delivery_id, tokens, at, audited_tokens=0, lease_id=None, resolved_at=None
    ):
        """Book an accepted delivery. Live callers supply a lease; replay may attribute directly."""
        _key(delivery_id, "delivery_id")
        _int(tokens, "tokens")
        _int(at, "at")
        if resolved_at is not None:
            _int(resolved_at, "resolved_at", minimum=at)
        _int(audited_tokens, "audited_tokens")
        if audited_tokens > tokens:
            raise ValueError("audited tokens exceed delivered tokens")
        with self._db(write=True) as db:
            spec = self._spec(db, order_id)
            old = db.execute(
                """SELECT d.*,a.hotkey FROM deliveries d JOIN awards a ON a.id=d.award_id
                WHERE d.order_id=? AND delivery_id=?""",
                (order_id, delivery_id),
            ).fetchone()
            if old:
                if (old["tokens"], old["hotkey"], old["audited_tokens"], old["lease_id"]) != (
                    tokens,
                    hotkey,
                    audited_tokens,
                    lease_id,
                ):
                    raise ValueError("conflicting delivery")
                return False
            if lease_id is not None:
                g = db.execute(
                    """SELECT g.*,a.hotkey FROM grants g JOIN awards a ON a.id=g.award_id
                    WHERE g.order_id=? AND lease_id=?""",
                    (order_id, lease_id),
                ).fetchone()
                if not g or g["hotkey"] != hotkey or g["state"] != "open" or tokens > g["max_tokens"] or at < g["at"]:
                    raise ValueError("delivery has no matching payable grant")
                award_id = g["award_id"]
                db.execute("UPDATE grants SET state='delivered' WHERE order_id=? AND lease_id=?", (order_id, lease_id))
            else:
                if (
                    spec["closed_at"] is not None
                    or not spec["opened_at"] <= at <= spec["opened_at"] + spec["duration_s"]
                ):
                    raise ValueError("direct replay delivery outside order")
                eligible = (
                    a
                    for a in self._awards(db, order_id)
                    if a["hotkey"] == hotkey
                    and a["started_at"] <= at
                    and a["ended_at"] is None
                    and a["tokens"] - sum(self._usage(db, a["id"])) >= tokens
                )
                a = next(eligible, None)
                if a is None:
                    raise ValueError("delivery exceeds remaining contract")
                award_id = a["id"]
            db.execute(
                "INSERT INTO deliveries VALUES (?,?,?,?,?,?,?)",
                (order_id, delivery_id, award_id, tokens, audited_tokens, at, lease_id),
            )
            if lease_id is not None:
                self._release_pending(db, order_id, award_id, at if resolved_at is None else resolved_at)
            # Previously granted work may be accepted after release; keep its earned reliability.
            db.execute(
                """UPDATE performance SET delivered=delivered+?,
                committed=max(committed,delivered+?) WHERE award_id=?""",
                (tokens, tokens, award_id),
            )
            return True

    def _finish(self, db, spec, award, at, reason, missed_notice):
        delivered, held = self._usage(db, award["id"])
        start = award["started_at"]
        deadline = spec["opened_at"] + spec["duration_s"]
        if at < start:
            raise ValueError("award has not yet started")
        active_until = min(at, deadline)
        if reason == "disappeared":
            active_until = deadline
        elif reason == "announced":
            active_until = min(deadline, at + missed_notice)
        seconds = self._active_seconds(db, spec["order_id"], start, active_until)
        commitment = max(delivered, min(award["tokens"], seconds * award["rate"]))
        db.execute("UPDATE awards SET ended_at=?,reason=? WHERE id=?", (at, reason, award["id"]))
        db.execute(
            "INSERT INTO performance VALUES (?,?,?,?,?)", (award["id"], award["coldkey"], at, commitment, delivered)
        )
        return min(
            award["tokens"] - delivered - held, award["rate"] * self._active_seconds(db, spec["order_id"], at, deadline)
        )

    def exit(self, order_id, hotkey, *, at, reason, expired_leases=0, penalty_rao=0, notice_s=0, event_id=None):
        """notice_s is the unserved portion of the notice, in seconds. A release makes no replacement."""
        if reason not in ("announced", "disappeared", "release"):
            raise ValueError("unknown exit reason")
        for name, value in (
            ("at", at),
            ("expired_leases", expired_leases),
            ("penalty_rao", penalty_rao),
            ("notice_s", notice_s),
        ):
            _int(value, name)
        request = _json(
            {
                "order_id": order_id,
                "hotkey": hotkey,
                "at": at,
                "reason": reason,
                "expired_leases": expired_leases,
                "penalty_rao": penalty_rao,
                "notice_s": notice_s,
            }
        )
        if event_id is None:
            event_id = hashlib.sha256(request.encode()).hexdigest()
        _key(event_id, "event_id")
        with self._db(write=True) as db:
            old = db.execute(
                "SELECT request,result FROM exits WHERE order_id=? AND event_id=?", (order_id, event_id)
            ).fetchone()
            if old:
                if old["request"] != request:
                    raise ValueError("conflicting exit event")
                return json.loads(old["result"])
            spec = self._spec(db, order_id)
            awards = [a for a in self._awards(db, order_id) if a["hotkey"] == hotkey and a["ended_at"] is None]
            if spec["closed_at"] is not None or not awards:
                raise ValueError("no active contract")
            if at > spec["opened_at"] + spec["duration_s"]:
                raise ValueError("exit after order deadline")
            if not self._working_at(db, order_id, at):
                reason = "service"
            remaining = 0
            for a in awards:
                free = self._finish(db, spec, a, at, reason, notice_s)
                remaining += free
                if reason in ("announced", "disappeared"):
                    delivered, held = self._usage(db, a["id"])
                    deadline = spec["opened_at"] + spec["duration_s"]
                    eligible = min(
                        a["tokens"] - delivered, a["rate"] * self._active_seconds(db, order_id, at, deadline)
                    )
                    if held and eligible > free:
                        db.execute(
                            "INSERT INTO pending_replacements VALUES (?,?,?,?)", (a["id"], event_id, eligible, free)
                        )

            rate = sum(a["rate"] for a in awards)
            penalty = expired_leases * penalty_rao if reason == "disappeared" else 0
            return self._exit_result(db, spec, event_id, request, awards[0], at, reason, rate, remaining, penalty)

    def _exit_result(self, db, spec, event_id, request, previous, at, reason, rate, tokens, penalty=0):
        replacements = []
        if reason in ("announced", "disappeared", "partial"):
            replacements = self._allocate(db, spec, at, rate, tokens)
            self._gap(db, spec["order_id"], at, rate, tokens, replacements, source=event_id, previous=previous)
        result = self._debt(event_id, spec["order_id"], previous, at, reason, replacements, penalty)
        db.execute("INSERT INTO exits VALUES (?,?,?,?)", (spec["order_id"], event_id, request, _json(result)))
        return result

    @staticmethod
    def _debt(event_id, job, previous, at, reason, replacements, penalty=0):
        extra = sum(max(0, a["price_micro"] - previous["price_micro"]) * a["tokens"] for a in replacements)
        return {
            "event_id": event_id,
            "order_id": job,
            "hotkey": previous["hotkey"],
            "coldkey": previous["coldkey"],
            "at": at,
            "reason": reason,
            "replacement_millirao": extra,
            "penalty_rao": penalty,
            "assessed_rao": extra // 1000 + penalty,
            "collected_rao": 0,
            "replaced_tokens": sum(a["tokens"] for a in replacements),
        }

    def supplemental_round(self, order_id, *, event_id, at, eligible_owners):
        """Reopen existing gaps at the original ceiling; fresh candidates still receive their signed price."""
        _int(at, "at")
        _key(event_id, "event_id")
        if eligible_owners is None:
            raise ValueError("a fresh ownership snapshot is required")
        with self._db(write=True) as db:
            old = db.execute(
                "SELECT at,result FROM extra_rounds WHERE order_id=? AND event_id=?", (order_id, event_id)
            ).fetchone()
            if old:
                if old["at"] != at:
                    raise ValueError("conflicting supplemental round")
                return json.loads(old["result"])
            spec = self._spec(db, order_id)
            deadline = spec["opened_at"] + spec["duration_s"]
            if spec["closed_at"] is not None or not spec["opened_at"] <= at < deadline:
                raise ValueError("supplemental round outside order")
            if not self._working_at(db, order_id, at):
                raise ValueError("order is paused")
            candidates = self._candidates(db, spec, at, eligible_owners)
            # A miner that already left or was downsized can recommit at a later signed revision.
            exited = dict(
                db.execute(
                    """SELECT hotkey,max(ended_at) FROM awards
                WHERE order_id=? AND ended_at IS NOT NULL GROUP BY hotkey""",
                    (order_id,),
                )
            )
            candidates = [c for c in candidates if c["effective_at"] > exited.get(c["hotkey"], -1)]
            result = {"event_id": event_id, "awards": [], "debts": []}
            for gap in db.execute(
                """SELECT * FROM gaps WHERE order_id=? AND at<=? AND tokens>0 AND rate>0
                                     ORDER BY at,id""",
                (order_id, at),
            ).fetchall():
                tokens = min(gap["tokens"], gap["rate"] * self._active_seconds(db, order_id, at, deadline))
                awards = self._allocate(db, spec, at, gap["rate"], tokens, candidates)
                if not awards:
                    continue
                result["awards"].extend(awards)
                db.execute(
                    """UPDATE gaps SET rate=rate-?,tokens=tokens-? WHERE id=?""",
                    (sum(a["rate"] for a in awards), sum(a["tokens"] for a in awards), gap["id"]),
                )
                if gap["source"] is not None:
                    delta_id = hashlib.sha256(_json([order_id, event_id, gap["source"]]).encode()).hexdigest()
                    debt = self._debt(delta_id, order_id, gap, at, "supplemental", awards)
                    debt["source_event_id"] = gap["source"]
                    result["debts"].append(debt)
            db.execute(
                "INSERT INTO extra_rounds VALUES (?,?,?,?,?)",
                (order_id, event_id, at, _json(candidates), _json(result)),
            )
            return result

    def reconcile_rate(
        self, order_id, hotkey, *, event_id, window_start, at, evidence_complete=False, min_active_s=180
    ):
        """Replay an independently attested interval of continuously offered, verifiable work.

        A missing service/queue attribution, pending grants, an interval shorter than the
        minimum or no clean audit cannot establish a shortfall. All are neutral. A live
        caller must independently supply this evidence before using this calculation.
        """
        _key(event_id, "event_id")
        _int(at, "at")
        _int(window_start, "window_start")
        _int(min_active_s, "min_active_s", 1)
        if type(evidence_complete) is not bool or at <= window_start:
            raise ValueError("invalid measurement window")
        request = _json(
            {
                "order_id": order_id,
                "hotkey": hotkey,
                "reason": "partial",
                "at": at,
                "window_start": window_start,
                "evidence_complete": evidence_complete,
                "min_active_s": min_active_s,
            }
        )
        with self._db(write=True) as db:
            old = db.execute(
                "SELECT request,result FROM exits WHERE order_id=? AND event_id=?", (order_id, event_id)
            ).fetchone()
            if old:
                if old["request"] != request:
                    raise ValueError("conflicting measurement event")
                return json.loads(old["result"])
            spec = self._spec(db, order_id)
            awards = [a for a in self._awards(db, order_id) if a["hotkey"] == hotkey and a["ended_at"] is None]
            deadline = spec["opened_at"] + spec["duration_s"]
            if (
                spec["closed_at"] is not None
                or not awards
                or at >= deadline
                or window_start < max(a["started_at"] for a in awards)
            ):
                raise ValueError("measurement outside current contract")
            measured, clean = db.execute(
                """SELECT coalesce(sum(d.tokens),0),coalesce(sum(d.audited_tokens),0)
                FROM deliveries d JOIN awards a ON a.id=d.award_id WHERE a.order_id=? AND a.hotkey=?
                AND a.ended_at IS NULL AND d.at>? AND d.at<=?""",
                (order_id, hotkey, window_start, at),
            ).fetchone()
            seconds = self._active_seconds(db, order_id, window_start, at)
            held = any(self._usage(db, a["id"])[1] for a in awards)
            old_rate = sum(a["rate"] for a in awards)
            rate = (measured + seconds - 1) // seconds if seconds else old_rate
            eligible_volume = sum(a["tokens"] - self._usage(db, a["id"])[0] for a in awards) + measured
            if (
                not evidence_complete
                or seconds < min_active_s
                or not clean
                or held
                or not self._working_at(db, order_id, at)
                or rate >= old_rate
                or eligible_volume < old_rate * seconds
            ):
                db.execute("INSERT INTO exits VALUES (?,?,?,?)", (order_id, event_id, request, "null"))
                return None
            remaining = sum(self._finish(db, spec, a, at, "partial", 0) for a in awards)
            retained = min(remaining, rate * self._active_seconds(db, order_id, at, deadline))
            if retained and rate:
                slots = (sum(a["slots"] for a in awards) * rate + old_rate - 1) // old_rate
                self._award(db, order_id, awards[0], at, rate, slots, retained)
            lost = remaining - retained
            # Count the surrendered part once. The retained commitment starts at this instant.
            uncharged = lost
            for a in awards:
                extra = min(
                    uncharged,
                    a["tokens"]
                    - db.execute("SELECT committed FROM performance WHERE award_id=?", (a["id"],)).fetchone()[0],
                )
                db.execute("UPDATE performance SET committed=committed+? WHERE award_id=?", (extra, a["id"]))
                uncharged -= extra
            return self._exit_result(db, spec, event_id, request, awards[0], at, "partial", old_rate - rate, lost)

    def debt_events(self, order_id):
        with self._db() as db:
            events = [
                json.loads(row[0])
                for row in db.execute("SELECT result FROM exits WHERE order_id=? ORDER BY event_id", (order_id,))
            ]
            for row in db.execute("SELECT result FROM extra_rounds WHERE order_id=? ORDER BY event_id", (order_id,)):
                events.extend(json.loads(row[0])["debts"])
            return [e for e in events if e is not None and e["assessed_rao"] > 0]

    def close(self, order_id, *, at, observed_only=False):
        """Release the order. Passive observation must use a separate book and skip performance."""
        _int(at, "at")
        if type(observed_only) is not bool:
            raise ValueError("observed_only must be boolean")
        with self._db(write=True) as db:
            spec = self._spec(db, order_id)
            if at < spec["opened_at"]:
                raise ValueError("close before opening")
            if spec["closed_at"] is None:
                if observed_only:
                    db.execute(
                        """UPDATE awards SET ended_at=?,reason='observation'
                        WHERE order_id=? AND ended_at IS NULL""",
                        (at, order_id),
                    )
                else:
                    for a in self._awards(db, order_id):
                        if a["ended_at"] is None:
                            self._finish(db, spec, a, at, "release", 0)
                db.execute("UPDATE orders SET closed_at=? WHERE order_id=?", (at, order_id))
            return self._summary(db, order_id)

    @staticmethod
    def _summary(db, order_id):
        prices = [
            row[1]
            for row in db.execute(
                """SELECT DISTINCT hotkey,price_micro FROM awards
            WHERE order_id=? ORDER BY price_micro""",
                (order_id,),
            )
        ]
        if not prices:
            return {"min_micro": None, "median_micro": None, "max_micro": None}
        middle = prices[(len(prices) - 1) // 2] + prices[len(prices) // 2]
        return {
            "min_micro": prices[0],
            "median_micro": middle // 2 if middle % 2 == 0 else middle / 2,
            "max_micro": prices[-1],
        }

    def summary(self, order_id):
        """The only cross-miner price data intended for publication, after closing."""
        with self._db() as db:
            spec = self._spec(db, order_id)
            return None if spec["closed_at"] is None else self._summary(db, order_id)
