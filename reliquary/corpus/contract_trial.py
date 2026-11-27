"""Single-turn contract trial with synthetic funding and no external side effects.

Use a fresh local ContractBook. One SQLite transaction owns the receipt, slot and
reservation. DrawLeases selects an eligible slot in memory; its JSON checkpoint
is deliberately unused. A live adapter must reconcile the bucket admission CAS
with these reservations before acknowledging work or releasing exposure.
"""

import json
import time
from contextlib import nullcontext
from fractions import Fraction

from reliquary.corpus.draw_leases import DrawLeases, Lease, free_draw_slots
from reliquary.corpus.order_contracts import ContractBook
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.token_auction import AuctionParams
from reliquary.protocol.corpus_submission import CorpusLeaseRequest, CorpusLeaseResponse
from reliquary.protocol.signatures import verify_corpus_lease_signature
from reliquary.validator import corpus_settlement

_TRIAL_PRICES = AuctionParams(ceiling=1.0, miner_alpha_per_period=1.0)


class _TransactionBook(ContractBook):
    """Run normal book operations inside the caller's already-open transaction."""

    def __init__(self, db):
        self.connection = db

    def _db(self, *, write=False):
        return nullcontext(self.connection)


class ContractTrial:
    """One signed lease per request. Receipt and verdict inputs are trusted local events."""

    def __init__(self, book, job, *, clock=time.time):
        if getattr(job, "episode", None) is not None:
            raise ValueError("trial requires a single-turn job")
        self.book, self.job, self.clock = book, job, clock
        self.order = job.job_id
        self.maximum = job.sampling.max_new_tokens * job.sampling.n
        if type(self.maximum) is not int or self.maximum <= 0:
            raise ValueError("invalid maximum payable tokens")
        shape = json.dumps(
            [job.prompt_start, job.prompt_count, job.slots_per_prompt, job.sampling.max_new_tokens, job.sampling.n]
        )
        with book._db(write=True) as db:
            db.execute("CREATE TABLE IF NOT EXISTS trial_jobs (job TEXT PRIMARY KEY, shape TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO trial_jobs VALUES (?,?)", (self.order, shape))
            if db.execute("SELECT shape FROM trial_jobs WHERE job=?", (self.order,)).fetchone()[0] != shape:
                raise ValueError("trial job shape changed")
            db.execute("""CREATE TABLE IF NOT EXISTS trial_leases (
                job TEXT NOT NULL, hotkey TEXT NOT NULL, nonce TEXT NOT NULL, issued INTEGER NOT NULL,
                id TEXT NOT NULL UNIQUE, prompt INTEGER NOT NULL, slot INTEGER NOT NULL,
                price INTEGER NOT NULL, expires REAL NOT NULL, state TEXT NOT NULL,
                receipt TEXT, tokens INTEGER, received INTEGER, PRIMARY KEY(job,hotkey,nonce),
                UNIQUE(job,receipt))""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS trial_occupied ON trial_leases(job,prompt,slot)
                WHERE state IN ('open','received','passed')""")

    def _expire(self, db, book, now):
        spec = book.order(self.order)
        if spec is None:
            raise ValueError("order not opened")
        cutoff = spec["closed_at"]
        expired = db.execute(
            """SELECT id FROM trial_leases WHERE job=? AND state='open'
            AND (expires<=? OR ?<=?)""",
            (self.order, now, cutoff, now),
        ).fetchall()
        for row in expired:
            self._release(db, book, row["id"], "expired", now)

    def _release(self, db, book, lease_id, state, now):
        if not book.cancel_grant(self.order, lease_id=lease_id, at=now):
            raise RuntimeError("trial reservation is missing")
        db.execute("UPDATE trial_leases SET state=? WHERE id=?", (state, lease_id))

    def expire(self):
        with self.book._db(write=True) as db:
            self._expire(db, _TransactionBook(db), int(self.clock()))

    @staticmethod
    def _response(row):
        if row["state"] != "open":
            return CorpusLeaseResponse(leased=False, reason=row["state"])
        return CorpusLeaseResponse(
            leased=True,
            reason="leased",
            lease_id=row["id"],
            prompt_index=row["prompt"],
            slot=row["slot"],
            expires_at=row["expires"],
        )

    def request(self, request: CorpusLeaseRequest):
        if request.job_id != self.order or request.count not in (None, 1):
            raise ValueError("trial requires one lease on the configured order")
        if not verify_corpus_lease_signature(request):
            raise ValueError("invalid lease signature")
        hotkey = request.miner_hotkey
        with self.book._db(write=True) as db:
            now = int(self.clock())
            book = _TransactionBook(db)
            self._expire(db, book, now)
            old = db.execute(
                "SELECT * FROM trial_leases WHERE job=? AND hotkey=? AND nonce=?", (self.order, hotkey, request.nonce)
            ).fetchone()
            if old:
                if old["issued"] != request.issued_at:
                    raise ValueError("conflicting lease nonce")
                return self._response(old)
            if abs(now - request.issued_at) > 60:
                raise ValueError("stale lease request")
            contract = book.contract(self.order, hotkey)
            if not contract:
                return CorpusLeaseResponse(leased=False, reason="no_contract")
            rows = db.execute("SELECT * FROM trial_leases WHERE job=?", (self.order,))
            occupied = set()
            ledger = SlotLedger(self.job.prompt_count, self.job.slots_per_prompt, prompt_start=self.job.prompt_start)
            for row in rows:
                if row["state"] in ("open", "received", "passed"):
                    occupied.add((row["prompt"], row["slot"]))
                elif row["state"] == "failed":
                    ledger.record_failure(row["prompt"], row["id"], slot=row["slot"], hotkey=row["hotkey"])
            slots = free_draw_slots(self.job, ledger, lambda prompt, slot: (prompt, slot) in occupied)
            leases = DrawLeases(self.job, clock=lambda: now, limit_per_hotkey=False)
            price = contract["price_micro"]
            # Trial's fixed unit of account is one alpha of emission per period.
            share = _TRIAL_PRICES.share_of(price / 1e6)
            lease = leases.grant(hotkey, slots, price=share, offer=share)
            if not isinstance(lease, Lease):
                return CorpusLeaseResponse(leased=False, reason="no_free_slot")
            if not book.grant(self.order, hotkey, lease_id=lease.lease_id, max_tokens=self.maximum, at=now):
                return CorpusLeaseResponse(leased=False, reason="no_reservation")
            spec = book.order(self.order)
            cutoff = min(lease.expires_at, spec["opened_at"] + spec["duration_s"])
            row = db.execute(
                """INSERT INTO trial_leases(job,hotkey,nonce,issued,id,prompt,slot,price,expires,state)
                VALUES (?,?,?,?,?,?,?,?,?,'open') RETURNING *""",
                (
                    self.order,
                    hotkey,
                    request.nonce,
                    request.issued_at,
                    lease.lease_id,
                    lease.prompt_index,
                    lease.slot,
                    price,
                    cutoff,
                ),
            ).fetchone()
            return self._response(row)

    def cancel(self, hotkey, lease_id):
        with self.book._db(write=True) as db:
            row = db.execute(
                "SELECT state FROM trial_leases WHERE job=? AND hotkey=? AND id=?", (self.order, hotkey, lease_id)
            ).fetchone()
            if not row or row["state"] != "open":
                return False
            self._release(db, _TransactionBook(db), lease_id, "cancelled", int(self.clock()))
            return True

    def receive(self, hotkey, lease_id, receipt, tokens):
        if not isinstance(receipt, str) or not receipt or type(tokens) is not int or not 0 < tokens <= self.maximum:
            raise ValueError("invalid trial receipt")
        with self.book._db(write=True) as db:
            now = int(self.clock())
            self._expire(db, _TransactionBook(db), now)
            row = db.execute(
                "SELECT * FROM trial_leases WHERE job=? AND id=? AND hotkey=?", (self.order, lease_id, hotkey)
            ).fetchone()
            if row and row["receipt"] == receipt and row["tokens"] == tokens:
                return True
            if not row or row["state"] != "open":
                return False
            if db.execute("SELECT 1 FROM trial_leases WHERE job=? AND receipt=?", (self.order, receipt)).fetchone():
                raise ValueError("conflicting receipt")
            db.execute(
                "UPDATE trial_leases SET state='received',receipt=?,tokens=?,received=? WHERE id=?",
                (receipt, tokens, now, lease_id),
            )
            return True

    def verdict(self, receipt, *, passed):
        if type(passed) is not bool:
            raise ValueError("expected a boolean verdict")
        state = "passed" if passed else "failed"
        with self.book._db(write=True) as db:
            book = _TransactionBook(db)
            row = db.execute("SELECT * FROM trial_leases WHERE job=? AND receipt=?", (self.order, receipt)).fetchone()
            if not row or row["state"] not in ("received", state):
                raise ValueError("missing receipt or conflicting verdict")
            if row["state"] == state:
                return
            if passed:
                book.record_delivery(
                    self.order,
                    row["hotkey"],
                    delivery_id=receipt,
                    tokens=row["tokens"],
                    audited_tokens=row["tokens"],
                    lease_id=row["id"],
                    at=row["received"],
                    resolved_at=int(self.clock()),
                )
                db.execute("UPDATE trial_leases SET state='passed' WHERE id=?", (row["id"],))
            else:
                self._release(db, book, row["id"], state, int(self.clock()))

    def _valuation(self):
        with self.book._db() as db:
            spec = _TransactionBook(db).order(self.order)
            rows = db.execute("SELECT * FROM trial_leases WHERE job=? AND state='passed'", (self.order,)).fetchall()
            millirao = {}
            for row in rows:
                hotkey = row["hotkey"]
                millirao[hotkey] = millirao.get(hotkey, 0) + row["tokens"] * row["price"]
            if sum(millirao.values()) > spec["budget_rao"] * 1000:
                raise RuntimeError("trial funding exceeded")
            verdicts = [
                {
                    "hotkey": r["hotkey"],
                    "token_count": r["tokens"],
                    "passed": True,
                    "received_at": r["received"],
                    "lease_price": _TRIAL_PRICES.share_of(r["price"] / 1e6),
                }
                for r in rows
            ]
            if any(corpus_settlement._period_priced(v) for v in verdicts):
                raise ValueError("legacy period pricing cannot settle contract bids")
            return corpus_settlement.rewards_for(verdicts, spec["budget_rao"] / 1e9), millirao

    def rewards(self):
        """Native pool shares from Core for one synthetic period. No payment or publication."""
        shares, _ = self._valuation()
        return shares

    def rao_breakdown(self):
        """Compare whole rao per hotkey in one snapshot, at the trial's one-alpha emission.

        The signed difference is valued minus contractual. Convert the exact binary
        share as the payment preview does, after aggregating the hotkey's receipts.
        """
        shares, millirao = self._valuation()
        result = {}
        for hotkey, value in millirao.items():
            contractual = value // 1000
            valued = int(Fraction(shares.get(hotkey, 0)) * 10**9)
            result[hotkey] = {
                "contractual_rao": contractual,
                "valued_rao": valued,
                "difference_rao": valued - contractual,
            }
        return result
