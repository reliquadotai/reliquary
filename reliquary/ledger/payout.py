"""Paying due obligations: ``transfer_stake`` from the collector's coldkey, each obligation once.

What is paid: the ledger's obligations that are ``due`` at the pass, of tasks
on the collector rail (``payout_rail: collector`` in the registry, chosen when
the task was added), for work periods from the rail's first
(``payout_rail_from``) on, valued at the rate frozen in the order's price
ceiling. Work of a native task, or from before the rail's first period, is
paid by the weights: paying it here too would pay it twice, so it is never
paid here, whatever is asked. Nothing is signed by default: ``run_pass``
plans, and signs and sends only with ``execute=True`` and a signer.

How: one ``transfer_stake`` per destination coldkey, the coldkey that owned
the hotkey at the end of the obligation's work period (noted in the journal
when first reserved, and kept for retries); it moves the collector's alpha
stake on its own hotkey to that coldkey, on the same subnet. At most
``batch_size`` of them go in one ``Utility.batch_all`` (all or nothing). A
destination owed less than ``min_transfer_rao`` in all waits for a later pass:
its obligations stay due and add up.

Once each: the journal (SQLite, one payer at a time) keeps every obligation
past ``due`` with the amount and destination it was first reserved with (a
retry sends those, never a recomputed amount), every batch's nonce, mortality
window and extrinsic hash, written before the batch is sent, and the
collector's nonce it expects next. There is one journal, on one machine: a
second, restored or rolled-back journal would not know what the first paid.
So a journal pays only once initialised at the chain's nonce (``initialise``),
and every pass stops while the collector's nonce on chain is not the one the
journal expects: another extrinsic was signed with the collector's key, or this
journal is not the one that paid. A person checks, then re-aligns it
(``resync``), with the reason.

A pass first settles every batch whose fate it does not know, from the chain:

- the collector's nonce at the finalized head has not passed the batch's: no
  extrinsic of that nonce is final. Past the batch's window it can never be,
  so its obligations are due again; inside the window the pass waits and
  sends nothing (a send that may still land is never repeated);
- it has passed: the batch's hash is looked up in its window. Not found,
  another extrinsic used the collector's key with that nonce (``foreign``).
  Found, its events say paid (``finalized``, reconciled on one
  ``StakeTransferred`` per transfer from the collector, on its hotkey and
  subnet), failed (``ExtrinsicFailed`` and no stake moved: due again), or
  neither (``unknown``). Paid with events that do not match the plan
  (``mismatch``), ``unknown`` and ``foreign`` stop every pass until a person
  settles the batch (``settle_by_hand``) with what they read on the chain;
- a read that cannot be right (a node's error decodes as an empty default:
  ``ChainReadError``) settles nothing: the pass stops, the next reads again.

Funding: only from the collector's stake at the finalized head, less what is
in flight and an operating reserve, oldest obligations first. The first
destination that does not fit whole is paid the oldest of its obligations
that fit (if they reach the minimum); nothing younger overtakes it.

Held, never paid until a person looks: an obligation valued at anything but
its frozen rate, one whose amount changed since it was first reserved, one to
the subnet owner's coldkey, and every obligation of a coldkey whose transfers
failed on chain ``MAX_ATTEMPTS`` times (the chain refuses it; ``clear_held``
once fixed). A coldkey whose transfer failed before goes in a batch of its
own, so it cannot revert others.

A surplus burn (``reliquary.ledger.surplus``) is one more kind of batch in
the same journal: it takes the collector's nonces, and gets the same care.

Receipts: every pass returns one (a dry run included), written by the caller
for publication: the plan and its digest, each batch (nonce, extrinsic hash,
block, events, fee in TAO), the collector's stake before and after, and the
totals. Every batch whose fate is known goes in a receipt: the journal marks
it once its receipt is written, and a pass lists any it finds unmarked (after
a crash).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from reliquary.ledger.obligations import (
    DUE,
    FINALIZED,
    RAO_PER_ALPHA,
    RATE_FROZEN,
    RECONCILED,
    RESERVED,
    SUBMITTED,
    Obligation,
    PayTerms,
    pay_instant,
    state_at,
)

logger = logging.getLogger(__name__)

PAYOUT_RECEIPT_SCHEMA = "reliquary/payout-receipt/v1"
DEFAULT_BATCH_SIZE = 32
# Blocks a signed batch stays valid (a power of two, as mortal eras are).
DEFAULT_ERA_PERIOD = 64
# A coldkey whose transfers failed on chain this many times (the chain refuses
# it: a coldkey reserved by a swap announcement, say, and one refused transfer
# reverts a whole batch_all) has all its obligations held until a person
# clears it. A coldkey whose transfer failed before goes alone.
MAX_ATTEMPTS = 3
MODE_DRY_RUN, MODE_EXECUTE = "dry-run", "execute"

# A batch's states. ``prepared``: reserved, not signed yet (never sent); ``submitted``: its hash is kept and it
# may be on the chain; then ``finalized`` (paid as planned), ``failed`` (included, reverted), ``expired`` (never
# included, its window over), and, until a person settles them, ``foreign`` (its nonce went to another
# extrinsic), ``unknown`` (included, its outcome unreadable) and ``mismatch`` (paid, not as planned). A person's
# decision makes ``foreign`` and ``unknown`` ``finalized`` (they landed) or ``void`` (they paid nothing).
BATCH_PREPARED, BATCH_SUBMITTED = "prepared", "submitted"
BATCH_FINALIZED, BATCH_FAILED, BATCH_EXPIRED = "finalized", "failed", "expired"
BATCH_FOREIGN, BATCH_UNKNOWN, BATCH_MISMATCH, BATCH_VOID = "foreign", "unknown", "mismatch", "void"
# Resolutions, never stored: still inside its window and not final; or the chain could not be read.
BATCH_PENDING, BATCH_UNREADABLE = "pending", "unreadable"
UNRESOLVED = (BATCH_PREPARED, BATCH_SUBMITTED)
NEEDS_PERSON = (BATCH_FOREIGN, BATCH_UNKNOWN, BATCH_MISMATCH)
# Batches whose nonce is spent on chain, by their own extrinsic or (``foreign``, ``void``) another's.
NONCE_SPENT = (BATCH_FINALIZED, BATCH_FAILED, BATCH_FOREIGN, BATCH_UNKNOWN, BATCH_MISMATCH, BATCH_VOID)
# Batches that moved the collector's stake.
PAID = (BATCH_FINALIZED, BATCH_MISMATCH)
# What an obligation's journal row says once it has left ``due``.
PAST_DUE = (RESERVED, SUBMITTED, FINALIZED, RECONCILED)
IN_FLIGHT = (RESERVED, SUBMITTED)

_TABLES = """
CREATE TABLE IF NOT EXISTS payout_obligations(
    obligation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, work_period INTEGER NOT NULL, part INTEGER NOT NULL,
    hotkey TEXT NOT NULL, coldkey TEXT NOT NULL, amount_rao INTEGER NOT NULL, state TEXT NOT NULL,
    batch_id TEXT, attempts INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS payout_obligations_batch ON payout_obligations(batch_id);
CREATE TABLE IF NOT EXISTS payout_batches(
    batch_id TEXT PRIMARY KEY, pass_day TEXT NOT NULL, state TEXT NOT NULL, transfers TEXT NOT NULL,
    nonce INTEGER, era_block INTEGER, era_period INTEGER, extrinsic_hash TEXT, block_number INTEGER,
    block_hash TEXT, fee_rao INTEGER, events TEXT, detail TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS payout_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payout_decisions(
    at REAL NOT NULL, kind TEXT NOT NULL, target TEXT NOT NULL, decision TEXT NOT NULL, evidence TEXT NOT NULL);
"""
# Columns added to a table after it was first released: (table, column, declaration). A journal that lacks
# them gets them all in one transaction (a read-only journal: in its in-memory snapshot).
_ADDED_COLUMNS = (
    ("payout_batches", "receipted", "INTEGER NOT NULL DEFAULT 0"),
    ("payout_batches", "kind", "TEXT NOT NULL DEFAULT 'transfers'"),
    ("payout_batches", "burn_rao", "INTEGER"),
)
# A batch's kind: transfers to miners (``transfers``), or one ``burn_alpha`` of the collector's surplus
# (``burn``, ``reliquary.ledger.surplus``). Both take the collector's nonces, so both share one journal.
KIND_TRANSFERS, KIND_BURN = "transfers", "burn"
# A burn whose fate a person must read may have burned: its amount stays spoken for.
BURN_IN_FLIGHT = (BATCH_PREPARED, BATCH_SUBMITTED, BATCH_UNKNOWN, BATCH_FOREIGN)
# What counts against a day's burn cap: every burn that burned, or may have.
BURN_COUNTED = (*BURN_IN_FLIGHT, BATCH_FINALIZED, BATCH_MISMATCH)


class PayoutError(RuntimeError):
    """The pass cannot go on safely; nothing more is sent."""


class ChainReadError(PayoutError):
    """A chain read that cannot be right (a node's error decodes as the storage's
    empty default, a block that cannot be fetched): nothing is settled on it."""


@dataclass(frozen=True, slots=True)
class PayoutConfig:
    netuid: int
    collector_coldkey: str
    collector_hotkey: str
    min_transfer_rao: int
    batch_size: int = DEFAULT_BATCH_SIZE
    reserve_rao: int = 0
    era_period: int = DEFAULT_ERA_PERIOD

    def __post_init__(self) -> None:
        for name in ("netuid", "min_transfer_rao", "batch_size", "reserve_rao", "era_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a whole number >= 0, got {value!r}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.era_period < 4 or self.era_period & (self.era_period - 1):
            raise ValueError(f"era_period must be a power of two of at least 4, got {self.era_period}")
        if not self.collector_coldkey or not self.collector_hotkey:
            raise ValueError("the collector's coldkey and hotkey are both needed")

    def to_dict(self) -> dict:
        return {"netuid": self.netuid, "collector_coldkey": self.collector_coldkey,
                "collector_hotkey": self.collector_hotkey, "min_transfer_rao": self.min_transfer_rao,
                "batch_size": self.batch_size, "reserve_rao": self.reserve_rao, "era_period": self.era_period}


@dataclass(frozen=True, slots=True)
class Transfer:
    """One ``transfer_stake`` of the collector's stake to ``coldkey``."""

    coldkey: str
    amount_rao: int
    obligation_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"coldkey": self.coldkey, "amount_rao": self.amount_rao,
                "obligation_ids": list(self.obligation_ids)}


@dataclass(frozen=True, slots=True)
class Inclusion:
    """Where a batch landed and what it did: its events normalized to
    ``{"module", "event", "attributes"}``. ``success``: True with an
    ``ExtrinsicSuccess`` event, False with an ``ExtrinsicFailed`` one, None when
    its events say neither (unreadable). ``fee_rao`` is TAO."""

    block_number: int
    block_hash: str
    success: bool | None
    events: tuple[Mapping, ...] = ()
    fee_rao: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SignedBatch:
    extrinsic_hash: str
    nonce: int
    era_block: int
    era_period: int
    payload: object = None


class PayoutChain(Protocol):
    """What the payer reads of the chain, and the one write: ``submit``. Every
    read raises ``ChainReadError`` rather than return a value it cannot trust."""

    async def finalized_block(self) -> int: ...

    async def collector_stake_rao(self, block: int) -> int: ...

    async def collector_nonce(self, block: int) -> int: ...

    async def hotkey_owner(self, hotkey: str, block: int) -> str | None: ...

    async def subnet_owner(self, block: int) -> tuple[str, str | None]: ...

    async def block_at(self, at: float) -> int: ...

    async def find_extrinsic(self, extrinsic_hash: str, first_block: int, last_block: int) -> Inclusion | None: ...

    async def submit(self, signed: SignedBatch) -> None: ...


class Signer(Protocol):
    coldkey: str

    async def sign(self, transfers: Sequence[Transfer], *, nonce: int, era_block: int,
                   era_period: int) -> SignedBatch: ...

    async def sign_burn(self, amount_rao: int, *, nonce: int, era_block: int, era_period: int) -> SignedBatch: ...


# -- the journal ---------------------------------------------------------------------------------------------------


def _migrate(db: sqlite3.Connection) -> None:
    db.executescript(_TABLES)
    db.execute("BEGIN IMMEDIATE")
    try:
        for table, column, declaration in _ADDED_COLUMNS:
            if column not in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


class Journal:
    """Every obligation past ``due`` and every batch, in one SQLite file that one
    payer holds at a time (an exclusive lock beside it). ``read_only`` (a dry
    run) works on a snapshot in memory, never the file and with no lock: a dry
    run settles and plans in it as an execution would, and writes nothing. A
    file that does not exist reads empty (``exists`` says so)."""

    def __init__(self, path: str | Path, *, read_only: bool = False, clock=time.time) -> None:
        self.path, self.read_only, self.clock = Path(path), read_only, clock
        self._lock = None
        self.exists = self.path.exists()
        if read_only:
            self.db = sqlite3.connect(":memory:", isolation_level=None)
            if self.exists:
                source = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
                try:
                    source.backup(self.db)
                finally:
                    source.close()
            _migrate(self.db)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = open(f"{self.path}.lock", "a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise PayoutError(f"another payer holds {self.path}") from exc
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        # On macOS fsync() leaves writes in the drive's cache; F_FULLFSYNC does not. A batch's hash must be
        # on disk before the batch is sent.
        self.db.execute("PRAGMA fullfsync=ON")
        self.db.execute("PRAGMA checkpoint_fullfsync=ON")
        _migrate(self.db)

    def snapshot(self) -> Journal:
        """A read-only copy in memory: what a dry run works on."""
        copy = object.__new__(Journal)
        copy.path, copy.read_only, copy.clock, copy._lock, copy.exists = self.path, True, self.clock, None, self.exists
        copy.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.backup(copy.db)
        _migrate(copy.db)
        return copy

    def close(self) -> None:
        self.db.close()
        if self._lock is not None and not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    @contextlib.contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _decide(self, db, kind: str, target: str, decision: str, evidence: str) -> None:
        db.execute("INSERT INTO payout_decisions(at, kind, target, decision, evidence) VALUES(?,?,?,?,?)",
                   (self.clock(), kind, target, decision, evidence))

    # -- what the journal knows of the collector's nonce

    def meta(self) -> dict[str, str]:
        return {key: value for key, value in self.db.execute("SELECT key, value FROM payout_meta")}

    def initialise(self, *, collector_coldkey: str, netuid: int, nonce: int, reason: str) -> None:
        """Start a journal at the collector's nonce on chain: the only way a
        journal comes to pay. ``reason`` is kept (a first journal, say)."""
        if self.meta():
            raise PayoutError("this journal is initialised already: re-align it with resync")
        if not str(reason).strip():
            raise PayoutError("say why this journal starts here")
        with self._transaction() as db:
            for key, value in (("collector_coldkey", str(collector_coldkey)), ("netuid", str(int(netuid))),
                               ("base_nonce", str(int(nonce))), ("initialised_at", repr(float(self.clock())))):
                db.execute("INSERT INTO payout_meta(key, value) VALUES(?, ?)", (key, value))
            self._decide(db, "initialise", str(collector_coldkey), f"nonce {int(nonce)}", str(reason))

    def resync(self, *, nonce: int, reason: str) -> int:
        """A person's decision, after checking what the collector's key signed
        that this journal did not: expect ``nonce`` next. Returns the nonce it
        expected before."""
        meta = self.meta()
        if not meta:
            raise PayoutError("this journal is not initialised")
        if not str(reason).strip():
            raise PayoutError("say what was checked on the chain")
        if self.unresolved():
            raise PayoutError("batches of unknown fate: run a pass to settle them first")
        expected = self.expected_nonce()
        if int(nonce) < expected:
            raise PayoutError(f"the chain's nonce {nonce} is below the {expected} this journal has seen spent")
        with self._transaction() as db:
            db.execute("UPDATE payout_meta SET value=? WHERE key='base_nonce'", (str(int(nonce)),))
            self._decide(db, "resync", meta["collector_coldkey"], f"nonce {expected} -> {int(nonce)}", str(reason))
        return expected

    def expected_nonce(self) -> int | None:
        """The collector's nonce on chain once every batch this journal knows
        is settled: None for a journal never initialised."""
        meta = self.meta()
        if "base_nonce" not in meta:
            return None
        marks = ",".join("?" * len(NONCE_SPENT))
        spent = self.db.execute(f"SELECT MAX(nonce) FROM payout_batches WHERE nonce IS NOT NULL AND state IN "
                                f"({marks})", NONCE_SPENT).fetchone()[0]
        return max(int(meta["base_nonce"]), 0 if spent is None else int(spent) + 1)

    def decisions(self) -> list[dict]:
        return [{"at": row[0], "kind": row[1], "target": row[2], "decision": row[3], "evidence": row[4]}
                for row in self.db.execute("SELECT at, kind, target, decision, evidence FROM payout_decisions "
                                           "ORDER BY at")]

    # -- obligations

    def rows(self, obligation_ids: Iterable[str]) -> dict[str, dict]:
        ids = list(obligation_ids)
        out: dict[str, dict] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self.db.execute(
                    f"SELECT obligation_id, coldkey, amount_rao, state, batch_id, attempts FROM payout_obligations "
                    f"WHERE obligation_id IN ({marks})", chunk):
                out[row[0]] = {"coldkey": row[1], "amount_rao": row[2], "state": row[3], "batch_id": row[4],
                               "attempts": row[5]}
        return out

    def due_rows(self) -> dict[str, dict]:
        """Every obligation the journal holds as due again (released by a
        batch that paid nothing), whatever its age."""
        return {row[0]: {"task_id": row[1], "work_period": row[2], "hotkey": row[3], "coldkey": row[4],
                         "amount_rao": row[5], "attempts": row[6]}
                for row in self.db.execute("SELECT obligation_id, task_id, work_period, hotkey, coldkey, amount_rao, "
                                           "attempts FROM payout_obligations WHERE state=?", (DUE,))}

    def paid_tasks(self) -> set[str]:
        """Every task the journal holds an obligation of."""
        return {str(row[0]) for row in self.db.execute("SELECT DISTINCT task_id FROM payout_obligations")}

    def in_flight_rao(self) -> int:
        """Transfers reserved or sent, and burns that may still burn: stake
        already spoken for."""
        marks = ",".join("?" * len(IN_FLIGHT))
        transfers = int(self.db.execute(f"SELECT COALESCE(SUM(amount_rao), 0) FROM payout_obligations "
                                        f"WHERE state IN ({marks})", IN_FLIGHT).fetchone()[0])
        burning = ",".join("?" * len(BURN_IN_FLIGHT))
        burns = int(self.db.execute(f"SELECT COALESCE(SUM(burn_rao), 0) FROM payout_batches WHERE kind=? "
                                    f"AND state IN ({burning})", (KIND_BURN, *BURN_IN_FLIGHT)).fetchone()[0])
        return transfers + burns

    def burned_on(self, day: date) -> int:
        """The rao asked of every burn reserved on UTC ``day`` that burned or
        may have: what counts against that day's cap."""
        marks = ",".join("?" * len(BURN_COUNTED))
        return int(self.db.execute(f"SELECT COALESCE(SUM(burn_rao), 0) FROM payout_batches WHERE kind=? AND "
                                   f"pass_day=? AND state IN ({marks})",
                                   (KIND_BURN, day.isoformat(), *BURN_COUNTED)).fetchone()[0])

    def reserve_burn(self, batch_id: str, day: date, amount_rao: int) -> None:
        """A burn of ``amount_rao`` of the collector's stake, prepared (not
        signed yet), counted against UTC ``day``'s cap."""
        if isinstance(amount_rao, bool) or not isinstance(amount_rao, int) or amount_rao <= 0:
            raise PayoutError(f"a burn needs a positive whole amount of rao, got {amount_rao!r}")
        now = self.clock()
        with self._transaction() as db:
            db.execute("INSERT INTO payout_batches(batch_id, pass_day, state, transfers, created_at, updated_at, kind, "
                       "burn_rao) VALUES(?,?,?,?,?,?,?,?)",
                       (batch_id, day.isoformat(), BATCH_PREPARED, "[]", now, now, KIND_BURN, amount_rao))

    def coldkey_attempts(self) -> dict[str, int]:
        """Each coldkey with a due obligation whose transfer failed on chain,
        and the most times one failed."""
        return {coldkey: int(count) for coldkey, count in self.db.execute(
            "SELECT coldkey, MAX(attempts) FROM payout_obligations WHERE state=? AND attempts > 0 GROUP BY coldkey",
            (DUE,))}

    def clear_held(self, *, evidence: str, coldkey: str | None = None, obligation_ids: Iterable[str] = ()) -> int:
        """A person's decision on a coldkey held after ``MAX_ATTEMPTS`` failed
        transfers (the destination's problem fixed), or on given obligations:
        plan them again. Returns how many due obligations were cleared."""
        ids = list(obligation_ids)
        if not str(evidence).strip():
            raise PayoutError("say what was checked")
        if coldkey is None and not ids:
            raise PayoutError("name a coldkey or obligations to clear")
        now = self.clock()
        cleared = 0
        with self._transaction() as db:
            if coldkey is not None:
                cleared += db.execute("UPDATE payout_obligations SET attempts=0, updated_at=? WHERE coldkey=? "
                                      "AND state=? AND attempts > 0", (now, coldkey, DUE)).rowcount
            for obligation_id in ids:
                cleared += db.execute("UPDATE payout_obligations SET attempts=0, updated_at=? WHERE obligation_id=? "
                                      "AND state=? AND attempts > 0", (now, obligation_id, DUE)).rowcount
            self._decide(db, "clear", coldkey or ",".join(ids), f"{cleared} obligation(s) cleared", str(evidence))
        return cleared

    # -- batches

    def batch(self, batch_id: str) -> dict | None:
        found = self.batches("batch_id = ?", (batch_id,))
        return found[0] if found else None

    def unresolved(self) -> list[dict]:
        marks = ",".join("?" * len(UNRESOLVED))
        return self.batches(f"state IN ({marks})", UNRESOLVED)

    def needs_person(self) -> list[dict]:
        """Batches waiting on a person's decision (``settle_by_hand``): until
        each is settled, no pass sends anything."""
        marks = ",".join("?" * len(NEEDS_PERSON))
        return self.batches(f"state IN ({marks})", NEEDS_PERSON)

    def unreported(self) -> list[dict]:
        """Batches whose fate is known and that no written receipt lists yet."""
        marks = ",".join("?" * len(UNRESOLVED))
        return self.batches(f"receipted = 0 AND state NOT IN ({marks})", UNRESOLVED)

    def mark_receipted(self, batch_ids: Iterable[str]) -> None:
        """A receipt listing these batches was written."""
        marks = ",".join("?" * len(UNRESOLVED))
        with self._transaction() as db:
            for batch_id in batch_ids:
                db.execute(f"UPDATE payout_batches SET receipted=1 WHERE batch_id=? AND state NOT IN ({marks})",
                           (batch_id, *UNRESOLVED))

    def known_hashes(self) -> set[str]:
        return {str(row[0]).lower() for row in self.db.execute(
            "SELECT extrinsic_hash FROM payout_batches WHERE extrinsic_hash IS NOT NULL")}

    def settle_by_hand(self, batch_id: str, *, landed: bool, evidence: str) -> None:
        """A person's decision on a batch that stopped the payer, after reading
        the chain: it ``landed`` (its transfers were made: paid) or not (it
        paid nothing: its obligations are due again). A ``mismatch`` batch
        landed by definition: it is settled once its destinations are checked."""
        batch = self.batch(batch_id)
        if batch is None or batch["state"] not in NEEDS_PERSON:
            raise PayoutError(f"batch {batch_id} is not waiting on a decision")
        if not str(evidence).strip():
            raise PayoutError("say what was read on the chain (extrinsic, block, events)")
        if batch["state"] == BATCH_MISMATCH and not landed:
            raise PayoutError(f"batch {batch_id} landed (its events are on chain): settle it as landed once its "
                              "destinations are checked")
        now = self.clock()
        with self._transaction() as db:
            if landed:
                db.execute("UPDATE payout_batches SET state=?, detail=?, updated_at=? WHERE batch_id=?",
                           (BATCH_FINALIZED, f"{batch['detail']}; settled by hand as landed: {evidence}", now,
                            batch_id))
                db.execute("UPDATE payout_obligations SET state=?, updated_at=? WHERE batch_id=? AND state IN (?,?)",
                           (FINALIZED, now, batch_id, RESERVED, SUBMITTED))
            else:
                db.execute("UPDATE payout_batches SET state=?, detail=?, updated_at=? WHERE batch_id=?",
                           (BATCH_VOID, f"{batch['detail']}; settled by hand as paying nothing: {evidence}", now,
                            batch_id))
                db.execute("UPDATE payout_obligations SET state=?, batch_id=NULL, updated_at=? WHERE batch_id=?",
                           (DUE, now, batch_id))
            self._decide(db, "settle", batch_id, "landed" if landed else "paid nothing", str(evidence))

    def batches(self, where: str = "1", args: Sequence = ()) -> list[dict]:
        columns = ("batch_id", "pass_day", "state", "transfers", "nonce", "era_block", "era_period",
                   "extrinsic_hash", "block_number", "block_hash", "fee_rao", "events", "detail", "created_at",
                   "receipted", "kind", "burn_rao")
        out = []
        for row in self.db.execute(f"SELECT {', '.join(columns)} FROM payout_batches WHERE {where} "
                                   f"ORDER BY created_at, batch_id", tuple(args)):
            item = dict(zip(columns, row))
            item["transfers"] = json.loads(item["transfers"])
            item["events"] = None if item["events"] is None else json.loads(item["events"])
            out.append(item)
        return out

    def reserve(self, batch_id: str, pass_day: date, transfers: Sequence[Transfer],
                obligations: Mapping[str, Obligation], coldkeys: Mapping[str, str]) -> None:
        """Every obligation of ``transfers`` from due to reserved, under a new
        batch, in one transaction. An obligation the journal holds past due,
        or with another amount or destination than it was first reserved
        with, refuses the whole batch."""
        now = self.clock()
        with self._transaction() as db:
            for transfer in transfers:
                for obligation_id in transfer.obligation_ids:
                    o = obligations[obligation_id]
                    changed = db.execute(
                        "INSERT INTO payout_obligations(obligation_id, task_id, work_period, part, hotkey, coldkey, "
                        "amount_rao, state, batch_id, attempts, updated_at) VALUES(?,?,?,?,?,?,?,?,?,0,?) "
                        "ON CONFLICT(obligation_id) DO UPDATE SET state=excluded.state, batch_id=excluded.batch_id, "
                        "updated_at=excluded.updated_at WHERE payout_obligations.state=? "
                        "AND payout_obligations.amount_rao=excluded.amount_rao "
                        "AND payout_obligations.coldkey=excluded.coldkey",
                        (obligation_id, o.task_id, o.work_period, o.part, o.hotkey, coldkeys[obligation_id],
                         o.amount_rao, RESERVED, batch_id, now, DUE)).rowcount
                    if changed != 1:
                        raise PayoutError(f"obligation {obligation_id} is past due in the journal, or its amount "
                                          "or destination is not the one first reserved")
            db.execute("INSERT INTO payout_batches(batch_id, pass_day, state, transfers, created_at, updated_at) "
                       "VALUES(?,?,?,?,?,?)",
                       (batch_id, pass_day.isoformat(), BATCH_PREPARED,
                        json.dumps([t.to_dict() for t in transfers], sort_keys=True), now, now))

    def mark_submitted(self, batch_id: str, signed: SignedBatch) -> None:
        """Before the batch is sent: what will let a later pass find it."""
        now = self.clock()
        with self._transaction() as db:
            changed = db.execute("UPDATE payout_batches SET state=?, nonce=?, era_block=?, era_period=?, "
                                 "extrinsic_hash=?, updated_at=? WHERE batch_id=? AND state=?",
                                 (BATCH_SUBMITTED, signed.nonce, signed.era_block, signed.era_period,
                                  signed.extrinsic_hash, now, batch_id, BATCH_PREPARED)).rowcount
            if changed != 1:
                raise PayoutError(f"batch {batch_id} is not prepared")
            db.execute("UPDATE payout_obligations SET state=?, updated_at=? WHERE batch_id=?",
                       (SUBMITTED, now, batch_id))

    def _land(self, db, batch_id: str, state: str, inclusion: Inclusion | None, detail: str, now: float) -> None:
        db.execute("UPDATE payout_batches SET state=?, block_number=?, block_hash=?, fee_rao=?, events=?, "
                   "detail=?, updated_at=? WHERE batch_id=?",
                   (state, None if inclusion is None else inclusion.block_number,
                    None if inclusion is None else inclusion.block_hash,
                    None if inclusion is None else inclusion.fee_rao,
                    None if inclusion is None else json.dumps(list(inclusion.events), sort_keys=True, default=str),
                    detail, now, batch_id))

    def mark_paid(self, batch_id: str, inclusion: Inclusion, *, reconciled: bool, detail: str) -> None:
        """Paid. Events that do not match the plan make it ``mismatch``: paid,
        and a stop for a person."""
        now = self.clock()
        with self._transaction() as db:
            self._land(db, batch_id, BATCH_FINALIZED if reconciled else BATCH_MISMATCH, inclusion, detail, now)
            db.execute("UPDATE payout_obligations SET state=?, updated_at=? WHERE batch_id=?",
                       (RECONCILED if reconciled else FINALIZED, now, batch_id))

    def mark_released(self, batch_id: str, state: str, detail: str, inclusion: Inclusion | None = None) -> None:
        """The batch paid nothing (failed, expired or never sent): its
        obligations are due again, their destinations and amounts kept. Only
        a failure on chain counts as a failed attempt."""
        now = self.clock()
        with self._transaction() as db:
            self._land(db, batch_id, state, inclusion, detail, now)
            db.execute("UPDATE payout_obligations SET state=?, batch_id=NULL, attempts=attempts+?, updated_at=? "
                       "WHERE batch_id=?", (DUE, 1 if state == BATCH_FAILED else 0, now, batch_id))

    def mark_unknown(self, batch_id: str, inclusion: Inclusion, detail: str) -> None:
        """Included, its outcome unreadable: its obligations stay sent (never
        paid twice) until a person decides."""
        now = self.clock()
        with self._transaction() as db:
            self._land(db, batch_id, BATCH_UNKNOWN, inclusion, detail, now)

    def mark_foreign(self, batch_id: str, detail: str) -> None:
        """The batch's nonce was used by another extrinsic: its obligations stay
        sent (never paid twice) until a person decides."""
        now = self.clock()
        with self._transaction() as db:
            db.execute("UPDATE payout_batches SET state=?, detail=?, updated_at=? WHERE batch_id=?",
                       (BATCH_FOREIGN, detail, now, batch_id))


# -- planning --------------------------------------------------------------------------------------------------------


def _age(obligation: Obligation) -> tuple:
    return (obligation.due_at, obligation.earned_at, obligation.work_period, obligation.part, obligation.task_id,
            obligation.hotkey, obligation.obligation_id)


def plan_transfers(candidates: Sequence[tuple[Obligation, str]], *, available_rao: int,
                   min_transfer_rao: int) -> tuple[list[Transfer], list[Obligation], list[Obligation]]:
    """(transfers, deferred, unfunded) of (obligation, destination coldkey)
    pairs. A destination owed less than ``min_transfer_rao`` in all is
    deferred. The others are funded in the order of their oldest obligation
    while ``available_rao`` lasts. The first that does not fit whole is paid
    the oldest of its obligations that fit, if they reach the minimum, and
    funding stops there: the rest of it, and every destination after, are
    unfunded."""
    groups: dict[str, list[Obligation]] = defaultdict(list)
    for obligation, coldkey in candidates:
        groups[coldkey].append(obligation)
    transfers: list[Transfer] = []
    deferred: list[Obligation] = []
    unfunded: list[Obligation] = []
    left = int(available_rao)
    stopped = False
    for coldkey, group in sorted(groups.items(), key=lambda kv: (min(_age(o) for o in kv[1]), kv[0])):
        group.sort(key=_age)
        total = sum(o.amount_rao for o in group)
        if total < min_transfer_rao:
            deferred.extend(group)
            continue
        if stopped:
            unfunded.extend(group)
            continue
        if total <= left:
            transfers.append(Transfer(coldkey=coldkey, amount_rao=total,
                                      obligation_ids=tuple(o.obligation_id for o in group)))
            left -= total
            continue
        stopped = True
        taken, sum_taken = 0, 0
        while taken < len(group) and sum_taken + group[taken].amount_rao <= left:
            sum_taken += group[taken].amount_rao
            taken += 1
        if taken and sum_taken >= min_transfer_rao:
            transfers.append(Transfer(coldkey=coldkey, amount_rao=sum_taken,
                                      obligation_ids=tuple(o.obligation_id for o in group[:taken])))
            left -= sum_taken
            unfunded.extend(group[taken:])
        else:
            unfunded.extend(group)
    return transfers, deferred, unfunded


def plan_digest(transfers: Sequence[Transfer]) -> str:
    """What the owner confirms besides the total: the destinations, amounts
    and obligations of a plan, hashed (16 hex digits)."""
    body = json.dumps(sorted([t.coldkey, int(t.amount_rao), sorted(t.obligation_ids)] for t in transfers),
                      separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def batches_of(transfers: Sequence[Transfer], size: int, *, alone: Collection[str] = ()) -> list[list[Transfer]]:
    """``transfers`` in batches of at most ``size``; a transfer to a coldkey in
    ``alone`` (one whose transfer failed before) in a batch of its own, so it
    cannot revert the others."""
    solo = [[t] for t in transfers if t.coldkey in alone]
    rest = [t for t in transfers if t.coldkey not in alone]
    return [list(rest[start:start + size]) for start in range(0, len(rest), size)] + solo


def _event_fields(event: Mapping, names: Sequence[str]) -> list | None:
    attributes = event.get("attributes")
    if isinstance(attributes, Mapping):
        return [attributes.get(name) for name in names]
    if isinstance(attributes, (list, tuple)) and len(attributes) >= len(names):
        return list(attributes[:len(names)])
    return None


def reconcile_events(transfers: Sequence[Mapping], inclusion: Inclusion, *, collector_coldkey: str,
                     collector_hotkey: str, netuid: int) -> tuple[bool, str]:
    """Whether a paid batch's events hold one ``StakeTransferred`` per transfer,
    from the collector, on its hotkey and subnet, to each destination. The
    event's fields are (origin coldkey, destination coldkey, hotkey, origin
    netuid, destination netuid, amount); its amount is a TAO value, not the
    alpha sent, so it is kept in the receipt and not compared."""
    want = Counter(str(t["coldkey"]) for t in transfers)
    got: Counter = Counter()
    for event in inclusion.events:
        if event.get("module") != "SubtensorModule" or event.get("event") != "StakeTransferred":
            continue
        fields = _event_fields(event, ("origin_coldkey", "destination_coldkey", "hotkey", "origin_netuid",
                                       "destination_netuid"))
        if fields is None:
            return False, f"a StakeTransferred event could not be read: {event.get('attributes')!r}"
        origin, destination, hotkey = (None if f is None else str(f) for f in fields[:3])
        try:
            subnets = {int(fields[3]), int(fields[4])}
        except (TypeError, ValueError):
            return False, f"a StakeTransferred event's subnets could not be read: {fields!r}"
        if origin != collector_coldkey or hotkey != collector_hotkey or subnets != {int(netuid)}:
            return False, f"a StakeTransferred event is not the collector's on subnet {netuid}: {fields!r}"
        got[destination] += 1
    if got != want:
        return False, f"StakeTransferred destinations {dict(got)} differ from the batch's {dict(want)}"
    return True, f"{sum(got.values())} StakeTransferred event(s), one per transfer"


def reconcile_burn(batch: Mapping, inclusion: Inclusion, *, collector_coldkey: str, collector_hotkey: str,
                   netuid: int) -> tuple[bool, str]:
    """Whether a burn's events hold one ``AlphaBurned`` of the collector's stake
    on its hotkey and subnet, of at most the amount asked (the chain burns
    what is there, at most). Its fields are (coldkey, hotkey, amount, netuid)."""
    burned = [e for e in inclusion.events
              if e.get("module") == "SubtensorModule" and e.get("event") == "AlphaBurned"]
    if len(burned) != 1:
        return False, f"{len(burned)} AlphaBurned event(s), one expected"
    fields = _event_fields(burned[0], ("coldkey", "hotkey", "amount", "netuid"))
    if fields is None:
        return False, f"an AlphaBurned event could not be read: {burned[0].get('attributes')!r}"
    coldkey, hotkey, amount, subnet = fields
    if str(coldkey) != collector_coldkey or str(hotkey) != collector_hotkey:
        return False, f"the AlphaBurned event is not the collector's: {fields!r}"
    try:
        amount, subnet = int(amount), int(subnet)
    except (TypeError, ValueError):
        return False, f"the AlphaBurned amount or subnet could not be read: {fields!r}"
    asked = int(batch.get("burn_rao") or 0)
    if subnet != int(netuid) or amount > asked:
        return False, f"AlphaBurned {amount} rao on subnet {subnet}: not at most the {asked} asked on {netuid}"
    return True, f"AlphaBurned {amount} rao of the {asked} asked"


def burned_rao(batch: Mapping) -> int:
    """What a final burn burned: its reconciled ``AlphaBurned`` amount, else 0
    (a burn the chain's events do not show is never counted as burned)."""
    if batch.get("state") != BATCH_FINALIZED:
        return 0
    for event in batch.get("events") or []:
        if event.get("module") == "SubtensorModule" and event.get("event") == "AlphaBurned":
            fields = _event_fields(event, ("coldkey", "hotkey", "amount", "netuid"))
            try:
                return int(fields[2]) if fields else 0
            except (TypeError, ValueError):
                return 0
    return 0


def moved_stake(inclusion: Inclusion) -> bool:
    """Whether a batch's events show the collector's stake moving."""
    return any(e.get("module") == "SubtensorModule" and e.get("event") in ("StakeTransferred", "AlphaBurned")
               for e in inclusion.events)


def failed_on_chain(inclusion: Inclusion) -> bool:
    return any(e.get("module") == "System" and e.get("event") == "ExtrinsicFailed" for e in inclusion.events)


# -- one pass ---------------------------------------------------------------------------------------------------------


async def resolve_batch(batch: Mapping, chain: PayoutChain) -> tuple[str, Inclusion | None, str]:
    """A batch whose fate is unknown, from the chain: ``finalized``, ``failed``,
    ``expired``, ``foreign``, ``unknown``; or, settling nothing, ``pending``
    (inside its window) or ``unreadable``."""
    if batch.get("extrinsic_hash") is None:
        # The hash is kept before anything is sent: without one, nothing was.
        return BATCH_EXPIRED, None, "never sent"
    nonce = int(batch["nonce"])
    era = int(batch["era_block"])
    last = era + int(batch["era_period"])
    try:
        head = await chain.finalized_block()
        spent = await chain.collector_nonce(head)
        if spent < nonce:
            # The batch was signed at the nonce the chain had then: it can only have grown since.
            return BATCH_UNREADABLE, None, (f"the collector's nonce reads {spent} at finalized block {head}, below "
                                            f"the batch's {nonce}: not a state the chain can be in")
        if spent == nonce:
            if head > last:
                return BATCH_EXPIRED, None, f"nonce {nonce} unused at finalized block {head}, past its window ({last})"
            return BATCH_PENDING, None, f"nonce {nonce} unused at finalized block {head}, window open until {last}"
        found = await chain.find_extrinsic(str(batch["extrinsic_hash"]), era, min(head, last))
    except ChainReadError as exc:
        return BATCH_UNREADABLE, None, f"the chain could not be read: {exc}"
    if found is None:
        return BATCH_FOREIGN, None, (f"nonce {nonce} was spent by an extrinsic other than {batch['extrinsic_hash']}: "
                                     f"blocks {era} to {min(head, last)} do not hold it")
    if found.success is True:
        return BATCH_FINALIZED, found, f"included in block {found.block_number}"
    if found.success is False and failed_on_chain(found) and not moved_stake(found):
        return BATCH_FAILED, found, f"failed in block {found.block_number}: {found.error}"
    return BATCH_UNKNOWN, found, (f"included in block {found.block_number}, but its events do not say whether it "
                                  "paid: a person must read the block")


def _settle(journal: Journal, batch: Mapping, state: str, inclusion: Inclusion | None, detail: str,
            config: PayoutConfig) -> dict:
    """Write a resolution down; what the receipt says of it."""
    if state == BATCH_FINALIZED:
        if batch.get("kind") == KIND_BURN:
            reconciled, why = reconcile_burn(batch, inclusion, collector_coldkey=config.collector_coldkey,
                                             collector_hotkey=config.collector_hotkey, netuid=config.netuid)
        else:
            reconciled, why = reconcile_events(batch["transfers"], inclusion,
                                               collector_coldkey=config.collector_coldkey,
                                               collector_hotkey=config.collector_hotkey, netuid=config.netuid)
        detail = f"{detail}; {why}"
        if not reconciled:
            state = BATCH_MISMATCH
            detail = f"{detail}: landed, not as planned; a person must check where the stake went"
        journal.mark_paid(batch["batch_id"], inclusion, reconciled=reconciled, detail=detail)
    elif state in (BATCH_FAILED, BATCH_EXPIRED):
        journal.mark_released(batch["batch_id"], state, detail, inclusion)
    elif state == BATCH_FOREIGN:
        journal.mark_foreign(batch["batch_id"], detail)
    elif state == BATCH_UNKNOWN:
        journal.mark_unknown(batch["batch_id"], inclusion, detail)
    return _batch_receipt(batch, state, inclusion, detail)


def _batch_receipt(batch: Mapping, state: str, inclusion: Inclusion | None, detail: str) -> dict:
    return {"batch_id": batch["batch_id"], "kind": batch.get("kind") or KIND_TRANSFERS,
            "burn_rao": batch.get("burn_rao"), "state": state, "nonce": batch.get("nonce"),
            "extrinsic_hash": batch.get("extrinsic_hash"), "era_block": batch.get("era_block"),
            "era_period": batch.get("era_period"),
            "block_number": None if inclusion is None else inclusion.block_number,
            "block_hash": None if inclusion is None else inclusion.block_hash,
            "fee_tao_rao": None if inclusion is None else inclusion.fee_rao,
            "events": None if inclusion is None else list(inclusion.events),
            "transfers": list(batch["transfers"]), "detail": detail}


def _row_receipt(row: Mapping) -> dict:
    """A journal batch as a receipt lists it."""
    return {"batch_id": row["batch_id"], "kind": row.get("kind") or KIND_TRANSFERS, "burn_rao": row.get("burn_rao"),
            "state": row["state"], "nonce": row.get("nonce"),
            "extrinsic_hash": row.get("extrinsic_hash"), "era_block": row.get("era_block"),
            "era_period": row.get("era_period"), "block_number": row.get("block_number"),
            "block_hash": row.get("block_hash"), "fee_tao_rao": row.get("fee_rao"), "events": row.get("events"),
            "transfers": list(row["transfers"]), "detail": row.get("detail")}


async def destinations(obligations: Sequence[Obligation], chain: PayoutChain,
                       journal: Journal) -> tuple[dict[str, str], list[Obligation]]:
    """Each obligation's destination coldkey: the one the journal noted when it
    was first reserved, else the hotkey's owner at the end of its work period
    (``earn_at``). The obligations whose hotkey had no owner then are
    returned apart."""
    noted = journal.rows(o.obligation_id for o in obligations)
    blocks: dict[float, int] = {}
    owners: dict[tuple[str, int], str | None] = {}
    found: dict[str, str] = {}
    ownerless: list[Obligation] = []
    for obligation in obligations:
        row = noted.get(obligation.obligation_id)
        if row is not None:
            found[obligation.obligation_id] = row["coldkey"]
            continue
        if obligation.earn_at not in blocks:
            blocks[obligation.earn_at] = await chain.block_at(obligation.earn_at)
        key = (obligation.hotkey, blocks[obligation.earn_at])
        if key not in owners:
            owners[key] = await chain.hotkey_owner(*key)
        if owners[key]:
            found[obligation.obligation_id] = str(owners[key])
        else:
            ownerless.append(obligation)
    return found, ownerless


def collector_rails(collector_tasks: Mapping[str, int] | Collection[str]) -> dict[str, int]:
    """Each collector-rail task and its first work period on the rail (0 when
    only the task is named)."""
    if isinstance(collector_tasks, Mapping):
        return {str(task): int(start or 0) for task, start in collector_tasks.items()}
    return {str(task): 0 for task in collector_tasks}


def terms_dict(terms: PayTerms) -> dict:
    return {"lag_days": terms.lag_days, "pay_at_seconds": terms.pay_at_seconds, "hold_seconds": terms.hold_seconds}


async def run_pass(obligations: Iterable[Obligation], *, journal: Journal, execute: bool = False,
                   **kwargs) -> dict:
    """One pass: settle what is unknown, check the journal against the chain,
    then plan, then (``execute``) pay. Returns the receipt. A dry run works on
    a snapshot of the journal: the file is never written and nothing is
    signed. Executing pays only when ``confirm_rao`` and ``confirm_plan`` name
    the pass's total and plan digest exactly (the owner's go, read off a dry
    run); otherwise it stops before reserving anything, and says both."""
    if execute or journal.read_only:
        return await _pass(obligations, journal=journal, execute=execute, **kwargs)
    snapshot = journal.snapshot()
    try:
        return await _pass(obligations, journal=snapshot, execute=False, **kwargs)
    finally:
        snapshot.close()


async def recover(journal: Journal, chain: PayoutChain, config: PayoutConfig, receipt: dict, *,
                  execute: bool) -> tuple[int, int]:
    """Every batch (transfers or burn) whose fate is unknown, from the chain,
    before anything new is sent; each listed in ``receipt["recovered"]``. A
    dry run settles them in the journal's snapshot, so what it plans next is
    what an execution would. Returns (pending, unreadable): batches still in
    their window (nothing new may be sent) and batches the chain could not
    tell (nothing is settled or sent)."""
    pending = unreadable = 0
    for batch in journal.unresolved():
        state, inclusion, detail = await resolve_batch(batch, chain)
        if state in (BATCH_PENDING, BATCH_UNREADABLE):
            pending += state == BATCH_PENDING
            unreadable += state == BATCH_UNREADABLE
            receipt["recovered"].append(_batch_receipt(batch, state, None, detail))
            continue
        entry = _settle(journal, batch, state, inclusion, detail, config)
        if not execute:
            entry["detail"] = f"(would settle) {entry['detail']}"
        receipt["recovered"].append(entry)
    return pending, unreadable


def _stop(receipt: dict, why: str) -> None:
    if receipt["stopped"] is None:
        receipt["stopped"] = why


async def check_journal(journal: Journal, chain: PayoutChain, config: PayoutConfig, receipt: dict, *,
                        pending: int, unreadable: int) -> tuple[int, int | None]:
    """What stops every send: a batch waiting on a person, one the chain could
    not tell, a journal never initialised or kept for another collector, and
    a collector's nonce that is not the one the journal expects. Returns (the
    finalized head, the nonce the journal expects)."""
    waiting = journal.needs_person()
    if waiting:
        _stop(receipt, f"batch(es) {', '.join(b['batch_id'] + ' (' + b['state'] + ')' for b in waiting)} wait on "
                       "a person: nothing is sent until each is settled by hand (reliquary ledger settle), with what "
                       "the chain shows")
    if unreadable:
        _stop(receipt, f"{unreadable} batch(es) could not be read from the chain: nothing is settled or sent; "
                       "run again")
    meta = journal.meta()
    head = await chain.finalized_block()
    spent = await chain.collector_nonce(head)
    expected = journal.expected_nonce()
    receipt["nonce"] = {"chain": spent, "journal": expected, "at_block": head}
    if not meta:
        _stop(receipt, f"the journal {journal.path} is not initialised" + ("" if journal.exists else " (no such file)")
                       + ": it knows nothing of what was paid. Initialise it once, at the chain's nonce, with "
                         "reliquary ledger journal-init; restore the journal that paid instead, if there is one")
    elif meta.get("collector_coldkey") != config.collector_coldkey or meta.get("netuid") != str(config.netuid):
        _stop(receipt, f"the journal belongs to collector {meta.get('collector_coldkey')} on subnet "
                       f"{meta.get('netuid')}, not {config.collector_coldkey} on {config.netuid}")
    elif not pending and spent != expected:
        _stop(receipt, f"the collector's nonce is {spent} at block {head}; this journal expects {expected}: the "
                       "collector's key signed an extrinsic this journal did not, or this journal is not the one "
                       "that paid (restored, copied or new). Nothing is sent. Check the collector's extrinsics "
                       "(reliquary ledger find-nonce), then re-align with reliquary ledger journal-init --resync "
                       "--reason ...")
    if pending:
        _stop(receipt, f"{pending} batch(es) sent earlier may still land: nothing new is sent until they are final "
                       "or expired")
    return head, expected


async def check_collector(chain: PayoutChain, config: PayoutConfig, receipt: dict, head: int) -> str:
    """The collector is not the subnet owner's (coldkey or hotkey) and its
    coldkey holds its hotkey; otherwise nothing is sent. Returns the subnet
    owner's coldkey."""
    owner_coldkey, owner_hotkey = await chain.subnet_owner(head)
    hotkey_holder = await chain.hotkey_owner(config.collector_hotkey, head)
    receipt["subnet_owner"] = {"coldkey": owner_coldkey, "hotkey": owner_hotkey}
    if config.collector_coldkey == owner_coldkey or config.collector_hotkey == owner_hotkey:
        _stop(receipt, "the collector is the subnet owner's (coldkey or hotkey): the collector must be a key of its "
                       "own")
    elif hotkey_holder != config.collector_coldkey:
        _stop(receipt, f"the collector's hotkey belongs to {hotkey_holder}, not to the collector's coldkey")
    return owner_coldkey


async def send_batch(journal: Journal, chain: PayoutChain, config: PayoutConfig, batch_id: str,
                     signed: SignedBatch, *, wait_seconds: float, poll_seconds: float, clock,
                     sleep) -> tuple[dict, bool]:
    """Keep a reserved batch's hash, send it, and read its fate from the chain
    until it is known or ``wait_seconds`` pass. Returns (its receipt entry,
    whether its fate is known)."""
    journal.mark_submitted(batch_id, signed)
    try:
        await chain.submit(signed)
    except Exception as exc:  # noqa: BLE001 - the send's fate is read from the chain, never assumed
        logger.warning("batch %s: send raised %r; reading the chain for its fate", batch_id, exc)
    deadline = clock() + wait_seconds
    while True:
        batch = journal.batch(batch_id)
        state, inclusion, detail = await resolve_batch(batch, chain)
        if state not in (BATCH_PENDING, BATCH_UNREADABLE):
            return _settle(journal, batch, state, inclusion, detail, config), True
        if clock() >= deadline:
            return _batch_receipt(batch, state, None, f"not final after {wait_seconds:.0f} s ({detail}): the next "
                                                    "pass settles it"), False
        await sleep(poll_seconds)


async def _pass(obligations: Iterable[Obligation], *, journal: Journal, chain: PayoutChain, config: PayoutConfig,
                collector_tasks: Mapping[str, int] | Collection[str], pass_day: date, terms: PayTerms, now: float,
                execute: bool = False, signer: Signer | None = None, confirm_rao: int | None = None,
                confirm_plan: str | None = None, wait_seconds: float = 600.0, poll_seconds: float = 12.0,
                clock=time.time, sleep=asyncio.sleep) -> dict:
    if execute:
        if signer is None:
            raise PayoutError("executing a pass needs a signer")
        if signer.coldkey != config.collector_coldkey:
            raise PayoutError("the signer's coldkey is not the collector's")
        if journal.read_only:
            raise PayoutError("executing a pass needs the journal open for writing")
    obligations = list(obligations)
    rails = collector_rails(collector_tasks)
    at = pay_instant(pass_day, terms)
    receipt: dict = {"schema": PAYOUT_RECEIPT_SCHEMA, "mode": MODE_EXECUTE if execute else MODE_DRY_RUN,
                     "pass_day": pass_day.isoformat(), "pass_at": at, "as_of": float(now), **config.to_dict(),
                     "terms": terms_dict(terms), "collector_rails": rails,
                     "unreported": [_row_receipt(b) for b in journal.unreported()],
                     "recovered": [], "batches": [], "stopped": None}

    # 1. What is not known yet, from the chain; then the journal against the chain; then the collector.
    pending, unreadable = await recover(journal, chain, config, receipt, execute=execute)
    head, expected = await check_journal(journal, chain, config, receipt, pending=pending, unreadable=unreadable)
    owner_coldkey = await check_collector(chain, config, receipt, head)

    # 2. What is due at this pass and the collector pays.
    due = [o for o in obligations if o.due_at <= at and state_at(o, now) == DUE]
    native = [o for o in due if o.task_id not in rails or o.work_period < rails[o.task_id]]
    mine = [o for o in due if o.task_id in rails and o.work_period >= rails[o.task_id]]
    unfrozen = [o for o in mine if o.rate_source != RATE_FROZEN]
    mine = [o for o in mine if o.rate_source == RATE_FROZEN]
    rows = journal.rows(o.obligation_id for o in mine)
    past = [o for o in mine if rows.get(o.obligation_id, {}).get("state", DUE) != DUE]
    open_ = [o for o in mine if rows.get(o.obligation_id, {}).get("state", DUE) == DUE]
    changed = [o for o in open_
               if o.obligation_id in rows and int(rows[o.obligation_id]["amount_rao"]) != o.amount_rao]
    changed_ids = {o.obligation_id for o in changed}
    open_ = [o for o in open_ if o.obligation_id not in changed_ids]
    listed = {o.obligation_id for o in obligations}
    journal_only = {oid: row for oid, row in journal.due_rows().items() if oid not in listed}
    found, ownerless = await destinations(open_, chain, journal)
    tried = journal.coldkey_attempts()
    refused = [o for o in open_ if tried.get(found.get(o.obligation_id), 0) >= MAX_ATTEMPTS]
    to_owner = [o for o in open_ if found.get(o.obligation_id) == owner_coldkey]
    held_ids = {o.obligation_id for o in refused + to_owner}
    payable = [o for o in open_ if o.obligation_id in found and o.obligation_id not in held_ids]
    stake = await chain.collector_stake_rao(head)
    in_flight = journal.in_flight_rao()
    available = max(0, stake - in_flight - config.reserve_rao)
    by_id = {o.obligation_id: o for o in payable}
    transfers, deferred, unfunded = plan_transfers([(o, found[o.obligation_id]) for o in payable],
                                                   available_rao=available, min_transfer_rao=config.min_transfer_rao)
    alone = {t.coldkey for t in transfers if tried.get(t.coldkey, 0) > 0}
    digest = plan_digest(transfers)
    receipt.update({
        "finalized_block": head, "collector_stake_rao": stake, "in_flight_rao": in_flight,
        "available_rao": available,
        "due": _count(due), "native_rail": _count(native), "already_past_due": _count(past),
        "deferred_below_minimum": _count(deferred), "unfunded": _count(unfunded), "ownerless": _count(ownerless),
        "held": {"rate_not_frozen": _listed(unfrozen), "amount_changed": _listed(changed),
                 "to_subnet_owner": _listed(to_owner), "coldkey_refused": _listed(refused)},
        "journal_only": {"obligations": len(journal_only),
                         "rao": sum(int(r["amount_rao"]) for r in journal_only.values()),
                         "obligation_ids": sorted(journal_only)},
        "planned": {"transfers": len(transfers), "rao": sum(t.amount_rao for t in transfers),
                    "obligations": sum(len(t.obligation_ids) for t in transfers)},
        "planned_transfers": [t.to_dict() for t in transfers], "plan_digest": digest, "alone": sorted(alone),
        "confirmed_rao": confirm_rao, "confirmed_plan": confirm_plan})
    if receipt["stopped"] is not None or not execute:
        return _close(receipt)
    planned = receipt["planned"]["rao"]
    if planned and (confirm_rao != planned or confirm_plan != digest):
        receipt["stopped"] = (f"this pass would pay {planned} rao ({planned / RAO_PER_ALPHA:.9f} alpha) in "
                              f"{len(transfers)} transfer(s), plan {digest}: confirm both with --confirm-rao "
                              f"{planned} --confirm-plan {digest}")
        return _close(receipt)

    # 3. Pay: one batch at a time, each final before the next takes the following nonce.
    coldkeys = {oid: t.coldkey for t in transfers for oid in t.obligation_ids}
    for group in batches_of(transfers, config.batch_size, alone=alone):
        tip = await chain.finalized_block()
        nonce = await chain.collector_nonce(tip)
        if nonce != expected:
            receipt["stopped"] = (f"the collector's nonce is {nonce} at block {tip}; this journal expects {expected}: "
                                  "another extrinsic was signed with the collector's key. Nothing more is sent")
            break
        batch_id = f"{pass_day.isoformat()}-{uuid.uuid4().hex[:12]}"
        journal.reserve(batch_id, pass_day, group, by_id, coldkeys)
        signed = await signer.sign(group, nonce=nonce, era_block=tip, era_period=config.era_period)
        entry, known = await send_batch(journal, chain, config, batch_id, signed, wait_seconds=wait_seconds,
                                        poll_seconds=poll_seconds, clock=clock, sleep=sleep)
        receipt["batches"].append(entry)
        if not known:
            receipt["stopped"] = f"batch {batch_id} {entry['detail']}"
            break
        if entry["state"] in NONCE_SPENT:
            expected = nonce + 1
        if entry["state"] in NEEDS_PERSON:
            receipt["stopped"] = entry["detail"]
            break
    return _close(receipt, await stake_after(chain))


async def stake_after(chain: PayoutChain) -> int | None:
    """The collector's stake once the pass is done; None if it cannot be read
    (the receipt is written all the same)."""
    try:
        return await chain.collector_stake_rao(await chain.finalized_block())
    except Exception as exc:  # noqa: BLE001 - a receipt is never lost to a last read
        logger.warning("collector stake after the pass could not be read: %r", exc)
        return None


def _count(obligations: Sequence[Obligation]) -> dict:
    return {"obligations": len(obligations), "rao": sum(o.amount_rao for o in obligations),
            "hotkeys": len({o.hotkey for o in obligations})}


def _listed(obligations: Sequence[Obligation]) -> dict:
    return {**_count(obligations), "obligation_ids": [o.obligation_id for o in obligations]}


def close_receipt(receipt: dict, stake_after: int | None = None) -> dict:
    """Totals over every batch the receipt lists, each once, of both kinds:
    ``paid`` (transfers) and ``burned`` (burns, as their events show them);
    the batches that landed not as planned; and the batches it reports (those
    whose fate is known)."""
    listed: dict[str, dict] = {}
    for batch in receipt["unreported"] + receipt["recovered"] + receipt["batches"]:
        listed.setdefault(batch["batch_id"], batch)
    landed = [b for b in listed.values() if b["state"] in PAID]
    paid = [b for b in landed if (b.get("kind") or KIND_TRANSFERS) == KIND_TRANSFERS]
    burns = [b for b in landed if b.get("kind") == KIND_BURN]

    def fees(batches) -> dict:
        total = sum(int(b.get("fee_tao_rao") or 0) for b in batches)
        return {"fee_tao_rao": total, "fee_tao": total / RAO_PER_ALPHA}

    receipt["paid"] = {"batches": len(paid), "transfers": sum(len(b["transfers"]) for b in paid),
                       "rao": sum(int(t["amount_rao"]) for b in paid for t in b["transfers"]), **fees(paid)}
    receipt["paid"]["alpha"] = receipt["paid"]["rao"] / RAO_PER_ALPHA
    receipt["burned"] = {"burns": len(burns), "rao": sum(burned_rao(b) for b in burns), **fees(burns)}
    receipt["burned"]["alpha"] = receipt["burned"]["rao"] / RAO_PER_ALPHA
    receipt["landed_not_as_planned"] = sorted(b["batch_id"] for b in landed if b["state"] == BATCH_MISMATCH)
    receipt["collector_stake_after_rao"] = stake_after
    receipt["reported_batch_ids"] = sorted(batch_id for batch_id, b in listed.items()
                                           if b["state"] not in (BATCH_PENDING, BATCH_UNREADABLE, *UNRESOLVED))
    return receipt


_close = close_receipt


def incomplete_receipt(journal: Journal, *, config: PayoutConfig, pass_day: date, now: float, execute: bool,
                       error: BaseException, schema: str = PAYOUT_RECEIPT_SCHEMA) -> dict:
    """What a pass that raised leaves: every batch the journal knows the fate
    of and no receipt lists yet, marked incomplete."""
    receipt = {"schema": schema, "mode": MODE_EXECUTE if execute else MODE_DRY_RUN,
               "pass_day": pass_day.isoformat(), "as_of": float(now), **config.to_dict(), "incomplete": True,
               "stopped": f"the pass raised {error!r}; the batches the journal settled are listed",
               "unreported": [_row_receipt(b) for b in journal.unreported()], "recovered": [], "batches": [],
               "planned_transfers": []}
    return _close(receipt)


def write_receipt(receipt: Mapping, directory: str | Path, *, prefix: str = "payout") -> Path:
    """The receipt as ``<directory>/<prefix>-<pass day>-<mode>-<time>.json``,
    never overwriting one."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(float(receipt["as_of"])))
    path = folder / f"{prefix}-{receipt['pass_day']}-{receipt['mode']}-{stamp}.json"
    serial = 1
    while path.exists():
        serial += 1
        path = folder / f"{prefix}-{receipt['pass_day']}-{receipt['mode']}-{stamp}-{serial}.json"
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
    return path


__all__ = [
    "BATCH_EXPIRED",
    "BATCH_FAILED",
    "BATCH_FINALIZED",
    "BATCH_FOREIGN",
    "BATCH_MISMATCH",
    "BATCH_PENDING",
    "BATCH_PREPARED",
    "BATCH_SUBMITTED",
    "BATCH_UNKNOWN",
    "BATCH_UNREADABLE",
    "BATCH_VOID",
    "ChainReadError",
    "Inclusion",
    "Journal",
    "KIND_BURN",
    "KIND_TRANSFERS",
    "MAX_ATTEMPTS",
    "MODE_DRY_RUN",
    "MODE_EXECUTE",
    "NEEDS_PERSON",
    "PAYOUT_RECEIPT_SCHEMA",
    "PayoutChain",
    "PayoutConfig",
    "PayoutError",
    "SignedBatch",
    "Signer",
    "Transfer",
    "batches_of",
    "burned_rao",
    "check_collector",
    "check_journal",
    "close_receipt",
    "collector_rails",
    "destinations",
    "incomplete_receipt",
    "plan_digest",
    "plan_transfers",
    "reconcile_burn",
    "reconcile_events",
    "recover",
    "resolve_batch",
    "run_pass",
    "send_batch",
    "stake_after",
    "write_receipt",
]
