"""Paying due obligations: ``transfer_stake`` from the collector's coldkey, each obligation once.

What is paid: the ledger's obligations that are ``due`` at the pass, of tasks
on the collector rail (``payout_rail: collector`` in the registry, chosen when
the task was added). A task on the native rail is paid by the weights: paying
it here too would pay its work twice, so it is never paid here, whatever is
asked. Nothing is signed by default: ``run_pass`` plans, and signs and sends
only with ``execute=True`` and a signer.

How: one ``transfer_stake`` per destination coldkey, the coldkey that owned the
hotkey when the obligation was earned; it moves the collector's alpha stake on
its own hotkey to that coldkey, on the same subnet. At most ``batch_size`` of
them go in one ``Utility.batch_all`` (all or nothing). A destination owed less
than ``min_transfer_rao`` in all waits for a later pass: its obligations stay
due and add up.

Once each: the journal (SQLite, one payer at a time) keeps every obligation
past ``due`` (reserved, submitted, finalized, reconciled) and every batch's
nonce, mortality window and extrinsic hash, written before the batch is sent.
There is one journal, on one machine: a second journal would not know what
the first paid.
A pass first settles every batch whose fate it does not know, from the chain:

- the collector's nonce at the finalized head has not passed the batch's: no
  extrinsic of that nonce is final. Past the batch's window it can never be,
  so its obligations are due again; inside the window the pass waits and
  sends nothing (a send that may still land is never repeated);
- it has passed: the batch's hash is looked up in its window. Found, its
  events say paid (``finalized``, then ``reconciled`` when one
  ``StakeTransferred`` per transfer is there) or failed (due again). Not
  found, another extrinsic used the collector's key with that nonce: the pass
  stops for a person to look.

Funding: only from the collector's stake at the finalized head, less what is
in flight and an operating reserve, oldest obligations first. A destination
that does not fit stops the funding there: nothing younger overtakes it.

Receipts: every pass returns one (a dry run included), written by the caller
for publication: the plan, each batch (nonce, extrinsic hash, block, events,
fee), the collector's stake before and after, and the totals.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
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
# An obligation whose batches failed or expired this many times is held out of
# every plan until a person looks: the chain may refuse its destination (a
# coldkey reserved by a swap announcement, say), and one refused transfer
# reverts a whole batch_all. A transfer that failed before goes alone.
MAX_ATTEMPTS = 3
MODE_DRY_RUN, MODE_EXECUTE = "dry-run", "execute"

# A batch's states. ``prepared``: reserved, not signed yet (never sent); ``submitted``: its hash is kept and it
# may be on the chain; then ``finalized`` (paid), ``failed`` (included, reverted), ``expired`` (never included,
# its window over) or ``foreign`` (its nonce went to another extrinsic: a person must look).
BATCH_PREPARED, BATCH_SUBMITTED = "prepared", "submitted"
BATCH_FINALIZED, BATCH_FAILED, BATCH_EXPIRED, BATCH_FOREIGN = "finalized", "failed", "expired", "foreign"
BATCH_PENDING = "pending"   # a resolution, not a stored state: still in its window, not final
UNRESOLVED = (BATCH_PREPARED, BATCH_SUBMITTED)
# What an obligation's journal row says once it has left ``due``.
PAST_DUE = (RESERVED, SUBMITTED, FINALIZED, RECONCILED)
IN_FLIGHT = (RESERVED, SUBMITTED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payout_obligations(
    obligation_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, work_period INTEGER NOT NULL, part INTEGER NOT NULL,
    hotkey TEXT NOT NULL, coldkey TEXT NOT NULL, amount_rao INTEGER NOT NULL, state TEXT NOT NULL,
    batch_id TEXT, attempts INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS payout_obligations_batch ON payout_obligations(batch_id);
CREATE TABLE IF NOT EXISTS payout_batches(
    batch_id TEXT PRIMARY KEY, pass_day TEXT NOT NULL, state TEXT NOT NULL, transfers TEXT NOT NULL,
    nonce INTEGER, era_block INTEGER, era_period INTEGER, extrinsic_hash TEXT, block_number INTEGER,
    block_hash TEXT, fee_rao INTEGER, events TEXT, detail TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
"""


class PayoutError(RuntimeError):
    """The pass cannot go on safely; nothing more is sent."""


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
    ``{"module", "event", "attributes"}``."""

    block_number: int
    block_hash: str
    success: bool
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
    """What the payer reads of the chain, and the one write: ``submit``."""

    async def finalized_block(self) -> int: ...

    async def collector_stake_rao(self, block: int) -> int: ...

    async def collector_nonce(self, block: int) -> int: ...

    async def hotkey_owner(self, hotkey: str, block: int) -> str | None: ...

    async def block_at(self, at: float) -> int: ...

    async def find_extrinsic(self, extrinsic_hash: str, first_block: int, last_block: int) -> Inclusion | None: ...

    async def submit(self, signed: SignedBatch) -> None: ...


class Signer(Protocol):
    coldkey: str

    async def sign(self, transfers: Sequence[Transfer], *, nonce: int, era_block: int,
                   era_period: int) -> SignedBatch: ...


# -- the journal ---------------------------------------------------------------------------------------------------


class Journal:
    """Every obligation past ``due`` and every batch, in one SQLite file that one
    payer holds at a time (an exclusive lock beside it). ``read_only`` opens it
    without the lock, for a dry run; a file that does not exist reads empty."""

    def __init__(self, path: str | Path, *, read_only: bool = False, clock=time.time) -> None:
        self.path, self.read_only, self.clock = Path(path), read_only, clock
        self._lock = None
        if read_only:
            uri = f"file:{self.path}?mode=ro" if self.path.exists() else "file::memory:"
            self.db = sqlite3.connect(uri, uri=True, isolation_level=None)
            if not self.path.exists():
                self.db.executescript(_SCHEMA)
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
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    @contextlib.contextmanager
    def _transaction(self):
        if self.read_only:
            raise PayoutError("the journal is open read-only")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

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

    def in_flight_rao(self) -> int:
        marks = ",".join("?" * len(IN_FLIGHT))
        return int(self.db.execute(f"SELECT COALESCE(SUM(amount_rao), 0) FROM payout_obligations "
                                   f"WHERE state IN ({marks})", IN_FLIGHT).fetchone()[0])

    def batch(self, batch_id: str) -> dict | None:
        found = self.batches(f"batch_id = ?", (batch_id,))
        return found[0] if found else None

    def unresolved(self) -> list[dict]:
        marks = ",".join("?" * len(UNRESOLVED))
        return self.batches(f"state IN ({marks})", UNRESOLVED)

    def clear_attempts(self, obligation_ids: Iterable[str]) -> int:
        """A person's decision on obligations held out after ``MAX_ATTEMPTS``
        failed attempts (the destination's problem fixed): plan them again.
        Returns how many were due and cleared."""
        ids = list(obligation_ids)
        now = self.clock()
        cleared = 0
        with self._transaction() as db:
            for obligation_id in ids:
                cleared += db.execute("UPDATE payout_obligations SET attempts=0, updated_at=? "
                                      "WHERE obligation_id=? AND state=?", (now, obligation_id, DUE)).rowcount
        return cleared

    def foreign(self) -> list[dict]:
        """Batches whose nonce another extrinsic took: until a person settles
        each (``settle_foreign``), no pass sends anything."""
        return self.batches("state = ?", (BATCH_FOREIGN,))

    def settle_foreign(self, batch_id: str, *, paid: bool, detail: str) -> None:
        """A person's decision on a ``foreign`` batch, after reading the chain:
        ``paid`` (its transfers were made after all) or not (its obligations
        are due again)."""
        batch = self.batch(batch_id)
        if batch is None or batch["state"] != BATCH_FOREIGN:
            raise PayoutError(f"batch {batch_id} is not waiting on a decision")
        now = self.clock()
        with self._transaction() as db:
            if paid:
                db.execute("UPDATE payout_batches SET state=?, detail=?, updated_at=? WHERE batch_id=?",
                           (BATCH_FINALIZED, f"settled by hand: {detail}", now, batch_id))
                db.execute("UPDATE payout_obligations SET state=?, updated_at=? WHERE batch_id=?",
                           (FINALIZED, now, batch_id))
            else:
                db.execute("UPDATE payout_batches SET state=?, detail=?, updated_at=? WHERE batch_id=?",
                           (BATCH_EXPIRED, f"settled by hand: {detail}", now, batch_id))
                db.execute("UPDATE payout_obligations SET state=?, batch_id=NULL, updated_at=? WHERE batch_id=?",
                           (DUE, now, batch_id))

    def batches(self, where: str = "1", args: Sequence = ()) -> list[dict]:
        columns = ("batch_id", "pass_day", "state", "transfers", "nonce", "era_block", "era_period",
                   "extrinsic_hash", "block_number", "block_hash", "fee_rao", "events", "detail", "created_at")
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
        batch, in one transaction. An obligation the journal holds past due
        (reserved, sent or paid) refuses the whole batch."""
        now = self.clock()
        with self._transaction() as db:
            for transfer in transfers:
                for obligation_id in transfer.obligation_ids:
                    o = obligations[obligation_id]
                    changed = db.execute(
                        "INSERT INTO payout_obligations(obligation_id, task_id, work_period, part, hotkey, coldkey, "
                        "amount_rao, state, batch_id, attempts, updated_at) VALUES(?,?,?,?,?,?,?,?,?,1,?) "
                        "ON CONFLICT(obligation_id) DO UPDATE SET state=excluded.state, batch_id=excluded.batch_id, "
                        "attempts=attempts+1, updated_at=excluded.updated_at WHERE payout_obligations.state=?",
                        (obligation_id, o.task_id, o.work_period, o.part, o.hotkey, coldkeys[obligation_id],
                         o.amount_rao, RESERVED, batch_id, now, DUE)).rowcount
                    if changed != 1:
                        raise PayoutError(f"obligation {obligation_id} is already past due in the journal")
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

    def mark_paid(self, batch_id: str, inclusion: Inclusion, *, reconciled: bool, detail: str) -> None:
        now = self.clock()
        with self._transaction() as db:
            db.execute("UPDATE payout_batches SET state=?, block_number=?, block_hash=?, fee_rao=?, events=?, "
                       "detail=?, updated_at=? WHERE batch_id=?",
                       (BATCH_FINALIZED, inclusion.block_number, inclusion.block_hash, inclusion.fee_rao,
                        json.dumps(list(inclusion.events), sort_keys=True, default=str), detail, now, batch_id))
            db.execute("UPDATE payout_obligations SET state=?, updated_at=? WHERE batch_id=?",
                       (RECONCILED if reconciled else FINALIZED, now, batch_id))

    def mark_released(self, batch_id: str, state: str, detail: str, inclusion: Inclusion | None = None) -> None:
        """The batch paid nothing (failed, expired or never sent): its
        obligations are due again, their destinations kept."""
        now = self.clock()
        with self._transaction() as db:
            db.execute("UPDATE payout_batches SET state=?, block_number=?, block_hash=?, fee_rao=?, events=?, "
                       "detail=?, updated_at=? WHERE batch_id=?",
                       (state, None if inclusion is None else inclusion.block_number,
                        None if inclusion is None else inclusion.block_hash,
                        None if inclusion is None else inclusion.fee_rao,
                        None if inclusion is None else json.dumps(list(inclusion.events), sort_keys=True,
                                                                  default=str),
                        detail, now, batch_id))
            db.execute("UPDATE payout_obligations SET state=?, batch_id=NULL, updated_at=? WHERE batch_id=?",
                       (DUE, now, batch_id))

    def mark_foreign(self, batch_id: str, detail: str) -> None:
        """The batch's nonce was used by another extrinsic: its obligations stay
        reserved (never paid twice) until a person decides."""
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
    while ``available_rao`` lasts; the first that does not fit stops it, and it
    and every one after are unfunded."""
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
        elif stopped or total > left:
            stopped = True
            unfunded.extend(group)
        else:
            transfers.append(Transfer(coldkey=coldkey, amount_rao=total,
                                      obligation_ids=tuple(o.obligation_id for o in group)))
            left -= total
    return transfers, deferred, unfunded


def batches_of(transfers: Sequence[Transfer], size: int, *, alone: Collection[str] = ()) -> list[list[Transfer]]:
    """``transfers`` in batches of at most ``size``; a transfer to a coldkey in
    ``alone`` (one that failed before) in a batch of its own, so it cannot
    revert the others."""
    solo = [[t] for t in transfers if t.coldkey in alone]
    rest = [t for t in transfers if t.coldkey not in alone]
    return [list(rest[start:start + size]) for start in range(0, len(rest), size)] + solo


def reconcile_events(transfers: Sequence[Mapping], inclusion: Inclusion, *, collector_coldkey: str,
                     collector_hotkey: str) -> tuple[bool, str]:
    """Whether a paid batch's events hold one ``StakeTransferred`` per transfer,
    from the collector, on its hotkey, to each destination. The event's fields
    are (origin coldkey, destination coldkey, hotkey, origin netuid,
    destination netuid, amount); its amount is a TAO value, not the alpha sent,
    so it is kept in the receipt and not compared."""
    want = Counter(str(t["coldkey"]) for t in transfers)
    got: Counter = Counter()
    for event in inclusion.events:
        if event.get("module") != "SubtensorModule" or event.get("event") != "StakeTransferred":
            continue
        attributes = event.get("attributes")
        if isinstance(attributes, Mapping):
            fields = [attributes.get(k) for k in ("origin_coldkey", "destination_coldkey", "hotkey")]
        elif isinstance(attributes, (list, tuple)) and len(attributes) >= 3:
            fields = list(attributes[:3])
        else:
            return False, f"a StakeTransferred event could not be read: {attributes!r}"
        origin, destination, hotkey = (None if f is None else str(f) for f in fields)
        if origin != collector_coldkey or hotkey != collector_hotkey:
            return False, f"a StakeTransferred event is not the collector's: {fields!r}"
        got[destination] += 1
    if got != want:
        return False, f"StakeTransferred destinations {dict(got)} differ from the batch's {dict(want)}"
    return True, f"{sum(got.values())} StakeTransferred event(s), one per transfer"


# -- one pass ---------------------------------------------------------------------------------------------------------


async def resolve_batch(batch: Mapping, chain: PayoutChain) -> tuple[str, Inclusion | None, str]:
    """A batch whose fate is unknown, from the chain: ``finalized``, ``failed``,
    ``expired``, ``foreign``, or ``pending`` (still inside its window)."""
    if batch.get("extrinsic_hash") is None:
        # The hash is kept before anything is sent: without one, nothing was.
        return BATCH_EXPIRED, None, "never sent"
    head = await chain.finalized_block()
    nonce = int(batch["nonce"])
    last = int(batch["era_block"]) + int(batch["era_period"])
    if await chain.collector_nonce(head) <= nonce:
        if head > last:
            return BATCH_EXPIRED, None, f"nonce {nonce} unused at finalized block {head}, past its window ({last})"
        return BATCH_PENDING, None, f"nonce {nonce} unused at finalized block {head}, window open until {last}"
    found = await chain.find_extrinsic(str(batch["extrinsic_hash"]), int(batch["era_block"]), min(head, last))
    if found is None:
        return BATCH_FOREIGN, None, (f"nonce {nonce} was used at or before finalized block {head} by an extrinsic "
                                     f"other than {batch['extrinsic_hash']}")
    if found.success:
        return BATCH_FINALIZED, found, f"included in block {found.block_number}"
    return BATCH_FAILED, found, f"failed in block {found.block_number}: {found.error}"


def _settle(journal: Journal, batch: Mapping, state: str, inclusion: Inclusion | None, detail: str,
            config: PayoutConfig) -> dict:
    """Write a resolution down; what the receipt says of it."""
    if state == BATCH_FINALIZED:
        reconciled, why = reconcile_events(batch["transfers"], inclusion, collector_coldkey=config.collector_coldkey,
                                           collector_hotkey=config.collector_hotkey)
        journal.mark_paid(batch["batch_id"], inclusion, reconciled=reconciled, detail=f"{detail}; {why}")
        detail = f"{detail}; {why}"
    elif state in (BATCH_FAILED, BATCH_EXPIRED):
        journal.mark_released(batch["batch_id"], state, detail, inclusion)
    elif state == BATCH_FOREIGN:
        journal.mark_foreign(batch["batch_id"], detail)
    return _batch_receipt(batch, state, inclusion, detail)


def _batch_receipt(batch: Mapping, state: str, inclusion: Inclusion | None, detail: str) -> dict:
    return {"batch_id": batch["batch_id"], "state": state, "nonce": batch.get("nonce"),
            "extrinsic_hash": batch.get("extrinsic_hash"), "era_block": batch.get("era_block"),
            "era_period": batch.get("era_period"),
            "block_number": None if inclusion is None else inclusion.block_number,
            "block_hash": None if inclusion is None else inclusion.block_hash,
            "fee_rao": None if inclusion is None else inclusion.fee_rao,
            "events": None if inclusion is None else list(inclusion.events),
            "transfers": list(batch["transfers"]), "detail": detail}


async def destinations(obligations: Sequence[Obligation], chain: PayoutChain,
                       journal: Journal) -> tuple[dict[str, str], list[Obligation]]:
    """Each obligation's destination coldkey: the one the journal noted when it
    was first reserved, else the hotkey's owner at the block it was earned in.
    The obligations whose hotkey had no owner then are returned apart."""
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
        if obligation.earned_at not in blocks:
            blocks[obligation.earned_at] = await chain.block_at(obligation.earned_at)
        key = (obligation.hotkey, blocks[obligation.earned_at])
        if key not in owners:
            owners[key] = await chain.hotkey_owner(*key)
        if owners[key]:
            found[obligation.obligation_id] = str(owners[key])
        else:
            ownerless.append(obligation)
    return found, ownerless


async def run_pass(obligations: Iterable[Obligation], *, journal: Journal, chain: PayoutChain, config: PayoutConfig,
                   collector_tasks: Collection[str], pass_day: date, terms: PayTerms, now: float,
                   execute: bool = False, signer: Signer | None = None, confirm_rao: int | None = None,
                   wait_seconds: float = 600.0, poll_seconds: float = 12.0, clock=time.time,
                   sleep=asyncio.sleep) -> dict:
    """One pass: settle what is unknown, then plan, then (``execute``) pay.
    Returns the receipt. A dry run writes nothing to the journal and signs
    nothing. Executing pays only when ``confirm_rao`` names the pass's total
    exactly (the owner's go, read off a dry run); otherwise it stops before
    reserving anything, and says the total."""
    if execute:
        if signer is None:
            raise PayoutError("executing a pass needs a signer")
        if signer.coldkey != config.collector_coldkey:
            raise PayoutError("the signer's coldkey is not the collector's")
        if journal.read_only:
            raise PayoutError("executing a pass needs the journal open for writing")
    collector_tasks = {str(task) for task in collector_tasks}
    at = pay_instant(pass_day, terms)
    receipt: dict = {"schema": PAYOUT_RECEIPT_SCHEMA, "mode": MODE_EXECUTE if execute else MODE_DRY_RUN,
                     "pass_day": pass_day.isoformat(), "pass_at": at, "as_of": float(now), **config.to_dict(),
                     "recovered": [], "batches": [], "stopped": None}

    # 1. What is not known yet, from the chain, before anything new is sent. A dry run settles nothing, but
    # plans as if it had (``would``), so the total it shows is the one an execution would ask to confirm.
    pending = 0
    would: dict[str, str] = {}
    would_rao = 0
    for batch in journal.unresolved():
        state, inclusion, detail = await resolve_batch(batch, chain)
        if state == BATCH_PENDING:
            pending += 1
            receipt["recovered"].append(_batch_receipt(batch, state, None, detail))
        elif execute:
            receipt["recovered"].append(_settle(journal, batch, state, inclusion, detail, config))
        else:
            receipt["recovered"].append(_batch_receipt(batch, state, inclusion, f"(would settle) {detail}"))
            if state in (BATCH_FINALIZED, BATCH_FAILED, BATCH_EXPIRED):
                would[batch["batch_id"]] = state
                would_rao += sum(int(t["amount_rao"]) for t in batch["transfers"])
        if state == BATCH_FOREIGN:
            receipt["stopped"] = detail
    waiting = [b["batch_id"] for b in journal.foreign()]
    if waiting and receipt["stopped"] is None:
        receipt["stopped"] = (f"batch(es) {', '.join(waiting)} lost their nonce to another extrinsic: nothing is "
                              "sent until each is settled by hand (Journal.settle_foreign)")
    # 2. What is due at this pass and the collector pays.
    due = [o for o in obligations if o.due_at <= at and state_at(o, now) == DUE]
    native = [o for o in due if o.task_id not in collector_tasks]
    mine = [o for o in due if o.task_id in collector_tasks]
    held = journal.rows(o.obligation_id for o in mine)

    def state_of(obligation_id: str) -> str:
        row = held.get(obligation_id)
        if row is None:
            return DUE
        settled = would.get(row["batch_id"])
        if settled == BATCH_FINALIZED:
            return FINALIZED
        return DUE if settled in (BATCH_FAILED, BATCH_EXPIRED) else row["state"]

    open_ = [o for o in mine if state_of(o.obligation_id) == DUE]
    attempts = {o.obligation_id: int(held.get(o.obligation_id, {}).get("attempts") or 0) for o in open_}
    stuck = [o for o in open_ if attempts[o.obligation_id] >= MAX_ATTEMPTS]
    open_ = [o for o in open_ if attempts[o.obligation_id] < MAX_ATTEMPTS]
    found, ownerless = await destinations(open_, chain, journal)
    head = await chain.finalized_block()
    stake = await chain.collector_stake_rao(head)
    in_flight = journal.in_flight_rao() - would_rao
    available = max(0, stake - in_flight - config.reserve_rao)
    by_id = {o.obligation_id: o for o in open_}
    transfers, deferred, unfunded = plan_transfers(
        [(o, found[o.obligation_id]) for o in open_ if o.obligation_id in found], available_rao=available,
        min_transfer_rao=config.min_transfer_rao)
    receipt.update({
        "finalized_block": head, "collector_stake_rao": stake, "in_flight_rao": in_flight,
        "available_rao": available,
        "due": _count(due), "native_rail": _count(native), "already_past_due": _count(
            [o for o in mine if o.obligation_id not in by_id]),
        "deferred_below_minimum": _count(deferred), "unfunded": _count(unfunded), "ownerless": _count(ownerless),
        "stuck": {**_count(stuck), "obligation_ids": [o.obligation_id for o in stuck]},
        "planned": {"transfers": len(transfers), "rao": sum(t.amount_rao for t in transfers),
                    "obligations": sum(len(t.obligation_ids) for t in transfers)},
        "transfers": [t.to_dict() for t in transfers], "confirmed_rao": confirm_rao})
    if receipt["stopped"] is not None:
        return _close(receipt)
    if pending:
        receipt["stopped"] = (f"{pending} batch(es) sent earlier may still land: nothing new is sent until they "
                              "are final or expired")
        return _close(receipt)
    if not execute:
        return _close(receipt)
    planned = receipt["planned"]["rao"]
    if planned and confirm_rao != planned:
        receipt["stopped"] = (f"this pass would pay {planned} rao ({planned / RAO_PER_ALPHA:.9f} alpha) in "
                              f"{len(transfers)} transfer(s): confirm that exact total with --confirm-rao {planned}")
        return _close(receipt)

    # 3. Pay: one batch at a time, each final before the next takes the following nonce.
    coldkeys = {oid: t.coldkey for t in transfers for oid in t.obligation_ids}
    retried = {oid for oid, count in attempts.items() if count > 0}
    alone = {t.coldkey for t in transfers if retried.intersection(t.obligation_ids)}
    for group in batches_of(transfers, config.batch_size, alone=alone):
        batch_id = f"{pass_day.isoformat()}-{uuid.uuid4().hex[:12]}"
        journal.reserve(batch_id, pass_day, group, by_id, coldkeys)
        tip = await chain.finalized_block()
        nonce = await chain.collector_nonce(tip)
        signed = await signer.sign(group, nonce=nonce, era_block=tip, era_period=config.era_period)
        journal.mark_submitted(batch_id, signed)
        try:
            await chain.submit(signed)
        except Exception as exc:  # noqa: BLE001 - the send's fate is read from the chain, never assumed
            logger.warning("payout batch %s: send raised %r; reading the chain for its fate", batch_id, exc)
        deadline = clock() + wait_seconds
        while True:
            batch = journal.batch(batch_id)
            state, inclusion, detail = await resolve_batch(batch, chain)
            if state != BATCH_PENDING:
                receipt["batches"].append(_settle(journal, batch, state, inclusion, detail, config))
                break
            if clock() >= deadline:
                receipt["batches"].append(_batch_receipt(batch, state, None, detail))
                receipt["stopped"] = f"batch {batch_id} not final after {wait_seconds:.0f} s: the next pass settles it"
                return _close(receipt, await chain.collector_stake_rao(await chain.finalized_block()))
            await sleep(poll_seconds)
        if state == BATCH_FOREIGN:
            receipt["stopped"] = detail
            break
    return _close(receipt, await chain.collector_stake_rao(await chain.finalized_block()))


def _count(obligations: Sequence[Obligation]) -> dict:
    return {"obligations": len(obligations), "rao": sum(o.amount_rao for o in obligations),
            "hotkeys": len({o.hotkey for o in obligations})}


def _close(receipt: dict, stake_after: int | None = None) -> dict:
    paid = [b for b in receipt["batches"] + receipt["recovered"] if b["state"] == BATCH_FINALIZED]
    receipt["paid"] = {"batches": len(paid),
                       "transfers": sum(len(b["transfers"]) for b in paid),
                       "rao": sum(int(t["amount_rao"]) for b in paid for t in b["transfers"]),
                       "fee_rao": sum(int(b["fee_rao"] or 0) for b in paid)}
    receipt["paid"]["alpha"] = receipt["paid"]["rao"] / RAO_PER_ALPHA
    receipt["collector_stake_after_rao"] = stake_after
    return receipt


def write_receipt(receipt: Mapping, directory: str | Path) -> Path:
    """The receipt as ``<directory>/payout-<pass day>-<mode>-<time>.json``,
    never overwriting one."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(float(receipt["as_of"])))
    path = folder / f"payout-{receipt['pass_day']}-{receipt['mode']}-{stamp}.json"
    serial = 1
    while path.exists():
        serial += 1
        path = folder / f"payout-{receipt['pass_day']}-{receipt['mode']}-{stamp}-{serial}.json"
    path.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
    return path


__all__ = [
    "BATCH_EXPIRED",
    "BATCH_FAILED",
    "BATCH_FINALIZED",
    "BATCH_FOREIGN",
    "BATCH_PENDING",
    "BATCH_PREPARED",
    "BATCH_SUBMITTED",
    "Inclusion",
    "Journal",
    "MAX_ATTEMPTS",
    "MODE_DRY_RUN",
    "MODE_EXECUTE",
    "PAYOUT_RECEIPT_SCHEMA",
    "PayoutChain",
    "PayoutConfig",
    "PayoutError",
    "SignedBatch",
    "Signer",
    "Transfer",
    "batches_of",
    "destinations",
    "plan_transfers",
    "reconcile_events",
    "resolve_batch",
    "run_pass",
    "write_receipt",
]
