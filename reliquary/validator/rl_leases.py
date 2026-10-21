"""The RL lease book (rl-async-design.md §4.2; implementation map §1.2): a miner reserves prompts before it
generates them, and every lease ends in a submission or an expiry (there is no release).

* **Exclusive.** At most one live lease per ``(environment, prompt, checkpoint k)``; once a lease of it is
  submitted, the key is closed for that k (the draw would be identical anyway). A SQLite partial unique index
  enforces it. An expired lease frees its key, and the next lessee gets the same draw (resolution C12: accepted,
  expiry earns nothing and cuts the quota).
* **Fair inside an interval.** ``LeaseCollector`` gathers the requests that arrive within ``interval_s`` (~1 s)
  and ``LeaseBook.grant`` resolves the whole batch at once: every (request, prompt) claim is shuffled with the
  validator's own randomness (``random.SystemRandom``) and taken in that order, so a contested prompt goes to a
  uniformly random requester that still has quota; arriving first inside the interval buys nothing.
* **Refusals say only** ``taken`` / ``cooling`` / ``blacklisted`` / ``quota`` / ``stale`` / ``banned`` (the
  caller's own audit ban, ``ban``, set by the submission path): never who holds a
  prompt, when its lease ends, or which leases are open. ``taken`` covers both an open lease of someone else and a key
  closed by a submission. A resend by the holder answers its own open lease again (a lost response is not a lost lease).
* **Prompt lifecycle** (§4.5): ``record_outcome`` (called by the submission path, P3, once a group is graded)
  appends the scan to a public history and moves the prompt: too easy a second time is blacklisted for the run (a
  first too-easy scan is not yet proof: it cools down like the others); in zone, too hard and truncated cool down
  until checkpoint ``latest + cooldown`` (``CooldownMap``'s half-open rule, in
  checkpoints: leasable again once the latest k reaches it; the per-env cooldown is read at each outcome). A
  never-scanned prompt has no row and is leasable.
  ``classify_group`` gives the verdict from the rewards and truncation flags. ``prompt_state`` is the miners'
  incremental view (no hotkey, no token): states and scans after a cursor.
* **Stale:** a lease on a k that is not adopted, or whose S_k is revealed, is refused (§4.6: leases only on
  checkpoints whose secret is unrevealed). S_k is revealed once k < ``latest - max_lag``, ``SECRET_REVEAL_LAG``
  (16) by default: late enough never to bind an honest miner; there is no other lag cap.
* **Expiry** is fixed per environment (``lease_expiry_s``: the env's max tokens at a slow reference speed, times a
  margin), never derived from the miner. ``expire_due`` sweeps (each ``grant`` and ``submit`` sweeps too) and
  cuts the hotkey's quota (``EXPIRY_BACKOFF``). A submission is judged at its arrival: a lease held by ``hold_submission`` (arrived
  on time, waiting for or in grading) is never expired by a sweep.
* **One lease of a prompt at a time** across checkpoints (``taken``): the same prompt cannot be leased at every
  unrevealed k at once. A request is cut to the hotkey's room (``quota - in_use``, capped by its share of the
  global cap) before any lookup.
* **Nonce:** ``nonce = H(NONCE_DOMAIN, S_k, env, prompt, k)`` with ``S_k`` a 32-byte secret drawn when k is
  adopted. ``H(COMMIT_DOMAIN, S_k, k)`` is published from adoption (``commitments()``) and ``S_k`` itself once k
  retires (older than ``latest - max_lag``: no lease on it can be granted any more), so anyone can check that
  every nonce of k was the deterministic one. Fields are length-prefixed under their own domain.

Persistence follows the service runtime (``services/runtime.py``): one SQLite file (WAL, ``synchronous=FULL``,
``BEGIN IMMEDIATE`` under an ``RLock``), created ``0600`` because it holds the unrevealed secrets. Open leases,
secrets, the latest k and the quota state (``QuotaController.attach``) survive a restart. A lease that fell due
while the validator was down is extended to ``now + RESTART_GRACE_S`` at boot: its holder could not submit.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import random
import json
import math
import secrets as _secrets
import sqlite3
import statistics
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reliquary.constants import MAX_TRUNCATED_PER_SUBMISSION, SIGMA_MIN_V4, prompt_cooldown_windows_for_environment
from reliquary.protocol.rl_prompt_ranges import PromptRows
from reliquary.validator.rl_lease_quota import QuotaController

TAKEN, COOLING, BLACKLISTED, QUOTA, STALE, BANNED = "taken", "cooling", "blacklisted", "quota", "stale", "banned"
FULL = "full"     # the global cap: buffered groups reached ``max_pending_groups``
REFUSALS = frozenset({TAKEN, COOLING, BLACKLISTED, QUOTA, STALE, BANNED, FULL})
OPEN, SUBMITTED, EXPIRED = "open", "submitted", "expired"
IN_ZONE, TOO_EASY, TOO_HARD, TRUNCATED = "in_zone", "too_easy", "too_hard", "truncated"
VERDICTS = frozenset({IN_ZONE, TOO_EASY, TOO_HARD, TRUNCATED})
COOLDOWN = "cooldown"                    # a prompt's stored state is COOLDOWN (until_k) or BLACKLISTED

NONCE_DOMAIN = b"reliquary/rl-lease-nonce/v1"
COMMIT_DOMAIN = b"reliquary/rl-lease-secret-commit/v1"
SECRET_BYTES = 32
MAX_REVEALED_LISTED = 64
MAX_PROMPTS_PER_REQUEST = 1024
INTERVAL_S = 1.0          # the collection interval inside which arrival order buys nothing
RESTART_GRACE_S = 120.0   # what a lease that fell due while the validator was down gets at boot
SECRET_REVEAL_LAG = 16    # S_k is revealed, and k stops being leasable, once k < latest - this
MAX_SCANS_PER_PAGE = 1000
HOTKEY_MAX_OPEN = 64          # PLACEHOLDER: the most open leases one hotkey may hold under a global cap


def _bound(domain: bytes, parts: list[bytes]) -> bytes:
    h = hashlib.sha256(domain)
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def _u64(value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise ValueError("a non-negative integer is required")
    return value.to_bytes(8, "big")


def lease_nonce(secret: bytes, environment: str, prompt_idx: int, checkpoint: int) -> bytes:
    """The forced draw's nonce for (env, prompt, k): the same for every lease of that key."""
    return _bound(NONCE_DOMAIN, [bytes(secret), environment.encode("utf-8"), _u64(prompt_idx), _u64(checkpoint)])


def secret_commitment(secret: bytes, checkpoint: int) -> bytes:
    return _bound(COMMIT_DOMAIN, [bytes(secret), _u64(checkpoint)])


def verify_reveal(commitment: bytes, checkpoint: int, secret: bytes) -> bool:
    try:
        return _secrets.compare_digest(secret_commitment(secret, checkpoint), bytes(commitment))
    except (TypeError, ValueError):
        return False


def lease_expiry_s(*, max_tokens: int, reference_tokens_per_s: float, margin: float) -> float:
    """A lease's fixed lifetime for an env: its longest generation at a slow reference speed, times a margin."""
    if not (max_tokens > 0 and reference_tokens_per_s > 0 and margin >= 1.0):
        raise ValueError("max_tokens and reference speed must be positive, margin >= 1")
    return float(max_tokens) / float(reference_tokens_per_s) * float(margin)


def classify_group(rewards, truncated, *, max_truncated: int = MAX_TRUNCATED_PER_SUBMISSION,
                   sigma_min: float = SIGMA_MIN_V4) -> str:
    """A graded group's verdict: ``truncated`` past ``max_truncated``, else ``in_zone`` at population sigma >=
    ``sigma_min`` (1/16 and 15/16: 0.242), else ``too_easy`` at mean >= 0.5 (16/16), else ``too_hard`` (0/16).
    Truncation comes first: blacklisting is final, so it needs a clean group."""
    rewards = [float(r) for r in rewards]
    if not rewards or len(rewards) != len(truncated) or not all(math.isfinite(r) for r in rewards):
        raise ValueError("one finite reward and one truncation flag per rollout")
    if sum(bool(t) for t in truncated) > max_truncated:
        return TRUNCATED
    if statistics.pstdev(rewards) >= sigma_min:
        return IN_ZONE
    return TOO_EASY if statistics.fmean(rewards) >= 0.5 else TOO_HARD


@dataclass(frozen=True)
class LeaseRequest:
    hotkey: str
    environment: str
    prompt_ids: tuple[int, ...]
    checkpoint: int


@dataclass(frozen=True)
class Lease:
    lease_id: str
    hotkey: str
    environment: str
    prompt_idx: int
    checkpoint: int
    nonce: str
    issued_at: float
    expires_at: float
    state: str

    def to_public(self) -> dict:
        """What the holder is told (the holder's own lease: no other lease is ever listed)."""
        return {"lease_id": self.lease_id, "environment": self.environment, "prompt_idx": self.prompt_idx,
                "checkpoint": self.checkpoint, "nonce": self.nonce, "issued_at": self.issued_at,
                "expires_at": self.expires_at}


@dataclass(frozen=True)
class Scan:
    """One graded group of a prompt, as published (no hotkey, no token). ``first_scan``: the prompt had never been
    scanned in the run (an out-of-zone first scan is paid 25 % after audit, P5)."""
    seq: int
    environment: str
    prompt_idx: int
    checkpoint: int
    rewards: tuple[float, ...]
    verdict: str
    first_scan: bool
    at: float

    def to_public(self) -> dict:
        return {"seq": self.seq, "environment": self.environment, "prompt_idx": self.prompt_idx,
                "checkpoint": self.checkpoint, "rewards": list(self.rewards), "verdict": self.verdict,
                "first_scan": self.first_scan, "at": self.at}


@dataclass
class LeaseAnswer:
    leases: list[Lease] = field(default_factory=list)
    refused: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"leases": [lease.to_public() for lease in self.leases],
                "refused": [{"prompt_idx": p, "reason": r} for p, r in sorted(self.refused.items())]}


@dataclass(frozen=True)
class LeaseBookConfig:
    expiry_s: Mapping[str, float]                              # per served env: ``lease_expiry_s``
    max_lag: int = SECRET_REVEAL_LAG
    # per env: ``(start, end)`` (``rl_prompt_ranges.rl_prompt_range``) or ``PromptRows`` (an episode env's rows)
    prompt_range: Mapping[str, Any] | None = None
    # Decision E: a static per-env cooldown set by an operator, read at each outcome so a live change applies to the
    # next cooldown. The profile's ``prompt_cooldown_windows`` counts checkpoints in the async protocol; the service
    # can pass its live schedule instead.
    cooldown_of: Callable[[str], int] = prompt_cooldown_windows_for_environment
    # The global cap (async trainer throttle, AReaL-style): no lease is granted while the buffered groups (queued or
    # taken) reach it, so a trainer slower than the miners holds them back instead of dropping their work. Open
    # leases are not counted (the quota and expiry bound them; a growing buffer is the oversupply signal), but one
    # hotkey may hold at most ``HOTKEY_MAX_OPEN`` of them (refused as ``quota``). None: no cap.
    max_pending_groups: int | None = None

    def __post_init__(self) -> None:
        if not self.expiry_s or any(not (float(v) > 0) for v in self.expiry_s.values()):
            raise ValueError("every served environment needs a positive lease expiry")
        if self.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if self.max_pending_groups is not None and self.max_pending_groups < 1:
            raise ValueError("max_pending_groups must be positive")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_lease_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_lease_secrets(k INTEGER PRIMARY KEY, secret BLOB NOT NULL, commitment TEXT NOT NULL,
    adopted_at REAL NOT NULL, revealed_at REAL);
CREATE TABLE IF NOT EXISTS rl_leases(lease_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, environment TEXT NOT NULL,
    prompt_idx INTEGER NOT NULL, k INTEGER NOT NULL, nonce TEXT NOT NULL, issued_at REAL NOT NULL,
    expires_at REAL NOT NULL, state TEXT NOT NULL, closed_at REAL);
CREATE UNIQUE INDEX IF NOT EXISTS rl_leases_live ON rl_leases(environment, prompt_idx, k)
    WHERE state IN ('open', 'submitted');
CREATE INDEX IF NOT EXISTS rl_leases_due ON rl_leases(state, expires_at);
CREATE INDEX IF NOT EXISTS rl_leases_holder ON rl_leases(hotkey, state);
CREATE TABLE IF NOT EXISTS rl_prompt_scans(seq INTEGER PRIMARY KEY AUTOINCREMENT, environment TEXT NOT NULL,
    prompt_idx INTEGER NOT NULL, k INTEGER NOT NULL, rewards TEXT NOT NULL, verdict TEXT NOT NULL,
    first_scan INTEGER NOT NULL, at REAL NOT NULL, state TEXT, until_k INTEGER);
CREATE TABLE IF NOT EXISTS rl_prompt_state(environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
    state TEXT NOT NULL, until_k INTEGER, seq INTEGER NOT NULL, PRIMARY KEY(environment, prompt_idx));
CREATE INDEX IF NOT EXISTS rl_prompt_state_seq ON rl_prompt_state(seq);
CREATE TABLE IF NOT EXISTS rl_hotkey_bans(hotkey TEXT PRIMARY KEY, until REAL NOT NULL);
"""
_COLUMNS = "lease_id, hotkey, environment, prompt_idx, k, nonce, issued_at, expires_at, state"


def _private_file(path: Path) -> None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)


class LeaseBook:
    def __init__(self, path: str | os.PathLike[str], config: LeaseBookConfig, *, quota: QuotaController,
                 clock: Callable[[], float] = time.time,
                 rng: random.Random | None = None) -> None:
        self.config = config
        self.quota = quota
        self.clock = clock
        self._rng = rng or random.SystemRandom()
        self.lock = threading.RLock()
        self._depth = 0
        self._full: set[str] = set()              # envs refused for the global cap since the last drain
        self._submitting: dict[str, int] = {}     # leases whose on-time submission is being graded: never expired
        path = Path(path)
        _private_file(path)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(_SCHEMA)
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(rl_prompt_scans)")}
            for column, kind in (("state", "TEXT"), ("until_k", "INTEGER")):   # a book from before the scan log
                if column not in columns:
                    self.db.execute(f"ALTER TABLE rl_prompt_scans ADD COLUMN {column} {kind}")
            self.db.commit()
            quota.attach(self.db, self.lock)
            self._boot()
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        with self.lock:
            self.db.close()

    @contextlib.contextmanager
    def _txn(self):
        """One transaction under the lock. Called inside ``transaction()`` on the same thread, it joins the outer
        transaction (which commits or rolls back everything)."""
        with self.lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            if self.db.in_transaction:
                raise RuntimeError("lease book transaction is already open")
            self.db.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()
            finally:
                self._depth = 0

    def transaction(self):
        """A transaction the book's own writes join: the submission path closes a lease, records its outcome and
        stores the group atomically. Other tables of the same file (``rl_submit``) are written inside it."""
        return self._txn()

    def _boot(self) -> None:
        now = float(self.clock())
        floor = now + RESTART_GRACE_S
        with self._txn():
            self.db.execute("UPDATE rl_leases SET expires_at=? WHERE state=? AND expires_at<?", (floor, OPEN, floor))
            counts = dict(self.db.execute("SELECT hotkey, COUNT(*) FROM rl_leases WHERE state=? GROUP BY hotkey",
                                          (OPEN,)).fetchall())
            self.quota.reconcile(counts)

    # --- checkpoints and secrets ----------------------------------------------------------------------------
    @property
    def latest_checkpoint(self) -> int | None:
        with self.lock:
            row = self.db.execute("SELECT value FROM rl_lease_meta WHERE key='latest'").fetchone()
        return None if row is None else int(row[0])

    def adopt_checkpoint(self, k: int) -> bytes:
        """Adopt checkpoint k as the latest: draw S_k, publish its commitment, reveal the secrets of retired k.
        Returns the commitment. Re-adopting the latest k is a no-op; an older k is refused."""
        _u64(k)
        now = float(self.clock())
        with self._txn():
            latest = self.latest_checkpoint
            if latest is not None and k < latest:
                raise ValueError(f"checkpoint {k} is older than the adopted {latest}")
            if latest != k:
                secret = os.urandom(SECRET_BYTES)
                self.db.execute("INSERT INTO rl_lease_secrets(k, secret, commitment, adopted_at) VALUES(?,?,?,?)",
                                (k, secret, secret_commitment(secret, k).hex(), now))
                self.db.execute("INSERT OR REPLACE INTO rl_lease_meta(key, value) VALUES('latest', ?)", (str(k),))
                retired = k - self.config.max_lag
                self.db.execute("UPDATE rl_lease_secrets SET revealed_at=? WHERE k<? AND revealed_at IS NULL",
                                (now, retired))
                # closed keys of retired k can never be asked again: keep the open ones (they still end)
                self.db.execute("DELETE FROM rl_leases WHERE k<? AND state<>?", (retired - self.config.max_lag, OPEN))
            row = self.db.execute("SELECT commitment FROM rl_lease_secrets WHERE k=?", (k,)).fetchone()
        return bytes.fromhex(row[0])

    def commitments(self) -> dict:
        """The commitments of the leasable checkpoints and the secrets of the latest retired ones."""
        with self.lock:
            latest = self.latest_checkpoint
            rows = self.db.execute("SELECT k, commitment, secret, revealed_at FROM rl_lease_secrets "
                                   "ORDER BY k").fetchall()
        active = [{"k": k, "commitment": c} for k, c, _, revealed in rows if revealed is None]
        revealed = [{"k": k, "commitment": c, "secret": bytes(s).hex()}
                    for k, c, s, r in rows if r is not None][-MAX_REVEALED_LISTED:]
        return {"latest": latest, "max_lag": self.config.max_lag, "active": active, "revealed": revealed,
                "nonce_domain": NONCE_DOMAIN.decode(), "commit_domain": COMMIT_DOMAIN.decode()}

    # --- requests -------------------------------------------------------------------------------------------
    def check_request(self, request: LeaseRequest) -> None:
        """ValueError when the request itself is malformed (the route answers it before any interval)."""
        if not isinstance(request.hotkey, str) or not request.hotkey:
            raise ValueError("hotkey required")
        if request.environment not in self.config.expiry_s:
            raise ValueError("environment not served")
        if type(request.checkpoint) is not int or request.checkpoint < 0:
            raise ValueError("checkpoint must be a non-negative integer")
        ids = request.prompt_ids
        if not ids or len(ids) > MAX_PROMPTS_PER_REQUEST:
            raise ValueError(f"between 1 and {MAX_PROMPTS_PER_REQUEST} prompts")
        if any(type(p) is not int or p < 0 for p in ids) or len(set(ids)) != len(ids):
            raise ValueError("prompt ids must be distinct non-negative integers")
        bounds = (self.config.prompt_range or {}).get(request.environment)
        if bounds is not None and any(p not in PromptRows.of(bounds) for p in ids):
            raise ValueError("prompt outside the environment's RL range")

    @staticmethod
    def _lease(row) -> Lease:
        lease_id, hotkey, env, prompt, k, nonce, issued, expires, state = row
        return Lease(lease_id, hotkey, env, int(prompt), int(k), nonce, float(issued), float(expires), state)

    def lease(self, lease_id: str) -> Lease | None:
        with self.lock:
            row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE lease_id=?", (lease_id,)).fetchone()
        return None if row is None else self._lease(row)

    def open_leases(self, hotkey: str, now: float | None = None) -> list[Lease]:
        """The hotkey's open, unexpired leases, earliest expiry first (``GET /rl/lease/mine``: a miner that lost
        its journal recovers them). Nobody else's lease is ever listed."""
        now = float(self.clock()) if now is None else now
        with self.lock:
            rows = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE hotkey=? AND state=? AND expires_at>? "
                                   "ORDER BY expires_at", (hotkey, OPEN, now)).fetchall()
        return [self._lease(row) for row in rows]

    def in_use(self, hotkey: str) -> int:
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM rl_leases WHERE hotkey=? AND state=?",
                                       (hotkey, OPEN)).fetchone()[0])

    def _expire(self, now: float) -> int:
        due = [self._lease(row) for row in self.db.execute(
            f"SELECT {_COLUMNS} FROM rl_leases WHERE state=? AND expires_at<=?", (OPEN, now)).fetchall()]
        due = [lease for lease in due if lease.lease_id not in self._submitting]
        for lease in due:
            self.db.execute("UPDATE rl_leases SET state=?, closed_at=? WHERE lease_id=?",
                            (EXPIRED, now, lease.lease_id))
            self.quota.record_expiry(lease.hotkey, lease, now=now)
        return len(due)

    def forfeit(self, lease_id: str, hotkey: str, now: float | None = None) -> Lease:
        """The miner gives up its open lease now (e.g. episodes that could not all be played): closed and charged
        exactly as its expiry (``record_expiry``), so it is never cheaper than delivering, only sooner; the quota
        slot is freed now rather than at the lease's end. LookupError: no open lease of ``hotkey`` by that id, or
        one whose submission is being judged."""
        now = float(self.clock()) if now is None else now
        with self._txn():
            row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE lease_id=?", (lease_id,)).fetchone()
            lease = None if row is None else self._lease(row)
            if (lease is None or lease.hotkey != hotkey or lease.state != OPEN
                    or lease.lease_id in self._submitting):
                raise LookupError("no open lease of this hotkey to forfeit")
            self.db.execute("UPDATE rl_leases SET state=?, closed_at=? WHERE lease_id=?",
                            (EXPIRED, now, lease.lease_id))
            self.quota.record_expiry(lease.hotkey, lease, now=now)
        return lease

    def expire_due(self, now: float | None = None) -> int:
        now = float(self.clock()) if now is None else now
        with self._txn():
            return self._expire(now)

    def grant(self, requests: list[LeaseRequest], now: float | None = None) -> list[LeaseAnswer]:
        """Resolve one interval's requests together. Raises ValueError if any request is malformed. A banned hotkey,
        or one without room (``quota - in_use``, and its share of the global cap), is refused up front; a request is
        cut to its hotkey's room in its own order (the rest refused as ``quota`` without a lookup)."""
        for request in requests:
            self.check_request(request)
        now = float(self.clock()) if now is None else now
        answers = [LeaseAnswer() for _ in requests]
        granted: dict[int, dict[int, Lease]] = {i: {} for i in range(len(requests))}
        cap = self.config.max_pending_groups
        with self._txn():
            self._expire(now)
            latest = self.latest_checkpoint
            secrets_by_k = dict(self.db.execute("SELECT k, secret FROM rl_lease_secrets WHERE revealed_at IS NULL"))
            room: dict[str, int] = {}
            held: dict[tuple, Lease] = {}                 # the requesters' own open leases: a resend needs no room
            for hotkey in {r.hotkey for r in requests}:
                if self._banned(hotkey, now):
                    room[hotkey] = -1
                    continue
                rows = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE hotkey=? AND state=?",
                                       (hotkey, OPEN)).fetchall()
                for lease in map(self._lease, rows):
                    held[(hotkey, lease.environment, lease.prompt_idx, lease.checkpoint)] = lease
                limit = self.quota.quota(hotkey, now)
                if cap is not None:
                    limit = min(limit, HOTKEY_MAX_OPEN)
                room[hotkey] = max(0, limit - len(rows))
            claims = []
            for i, request in enumerate(requests):
                hotkey, asked = request.hotkey, 0
                for prompt in request.prompt_ids:
                    resent = held.get((hotkey, request.environment, prompt, request.checkpoint))
                    if room[hotkey] < 0:
                        answers[i].refused[prompt] = BANNED
                    elif resent is not None:
                        granted[i][prompt] = resent
                    elif asked >= room[hotkey]:
                        answers[i].refused[prompt] = QUOTA
                    else:
                        asked += 1
                        claims.append((i, prompt))
            self._rng.shuffle(claims)
            pending = self._pending() if cap is not None else 0
            for i, prompt in claims:
                request = requests[i]
                env, k, hotkey = request.environment, request.checkpoint, request.hotkey
                if k not in secrets_by_k:                             # not adopted, or S_k revealed
                    answers[i].refused[prompt] = STALE
                    continue
                # one live lease of a prompt at any k (no parallel leases over the unrevealed checkpoints), and
                # a key closed by a submission at this k
                row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE environment=? AND prompt_idx=? "
                                      "AND (state=? OR (k=? AND state=?)) ORDER BY k=? DESC LIMIT 1",
                                      (env, prompt, OPEN, k, SUBMITTED, k)).fetchone()
                if row is not None:
                    answers[i].refused[prompt] = TAKEN
                    continue
                lifecycle = self.db.execute("SELECT state, until_k FROM rl_prompt_state WHERE environment=? "
                                            "AND prompt_idx=?", (env, prompt)).fetchone()
                if lifecycle is not None and lifecycle[0] == BLACKLISTED:
                    answers[i].refused[prompt] = BLACKLISTED
                    continue
                if lifecycle is not None and latest < lifecycle[1]:
                    answers[i].refused[prompt] = COOLING
                    continue
                if room[hotkey] <= 0:
                    answers[i].refused[prompt] = QUOTA
                    continue
                if cap is not None and pending >= cap:
                    answers[i].refused[prompt] = FULL
                    self._full.add(env)
                    continue
                lease = Lease(_secrets.token_hex(16), hotkey, env, prompt, k,
                              lease_nonce(bytes(secrets_by_k[k]), env, prompt, k).hex(), now,
                              now + float(self.config.expiry_s[env]), OPEN)
                self.db.execute(f"INSERT INTO rl_leases({_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?)",
                                (lease.lease_id, hotkey, env, prompt, k, lease.nonce, lease.issued_at,
                                 lease.expires_at, OPEN))
                room[hotkey] -= 1
                self.quota.record_lease(hotkey, now=now)
                granted[i][prompt] = lease
        for i, request in enumerate(requests):
            answers[i].leases = [granted[i][p] for p in request.prompt_ids if p in granted[i]]
        return answers

    def _pending(self) -> int:
        """Buffered groups (``rl_buffer``, queued or taken, when a ``TrainingBuffer`` shares the file): the global
        cap's measure. Open leases are bounded per hotkey (quota, its share of the cap) and by expiry."""
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rl_buffer'").fetchone():
            return int(self.db.execute("SELECT COUNT(*) FROM rl_buffer").fetchone()[0])
        return 0

    def drain_full(self) -> set[str]:
        """Envs refused for the global cap since the last drain (the price controller's oversupply signal)."""
        with self.lock:
            full, self._full = self._full, set()
        return full

    def _submittable(self, lease_id: str, hotkey: str, now: float) -> tuple[Lease | None, Exception | None]:
        row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE lease_id=?", (lease_id,)).fetchone()
        lease = None if row is None else self._lease(row)
        if lease is None:
            return None, LookupError("lease unknown")
        if lease.hotkey != hotkey:
            return lease, PermissionError("lease held by another hotkey")
        if lease.state != OPEN:
            return lease, LookupError(f"lease {lease.state}")
        if lease.expires_at <= now:
            self._expire(now)                                         # committed: late is expired
            return lease, LookupError("lease expired")
        if self._banned(hotkey, now):
            return lease, PermissionError("hotkey banned")
        return lease, None

    def check_submission(self, lease_id: str, hotkey: str, now: float | None = None) -> Lease:
        """What ``submit`` would refuse, without closing the lease: the submission path checks before it grades
        (nothing is graded for a lease that cannot be submitted)."""
        now = float(self.clock()) if now is None else now
        with self._txn():
            lease, failure = self._submittable(lease_id, hotkey, now)
        if failure is not None:
            raise failure
        return lease

    def hold_submission(self, lease_id: str, hotkey: str, arrived_at: float) -> Lease:
        """A submission arrived at ``arrived_at`` for an open lease of ``hotkey`` (``check_submission``'s checks,
        raised the same way): until ``release_submission``, the sweep does not expire the lease, so an on-time
        submission still waiting for or going through grading is judged at its arrival."""
        with self.lock:
            lease = self.check_submission(lease_id, hotkey, arrived_at)
            self._submitting[lease_id] = self._submitting.get(lease_id, 0) + 1
        return lease

    def release_submission(self, lease_id: str) -> None:
        with self.lock:
            left = self._submitting.pop(lease_id, 1) - 1
            if left > 0:
                self._submitting[lease_id] = left

    def submit(self, lease_id: str, hotkey: str, *, in_zone: bool, delivered: bool = True,
               now: float | None = None) -> Lease:
        """End a lease by its submission (whatever its outcome). Closes its (env, prompt, k) and frees its quota
        slot (credited only when paid: ``credit``); ``delivered=False`` (a group refused for integrity: no work)
        halves the quota.
        LookupError: unknown, already ended or expired; PermissionError: another hotkey's lease, or a banned
        hotkey."""
        now = float(self.clock()) if now is None else now
        with self._txn():
            lease, failure = self._submittable(lease_id, hotkey, now)
            if failure is None:
                self.db.execute("UPDATE rl_leases SET state=?, closed_at=? WHERE lease_id=?",
                                (SUBMITTED, now, lease_id))
                if delivered:
                    self.quota.record_submission(hotkey, lease, in_zone=bool(in_zone), now=now)
                else:
                    self.quota.record_expiry(hotkey, lease, now=now, cause="rejected")
        if failure is not None:
            raise failure
        return replace(lease, state=SUBMITTED)

    def credit(self, lease_id: str, now: float | None = None) -> None:
        """A submitted lease's group was paid: one delivery for its hotkey's quota (``QuotaController.record_paid``).
        A lease that is not (or no longer) on record as submitted is ignored."""
        now = float(self.clock()) if now is None else now
        with self._txn():
            row = self.db.execute("SELECT hotkey, issued_at, closed_at FROM rl_leases WHERE lease_id=? AND state=?",
                                  (lease_id, SUBMITTED)).fetchone()
            if row is not None:
                self.quota.record_paid(row[0], issued_at=float(row[1]), submitted_at=float(row[2]), now=now)

    # --- bans (set by the audit regime of the submission path) ----------------------------------------------
    def _banned(self, hotkey: str, now: float) -> bool:
        row = self.db.execute("SELECT until FROM rl_hotkey_bans WHERE hotkey=?", (hotkey,)).fetchone()
        return row is not None and now < row[0]

    def ban(self, hotkey: str, until: float) -> None:
        """No lease and no submission for ``hotkey`` before ``until`` (a running ban is never shortened)."""
        with self._txn():
            self.db.execute("INSERT INTO rl_hotkey_bans(hotkey, until) VALUES(?, ?) ON CONFLICT(hotkey) "
                            "DO UPDATE SET until=MAX(until, excluded.until)", (hotkey, float(until)))

    def banned_until(self, hotkey: str) -> float | None:
        with self.lock:
            row = self.db.execute("SELECT until FROM rl_hotkey_bans WHERE hotkey=?", (hotkey,)).fetchone()
        return None if row is None else float(row[0])

    # --- prompt lifecycle (§4.5) -----------------------------------------------------------------------------
    def record_outcome(self, environment: str, prompt_idx: int, checkpoint: int, rewards, verdict: str,
                       now: float | None = None) -> Scan:
        """Record a graded group of (env, prompt) leased at k and move the prompt: a second ``too_easy`` scan
        blacklists it for the run (one group is not proof enough to retire a prompt for good); any other scan cools
        it down until ``latest + cooldown`` (never shortening a running cooldown, never lifting a blacklist). The
        cooldown counts from the latest k, not the lease's: a lagged lease must not shorten it."""
        if environment not in self.config.expiry_s:
            raise ValueError("environment not served")
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        _u64(prompt_idx)
        _u64(checkpoint)
        rewards = tuple(float(r) for r in rewards)
        if not rewards or not all(math.isfinite(r) for r in rewards):
            raise ValueError("finite rewards required")
        cooldown = self.config.cooldown_of(environment)
        _u64(cooldown)
        now = float(self.clock()) if now is None else now
        with self._txn():
            prior = self.db.execute("SELECT state, until_k FROM rl_prompt_state WHERE environment=? AND prompt_idx=?",
                                    (environment, prompt_idx)).fetchone()
            easy_before = verdict == TOO_EASY and self.db.execute(
                "SELECT 1 FROM rl_prompt_scans WHERE environment=? AND prompt_idx=? AND verdict=? LIMIT 1",
                (environment, prompt_idx, TOO_EASY)).fetchone() is not None
            if easy_before or (prior is not None and prior[0] == BLACKLISTED):
                state, until = BLACKLISTED, None
            else:
                start = max(checkpoint, self.latest_checkpoint or 0)
                state, until = COOLDOWN, max(start + cooldown, (prior or (0, 0))[1])
            # The scan keeps the state it moved the prompt to: the published log (``rl_observations``) rebuilds
            # every prompt's state from the scans alone.
            seq = self.db.execute("INSERT INTO rl_prompt_scans(environment, prompt_idx, k, rewards, verdict, "
                                  "first_scan, at, state, until_k) VALUES(?,?,?,?,?,?,?,?,?)",
                                  (environment, prompt_idx, checkpoint, json.dumps(rewards), verdict,
                                   int(prior is None), now, state, until)).lastrowid
            self.db.execute("INSERT OR REPLACE INTO rl_prompt_state(environment, prompt_idx, state, until_k, seq) "
                            "VALUES(?,?,?,?,?)", (environment, prompt_idx, state, until, seq))
        return Scan(int(seq), environment, prompt_idx, checkpoint, rewards, verdict, prior is None, now)

    def prompt_state(self, since: int = 0, *, environment: str | None = None,
                     limit: int = MAX_SCANS_PER_PAGE) -> dict:
        """The miners' view after cursor ``since``: up to ``limit`` scans and the current state of the prompts they
        moved. Poll again with ``cursor`` (``more``: a next page is ready). A cooldown ends once ``latest`` reaches
        its ``until_k``; a prompt never listed is never scanned."""
        _u64(since)
        where, args = ("seq>?", [since]) if environment is None else ("seq>? AND environment=?", [since, environment])
        with self.lock:
            rows = self.db.execute(f"SELECT seq, environment, prompt_idx, k, rewards, verdict, first_scan, at FROM "
                                   f"rl_prompt_scans WHERE {where} ORDER BY seq LIMIT ?", (*args, limit + 1)).fetchall()
            more, rows = len(rows) > limit, rows[:limit]
            cursor = rows[-1][0] if rows else since
            states = self.db.execute(f"SELECT environment, prompt_idx, state, until_k FROM rl_prompt_state WHERE "
                                     f"{where} AND seq<=? ORDER BY seq", (*args, cursor)).fetchall()
            latest = self.latest_checkpoint
        scans = [Scan(seq, env, p, k, tuple(json.loads(r)), v, bool(f), at).to_public()
                 for seq, env, p, k, r, v, f, at in rows]
        return {"latest": latest, "cursor": cursor, "more": more, "scans": scans,
                "prompts": [{"environment": env, "prompt_idx": p, "state": state, "until_k": until}
                            for env, p, state, until in states]}


class LeaseCollector:
    """Batches the requests that arrive within one interval and resolves them together off the event loop."""

    def __init__(self, book: LeaseBook, *, interval_s: float = INTERVAL_S) -> None:
        self.book = book
        self.interval_s = float(interval_s)
        self._pending: list[tuple[LeaseRequest, asyncio.Future]] = []
        self._flush: asyncio.Task | None = None

    async def request(self, request: LeaseRequest) -> LeaseAnswer:
        self.book.check_request(request)                      # malformed: refused at once, never batched
        future = asyncio.get_running_loop().create_future()
        self._pending.append((request, future))
        if self._flush is None:
            self._flush = asyncio.create_task(self._resolve_after_interval())
        return await future

    async def _resolve_after_interval(self) -> None:
        try:
            await asyncio.sleep(self.interval_s)
        finally:
            batch, self._pending, self._flush = self._pending, [], None
        try:
            answers = await asyncio.to_thread(self.book.grant, [r for r, _ in batch])
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        for (_, future), answer in zip(batch, answers):
            if not future.done():
                future.set_result(answer)


__all__ = [
    "BANNED", "BLACKLISTED", "COMMIT_DOMAIN", "COOLDOWN", "COOLING", "EXPIRED", "FULL", "IN_ZONE", "NONCE_DOMAIN", "OPEN", "QUOTA",
    "REFUSALS", "SECRET_REVEAL_LAG", "STALE", "SUBMITTED", "TAKEN", "TOO_EASY", "TOO_HARD", "TRUNCATED", "VERDICTS",
    "Lease", "LeaseAnswer", "LeaseBook", "LeaseBookConfig", "LeaseCollector", "LeaseRequest", "Scan",
    "classify_group", "lease_expiry_s", "lease_nonce", "secret_commitment", "verify_reveal",
]
