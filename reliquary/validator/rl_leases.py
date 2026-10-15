"""The RL lease book (rl-async-design.md §4.2; implementation map §1.2): a miner reserves prompts before it
generates them, and every lease ends in a submission or an expiry (there is no release).

* **Exclusive.** At most one live lease per ``(environment, prompt, checkpoint k)``; once a lease of it is
  submitted, the key is closed for that k (the draw would be identical anyway). A SQLite partial unique index
  enforces it. An expired lease frees its key, and the next lessee gets the same draw (resolution C12: accepted,
  expiry earns nothing and halves the quota).
* **Fair inside an interval.** ``LeaseCollector`` gathers the requests that arrive within ``interval_s`` (~1 s)
  and ``LeaseBook.grant`` resolves the whole batch at once: every (request, prompt) claim is shuffled with the
  validator's own randomness (``random.SystemRandom``) and taken in that order, so a contested prompt goes to a
  uniformly random requester that still has quota; arriving first inside the interval buys nothing.
* **Refusals say only** ``taken`` / ``cooling`` / ``quota`` / ``stale``: never who holds a prompt, when its lease
  ends, or which leases are open. ``taken`` covers both an open lease of someone else and a key closed by a
  submission. A resend by the holder answers its own open lease again (a lost response is not a lost lease).
* **Cooling** (resolution C6): a prompt trained recently is refused. The book asks a ``cooling(env, prompt,
  latest_k)`` predicate; ``checkpoint_cooldown`` adapts the existing per-env ``CooldownMap`` with checkpoints as
  its unit (the trainer side records ``record_batched(prompt, k)`` when it trains a prompt, P4).
* **Stale:** a lease on a k that is not adopted, or older than ``latest - max_lag`` (S = 4), is refused.
* **Expiry** is fixed per environment (``lease_expiry_s``: the env's max tokens at a slow reference speed, times a
  margin), never derived from the miner. ``expire_due`` sweeps (each ``grant`` and ``submit`` sweeps too) and
  halves the hotkey's quota.
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
import secrets as _secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from reliquary.validator.rl_lease_quota import QuotaController

TAKEN, COOLING, QUOTA, STALE = "taken", "cooling", "quota", "stale"
REFUSALS = frozenset({TAKEN, COOLING, QUOTA, STALE})
OPEN, SUBMITTED, EXPIRED = "open", "submitted", "expired"

NONCE_DOMAIN = b"reliquary/rl-lease-nonce/v1"
COMMIT_DOMAIN = b"reliquary/rl-lease-secret-commit/v1"
SECRET_BYTES = 32
MAX_REVEALED_LISTED = 64
MAX_PROMPTS_PER_REQUEST = 1024
INTERVAL_S = 1.0          # the collection interval inside which arrival order buys nothing
RESTART_GRACE_S = 120.0   # what a lease that fell due while the validator was down gets at boot


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


def checkpoint_cooldown(maps: Mapping[str, object]) -> Callable[[str, int, int], bool]:
    """A ``cooling`` predicate over per-env ``CooldownMap``s whose unit is the checkpoint k."""
    def cooling(environment: str, prompt_idx: int, checkpoint: int) -> bool:
        cooldown = maps.get(environment)
        return cooldown is not None and cooldown.is_in_cooldown(prompt_idx, checkpoint)
    return cooling


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
    max_lag: int = 4
    prompt_range: Mapping[str, tuple[int, int]] | None = None  # per env: ``rl_prompt_ranges.rl_prompt_range``

    def __post_init__(self) -> None:
        if not self.expiry_s or any(not (float(v) > 0) for v in self.expiry_s.values()):
            raise ValueError("every served environment needs a positive lease expiry")
        if self.max_lag < 0:
            raise ValueError("max_lag must be non-negative")


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
"""
_COLUMNS = "lease_id, hotkey, environment, prompt_idx, k, nonce, issued_at, expires_at, state"


def _private_file(path: Path) -> None:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(path, 0o600)


class LeaseBook:
    def __init__(self, path: str | os.PathLike[str], config: LeaseBookConfig, *, quota: QuotaController,
                 clock: Callable[[], float] = time.time,
                 cooling: Callable[[str, int, int], bool] | None = None,
                 rng: random.Random | None = None) -> None:
        self.config = config
        self.quota = quota
        self.clock = clock
        self.cooling = cooling
        self._rng = rng or random.SystemRandom()
        self.lock = threading.RLock()
        path = Path(path)
        _private_file(path)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript(_SCHEMA)
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
        with self.lock:
            if self.db.in_transaction:
                raise RuntimeError("lease book transaction is already open")
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

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

    def _active_range(self) -> tuple[int, int] | None:
        latest = self.latest_checkpoint
        return None if latest is None else (max(0, latest - self.config.max_lag), latest)

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
        if bounds is not None and any(not (bounds[0] <= p < bounds[1]) for p in ids):
            raise ValueError("prompt outside the environment's RL range")

    @staticmethod
    def _lease(row) -> Lease:
        lease_id, hotkey, env, prompt, k, nonce, issued, expires, state = row
        return Lease(lease_id, hotkey, env, int(prompt), int(k), nonce, float(issued), float(expires), state)

    def lease(self, lease_id: str) -> Lease | None:
        with self.lock:
            row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE lease_id=?", (lease_id,)).fetchone()
        return None if row is None else self._lease(row)

    def in_use(self, hotkey: str) -> int:
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM rl_leases WHERE hotkey=? AND state=?",
                                       (hotkey, OPEN)).fetchone()[0])

    def _expire(self, now: float) -> int:
        due = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE state=? AND expires_at<=?",
                              (OPEN, now)).fetchall()
        for row in due:
            lease = self._lease(row)
            self.db.execute("UPDATE rl_leases SET state=?, closed_at=? WHERE lease_id=?",
                            (EXPIRED, now, lease.lease_id))
            self.quota.record_expiry(lease.hotkey, lease, now=now)
        return len(due)

    def expire_due(self, now: float | None = None) -> int:
        now = float(self.clock()) if now is None else now
        with self._txn():
            return self._expire(now)

    def grant(self, requests: list[LeaseRequest], now: float | None = None) -> list[LeaseAnswer]:
        """Resolve one interval's requests together. Raises ValueError if any request is malformed."""
        for request in requests:
            self.check_request(request)
        now = float(self.clock()) if now is None else now
        answers = [LeaseAnswer() for _ in requests]
        claims = [(i, p) for i, request in enumerate(requests) for p in request.prompt_ids]
        self._rng.shuffle(claims)
        granted: dict[int, dict[int, Lease]] = {i: {} for i in range(len(requests))}
        with self._txn():
            self._expire(now)
            active = self._active_range()
            secrets_by_k = dict(self.db.execute("SELECT k, secret FROM rl_lease_secrets WHERE revealed_at IS NULL"))
            in_use: dict[str, int] = {}
            for i, prompt in claims:
                request = requests[i]
                env, k, hotkey = request.environment, request.checkpoint, request.hotkey
                if active is None or not (active[0] <= k <= active[1]) or k not in secrets_by_k:
                    answers[i].refused[prompt] = STALE
                    continue
                row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE environment=? AND prompt_idx=? "
                                      "AND k=? AND state IN (?, ?)", (env, prompt, k, OPEN, SUBMITTED)).fetchone()
                if row is not None:
                    held = self._lease(row)
                    if held.state == OPEN and held.hotkey == hotkey:
                        granted[i][prompt] = held                     # the holder's resend
                    else:
                        answers[i].refused[prompt] = TAKEN
                    continue
                if self.cooling is not None and self.cooling(env, prompt, active[1]):
                    answers[i].refused[prompt] = COOLING
                    continue
                if hotkey not in in_use:
                    in_use[hotkey] = int(self.db.execute(
                        "SELECT COUNT(*) FROM rl_leases WHERE hotkey=? AND state=?", (hotkey, OPEN)).fetchone()[0])
                if in_use[hotkey] >= self.quota.quota(hotkey, now):
                    answers[i].refused[prompt] = QUOTA
                    continue
                lease = Lease(_secrets.token_hex(16), hotkey, env, prompt, k,
                              lease_nonce(bytes(secrets_by_k[k]), env, prompt, k).hex(), now,
                              now + float(self.config.expiry_s[env]), OPEN)
                self.db.execute(f"INSERT INTO rl_leases({_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?)",
                                (lease.lease_id, hotkey, env, prompt, k, lease.nonce, lease.issued_at,
                                 lease.expires_at, OPEN))
                in_use[hotkey] += 1
                self.quota.record_lease(hotkey, now=now)
                granted[i][prompt] = lease
        for i, request in enumerate(requests):
            answers[i].leases = [granted[i][p] for p in request.prompt_ids if p in granted[i]]
        return answers

    def submit(self, lease_id: str, hotkey: str, *, in_zone: bool, now: float | None = None) -> Lease:
        """End a lease by its submission (P3 calls this once the group is received, whatever its outcome). Closes
        its (env, prompt, k) and credits the hotkey's quota. LookupError: unknown, already ended or expired;
        PermissionError: another hotkey's lease."""
        now = float(self.clock()) if now is None else now
        failure: Exception | None = None
        with self._txn():
            row = self.db.execute(f"SELECT {_COLUMNS} FROM rl_leases WHERE lease_id=?", (lease_id,)).fetchone()
            lease = None if row is None else self._lease(row)
            if lease is None:
                failure = LookupError("lease unknown")
            elif lease.hotkey != hotkey:
                failure = PermissionError("lease held by another hotkey")
            elif lease.state != OPEN:
                failure = LookupError(f"lease {lease.state}")
            elif lease.expires_at <= now:
                self._expire(now)                                     # committed: late is expired
                failure = LookupError("lease expired")
            else:
                self.db.execute("UPDATE rl_leases SET state=?, closed_at=? WHERE lease_id=?",
                                (SUBMITTED, now, lease_id))
                self.quota.record_submission(hotkey, lease, in_zone=bool(in_zone), now=now)
        if failure is not None:
            raise failure
        return replace(lease, state=SUBMITTED)


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
    "COMMIT_DOMAIN", "COOLING", "EXPIRED", "NONCE_DOMAIN", "OPEN", "QUOTA", "REFUSALS", "STALE", "SUBMITTED",
    "TAKEN", "Lease", "LeaseAnswer", "LeaseBook", "LeaseBookConfig", "LeaseCollector", "LeaseRequest",
    "checkpoint_cooldown", "lease_expiry_s", "lease_nonce", "secret_commitment", "verify_reveal",
]
