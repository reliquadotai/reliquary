"""The admin host's index of customer env versions (sqlite), and the frozen rows validators read.

A version's record moves queued -> building -> qualifying -> qualified | refused, and qualified -> disabled.
Its owner may make a qualified version that names a license public: any account may then order it, for good
(orders already placed on it keep it). ``listed`` puts a public version in the catalog; the owner may take it
out again. Its ``source`` (where an imported env comes from) is shown with it.
The frozen prompts and the final record are also written create-only to the subnet bucket
(``reliquary/customer-envs/<account>/<name>/<sha256>/``): validators read them there, never this index.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from reliquary.customer_env import EnvId, parse_env_id

STATUSES = ("queued", "building", "qualifying", "qualified", "refused", "disabled")
REASONS = ("package_invalid", "build_failed", "start_failed", "freeze_failed", "examples_failed",
           "nondeterministic", "agentic_not_supported", "infrastructure")
SUBNET_PREFIX = "reliquary/customer-envs"


def subnet_key(env: EnvId, name: str) -> str:
    return f"{SUBNET_PREFIX}/{env.account}/{env.name}/{env.sha256}/{name}"


@dataclass
class Record:
    env_id: str
    account: str
    name: str
    sha256: str
    object_key: str
    status: str = "queued"
    reason: str | None = None
    detail: str | None = None
    kind: str | None = None
    uses: list[str] = field(default_factory=list)
    image_digest: str | None = None
    prompt_set: dict | None = None
    examples: list[dict] = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    created_at: float = 0.0
    qualified_at: float | None = None
    license: str | None = None
    source: dict | None = None
    public: bool = False
    listed: bool = False
    published_at: float | None = None

    def view(self) -> dict:
        """What the platform reads: everything but the bucket key."""
        out = asdict(self)
        out.pop("object_key")
        return out


class Registry:
    def __init__(self, path: str, *, clock=time.time) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS customer_envs (env_id TEXT PRIMARY KEY, account TEXT NOT NULL,"
                         " status TEXT NOT NULL, body TEXT NOT NULL)")
        self._db.execute("CREATE INDEX IF NOT EXISTS customer_envs_account ON customer_envs(account)")
        self._lock = threading.Lock()
        self._clock = clock

    def _put(self, record: Record) -> Record:
        if record.status not in STATUSES or (record.reason is not None and record.reason not in REASONS):
            raise ValueError(f"bad record state {record.status}/{record.reason}")
        self._db.execute("INSERT OR REPLACE INTO customer_envs VALUES (?,?,?,?)",
                         (record.env_id, record.account, record.status, json.dumps(asdict(record), sort_keys=True)))
        return record

    def submit(self, env: EnvId, object_key: str) -> tuple[Record, bool]:
        """The version's record, and whether this call created it (a resubmission is idempotent, except that
        an ``infrastructure`` refusal is queued again)."""
        with self._lock:
            existing = self.get(str(env))
            if existing is not None and not (existing.status == "refused" and existing.reason == "infrastructure"):
                return existing, False
            return self._put(Record(str(env), env.account, env.name, env.sha256, object_key,
                                    created_at=self._clock())), True

    def update(self, env_id: str, **changes) -> Record:
        with self._lock:
            record = self.get(env_id)
            if record is None:
                raise KeyError(env_id)
            for key, value in changes.items():
                if not hasattr(record, key):
                    raise AttributeError(key)
                setattr(record, key, value)
            if changes.get("status") == "qualified":
                record.qualified_at = self._clock()
            return self._put(record)

    def get(self, env_id: str) -> Record | None:
        row = self._db.execute("SELECT body FROM customer_envs WHERE env_id = ?", (env_id,)).fetchone()
        return Record(**json.loads(row[0])) if row else None

    def for_account(self, account: str) -> list[Record]:
        rows = self._db.execute("SELECT body FROM customer_envs WHERE account = ? ORDER BY env_id", (account,))
        return sorted((Record(**json.loads(r[0])) for r in rows), key=lambda r: r.created_at, reverse=True)

    def catalog(self) -> list[Record]:
        """The public, listed, qualified versions, newest first."""
        rows = self._db.execute("SELECT body FROM customer_envs WHERE status = 'qualified'"
                                " AND json_extract(body, '$.public') = 1 AND json_extract(body, '$.listed') = 1")
        return sorted((Record(**json.loads(r[0])) for r in rows), key=lambda r: r.published_at or 0, reverse=True)

    def publish(self, env_id: str, *, account: str, public: bool | None, listed: bool | None) -> Record:
        """The owner's change of visibility, or ValueError: public needs a qualified version with a license and
        is final; listed needs public."""
        with self._lock:
            record = self.get(env_id)
            if record is None:
                raise KeyError(env_id)
            if record.account != account:
                raise ValueError("customer_env_not_yours")
            if public is False and record.public:
                raise ValueError("customer_env_public_is_final: unlist it instead")
            if public and not record.public:
                if record.status != "qualified":
                    raise ValueError(f"customer_env_not_qualified: {record.status}")
                if not record.license:
                    raise ValueError("customer_env_license_missing: name the license in [tool.reliquary]")
                record.public, record.published_at = True, self._clock()
                if listed is None:
                    listed = True
            if listed and not record.public:
                raise ValueError("customer_env_not_public")
            if listed is not None:
                record.listed = bool(listed)
            return self._put(record)

    def in_flight(self, account: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM customer_envs WHERE account = ? AND status IN"
                                " ('queued','building','qualifying')", (account,)).fetchone()[0]

    def pending(self) -> Iterable[Record]:
        rows = self._db.execute("SELECT body FROM customer_envs WHERE status IN ('queued','building','qualifying')")
        return sorted((Record(**json.loads(r[0])) for r in rows), key=lambda r: r.created_at)


def resolve(registry: Registry, env: str, *, account: str, product: str) -> Record:
    """A qualified version an order of ``account`` may use for ``product``, or ValueError."""
    from reliquary.customer_env import check_order_env

    parse_env_id(env)
    record = registry.get(env)
    if record is None:
        raise ValueError("customer_env_unknown")
    if record.status != "qualified":
        raise ValueError(f"customer_env_not_qualified: {record.status}")
    check_order_env(env, account=account, product=product, uses=record.uses, public=record.public)
    return record


__all__ = ["REASONS", "Record", "Registry", "STATUSES", "SUBNET_PREFIX", "resolve", "subnet_key"]
