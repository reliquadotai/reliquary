"""How many RL leases a hotkey may hold open at once (rl-async-design.md §4.3).

The quota is measured, never declared. Per hotkey:

* ``cap`` follows AIMD. A new hotkey starts at ``initial`` (4). A *round* is one quota's worth of deliveries
  counted from the last change: when the round ends with the hotkey having held its full quota at some point
  (``saturated``) and no lease expired, ``cap`` doubles (slow start: 4 -> ~1,600 in ~9 rounds). Every expiry
  halves the quota (never below ``floor``) and ends slow start.
* ``target`` is Little's law over a sliding window: measured throughput (deliveries per second over the window,
  or over the hotkey's life when shorter) x measured mean lease-to-submit latency x ``headroom`` (1.5). It is the
  concurrency the hotkey actually sustains, plus room to grow.
* While in slow start the quota is ``cap``. A round that ends without the hotkey ever using its full quota means
  it found its own pace: it leaves slow start. Once steady, the quota is ``min(cap, target)`` (``cap`` alone until
  ``MIN_SAMPLES`` deliveries are in the window), so an idle or slowed hotkey cannot keep a large quota to block
  prompts with, and a growing one regains quota by about ``headroom`` per window and by doubling ``cap``.
* Everything is clamped to ``[floor, ceiling]``.

Out-of-zone submissions are delivered work like in-zone ones (the lease made the miner generate them).
``record_submission`` is what the submission path (P3) calls after it closed a lease (``LeaseBook.submit`` does
it for the caller); ``record_expiry`` is called by the lease book's sweep, and by ``LeaseBook.submit`` for a
submission refused for integrity (``cause="rejected"``: halved like an expiry).

Persistence: ``attach(db, lock)`` stores one JSON row per hotkey in the lease book's SQLite file (the latency
samples are kept as ``BUCKET_S`` buckets of (count, latency sum), so a row stays small at any rate). Without
``attach`` the controller is memory only (tests, tools).
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field

SLOW_START = "slow_start"
STEADY = "steady"
BUCKET_S = 10.0       # latency samples are kept as (count, sum) per bucket: a row stays small at any rate
MIN_SAMPLES = 8       # deliveries in the window before Little's law is trusted


@dataclass(frozen=True)
class QuotaConfig:
    initial: int = 4
    floor: int = 1
    ceiling: int = 2048
    headroom: float = 1.5
    window_s: float = 900.0

    def __post_init__(self) -> None:
        if not (1 <= self.floor <= self.initial <= self.ceiling):
            raise ValueError("quota bounds must satisfy 1 <= floor <= initial <= ceiling")
        if not self.headroom >= 1.0:
            raise ValueError("quota headroom must be >= 1")
        if not self.window_s >= BUCKET_S:
            raise ValueError(f"quota window must be at least {BUCKET_S} s")


def _counters() -> dict[str, int]:
    return {"leased": 0, "submitted": 0, "in_zone": 0, "out_of_zone": 0, "expired": 0, "rejected": 0}


@dataclass
class _Hotkey:
    cap: int
    phase: str = SLOW_START
    reason: str = "new hotkey: initial quota"
    in_use: int = 0
    since: float | None = None                  # first lease: the throughput span never predates it
    round_quota: int = 0                        # the quota the current round started at
    delivered: int = 0                          # deliveries in the current round
    saturated: bool = False                     # held its full quota during the current round
    counters: dict[str, int] = field(default_factory=_counters)
    buckets: dict[int, list[float]] = field(default_factory=dict)   # bucket -> [count, latency sum]

    def to_json(self) -> str:
        body = {key: getattr(self, key) for key in ("cap", "phase", "reason", "in_use", "since", "round_quota",
                                                     "delivered", "saturated", "counters")}
        body["buckets"] = [[b, c, s] for b, (c, s) in sorted(self.buckets.items())]
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "_Hotkey":
        body = json.loads(text)
        buckets = {int(b): [float(c), float(s)] for b, c, s in body.pop("buckets")}
        counters = {**_counters(), **{k: int(v) for k, v in body.pop("counters").items()}}
        return cls(**body, counters=counters, buckets=buckets)


class QuotaController:
    def __init__(self, config: QuotaConfig | None = None) -> None:
        self.config = config or QuotaConfig()
        self._hotkeys: dict[str, _Hotkey] = {}
        self._lock: threading.RLock = threading.RLock()
        self._db = None

    # --- persistence ----------------------------------------------------------------------------------------
    def attach(self, db, lock: threading.RLock) -> None:
        """Persist into ``db`` (the lease book's connection, guarded by ``lock``) and load what it holds."""
        with lock:
            db.execute("CREATE TABLE IF NOT EXISTS rl_lease_quota(hotkey TEXT PRIMARY KEY, state TEXT NOT NULL)")
            self._hotkeys = {hotkey: _Hotkey.from_json(state)
                             for hotkey, state in db.execute("SELECT hotkey, state FROM rl_lease_quota")}
            self._db, self._lock = db, lock

    def _save(self, hotkey: str, entry: _Hotkey) -> None:
        if self._db is None:
            return
        row = (hotkey, entry.to_json())
        if self._db.in_transaction:          # inside the lease book's transaction: committed with it
            self._db.execute("INSERT OR REPLACE INTO rl_lease_quota(hotkey, state) VALUES(?,?)", row)
        else:
            with self._db:
                self._db.execute("INSERT OR REPLACE INTO rl_lease_quota(hotkey, state) VALUES(?,?)", row)

    def reconcile(self, open_counts: dict[str, int]) -> None:
        """Align ``in_use`` with the lease book's open leases (after a restart)."""
        with self._lock:
            for hotkey in set(open_counts) | {h for h, e in self._hotkeys.items() if e.in_use}:
                entry = self._entry(hotkey)
                if entry.in_use != open_counts.get(hotkey, 0):
                    entry.in_use = open_counts.get(hotkey, 0)
                    self._save(hotkey, entry)

    # --- reading --------------------------------------------------------------------------------------------
    def _entry(self, hotkey: str) -> _Hotkey:
        entry = self._hotkeys.get(hotkey)
        if entry is None:
            entry = self._hotkeys[hotkey] = _Hotkey(cap=self.config.initial)
            entry.round_quota = self.config.initial
        return entry

    def _clamp(self, n: float) -> int:
        return max(self.config.floor, min(self.config.ceiling, int(n)))

    def _measure(self, entry: _Hotkey, now: float | None) -> tuple[int, float | None, float | None]:
        """(deliveries in the window, throughput per second, mean latency in seconds)."""
        if not entry.buckets:
            return 0, None, None
        last = max(entry.buckets)
        now = (last + 1) * BUCKET_S if now is None else now
        first = math.floor((now - self.config.window_s) / BUCKET_S)
        live = [(c, s) for b, (c, s) in entry.buckets.items() if b > first]
        count = sum(c for c, _ in live)
        if not count:
            return 0, None, None
        span = self.config.window_s if entry.since is None else min(self.config.window_s, now - entry.since)
        span = max(span, BUCKET_S)
        return int(count), count / span, sum(s for _, s in live) / count

    def _target(self, entry: _Hotkey, now: float | None) -> float | None:
        count, rate, latency = self._measure(entry, now)
        if count < MIN_SAMPLES or rate is None:
            return None
        return self.config.headroom * rate * latency

    def _quota(self, entry: _Hotkey, now: float | None) -> int:
        if entry.phase == SLOW_START:
            return self._clamp(entry.cap)
        target = self._target(entry, now)
        if target is None:
            return self._clamp(entry.cap)
        return self._clamp(min(entry.cap, math.ceil(target)))

    def quota(self, hotkey: str, now: float | None = None) -> int:
        with self._lock:
            return self._quota(self._hotkeys.get(hotkey) or _Hotkey(cap=self.config.initial), now)

    def in_use(self, hotkey: str) -> int:
        with self._lock:
            entry = self._hotkeys.get(hotkey)
            return 0 if entry is None else entry.in_use

    def state(self, hotkey: str, now: float | None = None) -> dict:
        """The explainable state published by ``GET /rl/quota``."""
        with self._lock:
            entry = self._hotkeys.get(hotkey) or _Hotkey(cap=self.config.initial)
            count, rate, latency = self._measure(entry, now)
            target = self._target(entry, now)
            quota = self._quota(entry, now)
            reason = entry.reason
            if entry.phase == STEADY and target is not None and quota < self._clamp(entry.cap):
                reason = (f"steady: throughput {rate * 60:.2f}/min x latency {latency:.1f}s x headroom "
                          f"{self.config.headroom:g} = {target:.1f}")
            return {
                "hotkey": hotkey, "quota": quota, "in_use": entry.in_use, "cap": entry.cap, "phase": entry.phase,
                "reason": reason, "counters": dict(entry.counters),
                "throughput_per_min": None if rate is None else rate * 60.0,
                "latency_s": latency, "target": target, "window_samples": count,
                "round": {"quota": entry.round_quota, "delivered": entry.delivered, "saturated": entry.saturated},
                "floor": self.config.floor, "ceiling": self.config.ceiling, "headroom": self.config.headroom,
                "window_s": self.config.window_s,
            }

    # --- events ---------------------------------------------------------------------------------------------
    def _new_round(self, entry: _Hotkey, now: float) -> None:
        entry.round_quota = self._quota(entry, now)
        entry.delivered = 0
        entry.saturated = entry.in_use >= entry.round_quota

    def record_lease(self, hotkey: str, *, now: float) -> None:
        with self._lock:
            entry = self._entry(hotkey)
            if entry.since is None:
                entry.since = now
            entry.in_use += 1
            entry.counters["leased"] += 1
            if entry.in_use >= self._quota(entry, now):
                entry.saturated = True
            self._save(hotkey, entry)

    def record_submission(self, hotkey: str, lease, *, in_zone: bool, now: float) -> None:
        """A lease ended by its submission, on time (any outcome: out of zone is delivered work)."""
        with self._lock:
            entry = self._entry(hotkey)
            entry.in_use = max(0, entry.in_use - 1)
            entry.counters["submitted"] += 1
            entry.counters["in_zone" if in_zone else "out_of_zone"] += 1
            bucket = entry.buckets.setdefault(math.floor(now / BUCKET_S), [0.0, 0.0])
            bucket[0] += 1
            bucket[1] += max(0.0, now - float(lease.issued_at))
            oldest = math.floor((now - self.config.window_s) / BUCKET_S)
            for b in [b for b in entry.buckets if b <= oldest]:
                del entry.buckets[b]
            entry.delivered += 1
            if entry.delivered >= max(1, entry.round_quota):
                if entry.saturated:
                    entry.cap = self._clamp(max(entry.cap, 2 * entry.round_quota))
                    entry.reason = (f"{'slow start' if entry.phase == SLOW_START else 'growth'}: full quota "
                                    f"{entry.round_quota} used and delivered on time, cap doubled")
                elif entry.phase == SLOW_START:
                    entry.phase = STEADY
                    entry.reason = f"steady: delivered {entry.round_quota} without using the full quota"
                self._new_round(entry, now)
            self._save(hotkey, entry)

    def record_expiry(self, hotkey: str, lease, *, now: float, cause: str = "expired") -> None:
        """A lease ran past its expiry without a submission (``cause="expired"``), or its submission was refused
        for integrity (``"rejected"``: malformed or forged work is no delivery): halve the quota."""
        if cause not in ("expired", "rejected"):
            raise ValueError("cause must be expired or rejected")
        with self._lock:
            entry = self._entry(hotkey)
            before = self._quota(entry, now)
            entry.in_use = max(0, entry.in_use - 1)
            entry.counters[cause] += 1
            entry.cap = self._clamp(before // 2)
            entry.phase = STEADY
            entry.reason = f"{cause} lease: quota halved from {before}"
            self._new_round(entry, now)
            self._save(hotkey, entry)


__all__ = ["SLOW_START", "STEADY", "QuotaConfig", "QuotaController"]
