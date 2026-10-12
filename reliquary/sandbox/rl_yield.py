"""What each miner's RL sandbox sessions turned into ("yield").

A session is decided when its precommit is settled: the runtime has settled the precommit's window
(``ServiceRuntime.window_uses``, asked by the RL wiring) and none of the precommit's sessions is held or
claimed (``SessionBook.precommits_held``). Each session of the precommit is then:

* a FAULT (neither useful nor wasted) when the machine or we failed it: a machine-signed ``aborted`` final,
  a drain (``voided``), a fault recorded on its session record (``runtime_exec_failed``: a proven runtime
  start failure; ``open_refused:<reason>``: a gateway-signed open refusal; ``validator_restarted``: an
  ``open_failed`` of a token we lost in a restart; ``grading_pool``: other miners' gradings held the pool), a
  window the runtime settled as aborted, a group of the precommit the proof handed back, and, unless a group of
  it was used: a precommit whose verdict is ours (``VALIDATOR_STAGES``: a group the checker verified that we
  then failed), or of the window a validator restart interrupted (``decide(resumed=True)``) unless a real
  refusal names it; and a session GRADED (it ran and finished: whatever else the miner's boxes did, expired,
  failed or exhausted their budget, stays waste) of an unused precommit whose opens met a capacity refusal of ours
  (``CAPACITY``) when fewer than M of the precommit's sessions were graded (the group could not be formed);
* USEFUL when a group of the precommit was trained or paid as first-scan exploration in that settlement
  (``used``): the M submitted sessions and the unselected over-generated ones alike;
* WASTED otherwise: expired, withdrawn, lapsed, ``box_failed`` (the agent's box), an unsigned
  ``open_failed`` (the miner's word: a box may run behind it), a refused group, a group accepted at intake but neither trained
  nor paid.

Verdicts. A precommit's verdict only rises: CAPACITY < OURS < ACCEPTED < REFUSED < HANDED_BACK,
so a real refusal is never cleared by a later "ours", nor by a later accepted group (a used group is useful
whatever the verdict; an unused one in the window a restart interrupted stays waste once a real refusal named
it). A refusal reached before the checker verified the group
(``NO_VERDICT_STAGES``: the precommit it names is unverified, and a miner can provoke these at will: a stale
directory, a busy checker, a window that is not open) gives no verdict at all; the miner retries it.

Capacity verdicts. At most ``capacity_allowance(hotkey)`` precommits of a hotkey get one per
window (the groups its quota lets it hold, set by the wiring with ``allow_capacity_verdicts``). The ledger itself
never asks who caused a saturation: the RL issuer leaves out a ``sandbox_saturated`` or ``open_queue_full`` refusal
only when the refused miner's operator (all its hotkeys, by the registration's coldkey; by hotkey when the issuer
has no ``operator_of``) holds more sessions than its hotkeys' allocations AND held at least half of the sessions on
the machines that could take the open (beyond the open's own group; the total never below the machines' reported
load), or of a full queue's tickets (beyond its own group's) (``SessionIssuer._self_caused``). A gateway's signed
open refusal is always ours (a fault on the record). Graded sessions a capacity verdict made neutral count toward
graduation (``YieldStats.capacity_neutral``), never toward the yield, so a hotkey cannot stay a newcomer on them;
``capacity_counts`` gives each hotkey's capacity verdicts and neutral sessions since the process started.

Keys. Everything is per hotkey. ``stats(hotkey)`` reads the snapshot of the last flush: the
yield is useful / decided over the hotkey's last ``YIELD_SESSIONS`` decided sessions, ``useful_24h`` its useful
sessions decided in the last ``USEFUL_HORIZON_S``. ``snapshot()`` is every hotkey's stats (the quota shares the
fleet with it), replaced whole at each flush with ``version`` raised.

Restart window. ``note_accepting(window)`` journals the window accepting groups now (None between
windows); ``accepting_at_last_flush()`` reads it back after a restart.

Threads. The event loop only journals (``note_opened``, the ``SessionOutcomes`` callbacks, ``decide``) in
memory, under a threading lock. ``flush`` writes the journal in one SQLite transaction and refreshes the
per-hotkey snapshot: call it in a worker thread. One flush at a time (``_flush_lock``); ``close`` waits for the
flush in progress. ``stats`` reads the snapshot (no I/O). A failed flush keeps its journal. A restart reloads the
pending sessions and verdicts from the file; ``reconcile`` then rebuilds from the restored session book the
opens an unflushed journal lost (and an accepted or handed-back group: their ``submitted`` records are
durable)."""
from __future__ import annotations

import heapq
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT
from reliquary.sandbox.capacity_refusals import CAPACITY_REFUSALS
from reliquary.sandbox.rl_capacity import USEFUL_HORIZON_S, YIELD_SESSIONS, YieldStats

logger = logging.getLogger(__name__)

PENDING, USEFUL, WASTED, FAULT = "pending", "useful", "wasted", "fault"
CAPACITY, OURS, REFUSED, ACCEPTED, HANDED_BACK = "capacity", "ours", "refused", "accepted", "handed_back"
# A verdict replaces another only when it ranks higher: a real refusal is never cleared by "ours".
_RANK = {CAPACITY: 0, OURS: 1, ACCEPTED: 2, REFUSED: 3, HANDED_BACK: 4}
FAULT_STATES = frozenset({"aborted", "voided"})
FAULT_STATUSES = frozenset({"aborted", "machine_drained", "handed_back"})
RUNTIME_EXEC_FAILED = "runtime_exec_failed"
OPEN_REFUSED = "open_refused"
VALIDATOR_RESTARTED = "validator_restarted"
GRADING_POOL = "grading_pool"
CAPACITY_REFUSED = "capacity_refused"
# Refusals of the validator's own making AFTER the checker verified the group as the precommit's: only these
# reach the ledger through a claimed group (``EpisodeGroupIntake.settle``). Neutral unless a group of the same
# precommit is used later. ``content_identity``: our prompt content digest missing at the batcher;
# ``service_eligibility``: the service runtime not active when the batcher took the group.
VALIDATOR_STAGES = frozenset({"episode_persist_failed", "admission_timeout", "admission_worker", "prompt_source",
                              "content_identity", "service_eligibility"})
# Refusals reached BEFORE the check: the precommit they name is unverified and a miner can provoke
# them at will, so they give no verdict (the miner retries them; a restart's window is ``decide(resumed=)``).
# Also the retryable refusals that are not ours (the miner's own groups or checks in flight: it
# resends them) and a drain (``admission_drain``, at shutdown too). Pinned: NO_VERDICT_STAGES ⊇ RETRYABLE_STAGES -
# VALIDATOR_STAGES. ``episode_prompt_source``: the validator's own prompt source failed inside the checker, so the
# precommit is still unverified (never ours) and the miner resends it.
NO_VERDICT_STAGES = frozenset({
    "episode_window_resumed", "episode_checker_busy", "episode_directory", "episode_session_busy",
    "episode_policy_stale", "episode_unserved", "episode_group_in_flight", "episode_checks_in_flight",
    "episode_prompt_source", "admission_drain"})
GRADED = "graded"
WITHDRAWN = "withdrawn"           # the session book's closed status of a withdrawal (``sessions.WITHDRAWN``)
MACHINE_FINALS = frozenset({GRADED, "expired", "box_failed", "budget_exhausted", "aborted", "handed_back"})
"""Closed statuses of a machine-signed final: the session's box ran. Anything else (a lapse with None, an
``open_failed``, a withdrawal, a drain) may be a token never presented to a machine."""
SCHEMA_VERSION = 1
PRUNE_EVERY_S = 3600.0
KEEP_DECIDED_S = 2 * USEFUL_HORIZON_S

_SCHEMA = """
CREATE TABLE IF NOT EXISTS yield_sessions(session_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL,
    precommit TEXT NOT NULL, window INTEGER NOT NULL, env TEXT NOT NULL, split TEXT NOT NULL,
    idx INTEGER NOT NULL, opened_at REAL NOT NULL, outcome TEXT NOT NULL, fault TEXT, decided_at REAL,
    held_s REAL NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS yield_sessions_hotkey ON yield_sessions(hotkey, outcome, opened_at);
CREATE INDEX IF NOT EXISTS yield_sessions_pending ON yield_sessions(outcome, window);
CREATE TABLE IF NOT EXISTS yield_precommits(precommit TEXT PRIMARY KEY, hotkey TEXT NOT NULL,
    verdict TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS yield_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _rl_precommit(engagement: Any) -> tuple[int, str] | None:
    """(window, precommit sha256) of an RL engagement, or None (a corpus session never loads the RL module)."""
    if not isinstance(engagement, str) or not engagement.startswith("rl:"):
        return None
    from reliquary.protocol.service_episode import EpisodeWireError, parse_rl_engagement

    try:
        window, sha, _seed = parse_rl_engagement(engagement)
    except EpisodeWireError:
        return None
    return window, sha


class YieldLedgerUnusable(RuntimeError):
    """The ledger's file is corrupted or of another schema version. Never started afresh in its place: a fresh
    ledger would make every low-yield hotkey a newcomer again."""


def _open(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30, check_same_thread=False)
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        tables = db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
        if version == 0 and tables == 0:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        elif version != SCHEMA_VERSION:
            raise YieldLedgerUnusable(f"the yield ledger {path} has schema version {version}, this validator reads "
                                      f"version {SCHEMA_VERSION}; it was left as it is")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript(_SCHEMA)
        db.execute("PRAGMA quick_check").fetchall()
    except sqlite3.DatabaseError as error:
        db.close()
        raise YieldLedgerUnusable(f"the yield ledger {path} is not a readable SQLite file ({error}); it was left as "
                                  "it is: restore it or move it aside deliberately") from error
    except BaseException:
        db.close()
        raise
    return db


class SessionYieldLedger:
    def __init__(self, path: str | Path, *, capacity_allowance: Callable[[str], int] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._capacity_allowance = capacity_allowance or (lambda hotkey: GROUPS_IN_FLIGHT)
        self._clock = clock
        self._lock = threading.Lock()          # the journal and the in-memory state
        self._flush_lock = threading.Lock()    # one flush (or close) at a time
        self._journal: list[tuple] = []
        self._pending: dict[str, dict[str, str]] = {}               # precommit -> {session: hotkey}
        self._windows: dict[str, int] = {}
        self._tasks: dict[str, tuple[str, str, str, int]] = {}      # precommit -> (hotkey, env, split, index)
        self._verdicts: dict[str, tuple[str, str]] = {}             # precommit -> (verdict, hotkey)
        self._decided: set[str] = set()
        # Per hotkey, since this process started: capacity verdicts given and sessions they made neutral (the
        # saturation metrics read them: a hotkey living on capacity verdicts shows the fleet is too thin for it).
        self._capacity_counts: dict[str, list[int]] = {}
        self._decided_counts: dict[str, int] = {}     # useful or wasted sessions decided since start, per hotkey
        self._recent: dict[str, tuple[tuple[float, str, bool], ...]] = {}   # hotkey -> last decided
        self._useful_24h: dict[str, int] = {}
        self._snapshot: dict[str, YieldStats] = {}
        self.version = 0
        self._accepting: int | None = None
        self._pruned_at = 0.0
        self._closed = False
        self._db = _open(Path(path))
        try:
            self._load()
        except sqlite3.DatabaseError as error:
            self._db.close()
            raise YieldLedgerUnusable(f"the yield ledger {path} cannot be read ({error}); it was left as it is: "
                                      "restore it or move it aside deliberately") from error

    # --- the event loop's side: memory only ---
    def note_opened(self, record: Any) -> None:
        found = _rl_precommit(getattr(record, "engagement", None))
        if found is None:
            return
        window, sha = found
        hotkey = str(record.hotkey)
        task = (hotkey, str(record.env), str(record.split), int(record.index))
        with self._lock:
            if record.session_id in self._decided:
                return
            sessions = self._pending.setdefault(sha, {})
            if record.session_id in sessions:
                return
            sessions[record.session_id] = hotkey
            self._windows[sha] = window
            self._tasks.setdefault(sha, task)
            self._journal.append(("open", record.session_id, hotkey, sha, window, *task[1:],
                                  float(record.issued_at)))

    def group_settled(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                      accepted: bool, stage: str | None = None) -> None:
        if not accepted and stage in NO_VERDICT_STAGES:
            return                          # an unverified precommit, a refusal the miner retries: no verdict
        verdict = ACCEPTED if accepted else OURS if stage in VALIDATOR_STAGES else REFUSED
        self._verdict(precommit_sha256, hotkey, verdict)

    def capacity_refused(self, *, hotkey: str, precommit_sha256: str) -> None:
        """An open of this precommit was refused for our capacity (saturated, the queue full, never placeable,
        an unserved env). Nothing is checked here about who caused it: the issuer calls this only for a
        saturation or a full queue its own operator's load did not cause (``SessionIssuer._self_caused``);
        unless a group of it is used or really refused, its sessions that ran are neither when the group could
        not be formed (``decide``). At most ``capacity_allowance(hotkey)`` precommits per hotkey and window."""
        with self._lock:
            window = self._windows.get(precommit_sha256)
            current = self._verdicts.get(precommit_sha256)
            if window is not None and current is None:
                taken = sum(1 for sha, (verdict, owner) in self._verdicts.items()
                            if verdict == CAPACITY and owner == hotkey and self._windows.get(sha) == window)
                if taken >= int(self._capacity_allowance(str(hotkey))):
                    logger.info("yield: capacity verdicts of %s in window %d at their cap", str(hotkey)[:12], window)
                    return
        if self._verdict(precommit_sha256, hotkey, CAPACITY):
            with self._lock:
                self._capacity_counts.setdefault(str(hotkey), [0, 0])[0] += 1

    def capacity_counts(self, hotkey: str | None = None):
        """``(verdicts, neutral sessions)`` of a hotkey since this process started, or, without one, every
        hotkey's (a copy)."""
        with self._lock:
            if hotkey is not None:
                return tuple(self._capacity_counts.get(str(hotkey), (0, 0)))
            return {key: tuple(value) for key, value in self._capacity_counts.items()}

    def session_counts(self) -> dict[str, tuple[int, int]]:
        """``(neutral, decided)`` sessions per hotkey since this process started, over the same span: the graded
        sessions a capacity verdict made neutral, and the useful or wasted ones (a copy)."""
        with self._lock:
            hotkeys = set(self._capacity_counts) | set(self._decided_counts)
            return {key: (self._capacity_counts.get(key, (0, 0))[1], self._decided_counts.get(key, 0))
                    for key in hotkeys}

    def allow_capacity_verdicts(self, allowance: Callable[[str], int]) -> None:
        """The per-hotkey, per-window cap on capacity verdicts (the wiring passes ``YieldQuota.groups_allowed``)."""
        self._capacity_allowance = allowance

    def note_accepting(self, window: int | None, at: float) -> None:
        """The window accepting groups now (None between windows): journalled, written at the next flush
        (the restart marks a window resumed only when it was accepting at the last flush)."""
        with self._lock:
            self._journal.append(("accepting", None if window is None else int(window), float(at)))

    def accepting_at_last_flush(self) -> int | None:
        return self._accepting

    def group_handed_back(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                          stage: str) -> None:
        self._verdict(precommit_sha256, hotkey, HANDED_BACK)

    def _verdict(self, sha: str, hotkey: str, verdict: str) -> bool:
        """Whether the verdict was taken (it ranks above the precommit's current one)."""
        with self._lock:
            sessions = self._pending.get(sha)
            if not sessions:
                return False                # no pending session of it: nothing a verdict could decide
            if any(owner != hotkey for owner in sessions.values()):
                logger.warning("yield: a %s verdict of %s names a precommit of another hotkey; ignored",
                               verdict, str(hotkey)[:12])
                return False
            current = self._verdicts.get(sha)
            if current is not None and _RANK[current[0]] >= _RANK[verdict]:
                return False
            self._verdicts[sha] = (verdict, hotkey)
            self._journal.append(("verdict", sha, hotkey, verdict, float(self._clock())))
            return True

    def pending_windows(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted({self._windows[sha] for sha, ids in self._pending.items()
                                 if ids and sha in self._windows}))

    def precommits_of(self, window: int) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(sha for sha, ids in self._pending.items()
                                if ids and self._windows.get(sha) == int(window)))

    def task_of(self, sha: str) -> tuple[str, str, str, int] | None:
        with self._lock:
            return self._tasks.get(sha)

    def decide(self, precommit_sha256: str, sessions: Iterable[tuple], *, used: bool, aborted: bool,
               now: float, resumed: bool = False, group_size: int | None = None) -> int:
        """Decide every pending session of a settled precommit. ``sessions``: the book's
        ``(session_id, state, closed_status, fault)`` of its records (a session the book no longer holds is
        decided by the precommit's verdict alone). ``used``: a group of it was trained or paid in its
        window's settlement; ``aborted``: the runtime settled that window as aborted; ``resumed``: its window is
        the one a validator restart interrupted (the restarted validator takes no group of it); ``group_size``:
        M of its env (a capacity verdict counts only when fewer than M sessions were graded; None: never)."""
        rows = {}
        for row in sessions:
            sid, state, status, fault, held = (tuple(row) + (None, None))[:5]
            rows[str(sid)] = (state, status, fault, held)
        with self._lock:
            pending = self._pending.pop(precommit_sha256, {})
            self._windows.pop(precommit_sha256, None)
            self._tasks.pop(precommit_sha256, None)
            verdict = (self._verdicts.pop(precommit_sha256, None) or (None, None))[0]
            graded = sum(1 for sid in pending if rows.get(sid, (None, None, None, None))[1] == GRADED)
            unformed = group_size is not None and graded < int(group_size)
            for session_id, hotkey in pending.items():
                state, status, fault, held = rows.get(session_id, (None, None, None, None))
                if fault is None and state in FAULT_STATES:
                    fault = state
                if fault is None and status in FAULT_STATUSES:
                    fault = status
                if fault is None and aborted:
                    fault = "window_aborted"
                if fault is None and verdict == HANDED_BACK:
                    fault = HANDED_BACK
                if fault is None and not used and verdict == OURS:
                    fault = "validator_refused"
                if fault is None and not used and verdict == CAPACITY and unformed and status == GRADED:
                    fault = CAPACITY_REFUSED       # it ran, and the group could not be formed
                    self._capacity_counts.setdefault(hotkey, [0, 0])[1] += 1
                if fault is None and not used and resumed and verdict != REFUSED:
                    fault = "window_resumed"
                outcome = FAULT if fault else USEFUL if used else WASTED
                if outcome != FAULT:
                    self._decided_counts[hotkey] = self._decided_counts.get(hotkey, 0) + 1
                self._journal.append(("decide", session_id, hotkey, outcome, fault, float(now), float(held or 0.0)))
                self._decided.add(session_id)
            return len(pending)

    def reconcile(self, records: Iterable[Any]) -> None:
        """After a restart: every RL session of the restored book is known, and an accepted (or handed back)
        group read back from its durable ``submitted`` records."""
        records = list(records)
        for record in records:
            self.note_opened(record)
        for record in records:
            found = _rl_precommit(getattr(record, "engagement", None))
            if found is None or getattr(record, "state", None) != "submitted":
                continue
            verdict = HANDED_BACK if getattr(record, "closed_status", None) == "handed_back" else ACCEPTED
            self._verdict(found[1], str(record.hotkey), verdict)

    def stats(self, hotkey: str) -> YieldStats:
        """The hotkey's yield stats (the snapshot of the last flush; per hotkey)."""
        return self._snapshot.get(str(hotkey), YieldStats())

    def snapshot(self) -> dict[str, YieldStats]:
        """Every hotkey's stats as of the last flush; replaced whole (never mutated) with ``version`` raised."""
        return self._snapshot

    @staticmethod
    def _stats_of(rows: tuple[tuple[float, str, bool], ...], useful_24h: tuple[int, float]) -> YieldStats:
        last = heapq.nlargest(YIELD_SESSIONS, rows, key=lambda row: (row[0], row[1]))
        return YieldStats(useful=sum(1 for row in last if row[2]), decided=len(last), useful_24h=int(useful_24h[0]),
                          useful_s_24h=float(useful_24h[1]), capacity_neutral=int(useful_24h[2]))

    # --- the worker thread's side ---
    def flush(self, now: float | None = None) -> int:
        now = float(self._clock()) if now is None else float(now)
        with self._flush_lock:
            if self._closed:
                return 0
            with self._lock:
                journal, self._journal = self._journal, []
            try:
                with self._db:
                    for entry in journal:
                        self._apply(entry)
            except BaseException:
                with self._lock:
                    self._journal[:0] = journal
                raise
            hotkeys = set(self._recent) | {entry[2] for entry in journal if entry[0] in ("open", "decide")}
            recent, useful = {}, {}
            for hotkey in hotkeys:
                recent[hotkey], useful[hotkey] = self._read_hotkey(hotkey, now)
            snapshot = {hotkey: self._stats_of(rows, useful[hotkey]) for hotkey, rows in recent.items()
                        if rows or useful[hotkey][2]}
            # Replaced whole (never mutated in place): the loop's readers see the old or the new snapshot.
            self._recent, self._useful_24h, self._snapshot = recent, useful, snapshot
            self.version += 1
            for entry in journal:
                if entry[0] == "accepting":
                    self._accepting = entry[1]
            if now - self._pruned_at >= PRUNE_EVERY_S:
                self._prune(now)
            return len(journal)

    def _apply(self, entry: tuple) -> None:
        kind = entry[0]
        if kind == "open":
            _, sid, hotkey, sha, window, env, split, index, opened_at = entry
            self._db.execute("INSERT OR IGNORE INTO yield_sessions(session_id, hotkey, precommit, window, env, "
                             "split, idx, opened_at, outcome) VALUES(?,?,?,?,?,?,?,?,?)",
                             (sid, hotkey, sha, int(window), env, split, int(index), opened_at, PENDING))
        elif kind == "verdict":
            _, sha, hotkey, verdict, at = entry
            row = self._db.execute("SELECT verdict FROM yield_precommits WHERE precommit=?", (sha,)).fetchone()
            if row is None or _RANK.get(row[0], -1) < _RANK[verdict]:
                self._db.execute("INSERT OR REPLACE INTO yield_precommits VALUES(?,?,?,?)", (sha, hotkey, verdict, at))
        elif kind == "decide":
            _, sid, _hotkey, outcome, fault, at, held = entry
            self._db.execute("UPDATE yield_sessions SET outcome=?, fault=?, decided_at=?, held_s=? WHERE session_id=? "
                             "AND outcome=?", (outcome, fault, at, held, sid, PENDING))
        elif kind == "accepting":
            _, window, _at = entry
            self._db.execute("INSERT OR REPLACE INTO yield_meta VALUES('accepting', ?)",
                             ("" if window is None else str(window),))
        else:
            raise ValueError(f"unknown yield journal entry {kind!r}")

    def _read_hotkey(self, hotkey: str, now: float) -> tuple[tuple[tuple[float, str, bool], ...],
                                                             tuple[int, float, int]]:
        rows = self._db.execute(
            "SELECT opened_at, session_id, outcome FROM yield_sessions WHERE hotkey=? AND outcome IN (?, ?) "
            "ORDER BY opened_at DESC, session_id DESC LIMIT ?", (hotkey, USEFUL, WASTED, YIELD_SESSIONS)).fetchall()
        count, seconds = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(held_s), 0) FROM yield_sessions WHERE hotkey=? AND outcome=? "
            "AND decided_at>=?", (hotkey, USEFUL, now - USEFUL_HORIZON_S)).fetchone()
        (neutral,) = self._db.execute(
            "SELECT COUNT(*) FROM yield_sessions WHERE hotkey=? AND outcome=? AND fault=?",
            (hotkey, FAULT, CAPACITY_REFUSED)).fetchone()
        return (tuple((float(at), str(sid), outcome == USEFUL) for at, sid, outcome in rows),
                (int(count), float(seconds), int(neutral)))


    def _prune(self, now: float) -> None:
        """Forget decided sessions older than two horizons that are not among their hotkey's last
        ``YIELD_SESSIONS`` decided ones, and verdicts no pending session needs."""
        with self._db:
            self._db.execute(
                "DELETE FROM yield_sessions WHERE outcome<>? AND decided_at<? AND session_id NOT IN ("
                " SELECT s.session_id FROM yield_sessions s WHERE s.hotkey=yield_sessions.hotkey AND s.outcome IN (?, ?)"
                " ORDER BY s.opened_at DESC, s.session_id DESC LIMIT ?)",
                (PENDING, now - KEEP_DECIDED_S, USEFUL, WASTED, YIELD_SESSIONS))
            self._db.execute("DELETE FROM yield_precommits WHERE at<? AND precommit NOT IN ("
                             " SELECT precommit FROM yield_sessions WHERE outcome=?)", (now - KEEP_DECIDED_S, PENDING))
        self._pruned_at = now
        with self._lock:
            if len(self._decided) > 1_000_000:
                self._decided.clear()

    def _load(self) -> None:
        for sid, hotkey, sha, window, env, split, index in self._db.execute(
                "SELECT session_id, hotkey, precommit, window, env, split, idx FROM yield_sessions WHERE outcome=?",
                (PENDING,)):
            self._pending.setdefault(sha, {})[sid] = hotkey
            self._windows[sha] = int(window)
            self._tasks.setdefault(sha, (hotkey, env, split, int(index)))
        for sha, hotkey, verdict in self._db.execute("SELECT precommit, hotkey, verdict FROM yield_precommits"):
            if sha in self._pending:
                self._verdicts[sha] = (verdict, hotkey)
        now = float(self._clock())
        self._decided = {sid for (sid,) in self._db.execute(
            "SELECT session_id FROM yield_sessions WHERE outcome<>? AND decided_at>=?", (PENDING, now - KEEP_DECIDED_S))}
        for (hotkey,) in self._db.execute("SELECT DISTINCT hotkey FROM yield_sessions").fetchall():
            self._recent[hotkey], self._useful_24h[hotkey] = self._read_hotkey(hotkey, now)
        self._snapshot = {hotkey: self._stats_of(rows, self._useful_24h[hotkey])
                          for hotkey, rows in self._recent.items() if rows or self._useful_24h[hotkey][2]}
        row = self._db.execute("SELECT value FROM yield_meta WHERE key='accepting'").fetchone()
        self._accepting = int(row[0]) if row is not None and row[0] else None

    def close(self) -> None:
        with self._flush_lock:
            if not self._closed:
                self._closed = True
                self._db.close()


def held_seconds(record: Any, now: float, *, presented: bool | None = None) -> float:
    """How long the session's box held its place: issue to its FIRST close with the machine's final (``ended_at``;
    ``closed_at`` for a record without it; now while open), capped at the token's expiry: the env-neutral volume of
    a useful session. A submission delayed to the window's end adds nothing. Only a session whose box ran counts:
    one closed by a machine-signed final (``ended_at`` set, or a ``MACHINE_FINALS`` status: a graded session
    withdrawn later keeps the first, never the second), or one the caller knows a machine listed
    (``presented=True``); a token never presented counts nothing, never its time to expiry. The wiring passes it to
    ``decide`` as each row's fifth element."""
    ran = (getattr(record, "closed_status", None) in MACHINE_FINALS or getattr(record, "ended_at", None) is not None
           if presented is None else bool(presented))
    if not ran:
        return 0.0
    end = getattr(record, "ended_at", None) or getattr(record, "closed_at", None) or now
    expires = getattr(record, "expires_at", None)
    if expires is not None:
        end = min(float(end), float(expires))
    return max(0.0, float(end) - float(getattr(record, "issued_at", end)))


def judged_status(record: Any) -> str | None:
    """The closed status the ledger judges a session by: its machine's signed final (``ended_status``) when the
    miner withdrew it after that final (a withdrawal rewrites ``closed_status``), else its ``closed_status``. A graded
    session withdrawn because its group could not form stays graded for the capacity verdict; a withdrawal never
    makes a session graded that was not, and every other status (a hand-back, a drained machine) is kept."""
    status = getattr(record, "closed_status", None)
    ended = getattr(record, "ended_status", None)
    if status == WITHDRAWN and ended is not None:
        return str(ended)
    return status


def decision_row(record: Any, now: float, *, presented: bool | None = None) -> tuple:
    """``decide``'s row of a session: ``(session_id, state, status, fault, held seconds)``."""
    return (record.session_id, record.state, judged_status(record), getattr(record, "fault", None),
            held_seconds(record, now, presented=presented))


CAPACITY_REASONS = CAPACITY_REFUSALS
"""The issuer's own set (``capacity_refusals``): an unused precommit that met one is neither useful nor wasted (for
its sessions that ran, in a group that could not be formed). The issuer leaves out a saturation or a full queue of
the refused miner's own operator's making before it tells the ledger."""


class LedgerEvents:
    """The RL issuer's ``SessionEvents``: every open goes to the ledger (faults live on the records), and every
    capacity refusal of an open marks its precommit."""

    def __init__(self, ledger: SessionYieldLedger) -> None:
        self._ledger = ledger

    def opened(self, record: Any) -> None:
        self._ledger.note_opened(record)

    def closed(self, record: Any, final: Mapping[str, Any] | None) -> None:
        return None

    def refused(self, hotkey: str, engagement: Any, refusal: Any) -> None:
        if getattr(refusal, "reason", None) not in CAPACITY_REASONS or not isinstance(engagement, Mapping):
            return
        precommit = engagement.get("precommit")
        sha = precommit.get("precommit_sha256") if isinstance(precommit, Mapping) else None
        if isinstance(sha, str):
            self._ledger.capacity_refused(hotkey=str(hotkey), precommit_sha256=sha)


__all__ = [
    "ACCEPTED",
    "CAPACITY",
    "CAPACITY_REASONS",
    "FAULT",
    "FAULT_STATES",
    "FAULT_STATUSES",
    "GRADED",
    "GRADING_POOL",
    "HANDED_BACK",
    "MACHINE_FINALS",
    "NO_VERDICT_STAGES",
    "OPEN_REFUSED",
    "OURS",
    "PENDING",
    "REFUSED",
    "RUNTIME_EXEC_FAILED",
    "SCHEMA_VERSION",
    "USEFUL",
    "VALIDATOR_RESTARTED",
    "VALIDATOR_STAGES",
    "WASTED",
    "LedgerEvents",
    "SessionYieldLedger",
    "YieldLedgerUnusable",
    "held_seconds",
]
