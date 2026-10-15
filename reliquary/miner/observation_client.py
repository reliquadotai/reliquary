"""Reference reader of a validator's published run observations, and an overridable prompt policy.

The validator publishes a signed, paged index of immutable segments on a public bucket
(``reliquary.services.publication``). This module is the miner side of that format: it polls the
head, verifies everything with the publication module's own verifiers (never reimplemented here),
and keeps a local SQLite table of every prompt's observations. The validator enforces no
exclusion: each miner decides which prompts to work on, through a ``PromptPolicy``.

Client loop (``ObservationClient.sync``)
    1. At most one head fetch per ``HEAD_MAX_AGE`` seconds (the head is published with
       ``max-age=15``); after an error the wait doubles up to ``BACKOFF_MAX``.
    2. The head is verified with ``verify_index(..., expected_run_id, min_last_number=<last number
       accepted>)``: a stale head is refused, so the table never goes backwards.
    3. Closed pages are walked once (``verify_page_entry`` then ``verify_index``), cached in the
       local database (they are immutable) and only the ones past the last applied segment are
       read. The head's own segments follow.
    4. Each new segment is downloaded, checked with ``verify_segment`` and applied in order. A
       segment and its position (``last_number`` / ``last_seq``) commit in ONE transaction, so a
       restart resumes after the last applied segment and never counts an event twice.

Local table (``ObservationTable``): one row per observation (compact), one row per (env, prompt).
Rows older than ``retain_windows`` windows are folded into the prompt row (scanned flag, first
scan status) and deleted, so memory is constant and disk grows with the retained windows only.

Public event semantics (``publication`` docstring): the LAST ``settle`` event of an id gives its
status; an observation event is never superseded, only settled.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from reliquary.services import publication
from reliquary.services.publication import (
    index_key, verify_index, verify_page_entry, verify_segment,
)

logger = logging.getLogger(__name__)

HEAD_MAX_AGE = 15.0          # seconds; the head is published with ``Cache-Control: max-age=15``
BACKOFF_MAX = 300.0
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_RETAIN_WINDOWS = 2000
SUCCESS_BPS = 5000           # a seed "succeeded" when its mean reward is at least this
TABLE_VERSION = 1
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
UNIFORM = ("uniform-high", "uniform-low")


class ObservationVerificationError(ValueError):
    """Something the validator published does not verify (signature, digest, order, gap)."""


# ------------------------------------------------------------------------------------------ table

def _scans(lane: str, status: str, proof: str, aborted: bool = False) -> bool:
    """Whether an observation holds the first scan of its prompt as published NOW.

    A proven training observation is a scan whatever its pay, unless its last settle says its window
    aborted (``window_aborted``: that window trained nothing and gave its prompts back; an event
    without the field means false). An exploration observation is a scan while it is entitled
    (``exploration_pending`` / ``exploration_paid``) and not failed; unpaid, forfeited and unproven
    observations (``already_scanned``, ``not_robust``, ``unaudited`` ...) are not."""
    if lane == "training":
        return proof == "proven" and not aborted
    return lane == "exploration" and status in ("exploration_pending", "exploration_paid") and proof != "failed"


@dataclass(frozen=True)
class PromptSummary:
    n_observations: int            # observation events ever applied for the prompt
    scanned: bool
    first_scan_status: str | None  # published status of the observation holding the first scan
    last_window: int | None
    verdicts: dict                 # verdict -> count over the retained, non-failed observations
    best_in_zone: dict | None      # latest in-zone evidence: pool_sha256, seeds, rewards_bps, window


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class ObservationTable:
    """SQLite-backed table of the run's observations (``path=None``: in memory, for tests).

    Thread safe: the engine syncs on a worker thread and reads on the event loop thread, so the
    connection is shared (``check_same_thread=False``) and every public method holds one lock."""

    def __init__(self, path: str | os.PathLike | None = None):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(":memory:" if path is None else str(path), isolation_level=None,
                                  check_same_thread=False)
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(
                n INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, env TEXT NOT NULL,
                prompt_idx INTEGER NOT NULL, window INTEGER NOT NULL, checkpoint_n INTEGER NOT NULL,
                lane TEXT NOT NULL, verdict TEXT NOT NULL, status TEXT NOT NULL, proof TEXT NOT NULL,
                reason TEXT, pool TEXT, seeds TEXT, rewards TEXT NOT NULL, uncertain TEXT NOT NULL,
                window_aborted INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS observations_prompt ON observations(env, prompt_idx);
            CREATE TABLE IF NOT EXISTS prompts(
                env TEXT NOT NULL, prompt_idx INTEGER NOT NULL, n_obs INTEGER NOT NULL DEFAULT 0,
                last_window INTEGER, scanned INTEGER NOT NULL DEFAULT 0, first_scan_status TEXT,
                PRIMARY KEY(env, prompt_idx));
            CREATE TABLE IF NOT EXISTS pages(first_number INTEGER PRIMARY KEY, raw BLOB NOT NULL);
        """)
        if "window_aborted" not in {r[1] for r in self.db.execute("PRAGMA table_info(observations)")}:
            self.db.execute("ALTER TABLE observations ADD COLUMN window_aborted INTEGER NOT NULL DEFAULT 0")
        stamped = self.get_meta("table_version")
        if stamped is None:
            self.set_meta("table_version", str(TABLE_VERSION))
        elif stamped != str(TABLE_VERSION):
            raise ValueError(f"observation table version {stamped} is not {TABLE_VERSION}; use a new directory")
        self.version = 0  # bumped on every change: callers cache derived sets on it

    # --- meta
    @_locked
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    @_locked
    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))

    @property
    def last_number(self) -> int:
        return int(self.get_meta("last_number") or 0)

    @property
    def last_seq(self) -> int:
        return int(self.get_meta("last_seq") or 0)

    @_locked
    def close(self) -> None:
        self.db.close()

    # --- writes
    @_locked
    def apply(self, event: dict) -> None:
        """Apply one published event (own transaction). Observations are idempotent on their id."""
        self.db.execute("BEGIN")
        try:
            self._apply(event)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")
        self.version += 1

    @_locked
    def apply_segment(self, number: int, last_seq: int, events: Iterable[dict], *, order_sha256: str | None = None) -> None:
        """Apply a whole segment and advance ``last_number`` / ``last_seq`` in ONE transaction."""
        self.db.execute("BEGIN")
        try:
            for event in events:
                self._apply(event)
            self.set_meta("last_number", str(number))
            self.set_meta("last_seq", str(last_seq))
            if order_sha256 is not None:
                self.set_meta("order_sha256", order_sha256)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")
        self.version += 1

    def _apply(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "observation":
            try:
                env, prompt = str(event["env"]), int(event["prompt_idx"])
                candidate = event.get("candidate") or {}
                row = (str(event["id"]), env, prompt, int(event["window"]), int(event.get("checkpoint_n", 0)),
                       str(event["lane"]), str(event["verdict"]), str(event["status"]), str(event["proof"]),
                       event.get("reason"), candidate.get("pool_sha256"),
                       None if "seeds" not in candidate else json.dumps(candidate["seeds"]),
                       json.dumps(event["rewards_bps"]), json.dumps(event.get("uncertain", [])))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ObservationVerificationError(f"malformed observation event: {exc!r}") from exc
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO observations(id,env,prompt_idx,window,checkpoint_n,lane,verdict,status,proof,"
                "reason,pool,seeds,rewards,uncertain) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row).rowcount == 1
            if inserted:
                self.db.execute(
                    "INSERT INTO prompts(env,prompt_idx,n_obs,last_window) VALUES(?,?,1,?) ON CONFLICT(env,prompt_idx) "
                    "DO UPDATE SET n_obs=n_obs+1, last_window=MAX(COALESCE(last_window,0), excluded.last_window)",
                    (env, prompt, row[3]))
        elif kind == "settle":
            # The last settle of an id wins: events are applied in sequence order. An id compacted away
            # (or never seen) is ignored.
            try:
                values = (str(event["status"]), str(event["proof"]), event.get("reason"),
                          int(bool(event.get("window_aborted", False))), str(event["id"]))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ObservationVerificationError(f"malformed settle event: {exc!r}") from exc
            self.db.execute("UPDATE observations SET status=?, proof=?, reason=?, window_aborted=? WHERE id=?", values)

    @_locked
    def cache_page(self, first_number: int, raw: bytes) -> None:
        self.db.execute("INSERT OR REPLACE INTO pages(first_number, raw) VALUES(?,?)", (first_number, raw))

    @_locked
    def drop_page(self, first_number: int) -> None:
        self.db.execute("DELETE FROM pages WHERE first_number=?", (first_number,))

    @_locked
    def cached_page(self, first_number: int) -> bytes | None:
        row = self.db.execute("SELECT raw FROM pages WHERE first_number=?", (first_number,)).fetchone()
        return None if row is None else bytes(row[0])

    @_locked
    def compact(self, retain_windows: int) -> int:
        """Fold observations older than ``retain_windows`` windows (before the newest seen) into their
        prompt row and delete them. Returns the number of rows deleted."""
        newest = self.db.execute("SELECT MAX(window) FROM observations").fetchone()[0]
        if newest is None:
            return 0
        cutoff = newest - retain_windows
        old = self.db.execute("SELECT DISTINCT env, prompt_idx FROM observations WHERE window<?", (cutoff,)).fetchall()
        if not old:
            return 0
        self.db.execute("BEGIN")
        try:
            deleted = 0
            for env, prompt in old:
                rows = self.db.execute("SELECT window, lane, status, proof, window_aborted FROM observations WHERE env=? "
                                       "AND prompt_idx=? ORDER BY n", (env, prompt)).fetchall()
                first = next((r for r in rows if _scans(r[1], r[2], r[3], bool(r[4]))), None)
                held = self.db.execute("SELECT scanned, first_scan_status FROM prompts WHERE env=? AND prompt_idx=?",
                                       (env, prompt)).fetchone()
                scanned = bool(held[0]) or first is not None
                status = held[1] if held[0] else (first[2] if first else None)
                self.db.execute("UPDATE prompts SET scanned=?, first_scan_status=? WHERE env=? AND prompt_idx=?",
                                (int(scanned), status, env, prompt))
                deleted += self.db.execute("DELETE FROM observations WHERE env=? AND prompt_idx=? AND window<?",
                                           (env, prompt, cutoff)).rowcount
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")
        self.version += 1
        return deleted

    # --- reads
    @staticmethod
    def _record(r) -> dict:
        return {"id": r[0], "env": r[1], "prompt_idx": r[2], "window": r[3], "checkpoint_n": r[4], "lane": r[5],
                "verdict": r[6], "status": r[7], "proof": r[8], "reason": r[9], "pool_sha256": r[10],
                "seeds": None if r[11] is None else json.loads(r[11]), "rewards_bps": json.loads(r[12]),
                "uncertain": json.loads(r[13]), "window_aborted": bool(r[14])}

    _COLUMNS = "id,env,prompt_idx,window,checkpoint_n,lane,verdict,status,proof,reason,pool,seeds,rewards,uncertain,window_aborted"

    @_locked
    def records(self, env: str, prompt_idx: int) -> list[dict]:
        """The retained observations of a prompt in the order they were published, with their latest status."""
        return [self._record(r) for r in self.db.execute(
            f"SELECT {self._COLUMNS} FROM observations WHERE env=? AND prompt_idx=? ORDER BY n", (env, int(prompt_idx)))]

    @_locked
    def prompts(self, env: str) -> list[int]:
        return [r[0] for r in self.db.execute("SELECT prompt_idx FROM prompts WHERE env=? ORDER BY prompt_idx", (env,))]

    @_locked
    def scanned_prompts(self, env: str) -> set[int]:
        out = {r[0] for r in self.db.execute("SELECT prompt_idx FROM prompts WHERE env=? AND scanned=1", (env,))}
        for prompt, lane, status, proof, aborted in self.db.execute(
                "SELECT prompt_idx, lane, status, proof, window_aborted FROM observations WHERE env=?", (env,)):
            if _scans(lane, status, proof, bool(aborted)):
                out.add(prompt)
        return out

    @_locked
    def summary(self, env: str, prompt_idx: int) -> PromptSummary | None:
        held = self.db.execute("SELECT n_obs, last_window, scanned, first_scan_status FROM prompts WHERE env=? AND "
                               "prompt_idx=?", (env, int(prompt_idx))).fetchone()
        if held is None:
            return None
        records = self.records(env, prompt_idx)
        first = next((r for r in records if _scans(r["lane"], r["status"], r["proof"], r["window_aborted"])), None)
        scanned = bool(held[2]) or first is not None
        status = held[3] if held[2] else (first["status"] if first else None)
        usable = [r for r in records if r["proof"] != "failed"]
        verdicts: dict = {}
        for r in usable:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        zone = [r for r in usable if r["verdict"] == "in-zone"]
        best = None
        if zone:
            r = max(zone, key=lambda x: (x["window"], x["checkpoint_n"]))
            best = {"pool_sha256": r["pool_sha256"], "seeds": r["seeds"], "rewards_bps": r["rewards_bps"],
                    "window": r["window"]}
        return PromptSummary(held[0], scanned, status, held[1], verdicts, best)

    @_locked
    def seed_rewards(self, env: str, prompt_idx: int, *, pool_sha256: str | None = None) -> dict[int, list[int]]:
        """Rewards (bps) per pool seed index over the retained non-failed observations; ``pool_sha256``
        restricts it to one pool. Uncertain positions (an unboxed 0) are left out."""
        out: dict[int, list[int]] = {}
        for r in self.records(env, prompt_idx):
            if r["proof"] == "failed" or r["seeds"] is None or (pool_sha256 and r["pool_sha256"] != pool_sha256):
                continue
            skip = set(r["uncertain"])
            for i, (seed, reward) in enumerate(zip(r["seeds"], r["rewards_bps"])):
                if i not in skip:
                    out.setdefault(int(seed), []).append(int(reward))
        return out


# ------------------------------------------------------------------------------------------ policy

class PromptPolicy(Protocol):
    """What a miner may replace. Only ``skip`` is required; the engine also uses ``prefers_unscanned``
    and ``preferred_seeds`` when the policy has them."""

    def skip(self, table: ObservationTable, env: str, prompt_idx: int, *, checkpoint_n: int) -> bool: ...


class DefaultPromptPolicy:
    """A small, documented, deterministic reference policy (not an optimal one).

    * ``skip``: a prompt is skipped when every non-failed observation of it was uniform (16/16 or
      0/16) and the latest one is (a) 16/16 at high confidence (proven or audited, or seen twice) --
      ``skip_all_pass`` -- or (b) 0/16 and fewer than ``retry_all_fail_after_checkpoints``
      checkpoints ago. Any in-zone, intermediate or below-threshold observation un-skips it.
    * ``prefers_unscanned``: a first scan pays an exploration entitlement (15 % of a training group)
      that nobody can get again, so a fraction ``unscanned_share`` of the rounds restricts the choice
      to never-scanned prompts (the engine falls back to the full set if none is left). The round
      is drawn from ``seed`` only.
    * ``preferred_seeds``: draws are deterministic per (pool, seed, position), so observations of the
      SAME pool tell exactly which seeds succeed. When both a success and a failure are known, the
      group takes about half of each (a mixed group is in zone), then unseen seeds. Without evidence
      of the same pool it returns None (the engine keeps its default: the first M seeds).
    """

    def __init__(self, *, skip_all_pass: bool = True, retry_all_fail_after_checkpoints: int = 3,
                 unscanned_share: float = 0.85):
        if not 0.0 <= unscanned_share <= 1.0:
            raise ValueError("unscanned_share must be in [0, 1]")
        self.skip_all_pass = skip_all_pass
        self.retry = retry_all_fail_after_checkpoints
        self.unscanned_share = unscanned_share

    def skip(self, table: ObservationTable, env: str, prompt_idx: int, *, checkpoint_n: int) -> bool:
        usable = [r for r in table.records(env, prompt_idx) if r["proof"] != "failed"]
        if not usable or any(r["verdict"] not in UNIFORM for r in usable):
            return False
        last = max(usable, key=lambda r: (r["checkpoint_n"], r["window"]))
        if last["verdict"] == "uniform-high":
            highs = [r for r in usable if r["verdict"] == "uniform-high"]
            confident = any(r["proof"] in ("proven", "audited") and not r["uncertain"] for r in highs) or len(highs) >= 2
            return self.skip_all_pass and confident
        return checkpoint_n - last["checkpoint_n"] < self.retry

    def prefers_unscanned(self, table: ObservationTable, env: str, *, seed: int) -> bool:
        digest = hashlib.sha256(f"unscanned:{seed}:{env}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.unscanned_share

    def preferred_seeds(self, table: ObservationTable, env: str, prompt_idx: int, *, pool_sha256: str,
                        group_size: int, pool_seeds: int) -> tuple[int, ...] | None:
        evidence = table.seed_rewards(env, prompt_idx, pool_sha256=pool_sha256)
        good, bad = [], []
        for seed in sorted(evidence):
            if not 0 <= seed < pool_seeds:
                continue
            (good if sum(evidence[seed]) / len(evidence[seed]) >= SUCCESS_BPS else bad).append(seed)
        if not good or not bad:
            return None
        half = group_size // 2
        chosen = good[:half] + bad[:group_size - half]
        chosen += [s for s in range(pool_seeds) if s not in evidence][:group_size - len(chosen)]
        chosen += [s for s in good + bad if s not in chosen][:group_size - len(chosen)]
        if len(chosen) != group_size:
            return None
        return tuple(sorted(chosen))


def load_policy(spec: str | None) -> PromptPolicy:
    """``None`` -> the default policy; ``"module:attr"`` -> the class (instantiated) or the object."""
    if not spec:
        return DefaultPromptPolicy()
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError("a prompt policy is named module:attr")
    target = getattr(importlib.import_module(module), attr)
    return target() if isinstance(target, type) else target


# ------------------------------------------------------------------------------------------ client

def _check_url(url: str) -> None:
    """Only https (http on localhost for tests): no file:, ftp: or any other scheme, redirects included."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1"):
        return
    raise ValueError("an observation source is read over https (http only on localhost)")


def _opener():
    import urllib.request

    class SameSchemes(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _check_url(urllib.parse.urljoin(req.full_url, newurl))
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(SameSchemes)


def _http_get(url: str) -> bytes:
    _check_url(url)
    with _opener().open(url, timeout=30) as response:  # noqa: S310 - scheme checked above
        body = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ObservationVerificationError("published object is larger than the download limit")
    return body


def default_directory() -> Path:
    return Path(os.environ.get("RELIQUARY_OBSERVATIONS_DIR") or "~/.reliquary/observations").expanduser()


class ObservationClient:
    def __init__(self, base_url: str, run_id: str, validator_hotkey: str, *, fetch: Callable[[str], bytes] | None = None,
                 directory: str | os.PathLike | None = None, table: ObservationTable | None = None,
                 clock: Callable[[], float] = time.monotonic, retain_windows: int = DEFAULT_RETAIN_WINDOWS):
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run id must be a plain identifier")
        _check_url(base_url)
        self.base = base_url.rstrip("/") + "/"
        self.run_id, self.validator_hotkey = run_id, validator_hotkey
        self.fetch = fetch or _http_get
        self.clock, self.retain_windows = clock, retain_windows
        if table is None:
            directory = Path(directory) if directory is not None else default_directory()
            directory.mkdir(parents=True, exist_ok=True)
            table = ObservationTable(directory / f"run-{run_id}.sqlite3")
        self.table = table
        for key, value in (("run_id", run_id), ("validator", validator_hotkey)):
            held = table.get_meta(key)
            if held is None:
                table.set_meta(key, value)
            elif held != value:
                raise ValueError(f"this observation table belongs to another {key}; use another directory")
        self._failures = 0
        self._next_poll = 0.0
        self._skip_cache: tuple = (-1, {})

    # --- loop
    def sync(self, *, force: bool = False) -> int:
        """Poll the head once if due and apply every new segment. Returns the segments applied (0 when
        not due). Raises (and backs off) when the network fails or something does not verify; segments
        applied before the failure stay applied."""
        now = self.clock()
        if not force and now < self._next_poll:
            return 0
        try:
            try:
                applied = self._sync_once()
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                if isinstance(exc, ObservationVerificationError):
                    raise
                raise ObservationVerificationError(f"malformed published document: {exc!r}") from exc
        except Exception:
            self._failures += 1
            self._next_poll = now + min(BACKOFF_MAX, HEAD_MAX_AGE * 2 ** self._failures)
            raise
        self._failures = 0
        self._next_poll = now + HEAD_MAX_AGE
        return applied

    def _get(self, key: str) -> bytes:
        return self.fetch(self.base + key)

    def _verify(self, raw: bytes, **kw) -> dict:
        try:
            return verify_index(raw, self.validator_hotkey, expected_run_id=self.run_id, **kw)
        except ValueError as exc:
            raise ObservationVerificationError(str(exc)) from exc

    def _sync_once(self) -> int:
        table = self.table
        last = table.last_number
        head = self._verify(self._get(index_key(self.run_id)), min_last_number=last)
        if head["kind"] != "head":
            raise ObservationVerificationError("the published index is not a head")
        pinned = table.get_meta("order_sha256")
        if pinned is not None and head["order_sha256"] != pinned:
            raise ObservationVerificationError("the published order changed")
        if head["last_number"] == last:
            return 0
        entries: list[dict] = []
        for page in head["pages"]:
            if page["last_number"] <= last:
                continue
            body = None
            cached = table.cached_page(page["first_number"])
            if cached is not None:
                try:
                    verify_page_entry(cached, page)
                    body = self._verify(cached)
                except ValueError:
                    logger.warning("cached index page %d failed verification; fetching it again", page["first_number"])
                    table.drop_page(page["first_number"])
            if body is None:
                raw = self._get(page["key"])
                try:
                    verify_page_entry(raw, page)
                    body = self._verify(raw)
                except ValueError as exc:
                    raise ObservationVerificationError(str(exc)) from exc
                table.cache_page(page["first_number"], raw)
            if (body["kind"] != "page" or body["first_number"] != page["first_number"]
                    or body["last_number"] != page["last_number"]):
                raise ObservationVerificationError("a page is not the one the head lists")
            entries += [e for e in body["segments"] if e["number"] > last]
        entries += [e for e in head["segments"] if e["number"] > last]
        applied = 0
        for entry in entries:
            if entry["number"] != table.last_number + 1:
                raise ObservationVerificationError("the published segments leave a gap after the last applied one")
            if entry["first_seq"] != table.last_seq + 1:   # also the first segment: the log starts at seq 1
                raise ObservationVerificationError("segment sequence does not continue the applied log")
            if entry["size"] > MAX_DOWNLOAD_BYTES:
                raise ObservationVerificationError("a published segment is larger than the download limit")
            raw = self._get(entry["key"])
            try:
                events = verify_segment(raw, entry)
            except ValueError as exc:
                raise ObservationVerificationError(str(exc)) from exc
            table.apply_segment(entry["number"], entry["last_seq"], events, order_sha256=head["order_sha256"])
            applied += 1
        if applied:
            table.compact(self.retain_windows)
        return applied

    # --- reads for the engine
    def skipped(self, env: str, *, policy: PromptPolicy, checkpoint_n: int) -> set[int]:
        if self._skip_cache[0] != self.table.version:
            self._skip_cache = (self.table.version, {})
        key = (env, checkpoint_n, id(policy))
        if key not in self._skip_cache[1]:
            self._skip_cache[1][key] = {p for p in self.table.prompts(env)
                                        if policy.skip(self.table, env, p, checkpoint_n=checkpoint_n)}
        return set(self._skip_cache[1][key])

    def scanned(self, env: str) -> set[int]:
        return self.table.scanned_prompts(env)
