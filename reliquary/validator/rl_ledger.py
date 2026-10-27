"""The async RL payment ledger (rl-async-design.md §4.5, §4.9; implementation map §1.11, §1.13, P5).

Input: the submission path's event stream (``RlSubmissions.events(since)``, in ``seq`` order, cursor persisted).

* ``trained``: pay FLAT per matched token: every rollout whose match is at least ``floor`` (0.90) earns its matched
  tokens x the env's price per token at ingestion (``AsyncPriceController.token_price``). A rollout under the floor
  is never paid (the trainer masks it; the ledger re-checks). Matched tokens are the trainer's exact per-rollout
  counts when it reports them (``report_training(matched_tokens=...)``), else ``floor(match x completion tokens)``.
* ``first_scan`` (out of zone, a never-scanned prompt; emitted at once when the group was not drawn for audit, at
  a passed audit when drawn): ``fraction`` (0.25) of AN IN-ZONE GROUP'S PAY, defined as the env's price per token
  at ingestion x the mean matched tokens of the env's last ``reference_groups`` (200) trained groups in this
  ledger (``default_group_tokens`` before any). The out-of-zone group's own length is never used: a 0/16 group is
  cheap to make long. ``first_scan_forfeited`` (failed or inconclusive audit) carries no money.
* ``ban``: every unsettled accrual of the hotkey is forfeited, and so is any payable event dated before the ban's
  end (``until``). Settled periods are not clawed back.

Settlement (epochs = the drand periods of ``corpus_periods``, 72 min = one Bittensor epoch): a period closes
``slack_s`` after its end; its accruals are summed per hotkey and normalised to the task cap (``scale = min(1, cap
/ total)``: the cap is the most a period pays, the unpaid rest burns). With ``env_pay_k`` set (registry
``rl_env_pay_k``, default off), each env's accruals are first scaled down to at most ``k x its target share x cap``
(the surplus burns): no env can take the period by being cheap to mine. With ``carry_periods`` > 0 the part of a
period over the cap is not burned at once but carried, per hotkey, into the next periods (``rl_pay_carry``), where
it is paid before the cap like a fresh accrual; a carry older than ``carry_periods`` periods burns. The trainer pays
in lumps (a step's groups are all reported at its end, and a step outlasts a period): without the carry a period
holding a step end is cut to the cap while the periods between steps pay nothing, and the gap burns
(teutonic-rl-async-v1, 2026-10-24: 67 % of the cap burned). The result is a period archive
(``rewards_by_hotkey``, ``work_period``, ``entry_period`` from ``cp.entry_for``) of the same shape as a corpus
period archive, so the unchanged weight setter (``weight_only._period_weights``: EMA decayed per period, capped
per task) pays it once the task is declared with ``params.settlement = "period-ema-v1"``. Archives are kept in the
ledger (``rl_pay_periods``) until ``publish`` writes them (``R2PeriodArchives``); an accrual that lands after its
period settled moves to the next open period. Nothing here talks to the chain.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from reliquary.constants import RL_ROLLOUT_MATCH_FLOOR
from reliquary.validator import corpus_periods as cp
from reliquary.validator.rl_submit import BAN, FIRST_SCAN, TRAINED

RL_PERIOD_ARCHIVE_SCHEMA = "reliquary/rl-async-period/v1"
BOARD_CACHE_S = 15.0       # the public board is rebuilt at most this often (it sums every accrual under the lock)
ACCRUED, SETTLED, FORFEITED = "accrued", "settled", "forfeited"
# Legacy 0 is unconfirmed and must keep its identity; 1 is confirmed.
# Only new state 2 proves no publication has been attempted.
_UNATTEMPTED = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_pay_cursor(id INTEGER PRIMARY KEY CHECK(id=0), seq INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS rl_accruals(seq INTEGER PRIMARY KEY, lease_id TEXT NOT NULL, hotkey TEXT NOT NULL,
    environment TEXT NOT NULL, kind TEXT NOT NULL, at REAL NOT NULL, period INTEGER NOT NULL,
    matched_tokens INTEGER NOT NULL, token_price REAL NOT NULL, amount REAL NOT NULL, status TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS rl_accruals_open ON rl_accruals(status, period);
CREATE INDEX IF NOT EXISTS rl_accruals_env ON rl_accruals(environment, kind, seq);
CREATE TABLE IF NOT EXISTS rl_pay_bans(hotkey TEXT PRIMARY KEY, until REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rl_pay_carry(hotkey TEXT NOT NULL, origin_period INTEGER NOT NULL, amount REAL NOT NULL,
    PRIMARY KEY(hotkey, origin_period));
CREATE TABLE IF NOT EXISTS rl_pay_periods(work_period INTEGER PRIMARY KEY, entry_period INTEGER NOT NULL,
    doc TEXT NOT NULL, published INTEGER NOT NULL);
"""


@dataclass(frozen=True)
class LedgerConfig:
    task_id: str
    cap: float                       # the task's emission cap: what one period pays at most
    floor: float = RL_ROLLOUT_MATCH_FLOOR  # per-rollout match floor (design §4.6)
    reference_groups: int = 200
    default_group_tokens: int = 16384
    slack_s: float = 120.0
    genesis: float = cp.PERIOD_EPOCH
    env_pay_k: float | None = None   # an env's period pay <= k x its target share x cap (None: off)
    carry_periods: int = 0           # a period's pay over the cap is carried this many periods before it burns
    carry_max_caps: float = 2.0      # the most carried at once, in caps (the oldest burns first; simulated: an
                                     # uncapped carry stalled at 2-3 caps after an outage and kept paying over)

    def __post_init__(self) -> None:
        if not (0 < self.cap <= 1 and 0 <= self.floor <= 1 and self.reference_groups > 0
                and self.default_group_tokens > 0 and self.slack_s >= 0
                and (self.env_pay_k is None or self.env_pay_k > 0) and self.carry_periods >= 0
                and self.carry_max_caps >= 0):
            raise ValueError("invalid ledger config")


def paid_tokens(detail: dict, floor: float) -> int:
    """Matched tokens of a trained group's rollouts at or above the floor."""
    match = detail["rollout_match"]
    counts = detail.get("matched_tokens")
    if counts is None:
        counts = [math.floor(m * n) for m, n in zip(match, detail["completion_tokens"])]
    return sum(int(c) for m, c in zip(match, counts) if m >= floor)


class RlPayLedger:
    def __init__(self, book, submissions, prices, config: LedgerConfig,
                 clock: Callable[[], float] = time.time) -> None:
        """``prices.token_price(env)`` prices a matched token (``AsyncPriceController``)."""
        self.book, self.submissions, self.prices, self.config, self.clock = book, submissions, prices, config, clock
        self._board: tuple[float, dict] | None = None
        with book.lock:
            book.db.executescript(_SCHEMA)
            book.db.execute("INSERT OR IGNORE INTO rl_pay_cursor(id, seq) VALUES(0, 0)")
            book.db.commit()

    # --- accrual -------------------------------------------------------------------------------------------
    def _open_period(self, at: float) -> int:
        last = self.book.db.execute("SELECT MAX(work_period) FROM rl_pay_periods").fetchone()[0]
        period = cp.period_of(at, self.config.genesis)
        return period if last is None else max(period, int(last) + 1)

    def _reference_tokens(self, environment: str) -> float:
        rows = self.book.db.execute("SELECT matched_tokens FROM rl_accruals WHERE environment=? AND kind=? "
                                    "ORDER BY seq DESC LIMIT ?",
                                    (environment, TRAINED, self.config.reference_groups)).fetchall()
        return sum(r[0] for r in rows) / len(rows) if rows else float(self.config.default_group_tokens)

    def _banned(self, hotkey: str, at: float) -> bool:
        row = self.book.db.execute("SELECT until FROM rl_pay_bans WHERE hotkey=?", (hotkey,)).fetchone()
        return row is not None and at < row[0]

    def ingest(self, limit: int = 10_000) -> int:
        """Consume the events after the cursor; returns how many were read. The cursor is read in the transaction
        that moves it: two concurrent ingests never accrue one event twice."""
        with self.book.transaction():
            db = self.book.db
            cursor = db.execute("SELECT seq FROM rl_pay_cursor WHERE id=0").fetchone()[0]
            events = self.submissions.events(cursor, limit=limit)
            if not events:
                return 0
            for event in events:
                kind, hotkey, at, detail = event["kind"], event["hotkey"], float(event["at"]), event["detail"]
                if kind == BAN:
                    db.execute("INSERT INTO rl_pay_bans(hotkey, until) VALUES(?,?) ON CONFLICT(hotkey) "
                               "DO UPDATE SET until=MAX(until, excluded.until)", (hotkey, float(detail["until"])))
                    db.execute("UPDATE rl_accruals SET status=? WHERE hotkey=? AND status=?",
                               (FORFEITED, hotkey, ACCRUED))
                    continue
                if kind == TRAINED:
                    environment = detail["environment"]
                    tokens = paid_tokens(detail, self.config.floor)
                    price = float(self.prices.token_price(environment))
                    amount = tokens * price
                elif kind == FIRST_SCAN:
                    row = db.execute("SELECT environment FROM rl_groups WHERE lease_id=?",
                                     (event["lease_id"],)).fetchone()
                    environment = row[0]
                    tokens = 0
                    price = float(self.prices.token_price(environment))
                    amount = float(detail["fraction"]) * price * self._reference_tokens(environment)
                else:
                    continue                      # untrained, first_scan_forfeited: no money
                status = FORFEITED if self._banned(hotkey, at) else ACCRUED
                db.execute("INSERT INTO rl_accruals(seq, lease_id, hotkey, environment, kind, at, period, "
                           "matched_tokens, token_price, amount, status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (event["seq"], event["lease_id"], hotkey, environment, kind, at, self._open_period(at),
                            tokens, price, amount, status))
            db.execute("UPDATE rl_pay_cursor SET seq=? WHERE id=0", (events[-1]["seq"],))
        return len(events)

    # --- settlement ----------------------------------------------------------------------------------------
    def settle(self, now: float | None = None) -> list[dict]:
        """Archive every closed period holding accruals, oldest first (each once); returns the new archives."""
        now = float(self.clock()) if now is None else now
        genesis = self.config.genesis
        closed = cp.closed_through(now=now, oldest_pending=None, genesis=genesis, slack=self.config.slack_s)
        out = []
        with self.book.transaction():
            db = self.book.db
            for period in self._due_periods(closed):
                rows = db.execute("SELECT hotkey, amount, matched_tokens, environment FROM rl_accruals WHERE "
                                  "status=? AND period=? ORDER BY seq", (ACCRUED, period)).fetchall()
                env_scale = self._env_scale(rows)
                accrued: dict[str, float] = {}
                for hotkey, amount, _, env in rows:
                    accrued[hotkey] = accrued.get(hotkey, 0.0) + amount * env_scale.get(env, 1.0)
                fresh = math.fsum(accrued.values())
                carried = self._take_carry(period)          # hotkey -> [(origin period, amount)], expired burned
                if not rows and not carried:
                    continue
                for hotkey, parts in carried.items():
                    accrued[hotkey] = accrued.get(hotkey, 0.0) + math.fsum(a for _, a in parts)
                total = math.fsum(accrued.values())
                scale = 1.0 if total <= self.config.cap else self.config.cap / total
                carried_out = self._carry(period, rows, env_scale, carried, 1.0 - scale) if scale < 1 else 0.0
                entered = [e for e, in db.execute("SELECT entry_period FROM rl_pay_periods").fetchall()]
                entry = cp.entry_for(cp.period_of(now, genesis) + 1, entered)
                doc = {"schema": RL_PERIOD_ARCHIVE_SCHEMA, "task_id": self.config.task_id, "mechanism": "rl-async",
                       "work_period": period, "entry_period": entry,
                       "rewards_by_hotkey": {h: accrued[h] * scale for h in sorted(accrued) if accrued[h] > 0},
                       "accrued_by_hotkey": {h: accrued[h] for h in sorted(accrued)}, "accrued_total": total,
                       "scale": scale, "cap": self.config.cap, "groups": len(rows),
                       "matched_tokens": sum(r[2] for r in rows)}
                if self.config.carry_periods:
                    doc.update({"accrued_fresh": fresh, "carried_in": total - fresh, "carried_out": carried_out,
                                "carry_periods": self.config.carry_periods})
                if self.config.env_pay_k is not None:
                    doc["env_scale"] = env_scale
                db.execute("INSERT INTO rl_pay_periods(work_period, entry_period, doc, published) VALUES(?,?,?,?)",
                           (period, entry, json.dumps(doc, sort_keys=True), _UNATTEMPTED))
                db.execute("UPDATE rl_accruals SET status=? WHERE status=? AND period=?", (SETTLED, ACCRUED, period))
                out.append(doc)
        return out

    def _due_periods(self, closed: int) -> list[int]:
        """Closed periods to archive, oldest first: those holding accruals, and, while a carry is pending, every
        period after the last archive (a period with no step end still pays the carry)."""
        db = self.book.db
        due = {p for p, in db.execute("SELECT DISTINCT period FROM rl_accruals WHERE status=? AND period<=?",
                                      (ACCRUED, closed)).fetchall()}
        if self.config.carry_periods and db.execute("SELECT 1 FROM rl_pay_carry LIMIT 1").fetchone():
            last = db.execute("SELECT MAX(work_period) FROM rl_pay_periods").fetchone()[0]
            starts = list(due) + ([int(last) + 1] if last is not None else [])
            due.update(range(min(starts) if starts else closed, closed + 1))
        return sorted(due)

    def _take_carry(self, period: int) -> dict[str, list[tuple[int, float]]]:
        """The carry pending before ``period`` (removed: what is not paid is carried again); older than
        ``carry_periods`` burns."""
        db = self.book.db
        out: dict[str, list[tuple[int, float]]] = {}
        for hotkey, origin, amount in db.execute("SELECT hotkey, origin_period, amount FROM rl_pay_carry WHERE "
                                                 "origin_period<? ORDER BY origin_period", (period,)).fetchall():
            if period - origin <= self.config.carry_periods and not self._banned(hotkey, float(self.clock())):
                out.setdefault(hotkey, []).append((int(origin), float(amount)))
        db.execute("DELETE FROM rl_pay_carry WHERE origin_period<?", (period,))
        return out

    def _carry(self, period: int, rows, env_scale, carried, unpaid: float) -> float:
        """Carry ``unpaid`` (the share over the cap) of each hotkey's fresh accrual (origin ``period``) and of
        what it had carried in (its origin kept); returns the amount carried."""
        if not self.config.carry_periods:
            return 0.0
        parts: dict[tuple[str, int], float] = {}
        for hotkey, amount, _, env in rows:
            key = (hotkey, period)
            parts[key] = parts.get(key, 0.0) + amount * env_scale.get(env, 1.0) * unpaid
        for hotkey, items in carried.items():
            for origin, amount in items:
                parts[(hotkey, origin)] = parts.get((hotkey, origin), 0.0) + amount * unpaid
        excess = math.fsum(parts.values()) - self.config.carry_max_caps * self.config.cap
        for origin in sorted({o for _, o in parts}):            # over the most: the oldest burns first
            if excess <= 0:
                break
            mine = [key for key in parts if key[1] == origin]
            held = math.fsum(parts[key] for key in mine)
            cut = min(excess, held)
            for key in mine:
                parts[key] *= 1.0 - cut / held if held > 0 else 0.0
            excess -= cut
        self.book.db.executemany(
            "INSERT INTO rl_pay_carry(hotkey, origin_period, amount) VALUES(?,?,?) ON CONFLICT(hotkey, origin_period) "
            "DO UPDATE SET amount=amount+excluded.amount",
            [(h, o, a) for (h, o), a in sorted(parts.items()) if a > 0])
        return math.fsum(parts.values())

    def _env_scale(self, rows) -> dict[str, float]:
        """Per env of a period's accrual rows: the factor holding its pay to ``env_pay_k x share x cap``."""
        k = self.config.env_pay_k
        if k is None:
            return {}
        targets = getattr(self.prices, "targets", {}) or {}
        totals: dict[str, float] = {}
        for _, amount, _, env in rows:
            totals[env] = totals.get(env, 0.0) + amount
        out = {}
        for env, total in sorted(totals.items()):
            limit = k * float(targets.get(env, 0.0)) * self.config.cap
            out[env] = 1.0 if total <= limit else (limit / total if total > 0 else 1.0)
        return out

    def unpublished(self) -> list[dict]:
        with self.book.lock:
            return [json.loads(d) for d, in self.book.db.execute(
                "SELECT doc FROM rl_pay_periods WHERE published<>1 ORDER BY work_period").fetchall()]

    async def publish(self, archives, now: float | None = None) -> int:
        """Reconcile unconfirmed archives; only never-attempted entries may move."""
        written = 0
        for doc in self.unpublished():
            existing = await archives.read(self.config.task_id, doc["work_period"], doc["entry_period"])
            if existing is not None and existing != doc:
                raise RuntimeError("period archive already holds another document")
            if existing is None:
                now_ = float(self.clock()) if now is None else now
                due = cp.period_of(now_, self.config.genesis) + 1
                with self.book.transaction():
                    state, stored = self.book.db.execute(
                        "SELECT published, doc FROM rl_pay_periods WHERE work_period=?",
                        (doc["work_period"],)).fetchone()
                    if state == 1:
                        continue
                    doc = json.loads(stored)
                    if state == _UNATTEMPTED and doc["entry_period"] < due:
                        entered = [e for e, in self.book.db.execute(
                            "SELECT entry_period FROM rl_pay_periods WHERE work_period<>?", (doc["work_period"],))]
                        doc = {**doc, "entry_period": cp.entry_for(due, entered)}
                        self.book.db.execute("UPDATE rl_pay_periods SET entry_period=?, doc=? WHERE work_period=?",
                                             (doc["entry_period"], json.dumps(doc, sort_keys=True), doc["work_period"]))
                    # Commit the identity before I/O; an interrupted PUT can finish later.
                    self.book.db.execute("UPDATE rl_pay_periods SET published=0 WHERE work_period=?",
                                         (doc["work_period"],))
                await archives.write(self.config.task_id, doc["work_period"], doc["entry_period"], doc)
            with self.book.transaction():
                self.book.db.execute("UPDATE rl_pay_periods SET published=1 WHERE work_period=?",
                                     (doc["work_period"],))
            written += 1
        return written

    # --- the board -----------------------------------------------------------------------------------------
    def balances(self) -> dict[str, dict]:
        """Per hotkey: ``accrued`` (unsettled, nominal), ``paid`` (settled into period archives, after the
        normalisation), ``forfeited`` (nominal) and the ban's end if any."""
        with self.book.lock:
            db = self.book.db
            rows = db.execute("SELECT hotkey, status, SUM(amount) FROM rl_accruals WHERE status<>? "
                              "GROUP BY hotkey, status", (SETTLED,)).fetchall()
            docs = [json.loads(d) for d, in db.execute("SELECT doc FROM rl_pay_periods").fetchall()]
            bans = dict(db.execute("SELECT hotkey, until FROM rl_pay_bans").fetchall())
        out: dict[str, dict] = {}

        def entry(hotkey: str) -> dict:
            return out.setdefault(hotkey, {"accrued": 0.0, "paid": 0.0, "forfeited": 0.0})

        for hotkey, status, amount in rows:
            entry(hotkey)[status] += float(amount)
        for doc in docs:
            for hotkey, amount in doc["rewards_by_hotkey"].items():
                entry(hotkey)["paid"] += float(amount)
        for hotkey, until in bans.items():
            entry(hotkey)["banned_until"] = until
        return {h: out[h] for h in sorted(out)}

    def _recent(self, environment: str) -> dict:
        """What the env's last ``reference_groups`` graded groups (every hotkey) turned into: the share trained or
        waiting for the trainer (in zone, not refused) and the share that were out-of-zone first scans. A miner
        reads expected pay per generated token as ``token_price x (trained_share + 0.25 x first_scan_share)``."""
        with self.book.lock:
            rows = self.book.db.execute(
                "SELECT verdict, training, first_scan FROM rl_groups WHERE environment=? AND verdict!='rejected' "
                "ORDER BY seq DESC LIMIT ?", (environment, self.config.reference_groups)).fetchall()
        n = len(rows)
        trained = sum(1 for verdict, training, _ in rows if verdict == "in_zone" and training != "rejected")
        first = sum(1 for verdict, _, first_scan in rows if verdict != "in_zone" and first_scan)
        return {"groups": n, "trained_share": trained / n if n else None,
                "first_scan_share": first / n if n else None}

    def board(self, now: float | None = None, hotkey: str | None = None) -> dict:
        """``GET /rl/pay`` (async): per-env prices and per-hotkey accrued / paid, rebuilt at most every
        ``BOARD_CACHE_S`` (public: one answer serves every caller in between)."""
        now = float(self.clock()) if now is None else now
        cached = self._board
        if cached is None or not 0.0 <= now - cached[0] < BOARD_CACHE_S:
            cached = self._board = (now, self._build_board(now))
        if hotkey is None:
            return cached[1]
        balances = cached[1]["hotkeys"]
        return {**cached[1], "hotkeys": {hotkey: balances.get(hotkey, {"accrued": 0.0, "paid": 0.0, "forfeited": 0.0})}}

    def _build_board(self, now: float) -> dict:
        balances = self.balances()
        with self.book.lock:
            periods = [(w, e, p) for w, e, p in self.book.db.execute(
                "SELECT work_period, entry_period, published FROM rl_pay_periods ORDER BY work_period DESC LIMIT 12")]
        environments = self.prices.board()
        for name, entry in environments.items():
            entry["recent"] = self._recent(name)
        return {"policy": "pay-per-matched-token/v1", "floor": self.config.floor, "cap": self.config.cap,
                "start_pool": getattr(self.prices, "start_pool", self.config.cap),
                "period": cp.period_of(now, self.config.genesis), "period_seconds": cp.PERIOD_SECONDS,
                "environments": environments, "hotkeys": balances,
                "settled_periods": [{"work_period": w, "entry_period": e, "published": p == 1}
                                    for w, e, p in periods],
                "note": ("Units are fractions of the task's emission cap. A trained rollout at or above the floor "
                         "earns its matched tokens x the env's token_price; an out-of-zone first scan earns 25 % of "
                         "token_price x the env's mean matched tokens per trained group. At multiplier 1 and the "
                         "target mix a 72-minute period pays start_pool; it pays at most the cap (scaled down when "
                         "over) and enters the weights through the per-period EMA; a ban forfeits what is not "
                         "settled.")}


__all__ = ["ACCRUED", "FORFEITED", "LedgerConfig", "RL_PERIOD_ARCHIVE_SCHEMA", "RlPayLedger", "SETTLED",
           "paid_tokens"]
