"""Opt-in public scalar projection of recorded RL groups. No wallet or miner imports."""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import time

MAX_ROWS = 1000
MAX_SQLITE_ROW_BYTES = 65536
SCHEMA = "reliquary/rl-miner-submissions/v1"
RUN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
HOTKEY = re.compile(r"[1-9A-HJ-NP-Za-km-z]{47,49}\Z")
ENV = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}\Z")
MODE_ENV = "RELIQUARY_RL_MINER_SUBMISSIONS_MODE"
START_ENV = "RELIQUARY_RL_MINER_SUBMISSIONS_START_AT"
logger = logging.getLogger(__name__)


def _number(value, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 2**53 - 1 or (integer and type(value) is not int):
        raise ValueError("Invalid scalar")
    return value


def read_snapshot(database, *, run_id, not_before, now=None):
    """Caller must qualify this DB and cutoff against the active native run first.

    A separate mode=ro connection, query-only transaction and bounded VM/time budget
    keep this export independent from the validator's writable connection/lock.
    Only post-cutoff groups are considered; this is not an all-time request log.
    """
    now = time.time() if now is None else now
    if not isinstance(run_id, str) or not RUN.fullmatch(run_id) or not 0 < not_before <= now:
        raise ValueError("Invalid run scope")
    uri = Path(database).resolve(strict=True).as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=0.5)
    deadline = time.monotonic() + 2
    operations = 0

    def budget():
        nonlocal operations
        operations += 1000
        return int(operations > 300_000 or time.monotonic() > deadline)

    try:
        # Bound values before SQLite/Python materializes them, not after fetchall().
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_SQLITE_ROW_BYTES)
        db.execute("PRAGMA query_only=ON")
        db.set_progress_handler(budget, 1000)
        db.execute("BEGIN")
        values = db.execute(
            "SELECT lease_id, hotkey, environment, k, issued_at, arrived_at, verdict, "
            "completion_tokens, rewards, training, audit FROM rl_groups "
            "WHERE issued_at>=? AND arrived_at>=? AND arrived_at<=? ORDER BY seq DESC LIMIT ?", (not_before, not_before, now, MAX_ROWS + 1)
        ).fetchall()
        rows = []
        for lease, hotkey, environment, checkpoint, issued, arrived, verdict, tokens, rewards, training, audit in values[:MAX_ROWS]:
            if (not isinstance(lease, str) or not 0 < len(lease) <= 128 or not HOTKEY.fullmatch(hotkey)
                or not ENV.fullmatch(environment) or verdict not in {"in_zone", "too_easy", "too_hard", "truncated", "rejected"}
                or training not in {None, "queued", "accepted", "rejected"}
                or audit not in {None, "not_drawn", "pending", "passed", "failed", "inconclusive"}
                or not 0 < issued <= arrived <= now):
                raise ValueError("Invalid group")
            reward_mean = None
            if rewards is not None:
                if len(rewards) > 16384:
                    raise ValueError("Oversized rewards")
                reward_values = json.loads(rewards)
                if not isinstance(reward_values, list) or not 0 < len(reward_values) <= 128 or any(type(r) not in (int, float) or not math.isfinite(r) for r in reward_values):
                    raise ValueError("Invalid rewards")
                reward_mean = statistics.fmean(reward_values)
                if not math.isfinite(reward_mean):
                    raise ValueError("Invalid mean")
            public_id = hashlib.sha256(("rl-submission/v1\0" + run_id + "\0" + lease).encode()).hexdigest()
            rows.append({"id": public_id, "hotkey": hotkey, "receivedAt": arrived * 1000,
                         "environment": environment, "checkpoint": _number(checkpoint, True),
                         "completionTokens": _number(tokens, True), "deliverySeconds": arrived - issued,
                         "rewardMean": reward_mean, "verdict": verdict, "training": training, "audit": audit})
        return {"schema": SCHEMA, "run_id": run_id, "observed_at": now * 1000,
                "not_before": not_before * 1000, "has_older": len(values) > MAX_ROWS, "rows": rows}
    finally:
        db.close()


class CachedExporter:
    def __init__(self, database, *, run_id, not_before):
        self.database, self.run_id, self.not_before = database, run_id, not_before
        self.lock, self.cached, self.last_attempt, self.failed = asyncio.Lock(), None, None, True

    async def get(self):
        async with self.lock:
            if self.last_attempt is None or time.monotonic() - self.last_attempt >= 60:
                self.last_attempt = time.monotonic()  # Failure also claims the minute.
                try:
                    candidate = await asyncio.to_thread(read_snapshot, self.database, run_id=self.run_id, not_before=self.not_before)
                except (ValueError, TypeError, OSError, sqlite3.Error, OverflowError):
                    self.failed = True
                else:
                    self.cached, self.failed = candidate, False
            return None if self.failed else self.cached


def build_router(database, *, run_id, not_before):
    """Mount explicitly on the active validator only after DB/run qualification."""
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse
    globals()["Request"] = Request
    router = APIRouter()
    exporter = CachedExporter(database, run_id=run_id, not_before=not_before)

    @router.get("/rl/miner-submissions")
    async def submissions(request: Request):
        if list(request.query_params.multi_items()) != [("run_id", run_id)]:
            return JSONResponse(status_code=400, content={"reason": "invalid_run_selection"})
        body = await exporter.get()
        if body is None:
            return JSONResponse(status_code=503, content={"reason": "submissions_unavailable"})
        return JSONResponse(content=body, headers={"Cache-Control": "no-store"})
    return router


def mount_miner_submissions(server, rl, *, environ=os.environ):
    """An optional scalar route: invalid export configuration must not stop mining."""
    if environ.get(MODE_ENV) != "enabled":
        return
    try:
        run_id = rl.settings.run
        not_before = float(environ.get(START_ENV, ""))
        if (not isinstance(run_id, str) or not RUN.fullmatch(run_id)
                or not math.isfinite(not_before) or not 0 < not_before <= time.time()):
            raise ValueError("Invalid run scope")
        with rl.book.lock:
            database = next(row[2] for row in rl.book.db.execute("PRAGMA database_list") if row[1] == "main")
        if not database:
            raise ValueError("A persistent journal is required")
        database = Path(database).resolve(strict=True)
        server.mount_rl_lease_router(build_router(database, run_id=run_id, not_before=not_before))
    except Exception:
        # Keep the optional feed unavailable without exposing paths or configuration.
        logger.warning("RL miner submission export unavailable; optional route not mounted")
