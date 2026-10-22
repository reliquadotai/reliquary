"""``POST /miner/setup``: keep the latest hardware each opted-in miner process declares (``protocol/miner_setup.py``).

Stored in the validator's state dir, read with ``reliquary miner-setups``; never served publicly, never scored.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from reliquary.protocol.miner_setup import MINER_SETUP_PATH, MinerSetupRequest, verify_miner_setup_signature
from reliquary.validator.corpus_registration import NOT_REGISTERED, UNAVAILABLE

logger = logging.getLogger(__name__)

DB_NAME = "miner-setups.sqlite3"
REQUEST_SKEW_S = 120
MIN_INTERVAL_S = 60          # per hotkey and instance
MAX_INSTANCES_PER_HOTKEY = 32
ACTIVE_WINDOW_S = 24 * 3600


class MinerSetupStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS setups (hotkey TEXT NOT NULL, instance_id TEXT NOT NULL, "
            "at INTEGER NOT NULL, received_at REAL NOT NULL, report TEXT NOT NULL, "
            "PRIMARY KEY (hotkey, instance_id))")
        self._db.commit()

    def put(self, hotkey: str, instance_id: str, at: int, received_at: float, report: dict,
            *, min_interval_s: float | None = None) -> bool:
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if min_interval_s is not None:
                last = self._db.execute("SELECT at, received_at FROM setups WHERE hotkey=? AND instance_id=?",
                                        (hotkey, instance_id)).fetchone()
                if last is not None and (at <= last[0] or received_at - last[1] < min_interval_s):
                    return False
            self._db.execute("INSERT OR REPLACE INTO setups VALUES (?, ?, ?, ?, ?)",
                             (hotkey, instance_id, at, received_at, json.dumps(report, sort_keys=True)))
            # Bound one hotkey's rows: its oldest instances go first.
            self._db.execute(
                "DELETE FROM setups WHERE hotkey=? AND instance_id NOT IN (SELECT instance_id FROM setups "
                "WHERE hotkey=? ORDER BY received_at DESC LIMIT ?)", (hotkey, hotkey, MAX_INSTANCES_PER_HOTKEY))
            return True

    def rows(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT hotkey, instance_id, at, received_at, report FROM setups ORDER BY hotkey, received_at"
            ).fetchall()
        return [{"hotkey": h, "instance_id": i, "at": a, "received_at": r, "report": json.loads(rep)}
                for h, i, a, r, rep in rows]


def summarize(rows: list[dict], *, now: float | None = None, active_window_s: float = ACTIVE_WINDOW_S) -> list[dict]:
    """One line per hotkey over its instances seen within ``active_window_s``: each card counted once."""
    now = time.time() if now is None else now
    by_hotkey: dict[str, list[dict]] = {}
    for row in rows:
        if now - row["received_at"] <= active_window_s:
            by_hotkey.setdefault(row["hotkey"], []).append(row)
    out = []
    for hotkey, items in sorted(by_hotkey.items()):
        cards: dict[str, str] = {}
        anonymous: list[str] = []
        notes = []
        for row in items:
            for gpu in row["report"].get("gpus", []):
                if gpu.get("id_hash"):
                    cards[gpu["id_hash"]] = gpu["name"]
                else:
                    anonymous.append(gpu["name"])
            if row["report"].get("note"):
                notes.append(row["report"]["note"])
        gpus = Counter(list(cards.values()) + anonymous)
        out.append({"hotkey": hotkey, "instances": len(items), "gpu_count": sum(gpus.values()),
                    "gpus": dict(gpus), "notes": sorted(set(notes)),
                    "last_seen": max(row["received_at"] for row in items)})
    return out


def build_miner_setup_router(store: MinerSetupStore, *, validator_hotkey: str,
                             registration: Callable[[str], Awaitable[str | None]] | None = None,
                             clock: Callable[[], float] = time.time, request_skew_s: int = REQUEST_SKEW_S,
                             min_interval_s: float = MIN_INTERVAL_S,
                             verify: Callable[..., bool] = verify_miner_setup_signature) -> APIRouter:
    router = APIRouter()

    def refuse(status: int, reason: str, **extra) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": reason, **extra})

    @router.post(MINER_SETUP_PATH)
    async def miner_setup(request: MinerSetupRequest):
        now = clock()
        if not request.signature:
            return refuse(401, "signature_required")
        if abs(int(now) - request.at) > request_skew_s:
            return refuse(409, "stale_request", max_skew_s=request_skew_s, now=int(now))
        if not await asyncio.to_thread(verify, request, validator_hotkey=validator_hotkey, path=MINER_SETUP_PATH):
            return refuse(403, "bad_signature")
        if registration is not None:
            try:
                why = await registration(request.miner_hotkey)
            except Exception as exc:
                logger.warning("miner setup registration check failed (%s)", type(exc).__name__)
                why = UNAVAILABLE
            if why == NOT_REGISTERED:
                return refuse(403, "hotkey_not_registered")
            if why is not None:
                return refuse(503, "registration_unavailable")
        instance = request.report.instance_id
        stored = await asyncio.to_thread(store.put, request.miner_hotkey, instance, request.at, now,
                                         request.report.model_dump(mode="json"), min_interval_s=min_interval_s)
        if not stored:
            return refuse(429, "too_frequent", min_interval_s=min_interval_s)
        return {"stored": True}

    return router


__all__ = ["DB_NAME", "MinerSetupStore", "build_miner_setup_router", "summarize"]
