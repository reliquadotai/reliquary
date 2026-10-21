"""The async RL observation log on R2 (next-rl-run-decisions.md §A and §D; rl-async-design.md §4.5).

Every scan the lease book records (``rl_prompt_scans``: env, prompt, checkpoint k, the 16 rewards, the verdict, the
first-scan flag, the time, and the state it moved the prompt to) is published as the window-era log is: immutable
gzip JSONL segments, one per flush (60 s), and a head index signed by the validator hotkey, with closed pages of 200
segments (``services/publication``, unchanged format). Only the root differs: ``rl/<run>/observations/`` in the
``reliquary-public`` bucket, beside the run's checkpoints. No hotkey, no token is ever published.

* ``ScanJournal``: the ``ObservationPublisher`` runtime over the lease book's file (segment plans and closed pages
  persisted before upload, so a restart rebuilds byte-identical segments).
* ``scan_publisher``: the publisher, writing through the public bucket's client (conditional PUT for immutables).
* ``ObservationBoard`` / ``GET /rl/observations?after=<number>``: the bucket is private, so the validator serves the
  head it last wrote plus presigned GET URLs for its pages and for the segments after ``after`` (at most
  ``MAX_URLS``). URLs are not signed: every byte is checked against the signed head.
* The reader is ``protocol/rl_scan_log.read_scan_log`` (the reference miner bootstraps its prompt table from it at
  start, then polls ``/rl/prompts/state`` from the last segment's seq).

Event (one per scan, canonical JSON): ``{"type": "scan", "id": seq, "env", "prompt_idx", "checkpoint", "rewards",
"verdict", "first_scan", "state", "until_k", "ts"}``; ``ts`` is the scan's time.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.services import publication as pub

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_scan_segments(number INTEGER PRIMARY KEY, first_seq INTEGER NOT NULL,
    last_seq INTEGER NOT NULL, flush_at REAL NOT NULL, sha256 TEXT, size INTEGER, checkpoints TEXT,
    committed INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS rl_scan_index_pages(first_number INTEGER PRIMARY KEY, body BLOB NOT NULL);
"""
FLUSH_EVERY_S = 60.0
MAX_URLS = 512           # segment URLs per /rl/observations answer (a reader asks again with a higher ``after``)
EVENT_KEYS = ("type", "id", "env", "prompt_idx", "checkpoint", "rewards", "verdict", "first_scan", "state",
              "until_k", "ts")


def order_sha256(task_id: str, run: str) -> str:
    """The head's ``order_sha256``: what the log is of (the window format's slot for the service order)."""
    return hashlib.sha256(canonical_json_bytes({"task_id": task_id, "run": run, "log": "rl-async-scans/v1"})).hexdigest()


def project_scans(events: list[dict], *, flush_at: float) -> list[dict]:
    """Published events: the allow-listed keys only (the journal builds nothing else; this keeps it so)."""
    return [{k: e[k] for k in EVENT_KEYS if k in e} for e in events]


class ScanJournal:
    """``ObservationPublisher``'s runtime over ``LeaseBook``: the scans are the log, seq is the scan's seq."""

    def __init__(self, book, *, order_sha256: str) -> None:
        self.book, self.db, self.lock = book, book.db, book.lock
        self.contract = SimpleNamespace(sha256=order_sha256)
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    def events(self, *, after: int = 0, limit: int = 1000) -> list[tuple[int, dict]]:
        with self.lock:
            rows = self.db.execute("SELECT seq, environment, prompt_idx, k, rewards, verdict, first_scan, at, state, "
                                   "until_k FROM rl_prompt_scans WHERE seq>? ORDER BY seq LIMIT ?",
                                   (int(after), int(limit))).fetchall()
        return [(seq, {"type": "scan", "id": seq, "env": env, "prompt_idx": p, "checkpoint": k,
                       "rewards": json.loads(r), "verdict": v, "first_scan": bool(f), "state": state,
                       "until_k": until, "ts": at})
                for seq, env, p, k, r, v, f, at, state, until in rows]

    def plan_segment(self, *, max_events: int, now: float | None = None) -> dict | None:
        """The unfinished segment if any (same range, same bytes after a restart), else the next one. A segment
        starts right after the previous one (``first_seq = last_seq + 1``), whatever seq gaps lie between."""
        with self.book.transaction():
            row = self.db.execute("SELECT number, first_seq, last_seq, flush_at FROM rl_scan_segments "
                                  "WHERE committed=0").fetchone()
            if row is not None:
                return {"number": row[0], "first_seq": row[1], "last_seq": row[2], "flush_at": row[3]}
            last_seq, last_number = self.db.execute("SELECT COALESCE(MAX(last_seq),0), COALESCE(MAX(number),0) "
                                                    "FROM rl_scan_segments").fetchone()
            top = self.db.execute("SELECT MAX(seq) FROM (SELECT seq FROM rl_prompt_scans WHERE seq>? ORDER BY seq "
                                  "LIMIT ?)", (last_seq, int(max_events))).fetchone()[0]
            if top is None:
                return None
            plan = {"number": last_number + 1, "first_seq": last_seq + 1, "last_seq": int(top),
                    "flush_at": float(int(time.time() if now is None else now))}
            self.db.execute("INSERT INTO rl_scan_segments(number, first_seq, last_seq, flush_at) VALUES(?,?,?,?)",
                            (plan["number"], plan["first_seq"], plan["last_seq"], plan["flush_at"]))
            return plan

    def commit_segment(self, number: int, *, sha256: str, size: int, windows: list[int],
                       checkpoints: list[int]) -> None:
        with self.book.transaction():
            self.db.execute("UPDATE rl_scan_segments SET sha256=?, size=?, checkpoints=?, committed=1 WHERE number=?",
                            (sha256, size, json.dumps(checkpoints), number))

    def index_page(self, first_number: int, build) -> bytes:
        """A closed page's signed bytes, stored at first build (a signature is not reproducible)."""
        with self.book.transaction():
            row = self.db.execute("SELECT body FROM rl_scan_index_pages WHERE first_number=?",
                                  (first_number,)).fetchone()
            if row is not None:
                return bytes(row[0])
            body = build()
            self.db.execute("INSERT INTO rl_scan_index_pages(first_number, body) VALUES(?,?)", (first_number, body))
            return body

    def published_segments(self, *, first_number: int = 1, last_number: int | None = None) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT number, first_seq, last_seq, sha256, size FROM rl_scan_segments WHERE "
                                   "committed=1 AND number>=? AND number<=? ORDER BY number",
                                   (first_number, last_number if last_number is not None else 2 ** 62)).fetchall()
        return [{"number": r[0], "first_seq": r[1], "last_seq": r[2], "sha256": r[3], "size": r[4]} for r in rows]

    def last_published_number(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COALESCE(MAX(number),0) FROM rl_scan_segments WHERE committed=1"
                                   ).fetchone()[0]


# --- the bucket -------------------------------------------------------------------------------------------------
def bucket_io(client, bucket: str) -> tuple[Callable[..., Awaitable[None]], Callable[[str], Awaitable[bytes | None]]]:
    """``put`` / ``get`` over a sync boto3 client (the public bucket's): immutables are written once
    (``If-None-Match: *``; an existing key must hold the same bytes), the head is replaced."""

    def _put(key: str, body: bytes, content_type: str, cache_control: str) -> None:
        from botocore.exceptions import ClientError

        immutable = cache_control == pub.SEGMENT_CACHE
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type, CacheControl=cache_control,
                              **({"IfNoneMatch": "*"} if immutable else {}))
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if not immutable or (code not in ("PreconditionFailed", "412") and status != 412):
                raise
            if client.get_object(Bucket=bucket, Key=key)["Body"].read() != body:
                raise ValueError(f"immutable object {key} already exists with different bytes") from exc

    def _get(key: str) -> bytes | None:
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            return None

    async def put(key: str, body: bytes, content_type: str, cache_control: str) -> None:
        await asyncio.to_thread(_put, key, body, content_type, cache_control)

    async def get(key: str) -> bytes | None:
        return await asyncio.to_thread(_get, key)

    return put, get


def remote_document_signer(client) -> Callable[[dict], str]:
    """Sign heads and pages through the remote signer's observation operation
    (``RemoteSignerClient.sign_rl_observation``): the controller holds no private key."""
    return lambda body: bytes(client.sign_rl_observation(dict(body))).hex()


def scan_publisher(book, *, run: str, task_id: str, wallet, put, get=None, clock=time.time, signer_client=None):
    """The scan log's publisher, signing with the remote signer when ``signer_client`` is given (``wallet`` is then
    its public identity), else with the local wallet."""
    from reliquary.shared.ckpt_index import observations_root

    journal = ScanJournal(book, order_sha256=order_sha256(task_id, run))
    signer = None if signer_client is None else remote_document_signer(signer_client)
    return pub.ObservationPublisher(journal, run_id=run, task_id=task_id, wallet=wallet, put=put, get=get,
                                    clock=clock, root=observations_root(run), project=project_scans, signer=signer)


# --- serving ------------------------------------------------------------------------------------------------------
class ObservationBoard:
    """``GET /rl/observations``: the head the publisher last wrote, with presigned URLs (the bucket is private)."""

    def __init__(self, publisher, *, presign: Callable[[str], str], ttl_seconds: int = 900,
                 clock: Callable[[], float] = time.time) -> None:
        self.publisher, self._presign, self._ttl, self._clock = publisher, presign, ttl_seconds, clock

    def answer(self, after: int = 0) -> dict | None:
        raw = self.publisher.head_bytes
        if raw is None:
            return None
        head = json.loads(raw)
        run, root = self.publisher.run_id, self.publisher.root
        keys = [page["key"] for page in head["pages"]]
        first = max(int(after), 0) + 1
        keys += [pub.segment_key(run, n, root=root)
                 for n in range(first, min(int(head["last_number"]), first + MAX_URLS - 1) + 1)]
        return {"index": head, "root": root, "urls": {key: self._presign(key) for key in keys},
                "urls_expire_at": int(self._clock()) + int(self._ttl)}


async def rl_observations_response(board: ObservationBoard | None, after: str = "0"):
    from fastapi.responses import JSONResponse

    try:
        after_n = int(after)
    except ValueError:
        return JSONResponse(status_code=422, content={"reason": "malformed_request"})
    answer = None if board is None else board.answer(after_n)
    if answer is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"}, headers={"Retry-After": "60"})
    return JSONResponse(content=answer, headers={"Cache-Control": "private, max-age=15"})


__all__ = ["FLUSH_EVERY_S", "MAX_URLS", "ObservationBoard", "ScanJournal", "bucket_io", "order_sha256",
           "project_scans", "remote_document_signer", "rl_observations_response", "scan_publisher"]
