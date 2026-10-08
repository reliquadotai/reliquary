"""Static, signed publication of the run observation log on R2 (decision D, validator side).

Layout (public bucket, ``RELIQUARY_OBSERVATIONS_BUCKET``):

* ``observations/run-<run_id>/seg-NNNNNN.jsonl.gz`` -- immutable segments (``Cache-Control: public,
  max-age=31536000, immutable``). One canonical JSON event per line, gzip with ``mtime=0``, no file
  name, level 9. A segment covers the log events with ``first_seq <= seq <= last_seq``.
* ``observations/run-<run_id>/index.json`` -- the index (``Cache-Control: public, max-age=15``),
  signed by the validator hotkey: ``signature`` = sr25519 over the canonical JSON of every other
  field (hex). ``verify_index`` is the miners' check.

Reader rules (part of the format):

* An id can have several events. Readers take the LAST ``settle`` event of an id (highest segment,
  then line order) as its status; an ``observation`` event is never superseded, only settled.
* ``ts`` of every event of a segment is the flush time of that segment (whole seconds): there is no
  finer timing, so ids that the validator settled together (e.g. all forfeits of one failed audit)
  cannot be clustered by timestamp inside a segment.
* Reasons ``probation_limit`` and ``banned`` are published as ``refused`` (R28): a reason must not
  reveal a submitter's state. Every other reason is published as is.
* No hotkey, token count, run salt or run metadata is ever serialized: events are projected on an
  explicit allow-list of keys. Transient statuses (``exploration_recording``,
  ``service_unproven_recording``) are never published.

Immutability: the segment range (and its flush time) is persisted in the runtime SQLite before the
upload, so a restart rebuilds byte-identical bytes. A segment key is written with a conditional PUT
(``If-None-Match: *``); if the key already exists the stored bytes are compared and must be equal.
Residual: R2 cannot forbid an operator or a leaked credential from overwriting a key; the index
carries each segment's sha256 and is signed, so a reader detects a swapped segment.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import math
import time

from reliquary.constants import SERVICE_OBSERVATION_FLUSH_SECONDS, SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS
from reliquary.protocol.release_contract import canonical_json_bytes

logger = logging.getLogger(__name__)

INDEX_SCHEMA = "service-observation-index/v1"
SEGMENT_CACHE = "public, max-age=31536000, immutable"
INDEX_CACHE = "public, max-age=15"
OBSERVATIONS_BUCKET_ENV = "RELIQUARY_OBSERVATIONS_BUCKET"

PUBLIC_KEYS = frozenset({"type", "id", "env", "dataset", "prompt_idx", "checkpoint_n", "checkpoint", "window",
                         "ts", "rewards_bps", "verdict", "candidate", "lane", "status", "proof", "reason",
                         "uncertain"})
TRANSIENT_STATUSES = frozenset({"exploration_recording", "service_unproven_recording"})
COARSENED_REASONS = frozenset({"probation_limit", "banned"})
REFUSED = "refused"


def segment_key(run_id: str, number: int) -> str:
    return f"observations/run-{run_id}/seg-{number:06d}.jsonl.gz"


def index_key(run_id: str) -> str:
    return f"observations/run-{run_id}/index.json"


def public_events(events: list[dict], *, flush_at: float) -> list[dict]:
    """The log's events as published: allow-listed keys, R28 reasons, timestamps at the flush time,
    transient statuses dropped."""
    stamp = float(math.floor(flush_at))
    out = []
    for event in events:
        if event.get("status") in TRANSIENT_STATUSES:
            continue
        item = {k: v for k, v in event.items() if k in PUBLIC_KEYS}
        item["ts"] = stamp
        if item.get("reason") in COARSENED_REASONS:
            item["reason"] = REFUSED
        out.append(item)
    return out


def encode_segment(events: list[dict]) -> bytes:
    return gzip.compress(b"".join(canonical_json_bytes(e) + b"\n" for e in events), compresslevel=9, mtime=0)


def build_index(*, run_id: str, task_id: str, order_sha256: str, validator_hotkey: str,
                segments: list[dict], generated_at: float) -> dict:
    return {"schema": INDEX_SCHEMA, "run_id": run_id, "task_id": task_id, "order_sha256": order_sha256,
            "validator_hotkey": validator_hotkey, "generated_at": generated_at,
            "segments": [{"key": segment_key(run_id, s["number"]), "number": s["number"],
                          "first_seq": s["first_seq"], "last_seq": s["last_seq"], "sha256": s["sha256"],
                          "size": s["size"], "windows": s["windows"], "checkpoints": s["checkpoints"]}
                         for s in segments]}


def sign_index(index: dict, wallet) -> dict:
    return {**index, "signature": bytes(wallet.hotkey.sign(canonical_json_bytes(index))).hex()}


def verify_index(index_bytes, validator_hotkey: str) -> dict:
    """The index body if its signature by ``validator_hotkey`` holds; ValueError otherwise."""
    from bittensor_wallet import Keypair
    try:
        document = json.loads(index_bytes) if isinstance(index_bytes, (bytes, bytearray, str)) else index_bytes
    except ValueError as exc:
        raise ValueError("observation index is not JSON") from exc
    if not isinstance(document, dict) or document.get("schema") != INDEX_SCHEMA:
        raise ValueError("unknown observation index")
    body = {k: v for k, v in document.items() if k != "signature"}
    if body.get("validator_hotkey") != validator_hotkey:
        raise ValueError("index names another validator hotkey (signature not checked)")
    try:
        ok = Keypair(ss58_address=validator_hotkey).verify(canonical_json_bytes(body),
                                                           bytes.fromhex(document["signature"]))
    except Exception as exc:
        raise ValueError("bad index signature") from exc
    if not ok:
        raise ValueError("bad index signature")
    return body


def r2_put(bucket: str):
    """The production ``put``: segments are written once (conditional PUT), the index is replaced."""
    from reliquary.infrastructure.storage import upload_bytes

    async def put(key: str, body: bytes, content_type: str, cache_control: str) -> None:
        await upload_bytes(key, body, content_type=content_type, cache_control=cache_control,
                           if_absent=cache_control == SEGMENT_CACHE, bucket_name=bucket)
    return put


class ObservationPublisher:
    def __init__(self, runtime, *, run_id: str, task_id: str, wallet, put, clock=time.time):
        self.runtime, self.run_id, self.task_id, self.wallet, self.put, self.clock = runtime, run_id, task_id, wallet, put, clock
        # After a restart the last index PUT is unknown: write it once on the first flush.
        self._index_dirty = True

    def _events(self, plan: dict) -> list[dict]:
        count = plan["last_seq"] - plan["first_seq"] + 1
        rows = self.runtime.events(after=plan["first_seq"] - 1, limit=count)
        return public_events([e for seq, e in rows if seq <= plan["last_seq"]], flush_at=plan["flush_at"])

    async def flush(self) -> dict | None:
        """Publish the next segment (if any) and the index. None when nothing was written."""
        plan = await asyncio.to_thread(self.runtime.plan_segment, max_events=SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS,
                                       now=self.clock())
        if plan is not None:
            events = await asyncio.to_thread(self._events, plan)
            body = await asyncio.to_thread(encode_segment, events)
            await self.put(segment_key(self.run_id, plan["number"]), body, "application/gzip", SEGMENT_CACHE)
            windows = sorted({e["window"] for e in events})
            checkpoints = sorted({e["checkpoint_n"] for e in events if "checkpoint_n" in e})
            await asyncio.to_thread(self.runtime.commit_segment, plan["number"],
                                    sha256=hashlib.sha256(body).hexdigest(), size=len(body),
                                    windows=windows[:1] + windows[-1:], checkpoints=checkpoints[:1] + checkpoints[-1:])
            self._index_dirty = True
        if not self._index_dirty:
            return None
        segments = await asyncio.to_thread(self.runtime.published_segments)
        if not segments:
            self._index_dirty = False
            return None
        index = sign_index(build_index(run_id=self.run_id, task_id=self.task_id,
                                       order_sha256=self.runtime.contract.sha256,
                                       validator_hotkey=self.wallet.hotkey.ss58_address,
                                       segments=segments, generated_at=self.clock()), self.wallet)
        await self.put(index_key(self.run_id), canonical_json_bytes(index), "application/json", INDEX_CACHE)
        self._index_dirty = False
        return index

    async def run(self, stop: asyncio.Event, interval: float = SERVICE_OBSERVATION_FLUSH_SECONDS) -> None:
        while not stop.is_set():
            try:
                while not stop.is_set() and await self.flush() is not None:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("observation publication failed; retrying next flush")
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
