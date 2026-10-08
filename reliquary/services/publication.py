"""Static, signed publication of the run observation log on R2 (decision D, validator side).

Layout (public bucket, ``RELIQUARY_OBSERVATIONS_BUCKET``):

* ``observations/run-<run_id>/seg-NNNNNN.jsonl.gz`` -- immutable segments (``Cache-Control: public,
  max-age=31536000, immutable``). One canonical JSON event per line, gzip with ``mtime=0``, no file
  name, level 9. A segment covers the log events with ``first_seq <= seq <= last_seq``.
* ``observations/run-<run_id>/index.json`` -- the HEAD index (``Cache-Control: public, max-age=15``):
  ``{schema, run_id, order_sha256, validator, generated_at, last_number, segments, pages}`` where
  ``segments`` are EVERY segment after the last closed page (at most ``PAGE_SEGMENTS - 1`` entries: number,
  key, first_seq, last_seq, sha256, size) and ``pages`` reference the closed pages (key, first_number,
  last_number, sha256, size). Pages and head together list each segment number exactly once.
* ``observations/run-<run_id>/index-NNNNNN-NNNNNN.json`` -- a closed PAGE: exactly ``PAGE_SEGMENTS`` (200)
  consecutive segment entries, written once (immutable ``Cache-Control``) BEFORE the head that references it. A
  miner walks the pages once, then polls the head only.
* Signature (head and pages alike): ``signature`` = sr25519 hex over ``b"reliquary/observation-index/v1\n"``
  + the canonical JSON of the object without its ``signature`` field. ``verify_index`` checks a head
  or a page (its result carries ``kind``), ``verify_segment`` checks a segment against its entry.

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
import threading
import time
import zlib

from reliquary.constants import SERVICE_OBSERVATION_FLUSH_SECONDS, SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS
from reliquary.protocol.release_contract import canonical_json_bytes

logger = logging.getLogger(__name__)

INDEX_SCHEMA = "service-observation-index/v1"
PAGE_SCHEMA = "service-observation-index-page/v1"
INDEX_DOMAIN = b"reliquary/observation-index/v1\n"
# R29: a page closes every PAGE_SEGMENTS segments; the head lists every segment after the last closed page
# (so it never holds more than PAGE_SEGMENTS - 1). There is no separate head size: it cannot leave a gap.
PAGE_SEGMENTS = 200
CONSECUTIVE_REMOTE_FAILURES_CRITICAL = 10
SEGMENT_CACHE = "public, max-age=31536000, immutable"
INDEX_CACHE = "public, max-age=15"
OBSERVATIONS_BUCKET_ENV = "RELIQUARY_OBSERVATIONS_BUCKET"

PUBLIC_KEYS = frozenset({"type", "id", "env", "dataset", "prompt_idx", "checkpoint_n", "checkpoint", "window",
                         "ts", "rewards_bps", "verdict", "candidate", "lane", "status", "proof", "reason",
                         "uncertain", "window_aborted"})
TRANSIENT_STATUSES = frozenset({"exploration_recording", "service_unproven_recording"})
COARSENED_REASONS = frozenset({"probation_limit", "banned"})
REFUSED = "refused"
# Every reason the runtime can publish (anything else is published as ``refused``).
PUBLIC_REASONS = frozenset({"already_scanned", "finalized", "zero_price", "cap", "order_inactive",
                            "exploration_disabled", "token_limit", "trained", "unaudited", REFUSED})


def segment_key(run_id: str, number: int) -> str:
    return f"observations/run-{run_id}/seg-{number:06d}.jsonl.gz"


def index_key(run_id: str) -> str:
    return f"observations/run-{run_id}/index.json"


def page_key(run_id: str, first_number: int, last_number: int) -> str:
    return f"observations/run-{run_id}/index-{first_number:06d}-{last_number:06d}.json"


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
        if "reason" in item and (item["reason"] in COARSENED_REASONS or item["reason"] not in PUBLIC_REASONS):
            item["reason"] = REFUSED
        out.append(item)
    return out


def encode_segment(events: list[dict]) -> bytes:
    return gzip.compress(b"".join(canonical_json_bytes(e) + b"\n" for e in events), compresslevel=9, mtime=0)


def segment_entry(run_id: str, s: dict) -> dict:
    return {"number": s["number"], "key": segment_key(run_id, s["number"]), "first_seq": s["first_seq"],
            "last_seq": s["last_seq"], "sha256": s["sha256"], "size": s["size"]}


def sign_document(body: dict, wallet) -> bytes:
    """The published bytes of a head or page: canonical JSON of ``body`` + its signature."""
    signature = bytes(wallet.hotkey.sign(INDEX_DOMAIN + canonical_json_bytes(body))).hex()
    return canonical_json_bytes({**body, "signature": signature})


def build_page(*, run_id: str, validator: str, segments: list[dict]) -> dict:
    return {"schema": PAGE_SCHEMA, "run_id": run_id, "validator": validator,
            "first_number": segments[0]["number"], "last_number": segments[-1]["number"],
            "segments": [segment_entry(run_id, s) for s in segments]}


def build_head(*, run_id: str, order_sha256: str, validator: str, segments: list[dict], pages: list[dict],
               last_number: int, generated_at: float) -> dict:
    return {"schema": INDEX_SCHEMA, "run_id": run_id, "order_sha256": order_sha256, "validator": validator,
            "generated_at": generated_at, "last_number": last_number,
            "segments": [segment_entry(run_id, s) for s in segments], "pages": pages}


def _check_entries(entries, run_id: str, *, first: int | None = None) -> None:
    if not isinstance(entries, list):
        raise ValueError("segments must be a list")
    previous = None
    for e in entries:
        if not isinstance(e, dict) or not all(k in e for k in ("number", "key", "first_seq", "last_seq", "sha256", "size")):
            raise ValueError("malformed segment entry")
        if e["key"] != segment_key(run_id, e["number"]):
            raise ValueError("segment key does not match run id and number")
        if e["first_seq"] > e["last_seq"]:
            raise ValueError("segment sequence range is inverted")
        if previous is not None and (e["number"] != previous["number"] + 1 or e["first_seq"] != previous["last_seq"] + 1):
            raise ValueError("segments are not contiguous")
        previous = e
    if first is not None and entries and entries[0]["number"] != first:
        raise ValueError("segments do not start where the index says")


def verify_index(index_bytes, validator_ss58: str, *, expected_run_id: str, min_last_number: int = 0) -> dict:
    """The body of a head or page index (plus ``kind``: ``"head"`` or ``"page"``) if its signature by
    ``validator_ss58`` holds, its run id is ``expected_run_id``, it is contiguous with well-named keys
    and (head) its ``last_number`` is at least ``min_last_number``. A head must start right after its
    last page (number ``pages * PAGE_SEGMENTS + 1``, also without page) and hold fewer than a page of
    entries. ValueError otherwise."""
    from bittensor_wallet import Keypair
    try:
        document = json.loads(index_bytes) if isinstance(index_bytes, (bytes, bytearray, str)) else index_bytes
    except ValueError as exc:
        raise ValueError("observation index is not JSON") from exc
    kinds = {INDEX_SCHEMA: "head", PAGE_SCHEMA: "page"}
    if not isinstance(document, dict) or document.get("schema") not in kinds:
        raise ValueError("unknown observation index")
    kind = kinds[document["schema"]]
    body = {k: v for k, v in document.items() if k != "signature"}
    if body.get("validator") != validator_ss58:
        raise ValueError("index names another validator (signature not checked)")
    try:
        ok = Keypair(ss58_address=validator_ss58).verify(INDEX_DOMAIN + canonical_json_bytes(body),
                                                         bytes.fromhex(document["signature"]))
    except Exception as exc:
        raise ValueError("bad index signature") from exc
    if not ok:
        raise ValueError("bad index signature")
    if body.get("run_id") != expected_run_id:
        raise ValueError("index belongs to another run")
    entries = body.get("segments")
    if kind == "page":
        if not isinstance(entries, list) or len(entries) != PAGE_SEGMENTS:
            raise ValueError("a page holds exactly %d segments" % PAGE_SEGMENTS)
        _check_entries(entries, expected_run_id, first=body["first_number"])
        if entries[-1]["number"] != body["last_number"]:
            raise ValueError("page last_number does not match its segments")
    else:
        _check_entries(entries, expected_run_id)
        if len(entries) >= PAGE_SEGMENTS:
            raise ValueError("head lists a whole page of segments")
        pages = body.get("pages")
        if not isinstance(pages, list):
            raise ValueError("pages must be a list")
        expected_first = 1
        for page in pages:
            if (page.get("first_number") != expected_first or page.get("last_number") != expected_first + PAGE_SEGMENTS - 1
                    or page.get("key") != page_key(expected_run_id, page["first_number"], page["last_number"])):
                raise ValueError("pages are not contiguous or badly named")
            expected_first += PAGE_SEGMENTS
        last = body.get("last_number")
        if not isinstance(last, int) or last < 0:
            raise ValueError("head last_number is malformed")
        if last != (entries[-1]["number"] if entries else expected_first - 1):
            raise ValueError("head last_number does not match its segments")
        if entries and entries[0]["number"] != expected_first:
            raise ValueError("head does not continue its pages")
        if last < min_last_number:
            raise ValueError("stale observation index")
    return {**body, "kind": kind}


def verify_page_entry(page_bytes: bytes, entry: dict) -> None:
    """A page's size and sha256 must be the ones the (verified) head states for it."""
    if len(page_bytes) != entry["size"] or hashlib.sha256(page_bytes).hexdigest() != entry["sha256"]:
        raise ValueError("page does not match its head entry")


MAX_SEGMENT_BYTES = 256 * 1024 * 1024   # decompressed size a reader accepts for one segment


def verify_segment(segment_bytes: bytes, entry: dict, *, max_size: int = MAX_SEGMENT_BYTES) -> list[dict]:
    """The events of a segment if its size and sha256 are the ones its (verified) index entry states.

    The decompressed size is capped at ``max_size`` (a signed segment is still not trusted to be small):
    above it, or when the gzip stream is truncated or followed by other bytes, ``ValueError``."""
    if len(segment_bytes) != entry["size"] or hashlib.sha256(segment_bytes).hexdigest() != entry["sha256"]:
        raise ValueError("segment does not match its index entry")
    decompressor = zlib.decompressobj(31)
    try:
        raw = decompressor.decompress(segment_bytes, max_size + 1)
    except zlib.error as exc:
        raise ValueError(f"segment is not a valid gzip stream: {exc}") from exc
    if len(raw) > max_size:
        raise ValueError("segment decompresses above the size limit")
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("segment is not exactly one complete gzip stream")
    return [json.loads(x) for x in raw.decode().splitlines()]


def r2_put(bucket: str):
    """The production ``put``: segments and pages are written once (conditional PUT), the head is replaced."""
    from reliquary.infrastructure.storage import upload_bytes

    async def put(key: str, body: bytes, content_type: str, cache_control: str) -> None:
        await upload_bytes(key, body, content_type=content_type, cache_control=cache_control,
                           if_absent=cache_control == SEGMENT_CACHE, bucket_name=bucket)
    return put


def r2_get(bucket: str):
    from reliquary.infrastructure.storage import download_bytes

    async def get(key: str) -> bytes | None:
        return await download_bytes(key, bucket_name=bucket)
    return get


class ObservationPublisher:
    def __init__(self, runtime, *, run_id: str, task_id: str, wallet, put, get=None, clock=time.time):
        self.runtime, self.run_id, self.task_id, self.wallet, self.put, self.get, self.clock = runtime, run_id, task_id, wallet, put, get, clock
        # After a restart the last index PUT is unknown: write it once on the first flush.
        self._index_dirty = True
        self._checked = get is None          # the remote head is compared once per process, before any upload
        self.disabled = False
        self._pages_written: set[int] = set()
        self._page_entries: dict[int, dict] = {}     # k -> head entry of closed page k (built once per process)
        self._page_pending: dict[int, bytes] = {}    # k -> bytes not yet uploaded
        self._remote_failures = 0
        self._calls = threading.Lock()       # held by every thread call: close() waits for the running one
        self._closed = False

    def _call(self, fn, *args, **kwargs):
        with self._calls:
            if self._closed:
                raise RuntimeError("observation publisher is closed")
            return fn(*args, **kwargs)

    async def _thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(self._call, fn, *args, **kwargs)

    async def close(self, timeout: float = 30.0) -> None:
        """Return once no thread call is running and none can start any more. Never gives up: the caller
        closes the runtime right after, which must not happen under a running thread."""
        def seal():
            with self._calls:
                self._closed = True
        sealing = asyncio.ensure_future(asyncio.to_thread(seal))
        while True:
            done, _ = await asyncio.wait({sealing}, timeout=timeout)
            if done:
                return
            logger.error("observation publisher still busy after %.0f s; still waiting", timeout)

    def _disable(self, why: str) -> None:
        self.disabled = True
        logger.critical("observation publication DISABLED for this process: %s (operator: start a new run id)", why)

    def _events(self, plan: dict) -> list[dict]:
        count = plan["last_seq"] - plan["first_seq"] + 1
        rows = self.runtime.events(after=plan["first_seq"] - 1, limit=count)
        return public_events([e for seq, e in rows if seq <= plan["last_seq"]], flush_at=plan["flush_at"])

    async def _check_remote(self) -> None:
        """Compare the published head and pages with the local journal: a restored (older) database must
        never overwrite what miners already saw. A missing head is a fresh run. Network errors propagate
        (retry, and the check is not marked done); only a completed check (or a disable) sets ``_checked``."""
        try:
            await self._compare_remote()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._remote_failures += 1
            if self._remote_failures % CONSECUTIVE_REMOTE_FAILURES_CRITICAL == 0:
                logger.critical("cannot read the published observation index (%d consecutive failures); "
                                "still retrying, nothing is published meanwhile", self._remote_failures)
            raise
        self._remote_failures = 0
        self._checked = True

    async def _compare_remote(self) -> None:
        raw = await self.get(index_key(self.run_id))
        if raw is None:
            return
        validator = self.wallet.hotkey.ss58_address
        try:
            remote = verify_index(raw, validator, expected_run_id=self.run_id)
        except ValueError as exc:
            return self._disable(f"the published index does not verify ({exc})")
        if remote["kind"] != "head":
            return self._disable("the published index is not a head")
        local_last = await self._thread(self.runtime.last_published_number)
        if remote["last_number"] > local_last:
            return self._disable(f"the published index is ahead of the local journal ({remote['last_number']} > {local_last})")
        if remote["segments"]:
            local = {s["number"]: s for s in await self._thread(
                self.runtime.published_segments, first_number=remote["segments"][0]["number"],
                last_number=remote["segments"][-1]["number"])}
            for e in remote["segments"]:
                if local.get(e["number"], {}).get("sha256") != e["sha256"]:
                    return self._disable(f"published segment {e['number']} differs from the local journal")
        for entry in remote["pages"]:
            page_raw = await self.get(entry["key"])
            if page_raw is None:
                return self._disable(f"the published head lists page {entry['key']} which is missing")
            try:
                verify_page_entry(page_raw, entry)
                page = verify_index(page_raw, validator, expected_run_id=self.run_id)
            except ValueError as exc:
                return self._disable(f"published page {entry['key']} does not verify ({exc})")
            if page["kind"] != "page" or page["first_number"] != entry["first_number"]:
                return self._disable(f"published page {entry['key']} is not the page the head lists")
            local = {s["number"]: s for s in await self._thread(
                self.runtime.published_segments, first_number=page["first_number"], last_number=page["last_number"])}
            for e in page["segments"]:
                if local.get(e["number"], {}).get("sha256") != e["sha256"]:
                    return self._disable(f"published segment {e['number']} (page) differs from the local journal")
            # Keep the remote bytes when the local journal never stored this page (signatures are not reproducible).
            stored = await self._thread(self.runtime.index_page, entry["first_number"], lambda: page_raw)
            if stored != page_raw:
                return self._disable(f"local page {entry['key']} differs from the published one")

    def _page_bytes(self, k: int) -> bytes:
        first = k * PAGE_SEGMENTS + 1

        def build() -> bytes:
            segments = self.runtime.published_segments(first_number=first, last_number=first + PAGE_SEGMENTS - 1)
            return sign_document(build_page(run_id=self.run_id, validator=self.wallet.hotkey.ss58_address,
                                            segments=segments), self.wallet)
        return self.runtime.index_page(first, build)

    def _closed_page_entries(self, closed: int) -> list[dict]:
        """Head entries of the first ``closed`` pages. Each page is read/built once per process; its bytes
        are kept only until uploaded."""
        for k in range(closed):
            if k in self._page_entries:
                continue
            body = self._page_bytes(k)
            first = k * PAGE_SEGMENTS + 1
            self._page_entries[k] = {"key": page_key(self.run_id, first, first + PAGE_SEGMENTS - 1),
                                     "first_number": first, "last_number": first + PAGE_SEGMENTS - 1,
                                     "sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
            if k not in self._pages_written:
                self._page_pending[k] = body
        return [self._page_entries[k] for k in range(closed)]

    def _head(self, last: int) -> tuple[bytes, dict]:
        closed = last // PAGE_SEGMENTS
        pages = self._closed_page_entries(closed)
        segments = self.runtime.published_segments(first_number=closed * PAGE_SEGMENTS + 1, last_number=last)
        head = build_head(run_id=self.run_id, order_sha256=self.runtime.contract.sha256,
                          validator=self.wallet.hotkey.ss58_address, segments=segments, pages=pages,
                          last_number=last, generated_at=self.clock())
        return sign_document(head, self.wallet), head

    async def flush(self) -> dict | None:
        """Publish the next segment (if any), the closed pages, then the head. None when nothing was written."""
        if self.disabled:
            return None
        if not self._checked:
            await self._check_remote()
            if self.disabled:
                return None
        plan = await self._thread(self.runtime.plan_segment, max_events=SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS,
                                  now=self.clock())
        if plan is not None:
            events = await self._thread(self._events, plan)
            body = await asyncio.to_thread(encode_segment, events)
            await self.put(segment_key(self.run_id, plan["number"]), body, "application/gzip", SEGMENT_CACHE)
            windows = sorted({e["window"] for e in events})
            checkpoints = sorted({e["checkpoint_n"] for e in events if "checkpoint_n" in e})
            await self._thread(self.runtime.commit_segment, plan["number"],
                               sha256=hashlib.sha256(body).hexdigest(), size=len(body),
                               windows=windows[:1] + windows[-1:], checkpoints=checkpoints[:1] + checkpoints[-1:])
            self._index_dirty = True
        if not self._index_dirty:
            return None
        last = await self._thread(self.runtime.last_published_number)
        if not last:
            self._index_dirty = False
            return None
        # Signing and canonicalisation run off the event loop. Pages first: the head references them.
        head_bytes, head = await self._thread(self._head, last)
        for k in sorted(self._page_pending):
            await self.put(self._page_entries[k]["key"], self._page_pending[k], "application/json", SEGMENT_CACHE)
            self._pages_written.add(k)
            del self._page_pending[k]
        await self.put(index_key(self.run_id), head_bytes, "application/json", INDEX_CACHE)
        self._index_dirty = False
        return head

    async def run(self, stop: asyncio.Event, interval: float = SERVICE_OBSERVATION_FLUSH_SECONDS) -> None:
        while not stop.is_set() and not self.disabled:
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
