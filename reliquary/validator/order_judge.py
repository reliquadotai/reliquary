"""The order control's judge in its own process (``RELIQUARY_ORDER_SPLIT_JUDGE=1``).

The front keeps HTTP, intake, ledgers, leases, the auction, the audit dispatcher
and its executor routes, and episode orders whole. A supervised child
(``order_judge_process``) runs each non-episode order's auditor, settler and
miner book on its own store connections and threads. Three links, all on unix
sockets in a private directory:

- accepted ids, front to judge: ``JudgeLink``/``ArrivalFeed`` (``corpus_feed``),
  whose coverage is the auditor's ``arrivals_covered``;
- the sync, front to judge: the orders to judge, their caps and prices, the
  coldkeys; its answer is the snapshot everything the front reads of a judged
  order comes from. A stale or missing snapshot reads as unknown: the order is
  held and nothing is sized, opened or drained on it;
- audits, judge to front: one framed stream carrying ``score`` requests and the
  dispatcher's quarantine events, through which ``RemotePoolView`` stands for
  ``PoolView``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import stat
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_feed import JudgeLink
from reliquary.validator.corpus_split import (
    HEALTHY_SECONDS,
    MAX_RESTART_BACKOFF_SECONDS,
    RESTART_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)

SPLIT_ENV = "RELIQUARY_ORDER_SPLIT_JUDGE"
DIR_ENV = "RELIQUARY_ORDER_JUDGE_DIR"
JUDGE_SOCKET = "judge.sock"
DISPATCH_SOCKET = "dispatch.sock"
LOCK_FILE = "judge.lock"
MAX_SOCKET_PATH = 108
SYNC_SECONDS = 2.0
SYNC_TIMEOUT_SECONDS = 5.0
# Past this the front knows nothing of the judge: its orders are held. A dead
# judge is seen at once; this is for one that hangs, and must outlast a busy front's loop.
SNAPSHOT_STALE_SECONDS = 30.0
# A child alive that answered no sync for this long, over at least HUNG_FAILURES tries, is killed and restarted.
# Counted in failed tries, so a front whose own loop stalled does not kill a healthy judge on waking.
HUNG_SECONDS = 120.0
HUNG_FAILURES = 5
# A verdict being written when its executor was quarantined is listed after
# the event: the event is delivered once more this much later.
RENOTIFY_SECONDS = 60.0
_HEADER = struct.Struct(">I")
# Frames above this are (de)serialized off the event loop.
OFF_LOOP_BYTES = 64 * 1024


def order_judge_enabled(env: Mapping[str, str] | None = None) -> bool:
    return (os.environ if env is None else env).get(SPLIT_ENV) == "1"


def private_run_dir(path: str | Path | None = None) -> Path:
    """The sockets' directory: ours alone, or refused."""
    run_dir = Path(path or os.environ.get(DIR_ENV) or f"/tmp/reliquary-order-judge-{os.getuid()}")
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = os.lstat(run_dir)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(f"{run_dir} must be a directory only this user can open")
    if len(os.fsencode(run_dir / DISPATCH_SOCKET)) >= MAX_SOCKET_PATH:
        raise RuntimeError(f"{run_dir} is too long for a unix socket path; set {DIR_ENV}")
    return run_dir


class JudgeUnavailable(RuntimeError):
    """The judge process cannot answer now."""


# -- the audit stream -------------------------------------------------------


def encode_frame(document: Any) -> bytes:
    data = json.dumps(document, separators=(",", ":")).encode()
    return _HEADER.pack(len(data)) + data


async def encode_frame_off_loop(document: Any) -> bytes:
    return await asyncio.to_thread(encode_frame, document)


async def read_frame(reader: asyncio.StreamReader) -> Any:
    size = _HEADER.unpack(await reader.readexactly(_HEADER.size))[0]
    data = await reader.readexactly(size)
    return await asyncio.to_thread(json.loads, data) if size > OFF_LOOP_BYTES else json.loads(data)


def scores_to_wire(scored) -> list:
    return [[row[0], [[c.exp_mismatches, c.mant_err_mean, c.mant_err_median] for c in row[1]],
             None if row[2] is None else list(row[2]), *row[3:]] for row in scored]


def scores_from_wire(rows) -> list:
    return [(row[0], tuple(ChunkResult(int(e), float(m), float(d)) for e, m, d in row[1]),
             None if row[2] is None else tuple(row[2]), *row[3:]) for row in rows]


class _Connection:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self._drain = asyncio.Lock()

    async def send(self, frame: bytes) -> None:
        self.writer.write(frame)
        async with self._drain:
            await self.writer.drain()


class DispatchServer:
    """The front's end of the audit stream: scores on the real dispatcher, and
    tells the judge of every quarantine. A judge that goes away has its waiting
    batches cancelled, so no card scores for nobody."""

    def __init__(self, dispatcher, path: str | Path, wiring: Callable[[str], Any]) -> None:
        self._dispatcher = dispatcher
        self.path = str(path)
        self._wiring = wiring
        self._connections: set[_Connection] = set()
        self._server: asyncio.AbstractServer | None = None
        self._events: set[asyncio.Task] = set()

    async def start(self) -> None:
        Path(self.path).unlink(missing_ok=True)
        self._dispatcher.subscribe(self._quarantined)
        self._dispatcher.subscribe_unparked(self._unparked)
        self._server = await asyncio.start_unix_server(self._serve, path=self.path)
        os.chmod(self.path, 0o600)

    async def close(self) -> None:
        self._dispatcher.unsubscribe(self._quarantined)
        self._dispatcher.unsubscribe_unparked(self._unparked)
        for task in list(self._events):
            task.cancel()
        if self._server is not None:
            self._server.close()
            self._server = None
        for connection in list(self._connections):
            connection.writer.close()

    async def _quarantined(self, executor_id: str) -> None:
        frame = encode_frame({"event": "quarantined", "executor": executor_id})
        for connection in list(self._connections):
            try:
                await connection.send(frame)
            except (ConnectionError, RuntimeError):
                logger.warning("order judge: quarantine of %s not delivered; its reconnect replays it",
                               executor_id)

    def _unparked(self, job_id: str, batch_id: int) -> None:
        """A batch parked on its losses got a scored answer: its records, in the judge, are tried again at once."""
        frame = encode_frame({"event": "unparked", "job": job_id, "batch": batch_id})
        for connection in list(self._connections):
            task = asyncio.ensure_future(self._send_event(connection, frame))
            self._events.add(task)
            task.add_done_callback(self._events.discard)

    @staticmethod
    async def _send_event(connection: "_Connection", frame: bytes) -> None:
        try:
            await connection.send(frame)
        except (ConnectionError, RuntimeError):
            logger.warning("order judge: an unpark was not delivered; its records are tried after their cool-down")

    async def _serve(self, reader, writer) -> None:
        connection = _Connection(writer)
        self._connections.add(connection)
        tasks: dict[int, asyncio.Task] = {}
        try:
            # Before any score: what was quarantined while no judge listened.
            await connection.send(encode_frame({
                "event": "hello", "quarantined": sorted(self._dispatcher.quarantined)}))
            while True:
                try:
                    request = await read_frame(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    return
                request_id, job_id = request["id"], str(request["job"])
                task = asyncio.ensure_future(self._score(connection, request_id, job_id, request["items"]))
                tasks[request_id] = task
                task.add_done_callback(lambda _, key=request_id: tasks.pop(key, None))
        finally:
            self._connections.discard(connection)
            for task in list(tasks.values()):
                task.cancel()
            await asyncio.gather(*list(tasks.values()), return_exceptions=True)
            writer.close()

    async def _score(self, connection: _Connection, request_id: int, job_id: str, items: list) -> None:
        from reliquary.corpus.eval_draw import has_draw
        from reliquary.validator.eval_control import BatchParked, ItemsUnreadable

        reply: dict[str, Any] = {"id": request_id}
        batches: list = []
        try:
            w = self._wiring(job_id)
            if w is None:
                raise RuntimeError(f"order {job_id} is not served here")
            job = w.job
            scored = await self._dispatcher.score(
                (job.checkpoint_repo, job.checkpoint_revision), w.proof, items, job_id,
                job.judge if has_draw(job) else None, batches=batches)
            reply["scores"] = scores_to_wire(scored)
        except ItemsUnreadable as exc:
            reply.update(error="unreadable", reasons=[[k, why] for k, why in sorted(exc.reasons.items())])
        except BatchParked as exc:
            scored = sorted(exc.scored.items())
            reply.update(error="parked", message=str(exc), items=exc.indexes, batches=exc.batch_ids,
                         scored=[[k, row] for k, row in zip((k for k, _ in scored),
                                                            scores_to_wire([row for _, row in scored]))])
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                # The judge is gone: no card scores this request's other batches for nobody.
                for batch in batches:
                    if not batch.future.done():
                        batch.future.cancel()
                raise
            # The batch alone was cancelled: its order was unwired.
            reply.update(error="runtime", message=f"the batch of order {job_id} was cancelled")
        except Exception as exc:  # noqa: BLE001 - told to the auditor, as raised in-process
            reply.update(error="value" if isinstance(exc, ValueError) else "runtime",
                         message=f"{type(exc).__name__}: {exc}")
        try:
            await connection.send(await encode_frame_off_loop(reply))
        except (ConnectionError, RuntimeError):
            logger.info("order judge: score reply of %s not delivered (judge gone)", job_id)


class _StreamLost(Exception):
    """The stream dropped with this request unanswered."""


class DispatchClient:
    """The judge's end of the audit stream, reconnecting for ever. A ``score``
    in flight when it drops is sent again: an audit only takes longer, and is
    never a validator-side error."""

    def __init__(self, path: str | Path, *, retry_seconds: float = 0.2,
                 renotify_seconds: float = RENOTIFY_SECONDS) -> None:
        self.path = str(path)
        self._retry = retry_seconds
        self._renotify = renotify_seconds
        self._ids = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._connection: _Connection | None = None
        self._up: asyncio.Event | None = None
        self._views: dict[str, RemotePoolView] = {}
        self.quarantined: set[str] = set()
        self._background: set[asyncio.Task] = set()

    def _up_event(self) -> asyncio.Event:
        if self._up is None:
            self._up = asyncio.Event()
        return self._up

    def connected(self) -> bool:
        return self._connection is not None

    def view(self, job_id: str) -> "RemotePoolView":
        view = RemotePoolView(self, str(job_id))
        self._views[str(job_id)] = view
        return view

    def forget(self, job_id: str) -> None:
        self._views.pop(str(job_id), None)

    async def score(self, job_id: str, items) -> list:
        frame_body = await asyncio.to_thread(json.dumps, items, separators=(",", ":"))
        while True:
            await self._up_event().wait()
            connection = self._connection
            if connection is None:
                continue
            self._ids += 1
            request_id = self._ids
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            head = json.dumps({"id": request_id, "job": job_id}, separators=(",", ":"))
            data = f'{head[:-1]},"items":{frame_body}}}'.encode()
            try:
                await connection.send(_HEADER.pack(len(data)) + data)
                reply = await future
            except (_StreamLost, ConnectionError):
                continue
            finally:
                self._pending.pop(request_id, None)
            return self._result(reply)

    @staticmethod
    def _result(reply: Mapping) -> list:
        from reliquary.validator.eval_control import BatchParked, ItemsUnreadable

        error = reply.get("error")
        if error == "unreadable":
            raise ItemsUnreadable({int(k): str(why) for k, why in reply.get("reasons") or ()})
        if error == "parked":
            scored = reply.get("scored") or ()
            rows = scores_from_wire([row for _, row in scored])
            raise BatchParked(reply.get("message", "parked"), indexes=reply.get("items"),
                              batch_ids=reply.get("batches") or (),
                              scored=dict(zip((int(k) for k, _ in scored), rows)))
        if error == "value":
            raise ValueError(reply.get("message", "refused"))
        if error is not None:
            raise RuntimeError(reply.get("message", "audit failed"))
        return scores_from_wire(reply["scores"])

    def _notify(self, executor_id: str) -> None:
        for view in list(self._views.values()):
            for listener in list(view.listeners):
                task = asyncio.ensure_future(self._call(listener, executor_id))
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    @staticmethod
    async def _call(listener, executor_id: str) -> None:
        try:
            await listener(executor_id)
        except Exception:
            logger.exception("re-audit after quarantining %s failed", executor_id)

    def _unparked(self, job_id: str, batch_id) -> None:
        view = self._views.get(job_id)
        for listener in list(view.unpark_listeners if view is not None else ()):
            try:
                listener(batch_id)
            except Exception:
                logger.exception("order judge: unpark of batch %s failed", batch_id)

    def _quarantine(self, executor_id: str) -> None:
        if executor_id in self.quarantined:
            return
        self.quarantined.add(executor_id)
        self._notify(executor_id)
        # What a pass was writing from its scores at the event is listed only after it.
        asyncio.get_running_loop().call_later(self._renotify, self._notify, executor_id)

    async def run(self) -> None:
        try:
            while True:
                try:
                    reader, writer = await asyncio.open_unix_connection(self.path)
                except OSError:
                    await asyncio.sleep(self._retry)
                    continue
                connection = _Connection(writer)
                self._connection = connection
                self._up_event().set()
                try:
                    while True:
                        message = await read_frame(reader)
                        if "event" in message:
                            if message["event"] == "hello":
                                for executor_id in message.get("quarantined", ()):
                                    self._quarantine(executor_id)
                            elif message["event"] == "quarantined":
                                self._quarantine(message["executor"])
                            elif message["event"] == "unparked":
                                self._unparked(str(message["job"]), message["batch"])
                            continue
                        future = self._pending.get(message.get("id"))
                        if future is not None and not future.done():
                            future.set_result(message)
                except (asyncio.IncompleteReadError, ConnectionError):
                    logger.warning("order judge: the audit stream dropped; reconnecting")
                finally:
                    self._connection = None
                    self._up_event().clear()
                    writer.close()
                    for future in list(self._pending.values()):
                        if not future.done():
                            future.set_exception(_StreamLost())
        finally:
            for task in list(self._background):
                task.cancel()


class RemotePoolView:
    """``PoolView`` over the audit stream: what ``CorpusAuditor`` uses of a
    remote dispatcher (``score``, ``subscribe``, ``connected``, ``close``)."""

    def __init__(self, client: DispatchClient, job_id: str) -> None:
        self._client = client
        self._job_id = job_id
        self.listeners: list = []
        self.unpark_listeners: list = []

    def connected(self) -> bool:
        return self._client.connected()

    def subscribe(self, listener) -> None:
        self.listeners.append(listener)

    def subscribe_unparked(self, listener) -> None:
        """``listener(batch_id)``: a batch of this order parked on its losses got a scored answer."""
        self.unpark_listeners.append(listener)

    def quarantined(self) -> set[str]:
        """Every executor the front has told this process is quarantined."""
        return set(self._client.quarantined)

    def close(self) -> None:
        self.listeners.clear()
        self.unpark_listeners.clear()
        self._client.forget(self._job_id)

    async def score(self, items) -> list:
        return await self._client.score(self._job_id, items)


# -- the front's view of a judged order --------------------------------------


class _Seeded:
    def __init__(self, front: "OrderJudgeFront", job_id: str) -> None:
        self._front, self._job_id = front, job_id

    def is_set(self) -> bool:
        return self._front.live(self._job_id)


class _RemoteHandle:
    """``w.auditor._remote`` as the job set's unwiring closes it."""

    def __init__(self, front: "OrderJudgeFront", job_id: str) -> None:
        self._front, self._job_id = front, job_id

    def close(self) -> None:
        self._front.forget(self._job_id)


class JudgedAuditor:
    """What the front reads of an order's auditor, from the last snapshot.
    Unknown (stale, not seeded): None where the reader takes it, else raises."""

    settled_count = None
    totals = None

    def __init__(self, front: "OrderJudgeFront", records, job_id: str, *, reopen_slots: bool) -> None:
        self._front, self._records, self._job_id = front, records, job_id
        self.seeded = _Seeded(front, job_id)
        self._remote = _RemoteHandle(front, job_id)
        self.reopen_slots = reopen_slots

    def _seeded_view(self) -> Mapping | None:
        view = self._front.snapshot_of(self._job_id)
        return view if view is not None and view.get("seeded") else None

    def _known(self) -> Mapping:
        view = self._seeded_view()
        if view is None:
            raise LookupError(f"the judge of {self._job_id} has no fresh snapshot")
        return view

    def audit_backlog(self) -> int:
        return int(self._known()["audit_backlog"])

    def audit_share(self) -> float:
        return float(self._known()["audit_share"])

    def audit_decisions(self) -> int:
        return int(self._known()["audit_decisions"])

    def audits_pending(self) -> int | None:
        view = self._seeded_view()
        return None if view is None else view["audits_pending"]

    def expected_audits(self, pending: int | None) -> float | None:
        if pending is None:
            return None
        view = self._seeded_view()
        return None if view is None else view["expected_audits"]

    @property
    def audits_idle_since(self) -> float | None:
        view = self._seeded_view()
        return None if view is None else view["audits_idle_since"]

    @property
    def audit_reasons(self) -> dict:
        view = self._seeded_view()
        return {} if view is None else dict(view["audit_reasons"])

    @property
    def ahead_wasted(self) -> int:
        view = self._seeded_view()
        return 0 if view is None else int(view["ahead_wasted"])

    async def pending_ids(self) -> list[str]:
        """For the drain check: never answered without a live, seeded judge."""
        self._known()
        submitted = await self._records.list_submission_ids(self._job_id)
        judged = set(await self._records.list_verdict_ids(self._job_id))
        return [sid for sid in submitted if sid not in judged]


class JudgedSettler:
    """What the front reads of an order's settler. The caps and prices reach
    the real one with the sync."""

    settled_count = None
    totals = None

    def __init__(self, front: "OrderJudgeFront", job_id: str) -> None:
        self._front, self._job_id = front, job_id

    def _view(self) -> Mapping | None:
        view = self._front.snapshot_of(self._job_id)
        return view if view is not None and view.get("seeded") else None

    @property
    def sizing_known(self) -> bool:
        return self._view() is not None

    @property
    def recent_paid(self) -> list[float]:
        view = self._view()
        return [] if view is None else list(view["recent_paid"])

    def open_value(self) -> float | None:
        view = self._view()
        return None if view is None else view["open_value"]

    def set_cap(self, cap: float) -> None:
        return None

    def set_max_price(self, price) -> None:
        return None

    def set_max_input_price(self, price) -> None:
        return None


class OrderJudgeLink(JudgeLink):
    """``JudgeLink`` whose status reads say so when the judge cannot answer."""

    async def stats(self, job_id: str) -> dict:
        return await self._read(super().stats(job_id), job_id)

    async def miner(self, job_id: str, hotkey: str) -> dict:
        return await self._read(super().miner(job_id, hotkey), job_id)

    @staticmethod
    async def _read(call, job_id: str) -> dict:
        import httpx

        try:
            judged = await call
        except (httpx.HTTPError, OSError) as exc:
            raise JudgeUnavailable(f"the judge of {job_id}: {type(exc).__name__}") from exc
        if judged is None:
            raise JudgeUnavailable(f"the judge does not serve {job_id} yet")
        return judged


def entry_to_wire(entry) -> dict:
    return {f.name: getattr(entry, f.name) for f in fields(entry)}


def _wire_entry(entry) -> tuple[str, dict]:
    document = json.loads(json.dumps(entry_to_wire(entry), default=dict, sort_keys=True))
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(), document


@dataclass
class OrderJudgeConfig:
    """What the child needs, picklable."""

    run_dir: str
    parent_pid: int
    child_init: str | None = None
    settle_seconds: float = 60.0
    auditor_kwargs: dict | None = None


class OrderJudgeFront:
    """The front's side of the judge process: the orders it judges, the sync
    and its snapshot, the audit stream, and the child's supervision."""

    def __init__(self, *, dispatcher, run_dir: str | Path, coldkeys: Callable[[], Mapping] | None = None,
                 clock: Callable[[], float] = time.monotonic, stale_seconds: float = SNAPSHOT_STALE_SECONDS,
                 sync_seconds: float = SYNC_SECONDS, spawn: bool = True, child_init: str | None = None,
                 settle_seconds: float = 60.0, auditor_kwargs: dict | None = None,
                 backoff_seconds: float = RESTART_BACKOFF_SECONDS, hung_seconds: float = HUNG_SECONDS,
                 hung_failures: int = HUNG_FAILURES) -> None:
        self.run_dir = private_run_dir(run_dir)
        self._dispatcher = dispatcher
        self._coldkeys = coldkeys
        self._clock = clock
        self._stale = stale_seconds
        self._sync_every = sync_seconds
        self._spawn = spawn
        self._child_init = child_init
        self._settle_seconds = settle_seconds
        self._auditor_kwargs = auditor_kwargs
        self._backoff = backoff_seconds
        self._hung_seconds = hung_seconds
        self._hung_failures = hung_failures
        # Since when no sync has been answered, and how many tries failed since.
        self._unanswered_since: float | None = None
        self._unanswered = 0
        self.link = OrderJudgeLink(self.run_dir / JUDGE_SOCKET, [])
        self.dispatch = DispatchServer(dispatcher, self.run_dir / DISPATCH_SOCKET, self.wiring_of)
        self._jobs: dict[str, Any] = {}
        self._latest: dict[str, Any] = {}
        self._wire_docs: dict[str, tuple[Any, str, dict]] = {}
        self._snapshot: Mapping | None = None
        self._snapshot_at = float("-inf")
        self._boot: str | None = None
        self._state_writes: dict[str, int] = {}
        # The front's job store, and the judge's ledger writes as last reported per order.
        self._job_store = None
        self._ledger_writes: dict[str, int] = {}
        # Bumped when the judge is seen gone: a reply already in flight then proves nothing.
        self._downs = 0
        self.process = None
        self.starts = 0
        self._stopping = False

    @classmethod
    def from_env(cls, *, dispatcher, coldkeys=None) -> "OrderJudgeFront":
        return cls(dispatcher=dispatcher, run_dir=private_run_dir(), coldkeys=coldkeys)

    # -- the orders ----------------------------------------------------------

    def wiring_of(self, job_id: str):
        return self._jobs.get(str(job_id))

    def attach(self, w, *, records, store=None) -> None:
        """A non-episode order: its judge is in the child; every read of it here comes from the snapshot."""
        from reliquary.corpus.eval_draw import has_draw
        from reliquary.eval.prompt_source import is_eval_source

        job_id = str(w.job.job_id)
        w.judge_link = self.link
        w.miners = None
        w.auditor = JudgedAuditor(self, records, job_id,
                                  reopen_slots=is_eval_source(w.job.prompt_source) or has_draw(w.job))
        w.settler = JudgedSettler(self, job_id)

        def on_accepted(submission_id: str) -> None:
            w.stats.accepted()
            self.link.accepted(job_id, submission_id)

        w.on_accepted = on_accepted
        self._jobs[job_id] = w
        if store is not None:
            self._job_store = store

    def forget(self, job_id: str) -> None:
        """The order drained and is unwired: the next sync stops it there."""
        self._jobs.pop(job_id, None)
        self._wire_docs.pop(job_id, None)
        self._state_writes.pop(job_id, None)
        self._ledger_writes.pop(job_id, None)
        self._dispatcher.forget_job(job_id)

    def watching(self, read_entries):
        """The registry reader, remembering the entries: a restarted judge is wired from the latest."""

        async def read():
            entries = await read_entries()
            self._latest = dict(entries)
            return entries

        return read

    # -- the snapshot --------------------------------------------------------

    def snapshot_of(self, job_id: str) -> Mapping | None:
        """The job's part of a fresh snapshot, or None."""
        if self._snapshot is None or self._clock() - self._snapshot_at > self._stale:
            return None
        return (self._snapshot.get("jobs") or {}).get(str(job_id))

    def live(self, job_id: str) -> bool:
        view = self.snapshot_of(job_id)
        return view is not None and bool(view.get("seeded"))

    def mark_down(self) -> None:
        self._snapshot = None
        self._downs += 1
        self._sync_lease_clocks()

    def _sync_lease_clocks(self) -> None:
        """Each order's leases see a hold start or end now, whatever the miners are doing."""
        for job_id, w in list(self._jobs.items()):
            sync = getattr(w, "sync_lease_clock", None)
            if sync is None:
                continue
            try:
                sync()
            except Exception:
                logger.exception("order judge: the lease clock of %s could not be synced", job_id)

    def _wanted(self, job_id: str, w, known: Mapping) -> dict:
        latest = self._latest.get(str(w.entry.task_id))
        entry = latest if latest is not None and str(getattr(latest, "job_id", "")) == job_id else w.entry
        cached = self._wire_docs.get(job_id)
        if cached is None or cached[0] is not entry:
            sha, document = _wire_entry(entry)
            cached = self._wire_docs[job_id] = (entry, sha, document)
        spec: dict[str, Any] = {"sha": cached[1], "cap": float(w.cap)}
        if known.get(job_id) != cached[1]:
            spec["entry"] = cached[2]
        return spec

    def _sync_body(self) -> dict:
        previous = self._snapshot or {}
        known = previous.get("entries") or {}
        body: dict[str, Any] = {"jobs": {job_id: self._wanted(job_id, w, known)
                                         for job_id, w in list(self._jobs.items())}}
        if self._coldkeys is None:
            body["operators"] = {"version": "none", "coldkeys": None}
        else:
            coldkeys = dict(self._coldkeys())
            version = hashlib.sha256(json.dumps(coldkeys, sort_keys=True).encode()).hexdigest()
            body["operators"] = {"version": version}
            if previous.get("operators_version") != version:
                body["operators"]["coldkeys"] = coldkeys
        return body

    def _take(self, reply: Mapping) -> None:
        boot = reply.get("boot")
        new_judge = boot != self._boot
        if new_judge:
            logger.info("order judge: process %s answers (was %s)", boot, self._boot)
            self._boot = boot
        self._forget_ledgers(reply.get("ledger_writes") or {}, new_judge)
        self._snapshot, self._snapshot_at = reply, self._clock()
        for job_id, view in (reply.get("jobs") or {}).items():
            w = self._jobs.get(job_id)
            if w is None:
                continue
            writes = view.get("state_writes")
            if writes is not None and writes != self._state_writes.get(job_id):
                # The judge moved a miner's state: suspicion is seen at the next lease, not 30 s later.
                self._state_writes[job_id] = writes
                w.miner_states.forget_mirror()
            for reason in view.get("attention") or ():
                reasons = self._dispatcher.attention[job_id]
                if reason not in reasons:
                    reasons.append(reason)
                    del reasons[:-10]

    def _forget_ledgers(self, writes: Mapping[str, int], new_judge: bool) -> None:
        """The judge reopened a failed slot: the front's memory of that ledger is stale. A new judge
        may have written before its last one could say so."""
        forget = getattr(self._job_store, "forget_ledgers", None)
        for job_id in self._jobs:
            count = int(writes.get(job_id, 0))
            if new_judge or count != self._ledger_writes.get(job_id, 0):
                self._ledger_writes[job_id] = count
                if forget is not None:
                    forget(job_id)

    async def sync_once(self) -> bool:
        import httpx

        downs = self._downs
        try:
            reply = await self.link._client.post("/sync", self._sync_body(), timeout=SYNC_TIMEOUT_SECONDS)
        except (httpx.ConnectError, FileNotFoundError, ConnectionError):
            self._sync_unanswered()
            self.mark_down()                        # nobody on the socket
            return False
        except Exception as exc:  # noqa: BLE001 - it goes stale, then its orders are held
            self._sync_unanswered()
            logger.warning("order judge: sync failed: %r", exc)
            return False
        if downs != self._downs:
            return False
        self._unanswered_since, self._unanswered = None, 0
        self._take(reply)
        return True

    def _sync_unanswered(self) -> None:
        if self._unanswered_since is None:
            self._unanswered_since = self._clock()
        self._unanswered += 1

    def hung(self) -> bool:
        """The child has answered no sync for ``hung_seconds``, over at least ``hung_failures`` tries."""
        return (self._unanswered_since is not None and self._unanswered >= self._hung_failures
                and self._clock() - self._unanswered_since >= self._hung_seconds)

    async def _sync_forever(self) -> None:
        while True:
            try:
                await self.sync_once()
            except Exception:
                logger.exception("order judge: sync pass failed")
            self._sync_lease_clocks()
            await self._load_quarantined()
            await asyncio.sleep(self._sync_every)

    # -- the child -----------------------------------------------------------

    def _start_child(self) -> None:
        from reliquary.validator.order_judge_process import judge_main

        # From the event loop's thread: PR_SET_PDEATHSIG follows the thread that forked.
        context = multiprocessing.get_context("spawn")
        config = OrderJudgeConfig(run_dir=str(self.run_dir), parent_pid=os.getpid(),
                                  child_init=self._child_init, settle_seconds=self._settle_seconds,
                                  auditor_kwargs=self._auditor_kwargs)
        self.process = context.Process(target=judge_main, args=(config,), name="order-judge", daemon=False)
        self.process.start()
        self.starts += 1
        logger.info("order judge: started (pid %d, start %d)", self.process.pid, self.starts)

    async def _supervise(self) -> None:
        failures = 0
        while not self._stopping:
            self._start_child()
            self._unanswered_since, self._unanswered = None, 0
            started = time.monotonic()
            while self.process.is_alive():
                if self.hung():
                    logger.error("order judge: pid %d answered no sync for %.0f s (%d tries); killing it",
                                 self.process.pid, self._clock() - self._unanswered_since, self._unanswered)
                    self.process.kill()
                    self._unanswered_since, self._unanswered = None, 0
                await asyncio.sleep(0.2)
            code = self.process.exitcode
            self.process.join(0)
            self.mark_down()
            if self._stopping:
                return
            lived = time.monotonic() - started
            failures = 1 if lived >= HEALTHY_SECONDS else failures + 1
            delay = min(MAX_RESTART_BACKOFF_SECONDS, self._backoff * 2 ** (failures - 1))
            logger.error("order judge: exited (code %s) after %.0f s; restarting in %.0f s; its orders are held",
                         code, lived, delay)
            await asyncio.sleep(delay)

    async def _load_quarantined(self) -> None:
        """What the registry says was quarantined, told to the judge as a live quarantine (once per executor)."""
        if not hasattr(self._dispatcher._directory, "quarantined_ids"):
            return
        try:
            await self._dispatcher.load_quarantined()
        except Exception:
            logger.exception("order judge: registry quarantines not loaded; retrying")

    async def run(self) -> None:
        await self.dispatch.start()
        # Before the child: its first hello already names what was quarantined before this restart.
        refresh = getattr(self._dispatcher._directory, "maybe_refresh", None)
        if refresh is not None:
            await refresh()
        await self._load_quarantined()
        workers = [self.link.run(), self._sync_forever()]
        if self._spawn:
            workers.append(self._supervise())
        try:
            await asyncio.gather(*workers)
        finally:
            await self.dispatch.close()

    def stop(self, grace_seconds: float = 20.0) -> None:
        self._stopping = True
        process = self.process
        if process is None or not process.is_alive():
            return
        process.terminate()
        process.join(grace_seconds)
        if process.is_alive():
            logger.warning("order judge: ignored SIGTERM; killing")
            process.kill()
            process.join(5)


__all__ = [
    "DIR_ENV",
    "DispatchClient",
    "DispatchServer",
    "JudgeUnavailable",
    "JudgedAuditor",
    "JudgedSettler",
    "OrderJudgeConfig",
    "OrderJudgeFront",
    "OrderJudgeLink",
    "RemotePoolView",
    "SPLIT_ENV",
    "order_judge_enabled",
    "private_run_dir",
]
