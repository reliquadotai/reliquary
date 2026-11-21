"""The order control's judge in its own process: a real child on a real unix
socket, one on-disk bucket, the front's real dispatcher and a fake executor."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as record_store
from reliquary.corpus.audit_policy import MinerState, drawn
from reliquary.infrastructure import storage
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, TaskEntry
from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditResult
from reliquary.validator.corpus_job_status import JobStats
from reliquary.validator.corpus_miner_states import MinerStates
from reliquary.validator.corpus_service import CorpusJobRoutes
from reliquary.validator.eval_control import HELD_AWAITING_AUDIT, ORDER_TRUST_ID, PairedAuditDispatcher
from reliquary.validator.order_judge import OrderJudgeFront, order_judge_enabled
from tests.unit import corpus_split_harness as h
from tests.unit.corpus_split_fakes import FORGED, PROOF
from tests.unit.test_eval_control_process import world  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401

JOB = "order-gen-a"
HOTKEYS = h.HOTKEYS[:4]
EXECUTOR = {"executor_id": "e1", "model_id": "org/Frozen", "model_revision": "abc123",
            "provider_id": "p1", "host": "h1"}
FAST = {"_accept_slack": 5.0, "_rescan_every": 0.5}
CONST_BEACON = "c3" * 32
SAMPLED = {"audit_q": 0.5, "audit_probation_submissions": 1, "audit_hold_seconds": 600.0}
# The arrivals rule: what a record's stake would add is not under test here.
ARRIVALS = {"_rescan_every": 0.3, "stake_at_receipt": False}


def init_child() -> None:
    """Every judge child: the on-disk bucket and a tokenizer without a download."""
    h.install()
    from reliquary.protocol import profiles
    from reliquary.validator import corpus_validator, eval_control
    from tests.unit import corpus_split_fakes as fakes

    eval_control.load_cpu_tokenizer = lambda repo, revision: (fakes.Tokenizer(), fakes.VOCAB)
    profiles.toploc_proof = lambda profile: fakes.PROOF

    async def no_chain(executor):
        return None

    corpus_validator.warm_drand_chain = no_chain


def _entry(**params) -> TaskEntry:
    return TaskEntry(task_id=JOB, profile_id=ACTIVE_PROTOCOL_PROFILE.profile_id, profile_sha256="0" * 64,
                     mechanism=MECHANISM_CORPUS_GENERATION,
                     params={"cap": 0.1, "audit_q": 1.0, "audit_probation_submissions": 5,
                             "audit_hold_seconds": 60.0, "audit_ban_after_failures": 1000,
                             "settlement": "period-ema-v1", **params},
                     status="active", retired_at=None,
                     contract=ACTIVE_PROTOCOL_PROFILE.to_generation_contract(), job_id=JOB)


def _put(root: Path, *, received: float, forged: bool = False, hotkey: str | None = None,
         rng=random.Random(11)) -> str:
    sid = "%064x" % rng.getrandbits(256)
    h.put_raw(root, record_store._key(JOB, "submissions", sid), h.synth_stub(
        sid=sid, job_id=JOB, hotkey=hotkey or HOTKEYS[rng.randrange(len(HOTKEYS))], received_at=received,
        n=256, forged=forged))
    return sid


def _put_undrawn(root: Path, *, received: float, q: float, hotkey: str | None = None,
                 rng=random.Random(23)) -> str:
    """A record the constant beacon leaves undrawn at ``q``: no audit is ever asked for it."""
    while True:
        sid = "%064x" % rng.getrandbits(256)
        if not drawn(CONST_BEACON, sid, q):
            break
    h.put_raw(root, record_store._key(JOB, "submissions", sid), h.synth_stub(
        sid=sid, job_id=JOB, hotkey=hotkey or HOTKEYS[rng.randrange(len(HOTKEYS))], received_at=received, n=256))
    return sid


def _trust(root: Path) -> None:
    """Every hotkey past probation in the orders' shared trust document."""
    h.put_raw(root, record_store._miners_key(ORDER_TRUST_ID), json.dumps(
        {hotkey: MinerState(audited_passed=1000).to_dict() for hotkey in HOTKEYS}).encode())


def _verdicts(root: Path) -> dict:
    return h.listed(root, JOB, "verdicts")


def _voided(root: Path) -> dict:
    return h.listed(root, JOB, "voided")


async def _until(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class _Executor:
    """Claims from the real dispatcher; honest unless ``lies`` (then a forged completion passes)."""

    def __init__(self, dispatcher, executor_id="e1", *, lies=False) -> None:
        self.dispatcher, self.lies = dispatcher, lies
        self.document = {**EXECUTOR, "executor_id": executor_id, "provider_id": f"p-{executor_id}",
                         "host": f"h-{executor_id}"}
        self.paused = False
        self.scored = 0

    async def run(self) -> None:
        eid = self.document["executor_id"]
        while True:
            self.dispatcher.heartbeat(eid, self.document)
            lease = None if self.paused else self.dispatcher.claim(self.document, (AUDIT_PROTOCOL,))
            if lease is None:
                await asyncio.sleep(0.02)
                continue
            scores = []
            for item in lease["items"]:
                forged = FORGED in item["tokens"][item["prompt_len"]:] and not self.lies
                chunk = [1000, 99.5, 99.25] if forged else [0, 0.0125, 0.01]
                scores.append({"status": "ok", "chunks": [chunk] * len(item["proofs"])})
            await self.dispatcher.result(eid, lease["lease_id"], AuditResult.model_validate({"scores": scores}))
            self.scored += len(scores)


class _Control:
    """The front's half, around a real judge child."""

    def __init__(self, root: Path, run_dir: Path, *, entry=None, auditor=None) -> None:
        self.root = root
        self.dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        self.front = OrderJudgeFront(
            dispatcher=self.dispatcher, run_dir=run_dir, spawn=True, sync_seconds=0.1, stale_seconds=1.5,
            backoff_seconds=0.2, settle_seconds=0.5, auditor_kwargs=auditor or FAST,
            child_init="tests.unit.test_order_judge_process:init_child")
        entry = entry or _entry()
        self.records = record_store.BucketRecordStore()
        self.w = SimpleNamespace(
            entry=entry, cap=float(entry.params["cap"]), job=None, stats=JobStats(),
            proof=PROOF,
            miner_states=MinerStates(self.records, JOB))
        self.task = None
        self.feed_open = True

    async def start(self) -> "_Control":
        self.w.job, _ = await job_store.BucketJobStore().read_job(JOB)
        self.front.attach(self.w, records=self.records)
        flush = self.front.link.flush_once

        async def gated():
            if not self.feed_open:
                return False
            return await flush()

        self.front.link.flush_once = gated
        self.task = asyncio.ensure_future(self.front.run())
        return self

    async def close(self) -> None:
        await asyncio.to_thread(self.front.stop, 5.0)
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    root = tmp_path / "bucket"
    root.mkdir()
    monkeypatch.setenv(h.ROOT_ENV, str(root))
    monkeypatch.setenv(h.LATENCY_ENV, "0,0,0,0,0")
    client = lambda **kw: h.FileS3(root)  # noqa: E731
    for module in (job_store, record_store, storage):
        monkeypatch.setattr(module, "get_s3_client", client)
    h.seed_bucket(root, [JOB], hotkeys=HOTKEYS, audited_passed=0)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    return root, run_dir


def test_the_split_is_off_unless_asked():
    assert not order_judge_enabled({})
    assert not order_judge_enabled({"RELIQUARY_ORDER_SPLIT_JUDGE": "0"})
    assert order_judge_enabled({"RELIQUARY_ORDER_SPLIT_JUDGE": "1"})


def test_an_accepted_id_is_audited_through_the_front_and_settled(bucket):
    root, run_dir = bucket

    async def go():
        control = await _Control(root, run_dir).start()
        executor = _Executor(control.dispatcher)
        scoring = asyncio.ensure_future(executor.run())
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            sid = _put(root, received=time.time() - 40)
            control.front.link.accepted(JOB, sid)
            await _until(lambda: sid in _verdicts(root), 30, "the verdict")
            verdict = _verdicts(root)[sid]
            assert verdict["passed"] and verdict["audited"] and verdict["scored_by"] == ["e1"]
            assert executor.scored >= 1
            await _until(lambda: sid in (h.settlement(root, JOB).get("settled") or ()), 60, "its settlement")
            stats = await control.front.link.stats(JOB)
            assert stats["settled_count"] >= 1
        finally:
            scoring.cancel()
            await control.close()

    asyncio.run(go())


def test_no_unaudited_pass_before_coverage_even_across_restarts(bucket, monkeypatch):
    root, run_dir = bucket
    monkeypatch.setenv(h.BEACON_ENV, "const")
    _trust(root)
    hold = SAMPLED["audit_hold_seconds"]

    async def go():
        now = time.time()
        # Past receipt + hold + the real 420 s slack: only the feed's coverage keeps them unpaid.
        sids = [_put_undrawn(root, received=now - hold - 430 - k * 0.01, q=0.5, hotkey=HOTKEYS[k % 4])
                for k in range(8)]
        for hotkey in HOTKEYS:   # enough arrivals inside the hold: off the slow-hotkey rule
            for k in range(3):
                _put_undrawn(root, received=now - 100 - k, q=0.5, hotkey=hotkey)
        control = _Control(root, run_dir, entry=_entry(**SAMPLED), auditor=ARRIVALS)
        control.feed_open = False
        await control.start()
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            await asyncio.sleep(2.0)
            assert not (set(sids) & set(_verdicts(root))), "passed with no coverage from the front"
            # A judge restart: still nothing, whatever it re-read.
            first = control.front.process.pid
            os.kill(first, signal.SIGKILL)
            await _until(lambda: not control.front.live(JOB), 5, "the hold")
            await _until(lambda: control.front.process.pid != first and control.front.live(JOB), 90, "the restart")
            await asyncio.sleep(2.0)
            assert not (set(sids) & set(_verdicts(root))), "passed with no coverage after a judge restart"
            # A new front's epoch: covered once its listing is done, and the passes land.
            control.front.link.epoch = "another-front"
            control.feed_open = True
            await _until(lambda: set(sids) <= set(_verdicts(root)), 60, "the unaudited passes")
            for sid in sids:
                verdict = _verdicts(root)[sid]
                assert verdict["passed"] and not verdict["audited"]
        finally:
            await control.close()

    asyncio.run(go())


def test_a_judge_killed_mid_pass_is_held_restarted_and_loses_nothing(bucket):
    root, run_dir = bucket

    async def go():
        now = time.time()
        control = await _Control(root, run_dir).start()
        executor = _Executor(control.dispatcher)
        executor.paused = True
        scoring = asyncio.ensure_future(executor.run())
        routes = CorpusJobRoutes({JOB: object()})
        routes.hold(JOB, lambda: control.front.live(JOB), why=lambda: HELD_AWAITING_AUDIT, latch=False)
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            assert not routes.is_held(JOB)
            sids = [_put(root, received=now - 40 - k * 0.01) for k in range(24)]
            for sid in sids:
                control.front.link.accepted(JOB, sid)
            await _until(lambda: control.dispatcher.pending_items(JOB) > 0, 20, "a batch in flight")
            first = control.front.process.pid
            os.kill(first, signal.SIGKILL)
            await _until(lambda: routes.is_held(JOB), 3, "the hold")
            assert routes.held_reason(JOB) == HELD_AWAITING_AUDIT
            # Its batches are withdrawn: no card scores for a judge that is gone.
            await _until(lambda: control.dispatcher.pending_items(JOB) == 0, 5, "the batches' cancellation")
            # Accepted while it is down: kept for the next one.
            late = _put(root, received=time.time() - 35)
            control.front.link.accepted(JOB, late)
            await _until(lambda: control.front.process.pid != first and control.front.live(JOB), 90, "the restart")
            assert not routes.is_held(JOB)
            executor.paused = False
            wanted = {*sids, late}
            await _until(lambda: wanted <= set(_verdicts(root)), 60, "every verdict")
            await _until(lambda: wanted <= set(h.settlement(root, JOB).get("settled") or ()), 60, "every settlement")
            log = root / "archives" / "writes.jsonl"
            windows = [(row["task"], row["window"]) for row in map(json.loads, log.read_text().splitlines())]
            assert len(windows) == len(set(windows)), "a period archived twice"
            settled = h.settlement(root, JOB)["settled"]
            assert len(settled) == len(set(settled))
        finally:
            scoring.cancel()
            await control.close()

    asyncio.run(go())


def test_a_quarantined_executor_is_re_audited_through_the_remote_view(bucket):
    root, run_dir = bucket

    async def go():
        control = await _Control(root, run_dir).start()
        liar = _Executor(control.dispatcher, "liar", lies=True)
        lying = asyncio.ensure_future(liar.run())
        honest_task = None
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            forged = _put(root, received=time.time() - 30, forged=True, hotkey=HOTKEYS[0])
            control.front.link.accepted(JOB, forged)
            await _until(lambda: forged in _verdicts(root), 30, "the lying verdict")
            assert _verdicts(root)[forged]["passed"] and _verdicts(root)[forged]["scored_by"] == ["liar"]
            lying.cancel()
            honest = _Executor(control.dispatcher, "honest")
            honest_task = asyncio.ensure_future(honest.run())
            # The pass's bookkeeping lands right after its verdict.
            await asyncio.sleep(0.5)
            await control.dispatcher.quarantine("liar", "test")
            await _until(lambda: forged in _voided(root), 30, "the re-audit's void")
            assert _voided(root)[forged]["executor_id"] == "liar"
            assert honest.scored >= 2      # the failure is confirmed by a second audit
        finally:
            lying.cancel()
            if honest_task is not None:
                honest_task.cancel()
            await control.close()

    asyncio.run(go())


def test_a_stale_or_missing_snapshot_holds_and_never_drains(bucket):
    root, run_dir = bucket

    async def go():
        clock = SimpleNamespace(t=100.0)
        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=False, clock=lambda: clock.t,
                                stale_seconds=10.0)
        records = record_store.BucketRecordStore()
        w = SimpleNamespace(entry=_entry(), cap=0.1, job=(await job_store.BucketJobStore().read_job(JOB))[0],
                            stats=JobStats(), proof=PROOF,
                            miner_states=MinerStates(records, JOB))
        front.attach(w, records=records)
        routes = CorpusJobRoutes({JOB: object()})
        routes.hold(JOB, lambda: w.auditor.seeded.is_set(), latch=False)

        async def refused() -> None:
            assert routes.is_held(JOB)
            assert w.auditor.audits_pending() is None and w.auditor.expected_audits(None) is None
            assert w.auditor.audits_idle_since is None and not w.settler.sizing_known
            assert w.settler.open_value() is None and w.settler.recent_paid == []
            for read in (w.auditor.audit_backlog, w.auditor.audit_share, w.auditor.audit_decisions):
                with pytest.raises(LookupError):
                    read()
            with pytest.raises(LookupError):
                await w.auditor.pending_ids()

        await refused()                       # no snapshot yet
        view = {"seeded": True, "audit_backlog": 3, "audit_share": 0.4, "audit_decisions": 9, "audits_pending": 2,
                "expected_audits": 0.8, "audits_idle_since": None, "audit_reasons": {"drawn": 1}, "ahead_wasted": 0,
                "recent_paid": [0.01], "open_value": 0.02, "state_writes": 1, "attention": []}
        front._take({"boot": "b1", "jobs": {JOB: view}})
        assert not routes.is_held(JOB) and w.auditor.audit_backlog() == 3 and w.settler.open_value() == 0.02
        assert await w.auditor.pending_ids() == []
        clock.t += 10.1
        await refused()                       # stale
        front._take({"boot": "b2", "jobs": {JOB: {"seeded": False, "attention": []}}})
        await refused()                       # a new judge, not seeded
        front._take({"boot": "b2", "jobs": {JOB: view}})
        assert not routes.is_held(JOB)
        front.mark_down()
        await refused()                       # its process is gone

    asyncio.run(go())


def test_a_reply_in_flight_when_the_judge_died_proves_nothing(bucket):
    _, run_dir = bucket

    async def go():
        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=False)

        class Late:
            path = "judge.sock"

            async def post(self, path, body, **kw):
                front.mark_down()               # seen dead while this answer was on its way
                return {"boot": "b1", "jobs": {JOB: {"seeded": True}}}

        front.link._client = Late()
        assert not await front.sync_once()
        assert not front.live(JOB)

    asyncio.run(go())


def test_the_front_forgets_a_ledger_the_judge_wrote(bucket):
    _, run_dir = bucket

    class Store:
        forgotten = 0

        def forget_ledgers(self, job_id):
            assert job_id == JOB
            self.forgotten += 1

    dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
    front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=False)
    store = Store()
    w = SimpleNamespace(entry=_entry(), cap=0.1, job=SimpleNamespace(job_id=JOB, prompt_source="x", sampling=None),
                        stats=JobStats(), proof=PROOF, miner_states=None)
    w.job.sampling = SimpleNamespace(draw=None)
    front.attach(w, records=None, store=store)
    front._take({"boot": "b1", "jobs": {}, "ledger_writes": {}})
    assert store.forgotten == 1                 # a new judge may have written before it could say so
    front._take({"boot": "b1", "jobs": {}, "ledger_writes": {}})
    assert store.forgotten == 1
    front._take({"boot": "b1", "jobs": {}, "ledger_writes": {JOB: 1}})
    assert store.forgotten == 2
    front._take({"boot": "b1", "jobs": {}, "ledger_writes": {JOB: 1}})
    assert store.forgotten == 2
    front._take({"boot": "b2", "jobs": {}, "ledger_writes": {}})
    assert store.forgotten == 3


def test_a_forgotten_ledger_is_read_again_and_a_read_in_flight_keeps_no_memory(monkeypatch):
    from reliquary.infrastructure.corpus_job_store import BucketJobStore

    versions = iter([({"v": 1}, "e1"), ({"v": 2}, "e2"), ({"v": 3}, "e3")])
    gate = {"event": None}

    async def read(job_id, **kw):
        value = next(versions)
        if gate["event"] is not None:
            await gate["event"].wait()
        return value

    monkeypatch.setattr(job_store, "read_ledgers", read)

    async def go():
        store = BucketJobStore()
        assert (await store.read_ledgers(JOB))[1] == "e1"
        assert (await store.read_ledgers(JOB))[1] == "e1"      # remembered
        store.forget_ledgers(JOB)
        gate["event"] = asyncio.Event()
        in_flight = asyncio.ensure_future(store.read_ledgers(JOB))
        await asyncio.sleep(0)
        store.forget_ledgers(JOB)               # written elsewhere while that read was on its way
        gate["event"].set()
        assert (await in_flight)[1] == "e2"
        gate["event"] = None
        assert (await store.read_ledgers(JOB))[1] == "e3"

    asyncio.run(go())


def _stream(dispatcher, run_dir, w):
    from reliquary.validator.order_judge import DispatchClient, DispatchServer

    server = DispatchServer(dispatcher, run_dir / "dispatch.sock", lambda job_id: w if job_id == JOB else None)
    return server, DispatchClient(run_dir / "dispatch.sock", retry_seconds=0.02, renotify_seconds=0.2)


def _draw_item() -> dict:
    return {"tokens": [1, 2, 3, 4], "prompt_len": 2, "proofs": None,
            "draw": {"schema": "x", "warp": {"top_k": 0}, "nested": [1, {"a": 0.1}]}}


def test_the_stream_carries_scores_exactly_and_a_parked_batch_as_parked(bucket):
    from reliquary.protocol.toploc import ChunkResult
    from reliquary.validator.eval_control import BatchParked

    _, run_dir = bucket
    diagnostics = {"positions": [1, 2], "pick": [0.125, 1e-300]}

    class Dispatcher(PairedAuditDispatcher):
        outcome = "ok"

        async def score(self, model, proof, items, job_id="", judge=None, batches=None):
            assert items == [_draw_item()] and proof is PROOF and model == ("org/Frozen", "abc123")
            if self.outcome == "parked":
                raise BatchParked("no agreeing pair")
            if self.outcome == "refused":
                raise ValueError("bad row")
            return [("ok", (ChunkResult(1, 0.1, 1 / 3),), ("e1", "e2"), diagnostics)]

    async def go():
        dispatcher = Dispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        w = SimpleNamespace(job=SimpleNamespace(checkpoint_repo="org/Frozen", checkpoint_revision="abc123",
                                                sampling=SimpleNamespace(draw=None)), proof=PROOF)
        server, client = _stream(dispatcher, run_dir, w)
        await server.start()
        reader = asyncio.ensure_future(client.run())
        view = client.view(JOB)
        try:
            scored = await view.score([_draw_item()])
            assert scored == [("ok", (ChunkResult(1, 0.1, 1 / 3),), ("e1", "e2"), diagnostics)]
            dispatcher.outcome = "parked"
            with pytest.raises(BatchParked):
                await view.score([_draw_item()])
            dispatcher.outcome = "refused"
            with pytest.raises(ValueError):
                await view.score([_draw_item()])
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await server.close()

    asyncio.run(go())


def test_a_dropped_stream_resends_and_withdraws_what_nobody_waits_for(bucket):
    _, run_dir = bucket

    async def go():
        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        w = SimpleNamespace(job=SimpleNamespace(checkpoint_repo="org/Frozen", checkpoint_revision="abc123",
                                                sampling=SimpleNamespace(draw=None)), proof=PROOF)
        server, client = _stream(dispatcher, run_dir, w)
        await server.start()
        reader = asyncio.ensure_future(client.run())
        view = client.view(JOB)
        item = {"tokens": [1, 2, 3, 4], "prompt_len": 2, "proofs": ["p"]}
        call = asyncio.ensure_future(view.score([item]))
        executor = None
        try:
            await _until(lambda: dispatcher.pending_items(JOB) == 1, 5, "the batch")
            for connection in list(server._connections):
                connection.writer.close()       # the stream drops, the judge lives
            # The first request's batch is withdrawn; its resend is the one waiting.
            await _until(lambda: client.connected() and dispatcher.pending_items(JOB) == 1
                         and sum(1 for q in dispatcher._queues.values() for b in q if b.future.cancelled()) == 1,
                         5, "the resend")
            executor = asyncio.ensure_future(_Executor(dispatcher).run())
            scored = await asyncio.wait_for(call, 5)
            assert scored[0][0] == "ok" and scored[0][2] == ("e1",)
        finally:
            for task in (call, reader, executor):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (call, reader, executor) if t), return_exceptions=True)
            await server.close()

    asyncio.run(go())


def test_a_quarantine_reaches_every_view_again_after_what_was_being_written(bucket):
    _, run_dir = bucket

    async def go():
        dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
        server, client = _stream(dispatcher, run_dir, None)
        await server.start()
        reader = asyncio.ensure_future(client.run())
        heard = []

        async def listener(executor_id):
            heard.append(executor_id)

        view = client.view(JOB)
        view.subscribe(listener)
        try:
            await _until(client.connected, 5, "the stream")
            await asyncio.sleep(0.05)
            await dispatcher.quarantine("e9", "test")
            await _until(lambda: len(heard) == 2, 3, "the event, then its repeat")
            assert heard == ["e9", "e9"]
            view.close()
            await dispatcher.quarantine("e8", "test")
            await asyncio.sleep(0.4)
            assert "e8" not in heard            # a closed order starts no re-audit
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await server.close()

    asyncio.run(go())


def test_the_control_holds_a_judged_order_and_keeps_its_size_unknown_when_stale(world, monkeypatch, tmp_path):
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_validator
    from reliquary.validator.eval_control import EvalExecutorDirectory, build_eval_control

    registry, _ = world

    async def warm(executor):
        return None

    monkeypatch.setattr(corpus_validator, "warm_drand_chain", warm)

    async def read_entries():
        return dict(registry["entries"])

    async def go():
        clock = SimpleNamespace(t=100.0)
        directory = EvalExecutorDirectory(list_documents=lambda: asyncio.sleep(0, []))
        dispatcher = PairedAuditDispatcher(directory=directory)
        run_dir = tmp_path / "r"
        run_dir.mkdir(mode=0o700)
        front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=False, clock=lambda: clock.t,
                                stale_seconds=10.0)

        class Tokenizer:
            chat_template = "x"

            def apply_chat_template(self, messages, **kwargs):
                return f"<u>{messages[0]['content']}</u>"

            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 90 for c in text]

        app, job_set = build_eval_control(
            store=BucketJobStore(), records=BucketRecordStore(), dispatcher=dispatcher, directory=directory,
            verify_signature=lambda request: True, tokenizer_for=lambda repo, revision: (Tokenizer(), 100),
            read_entries=read_entries, order_judge=front)
        await job_set.refresh()
        job_id, other = "order-eval-a", "order-eval-b"
        assert sorted(job_set.served) == [job_id, other] and sorted(front._jobs) == [job_id, other]
        # Nothing runs here for a judged order, and no listener is left on the dispatcher.
        assert all(not tasks for tasks in job_set._tasks.values()) and dispatcher._listeners == []
        routes = job_set._routes
        assert routes.held_reason(job_id) == HELD_AWAITING_AUDIT
        # What the submit route announces goes to the judge's feed, in order.
        job_set.served[job_id].on_accepted("a" * 64)
        assert [row[1:] for row in front.link._queue] == [(job_id, "a" * 64)]
        view = {"seeded": True, "audit_backlog": 0, "audit_share": 0.2, "audit_decisions": 3, "audits_pending": 0,
                "expected_audits": 0.0, "audits_idle_since": 90.0, "audit_reasons": {}, "ahead_wasted": 0,
                "recent_paid": [0.01], "open_value": 0.02, "state_writes": 0, "attention": ["background work stopped"]}
        front._take({"boot": "b1", "jobs": {job_id: view, other: view}})
        # Seeded, but no executor of the model claims yet.
        assert routes.held_reason(job_id) == HELD_AWAITING_AUDIT
        document = {"executor_id": "e1", "model_id": "customer/A", "model_revision": "a" * 40,
                    "provider_id": "p", "host": "h"}
        clock_now = dispatcher._clock()
        dispatcher._documents["e1"], dispatcher._seen["e1"] = document, clock_now
        dispatcher._protocols["e1"] = (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3)
        assert routes.held_reason(job_id) is None and job_id in routes.open_jobs()
        assert "background work stopped" in dispatcher.attention[job_id]

        async def stats(job):
            return {"unsettled": [0, 0, 0], "settled_count": 0, "totals": {}, "last_period": None}

        front.link.stats = stats
        status = await app.state.control_status()
        assert status["jobs"][job_id]["audits_pending"] == 0
        assert status["jobs"][job_id]["sizing"]["open_value"] == 0.02
        assert status["models"][f"customer/A@{'a' * 40}"]["expected_audits"] == 0
        # Open once: a card lost later does not close it, a judge lost does.
        dispatcher._seen["e1"] = -1e12
        assert routes.held_reason(job_id) is None
        clock.t += 10.5
        assert routes.held_reason(job_id) == HELD_AWAITING_AUDIT and job_id not in routes.open_jobs()
        job_set._status_cache.clear()
        status = await app.state.control_status()
        assert status["jobs"][job_id]["audits_pending"] is None
        assert "sizing" not in status["jobs"][job_id]
        assert status["jobs"][job_id]["audits_idle_since"] is None
        assert status["models"][f"customer/A@{'a' * 40}"]["expected_audits"] is None
        # Never drained on an unknown judge: a retired order stays wired.
        from dataclasses import replace

        registry["entries"][job_id] = replace(registry["entries"][job_id], status="retired", retired_at=5)
        await job_set.refresh()
        await job_set.refresh()
        assert job_id in job_set.served
        clock.t += 1.0
        front._take({"boot": "b1", "jobs": {job_id: view, other: view}})
        assert routes.held_reason(other) is not None or other in routes.open_jobs()
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())


def _gil_hog(stop: threading.Event) -> None:
    while not stop.is_set():
        sum(i * i for i in range(2000))


def _pass_seconds(log: Path, start: int) -> float:
    """The judge's own account of its passes logged past byte ``start``: seconds spent in them."""
    with open(log, "rb") as handle:
        handle.seek(start)
        text = handle.read().decode(errors="replace")
    return sum(float(m) for m in re.findall(r"corpus judge pass: job=\S+ ids=\d+ total=([0-9.]+)s", text))


def test_heavy_cpu_in_the_front_does_not_slow_a_judge_pass(bucket, monkeypatch, tmp_path):
    """Unaudited passes need only the judge: the seconds its passes take, the front's GIL idle, then saturated."""
    root, run_dir = bucket
    log = tmp_path / "judge.log"
    monkeypatch.setenv(h.LOG_ENV, str(log))
    monkeypatch.setenv(h.BEACON_ENV, "const")
    monkeypatch.setenv(h.LATENCY_ENV, "0.002,0.004,0.004,0.008,0.004")
    _trust(root)
    hold = SAMPLED["audit_hold_seconds"]

    async def timed(control, count: int) -> float:
        now = time.time()
        # Enough arrivals inside the hold, read before the records they vouch for.
        for hotkey in HOTKEYS:
            for k in range(3):
                control.front.link.accepted(JOB, _put_undrawn(root, received=now - 100 - k, q=0.5, hotkey=hotkey))
        await asyncio.sleep(1.5)
        sids = [_put_undrawn(root, received=now - hold - 430 - k * 0.001, q=0.5, hotkey=HOTKEYS[k % 4])
                for k in range(count)]
        start = log.stat().st_size
        for sid in sids:
            control.front.link.accepted(JOB, sid)
        base = root / "objects" / record_store._prefix(JOB, "verdicts")
        wanted = {f"{sid}.json" for sid in sids}
        # Listed, not parsed: this poll runs under the hogs.
        await _until(lambda: base.exists() and wanted <= set(os.listdir(base)), 120, "the passes")
        await asyncio.sleep(0.5)                 # the pass's closing log line
        return _pass_seconds(log, start)

    async def go():
        control = await _Control(root, run_dir, entry=_entry(**SAMPLED), auditor=ARRIVALS).start()
        stop = threading.Event()
        hogs = [threading.Thread(target=_gil_hog, args=(stop,), daemon=True) for _ in range(4)]
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            idle = await timed(control, 150)
            for hog in hogs:
                hog.start()
            loaded = await timed(control, 150)
        finally:
            stop.set()
            await control.close()
        out = os.environ.get("ORDER_JUDGE_LOAD_OUT")
        if out:
            Path(out).write_text(json.dumps({"idle_seconds": idle, "loaded_seconds": loaded}))
        assert idle > 0 and loaded > 0
        # Loose (a shared box): a judge on the front's GIL takes many times longer.
        assert loaded < max(3.0 * idle, idle + 2.0)

    asyncio.run(go())
