"""The order control's judge in its own process: a real child on a real unix
socket, one on-disk bucket, the front's real dispatcher and a fake executor."""

from __future__ import annotations

import asyncio
import hashlib
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

from reliquary.infrastructure import corpus_executor_store as executor_store
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

    def __init__(self, root: Path, run_dir: Path, *, entry=None, auditor=None, directory=None) -> None:
        self.root = root
        self.dispatcher = PairedAuditDispatcher(
            directory=directory or SimpleNamespace(revoke_locally=lambda e: None))
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
    for module in (job_store, record_store, storage, executor_store):
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


@pytest.mark.parametrize("quarantined_while_down", [True, False])
def test_a_restarted_judge_re_audits_what_a_quarantined_executor_scored(bucket, quarantined_while_down):
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
            await asyncio.gather(lying, return_exceptions=True)
            honest = _Executor(control.dispatcher, "honest")
            honest_task = asyncio.ensure_future(honest.run())
            first = control.front.process.pid
            os.kill(first, signal.SIGKILL)
            await _until(lambda: not control.front.live(JOB), 5, "the hold")
            if quarantined_while_down:
                await control.dispatcher.quarantine("liar", "test")
            await _until(lambda: control.front.process.pid != first and control.front.live(JOB), 90, "the restart")
            if not quarantined_while_down:
                await control.dispatcher.quarantine("liar", "test")
            await _until(lambda: forged in _voided(root), 30, "the re-audit's void")
            assert _voided(root)[forged]["executor_id"] == "liar"
            assert honest.scored >= 2
        finally:
            lying.cancel()
            if honest_task is not None:
                honest_task.cancel()
            await control.close()

    asyncio.run(go())


async def _registered(*executor_ids) -> None:
    for executor_id in executor_ids:
        await executor_store.register_executor(
            executor_id=executor_id, token_sha256=hashlib.sha256(executor_id.encode()).hexdigest(),
            model_id="org/Frozen", model_revision="abc123", expires_at=4e9, now=1.0, provider_id=f"p-{executor_id}",
            host=f"h-{executor_id}", scope="eval")


@pytest.mark.parametrize("then_revoked", [False, True])
def test_a_quarantine_from_before_a_front_restart_is_re_audited(bucket, then_revoked):
    from reliquary.validator.eval_control import EvalExecutorDirectory

    root, run_dir = bucket

    async def go():
        await _registered("liar", "honest")
        first = await _Control(root, run_dir, directory=EvalExecutorDirectory()).start()
        lying = asyncio.ensure_future(_Executor(first.dispatcher, "liar", lies=True).run())
        try:
            await _until(lambda: first.front.live(JOB), 90, "the judge's seed")
            forged = _put(root, received=time.time() - 30, forged=True, hotkey=HOTKEYS[0])
            first.front.link.accepted(JOB, forged)
            await _until(lambda: forged in _verdicts(root), 30, "the lying verdict")
            assert _verdicts(root)[forged]["scored_by"] == ["liar"]
        finally:
            lying.cancel()
            await asyncio.gather(lying, return_exceptions=True)
            await first.close()
        # Quarantined as the front went down: no judge heard of it.
        await executor_store.set_executor_status("liar", "quarantined", reason="test", scope="eval")
        if then_revoked:
            await executor_store.set_executor_status("liar", "revoked", reason="card released")
        second = await _Control(root, run_dir, directory=EvalExecutorDirectory()).start()
        honest = _Executor(second.dispatcher, "honest")
        scoring = asyncio.ensure_future(honest.run())
        try:
            await _until(lambda: forged in _voided(root), 120, "the re-audit's void")
            assert _voided(root)[forged]["executor_id"] == "liar"
            assert second.dispatcher.quarantined == {"liar"}     # the active one is not touched
        finally:
            scoring.cancel()
            await second.close()

    asyncio.run(go())


def _registry(documents):
    from reliquary.validator.eval_control import EvalExecutorDirectory

    async def read():
        if isinstance(documents["now"], Exception):
            raise documents["now"]
        return documents["now"]

    return EvalExecutorDirectory(list_documents=read, refresh_seconds=0.0)


def _document(executor_id, status="quarantined", scope="eval", **extra):
    return {"executor_id": executor_id, "token_sha256": executor_id * 8, "status": status, "scope": scope, **extra}


def test_the_registry_feeds_each_quarantine_once_and_writes_nothing_back():
    documents = {"now": [_document("q"), _document("ok", "active"), _document("off", "revoked"),
                         _document("corpus", scope="corpus"), _document("gone", "revoked", was_quarantined=True)]}
    directory, written, heard = _registry(documents), [], []

    async def write(executor_id, reason):
        written.append(executor_id)

    async def listener(executor_id):
        heard.append(executor_id)

    async def go():
        dispatcher = PairedAuditDispatcher(directory=directory, quarantine=write)
        dispatcher.subscribe(listener)
        await directory.refresh()
        assert await dispatcher.load_quarantined() == ["gone", "q"]
        assert await dispatcher.load_quarantined() == []
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sorted(heard) == ["gone", "q"] and written == []
        assert dispatcher.quarantined == {"gone", "q"} and not dispatcher._unwritten
        # A failed read, then an empty one: nothing cleared, nothing invented.
        documents["now"] = RuntimeError("the bucket is down")
        await directory.maybe_refresh()
        assert await dispatcher.load_quarantined() == []
        documents["now"] = []
        await directory.refresh()
        assert await dispatcher.load_quarantined() == [] and dispatcher.quarantined == {"gone", "q"}

    asyncio.run(go())


def test_a_revoke_keeps_the_quarantine_only_of_a_quarantined_executor(bucket):
    async def go():
        await _registered("q", "plain")
        await executor_store.set_executor_status("q", "quarantined", reason="lied", scope="eval")
        await executor_store.set_executor_status("q", "revoked", reason="card released")
        await executor_store.set_executor_status("plain", "revoked", reason="card released")
        assert (await executor_store.read_executor("q"))["was_quarantined"] is True
        assert "was_quarantined" not in await executor_store.read_executor("plain")

    asyncio.run(go())


def test_with_the_flag_off_a_dispatcher_run_adopts_no_registry_quarantine():
    documents = {"now": [_document("q")]}

    async def go():
        dispatcher = PairedAuditDispatcher(directory=_registry(documents))
        running = asyncio.ensure_future(dispatcher.run(sweep_seconds=0.01))
        await asyncio.sleep(0.1)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        assert dispatcher.quarantined == set()

    asyncio.run(go())


def test_a_hung_judge_is_killed_and_restarted(bucket, monkeypatch):
    from reliquary.validator import order_judge

    root, run_dir = bucket
    monkeypatch.setattr(order_judge, "SYNC_TIMEOUT_SECONDS", 1.0)

    async def go():
        control = await _Control(root, run_dir).start()
        try:
            await _until(lambda: control.front.live(JOB), 90, "the judge's seed")
            # Short limits only from here: a child's start takes longer than this one allows.
            control.front._hung_seconds, control.front._hung_failures = 4.0, 2
            first = control.front.process.pid
            os.kill(first, signal.SIGSTOP)          # alive, answering nothing
            await _until(lambda: not control.front.live(JOB), 10, "the hold")
            assert control.front.process.pid == first and control.front.process.is_alive()
            await _until(lambda: control.front.process.pid != first, 30, "the kill")
            control.front._hung_seconds, control.front._hung_failures = order_judge.HUNG_SECONDS, order_judge.HUNG_FAILURES
            await _until(lambda: control.front.live(JOB), 120, "the restart")
            assert control.front.starts == 2
        finally:
            await control.close()

    asyncio.run(go())


def test_a_front_that_stalled_does_not_kill_a_healthy_judge(bucket):
    _, run_dir = bucket
    clock = SimpleNamespace(t=0.0)
    dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
    front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=False, clock=lambda: clock.t)
    assert not front.hung()
    front._sync_unanswered()                    # one try failed, then the loop stalled past the limit
    clock.t += 10 * order_judge_module().HUNG_SECONDS
    assert not front.hung()
    for _ in range(order_judge_module().HUNG_FAILURES):
        front._sync_unanswered()
    assert front.hung()


def order_judge_module():
    from reliquary.validator import order_judge

    return order_judge


def _late_auditor(monkeypatch, remote):
    from reliquary.validator import corpus_auditor, eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    async def records_a_pass(self, ids, records, draws, *, deferred=None):
        for sid in ids:
            self._remote_scored.setdefault("X", []).append(sid)
        return [], set()

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "_audit_records", records_a_pass)
    return eval_control.eval_auditor(job_id="j", records=object(), tokenizer=None, proof=PROOF, vocab_size=10,
                                     job=draw_job(), job_store=object(), remote=remote)


def test_a_pass_written_after_its_executors_quarantine_is_re_audited_at_once(monkeypatch):
    remote = SimpleNamespace(subscribe=lambda listener: None, quarantined=lambda: {"X"})
    auditor = _late_auditor(monkeypatch, remote)
    audited, attempts = [], []

    async def reaudit(executor_id):
        attempts.append(executor_id)
        listed = auditor._remote_scored.pop(executor_id, [])
        if len(attempts) == 1:
            raise RuntimeError("the store hiccuped")    # what it took is not lost
        audited.extend(listed)
        return []

    monkeypatch.setattr(auditor, "reaudit_executor", reaudit)
    monkeypatch.setattr("reliquary.validator.eval_control.REOPEN_RETRY_SECONDS", 0.01)

    async def go():
        await auditor._audit_records(["late"], [{}], {})
        await _until(lambda: not auditor._late_reaudits, 3, "the re-audit")
        assert audited == ["late"] and len(attempts) == 2

    asyncio.run(go())


def test_a_single_process_view_schedules_no_late_re_audit(monkeypatch):
    auditor = _late_auditor(monkeypatch, SimpleNamespace(subscribe=lambda listener: None))

    async def go():
        await auditor._audit_records(["late"], [{}], {})
        assert not auditor._late_reaudits and auditor._remote_scored == {"X": ["late"]}

    asyncio.run(go())


def _settling_host(restore_wait):
    from reliquary.validator.order_judge_process import OrderJudgeHost

    return OrderJudgeHost(records=None, store=None, client=None, feed=None, settle_seconds=0.01,
                          tokenizer_for=lambda *a: None, restore_wait_seconds=restore_wait)


def _settling_wiring():
    settled = []

    async def settle_once():
        settled.append(1)

    auditor = SimpleNamespace(seeded=asyncio.Event(), scores_restored=asyncio.Event())
    auditor.seeded.set()
    return SimpleNamespace(auditor=auditor, settler=SimpleNamespace(settle_once=settle_once),
                           job=SimpleNamespace(job_id=JOB)), settled


def test_the_first_settlement_waits_for_the_restored_scores():
    async def go():
        w, settled = _settling_wiring()
        task = asyncio.ensure_future(_settling_host(30.0)._settle_forever(w))
        await asyncio.sleep(0.05)
        assert not settled
        w.auditor.scores_restored.set()
        await _until(lambda: settled, 2, "the settlement")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())


def test_a_restore_that_keeps_failing_never_stops_settlement_for_good():
    async def go():
        w, settled = _settling_wiring()
        host = _settling_host(0.05)
        task = asyncio.ensure_future(host._settle_forever(w))
        await _until(lambda: settled, 2, "the settlement")
        assert any("restored" in reason for reason in host._attention[JOB])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())


def test_restoring_scores_lists_each_pass_once(monkeypatch):
    from reliquary.validator import eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    verdicts = {"a": {"passed": True, "audited": True, "scored_by": ["X", "X"], "audited_at": 1.0},
                "b": {"passed": True, "audited": False, "scored_by": ["X"]},
                "c": {"passed": False, "audited": True, "scored_by": ["X"]}}

    class Records:
        async def list_verdict_ids(self, job_id):
            return list(verdicts)

        async def list_voided_ids(self, job_id):
            return []

        async def read_settlement(self, job_id):
            return {"settled": []}, None

        async def read_verdict(self, job_id, sid):
            return verdicts[sid]

    remote = SimpleNamespace(subscribe=lambda listener: None, quarantined=lambda: set())
    auditor = eval_control.eval_auditor(job_id="j", records=Records(), tokenizer=None, proof=PROOF, vocab_size=10,
                                        job=draw_job(), job_store=object(), remote=remote, restores_scored=True)
    auditor._remote_scored["X"] = ["a"]
    assert asyncio.run(auditor.restore_scored()) == 0
    assert auditor._remote_scored == {"X": ["a"]}


class _Record(dict):
    """A record a weak reference can follow."""


def _restart_reaudit(monkeypatch, slice_records):
    """A restarted judge (`restore_scored`) whose 10 passes executor X scored, X quarantined since. Both audits
    refuse ``run.charged`` (a charge) and ``run.drift`` (a drift, no penalty). ``run.events``: each audit (its ids,
    the records alive then) and each effect, in order; ``run.most_live``: the most records alive at once."""
    import weakref

    from reliquary.validator import corpus_auditor, eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    monkeypatch.setattr(corpus_auditor, "REAUDIT_SLICE_RECORDS", slice_records)
    job = draw_job()
    sids = [f"{i:064x}" for i in range(1, 11)]
    run = SimpleNamespace(sids=sids, charged=sids[2], drift=sids[7], events=[], live=0, most_live=0)

    def dropped():
        run.live -= 1

    class Records:
        async def list_verdict_ids(self, job_id):
            return list(sids)

        async def list_voided_ids(self, job_id):
            return []

        async def read_settlement(self, job_id):
            return {"settled": []}, None

        async def read_verdict(self, job_id, sid):
            return {"passed": True, "audited": True, "scored_by": ["X"], "audited_at": 1.0}

        async def read_submission(self, job_id, sid):
            await asyncio.sleep(0.001 * (int(sid, 16) % 3))            # out of order, as the store answers
            record = _Record(submission_id=sid, hotkey="5B" if sid == run.charged else "5A",
                             received_at=float(int(sid, 16)), token_count=10, completions=[])
            run.live += 1
            run.most_live = max(run.most_live, run.live)
            weakref.finalize(record, dropped)
            return record

        async def write_voided(self, job_id, sid, document):
            run.events.append(("void", sid, document["executor_id"], document.get("penalty", True)))
            return True

    remote = SimpleNamespace(subscribe=lambda listener: None, quarantined=lambda: {"X"})
    auditor = eval_control.eval_auditor(
        job_id=job.job_id, records=Records(), tokenizer=None, proof=PROOF, vocab_size=10, job=job,
        job_store=object(), remote=remote, restores_scored=True,
        on_voided=lambda sid, document: run.events.append(("reported", sid)))

    async def audit(records, *, local=False):
        await asyncio.sleep(0)                  # answered later, as the GPU or the executors are
        run.events.append(("audit", [record["submission_id"] for record in records], run.live))
        return [{"passed": False, "reason": "draw_mismatch", **({"penalty": False} if sid == run.drift else {})}
                if sid in (run.charged, run.drift) else {"passed": True}
                for sid in (record["submission_id"] for record in records)]

    async def escalate(failures, now, received=None):
        run.events.append(("escalate", {hotkey: sorted(ids) for hotkey, ids in failures.items()}, dict(received)))
        return {}

    async def reopen(sid):
        run.events.append(("reopen", sid))

    async def ledger(sid):
        run.events.append(("ledger", sid))

    monkeypatch.setattr(auditor, "_audit_outcomes", audit)
    monkeypatch.setattr(auditor, "_escalate", escalate)
    monkeypatch.setattr(auditor, "_reopen_slot", reopen)
    monkeypatch.setattr(auditor, "_record_failure", ledger)
    run.restored = asyncio.run(auditor.restore_scored())
    run.audits = [event[1:] for event in run.events if event[0] == "audit"]
    run.effects = [event for event in run.events if event[0] != "audit"]
    return run


def test_a_restart_re_audits_a_quarantined_executors_passes_a_slice_at_a_time(monkeypatch):
    """At each judge start every pass scored by an executor still quarantined is read and re-audited: all at once,
    2,000 records held ~450 MiB with their audit's copies (synthetic benchmark: fake store, synthetic records).
    A slice at a time, a record is dropped once its first audit passes, a refusal once its own audit confirms it;
    which records are re-audited, and every effect and its order, are as with one slice of them all."""
    run = _restart_reaudit(monkeypatch, 4)
    whole = _restart_reaudit(monkeypatch, 1000)
    assert run.restored == whole.restored == 10
    first = [ids for ids, _ in run.audits if len(ids) > 1 or ids[0] not in (run.charged, run.drift)]
    assert first == [run.sids[0:4], run.sids[4:8], run.sids[8:10]]          # every pass once, in order
    assert whole.audits[0][0] == whole.sids
    # Each refusal's own audit, right after its slice's first: its record alone is alive then (10 if all are held).
    assert [audit for audit in run.audits if audit[0] in ([run.charged], [run.drift])] == [
        ([run.charged], 1), ([run.drift], 1)]
    assert run.most_live <= 4                # a slice (10 when read all at once)
    assert run.effects == whole.effects == [
        ("escalate", {"5B": [run.charged]}, {run.charged: float(int(run.charged, 16)),
                                             run.drift: float(int(run.drift, 16))}),
        ("void", run.charged, "X", True), ("reported", run.charged),
        ("void", run.drift, "X", False), ("reopen", run.drift), ("reported", run.drift),
        ("ledger", run.charged), ("ledger", run.drift)]


def test_a_single_process_auditor_reads_no_verdict_back_at_its_start():
    from reliquary.validator import eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    auditor = eval_control.eval_auditor(job_id="j", records=object(), tokenizer=None, proof=PROOF, vocab_size=10,
                                        job=draw_job(), job_store=object(), reopen_slots=True)
    assert auditor.restores_scored is False


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


def test_a_reopen_that_lost_every_ledger_attempt_is_retried_until_it_lands(monkeypatch):
    from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict
    from reliquary.validator import corpus_service, eval_control
    from tests.unit.test_eval_draw import _job as draw_job

    monkeypatch.setattr(eval_control, "REOPEN_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(eval_control, "MAX_REOPEN_RETRY_SECONDS", 0.02)
    tries = []

    async def record_failure(store, job, prompt, sid, **kw):
        tries.append(sid)
        if len(tries) < 6:
            raise CorpusStoreConflict("lost the ledger race")
        return True

    monkeypatch.setattr(corpus_service, "record_prompt_failure", record_failure)

    class Records:
        async def read_submission(self, job_id, sid):
            return {"prompt_index": 0, "cursor": 0, "hotkey": "5A"}

    def auditor(retry):
        return eval_control.eval_auditor(
            job_id="j", records=Records(), tokenizer=None, proof=PROOF, vocab_size=10, job=draw_job(),
            job_store=object(), reopen_slots=True, retry_reopens=retry)

    async def go():
        once = auditor(False)
        await once._record_failure("s" * 64)
        await asyncio.sleep(0.05)
        assert len(tries) == 1 and not once._reopening     # flag off: left to the next start, as ever
        tries.clear()
        retried = auditor(True)
        await retried._record_failure("s" * 64)
        await retried._record_failure("s" * 64)            # one retry per record
        await _until(lambda: not retried._reopening, 5, "the reopen")
        assert len(tries) == 6

    asyncio.run(go())


class _LedgerStore:
    """A ledger behind a compare-and-swap, that loses the first ``lose`` writes."""

    def __init__(self, job, lose=0):
        self.snapshot = {"schema": "reliquary/corpus-ledgers/v2",
                         "slots": {str(job.prompt_start + i): job.slots_per_prompt for i in range(job.prompt_count)},
                         "cursors": {}, "seen_pending": [], "seen_segments": []}
        self.version, self.writes, self.lose = 0, 0, lose

    async def read_ledgers(self, job_id):
        return json.loads(json.dumps(self.snapshot)), str(self.version)

    async def write_ledgers(self, job_id, snapshot, etag):
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        if self.lose:
            self.lose -= 1
            raise CorpusStoreConflict("the front wrote first")
        assert etag == str(self.version)
        self.snapshot, self.version = snapshot, self.version + 1
        self.writes += 1


class _ForfeitRecords:
    def __init__(self, job, count):
        self.submissions = {f"{i:064x}": {"prompt_index": job.prompt_start + i % job.prompt_count, "cursor": 0,
                                          "hotkey": "5A"} for i in range(count)}
        self.voided = {}

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def write_voided(self, job_id, sid, document):
        self.voided[sid] = document

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return {"settled": []}, "etag"


def _forfeit_auditor(job, records, store, retry):
    from reliquary.validator import eval_control

    auditor = eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                        vocab_size=10, job=job, job_store=store, reopen_slots=True,
                                        retry_reopens=retry)
    auditor._paid["5A"] = [(1.0, sid) for sid in records.submissions]
    auditor._paid_loaded = True
    return auditor, {"5A": MinerState(forfeit_before=2.0)}


def test_a_forfeit_reopens_every_slot_in_one_ledger_write_before_the_status_hears():
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    records, store = _ForfeitRecords(job, 4), _LedgerStore(job)
    auditor, states = _forfeit_auditor(job, records, store, retry=False)
    writes_at_report = []
    auditor._report_voided = lambda sid, document: writes_at_report.append(store.writes)

    async def go():
        assert set(await auditor._void_forfeited(states, 3.0)) == set(records.submissions)
        await auditor._reopen_slots(sorted(records.submissions))        # already recorded: nothing to write

    asyncio.run(go())
    assert store.writes == 1 and writes_at_report == [1] * 4
    recorded = [entry for entries in store.snapshot["failed"].values() for entry in entries]
    assert sorted(entry[:12] for entry in recorded) == sorted(sid[:12] for sid in records.submissions)
    assert all(entry.endswith("@0@5A") for entry in recorded)


def test_a_forfeit_that_lost_its_ledger_race_lands_whole_on_one_retry(monkeypatch):
    from reliquary.corpus.slots import _failure_id
    from reliquary.validator import eval_control
    from reliquary.validator.corpus_service import DEFAULT_WRITE_ATTEMPTS
    from tests.unit.test_eval_draw import _job as draw_job

    monkeypatch.setattr(eval_control, "REOPEN_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(eval_control, "MAX_REOPEN_RETRY_SECONDS", 0.02)
    job = draw_job()
    records, store = _ForfeitRecords(job, 4), _LedgerStore(job, lose=DEFAULT_WRITE_ATTEMPTS + 1)
    auditor, states = _forfeit_auditor(job, records, store, retry=True)

    async def go():
        await auditor._void_forfeited(states, 3.0)
        assert store.writes == 0 and auditor._reopen_backlog == set(records.submissions)
        await _until(lambda: not auditor._reopening, 5, "the backlog")

    asyncio.run(go())
    assert store.writes == 1 and not auditor._reopen_backlog
    recorded = {_failure_id(entry) for entries in store.snapshot["failed"].values() for entry in entries}
    assert recorded == {sid[:12] for sid in records.submissions}


def test_a_forfeit_larger_than_a_read_slice_still_reopens_in_one_ledger_write(monkeypatch):
    """Read a slice at a time, recorded all at once (#571): never a partial reopen."""
    from reliquary.validator import corpus_auditor
    from tests.unit.test_eval_draw import _job as draw_job

    monkeypatch.setattr(corpus_auditor, "RECONCILE_SLICE_IDS", 3)
    job = draw_job()
    records, store = _ForfeitRecords(job, 10), _LedgerStore(job)       # 10 prompts of 30: no shared prompt
    auditor, states = _forfeit_auditor(job, records, store, retry=False)
    assert set(asyncio.run(auditor._void_forfeited(states, 3.0))) == set(records.submissions)
    assert store.writes == 1
    recorded = [entry for entries in store.snapshot["failed"].values() for entry in entries]
    assert len(recorded) == 10 and all(entry.endswith("@0@5A") for entry in recorded)


def test_a_reconcile_of_more_failures_than_a_slice_records_them_in_one_write(monkeypatch):
    from reliquary.validator import corpus_auditor
    from tests.unit.test_eval_draw import _Verdicts, _draw_auditor, _job as draw_job

    monkeypatch.setattr(corpus_auditor, "RECONCILE_SLICE_IDS", 4)
    job = draw_job()
    sids = [f"{i:012x}" + "0" * 52 for i in range(1, 11)]                # distinct 12-hex ledger keys
    records = _Verdicts({sid: {"prompt_index": job.prompt_start + i, "cursor": 0, "hotkey": "5A"}
                         for i, sid in enumerate(sids)})
    records.verdicts = {sid: {"passed": False} for sid in sids}
    store = _LedgerStore(job)
    auditor = _draw_auditor(job, records, store)

    async def go():
        assert await auditor.reconcile_failures() == 10
        assert store.writes == 1
        assert await auditor.reconcile_failures() == 10                  # all recorded already: no write
        assert store.writes == 1

    asyncio.run(go())
    recorded = {entry.split("@", 1)[0] for entries in store.snapshot["failed"].values() for entry in entries}
    assert recorded == {sid[:12] for sid in sids}


@pytest.mark.parametrize("judge", ["order", "corpus"])
def test_a_reconcile_reads_only_the_failures_its_ledger_may_lack(judge, monkeypatch):
    """A judge start read every failed record of every open order again, though their ledgers record nearly all.
    A recorded one whose ledger key no other verdict carries is not read; one whose key another verdict shares,
    here on another prompt, is read and recorded as before: the ledger tells failures apart by key within one
    prompt only."""
    from reliquary.validator import corpus_service, eval_control
    from tests.unit.test_eval_draw import _Verdicts, _draw_auditor, _job as draw_job

    job = draw_job()
    old = [f"{i:012x}" + "0" * 52 for i in range(1, 6)]                  # distinct 12-hex ledger keys
    new, twin = f"{6:012x}" + "0" * 52, old[0][:12] + "f" * 52             # twin: old[0]'s key, another prompt
    prompts = {**{sid: index for index, sid in enumerate(old)}, new: 5, twin: 9}
    records = _Verdicts({sid: {"prompt_index": job.prompt_start + index, "cursor": 0, "hotkey": "5A"}
                         for sid, index in prompts.items()})
    records.verdicts = {**{sid: {"passed": False} for sid in old}, "9" * 64: {"passed": True}}
    reads, read_submission = [], records.read_submission

    async def counted(job_id, sid):
        reads.append(sid)
        return await read_submission(job_id, sid)

    async def unreadable(store, job):
        raise ConnectionError("ledger read failed")

    records.read_submission = counted
    store = _LedgerStore(job)
    auditor = (eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                         vocab_size=10, job=job, job_store=store, reopen_slots=True)
               if judge == "order" else _draw_auditor(job, records, store))

    async def reconcile():
        reads.clear()
        return await auditor.reconcile_failures(), sorted(reads)

    async def go():
        assert await reconcile() == (5, sorted(old))
        records.verdicts.update({new: {"passed": False}, twin: {"passed": False}})
        assert await reconcile() == (7, sorted([old[0], new, twin]))          # the new one, and the shared key
        assert await reconcile() == (7, sorted([old[0], twin]))               # the shared key alone
        with monkeypatch.context() as patch:
            patch.setattr(corpus_service, "recorded_failure_ids", unreadable)
            assert await reconcile() == (7, sorted([*old, new, twin]))        # ledger unread: every one

    asyncio.run(go())
    assert store.writes == 2
    keys = {int(index): [entry.split("@", 1)[0] for entry in entries]
            for index, entries in store.snapshot["failed"].items()}
    assert keys == {**{job.prompt_start + index: [sid[:12]] for index, sid in enumerate(old)},
                    job.prompt_start + 5: [new[:12]], job.prompt_start + 9: [twin[:12]]}


class _YieldingLedgerStore(_LedgerStore):
    """A ledger read that lets the loop run before it takes its snapshot, as a store's read does."""

    async def read_ledgers(self, job_id):
        for _ in range(3):
            await asyncio.sleep(0)
        return await super().read_ledgers(job_id)


def _twins(job, records_class=None):
    """A failure whose ledger write was lost (a stop in between) and a twin: another submission with its 12-hex
    ledger key, on another prompt. No verdict written yet."""
    from tests.unit.test_eval_draw import _Verdicts

    lost = f"{1:012x}" + "0" * 52
    twin = lost[:12] + "f" * 52
    records = (records_class or _Verdicts)({
        lost: {"prompt_index": job.prompt_start, "cursor": 0, "hotkey": "5A"},
        twin: {"prompt_index": job.prompt_start + 9, "cursor": 0, "hotkey": "5A"}})
    return lost, twin, records


def _reconciling(judge, job, records, store):
    from reliquary.validator import eval_control
    from tests.unit.test_eval_draw import _draw_auditor

    if judge == "order":
        return eval_control.eval_auditor(job_id=job.job_id, records=records, tokenizer=None, proof=PROOF,
                                         vocab_size=10, job=job, job_store=store, reopen_slots=True)
    return _draw_auditor(job, records, store)


def _failed_keys(store):
    return {int(index): [entry.split("@", 1)[0] for entry in entries]
            for index, entries in store.snapshot["failed"].items()}


@pytest.mark.parametrize("judge", ["order", "corpus"])
def test_a_reconcile_reads_its_ledger_before_it_lists_the_verdicts(judge):
    """The twin is judged while the verdicts are listed: its verdict lands past the listing, then its failure in
    the ledger. A ledger read after the listing (or beside it) would find the shared key recorded with a single
    listed verdict carrying it, and skip the lost failure. Read first, the ledger holds no failure whose verdict
    the listing misses: the lost failure is read and recorded at its own prompt."""
    from reliquary.validator.corpus_service import record_prompt_failure
    from tests.unit.test_eval_draw import _job as draw_job

    job = draw_job()
    lost, twin, records = _twins(job)
    records.verdicts = {lost: {"passed": False}}
    store, after = _YieldingLedgerStore(job), _LedgerStore(job)
    asyncio.run(record_prompt_failure(after, job, job.prompt_start + 9, twin, slot=0, hotkey="5A"))
    listed = records.list_verdict_ids

    async def listing_while_the_twin_is_judged(job_id):
        ids = await listed(job_id)
        records.verdicts[twin] = {"passed": False}
        store.snapshot, store.version = after.snapshot, store.version + 1
        records.list_verdict_ids = listed
        return ids

    records.list_verdict_ids = listing_while_the_twin_is_judged
    assert asyncio.run(_reconciling(judge, job, records, store).reconcile_failures()) == 1
    assert _failed_keys(store) == {job.prompt_start: [lost[:12]], job.prompt_start + 9: [twin[:12]]}


def test_a_listed_pass_sharing_a_recorded_key_still_has_the_failure_read():
    """The one failure recorded before its verdict: when a verdict's create conflicts and the re-read finds none
    (another create of that id still in flight), `EvalAuditor._write` records the failure of the verdict it tried
    to write, and the verdict that lands may pass. A listed verdict counts toward a shared key whatever it says:
    counted only if failed, the twin's pass would leave the lost failure's key carried by one failed verdict, and
    the lost failure unread."""
    from tests.unit.test_eval_draw import _Verdicts, _job as draw_job

    class _InFlight(_Verdicts):
        async def write_verdict(self, job_id, submission_id, verdict):
            if submission_id == twin:
                return False                        # another create of it, in flight: not readable yet
            return await super().write_verdict(job_id, submission_id, verdict)

    job = draw_job()
    lost, twin, records = _twins(job, _InFlight)
    store = _LedgerStore(job)
    auditor = _reconciling("order", job, records, store)

    async def go():
        assert await auditor._write(twin, {"passed": False, "reason": "draw_mismatch"}) == (None, False)
        assert _failed_keys(store) == {job.prompt_start + 9: [twin[:12]]}
        records.verdicts[twin] = {"passed": True}           # the create in flight lands: a pass
        records.verdicts[lost] = {"passed": False}          # its ledger write lost
        return await auditor.reconcile_failures()

    assert asyncio.run(go()) == 1
    assert _failed_keys(store) == {job.prompt_start: [lost[:12]], job.prompt_start + 9: [twin[:12]]}


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
