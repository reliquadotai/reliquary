"""The whole order control with its judge apart: the real ``build_eval_control`` routes, a real judge child, the real
lease clock and auction, on an on-disk bucket with a fake executor. The judge is SIGKILLed with leases live and lapsed;
the front's own loop is stalled around the stale-snapshot limit."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import secrets
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.validator.order_judge import SNAPSHOT_STALE_SECONDS
from tests.unit import corpus_split_harness as h
from tests.unit import test_order_judge_draw as draw

JOB = draw.JOB
LEASE_SECONDS = 8.0
# The restart's backoff: the hold outlasts a lease and its late window whatever the child's start takes.
BACKOFF_SECONDS = 2 * LEASE_SECONDS + 4
DONE, LATE, LIVE, GHOST, SPARED, TAKER = "5Done", "5Late", "5Live", "5Ghost", "5Spared", "5Taker"
HOTKEYS = (DONE, LATE, LIVE, GHOST, SPARED, TAKER)
PAID = (DONE, LATE, LIVE, TAKER)


def _entry():
    return draw._entry(pricing="auction", auction_initial_leases=16)


async def _control(root: Path, run_dir: Path, *, stale_seconds: float, lease_seconds: float) -> SimpleNamespace:
    from reliquary.corpus import draw_leases
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.protocol import signatures
    from reliquary.validator import eval_control
    from reliquary.validator.eval_control import EvalExecutorDirectory, PairedAuditDispatcher, build_eval_control
    from reliquary.validator.order_judge import OrderJudgeFront
    from tests.unit import corpus_split_fakes as fakes

    signatures.verify_corpus_lease_signature = lambda request: True
    eval_control.order_job_refusal = lambda entry, job: None
    eval_control.qualification_refusal = lambda *args, **kw: None
    draw_leases.lease_seconds = lambda job: lease_seconds
    # The same slots in the same order in every run: the same submission ids.
    draw_leases.secrets = SimpleNamespace(randbits=lambda bits: 11, token_hex=secrets.token_hex)

    async def declared(job_id):
        return {}

    async def read_entries():
        return {JOB: _entry()}

    async def documents():
        return []

    directory = EvalExecutorDirectory(list_documents=documents)
    dispatcher = PairedAuditDispatcher(directory=directory)
    front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=True, sync_seconds=0.2,
                            stale_seconds=stale_seconds, settle_seconds=0.5, auditor_kwargs=draw.AUDITOR,
                            backoff_seconds=BACKOFF_SECONDS,
                            child_init="tests.unit.test_order_judge_draw:init_child")
    app, job_set = build_eval_control(
        store=BucketJobStore(), records=BucketRecordStore(), dispatcher=dispatcher, directory=directory,
        verify_signature=lambda request: True, verify_skip_signature=lambda request: True,
        tokenizer_for=lambda repo, revision: (fakes.Tokenizer(), fakes.VOCAB),
        qualification_store=SimpleNamespace(), order_jobs=SimpleNamespace(read=declared),
        read_entries=read_entries, refresh_every_seconds=1.0, order_judge=front)
    await job_set.refresh()
    assert JOB in job_set.served
    background = [asyncio.ensure_future(front.run()), asyncio.ensure_future(job_set.run()),
                  asyncio.ensure_future(draw._executor(dispatcher))]
    routes = job_set._routes
    await draw._until(lambda: routes.held_reason(JOB) is None, 90, "the order's opening")
    return SimpleNamespace(app=app, job_set=job_set, routes=routes, front=front, background=background,
                           wiring=job_set.served[JOB], router=routes.routers[JOB])


async def _close(control) -> None:
    for task in control.background:
        task.cancel()
    await asyncio.wait(control.background, timeout=5)
    await asyncio.to_thread(control.front.stop, 5.0)


class Miners:
    def __init__(self, client, control) -> None:
        self.client, self.control = client, control

    def _signed(self, hotkey: str) -> dict:
        return {"job_id": JOB, "miner_hotkey": hotkey, "issued_at": int(time.time()),
                "nonce": secrets.token_hex(16), "signature": "ok"}

    async def ask(self, hotkey: str) -> None:
        body = {**self._signed(hotkey), "micro_alpha_per_mtoken": 200_000, "capacity": 1}
        answer = (await self.client.post(f"/corpus/jobs/{JOB}/ask", json=body)).json()
        assert answer["accepted"], answer

    async def lease(self, hotkey: str) -> dict:
        answer = (await self.client.post(f"/corpus/jobs/{JOB}/lease", json=self._signed(hotkey))).json()
        assert answer["leased"], answer
        return answer

    async def submit(self, hotkey: str, lease: dict):
        from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
        from reliquary.protocol.signatures import corpus_submission_id
        from tests.unit.test_corpus_service import CHECKPOINT, EOS, _faithful_prompt, _text_for

        tokens = [7] * 99 + [EOS]
        request = CorpusSubmissionRequest(
            job_id=JOB, miner_hotkey=hotkey, cursor=lease["slot"], prompt_index=lease["prompt_index"],
            checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(lease["prompt_index"]),
            completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok")
        response = await self.client.post(f"/corpus/jobs/{JOB}/submit", json=request.model_dump())
        return response, corpus_submission_id(request)

    async def delivered(self, hotkey: str, lease: dict) -> str:
        response, sid = await self.submit(hotkey, lease)
        assert response.status_code == 200 and response.json()["accepted"], response.text
        return sid

    async def poll(self) -> None:
        await self.client.get("/corpus/jobs")


def _book(control):
    return control.router.draw_leases[JOB]


def _auction(control):
    return control.router.auction


def _charges(control, hotkeys=HOTKEYS) -> dict:
    book, auction = _book(control), _auction(control)
    return {hotkey: (round(auction.debt_alpha(hotkey), 15), len(auction._missed.get(hotkey, ())),
                     len(book._unused.get(hotkey, ()))) for hotkey in hotkeys
            if auction.debt_alpha(hotkey) or auction._missed.get(hotkey) or book._unused.get(hotkey)}


def _deadlines(control) -> dict[str, float]:
    book = _book(control)
    return {lease.lease_id: lease.expires_at for store in (book._by_id, book._lapsed) for lease in store.values()}


async def _run(mode: str, root: Path, run_dir: Path) -> dict:
    import httpx

    control = await _control(root, run_dir, stale_seconds=5.0, lease_seconds=LEASE_SECONDS)
    report: dict = {}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=control.app), base_url="http://front",
                                     timeout=30) as client:
            miners = Miners(client, control)
            for hotkey in HOTKEYS:
                await miners.ask(hotkey)
            ids = {}
            done = await miners.lease(DONE)
            ids[DONE] = await miners.delivered(DONE, done)
            await draw._until(lambda: ids[DONE] in h.listed(root, JOB, "verdicts"), 30, "the first verdict")
            late, spared = await miners.lease(LATE), await miners.lease(SPARED)
            # Leased before those two lapse: a lapsed slot is open, and would be granted first.
            await asyncio.sleep(LEASE_SECONDS - 2.0)
            live, ghost = await miners.lease(LIVE), await miners.lease(GHOST)
            await asyncio.sleep(2.5)
            await miners.poll()
            book = _book(control)
            assert {late["lease_id"], spared["lease_id"]} <= set(book._lapsed)
            running = {live["lease_id"], ghost["lease_id"]}
            assert running <= set(book._by_id) and not _charges(control)

            if mode == "kill":
                before = _deadlines(control)
                pid = control.front.process.pid
                os.kill(pid, signal.SIGKILL)
                await draw._until(lambda: control.routes.held_reason(JOB) == draw_held(), 3, "the front's hold")
                held_from = time.monotonic()
                await miners.poll()
                assert book._stopped_at is not None, "the clock did not stop"
                refused, _ = await miners.submit(LIVE, live)
                assert (refused.status_code, refused.json()["detail"]) == (409, "job_paused")
                while time.monotonic() - held_from < 2 * LEASE_SECONDS + 1:
                    await miners.poll()
                    assert not _charges(control), "charged during the hold"
                    assert running <= set(book._by_id), "a lease lapsed during the hold"
                    await asyncio.sleep(0.2)
                assert _deadlines(control) == before, "a deadline moved before the hold ended"
                await draw._until(lambda: control.front.process.pid != pid and control.routes.held_reason(JOB) is None,
                                  120, "the judge's return")
                held_for = time.monotonic() - held_from
                await miners.poll()
                assert book._stopped_at is None, "the clock did not restart"
                after = _deadlines(control)
                moved = {lease_id: after[lease_id] - before[lease_id] for lease_id in before}
                report["hold_seconds"] = held_for
                report["moved_seconds"] = sorted(moved.values())
                assert max(moved.values()) - min(moved.values()) < 1e-6, "leases moved by different lengths"
                assert abs(next(iter(moved.values())) - held_for) < 3.0, (moved, held_for)
                assert running <= set(book._by_id) and not _charges(control)

            ids[LATE] = await miners.delivered(LATE, late)
            ids[LIVE] = await miners.delivered(LIVE, live)
            taker = await miners.lease(TAKER)
            assert (taker["prompt_index"], taker["slot"]) == (spared["prompt_index"], spared["slot"])
            ids[TAKER] = await miners.delivered(TAKER, taker)
            report["spared_charged"] = SPARED in _charges(control)
            assert not {LATE, LIVE, TAKER, DONE} & set(_charges(control))

            # The clock runs again: the one that never delivers is charged when its moved time is up, not before.
            ghost_lease = next(lease for lease in book._by_id.values() if lease.hotkey == GHOST)
            charged_at = ghost_lease.expires_at + book.seconds
            while time.time() < charged_at - 0.5:
                await miners.poll()
                assert GHOST not in _charges(control), "charged before its moved time was up"
                await asyncio.sleep(0.2)
            await asyncio.sleep(1.0)
            await miners.poll()
            report["ghost_charge"] = _charges(control).get(GHOST, (0, 0, 0))[1:]

            wanted = set(ids.values())
            await draw._until(lambda: wanted <= set(h.listed(root, JOB, "verdicts")), 60, "every verdict")
            await draw._until(lambda: wanted <= set(h.settlement(root, JOB).get("settled") or ()), 90, "every settlement")
            verdicts = h.listed(root, JOB, "verdicts")
            report["verdicts"] = {hotkey: {key: verdicts[sid].get(key) for key in (
                "hotkey", "passed", "audited", "reason", "forced_draw", "token_count", "lease_price", "lease_debit",
                "scored_by")} for hotkey, sid in ids.items()}
    finally:
        await _close(control)
    rewards: dict[str, float] = {}
    for row in map(json.loads, (root / "archives" / "writes.jsonl").read_text().splitlines()):
        for hotkey, value in row["rewards"].items():
            rewards[hotkey] = rewards.get(hotkey, 0.0) + float(value)
    report["rewards"] = rewards
    return report


def draw_held() -> str:
    from reliquary.validator.eval_control import HELD_AWAITING_AUDIT

    return HELD_AWAITING_AUDIT


async def _stalled(root: Path, run_dir: Path, seconds: float) -> dict:
    """The front's event loop blocked for ``seconds``, at the real stale limit, with one lease running."""
    import httpx

    control = await _control(root, run_dir, stale_seconds=SNAPSHOT_STALE_SECONDS, lease_seconds=600.0)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=control.app), base_url="http://front",
                                     timeout=30) as client:
            miners = Miners(client, control)
            await miners.ask(LIVE)
            lease = await miners.lease(LIVE)
            before = _deadlines(control)[lease["lease_id"]]
            pid = control.front.process.pid
            await asyncio.sleep(0.5)                # a fresh snapshot just before
            synced = control.front._snapshot_at
            time.sleep(seconds)                     # blocks the loop: no sync, no route
            age = control.front._clock() - synced
            held_at_once = control.routes.held_reason(JOB) is not None
            await miners.poll()
            stopped = _book(control)._stopped_at is not None
            started = time.monotonic()
            await draw._until(lambda: control.routes.held_reason(JOB) is None, 20, "the order's reopening")
            recovered = time.monotonic() - started
            await miners.poll()
            assert _book(control)._stopped_at is None
            # How much of the stall the lease got back.
            credited = _deadlines(control)[lease["lease_id"]] - before
            sid = await miners.delivered(LIVE, lease)
            await draw._until(lambda: sid in h.listed(root, JOB, "verdicts"), 30, "the verdict")
            return {"stall_seconds": seconds, "snapshot_age": age, "held_at_once": held_at_once,
                    "clock_stopped": stopped, "recovered_seconds": recovered, "credited_seconds": credited,
                    "judge_restarted": control.front.process.pid != pid, "charges": _charges(control, (LIVE,))}
    finally:
        await _close(control)


def _scenario(kind: str, root: str, run_dir: str, out: str, stall: float) -> None:
    import logging

    from reliquary.shared.log_levels import unmute_reliquary_loggers

    draw.init_child()
    logging.basicConfig(level=logging.WARNING, force=True)
    unmute_reliquary_loggers()
    draw._seed(Path(root), prompts=len(HOTKEYS) - 1, hotkeys=HOTKEYS)
    if kind == "stall":
        result = asyncio.run(_stalled(Path(root), Path(run_dir), stall))
    else:
        result = asyncio.run(_run(kind, Path(root), Path(run_dir)))
    Path(out).write_text(json.dumps(result))
    os._exit(0)


def _outcome(kind: str, base: Path, monkeypatch, stall: float = 0.0, *, timeout: float = 280) -> dict:
    base.mkdir(parents=True)
    root, run_dir = base / "bucket", base / "run"
    root.mkdir()
    run_dir.mkdir(mode=0o700)
    monkeypatch.setenv(h.ROOT_ENV, str(root))
    monkeypatch.setenv(h.LATENCY_ENV, "0,0,0,0,0")
    monkeypatch.setenv(h.LOG_ENV, str(base / "log"))
    out = base / "out.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_scenario, args=(kind, str(root), str(run_dir), str(out), stall))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(5)
    assert out.exists(), (base / "log").read_text()[-3500:] if (base / "log").exists() else "no log"
    result = json.loads(out.read_text())
    target = os.environ.get("ORDER_JUDGE_INTEGRATED_OUT")
    if target:
        with open(target, "a") as handle:
            handle.write(json.dumps({"kind": kind, **{k: v for k, v in result.items() if k != "verdicts"}}) + "\n")
    return result


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    try:
        path = tmp_path_factory.mktemp("run")
        return {kind: _outcome(kind, path / kind, patch) for kind in ("clean", "kill")}
    finally:
        patch.undo()


def test_a_judge_killed_mid_run_ends_in_the_verdicts_and_pay_of_an_uninterrupted_run(runs):
    assert runs["kill"]["verdicts"] == runs["clean"]["verdicts"]
    assert runs["kill"]["rewards"] == pytest.approx(runs["clean"]["rewards"])
    assert set(runs["kill"]["rewards"]) == set(PAID)


def test_the_leases_moved_on_by_the_hold(runs):
    kill = runs["kill"]
    assert kill["hold_seconds"] > 2 * LEASE_SECONDS
    assert kill["moved_seconds"][0] == pytest.approx(kill["hold_seconds"], abs=3.0)


def test_a_lapsed_lease_taken_after_a_hold_is_spared_and_without_one_is_not(runs):
    assert runs["kill"]["spared_charged"] is False
    assert runs["clean"]["spared_charged"] is True


def test_a_miner_that_never_delivers_is_still_charged_once(runs):
    assert runs["kill"]["ghost_charge"] == [1, 1] == runs["clean"]["ghost_charge"]


@pytest.mark.parametrize("stall, held", [(SNAPSHOT_STALE_SECONDS - 2.0, False), (SNAPSHOT_STALE_SECONDS + 2.0, True)])
def test_a_stalled_front_loop_around_the_stale_limit(tmp_path, monkeypatch, stall, held):
    result = _outcome("stall", tmp_path / "stall", monkeypatch, stall)
    assert result["held_at_once"] is held and result["clock_stopped"] is held
    assert (result["snapshot_age"] > SNAPSHOT_STALE_SECONDS) is held
    assert result["judge_restarted"] is False       # a slow front is no reason to lose the judge
    assert result["recovered_seconds"] < 10.0
    assert result["charges"] == {}
