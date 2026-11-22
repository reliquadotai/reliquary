"""A real draw order through the front's routes, judged in one process and in the
judge child: the same submissions must end in the same verdicts, audit reasons,
reopened slots and pay."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit import corpus_split_harness as h

JOB = "order-gen-a"
EMISSION = 100.0
ALPHA = 0.5
GOOD, CHEAT, DRIFT, REDO = "5Good", "5Cheat", "5Drift", "5Redo"
TOKENS = {GOOD: 7, CHEAT: 8, DRIFT: 9, REDO: 7}
# What the fake executor reads off a sample: followed, another model's, a near miss.
DIAGNOSTICS = {7: {"positions": 100, "stochastic": 50, "exact": 100, "max_miss": 0.0, "last_miss": 0.0},
               8: {"positions": 100, "stochastic": 100, "exact": 10, "max_miss": 0.9, "last_miss": 0.0},
               9: {"positions": 40, "stochastic": 40, "exact": 20, "max_miss": 0.0, "last_miss": 0.0}}
AUDITOR = {"_accept_slack": 5.0, "_rescan_every": 0.5}
PERIOD_ENV = "ORDER_JUDGE_TEST_PERIOD"


def init_child() -> None:
    h.install()
    from reliquary.protocol import profiles
    from reliquary.validator import corpus_validator, eval_control
    from tests.unit import corpus_split_fakes as fakes

    eval_control.load_cpu_tokenizer = lambda repo, revision: (fakes.Tokenizer(), fakes.VOCAB)
    profiles.toploc_proof = lambda profile: fakes.PROOF

    async def no_chain(executor):
        return None

    corpus_validator.warm_drand_chain = no_chain
    if os.environ.get(PERIOD_ENV):
        from reliquary.validator import corpus_periods

        corpus_periods.PERIOD_SECONDS = int(os.environ[PERIOD_ENV])


def _entry(**params):
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, TaskEntry

    return TaskEntry(
        task_id=JOB, profile_id=ACTIVE_PROTOCOL_PROFILE.profile_id, profile_sha256="0" * 64,
        mechanism=MECHANISM_CORPUS_GENERATION,
        params={"cap": 0.5, "audit_q": 1.0, "audit_probation_submissions": 1, "audit_hold_seconds": 60.0,
                "audit_ban_after_failures": 1000, "settlement": "period-ema-v1",
                "max_price_per_token": ALPHA / 1e6 / EMISSION, "max_price_alpha_per_mtoken": ALPHA,
                "max_price_miner_alpha_per_period": EMISSION, "max_price_set_at": 1.0, **params},
        status="active", retired_at=None, contract=ACTIVE_PROTOCOL_PROFILE.to_generation_contract(), job_id=JOB)


def _seed(root: Path, *, prompts: int = 3, hotkeys=tuple(TOKENS)) -> None:
    from reliquary.corpus.audit_policy import MinerState
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure import corpus_record_store as record_store
    from reliquary.validator.eval_control import ORDER_TRUST_ID

    raw = h.manifest(JOB, max_new_tokens=4096)
    raw.update(prompt_count=prompts, slots_per_prompt=1, seed=5, sampling={**raw["sampling"], "draw": "gumbel_max"})
    h.put_raw(root, job_store._job_key(JOB), json.dumps(raw).encode())
    h.put_raw(root, record_store._miners_key(ORDER_TRUST_ID), json.dumps(
        {hotkey: MinerState(audited_passed=1000).to_dict() for hotkey in hotkeys}).encode())


async def _until(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def _executor(dispatcher) -> None:
    from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AuditResult

    document = {"executor_id": "e1", "model_id": "org/Frozen", "model_revision": "abc123",
                "provider_id": "p1", "host": "h1"}
    while True:
        dispatcher.heartbeat("e1", document)
        lease = dispatcher.claim(document, (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3))
        if lease is None:
            await asyncio.sleep(0.02)
            continue
        assert lease["protocol"] == AUDIT_PROTOCOL_V3
        scores = []
        for item in lease["items"]:
            assert item["draw"] and not item["proofs"]
            scores.append({"status": "ok", "chunks": [],
                           "draw": DIAGNOSTICS[item["tokens"][item["prompt_len"]]]})
        await dispatcher.result("e1", lease["lease_id"], AuditResult.model_validate({"scores": scores}))


def _signed(hotkey: str) -> dict:
    return {"job_id": JOB, "miner_hotkey": hotkey, "issued_at": int(time.time()),
            "nonce": secrets.token_hex(16), "signature": "ok"}


async def _submit(client, hotkey: str, lease: dict, *, token: int | None = None) -> dict:
    from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
    from tests.unit.test_corpus_service import CHECKPOINT, EOS, _faithful_prompt, _text_for

    length = 39 if hotkey == DRIFT else 99     # the near miss is short: its refusal is never charged
    tokens = [TOKENS[hotkey] if token is None else token] * length + [EOS]
    request = CorpusSubmissionRequest(
        job_id=JOB, miner_hotkey=hotkey, cursor=lease["slot"], prompt_index=lease["prompt_index"],
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(lease["prompt_index"]),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok")
    from reliquary.protocol.signatures import corpus_submission_id

    answer = (await client.post("/corpus/submit", json=request.model_dump())).json()
    return {**answer, "submission_id": corpus_submission_id(request)}


async def _start(mode: str, root: Path, run_dir: Path, entry) -> SimpleNamespace:
    """The front's routes over a real dispatcher and fake executor, the order judged in this process or the child."""
    from fastapi import FastAPI

    from reliquary.corpus.token_auction import AuctionParams, TokenAuction
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.protocol import profiles
    from reliquary.validator import corpus_service
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_validator import build_corpus_audit_wiring
    from reliquary.validator.eval_control import (
        ORDER_TRUST_ID, PairedAuditDispatcher, _suspect_reader, wire_order_judge,
    )
    from reliquary.validator.order_judge import OrderJudgeFront
    from tests.unit import corpus_split_fakes as fakes
    from tests.unit.test_corpus_service import _Renderer

    store, records = BucketJobStore(), BucketRecordStore()
    job, _ = await store.read_job(JOB)
    params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
        entry=entry, job=job, records=records, trust_id=ORDER_TRUST_ID)
    proof = profiles.toploc_proof(None)
    w = SimpleNamespace(entry=entry, cap=0.5, job=job, tokenizer=fakes.Tokenizer(), vocab_size=fakes.VOCAB,
                        proof=proof, stats=JobStats(), audit_params=params, miner_states=miner_states)
    dispatcher = PairedAuditDispatcher(directory=SimpleNamespace(revoke_locally=lambda e: None))
    scoring = asyncio.ensure_future(_executor(dispatcher))
    # A fake executor that broke says so at once, instead of a timeout on a verdict.
    scoring.add_done_callback(lambda t: t.cancelled() or (t.exception() and os._exit(3)
                                                          if print("executor failed:", repr(t.exception()),
                                                                   flush=True) else None))
    background = [scoring]
    front = None
    if mode == "child":
        front = OrderJudgeFront(dispatcher=dispatcher, run_dir=run_dir, spawn=True, sync_seconds=0.1,
                                stale_seconds=5.0, settle_seconds=0.5, auditor_kwargs=AUDITOR,
                                child_init="tests.unit.test_order_judge_draw:init_child")
        front.attach(w, records=records, store=store)
        background.append(asyncio.ensure_future(front.run()))
        await _until(lambda: front.live(JOB), 90, "the judge's seed")
    else:
        wire_order_judge(w, records=records, job_store=store, params=params, miner_states=miner_states,
                         beacon=beacon, round_at=round_at,
                         remote=dispatcher.view(job.checkpoint_repo, job.checkpoint_revision, proof, job.job_id,
                                                job.judge))
        for name, value in AUDITOR.items():
            setattr(w.auditor, name, value)

        async def settle():
            await w.auditor.seeded.wait()
            while True:
                await w.settler.settle_once()
                await asyncio.sleep(0.5)

        def on_accepted(submission_id: str) -> None:
            w.stats.accepted()
            w.auditor.enqueue(submission_id)

        w.on_accepted = on_accepted
        background += [asyncio.ensure_future(w.auditor.run()), asyncio.ensure_future(settle())]
        await _until(w.auditor.seeded.is_set, 30, "the seed")

    auction = TokenAuction(AuctionParams(ceiling=ALPHA / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                                         initial_leases=8), mean_tokens=100)
    app = FastAPI()
    app.include_router(corpus_service.build_corpus_router(
        job_id=JOB, store=store, tokenizer=fakes.Tokenizer(), renderer=_Renderer(),
        verify_signature=lambda request: True, verify_lease_signature=lambda request: True,
        prompt_job_for=corpus_service.prompt_job_for_spec, records=records, on_accepted=w.on_accepted,
        proof_chunk_tokens=None, vocab_size=fakes.VOCAB, is_banned=is_banned,
        suspect_until=_suspect_reader(miner_states, params), job=job, durable_records=False, auction=auction))
    return SimpleNamespace(app=app, w=w, job=job, store=store, front=front, background=background)


async def _stop(control) -> None:
    for task in control.background:
        task.cancel()
    await asyncio.wait(control.background, timeout=5)
    if control.front is not None:
        await asyncio.to_thread(control.front.stop, 5.0)


async def _run(mode: str, root: Path, run_dir: Path) -> dict:
    import httpx

    from reliquary.infrastructure.corpus_job_store import BucketJobStore

    control = await _start(mode, root, run_dir, _entry())
    app, w, job, front = control.app, control.w, control.job, control.front
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://front",
                                     timeout=30) as client:
            for hotkey in TOKENS:
                body = {**_signed(hotkey), "micro_alpha_per_mtoken": int(0.2 * 1e6),
                        "capacity": 2 if hotkey == REDO else 1}
                assert (await client.post("/corpus/ask", json=body)).json()["accepted"]

            async def lease(hotkey):
                return (await client.post("/corpus/lease", json=_signed(hotkey))).json()

            slots, ids = {}, {}
            for hotkey in (GOOD, CHEAT, DRIFT):
                granted = await lease(hotkey)
                assert granted["leased"], granted
                slots[hotkey] = (granted["prompt_index"], granted["slot"])
                answer = await _submit(client, hotkey, granted)
                assert answer["accepted"], answer
                ids[hotkey] = answer["submission_id"]

            def verdicts():
                return h.listed(root, JOB, "verdicts")

            await _until(lambda: set(ids.values()) <= set(verdicts()), 60, "the three verdicts")

            # The failed audits reopened their slots in the ledger the front leases from.
            leases, deadline, refusal = [], time.monotonic() + 30, None
            while len(leases) < 2 and time.monotonic() < deadline:
                granted = await lease(REDO)
                if granted.get("leased"):
                    leases.append(granted)
                else:
                    refusal = granted
                    await asyncio.sleep(0.1)
            assert sorted((g["prompt_index"], g["slot"]) for g in leases) == sorted(
                (slots[CHEAT], slots[DRIFT])), (leases, slots, refusal)
            redone = []
            for granted in leases:
                answer = await _submit(client, REDO, granted)
                assert answer["accepted"], answer
                redone.append(answer["submission_id"])
            await _until(lambda: set(redone) <= set(verdicts()), 60, "the redone verdicts")

            # The charged failure reaches the front's lease route: no new work for that hotkey.
            suspected = None
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                suspected = (await lease(CHEAT)).get("reason")
                if suspected == "miner_suspect":
                    break
                await asyncio.sleep(0.2)

            everything = {*ids.values(), *redone}
            await _until(lambda: everything <= set(h.settlement(root, JOB).get("settled") or ()), 90,
                         "every settlement")
            if front is not None:
                await _until(lambda: front.live(JOB) and sum(w.auditor.audit_reasons.values()) >= 5, 15,
                             "the snapshot's reasons")
            reasons = dict(w.auditor.audit_reasons)
            job_state = await client.get("/corpus/job")
            all_verdicts = verdicts()
    except BaseException:
        logging.getLogger(__name__).exception("the %s scenario failed", mode)
        raise
    finally:
        await _stop(control)

    def shown(sid: str) -> dict:
        verdict = all_verdicts[sid]
        return {key: verdict.get(key) for key in (
            "hotkey", "passed", "audited", "reason", "penalty", "forced_draw", "token_count", "lease_price",
            "lease_debit", "scored_by")}

    rewards: dict[str, float] = {}
    log = root / "archives" / "writes.jsonl"
    for row in map(json.loads, log.read_text().splitlines()):
        for hotkey, value in row["rewards"].items():
            rewards[hotkey] = rewards.get(hotkey, 0.0) + float(value)
    assert job_state.status_code == 200
    snapshot, _ = await BucketJobStore().read_ledgers(JOB)
    from reliquary.validator.corpus_service import rebuild_ledgers

    ledger = rebuild_ledgers(job, snapshot)
    return {"verdicts": {name: shown(ids[name]) for name in (GOOD, CHEAT, DRIFT)},
            "redone": sorted((shown(sid) for sid in redone), key=json.dumps),
            "reasons": reasons, "suspected": suspected, "rewards": rewards,
            "ledger": {"filled": ledger.slots.filled, "total": ledger.slots.total,
                       "attempts": sorted(ledger.slots.slot_attempt(p, 0) for p in range(3))}}


# -- sampled: q < 1, unaudited passes, the stake ---------------------------------

HONEST = ("5Hon1", "5Hon2")
SAMPLED_HOTKEYS = (*HONEST, CHEAT)
SAMPLED_PROMPTS = 18
SAMPLED = {"audit_q": 0.5, "audit_hold_seconds": 20.0}
SAMPLED_PERIOD_SECONDS = 60
ROUND_SECONDS = 3                       # ``corpus_split_harness.round_at``


async def _round_start(*, period: bool) -> None:
    """Just after a drand round begins (and with ``period``, a settlement period): each wave in one round, both
    waves in one period, since a record's stake counts its own period's."""
    from reliquary.validator import corpus_periods as cp

    while True:
        now = time.time()
        if now % ROUND_SECONDS < 0.3 and (not period or (now - cp.PERIOD_EPOCH) % cp.PERIOD_SECONDS < 3.0):
            return
        await asyncio.sleep(0.02)


async def _run_sampled(mode: str, root: Path, run_dir: Path) -> dict:
    import httpx

    from reliquary.corpus import draw_leases

    # The same slots in the same order in both runs: the same submission ids, so the same draws.
    draw_leases.secrets = SimpleNamespace(randbits=lambda bits: 11, token_hex=secrets.token_hex)
    control = await _start(mode, root, run_dir, _entry(**SAMPLED))
    labels: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=control.app), base_url="http://front",
                                     timeout=30) as client:
            for hotkey in SAMPLED_HOTKEYS:
                body = {**_signed(hotkey), "micro_alpha_per_mtoken": int(0.2 * 1e6), "capacity": 1}
                assert (await client.post("/corpus/ask", json=body)).json()["accepted"]

            async def wave(first: int, last: int, forged: bool) -> None:
                for k in range(first, last):
                    hotkey = SAMPLED_HOTKEYS[k % 3]
                    granted = (await client.post("/corpus/lease", json=_signed(hotkey))).json()
                    assert granted["leased"], granted
                    answer = await _submit(client, hotkey, granted, token=8 if forged and hotkey == CHEAT else 7)
                    assert answer["accepted"], answer
                    labels[answer["submission_id"]] = f"{hotkey}:{granted['prompt_index']}"

            # Audited first (no stake yet): what the second wave's stake then counts.
            await _round_start(period=True)
            await wave(0, 3, False)
            await _until(lambda: set(labels) <= set(h.listed(root, JOB, "verdicts")), 30, "the first wave")
            await _round_start(period=False)
            await wave(3, SAMPLED_PROMPTS, True)
            await _until(lambda: set(labels) <= set(h.listed(root, JOB, "verdicts")), 120, "every sampled verdict")
            await _until(lambda: set(labels) <= set(h.settlement(root, JOB).get("settled") or ()), 120,
                         "every sampled settlement")
            if control.front is not None:
                await _until(lambda: control.front.live(JOB)
                             and sum(control.w.auditor.audit_reasons.values()) > 0, 15, "the snapshot's reasons")
                await asyncio.sleep(0.5)
            reasons = dict(control.w.auditor.audit_reasons)
    finally:
        await _stop(control)
    verdicts, voided = h.listed(root, JOB, "verdicts"), h.listed(root, JOB, "voided")

    def shown(sid: str) -> dict:
        verdict = verdicts[sid]
        draw = verdict.get("draw") or {}
        return {**{key: verdict.get(key) for key in ("hotkey", "passed", "audited", "reason", "penalty", "forced_draw",
                                                     "token_count", "lease_price", "lease_debit", "scored_by")},
                "drawn": draw.get("drawn"), "voided": sid in voided}

    rewards: dict[str, float] = {}
    for row in map(json.loads, (root / "archives" / "writes.jsonl").read_text().splitlines()):
        for hotkey, value in row["rewards"].items():
            rewards[hotkey] = rewards.get(hotkey, 0.0) + float(value)
    return {"verdicts": {label: shown(sid) for sid, label in labels.items()}, "reasons": reasons,
            "rewards": rewards}


def _scenario(mode: str, root: str, run_dir: str, out: str, kind: str = "full") -> None:
    from reliquary.shared.log_levels import unmute_reliquary_loggers

    init_child()
    logging.basicConfig(level=logging.WARNING, force=True)
    unmute_reliquary_loggers()
    if kind == "sampled":
        _seed(Path(root), prompts=SAMPLED_PROMPTS, hotkeys=SAMPLED_HOTKEYS)
        result = asyncio.run(_run_sampled(mode, Path(root), Path(run_dir)))
    else:
        _seed(Path(root))
        result = asyncio.run(_run(mode, Path(root), Path(run_dir)))
    Path(out).write_text(json.dumps(result))
    os._exit(0)


def _outcome(mode: str, tmp_path: Path, monkeypatch, kind: str = "full") -> dict:
    base = tmp_path / f"{kind}-{mode}"
    root, run_dir = base / "bucket", base / "run"
    root.mkdir(parents=True)
    run_dir.mkdir(mode=0o700)
    monkeypatch.setenv(h.ROOT_ENV, str(root))
    monkeypatch.setenv(h.LATENCY_ENV, "0,0,0,0,0")
    monkeypatch.setenv(h.LOG_ENV, str(base / "log"))
    out = base / "out.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_scenario, args=(mode, str(root), str(run_dir), str(out), kind))
    process.start()
    process.join(280)
    if process.is_alive():
        process.kill()
        process.join(5)
    assert out.exists(), (base / "log").read_text()[-3000:] if (base / "log").exists() else "no log"
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    try:
        path = tmp_path_factory.mktemp("draw")
        return {mode: _outcome(mode, path, patch) for mode in ("single", "child")}
    finally:
        patch.undo()


def test_a_draw_order_ends_the_same_judged_in_the_child(outcomes):
    assert outcomes["child"] == outcomes["single"]


def test_the_child_passes_refuses_and_charges_as_the_draw_says(outcomes):
    verdicts = outcomes["child"]["verdicts"]
    assert verdicts[GOOD]["passed"] and verdicts[GOOD]["audited"] and verdicts[GOOD]["scored_by"] == ["e1"]
    assert verdicts[GOOD]["lease_price"] > 0
    for name in (CHEAT, DRIFT):
        assert not verdicts[name]["passed"] and verdicts[name]["reason"] == "draw_mismatch"
    assert verdicts[CHEAT]["forced_draw"]["penalty"] is True
    assert verdicts[DRIFT]["penalty"] is False
    assert outcomes["child"]["suspected"] == "miner_suspect"
    assert outcomes["child"]["reasons"] == {"full": 5}


def test_a_failed_audit_in_the_child_reopens_its_slot_for_the_fronts_leases(outcomes):
    child = outcomes["child"]
    assert all(v["passed"] and v["hotkey"] == REDO for v in child["redone"]) and len(child["redone"]) == 2
    # Each failure added an attempt: three samples kept, two more slots to fill, all filled.
    assert child["ledger"]["filled"] == child["ledger"]["total"] == 5
    assert child["ledger"]["attempts"] == [0, 1, 1]


def test_only_what_passed_is_paid(outcomes):
    rewards = outcomes["child"]["rewards"]
    assert rewards[GOOD] > 0 and rewards[REDO] > 0
    assert not rewards.get(CHEAT) and not rewards.get(DRIFT)


@pytest.fixture(scope="module")
def sampled(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    try:
        patch.setenv(h.BEACON_ENV, "const")
        patch.setenv(PERIOD_ENV, str(SAMPLED_PERIOD_SECONDS))
        path = tmp_path_factory.mktemp("sampled")
        return {mode: _outcome(mode, path, patch, "sampled") for mode in ("single", "child")}
    finally:
        patch.undo()


def _comparable(outcome: dict) -> dict:
    """All of it, but for one race the rule allows: a record of a hotkey just caught is audited either on its draw or
    as suspect, by whether the pass deciding it came after the failure. Audited and refused both ways."""
    reasons = dict(outcome["reasons"])
    caught = reasons.pop("drawn", 0) + reasons.pop("suspect", 0)
    verdicts = {label: {k: v for k, v in verdict.items() if not (verdict["hotkey"] == CHEAT and k == "drawn")}
                for label, verdict in outcome["verdicts"].items()}
    return {**outcome, "reasons": {**reasons, "drawn_or_suspect": caught}, "verdicts": verdicts}


def test_a_sampled_draw_order_ends_the_same_judged_in_the_child(sampled):
    assert _comparable(sampled["child"]) == _comparable(sampled["single"])


def test_the_sampled_run_exercises_unaudited_passes_the_stake_and_a_forfeit(sampled):
    child = sampled["child"]
    verdicts = child["verdicts"].values()
    honest = [v for v in verdicts if v["hotkey"] in HONEST]
    assert all(v["passed"] for v in honest)
    assert any(not v["audited"] for v in honest), "no unaudited pass"
    assert any(v["audited"] for v in honest), "no audit"
    assert child["reasons"].get("slow_stake"), "the stake never decided an audit"
    assert child["reasons"].get("drawn"), "the draw never decided an audit"
    assert not any(v["passed"] and not v["voided"] for v in verdicts if v["hotkey"] == CHEAT)
    assert not child["rewards"].get(CHEAT) and all(child["rewards"].get(hotkey, 0) > 0 for hotkey in HONEST)
