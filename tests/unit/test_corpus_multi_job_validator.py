"""One corpus validator process, one loaded model, several jobs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.validator.corpus_validator import multi_job_refusal
from tests.unit.test_corpus_service import (  # noqa: F401
    CHECKPOINT, EOS, _Tokenizer, _faithful_prompt, _manifest, _r2_client, _text_for,
    fake_r2, seeded_job,
)
from tests.unit.test_corpus_validator import _profile, fixed_drand_chain, wired_records  # noqa: F401


def _entry(task_id, job_id, cap=0.1, **params):
    return SimpleNamespace(task_id=task_id, job_id=job_id, mechanism="corpus-generation",
                           params={"cap": cap, **params}, contract=None)


def _job(job_id, **kw):
    fields = dict(job_id=job_id, checkpoint_repo="org/Frozen", checkpoint_revision="abc123",
                  checkpoint_sha256=CHECKPOINT)
    fields.update(kw)
    return SimpleNamespace(**fields)


def _same_proof(entry):
    return "toploc-a"


def test_http_shutdown_cancels_and_awaits_all_corpus_services():
    from reliquary.validator.corpus_validator import _run_corpus_services

    async def exercise():
        started = [asyncio.Event(), asyncio.Event()]
        cleaned = []
        warmed = asyncio.Event()

        async def service(index):
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.append(index)

        async def warm():
            warmed.set()

        class _Server:
            async def serve(self):
                await asyncio.gather(*(event.wait() for event in started), warmed.wait())
                await asyncio.sleep(0)

        await asyncio.wait_for(_run_corpus_services(
            _Server(), [service(0), warm(), service(1)]), timeout=1)
        assert sorted(cleaned) == [0, 1]
        assert all(task is asyncio.current_task() for task in asyncio.all_tasks())

    asyncio.run(exercise())


def test_corpus_service_failure_propagates_and_stops_the_server():
    from reliquary.validator.corpus_validator import _run_corpus_services

    async def exercise():
        serving = asyncio.Event()
        cleaned = []

        class _Server:
            async def serve(self):
                serving.set()
                try:
                    await asyncio.Future()
                finally:
                    await asyncio.sleep(0)
                    cleaned.append("server")

        async def failure():
            await serving.wait()
            raise RuntimeError("fixture corpus service failure")

        async def background():
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.append("background")

        with pytest.raises(RuntimeError, match="fixture corpus service failure"):
            await _run_corpus_services(_Server(), [failure(), background()])
        assert sorted(cleaned) == ["background", "server"]
        assert all(task is asyncio.current_task() for task in asyncio.all_tasks())

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX signal exit status")
def test_real_http_sigterm_awaits_background_cleanup_before_process_exit():
    script = textwrap.dedent("""
        import asyncio
        import os
        import signal
        import uvicorn
        from reliquary.validator.corpus_validator import _run_corpus_services

        async def app(scope, receive, send):
            pass

        async def main():
            server = uvicorn.Server(uvicorn.Config(
                app, host="127.0.0.1", port=0, lifespan="off", ws="none",
                log_level="critical"))

            async def background(index):
                try:
                    if index == 0:
                        while not server.started:
                            await asyncio.sleep(0.01)
                        os.kill(os.getpid(), signal.SIGTERM)
                    await asyncio.Future()
                finally:
                    await asyncio.sleep(0.02 * (index + 1))
                    print(f"background-cleaned-{index}", flush=True)

            await _run_corpus_services(server, [background(0), background(1)])

        asyncio.run(main())
    """)
    result = subprocess.run([sys.executable, "-u", "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == -signal.SIGTERM, result.stderr
    assert sorted(result.stdout.splitlines()) == ["background-cleaned-0", "background-cleaned-1"]


def test_jobs_on_one_checkpoint_start():
    pairs = [(_entry("corpus-math", "math"), _job("math")),
             (_entry("corpus-code", "code"), _job("code"))]
    assert multi_job_refusal(pairs, proof_of=_same_proof) is None


@pytest.mark.parametrize("field,value", [("checkpoint_repo", "org/Other"),
                                         ("checkpoint_revision", "def456"),
                                         ("checkpoint_sha256", "b" * 64)])
def test_jobs_on_different_checkpoints_refuse_naming_the_field(field, value):
    pairs = [(_entry("corpus-math", "math"), _job("math")),
             (_entry("corpus-code", "code"), _job("code", **{field: value}))]
    refusal = multi_job_refusal(pairs, proof_of=_same_proof)
    assert refusal is not None
    assert field in refusal and "math" in refusal and "code" in refusal


def test_different_toploc_proofs_refuse():
    pairs = [(_entry("corpus-math", "math"), _job("math")),
             (_entry("corpus-code", "code"), _job("code"))]
    refusal = multi_job_refusal(pairs, proof_of=lambda e: e.task_id)
    assert refusal is not None and "toploc" in refusal and "corpus-code" in refusal


def test_two_tasks_naming_one_job_refuse():
    pairs = [(_entry("corpus-math", "math"), _job("math")),
             (_entry("corpus-math-2", "math"), _job("math"))]
    refusal = multi_job_refusal(pairs, proof_of=_same_proof)
    assert refusal is not None and "math" in refusal


def test_the_default_proof_reads_each_entrys_own_contract():
    from reliquary.protocol.profiles import PROFILES, PROOF_SCHEME_TOPLOC, to_generation_contract
    from reliquary.cli.main import _with_enforced_toploc

    base = _with_enforced_toploc(to_generation_contract(next(iter(PROFILES))))
    other = {**base, "proofs": [dict(p) for p in base["proofs"]]}
    toploc = next(p for p in other["proofs"] if p["scheme"] == PROOF_SCHEME_TOPLOC)
    toploc["mant_mean_threshold"] = toploc["mant_mean_threshold"] + 10.0
    pairs = [(replace_contract(_entry("corpus-math", "math"), base), _job("math")),
             (replace_contract(_entry("corpus-code", "code"), other), _job("code"))]
    assert "toploc" in multi_job_refusal(pairs)
    pairs[1] = (replace_contract(_entry("corpus-code", "code"), base), _job("code"))
    assert multi_job_refusal(pairs) is None


def replace_contract(entry, contract):
    entry.contract = contract
    return entry


# --------------------------------------------------------------------------
# run_corpus_validator over two jobs: the real startup, stubbed at the GPU
# --------------------------------------------------------------------------


class _Stop(Exception):
    pass


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=200_000)


@pytest.fixture
def booted(seeded_job, fake_r2, wired_records, fixed_drand_chain, monkeypatch):
    """Drive the real `run_corpus_validator` for two jobs until the HTTP
    server would start serving; returns what it built."""
    import huggingface_hub
    import uvicorn

    import reliquary.corpus.encoding as encoding
    import reliquary.protocol.profiles as profiles
    import reliquary.protocol.signatures as signatures
    import reliquary.shared.modeling as modeling
    from reliquary.validator import corpus_auditor, corpus_settlement
    from reliquary.validator.corpus_validator import run_corpus_validator

    asyncio.run(job_store.write_job({**_manifest(), "job_id": "swe-v2"}, None, **fake_r2))
    loads = {"snapshot": 0, "model": 0, "tokenizer": 0}
    built = {"auditors": [], "settled": [], "settlers": [], "app": None}

    def snapshot(repo, revision=None):
        loads["snapshot"] += 1
        return "/nonexistent/checkpoint"

    def tokenizer(path):
        loads["tokenizer"] += 1
        return _Tokenizer()

    def model(path, **kw):
        loads["model"] += 1
        return _Model()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(encoding, "checkpoint_fingerprint", lambda d: CHECKPOINT)
    monkeypatch.setattr(modeling, "load_tokenizer", tokenizer)
    monkeypatch.setattr(modeling, "load_text_only_model", model)
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", _profile())
    # A fake hotkey signs nothing; the ban check sits behind the signature one.
    monkeypatch.setattr(signatures, "verify_corpus_signature", lambda request: True)

    async def run(self):
        built["auditors"].append(self)
        await asyncio.sleep(3600)

    async def settle_once(self):
        built["settled"].append((self._task_id, self._job_id, self._cap))
        built["settlers"].append(self)
        return None

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", run)
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", settle_once)

    class _Server:
        def __init__(self, config):
            built["app"] = config.app

        async def serve(self):
            await asyncio.sleep(0.05)
            import httpx

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=built["app"]),
                                         base_url="http://corpus") as client:
                built["runtime_at_start"] = (await client.get("/corpus/runtime-contract")).json()
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)

    jobs = [(_entry("corpus-math", "swe-v1", cap=0.1, audit_q=1.0), 0.1),
            (_entry("corpus-code", "swe-v2", cap=0.2, audit_q=0.5), 0.2)]
    with pytest.raises(_Stop):
        asyncio.run(run_corpus_validator(
            jobs=jobs, wallet=None, netuid=0, signer_client=None, http_host="127.0.0.1",
            http_port=0, set_weights=False, registration_gate=False,
        ))
    return SimpleNamespace(loads=loads, **built)


def test_the_model_is_loaded_once_for_both_jobs(booted):
    assert booted.loads == {"snapshot": 1, "model": 1, "tokenizer": 1}


def test_runtime_contract_attests_the_loaded_checkpoint_and_wired_jobs(booted):
    from reliquary.protocol.release_contract import canonical_sha256

    runtime = TestClient(booted.app).get("/corpus/runtime-contract")
    assert runtime.status_code == 200
    document = runtime.json()
    assert document["ready"] is False  # The test's server has stopped.
    assert booted.runtime_at_start["ready"] is True
    assert booted.runtime_at_start["audit_ready"] is True
    assert document["checkpoint_sha256"] == CHECKPOINT
    assert document["contract_sha256"] == canonical_sha256(document["contract"])
    assert document["registry_refresh_enabled"] is False
    assert sorted(document["jobs"]) == ["swe-v1", "swe-v2"]


def test_each_job_gets_its_own_auditor_on_one_shared_gpu_lock(booted):
    auditors = sorted(booted.auditors, key=lambda a: a._job_id)
    assert [a._job_id for a in auditors] == ["swe-v1", "swe-v2"]
    assert isinstance(auditors[0]._gpu_lock, asyncio.Lock)
    assert auditors[0]._gpu_lock is auditors[1]._gpu_lock
    assert auditors[0]._model is auditors[1]._model
    # Per-job audit params and per-job miner states.
    assert [a._params.q for a in auditors] == [1.0, 0.5]
    assert [a._miner_states._job_id for a in auditors] == ["swe-v1", "swe-v2"]


def test_each_task_settles_its_own_job_under_its_own_cap(booted):
    assert sorted(booted.settled) == [("corpus-code", "swe-v2", 0.2), ("corpus-math", "swe-v1", 0.1)]


def test_each_auditors_verdicts_feed_its_own_jobs_settler(booted):
    from reliquary.validator.corpus_settlement import SETTLE_FULL_LIST_SECONDS

    settlers = {s._job_id: s for s in booted.settlers}
    for auditor in booted.auditors:
        auditor._on_verdict("e" * 64, {"hotkey": "h", "token_count": 1, "passed": True})
    for job_id, settler in settlers.items():
        assert settler._full_list_every == SETTLE_FULL_LIST_SECONDS
        assert settler._unsettled == {"e" * 64}


def test_the_app_serves_both_jobs(booted):
    client = TestClient(booted.app)
    assert client.get("/corpus/jobs").json() == {"jobs": ["swe-v1", "swe-v2"]}
    assert client.get("/corpus/jobs/swe-v2/job").json()["job_id"] == "swe-v2"
    # The legacy read answers for the first task listed, as RELIQUARY_TASK_ID orders them.
    assert client.get("/corpus/job").json()["job_id"] == "swe-v1"


def test_a_ban_on_one_job_is_not_a_ban_on_the_other(booted):
    auditors = {a._job_id: a for a in booted.auditors}
    asyncio.run(auditors["swe-v1"]._miner_states.update(
        "5Hot", lambda m: replace(m, banned_until=4_102_444_800.0)))
    client = TestClient(booted.app)

    def submit(job_id):
        tokens = [7] * 16 + [EOS]
        return client.post("/corpus/submit", json={
            "job_id": job_id, "miner_hotkey": "5Hot", "cursor": 0, "prompt_index": 0,
            "checkpoint_sha256": CHECKPOINT, "rendered_prompt": _faithful_prompt(0),
            "completions": [{"tokens": tokens, "text": _text_for(tokens)}], "signature": "ok",
        }).json()

    assert submit("swe-v1")["reason"] == "miner_banned"
    # Past the ban check on the other job (the real renderer then judges the
    # stand-in prompt, which is not this test's concern).
    assert submit("swe-v2")["reason"] != "miner_banned"


def test_a_mismatched_second_job_refuses_before_any_download(
    seeded_job, fake_r2, monkeypatch
):
    import huggingface_hub

    from reliquary.validator.corpus_validator import run_corpus_validator

    asyncio.run(job_store.write_job(
        {**_manifest(), "job_id": "swe-v2", "checkpoint_sha256": "b" * 64}, None, **fake_r2))
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: pytest.fail("downloaded"))

    with pytest.raises(RuntimeError, match="checkpoint_sha256"):
        asyncio.run(run_corpus_validator(
            jobs=[(_entry("corpus-math", "swe-v1"), 0.1), (_entry("corpus-code", "swe-v2"), 0.1)],
            wallet=None, netuid=0, signer_client=None, http_host="127.0.0.1", http_port=0,
            set_weights=False, registration_gate=False,
        ))


# --------------------------------------------------------------------------
# Ledger v2: each job migrated and its seen index preloaded at startup
# --------------------------------------------------------------------------


@pytest.fixture
def v1_ledgers(_r2_client, monkeypatch):
    """Both jobs' ledgers still v1, and every router's seen index recorded."""
    from reliquary.infrastructure import corpus_job_store as store_module
    from reliquary.validator import corpus_service
    from tests.unit.test_corpus_ledger_migration import _digests

    ledgers = {}
    for job_id, salt in (("swe-v1", "a"), ("swe-v2", "b")):
        seen = _digests(5000, salt)
        ledgers[job_id] = {"schema": corpus_service.LEDGER_SCHEMA_V1,
                           "slots": {str(i): 8 for i in range(len(seen) // 8)},
                           "cursors": {"5Hot": 1}, "seen": seen}
        _r2_client.objects[f"reliquary/corpus/jobs/{job_id}/ledgers.json"] = (
            store_module._encode(ledgers[job_id]), '"seeded"')
    indexes = {}
    real = corpus_service.build_corpus_router

    def recording(**kwargs):
        indexes[kwargs["job_id"]] = kwargs.get("seen_index")
        return real(**kwargs)

    monkeypatch.setattr(corpus_service, "build_corpus_router", recording)
    return SimpleNamespace(ledgers=ledgers, indexes=indexes, r2=_r2_client)


def test_each_jobs_ledger_is_migrated_and_its_index_preloaded(v1_ledgers, booted):
    import json

    from reliquary.validator.corpus_service import LEDGER_SCHEMA_V2

    for job_id, v1 in v1_ledgers.ledgers.items():
        ledger = json.loads(v1_ledgers.r2.objects[f"reliquary/corpus/jobs/{job_id}/ledgers.json"][0])
        assert ledger["schema"] == LEDGER_SCHEMA_V2
        assert f"reliquary/corpus/jobs/{job_id}/ledgers.v1-backup.json" in v1_ledgers.r2.objects
        index = v1_ledgers.indexes[job_id]
        # Its own job's index, with that job's sealed digests and no other's.
        assert index._job_id == job_id and len(index) > 0
        sealed = set(index)
        assert sealed | set(ledger["seen_pending"]) == set(v1["seen"])
    assert not set(v1_ledgers.indexes["swe-v1"]) & set(v1_ledgers.indexes["swe-v2"])


def test_a_refused_start_migrates_no_ledger(v1_ledgers, seeded_job, fake_r2):
    """The manifests are checked against each other before any ledger is touched:
    a start that refuses leaves every v1 ledger exactly as it was."""
    from reliquary.validator.corpus_validator import run_corpus_validator

    asyncio.run(job_store.write_job(
        {**_manifest(), "job_id": "swe-v2", "checkpoint_sha256": "b" * 64}, None, **fake_r2))
    before = dict(v1_ledgers.r2.objects)

    with pytest.raises(RuntimeError, match="checkpoint_sha256"):
        asyncio.run(run_corpus_validator(
            jobs=[(_entry("corpus-math", "swe-v1"), 0.1), (_entry("corpus-code", "swe-v2"), 0.1)],
            wallet=None, netuid=0, signer_client=None, http_host="127.0.0.1", http_port=0,
            set_weights=False, registration_gate=False,
        ))
    after = {k: v for k, v in v1_ledgers.r2.objects.items() if "ledger" in k or "/seen/" in k}
    assert after == {k: v for k, v in before.items() if "ledger" in k or "/seen/" in k}


def test_the_miner_status_route_serves_each_job_from_its_own_wiring(booted):
    hotkey = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    auditor = next(a for a in booted.auditors if a._job_id == "swe-v2")
    # The auditor's real verdict report reaches the book.
    auditor._report("f" * 64, {"hotkey": hotkey, "passed": False, "audited": True,
                               "reason": "mant_err_median", "audited_at": 5.0,
                               "token_count": 9, "worst_exp": 0, "worst_mant_mean": 0.01,
                               "worst_mant_median": 0.02})
    client = TestClient(booted.app)
    v2 = client.get(f"/corpus/jobs/swe-v2/miners/{hotkey}")
    assert v2.status_code == 200, v2.text
    assert v2.json()["cap"] == 0.2 and v2.json()["failed"] == 1
    assert v2.json()["recent_failures"][0]["reason"] == "mant_err_median"
    assert "mant_median" in v2.json()["toploc_thresholds"]
    v1 = client.get(f"/corpus/jobs/swe-v1/miners/{hotkey}").json()
    assert v1["cap"] == 0.1 and v1["failed"] == 0 and v1["audit_state"] == "probation"
    assert client.get(f"/corpus/miners/{hotkey}").json()["job_id"] == "swe-v1"
