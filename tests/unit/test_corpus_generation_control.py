"""Pinned, disjoint rolling ownership using the existing scorer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from tests.unit.test_corpus_hot_jobs import ENV, _hot_entry, _hot_job
from tests.unit.test_corpus_validator import _profile


@pytest.fixture
def startup(monkeypatch, tmp_path):
    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.infrastructure import corpus_job_store, task_registry_store
    from reliquary.protocol import profiles
    from reliquary.validator import corpus_validator

    (tmp_path / "model.safetensors").write_bytes(b"pinned test checkpoint")
    profile = _profile()
    contract = {**profile.to_generation_contract(), "environments": {"src": ENV}, "protocol_version": 5}
    profile.to_generation_contract = lambda: contract
    entry = _hot_entry(job_id="order-gen-ops-fresh")
    entry.contract = contract
    job = _hot_job(job_id=entry.job_id, submit="scoped", episode=None,
                   checkpoint_sha256=checkpoint_fingerprint(tmp_path))
    calls = []
    facts = {"model_id": profile.model_id, "model_revision": profile.model_revision, "vocab_size": 10}

    async def read_registry(**kw):
        return {entry.task_id: entry}, "etag"

    async def read_job(job_id):
        return job, "etag"

    async def run(**kw):
        calls.append(kw)

    monkeypatch.setattr(task_registry_store, "read_registry", read_registry)
    monkeypatch.setattr(corpus_job_store, "read_job", read_job)
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", profile)
    monkeypatch.setattr(profiles, "profile_from_contract", lambda c: profile)
    monkeypatch.setattr(corpus_validator, "run_corpus_validator", run)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **kw: httpx.MockTransport(
        lambda request: httpx.Response(200, json=facts)))
    return SimpleNamespace(entry=entry, job=job, facts=facts, calls=calls, directory=tmp_path)


def invoke(startup):
    from reliquary.validator.generation_control import run_generation_control

    asyncio.run(run_generation_control(task_ids=[startup.entry.task_id],
                                       checkpoint_dir=str(startup.directory),
                                       gpu_run_dir=str(startup.directory)))


def test_shared_generation_start_validates_pins_and_selects_only_its_owner(startup):
    invoke(startup)
    (call,) = startup.calls
    assert call["generation_only"] and call["remote_audit"]
    assert call["set_weights"] is False
    assert call["split"].links == {}
    assert call["split"].fingerprint == startup.job.checkpoint_sha256
    assert call["jobs"] == [(startup.entry, 0.1)]
    assert asyncio.run(call["read_registry"]()) == {startup.entry.task_id: startup.entry}


def test_shared_generation_control_can_boot_empty_without_a_placeholder_task(startup):
    from reliquary.validator.generation_control import run_generation_control

    asyncio.run(run_generation_control(task_ids=[], checkpoint_dir=str(startup.directory),
                                       gpu_run_dir=str(startup.directory)))
    assert startup.calls[0]["jobs"] == []
    assert startup.calls[0]["generation_only"] is True


@pytest.mark.parametrize("invalid", ["legacy", "gpu", "checkpoint", "unscoped", "episode"])
def test_bad_owner_or_runtime_binding_never_starts_a_writer(startup, invalid):
    if invalid == "legacy":
        startup.entry.job_id = "order-ops-existing"
    elif invalid == "gpu":
        startup.facts["model_revision"] = "different"
    elif invalid == "checkpoint":
        startup.job.checkpoint_sha256 = "0" * 64
    elif invalid == "unscoped":
        startup.job.submit = "legacy"
    else:
        startup.job.episode = object()
    with pytest.raises(ValueError):
        invoke(startup)
    assert startup.calls == []


def test_generation_control_cli_preserves_explicit_initial_ownership(monkeypatch):
    from typer.testing import CliRunner
    from reliquary.cli.main import app
    from reliquary.validator import generation_control

    calls = []

    async def run(**kw):
        calls.append(kw)

    monkeypatch.setattr(generation_control, "run_generation_control", run)
    result = CliRunner().invoke(app, ["corpus", "generation-control", "--task-id", "fresh-task",
                                     "--checkpoint-dir", "/model", "--gpu-run-dir", "/scorer"])
    assert result.exit_code == 0, result.output
    assert calls[0]["task_ids"] == ["fresh-task"]
    assert calls[0]["http_host"] == "127.0.0.1"
    assert calls[0]["http_port"] == 8792


@pytest.mark.parametrize("initial_jobs", [False, True])
@pytest.mark.parametrize("scorer_ready", [False, True, None])
def test_shared_generation_mode_never_loads_another_model_and_keeps_audit_rechecks(
    seeded_job, fake_r2, fixed_drand_chain, wired_records, monkeypatch, initial_jobs,  # noqa: F811
    scorer_ready,
):
    import huggingface_hub
    import uvicorn
    from reliquary.infrastructure import corpus_executor_store, corpus_job_store
    from reliquary.protocol import profiles
    from reliquary.shared import modeling
    from reliquary.validator import corpus_auditor, corpus_feed, corpus_gpu, corpus_period_settlement
    from reliquary.validator.corpus_validator import run_corpus_validator
    from tests.unit.test_corpus_audit_remote import _R2
    from tests.unit.test_corpus_multi_job_validator import _entry
    from tests.unit.test_corpus_service import CHECKPOINT, PROMPT_SOURCE, _Tokenizer, _manifest

    job_id = "order-gen-ops-fresh"
    asyncio.run(corpus_job_store.write_job({**_manifest(), "job_id": job_id, "submit": "scoped"}, None, **fake_r2))
    profile = _profile()
    contract = {**profile.to_generation_contract(), "environments": {PROMPT_SOURCE: ENV}, "protocol_version": 5}
    profile.to_generation_contract = lambda: contract
    entry = _entry("fresh-task", job_id)
    entry.contract = contract
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", profile)
    monkeypatch.setattr(profiles, "profile_from_contract", lambda c: profile)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: _Tokenizer())
    monkeypatch.setattr(corpus_executor_store, "get_s3_client", lambda **kw: _R2())

    def forbidden(*args, **kw):
        raise AssertionError("a shared generation control must not download or load a GPU model")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(modeling, "load_text_only_model", forbidden)

    async def info(path):
        return {"vocab_size": 4096}

    async def gpu_info(self, path):
        assert path == "/info"
        facts = {"model_id": profile.model_id, "model_revision": profile.model_revision}
        if scorer_ready is not None:
            facts["ready"] = scorer_ready
        return facts

    async def idle(self):
        await asyncio.Future()

    async def settle(self):
        return None

    monkeypatch.setattr(corpus_gpu, "read_info", info)
    monkeypatch.setattr(corpus_feed.UdsClient, "get", gpu_info)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", settle)
    scored = []

    class Scorer:
        def __init__(self, *args, **kw):
            pass

        async def __call__(self, rows):
            scored.append(rows)
            return [("ok", ())], 0.0, 0.0

    monkeypatch.setattr(corpus_gpu, "GpuScorer", Scorer)

    class Stop(Exception):
        pass

    class Server:
        def __init__(self, config):
            self.app = config.app

        async def serve(self):
            await asyncio.sleep(0)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://corpus") as client:
                observed = (await client.get("/corpus/runtime-contract")).json()
            assert observed["audit_ready"] is (scorer_ready is not False)
            assert observed["ready"] is (scorer_ready is not False)
            runtime = self.app.state.corpus_runtime_contract
            assert runtime["execution_scope"] == "generation-operations"
            assert runtime["durable_executor_attempts"] is True
            remote = self.app.state.corpus_audit_remote
            assert remote._attempt_store is not None
            assert self.app.state.corpus_jobs._wire_retired
            assert self.app.state.corpus_jobs._finals is not None
            assert bool(self.app.state.corpus_jobs.served) == initial_jobs
            for wiring in self.app.state.corpus_jobs.served.values():
                assert wiring.settler._archives._served_only
                wiring.settler._archives.refuse_unserved(wiring.entry.task_id)
                with pytest.raises(RuntimeError, match="refusing to archive"):
                    wiring.settler._archives.refuse_unserved("corpus-legacy")
            assert await remote._local_scores([{"tokens": [1, 2], "prompt_len": 1, "proofs": []}]) == [("ok", ())]
            raise Stop()

    monkeypatch.setattr(uvicorn, "Server", Server)
    split = SimpleNamespace(directory="/pinned", fingerprint=CHECKPOINT, proof=profile.proofs[0],
                            run_dir="/shared", links={})
    with pytest.raises(Stop):
        asyncio.run(run_corpus_validator(jobs=[(entry, 0.1)] if initial_jobs else [], wallet=None, netuid=0,
                                        signer_client=None, http_host="127.0.0.1", http_port=0,
                                        set_weights=False, registration_gate=False, remote_audit=True,
                                        generation_only=True, split=split))
    assert scored == [[([1, 2], 1, [])]]


from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: E402,F401
from tests.unit.test_corpus_validator import fixed_drand_chain, wired_records  # noqa: E402,F401
