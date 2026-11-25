"""R1: the corpus validator's job set changes without a restart.

A new active corpus entry on this process's model is wired, one on another
model is ignored, one the binary cannot serve is refused (never a crash); a
retired entry stops admitting at once, keeps draining, then is unwired.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest

from reliquary.validator.corpus_hot_jobs import (
    OTHER_MODEL,
    REFUSED,
    CorpusJobSet,
    hot_job_refusal,
    job_drained,
)
from reliquary.validator.corpus_service import CorpusJobRoutes, build_corpus_jobs_router
from tests.unit.test_corpus_service import CHECKPOINT, _manifest, _r2_client, fake_r2  # noqa: F401
from tests.unit.test_corpus_validator import fixed_drand_chain, wired_records  # noqa: F401


# --------------------------------------------------------------------------
# hot_job_refusal
# --------------------------------------------------------------------------

ENV = {"prompt_template": {"id": "t"}, "max_new_tokens": 64}
PROOF = SimpleNamespace(scheme="toploc-v1", mode="enforce")


def _profile(model="org/Frozen", revision="abc123", proof=PROOF):
    return SimpleNamespace(model_id=model, model_revision=revision, proofs=(proof,))


def _hot_entry(task_id="corpus-b", job_id="job-b", env=None, status="active", cap=0.1,
               model="org/Frozen", revision="abc123", protocol_version=5):
    return SimpleNamespace(
        task_id=task_id, job_id=job_id, mechanism="corpus-generation", status=status,
        params={"cap": cap, "settlement": "period-ema-v1"},
        contract={"environments": {"src": env or ENV}, "protocol_version": protocol_version,
                  "model_id": model, "model_revision": revision},
    )


def _hot_job(job_id="job-b", **kw):
    fields = dict(job_id=job_id, prompt_source="src", checkpoint_repo="org/Frozen",
                  checkpoint_revision="abc123", checkpoint_sha256=CHECKPOINT)
    fields.update(kw)
    return SimpleNamespace(**fields)


def _refusal(entry, job=None, process_contract=None, **kw):
    import reliquary.protocol.profiles as profiles

    def toploc(profile):
        return profile.proofs[0]

    real = profiles.toploc_proof
    profiles.toploc_proof = toploc
    try:
        return hot_job_refusal(
            entry, job or _hot_job(), process_profile=kw.pop("process_profile", _profile()),
            process_contract=process_contract or {"environments": {"src": ENV},
                                                  "protocol_version": 5},
            fingerprint=kw.pop("fingerprint", CHECKPOINT),
            profile_of=lambda e: _profile(e.contract["model_id"], e.contract["model_revision"]),
            generation_only=kw.pop("generation_only", False),
        )
    finally:
        profiles.toploc_proof = real


def test_an_entry_on_this_model_with_this_environment_is_admitted():
    assert _refusal(_hot_entry()) is None


def test_generation_ownership_is_disjoint_from_legacy_and_other_orders():
    from reliquary.validator.corpus_hot_jobs import generation_entry_screen, order_entry_screen

    entry = _hot_entry(job_id="order-gen-ops-new")
    job = _hot_job(job_id=entry.job_id, submit="scoped")
    assert order_entry_screen(entry)[0] == OTHER_MODEL
    assert generation_entry_screen(entry) is None
    assert _refusal(entry, job, generation_only=True) is None
    assert _refusal(entry, job)[0] == OTHER_MODEL
    for job_id in ("legacy-job", "order-ops-old", "order-gen-customer", "order-eval-ops-new"):
        assert generation_entry_screen(_hot_entry(job_id=job_id))[0] == OTHER_MODEL


def test_generation_ownership_keeps_pin_and_scoped_single_turn_requirements():
    entry = _hot_entry(job_id="order-gen-ops-new")
    job = _hot_job(job_id=entry.job_id, submit="scoped")
    assert _refusal(entry, _hot_job(job_id=entry.job_id, submit="scoped", episode=object()),
                    generation_only=True)[0] == REFUSED
    assert _refusal(entry, _hot_job(job_id=entry.job_id, submit="legacy"),
                    generation_only=True)[0] == REFUSED
    assert _refusal(entry, job, generation_only=True, fingerprint="0" * 64)[0] == REFUSED


def test_generation_ownership_uses_the_existing_configured_admin_prefix(monkeypatch):
    from reliquary.validator.corpus_hot_jobs import generation_entry_screen, order_entry_screen

    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "pilot-")
    entry = _hot_entry(job_id="pilot-gen-ops-new")
    assert generation_entry_screen(entry) is None
    assert order_entry_screen(entry)[0] == OTHER_MODEL
    assert generation_entry_screen(_hot_entry(job_id="order-gen-ops-new"))[0] == OTHER_MODEL


def test_an_entry_for_another_model_is_ignored_not_refused():
    kind, why = _refusal(_hot_entry(model="org/Other"))
    assert kind == OTHER_MODEL and "org/Other" in why
    kind, _ = _refusal(_hot_entry(revision="def456"))
    assert kind == OTHER_MODEL


def test_an_entry_whose_environment_the_process_renders_differently_is_refused():
    kind, why = _refusal(_hot_entry(env={**ENV, "max_new_tokens": 128}))
    assert kind == REFUSED and "src" in why


def test_an_entry_whose_source_the_process_does_not_declare_is_refused():
    kind, _ = _refusal(_hot_entry(), process_contract={"environments": {}, "protocol_version": 5})
    assert kind == REFUSED


def test_a_job_on_another_checkpoint_than_the_loaded_one_is_refused():
    kind, why = _refusal(_hot_entry(), fingerprint="b" * 64)
    assert kind == REFUSED and "fingerprint" in why


def test_a_contract_less_entry_is_refused():
    entry = _hot_entry()
    entry.contract = None
    assert _refusal(entry)[0] == REFUSED


# --------------------------------------------------------------------------
# CorpusJobSet over fakes
# --------------------------------------------------------------------------


class _Router:
    def __init__(self, job_id):
        self.job_id = job_id
        self.submits = []

    async def corpus_job(self):
        return {"job_id": self.job_id}

    async def corpus_cursor(self, hotkey):
        return {"hotkey": hotkey, "cursor": 0}

    async def corpus_next(self, hotkey):
        return {"cursor": 0}

    async def skip_corpus(self, request):
        raise AssertionError("not reached")

    async def submit_corpus(self, request):
        self.submits.append(request)
        raise AssertionError("not reached")


class _Harness:
    def __init__(self, entries, *, admit=None, wire_error=None, clock=None, screen=None):
        self.entries = entries
        self.routes = CorpusJobRoutes()
        self.wired, self.running, self.cancelled = [], [], []
        self.drained = {}
        self.caps = {}
        self.wire_error = wire_error
        # Raised by the registry read, and by every manifest read.
        self.registry_error = self.read_error = None
        # Awaited by every manifest read first: what a miner sees while a pass works on a job.
        self.mid_pass = None

        async def wire(entry, cap, job):
            if self.wire_error is not None:
                raise self.wire_error
            self.wired.append(entry.task_id)
            settler = SimpleNamespace(set_cap=lambda c, t=entry.task_id: self.caps.__setitem__(t, c))
            return SimpleNamespace(entry=entry, cap=cap, job=job, settler=settler)

        async def forever(name):
            self.running.append(name)
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise

        async def read_entries():
            if self.registry_error is not None:
                raise self.registry_error
            return {e.task_id: e for e in self.entries}

        async def read_job(job_id):
            if self.mid_pass is not None:
                await self.mid_pass()
            if self.read_error is not None:
                raise self.read_error
            return _hot_job(job_id)

        async def drained(wiring):
            return self.drained.get(wiring.entry.job_id, False)

        self.set = CorpusJobSet(
            routes=self.routes, router_for=lambda w: _Router(w.entry.job_id), wire=wire,
            jobs_of=lambda w: [forever(f"audit:{w.entry.job_id}"), forever(f"settle:{w.entry.job_id}")],
            read_entries=read_entries, read_job=read_job,
            admit=admit or (lambda entry, job: None), drained=drained, screen=screen,
            **({} if clock is None else {"clock": clock}),
        )

    def client(self):
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(build_corpus_jobs_router(self.routes, legacy=True))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_a_new_active_entry_is_wired_routed_and_started():
    async def go():
        h = _Harness([_hot_entry()])
        await h.set.refresh()
        await asyncio.sleep(0)
        assert h.wired == ["corpus-b"]
        assert sorted(h.running) == ["audit:job-b", "settle:job-b"]
        async with h.client() as client:
            assert (await client.get("/corpus/jobs")).json()["jobs"] == ["job-b"]
            assert (await client.get("/corpus/jobs/job-b/job")).json() == {"job_id": "job-b"}
        await h.set.refresh()
        assert h.wired == ["corpus-b"]

    asyncio.run(go())


def test_an_entry_for_another_model_is_logged_once(caplog):
    async def go():
        h = _Harness([_hot_entry()], admit=lambda e, j: (OTHER_MODEL, "another model"))
        with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_hot_jobs"):
            await h.set.refresh()
            await h.set.refresh()
        assert h.wired == []
        assert [r.message for r in caplog.records].count("corpus task corpus-b ignored: another model") == 1

    asyncio.run(go())


def test_an_entry_whose_wiring_raises_is_refused_once_and_nothing_crashes(caplog):
    from reliquary.validator.corpus_service import CorpusPromptSourceError

    async def go():
        h = _Harness([_hot_entry()], wire_error=CorpusPromptSourceError("no renderer"))
        with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_hot_jobs"):
            await h.set.refresh()
            await h.set.refresh()
        assert h.set.served == {}
        refusals = [r for r in caplog.records if "refused" in r.message]
        assert len(refusals) == 1 and "job-b" in refusals[0].message

    asyncio.run(go())


def test_retired_stops_admission_drains_then_unwires():
    async def go():
        entry = _hot_entry()
        h = _Harness([entry])
        await h.set.refresh()
        await asyncio.sleep(0)
        h.entries = [_hot_entry(status="retired")]
        await h.set.refresh()
        # Admission stopped at once; the job's reads and its tasks are still up.
        async with h.client() as client:
            for method, path, body in (
                ("get", "/corpus/jobs/job-b/next/5Hot", None),
                ("get", "/corpus/next/5Hot", None),
            ):
                response = await getattr(client, method)(path)
                assert response.status_code == 410 and response.json() == {"detail": "job_retired"}
            assert (await client.get("/corpus/jobs/job-b/job")).status_code == 200
            listed = (await client.get("/corpus/jobs")).json()
            assert listed["jobs"] == []
            assert {o["job_id"]: o["state"] for o in listed["orders"]}["job-b"] == "draining"
        assert "job-b" in h.set.served and h.cancelled == []
        # Drained: its auditor and settler stop and its routes leave.
        h.drained["job-b"] = True
        await h.set.refresh()
        await asyncio.sleep(0)
        assert h.set.served == {} and sorted(h.cancelled) == ["audit:job-b", "settle:job-b"]
        assert "job-b" in h.set.finished
        async with h.client() as client:
            assert (await client.get("/corpus/jobs/job-b/next/5Hot")).status_code == 410
        # Never wired again, even while its entry still reads retired.
        await h.set.refresh()
        assert h.wired == ["corpus-b"]

    asyncio.run(go())


def test_pause_keeps_draining_and_resume_reuses_the_same_running_job():
    async def go():
        entry = _hot_entry(cap=0)
        h = _Harness([entry])
        await h.set.refresh()
        await asyncio.sleep(0)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def admitted_write():
            async with h.routes.admission("job-b"):
                entered.set()
                await release.wait()

        pending = asyncio.create_task(admitted_write())
        await entered.wait()
        entry.admission = "paused"
        await h.set.refresh()
        await h.set.refresh()
        assert h.routes.admission_pending("job-b")
        assert h.cancelled == [] and h.wired == ["corpus-b"]
        async with h.client() as client:
            assert (await client.get("/corpus/jobs/job-b/next/5Hot")).status_code == 409
            assert (await client.get("/corpus/next/5Hot")).status_code == 409
            assert (await client.get("/corpus/jobs/job-b/job")).status_code == 200
            assert (await client.get("/corpus/jobs/job-b/cursor/5Hot")).status_code == 200
            assert (await client.get("/corpus/jobs")).json()["jobs"] == []
            paused = (await client.get("/corpus/jobs")).json()["orders"]
            assert [(o["job_id"], o["state"]) for o in paused if o["job_id"] == "job-b"] == [("job-b", "paused")]
        release.set()
        await pending
        h.drained["job-b"] = True
        await h.set.refresh()
        assert "job-b" in h.set.served and h.cancelled == []
        entry.admission = "open"
        await h.set.refresh()
        async with h.client() as client:
            assert (await client.get("/corpus/jobs/job-b/next/5Hot")).status_code == 200
            assert (await client.get("/corpus/jobs")).json()["jobs"] == ["job-b"]
        assert h.wired == ["corpus-b"] and h.caps == {}
        for tasks in h.set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())


def test_a_retired_job_refuses_submit_and_skip_with_410():
    from tests.unit.test_corpus_multi_job_service import _body

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")])
        await h.set.refresh()
        h.entries = [_hot_entry(job_id="swe-v1", status="retired")]
        await h.set.refresh()
        async with h.client() as client:
            for path in ("/corpus/submit", "/corpus/jobs/swe-v1/submit"):
                response = await client.post(path, json=_body("swe-v1"))
                assert response.status_code == 410, path
                assert response.json() == {"detail": "job_retired"}
            skip = {"job_id": "swe-v1", "miner_hotkey": "5Hot", "cursor": 0, "prompt_index": 0,
                    "to_cursor": 1, "signature": "ok"}
            for path in ("/corpus/skip", "/corpus/jobs/swe-v1/skip"):
                response = await client.post(path, json=skip)
                assert response.status_code == 410, (path, response.text)

    asyncio.run(go())


def test_a_paused_job_refuses_legacy_and_scoped_submit_and_skip_before_storage():
    from tests.unit.test_corpus_multi_job_service import _body

    async def go():
        entry = _hot_entry(job_id="swe-v1")
        entry.admission = "paused"
        h = _Harness([entry])
        await h.set.refresh()
        async with h.client() as client:
            skip = {"job_id": "swe-v1", "miner_hotkey": "5Hot", "cursor": 0, "prompt_index": 0,
                    "to_cursor": 1, "signature": "ok"}
            for path, body in (("/corpus/submit", _body("swe-v1")),
                               ("/corpus/jobs/swe-v1/submit", _body("swe-v1")),
                               ("/corpus/skip", skip), ("/corpus/jobs/swe-v1/skip", skip)):
                response = await client.post(path, json=body)
                assert response.status_code == 409 and response.json() == {"detail": "job_paused"}

    asyncio.run(go())


def test_a_cap_change_reaches_the_running_settler():
    async def go():
        h = _Harness([_hot_entry(cap=0.1)])
        await h.set.refresh()
        h.entries = [_hot_entry(cap=0.25)]
        await h.set.refresh()
        assert h.caps == {"corpus-b": 0.25}
        assert h.set.served["job-b"].cap == 0.25

    asyncio.run(go())


def test_an_unreadable_registry_changes_nothing():
    async def go():
        h = _Harness([_hot_entry()])
        await h.set.refresh()

        async def broken():
            raise OSError("r2 down")

        h.set._read_entries = broken
        await h.set.refresh()
        assert list(h.set.served) == ["job-b"] and not h.routes.retired

    asyncio.run(go())


def test_a_failing_background_task_is_raised_out_of_run():
    class Halted(Exception):
        pass

    async def go():
        routes = CorpusJobRoutes()

        async def boom():
            raise Halted()

        job_set = CorpusJobSet(routes=routes, router_for=None, wire=None,
                               jobs_of=lambda w: [boom()], refresh_every_seconds=0.01)
        job_set.adopt(SimpleNamespace(entry=SimpleNamespace(task_id="t", job_id="j")))
        await job_set.run()

    with pytest.raises(Halted):
        asyncio.run(go())


def test_job_drained_needs_every_submission_judged_and_every_verdict_settled():
    class Records:
        def __init__(self):
            self.verdicts, self.state = [], {}

        async def list_verdict_ids(self, job_id):
            return self.verdicts

        async def list_submission_ids(self, job_id):
            return ["s1"]

        async def read_job(self, job_id):
            from reliquary.corpus.job import parse_job

            return parse_job(_manifest()), None

        async def read_ledgers(self, job_id):
            return {"schema": "reliquary/corpus-ledgers/v2", "slots": {},
                    "cursors": {}, "seen_pending": [], "seen_segments": []}, None

        async def read_settlement(self, job_id):
            return self.state, None

    class Auditor:
        pending = ["s1"]

        async def pending_ids(self):
            return self.pending

    records, auditor = Records(), Auditor()

    def drained():
        return asyncio.run(job_drained(auditor=auditor, records=records, job_id="j"))

    assert drained() is False
    auditor.pending, records.verdicts = [], ["s1"]
    assert drained() is False
    records.state = {"settled": ["s1"], "pending": {"window": 3}}
    assert drained() is False
    records.state = {"settled": ["s1"], "pending": None}
    assert drained() is True


# --------------------------------------------------------------------------
# The real run_corpus_validator: boot one job, hot-add a second, retire it
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


_SS58 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def _registry_entry(tmp_path, suffix, task_id, job_id, status="active"):
    from dataclasses import replace

    from reliquary.shared.task_registry import TaskEntry
    import json

    raw = json.loads((tmp_path / f"entry{suffix}.json").read_text())
    entry = TaskEntry(**{**raw, "task_id": task_id, "job_id": job_id})
    return replace(entry, status=status, retired_at=1 if status == "retired" else None)


def test_a_hot_added_job_serves_and_a_retired_job_drains_without_a_restart(
    tmp_path, fake_r2, wired_records, fixed_drand_chain, monkeypatch,  # noqa: F811
):
    import huggingface_hub
    import uvicorn

    import reliquary.corpus.encoding as encoding
    import reliquary.protocol.profiles as profiles
    import reliquary.shared.modeling as modeling
    from reliquary.protocol.profiles import profile_from_contract
    from reliquary.validator import corpus_auditor, corpus_period_settlement
    from reliquary.validator.corpus_validator import run_corpus_validator
    from reliquary.validator.task_config import merge_corpus_contracts
    from tests.unit.test_corpus_multi_job_startup import _declare

    contracts = _declare(tmp_path, fake_r2)
    merged = merge_corpus_contracts({"corpus-math": contracts["a"], "corpus-code": contracts["b"]})
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", profile_from_contract(merged))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None: "/x")
    monkeypatch.setattr(encoding, "checkpoint_fingerprint", lambda d: CHECKPOINT)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: SimpleNamespace())
    monkeypatch.setattr(modeling, "load_text_only_model", lambda path, **kw: _Model())

    runs, settles = [], []

    async def idle(self):
        runs.append(self._job_id)
        await asyncio.sleep(3600)

    async def settle_once(self):
        settles.append((self._task_id, self._cap))

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", settle_once)

    math = _registry_entry(tmp_path, "-a", "corpus-math", "math-v1")
    code = _registry_entry(tmp_path, "-b", "corpus-code", "code-v1")
    other = _registry_entry(tmp_path, "-b", "corpus-other", "other-v1")
    other = SimpleNamespace(**{**{f: getattr(other, f) for f in other.__slots__},
                               "contract": {**other.contract, "model_id": "org/Elsewhere"}})
    registry = {"corpus-math": math}

    async def read_entries():
        return dict(registry)

    seen = {}

    class _Server:
        def __init__(self, config):
            self.app = config.app

        async def serve(self):
            job_set = self.app.state.corpus_jobs
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                seen["before"] = (await client.get("/corpus/jobs")).json()
                registry.update({"corpus-code": code, "corpus-other": other})
                await job_set.refresh()
                await asyncio.sleep(0.01)
                seen["after_add"] = (await client.get("/corpus/jobs")).json()
                seen["code_job"] = (await client.get("/corpus/jobs/code-v1/job")).json()
                seen["code_contract"] = (await client.get("/corpus/jobs/code-v1/contract")).json()
                seen["legacy"] = (await client.get("/corpus/job")).json()["job_id"]
                seen["status_open"] = (await client.get("/corpus/jobs/code-v1/status")).json()
                seen["miner_hot"] = (await client.get(f"/corpus/jobs/code-v1/miners/{_SS58}")).json()
                seen["miner_boot"] = (await client.get(f"/corpus/miners/{_SS58}")).json()
                registry["corpus-code"] = _registry_entry(
                    tmp_path, "-b", "corpus-code", "code-v1", status="retired")
                await job_set.refresh()
                await job_set.refresh()
                seen["retired_next"] = (await client.get("/corpus/jobs/code-v1/next/5Hot")).status_code
                seen["served_after_drain"] = sorted(job_set.served)
                seen["status_drained"] = (await client.get("/corpus/jobs/code-v1/status")).json()
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)
    with pytest.raises(_Stop):
        asyncio.run(run_corpus_validator(
            entry=_registry_entry(tmp_path, "-a", "corpus-math", "math-v1"), cap=0.1,
            wallet=None, netuid=0, signer_client=None, http_host="127.0.0.1", http_port=0,
            set_weights=False, registration_gate=False, read_registry=read_entries,
            refresh_every_seconds=3600,
        ))
    assert seen["before"]["jobs"] == ["math-v1"]
    assert seen["after_add"]["jobs"] == ["code-v1", "math-v1"]
    assert seen["code_job"]["job_id"] == "code-v1"
    assert seen["code_contract"] == contracts["b"]
    assert seen["legacy"] == "math-v1"
    assert sorted(runs) == ["code-v1", "math-v1"]
    assert ("corpus-code", 0.1) in settles
    assert seen["retired_next"] == 410
    # Nothing was submitted, so the retired job drained at once and left.
    assert seen["served_after_drain"] == ["math-v1"]
    assert seen["status_open"]["state"] == "open"
    assert seen["status_open"]["submissions_accepted"] == 0
    assert seen["status_drained"]["state"] == "drained"
    # The miner status route serves a boot job and a hot-added one alike.
    assert seen["miner_hot"]["job_id"] == "code-v1" and seen["miner_hot"]["cap"] == 0.1
    assert seen["miner_hot"]["audit_state"] == "probation"
    assert seen["miner_boot"]["job_id"] == "math-v1"


# --------------------------------------------------------------------------
# Review fixes: I1 (drain race), I8 (transient wiring errors)
# --------------------------------------------------------------------------


def test_a_submit_in_flight_holds_the_job_wired_until_it_returns():
    from tests.unit.test_corpus_multi_job_service import _body

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")])
        await h.set.refresh()
        router = h.routes.routers["swe-v1"]
        release = asyncio.Event()

        async def slow_submit(request):
            await release.wait()
            return {"reason": "accepted", "accepted": True}

        router.submit_corpus = slow_submit
        h.drained["swe-v1"] = True
        async with h.client() as client:
            submit = asyncio.ensure_future(client.post("/corpus/submit", json=_body("swe-v1")))
            for _ in range(20):
                await asyncio.sleep(0)
            assert h.routes.in_flight["swe-v1"] == 1
            h.entries = [_hot_entry(job_id="swe-v1", status="retired")]
            await h.set.refresh()
            await h.set.refresh()
            assert "swe-v1" in h.set.served  # the admitted submit is still running
            release.set()
            assert (await submit).status_code == 200
        assert h.routes.in_flight["swe-v1"] == 0
        await h.set.refresh()
        assert "swe-v1" not in h.set.served

    asyncio.run(go())


def test_a_job_is_never_unwired_in_the_refresh_that_retired_it():
    async def go():
        h = _Harness([_hot_entry()])
        await h.set.refresh()
        h.drained["job-b"] = True
        h.entries = [_hot_entry(status="retired")]
        await h.set.refresh()
        assert "job-b" in h.set.served
        await h.set.refresh()
        assert "job-b" not in h.set.served

    asyncio.run(go())


def test_a_transient_wiring_error_is_retried_on_the_next_refresh():
    async def go():
        h = _Harness([_hot_entry()], wire_error=OSError("r2 timeout"))
        await h.set.refresh()
        assert h.set.served == {}
        h.wire_error = None
        await h.set.refresh()
        assert list(h.set.served) == ["job-b"]

    asyncio.run(go())


def test_a_job_the_registry_lists_but_not_wired_yet_is_not_ready_never_unknown():
    # A restart re-wires its jobs one by one: a miner asking for one meanwhile hears a 503 it retries, not the
    # 404 that stops it and cancels its running episodes.
    from tests.unit.test_corpus_multi_job_service import _body

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], wire_error=OSError("r2 timeout"))
        async with h.client() as client:
            # Before the registry is read: any job may be one about to be wired.
            for response in (await client.get("/corpus/jobs/swe-v1/job"),
                             await client.post("/corpus/jobs/swe-v1/submit", json=_body("swe-v1"))):
                assert response.status_code == 503 and response.json() == {"detail": "job_not_ready"}
                assert response.headers["Retry-After"] == "10"
            await h.set.refresh()
            # Its wiring failed for now and is retried: still not ready. A job the registry does not list is unknown.
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503
            unknown = await client.get("/corpus/jobs/other/job")
            assert unknown.status_code == 404 and unknown.json() == {"detail": "corpus_job_not_served"}
            h.wire_error = None
            await h.set.refresh()
            assert (await client.get("/corpus/jobs/swe-v1/job")).json() == {"job_id": "swe-v1"}
            assert h.routes.restoring == set()

    asyncio.run(go())


def test_a_job_refused_for_good_is_unknown_not_waited_for():
    from reliquary.validator.corpus_service import CorpusPromptSourceError

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], wire_error=CorpusPromptSourceError("no renderer"))
        await h.set.refresh()
        async with h.client() as client:
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 404
        h2 = _Harness([_hot_entry(job_id="swe-v1")], admit=lambda e, j: (OTHER_MODEL, "another model"))
        await h2.set.refresh()
        assert not h2.routes.not_ready("swe-v1")

    asyncio.run(go())


def test_the_order_list_answers_503_until_the_registry_is_read_then_shows_jobs_being_wired():
    # An order follower stops the miner of an order missing from two reads in a row: before the registry is read the
    # list would miss every order, and a restart wires them one by one for minutes. A failed read it retries.
    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], wire_error=OSError("r2 timeout"))
        async with h.client() as client:
            booting = await client.get("/corpus/jobs")
            assert booting.status_code == 503 and booting.json() == {"detail": "orders_not_ready"}
            assert booting.headers["Retry-After"] == "10"
            await h.set.refresh()
            listed = (await client.get("/corpus/jobs")).json()
            assert listed["jobs"] == [] and listed["orders"] == [{"job_id": "swe-v1", "state": "restoring"}]
            h.wire_error = None
            await h.set.refresh()
            listed = (await client.get("/corpus/jobs")).json()
            assert listed["jobs"] == ["swe-v1"]
            assert [(o["job_id"], o["state"]) for o in listed["orders"]] == [("swe-v1", "open")]

    asyncio.run(go())


@pytest.mark.parametrize("fault", ["ledger", "manifest"])
def test_a_job_that_fails_every_pass_is_unknown_again_after_its_attempts(caplog, fault):
    # A fault the next pass repeats (a ledger or manifest that does not parse) is not a restart: its miners stop on
    # the 404, as before, instead of waiting on it forever. Its wiring is still retried.
    from reliquary.corpus.job import JobError
    from reliquary.validator.corpus_hot_jobs import RESTORE_ATTEMPTS
    from reliquary.validator.corpus_service import LedgerSnapshotError

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")])
        if fault == "ledger":
            h.wire_error = LedgerSnapshotError("job 'swe-v1' repeats a pending record")
        else:
            h.read_error = JobError("stored job 'swe-v1' is not JSON")
        async with h.client() as client:
            for _ in range(RESTORE_ATTEMPTS - 1):
                await h.set.refresh()
                assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503
            retried = []

            async def mid_pass():
                retried.append(((await client.get("/corpus/jobs/swe-v1/job")).status_code,
                                (await client.get("/corpus/jobs")).json()["orders"]))

            with caplog.at_level(logging.ERROR, logger="reliquary.validator.corpus_hot_jobs"):
                await h.set.refresh()
                h.mid_pass = mid_pass
                await h.set.refresh()
            # Given up: a pass retrying it does not make it `restoring` again while it runs, or a follower would
            # see it listed in every pass and never miss it twice in a row.
            assert retried == [(404, [])]
            h.mid_pass = None
            unknown = await client.get("/corpus/jobs/swe-v1/job")
            assert unknown.status_code == 404 and unknown.json() == {"detail": "corpus_job_not_served"}
            assert (await client.get("/corpus/jobs")).json()["orders"] == []
            given_up = [r.message for r in caplog.records if "still not wired" in r.message]
            assert len(given_up) == 1 and given_up[0].startswith(
                f"corpus job swe-v1: still not wired after {RESTORE_ATTEMPTS} attempt(s)")
            h.wire_error = h.read_error = None
            await h.set.refresh()
            assert (await client.get("/corpus/jobs/swe-v1/job")).json() == {"job_id": "swe-v1"}

    asyncio.run(go())


def test_a_job_waited_for_since_the_start_is_unknown_again_after_restore_seconds():
    from reliquary.validator.corpus_hot_jobs import RESTORE_ATTEMPTS, RESTORE_SECONDS

    now = [1000.0]

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], wire_error=OSError("r2 timeout"), clock=lambda: now[0])
        async with h.client() as client:
            now[0] += RESTORE_SECONDS - 60          # a long first pass after the start, failed for now
            await h.set.refresh()
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503
            now[0] += 60                            # its next pass: fewer than RESTORE_ATTEMPTS, but too late
            await h.set.refresh()
            assert RESTORE_ATTEMPTS > 2
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 404

    asyncio.run(go())


def test_a_retry_starting_past_restore_seconds_keeps_its_job_restoring_until_it_is_wired():
    # A restart's first pass fails after 14 minutes, and its retry starts 15 minutes after the start and wires the
    # job in 3 more. That retry is the attempt that may wire it: while it runs the job answers 503 and is listed
    # `restoring`. Given up as the retry started, it answered 404 and left the list while it was wired, and a
    # follower reading the list then stopped its miner.
    from reliquary.validator.corpus_hot_jobs import RESTORE_SECONDS

    now = [1000.0]
    restoring = [{"job_id": "swe-v1", "state": "restoring"}]

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], wire_error=OSError("r2 timeout"), clock=lambda: now[0])
        seen = []
        async with h.client() as client:
            def taking(seconds):
                async def mid_pass():
                    seen.append(((await client.get("/corpus/jobs/swe-v1/job")).status_code,
                                 (await client.get("/corpus/jobs")).json()["orders"]))
                    now[0] += seconds
                return mid_pass

            h.mid_pass = taking(RESTORE_SECONDS - 60)
            await h.set.refresh()
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503
            now[0] += 60
            h.wire_error, h.mid_pass = None, taking(180)
            await h.set.refresh()
            assert seen == [(503, restoring), (503, restoring)]
            assert (await client.get("/corpus/jobs/swe-v1/job")).json() == {"job_id": "swe-v1"}
            assert [o["state"] for o in (await client.get("/corpus/jobs")).json()["orders"]] == ["open"]

    asyncio.run(go())


def test_a_job_the_screen_passes_over_is_never_restoring(monkeypatch):
    # The registry lists every control's jobs. One the screen passes over from its entry alone (an order job here,
    # on a corpus validator) is unknown from the first pass on, not listed `restoring` before that pass reaches it.
    from reliquary.validator.corpus_hot_jobs import order_entry_screen

    monkeypatch.delenv("RELIQUARY_ADMIN_TASK_PREFIX", raising=False)

    async def go():
        h = _Harness([_hot_entry(task_id="corpus-a", job_id="swe-v1"),
                      _hot_entry(task_id="corpus-b", job_id="order-eval-x")],
                     wire_error=OSError("r2 timeout"), screen=order_entry_screen)
        seen = []
        async with h.client() as client:
            async def mid_pass():
                seen.append(((await client.get("/corpus/jobs/order-eval-x/job")).status_code,
                             (await client.get("/corpus/jobs")).json()["orders"]))

            h.mid_pass = mid_pass
            await h.set.refresh()
            assert seen == [(404, [{"job_id": "swe-v1", "state": "restoring"}])]
            assert (await client.get("/corpus/jobs/order-eval-x/job")).status_code == 404
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503

    asyncio.run(go())


@pytest.mark.parametrize("bound", ["attempts", "seconds"])
def test_an_unreadable_registry_ends_booting_after_its_bound(caplog, bound):
    from reliquary.validator.corpus_hot_jobs import RESTORE_ATTEMPTS, RESTORE_SECONDS

    now = [1000.0]

    async def go():
        h = _Harness([_hot_entry(job_id="swe-v1")], clock=lambda: now[0])
        h.registry_error = OSError("r2 down")
        reads = RESTORE_ATTEMPTS if bound == "attempts" else 1
        async with h.client() as client:
            for _ in range(reads - 1):
                await h.set.refresh()
                assert (await client.get("/corpus/jobs")).status_code == 503
                assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503
            if bound == "seconds":
                now[0] += RESTORE_SECONDS
            with caplog.at_level(logging.ERROR, logger="reliquary.validator.corpus_hot_jobs"):
                await h.set.refresh()
            listed = await client.get("/corpus/jobs")
            assert listed.status_code == 200 and listed.json()["orders"] == []
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 404
            assert any("the task registry is still unread" in r.message for r in caplog.records)
            # Read at last: the job it lists is waited for again until it is wired.
            h.registry_error, h.wire_error = None, OSError("r2 timeout")
            await h.set.refresh()
            assert (await client.get("/corpus/jobs/swe-v1/job")).status_code == 503

    asyncio.run(go())
