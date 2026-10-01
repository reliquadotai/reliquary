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
        params={"cap": cap},
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
        )
    finally:
        profiles.toploc_proof = real


def test_an_entry_on_this_model_with_this_environment_is_admitted():
    assert _refusal(_hot_entry()) is None


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
    def __init__(self, entries, *, admit=None, wire_error=None):
        self.entries = entries
        self.routes = CorpusJobRoutes()
        self.wired, self.running, self.cancelled = [], [], []
        self.drained = {}
        self.caps = {}
        self.wire_error = wire_error

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
            return {e.task_id: e for e in self.entries}

        async def read_job(job_id):
            return _hot_job(job_id)

        async def drained(wiring):
            return self.drained.get(wiring.entry.job_id, False)

        self.set = CorpusJobSet(
            routes=self.routes, router_for=lambda w: _Router(w.entry.job_id), wire=wire,
            jobs_of=lambda w: [forever(f"audit:{w.entry.job_id}"), forever(f"settle:{w.entry.job_id}")],
            read_entries=read_entries, read_job=read_job,
            admit=admit or (lambda entry, job: None), drained=drained,
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
            assert (await client.get("/corpus/jobs")).json() == {"jobs": ["job-b"]}
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
            assert (await client.get("/corpus/jobs")).json() == {"jobs": []}
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
    from reliquary.validator import corpus_auditor, corpus_settlement
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
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", settle_once)

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
    assert seen["before"] == {"jobs": ["math-v1"]}
    assert seen["after_add"] == {"jobs": ["code-v1", "math-v1"]}
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
