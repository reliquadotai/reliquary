"""R2: `GET /corpus/jobs/{job_id}/status`, public, from in-memory state, cached."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.corpus.slots import SlotLedger
from reliquary.validator.corpus_hot_jobs import CorpusJobSet
from reliquary.validator.corpus_job_status import JobStats
from reliquary.validator.corpus_service import CorpusJobRoutes

HOUR = 3600.0
TOTALS = {"verdicts": 10, "passed": 8, "verified_tokens": 80, "complete": True}


def _verdict(hotkey="5Hot", passed=True, audited=True, tokens=10):
    return {"hotkey": hotkey, "passed": passed, "audited": audited, "token_count": tokens}


# --------------------------------------------------------------------------
# JobStats
# --------------------------------------------------------------------------


def test_stats_hold_only_unsettled_verdicts():
    stats = JobStats()
    stats.observe("a", _verdict(tokens=10))
    stats.observe("a", _verdict(tokens=10))
    stats.observe("b", _verdict(passed=False, tokens=7))
    stats.observe("c", _verdict(audited=False, tokens=5))
    stats.observe("d", None)
    assert stats.unsettled() == (3, 2, 15)
    stats.settled(["a", "b"])
    assert stats.unsettled() == (1, 1, 5)


def test_accepted_last_hour_forgets_older_acceptances():
    now = [1000.0]
    stats = JobStats(clock=lambda: now[0])
    stats.accepted()
    now[0] += HOUR - 1
    stats.accepted()
    assert stats.accepted_last_hour() == 2
    now[0] += 2
    assert stats.accepted_last_hour() == 1


def test_the_settler_persists_totals_once_per_settled_verdict():
    from tests.unit.test_corpus_settlement import _Archives, _Records, _settler, _v

    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30, ok=False)})
    settled_ids = []
    settler = _settler(records, _Archives(46000))
    settler.on_settled = settled_ids.extend
    asyncio.run(settler.settle_once())
    assert records.state["totals"] == {"verdicts": 2, "passed": 1, "verified_tokens": 10,
                                       "complete": True}
    assert settler.totals == records.state["totals"] and sorted(settled_ids) == sorted(records.verdicts)
    records.verdicts["3" * 64] = _v("A", 5)
    records.fail_state_writes_after = 1  # the pending write lands, the finish crashes
    archives = _Archives(46001)
    crashed = _settler(records, archives)
    with pytest.raises(OSError):
        asyncio.run(crashed.settle_once())
    records.fail_state_writes_after = None
    asyncio.run(_settler(records, archives).settle_once())
    assert records.state["totals"]["verdicts"] == 3 and records.state["totals"]["passed"] == 2


def test_a_job_settled_before_totals_existed_says_its_counts_are_incomplete():
    from tests.unit.test_corpus_settlement import _Archives, _Records, _settler

    records = _Records({})
    records.state = {"settled": ["9" * 64], "last_window": 3}
    settler = _settler(records, _Archives(None))
    asyncio.run(settler.settle_once())
    assert settler.totals["complete"] is False


# --------------------------------------------------------------------------
# The status of a served job
# --------------------------------------------------------------------------


def _job(prompt_count=3, slots=2):
    return SimpleNamespace(job_id="job-b", prompt_count=prompt_count, prompt_start=0,
                           slots_per_prompt=slots)


class _Router:
    def __init__(self, job, consumed):
        self.job, self.consumed, self.reads = job, consumed, 0

    async def ledger_state(self, job=None):
        assert job is self.job  # the wired manifest: no manifest read
        self.reads += 1
        slots = SlotLedger(prompt_start=0, prompt_count=self.job.prompt_count,
                           slots_per_prompt=self.job.slots_per_prompt)
        for index, count in self.consumed.items():
            for _ in range(count):
                slots.consume(index)
        return self.job, SimpleNamespace(slots=slots)


def _set(consumed, *, now):
    job = _job()
    routes = CorpusJobRoutes()
    router = _Router(job, consumed)
    routes.add("job-b", router)
    stats = JobStats(clock=lambda: now[0])
    wiring = SimpleNamespace(entry=SimpleNamespace(task_id="corpus-b", job_id="job-b"),
                             job=job, stats=stats,
                             settler=SimpleNamespace(settled_count=4, totals=TOTALS))
    job_set = CorpusJobSet(routes=routes, router_for=None, wire=None, jobs_of=lambda w: [],
                           clock=lambda: now[0])
    job_set.served["job-b"] = wiring
    return job_set, router, stats


def test_a_served_job_reports_every_field_and_no_hotkey():
    now = [10_000.0]
    job_set, _, stats = _set({0: 2, 1: 1}, now=now)
    for sid, verdict in (("a", _verdict(tokens=10)), ("b", _verdict(passed=False, tokens=3))):
        stats.observe(sid, verdict)
    stats.accepted()
    status = asyncio.run(job_set.status("job-b"))
    assert status == {
        "job_id": "job-b", "state": "open", "prompts_total": 3, "prompts_full": 1,
        "submissions_accepted": 3, "audited": 12, "passed": 9, "verified_tokens": 90,
        "settled": 4, "accepted_last_hour": 1, "counts_complete": True,
    }
    assert "5Hot" not in json.dumps(status)


def test_a_job_with_every_prompt_full_is_full_then_retired_then_drained():
    now = [10_000.0]
    job_set, _, _ = _set({0: 2, 1: 2, 2: 2}, now=now)
    assert asyncio.run(job_set.status("job-b"))["state"] == "full"
    job_set._routes.retire("job-b")
    now[0] += 31
    assert asyncio.run(job_set.status("job-b"))["state"] == "retired"

    async def drained(wiring):
        return True

    job_set._drained = drained
    asyncio.run(job_set._maybe_unwire("job-b"))
    final = asyncio.run(job_set.status("job-b"))
    assert final["state"] == "drained" and final["submissions_accepted"] == 6


def test_the_status_is_cached_and_the_ledger_read_at_most_once_per_period():
    now = [10_000.0]
    job_set, router, stats = _set({0: 1}, now=now)
    first = asyncio.run(job_set.status("job-b"))
    stats.observe("a", _verdict())
    assert asyncio.run(job_set.status("job-b")) == first
    assert router.reads == 1
    now[0] += 30.5
    assert asyncio.run(job_set.status("job-b"))["passed"] == 9
    assert router.reads == 2


def test_concurrent_requests_on_an_expired_cache_recompute_once():
    now = [10_000.0]
    job_set, router, _ = _set({0: 1}, now=now)

    async def go():
        return await asyncio.gather(*(job_set.status("job-b") for _ in range(5)))

    assert len({id(s) for s in asyncio.run(go())}) == 1
    assert router.reads == 1


def test_an_unknown_job_has_no_status():
    now = [0.0]
    job_set, _, _ = _set({}, now=now)
    assert asyncio.run(job_set.status("nope")) is None


def test_a_ledger_read_that_fails_serves_the_last_status_or_raises():
    now = [10_000.0]
    job_set, router, _ = _set({0: 1}, now=now)
    first = asyncio.run(job_set.status("job-b"))

    async def broken(job=None):
        raise OSError("r2 down")

    router.ledger_state = broken
    now[0] += 31
    assert asyncio.run(job_set.status("job-b")) == first
    job_set._status_cache.clear()
    with pytest.raises(OSError):
        asyncio.run(job_set.status("job-b"))


# --------------------------------------------------------------------------
# The route, the auditor hook and the settler count
# --------------------------------------------------------------------------


def test_the_route_serves_the_status_and_404s_an_unknown_job(seeded_job):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("swe-v1",))
    app = client.app
    now = [10_000.0]
    job_set = CorpusJobSet(routes=app.state.corpus_routes, router_for=None, wire=None,
                           jobs_of=lambda w: [], clock=lambda: now[0])
    job_set.served["swe-v1"] = SimpleNamespace(
        entry=SimpleNamespace(task_id="t", job_id="swe-v1"), job=seeded_job.job,
        stats=JobStats(), settler=SimpleNamespace(settled_count=0, totals=None))
    app.state.corpus_jobs = job_set
    status = client.get("/corpus/jobs/swe-v1/status")
    assert status.status_code == 200, status.text
    assert status.json()["prompts_total"] == seeded_job.job.prompt_count
    assert status.json()["submissions_accepted"] == 0
    assert status.json()["counts_complete"] is False
    assert client.get("/corpus/jobs/nope/status").status_code == 404


def test_the_auditor_reports_every_verdict_it_writes_or_finds():
    from tests.unit.test_corpus_auditor import _Records
    from reliquary.validator.corpus_auditor import CorpusAuditor

    seen = []
    records = _Records({})
    records.verdicts["b"] = _verdict(tokens=3)
    auditor = CorpusAuditor(job_id="j", records=records, model=None, tokenizer=None,
                            proof=None, on_verdict=lambda sid, v: seen.append((sid, v)))
    asyncio.run(auditor._write("a", _verdict(tokens=5)))
    asyncio.run(auditor._write("b", _verdict(tokens=9)))
    assert seen == [("a", _verdict(tokens=5)), ("b", _verdict(tokens=3))]


def test_the_settler_keeps_the_count_it_settled():
    from tests.unit.test_corpus_settlement import _Archives, _Records, _settler, _v

    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30)})
    settler = _settler(records, _Archives(46000))
    assert settler.settled_count is None
    asyncio.run(settler.settle_once())
    assert settler.settled_count == 2


from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: E402,F401


def test_with_the_flags_off_boot_reads_no_verdict(
    seeded_job, fake_r2, wired_records, fixed_drand_chain, monkeypatch,  # noqa: F811
):
    """I2: the status route costs prod nothing at boot."""
    import huggingface_hub
    import uvicorn

    import reliquary.corpus.encoding as encoding
    import reliquary.protocol.profiles as profiles
    import reliquary.shared.modeling as modeling
    from reliquary.infrastructure import corpus_record_store
    from reliquary.validator import corpus_auditor, corpus_settlement
    from reliquary.validator.corpus_validator import run_corpus_validator
    from tests.unit.test_corpus_multi_job_validator import _Model, _entry
    from tests.unit.test_corpus_service import CHECKPOINT, _Tokenizer
    from tests.unit.test_corpus_validator import _profile

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None: "/x")
    monkeypatch.setattr(encoding, "checkpoint_fingerprint", lambda d: CHECKPOINT)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: _Tokenizer())
    monkeypatch.setattr(modeling, "load_text_only_model", lambda path, **kw: _Model())
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", _profile())
    reads = []
    for name in ("list_verdict_ids", "read_verdict", "list_submission_ids"):
        real = getattr(corpus_record_store, name)

        async def counted(*a, _real=real, _name=name, **kw):
            reads.append(_name)
            return await _real(*a, **kw)

        monkeypatch.setattr(corpus_record_store, name, counted)

    async def idle(self):
        await asyncio.sleep(3600)

    async def settle_once(self):
        return None

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_settlement.CorpusSettler, "settle_once", settle_once)

    class _Stop(Exception):
        pass

    class _Server:
        def __init__(self, config):
            pass

        async def serve(self):
            await asyncio.sleep(0.05)
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)
    with pytest.raises(_Stop):
        asyncio.run(run_corpus_validator(
            jobs=[(_entry("corpus-math", "swe-v1"), 0.1)], wallet=None, netuid=0,
            signer_client=None, http_host="127.0.0.1", http_port=0, set_weights=False,
            registration_gate=False))
    assert reads == []


from tests.unit.test_corpus_validator import fixed_drand_chain, wired_records  # noqa: E402,F401
