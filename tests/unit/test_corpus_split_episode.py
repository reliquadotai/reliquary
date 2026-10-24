"""An episode job on the split validator: the front serves it (intake, audit
through the GPU process, grade dispatcher and router, grader, payment gate)
when no judge group names it; a judge group naming it is refused at startup,
since judge processes host no grader."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.job import parse_job
from reliquary.environment.agentic_swe import SweSource
from reliquary.validator.corpus_split import plan_groups
from reliquary.validator.corpus_validator import split_episode_refusal
from tests.unit import corpus_split_fakes as fakes
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_trajectory_parse import R

EPISODE_ID, SINGLE_ID = "swe-agentic-v1", "math-v1"
EPISODE_TASK, SINGLE_TASK = "corpus-swe", "corpus-math"


def _episode_job(job_id=EPISODE_ID, version=None):
    raw = _manifest(prompt_count=3, job_id=job_id)
    if version is not None:
        raw["episode"]["env"]["version"] = version
    return parse_job(raw)


def _single_job():
    return parse_job(_manifest(with_episode=False, job_id=SINGLE_ID,
                               prompt_source="openmathinstruct", renderer_id="x-v1"))


def _entry(task_id, job_id):
    return SimpleNamespace(task_id=task_id, job_id=job_id,
                           params={"cap": 0.1, "settlement": "period-ema-v1"},
                           mechanism="corpus-generation", status="active", retired_at=None,
                           contract=None)


# -- the plan: which process judges an episode job ------------------------------

JOBS = [(SINGLE_TASK, SINGLE_ID), (EPISODE_TASK, EPISODE_ID)]


def test_star_leaves_an_episode_job_in_the_front():
    assert plan_groups(None, JOBS, front_only={EPISODE_ID}) == [[SINGLE_ID]]
    assert plan_groups("*", JOBS, front_only={EPISODE_ID}) == [[SINGLE_ID]]
    assert plan_groups(SINGLE_TASK, JOBS, front_only={EPISODE_ID}) == [[SINGLE_ID]]


@pytest.mark.parametrize("value", [EPISODE_ID, EPISODE_TASK, f"{SINGLE_ID},{EPISODE_ID}",
                                   f"{SINGLE_ID};{EPISODE_TASK}"])
def test_a_judge_group_naming_an_episode_job_is_refused(value):
    with pytest.raises(ValueError, match="episode job .* judge processes host no grader"):
        plan_groups(value, JOBS, front_only={EPISODE_ID})


def test_the_plan_of_single_turn_jobs_is_unchanged():
    assert plan_groups(None, JOBS) == [[SINGLE_ID], [EPISODE_ID]]
    assert plan_groups(None, JOBS, front_only=()) == [[SINGLE_ID], [EPISODE_ID]]


def test_a_judge_process_refuses_an_episode_job(monkeypatch):
    from reliquary.infrastructure import corpus_job_store
    from reliquary.validator.corpus_judge_process import run_corpus_judges

    class _Store:
        async def read_job(self, job_id):
            return _episode_job(), None

    monkeypatch.setattr(corpus_job_store, "BucketJobStore", _Store)
    with pytest.raises(RuntimeError, match="judge processes host no grader"):
        asyncio.run(run_corpus_judges(served=[(_entry(EPISODE_TASK, EPISODE_ID), 0.1)],
                                      directory="/nonexistent", run_dir="/nonexistent",
                                      proof=fakes.PROOF, socket_path="/nonexistent/s"))


# -- the front: started for real, stubbed at storage, GPU and HTTP --------------


class _Stop(Exception):
    pass


class _Link:
    def __init__(self):
        self.accepted_ids = []
        self.pending_record_arrivals = {}

    def accepted(self, job_id, submission_id):
        self.accepted_ids.append((job_id, submission_id))

    async def run(self):
        await asyncio.sleep(3600)


class _Records:
    """Nothing is read or written before the server would serve."""


@pytest.fixture
def front(monkeypatch, tmp_path):
    import uvicorn

    import reliquary.shared.modeling as modeling
    from reliquary.infrastructure import corpus_executor_store, corpus_job_store
    from reliquary.infrastructure import corpus_record_store
    from reliquary.validator import (
        agentic_intake, corpus_auditor, corpus_gpu, corpus_grade_remote, corpus_grading,
        corpus_judge_threads, corpus_service, corpus_validator,
    )

    jobs = {EPISODE_ID: _episode_job(), SINGLE_ID: _single_job(),
            "swe-agentic-v2": _episode_job("swe-agentic-v2"),
            "swe-agentic-other": _episode_job("swe-agentic-other", version="e" * 40)}
    # ``fail``: step name -> exceptions to raise there, one per call, in order.
    calls = SimpleNamespace(migrated=[], recovered=[], intakes=[], leases=[], fail={})

    def maybe_fail(step):
        pending = calls.fail.get(step)
        if pending:
            raise pending.pop(0)

    class _Store:
        async def read_job(self, job_id):
            return jobs.get(job_id), None

    async def migrate(store, job):
        if job.episode is not None:
            maybe_fail("migrate")
        calls.migrated.append(str(job.job_id))
        return None

    async def recover(store, records, job):
        if job.episode is not None:
            maybe_fail("recover")
        assert str(job.job_id) in calls.migrated
        calls.recovered.append(str(job.job_id))
        return []

    def intake(job, **kw):
        maybe_fail("intake")
        assert str(job.job_id) in calls.recovered
        calls.intakes.append((str(job.job_id), kw))
        return SimpleNamespace(renderer=R, source=SweSource([("i0", "p"), ("i1", "q")]))

    async def no_executors(**kw):
        maybe_fail("registry")
        return []

    async def info(run_dir, **kw):
        return {"vocab_size": fakes.VOCAB}

    async def idle(self, *a, **kw):
        await asyncio.sleep(3600)

    async def idle_settle(*a, **kw):
        await asyncio.sleep(3600)

    monkeypatch.setattr(corpus_job_store, "BucketJobStore", _Store)
    monkeypatch.setattr(corpus_record_store, "BucketRecordStore", _Records)
    monkeypatch.setattr(corpus_judge_threads, "judge_record_store", lambda *a, **kw: _Records())
    monkeypatch.setattr(corpus_service, "migrate_ledgers_at_startup", migrate)
    monkeypatch.setattr(corpus_service, "recover_pending_records", recover)
    monkeypatch.setattr(corpus_service, "renderer_for_job", lambda job, encode, **kw: object())
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: fakes.Tokenizer())
    monkeypatch.setattr(corpus_gpu, "read_info", info)
    monkeypatch.setattr(agentic_intake, "build_episode_intake", intake)
    grade_renderers = []

    def grade_renderer(job, **kw):
        grade_renderers.append(SimpleNamespace(job=str(job.job_id)))
        return grade_renderers[-1]

    monkeypatch.setattr(agentic_intake, "build_grade_renderer", grade_renderer)
    def lease(job, **kw):
        maybe_fail("lease")
        calls.leases.append(str(job.job_id))

    monkeypatch.setattr(corpus_grade_remote, "check_replay_lease", lease)
    # Hot adds go through the real job set; the contract check is not under test here.
    from reliquary.validator import corpus_hot_jobs

    monkeypatch.setattr(corpus_hot_jobs, "hot_job_refusal", lambda *a, **kw: None)
    monkeypatch.setattr(corpus_executor_store, "list_executors", no_executors)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_grading.CorpusGrader, "run", idle)
    monkeypatch.setattr(corpus_validator, "settle_forever", idle_settle)
    # A hot add reads its entry's contract; the renderer it checks is a stub here.
    monkeypatch.setattr(corpus_validator, "_entry_profile", lambda entry: None)
    built = {}

    async def no_entries():
        return {}

    class _Server:
        def __init__(self, config):
            built["app"] = config.app
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)

    def start(served_ids, linked_ids=(), hot=False):
        from reliquary.validator.corpus_split import FrontSplit

        tasks = {EPISODE_ID: EPISODE_TASK, SINGLE_ID: SINGLE_TASK,
                 "swe-agentic-v2": "corpus-swe-2", "swe-agentic-other": "corpus-swe-3"}
        link = _Link()
        split = FrontSplit(directory=str(tmp_path), fingerprint="c" * 64, proof=fakes.PROOF,
                           run_dir=str(tmp_path), links={j: link for j in linked_ids})
        with pytest.raises(_Stop):
            asyncio.run(corpus_validator.run_corpus_validator(
                jobs=[(_entry(tasks[j], j), 0.1) for j in served_ids], wallet=None, netuid=81,
                signer_client=None, http_host="127.0.0.1", http_port=0, set_weights=False,
                registration_gate=False, split=split,
                read_registry=no_entries if hot else None))
        app = built["app"]
        return SimpleNamespace(app=app, served=app.state.corpus_jobs.served, link=link,
                               job_set=app.state.corpus_jobs, split=split)

    return SimpleNamespace(start=start, calls=calls, jobs=jobs)


def _post(app, path, body):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    if body is None:
        return client.post(path).status_code
    return client.post(path, json=body, headers={"Authorization": "Bearer nope"}).status_code


def test_the_front_serves_an_episode_job_no_group_names(front):
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from reliquary.validator.corpus_gpu import GpuScorer
    from reliquary.validator.corpus_grading import CorpusGrader

    started = front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID])
    w = started.served[EPISODE_ID]
    # Intake: the episode intake on the job's route, built with the GPU process's vocabulary.
    assert w.episode_intake is not None
    (job_id, kw), = front.calls.intakes
    assert job_id == EPISODE_ID and kw["vocab_size"] == fakes.VOCAB
    # Audit v2 in the front, scored by the GPU process (spans cross its wire).
    assert getattr(w, "judge_link", None) is None
    assert isinstance(w.auditor, CorpusAuditor) and isinstance(w.auditor._scorer, GpuScorer)
    assert w.auditor._model is None and w.auditor._vocab_size == fakes.VOCAB
    # The grader leases to the one grade dispatcher, and gates payment.
    assert isinstance(w.grader, CorpusGrader)
    assert w.grader._dispatcher is started.app.state.corpus_grade_remote
    assert w.settler._ready == w.grader.ready
    # I2: its own renderer and threads, never the intake's lock or the default executor.
    assert w.grader._renderer is w.grade_renderer and w.grade_renderer is not w.episode_intake.renderer
    assert w.grader._parse_executor is not None and w.grader._beacon_executor is not None
    assert front.calls.leases == [EPISODE_ID]
    # The grade executors reach the front: its routes are mounted.
    claim = {"executor_id": "g0", "env_package": "reliquary-swe", "env_version": "0" * 40}
    assert _post(started.app, "/corpus/internal/grade/claim", claim) == 401     # mounted, gated
    # The runbook's reachability probe: a bare POST is 422 (routed), never 404.
    assert _post(started.app, "/corpus/internal/grade/claim", None) == 422
    # The single-turn job still goes to its judge process, as before.
    single = started.served[SINGLE_ID]
    assert single.judge_link is started.link and getattr(single, "grader", None) is None
    assert single.pending_record_arrivals is started.link.pending_record_arrivals[SINGLE_ID]
    assert set(front.calls.recovered) == {SINGLE_ID, EPISODE_ID}


def test_single_turn_jobs_alone_mount_no_grade_route(front):
    started = front.start([SINGLE_ID], linked_ids=[SINGLE_ID])
    assert not hasattr(started.app.state, "corpus_grade_remote")
    claim = {"executor_id": "g0", "env_package": "reliquary-swe", "env_version": "0" * 40}
    assert _post(started.app, "/corpus/internal/grade/claim", claim) == 404
    assert _post(started.app, "/corpus/internal/grade/claim", None) == 404
    assert front.calls.intakes == [] and front.calls.leases == []


def test_an_episode_job_linked_to_a_judge_is_refused_before_anything(front):
    from reliquary.validator import corpus_validator
    from reliquary.validator.corpus_validator import SPLIT_EPISODE_REFUSAL

    with pytest.raises(RuntimeError) as refused:
        asyncio.run(corpus_validator.run_corpus_validator(
            jobs=[(_entry(EPISODE_TASK, EPISODE_ID), 0.1)], wallet=None, netuid=81,
            signer_client=None, http_host="127.0.0.1", http_port=0, set_weights=False,
            registration_gate=False,
            split=SimpleNamespace(links={EPISODE_ID: _Link()}, directory="/x",
                                  fingerprint="c" * 64, proof=fakes.PROOF, run_dir="/x")))
    assert str(refused.value) == SPLIT_EPISODE_REFUSAL.format(job_id=EPISODE_ID)
    assert "judge processes host no grader" in str(refused.value)
    assert front.calls.migrated == []                     # no ledger touched


def test_split_episode_refusal_names_only_a_linked_episode_job():
    from reliquary.validator.corpus_hot_jobs import REFUSED

    episode, single = _episode_job(), _single_job()
    split = SimpleNamespace(links={SINGLE_ID: object()})
    assert split_episode_refusal(None, episode) is None
    assert split_episode_refusal(split, episode) is None
    assert split_episode_refusal(split, single) is None
    kind, why = split_episode_refusal(SimpleNamespace(links={EPISODE_ID: object()}), episode)
    assert kind == REFUSED and EPISODE_ID in why


# -- hot adds: the single process's rules ---------------------------------------


def test_a_hot_episode_job_joins_a_front_started_with_one_on_its_pin(front):
    started = front.start([EPISODE_ID])
    entry = _entry("corpus-swe-2", "swe-agentic-v2")
    # The split adds no refusal of its own for a job no group names.
    assert split_episode_refusal(started.split, front.jobs["swe-agentic-v2"]) is None
    w = asyncio.run(started.job_set._wire(entry, 0.1, front.jobs["swe-agentic-v2"]))
    assert w.grader is not None and w.auditor._scorer is not None
    assert w.grader._dispatcher is started.app.state.corpus_grade_remote


def test_a_hot_episode_job_on_another_pin_is_refused_not_crashing(front):
    started = front.start([EPISODE_ID])
    entry = _entry("corpus-swe-3", "swe-agentic-other")
    with pytest.raises(ValueError, match="restart it to grade the job"):
        asyncio.run(started.job_set._wire(entry, 0.1, front.jobs["swe-agentic-other"]))


def test_a_hot_episode_job_on_a_front_without_one_is_refused_not_crashing(front):
    started = front.start([SINGLE_ID], linked_ids=[SINGLE_ID])
    entry = _entry(EPISODE_TASK, EPISODE_ID)
    assert split_episode_refusal(started.split, front.jobs[EPISODE_ID]) is None
    with pytest.raises(ValueError, match="started without one"):
        asyncio.run(started.job_set._wire(entry, 0.1, front.jobs[EPISODE_ID]))


# -- the supervisor: the plan refuses before any child starts --------------------


@pytest.mark.parametrize("judges,groups", [(EPISODE_TASK, None), ("*", [[SINGLE_ID]]),
                                           (None, [[SINGLE_ID]])])
def test_the_supervisor_plans_episode_jobs_into_the_front(monkeypatch, judges, groups):
    from reliquary.validator import corpus_split

    async def preflight(served):
        return SimpleNamespace(directory="/x", fingerprint="c" * 64, proof=fakes.PROOF,
                               model_id="m", model_revision="r", jobs=JOBS,
                               episode_jobs=[EPISODE_ID])

    started = []

    class _Supervisor:
        def __init__(self, spec):
            started.append(spec)

        async def run(self):
            return None

    monkeypatch.setattr(corpus_split, "preflight", preflight)
    monkeypatch.setattr(corpus_split, "Supervisor", _Supervisor)
    if judges is None:
        monkeypatch.delenv(corpus_split.JUDGES_ENV, raising=False)
    else:
        monkeypatch.setenv(corpus_split.JUDGES_ENV, judges)
    run = corpus_split.run_corpus_split(served=[], netuid=81, http_host="h", http_port=1,
                                        set_weights=False)
    if groups is None:
        with pytest.raises(ValueError, match="judge processes host no grader"):
            asyncio.run(run)
        assert started == []                                # no child started
    else:
        asyncio.run(run)
        assert started[0].groups == groups and started[0].group_of(EPISODE_ID) is None


# -- I1: an episode job that cannot be wired at startup is left out, alone --------


def _listed_jobs(app):
    from fastapi.testclient import TestClient

    return TestClient(app).get("/corpus/jobs").json()["jobs"]


@pytest.mark.parametrize("step,exc", [
    ("intake", RuntimeError("reliquary-swe is not installed at the pin")),
    ("intake", OSError("HF Hub unreachable")),
    ("lease", RuntimeError("the replay lease is shorter than the task's replay work")),
    ("registry", OSError("R2 unreachable")),
    ("migrate", OSError("R2 unreachable")),
    ("recover", OSError("accepted body is not readable")),
])
def test_an_episode_job_that_fails_to_wire_leaves_the_others_served(front, step, exc,
                                                                    monkeypatch):
    from reliquary.validator import corpus_validator

    errors = []
    monkeypatch.setattr(corpus_validator.logger, "error",
                        lambda msg, *a, **kw: errors.append(msg % a))
    front.calls.fail[step] = [exc]
    started = front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID])
    # The front is up and the single-turn job served, as before.
    assert set(started.served) == {SINGLE_ID}
    assert started.served[SINGLE_ID].judge_link is started.link
    # The episode job is not served (no route), and says so loudly.
    assert _listed_jobs(started.app) == [SINGLE_ID]
    assert EPISODE_TASK in started.app.state.corpus_unserved
    assert any("EPISODE JOB swe-agentic-v1 IS NOT SERVED" in m and str(exc) in m
               for m in errors), errors


def test_a_front_whose_only_job_fails_to_wire_refuses_to_start(front):
    from reliquary.validator import corpus_validator
    from reliquary.validator.corpus_split import FrontSplit

    front.calls.fail["intake"] = [RuntimeError("reliquary-swe is not installed at the pin")]
    with pytest.raises(RuntimeError, match="no corpus job left to serve"):
        asyncio.run(corpus_validator.run_corpus_validator(
            jobs=[(_entry(EPISODE_TASK, EPISODE_ID), 0.1)], wallet=None, netuid=81,
            signer_client=None, http_host="127.0.0.1", http_port=0, set_weights=False,
            registration_gate=False,
            split=FrontSplit(directory="/x", fingerprint="c" * 64, proof=fakes.PROOF,
                             run_dir="/x", links={})))


def test_an_episode_job_left_out_at_start_is_wired_by_a_later_refresh(front):
    """A transient failure (the HF Hub down at the front's start): the hot job
    set retries the job at its refresh, through the real job set."""
    front.calls.fail["intake"] = [OSError("HF Hub unreachable")]
    started = front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID])
    assert EPISODE_ID not in started.served
    entry = _entry(EPISODE_TASK, EPISODE_ID)
    asyncio.run(started.job_set._consider(entry))
    w = started.job_set.served[EPISODE_ID]
    assert w.grader._dispatcher is started.app.state.corpus_grade_remote
    # Served now: no longer listed as not served.
    assert EPISODE_TASK not in started.app.state.corpus_unserved


def test_a_job_the_lease_check_refuses_at_start_is_never_wired_by_a_refresh(front):
    """P26: a refresh retries the job, but the lease check is run again first
    and its refusal is permanent."""
    refusal = RuntimeError("the replay lease is shorter than the task's replay work")
    front.calls.fail["lease"] = [refusal, RuntimeError(str(refusal))]
    started = front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID], hot=True)
    assert EPISODE_ID not in started.served
    asyncio.run(started.job_set._consider(_entry(EPISODE_TASK, EPISODE_ID)))
    assert EPISODE_ID not in started.job_set.served
    assert EPISODE_TASK in started.job_set._passed_over          # refused for good
    assert front.calls.intakes == []                             # before any intake


def test_a_second_hot_job_on_the_pin_is_lease_checked_too(front):
    started = front.start([EPISODE_ID], hot=True)
    front.calls.fail["lease"] = [RuntimeError("the replay lease is shorter ...")]
    front.calls.intakes.clear()
    with pytest.raises(ValueError, match="replay lease"):
        asyncio.run(started.job_set._wire(_entry("corpus-swe-2", "swe-agentic-v2"), 0.1,
                                          front.jobs["swe-agentic-v2"]))
    assert front.calls.intakes == []
    assert front.calls.leases == [EPISODE_ID]                    # the refused one not listed


def test_a_transient_lease_check_failure_stays_retried(front):
    started = front.start([EPISODE_ID], hot=True)
    front.calls.fail["lease"] = [OSError("HF Hub unreachable")]
    with pytest.raises(OSError):
        asyncio.run(started.job_set._wire(_entry("corpus-swe-2", "swe-agentic-v2"), 0.1,
                                          front.jobs["swe-agentic-v2"]))


def test_a_pin_this_binary_does_not_have_is_refused_for_good(monkeypatch):
    from reliquary.environment import agentic_swe
    from reliquary.validator import agentic_intake

    monkeypatch.setattr(agentic_swe, "episode_support_refusal",
                        lambda *a, **kw: "reliquary-swe is installed at another commit")
    with pytest.raises(ValueError, match="another commit"):
        agentic_intake.build_episode_intake(_episode_job(), checkpoint_dir="/x",
                                            tokenizer=fakes.Tokenizer(),
                                            vocab_size=fakes.VOCAB, chunk_tokens=32)


@pytest.mark.parametrize("step,retried", [("intake", True), ("registry", False),
                                          ("lease", False)])
def test_the_not_served_log_promises_a_retry_only_when_there_is_one(front, monkeypatch,
                                                                    step, retried):
    from reliquary.validator import corpus_validator

    errors = []
    monkeypatch.setattr(corpus_validator.logger, "error",
                        lambda msg, *a, **kw: errors.append(msg % a))
    front.calls.fail[step] = [RuntimeError("boom")]
    front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID], hot=True)
    (message,) = [m for m in errors if "IS NOT SERVED" in m]
    assert ("retries it at each refresh" in message) is retried, message
    assert ("restart" in message), message


def test_a_hot_episode_job_on_another_pin_is_passed_over_by_the_job_set(front):
    started = front.start([EPISODE_ID])
    entry = _entry("corpus-swe-3", "swe-agentic-other")
    asyncio.run(started.job_set._consider(entry))       # never raises
    assert "swe-agentic-other" not in started.job_set.served
    assert "corpus-swe-3" in started.job_set._passed_over


# -- review minors -----------------------------------------------------------------


def test_the_production_judges_plan_is_unchanged_with_an_episode_job():
    """corpus-01's string (2026-10-03, code-v2 added): judge-0 math, judge-1 the
    rest; the SWE episode job, named nowhere, stays in the front."""
    production = ("math-omi-qwen38-27b-v1;code-qwen38-27b-v1,if-qwen38-27b-v1,"
                  "logic-qwen38-27b-v1,code-qwen38-27b-v2")
    jobs = [("corpus-math-omi-v1", "math-omi-qwen38-27b-v1"),
            ("corpus-code-v1", "code-qwen38-27b-v1"), ("corpus-if-v1", "if-qwen38-27b-v1"),
            ("corpus-logic-v1", "logic-qwen38-27b-v1"), ("corpus-code-v2", "code-qwen38-27b-v2")]
    expected = [["math-omi-qwen38-27b-v1"],
                ["code-qwen38-27b-v1", "if-qwen38-27b-v1", "logic-qwen38-27b-v1",
                 "code-qwen38-27b-v2"]]
    assert plan_groups(production, jobs) == expected
    assert plan_groups(production, jobs + [(EPISODE_TASK, EPISODE_ID)],
                       front_only={EPISODE_ID}) == expected
    with pytest.raises(ValueError, match="judge processes host no grader"):
        plan_groups(production + f",{EPISODE_ID}", jobs + [(EPISODE_TASK, EPISODE_ID)],
                    front_only={EPISODE_ID})


def _drand_threads(monkeypatch):
    from reliquary.infrastructure import drand

    seen = []

    def chain():
        import threading

        seen.append(threading.current_thread().name)
        return {"genesis_time": 1_692_803_367, "period": 3}

    monkeypatch.setattr(drand, "get_current_chain", chain)
    return seen


def test_the_front_resolves_the_drand_chain_off_the_loop_for_an_episode_job(front, monkeypatch):
    seen = _drand_threads(monkeypatch)
    front.start([SINGLE_ID, EPISODE_ID], linked_ids=[SINGLE_ID])
    assert seen and all(name.startswith("corpus-judge-drand") for name in seen), seen


def test_a_front_without_an_episode_job_never_reads_the_drand_chain_at_start(front, monkeypatch):
    seen = _drand_threads(monkeypatch)
    front.start([SINGLE_ID], linked_ids=[SINGLE_ID])
    assert seen == []                                   # single-turn: unchanged


def test_a_hot_episode_job_a_front_cannot_grade_is_refused_before_its_intake(front):
    started = front.start([SINGLE_ID], linked_ids=[SINGLE_ID])
    with pytest.raises(ValueError, match="started without one"):
        asyncio.run(started.job_set._wire(_entry(EPISODE_TASK, EPISODE_ID), 0.1,
                                          front.jobs[EPISODE_ID]))
    assert front.calls.intakes == []                    # no HF download for nothing
    other = front.start([EPISODE_ID])
    front.calls.intakes.clear()
    with pytest.raises(ValueError, match="restart it to grade the job"):
        asyncio.run(other.job_set._wire(_entry("corpus-swe-3", "swe-agentic-other"), 0.1,
                                        front.jobs["swe-agentic-other"]))
    assert front.calls.intakes == []


def test_the_task_set_is_loaded_once_for_the_lease_check_and_the_intake(monkeypatch):
    import sys
    import types

    from reliquary.environment import agentic_swe
    from reliquary.validator import agentic_intake, agentic_replay, corpus_grade_remote

    loads = []
    corpus = types.ModuleType("reliquary_swe.corpus")
    corpus.load_swesmith_rows = lambda n: loads.append(n) or [SimpleNamespace(
        instance_id=f"i{k}", workdir="/w", problem_statement="p") for k in range(3)]
    taskset = types.ModuleType("reliquary_swe.taskset")
    taskset.PROMPT = "{workdir} {problem_statement}"
    package = types.ModuleType("reliquary_swe")
    package.corpus, package.taskset = corpus, taskset
    monkeypatch.setitem(sys.modules, "reliquary_swe", package)
    monkeypatch.setitem(sys.modules, "reliquary_swe.corpus", corpus)
    monkeypatch.setitem(sys.modules, "reliquary_swe.taskset", taskset)
    agentic_swe.load_swe_source.cache_clear()
    monkeypatch.setattr(agentic_replay, "swesmith_task", lambda instance_id: instance_id)
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda *a, **kw: None)
    monkeypatch.setattr(agentic_swe, "load_turn_renderer", lambda d: R)
    job = _episode_job()
    try:
        assert corpus_grade_remote._job_task(job, 0) == "i0"
        intake = agentic_intake.build_episode_intake(job, checkpoint_dir="/x",
                                                     tokenizer=fakes.Tokenizer(),
                                                     vocab_size=fakes.VOCAB, chunk_tokens=32)
        assert intake.source.instance_id(1) == "i1"
    finally:
        agentic_swe.load_swe_source.cache_clear()
    assert loads == [job.episode.env.num_images]        # once, not once per caller
