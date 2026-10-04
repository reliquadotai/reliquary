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
    return SimpleNamespace(task_id=task_id, job_id=job_id, params={"cap": 0.1},
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
    calls = SimpleNamespace(migrated=[], intakes=[], leases=[])

    class _Store:
        async def read_job(self, job_id):
            return jobs.get(job_id), None

    async def migrate(store, job):
        calls.migrated.append(str(job.job_id))
        return None

    def intake(job, **kw):
        calls.intakes.append((str(job.job_id), kw))
        return SimpleNamespace(renderer=R, source=SweSource([("i0", "p"), ("i1", "q")]))

    async def no_executors(**kw):
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
    monkeypatch.setattr(corpus_service, "renderer_for_job", lambda job, encode, **kw: object())
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: fakes.Tokenizer())
    monkeypatch.setattr(corpus_gpu, "read_info", info)
    monkeypatch.setattr(agentic_intake, "build_episode_intake", intake)
    monkeypatch.setattr(corpus_grade_remote, "check_replay_lease",
                        lambda job, **kw: calls.leases.append(str(job.job_id)))
    monkeypatch.setattr(corpus_executor_store, "list_executors", no_executors)
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(corpus_grading.CorpusGrader, "run", idle)
    monkeypatch.setattr(corpus_validator, "settle_forever", idle_settle)
    # A hot add reads its entry's contract; the renderer it checks is a stub here.
    monkeypatch.setattr(corpus_validator, "_entry_profile", lambda entry: None)
    built = {}

    class _Server:
        def __init__(self, config):
            built["app"] = config.app
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)

    def start(served_ids, linked_ids=()):
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
                registration_gate=False, split=split))
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
    assert front.calls.leases == [EPISODE_ID]
    # The grade executors reach the front: its routes are mounted.
    claim = {"executor_id": "g0", "env_package": "reliquary-swe", "env_version": "0" * 40}
    assert _post(started.app, "/corpus/internal/grade/claim", claim) == 401     # mounted, gated
    # The runbook's reachability probe: a bare POST is 422 (routed), never 404.
    assert _post(started.app, "/corpus/internal/grade/claim", None) == 422
    # The single-turn job still goes to its judge process, as before.
    single = started.served[SINGLE_ID]
    assert single.judge_link is started.link and getattr(single, "grader", None) is None


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
