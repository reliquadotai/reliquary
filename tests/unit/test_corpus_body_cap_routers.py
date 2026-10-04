"""Every corpus router carries its job's submit body cap (ruling P10), and a hot
episode job the split validator cannot serve is refused for good."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.job import parse_job
from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
from tests.unit.test_corpus_job_episode import _manifest

MIB = 1024 * 1024


def _big_job():
    raw = _manifest(with_episode=False, prompt_count=3)
    raw["sampling"] = {**raw["sampling"], "max_new_tokens": 32768, "n": 16}
    return parse_job(raw)


def _accepts_twelve_megabytes(router):
    app = FastAPI()
    app.include_router(router)
    junk = b"x" * 12_000_000      # not JSON: past the cap check it answers 422, never 413
    assert TestClient(app).post("/corpus/submit", content=junk).status_code == 422


def test_the_eval_control_router_carries_its_jobs_cap():
    from reliquary.validator.eval_control import build_eval_control

    job = _big_job()
    _, job_set = build_eval_control(
        store=MagicMock(), records=MagicMock(), dispatcher=MagicMock(), directory=MagicMock(),
        verify_signature=lambda request: True)
    w = SimpleNamespace(entry=SimpleNamespace(job_id=job.job_id), tokenizer=MagicMock(),
                        renderer=None, prompt_job_for=None, on_accepted=None, job=job,
                        proof=SimpleNamespace(chunk_tokens=32), vocab_size=None,
                        is_banned=None, seen_index=None)
    _accepts_twelve_megabytes(job_set._router_for(w))


def test_the_single_job_server_mount_carries_its_jobs_cap():
    from reliquary.validator.server import ValidatorServer

    job = _big_job()
    holder = SimpleNamespace(app=FastAPI())
    entry = SimpleNamespace(job_id=job.job_id, mechanism=MECHANISM_CORPUS_GENERATION)
    assert ValidatorServer.mount_corpus_router(
        holder, entry, store=MagicMock(), tokenizer=MagicMock(), renderer=None,
        verify_signature=lambda request: True, job=job)
    junk = b"x" * 12_000_000
    assert TestClient(holder.app).post("/corpus/submit", content=junk).status_code == 422


def test_a_hot_episode_job_in_a_split_judge_group_is_refused_for_good():
    from types import SimpleNamespace

    from reliquary.validator.corpus_hot_jobs import REFUSED
    from reliquary.validator.corpus_validator import split_episode_refusal

    episode, single = parse_job(_manifest(prompt_count=3)), parse_job(_manifest(with_episode=False))
    linked = SimpleNamespace(links={str(episode.job_id): object()})
    assert split_episode_refusal(None, episode) is None
    assert split_episode_refusal(linked, single) is None
    # The front serves an episode job no judge group names.
    assert split_episode_refusal(SimpleNamespace(links={}), episode) is None
    kind, why = split_episode_refusal(linked, episode)
    assert kind == REFUSED and "judge" in why
