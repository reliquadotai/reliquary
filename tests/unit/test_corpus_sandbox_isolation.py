"""A sandbox-service failure at startup disables the signed-sandbox jobs only:
their intake refuses (retryable), every other job keeps its intake and serves."""

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.validator.corpus_validator import (
    SANDBOX_UNAVAILABLE,
    SandboxUnavailableIntake,
    start_sandbox_services,
)
from tests.unit.test_corpus_route_episode import (  # noqa: F401  (fixtures)
    TOKENIZER, _request, episode_store,
)
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)
from tests.unit.test_corpus_route_records import _Records


def _wired(job_id, execution=None, harness="bash"):
    episode = (None if execution is None and harness is None else
               SimpleNamespace(execution=execution or "replay", harness=harness))
    return SimpleNamespace(job=SimpleNamespace(job_id=job_id, episode=episode),
                           entry=SimpleNamespace(task_id=f"task-{job_id}"),
                           episode_intake=object())


class _Services:
    def __init__(self, fail: bool):
        self.fail, self.forgotten = fail, []

    async def start(self):
        if self.fail:
            raise TimeoutError("R2 session restore timed out")

    def forget(self, job_id):
        self.forgotten.append(job_id)


def test_a_failed_start_disables_only_the_signed_jobs():
    signed = _wired("signed", execution="signed_sandbox")
    replay, suite, text = _wired("replay"), _wired("suite", harness="suite"), _wired("text", harness=None)
    others = {w.job.job_id: w.episode_intake for w in (replay, suite, text)}
    services, disabled = _Services(fail=True), []

    started = asyncio.run(start_sandbox_services(
        services, [signed, replay, suite, text], on_disabled=lambda w, why: disabled.append(w)))

    assert started is False
    assert isinstance(signed.episode_intake, SandboxUnavailableIntake)
    assert services.forgotten == ["signed"]
    assert disabled == [signed]
    assert {w.job.job_id: w.episode_intake for w in (replay, suite, text)} == others


def test_a_good_start_changes_nothing():
    signed, replay = _wired("signed", execution="signed_sandbox"), _wired("replay")
    before = (signed.episode_intake, replay.episode_intake)
    services = _Services(fail=False)
    assert asyncio.run(start_sandbox_services(services, [signed, replay])) is True
    assert (signed.episode_intake, replay.episode_intake) == before
    assert services.forgotten == []


def test_the_disabled_intake_answers_a_retryable_503_and_consumes_nothing(episode_store):
    from reliquary.validator.corpus_service import build_corpus_router

    records, accepted = _Records(), []
    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-agentic-v1", store=episode_store, tokenizer=TOKENIZER, renderer=None,
        verify_signature=lambda r: True, records=records, on_accepted=accepted.append,
        proof_chunk_tokens=32, episode_intake=SandboxUnavailableIntake("restore failed")))
    response = TestClient(app).post("/corpus/submit", json=_request().model_dump())

    assert response.status_code == 503
    assert response.json()["detail"] == SANDBOX_UNAVAILABLE
    assert int(response.headers["Retry-After"]) > 0
    assert accepted == []
