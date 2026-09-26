"""Only hotkeys registered on the subnet may mine the corpus task: an
unregistered key is never paid, so its work would only spend audit GPU."""

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.validator.corpus_registration import (
    NOT_REGISTERED,
    UNAVAILABLE,
    RegisteredHotkeys,
)
from tests.unit.test_corpus_route_ban import _submit
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    EOS,
    _Tokenizer,
    _r2_client,
    fake_r2,
    seeded_job,
)

TTL, GRACE = 100.0, 200.0


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def _registry(load, clock):
    return RegisteredHotkeys(load=load, clock=clock, ttl_seconds=TTL, grace_seconds=GRACE)


def _loads(*results):
    """A chain read returning each result in turn; an exception is raised."""
    queue = list(results)

    async def load():
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return load


def test_before_any_snapshot_every_hotkey_is_unavailable_not_refused():
    registry = _registry(_loads(), _Clock())
    assert asyncio.run(registry.reason("5Hot")) == UNAVAILABLE


def test_a_registered_hotkey_is_admitted_and_an_unregistered_one_refused():
    clock = _Clock()
    registry = _registry(_loads({"5Hot"}), clock)
    assert asyncio.run(registry.refresh()) is True
    assert asyncio.run(registry.reason("5Hot")) is None
    assert asyncio.run(registry.reason("5Other")) == NOT_REGISTERED


def test_a_miss_on_a_snapshot_older_than_the_ttl_is_unavailable():
    """It may predate the miner's registration: retry, don't refuse."""
    clock = _Clock()
    registry = _registry(_loads({"5Hot"}), clock)
    asyncio.run(registry.refresh())
    clock.now += TTL + 1
    assert asyncio.run(registry.reason("5Other")) == UNAVAILABLE
    assert asyncio.run(registry.reason("5Hot")) is None


def test_past_the_grace_even_a_registered_hotkey_is_unavailable():
    clock = _Clock()
    registry = _registry(_loads({"5Hot"}), clock)
    asyncio.run(registry.refresh())
    clock.now += GRACE + 1
    assert asyncio.run(registry.reason("5Hot")) == UNAVAILABLE


def test_a_failed_or_empty_refresh_keeps_the_last_snapshot():
    clock = _Clock()
    registry = _registry(_loads({"5Hot"}, RuntimeError("rpc down"), set()), clock)
    asyncio.run(registry.refresh())
    clock.now += 10
    assert asyncio.run(registry.refresh()) is False
    assert asyncio.run(registry.refresh()) is False
    assert asyncio.run(registry.reason("5Hot")) is None
    assert asyncio.run(registry.reason("5Other")) == NOT_REGISTERED


# -- the route -------------------------------------------------------------

def _client(seeded_job, registration):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=seeded_job.store, tokenizer=_Tokenizer(),
        renderer=seeded_job.renderer,
        verify_signature=lambda request: request.signature != "bad",
        prompt_job_for=seeded_job.prompt_job_for,
        registration=registration,
    ))
    return TestClient(app)


def _answering(reason):
    calls = []

    async def registration(hotkey):
        calls.append(hotkey)
        return reason

    registration.calls = calls
    return registration


def test_an_unregistered_hotkey_is_refused_before_the_store_is_touched(seeded_job):
    reads, writes = seeded_job.store.job_reads, seeded_job.ledger_writes()
    body = _submit(_client(seeded_job, _answering(NOT_REGISTERED)),
                   tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is False
    assert body["reason"] == "hotkey_not_registered"
    assert seeded_job.store.job_reads == reads
    assert seeded_job.ledger_writes() == writes


def test_an_unknown_registration_is_a_retryable_503(seeded_job):
    reads, writes = seeded_job.store.job_reads, seeded_job.ledger_writes()
    response = _submit(_client(seeded_job, _answering(UNAVAILABLE)), tokens=[7] * 16 + [EOS])
    assert response.status_code == 503
    assert response.json()["detail"] == "corpus_registration_unavailable"
    assert seeded_job.store.job_reads == reads
    assert seeded_job.ledger_writes() == writes


def test_a_registered_hotkey_is_admitted(seeded_job):
    body = _submit(_client(seeded_job, _answering(None)), tokens=[7] * 16 + [EOS]).json()
    assert body["accepted"] is True


def test_a_bad_signature_never_reaches_the_registration_check(seeded_job):
    registration = _answering(None)
    body = _submit(_client(seeded_job, registration), tokens=[7] * 16 + [EOS],
                   signature="bad").json()
    assert body["reason"] == "bad_signature"
    assert registration.calls == []
