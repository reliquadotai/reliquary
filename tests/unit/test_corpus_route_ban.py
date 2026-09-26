"""A banned miner is refused at the route, before the ledgers are touched."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    EOS,
    _Tokenizer,
    _faithful_prompt,
    _r2_client,
    _text_for,
    fake_r2,
    seeded_job,
)


def _client(seeded_job, *, is_banned=None):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=seeded_job.store,
            tokenizer=_Tokenizer(),
            renderer=seeded_job.renderer,
            verify_signature=lambda request: request.signature != "bad",
            prompt_job_for=seeded_job.prompt_job_for,
            is_banned=is_banned,
        )
    )
    return TestClient(app)


def _submit(client, *, tokens, hotkey="5Hot", signature="ok", prompt_index=0):
    return client.post(
        "/corpus/submit",
        json={
            "job_id": "swe-v1",
            "miner_hotkey": hotkey,
            "cursor": 0,
            "prompt_index": prompt_index,
            "checkpoint_sha256": "a" * 64,
            "rendered_prompt": _faithful_prompt(prompt_index),
            "completions": [{"tokens": tokens, "text": _text_for(tokens)}],
            "signature": signature,
        },
    )


async def _banned(hotkey):
    return hotkey == "5Banned"


async def _nobody_banned(hotkey):
    return False


def test_a_banned_hotkey_is_refused_and_nothing_is_written(seeded_job):
    before_reads = seeded_job.store.job_reads
    before_writes = seeded_job.ledger_writes()
    client = _client(seeded_job, is_banned=_banned)

    body = _submit(client, tokens=[7] * 16 + [EOS], hotkey="5Banned").json()

    assert body["accepted"] is False
    assert body["reason"] == "miner_banned"
    assert seeded_job.store.job_reads == before_reads
    assert seeded_job.ledger_writes() == before_writes


def test_an_unbanned_hotkey_is_admitted(seeded_job):
    client = _client(seeded_job, is_banned=_nobody_banned)

    body = _submit(client, tokens=[7] * 16 + [EOS], hotkey="5Hot").json()

    assert body["accepted"] is True


def test_is_banned_none_changes_nothing(seeded_job):
    client = _client(seeded_job, is_banned=None)

    body = _submit(client, tokens=[7] * 16 + [EOS], hotkey="5Banned").json()

    assert body["accepted"] is True


async def _raising_ban(hotkey):
    raise RuntimeError("miners.json unreadable")


def test_is_banned_raising_gives_503_and_writes_nothing(seeded_job):
    before_reads = seeded_job.store.job_reads
    before_writes = seeded_job.ledger_writes()
    client = _client(seeded_job, is_banned=_raising_ban)

    response = _submit(client, tokens=[7] * 16 + [EOS], hotkey="5Hot")

    assert response.status_code == 503
    assert response.json()["detail"] == "corpus_store_unavailable"
    assert seeded_job.store.job_reads == before_reads
    assert seeded_job.ledger_writes() == before_writes


def test_a_bad_signature_is_refused_before_the_ban_check_runs(seeded_job):
    """The ban check sits after the signature check, so an unsigned request
    cannot use it to probe whether a hotkey is banned."""
    calls = []

    async def _spying_ban(hotkey):
        calls.append(hotkey)
        return True

    client = _client(seeded_job, is_banned=_spying_ban)

    body = _submit(
        client, tokens=[7] * 16 + [EOS], hotkey="5Hot", signature="bad"
    ).json()

    assert body["reason"] == "bad_signature"
    assert calls == []
