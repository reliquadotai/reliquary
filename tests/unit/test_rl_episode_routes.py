"""POST /rl/episodes/precommit (plan 2C, Task 4)."""
import time

import httpx
import pytest
from fastapi import FastAPI

bt = pytest.importorskip("bittensor")

from reliquary.protocol.service_episode import EpisodePrecommit  # noqa: E402
from reliquary.protocol.signatures import build_episode_precommit_binding  # noqa: E402
from reliquary.sandbox.rl_routes import build_episode_precommit_router, episode_precommit_path  # noqa: E402
from reliquary.sandbox.sessions import SandboxPolicy  # noqa: E402
from reliquary.validator.corpus_registration import NOT_REGISTERED  # noqa: E402
from tests.unit.episode_v2_fixtures import episode_precommit, episode_runtime  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
OTHER = bt.Keypair.create_from_uri("//Dave")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
ELSEWHERE = bt.Keypair.create_from_uri("//Charlie")
PATH = episode_precommit_path()


def body(precommit, *, at=None, audience=VALIDATOR, signer=MINER):
    at = int(time.time()) if at is None else at
    value = {"miner_hotkey": signer.ss58_address, "at": at, "precommit": precommit.to_dict(), "signature": ""}
    value["signature"] = signer.sign(build_episode_precommit_binding(
        value["precommit"], at=at, validator_hotkey=audience.ss58_address, path=PATH)).hex()
    return value


def app(rt, *, window=1, registration=None):
    current = {"window": window}
    application = FastAPI()
    application.include_router(build_episode_precommit_router(
        record=rt.record_episode_precommit, current_window=lambda: current["window"],
        validator_hotkey=VALIDATOR.ss58_address, policy=SandboxPolicy(), registration=registration))
    return application, current


async def post(application, value):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application),
                                 base_url="http://validator") as client:
        return await client.post(PATH, json=value)


async def test_an_honest_precommit_is_recorded_and_a_resend_is_idempotent(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    precommit = episode_precommit(rt.contract, hotkey=MINER.ss58_address)
    first = await post(application, body(precommit))
    assert first.status_code == 200, first.json()
    assert first.json() == {"precommit_sha256": precommit.sha256, "created": True}
    again = await post(application, body(precommit))
    assert again.status_code == 200 and again.json()["created"] is False
    assert rt.episode_precommit(precommit.sha256) == precommit


async def test_a_precommit_signed_for_another_validator_is_refused(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address),
                                          audience=ELSEWHERE))
    assert (answer.status_code, answer.json()["reason"]) == (403, "bad_signature")


async def test_a_stale_request_is_refused(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address),
                                          at=int(time.time()) - 10_000))
    assert (answer.status_code, answer.json()["reason"]) == (400, "stale_request")


async def test_between_windows_the_miner_retries(tmp_path):
    rt = episode_runtime(tmp_path)
    application, current = app(rt)
    current["window"] = None
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address)))
    assert (answer.status_code, answer.json()["reason"]) == (503, "window_not_open")
    assert int(answer.headers["Retry-After"]) >= 1


async def test_a_precommit_for_another_window_is_stale(tmp_path):
    rt = episode_runtime(tmp_path)
    application, current = app(rt)
    current["window"] = 2
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address)))
    assert (answer.status_code, answer.json()["reason"]) == (409, "precommit_stale")
    assert answer.json()["detail"] == {"window": 2}


async def test_a_precommit_naming_another_hotkey_is_refused(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address),
                                          signer=OTHER))
    assert (answer.status_code, answer.json()["reason"]) == (403, "hotkey_mismatch")


async def test_a_precommit_the_window_refuses_names_its_reason(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    wrong = EpisodePrecommit.from_dict({**episode_precommit(rt.contract, hotkey=MINER.ss58_address).to_dict(),
                                        "pool_sha256": "e" * 64})
    answer = await post(application, body(wrong))
    assert (answer.status_code, answer.json()["reason"]) == (409, "pool_mismatch")


async def test_a_malformed_precommit_is_refused(tmp_path):
    rt = episode_runtime(tmp_path)
    application, _ = app(rt)
    value = body(episode_precommit(rt.contract, hotkey=MINER.ss58_address))
    value["precommit"]["schema"] = "other"
    answer = await post(application, value)
    assert (answer.status_code, answer.json()["reason"]) == (422, "precommit_invalid")


async def test_an_unregistered_hotkey_is_refused(tmp_path):
    rt = episode_runtime(tmp_path)

    async def not_registered(hotkey):
        return NOT_REGISTERED

    application, _ = app(rt, registration=not_registered)
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address)))
    assert (answer.status_code, answer.json()["reason"]) == (403, "hotkey_not_registered")
    assert rt.db.execute("SELECT COUNT(*) FROM service_episode_precommits").fetchone()[0] == 0


async def test_a_past_window_is_refused_before_the_runtime_is_called(tmp_path):
    rt = episode_runtime(tmp_path)
    calls = []
    application = FastAPI()
    application.include_router(build_episode_precommit_router(
        record=lambda value: calls.append(value), current_window=lambda: 5,
        validator_hotkey=VALIDATOR.ss58_address, policy=SandboxPolicy()))
    precommit = episode_precommit(rt.contract, hotkey=MINER.ss58_address)
    assert precommit.window < 5
    answer = await post(application, body(precommit))
    assert (answer.status_code, answer.json()["reason"]) == (409, "precommit_stale")
    assert answer.json()["detail"] == {"window": 5} and calls == []


async def test_a_bad_signature_never_reaches_the_runtime(tmp_path):
    rt = episode_runtime(tmp_path)
    calls = []
    application = FastAPI()
    application.include_router(build_episode_precommit_router(
        record=lambda value: calls.append(value), current_window=lambda: 1,
        validator_hotkey=VALIDATOR.ss58_address, policy=SandboxPolicy()))
    answer = await post(application, body(episode_precommit(rt.contract, hotkey=MINER.ss58_address),
                                          audience=ELSEWHERE))
    assert answer.status_code == 403 and calls == []
