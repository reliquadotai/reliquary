"""Bounded miner state is additive and preserves the legacy state contract."""

import httpx
import pytest
from fastapi.testclient import TestClient

from reliquary.miner.submitter import get_miner_state_v1
from reliquary.protocol.submission import (
    MinerState,
    WindowState,
    decode_cooldown_bitmap,
    encode_cooldown_bitmap,
)
from reliquary.validator.cooldown import CooldownMap

from tests.unit.test_validator_server import (
    FakeEnv,
    ValidatorServer,
    _batcher,
)


class _CodeEnv(FakeEnv):
    name = "opencodeinstruct"


def _server() -> ValidatorServer:
    math_cooldown = CooldownMap(cooldown_windows=1000)
    math_cooldown.record_batched(11, 490)
    code_cooldown = CooldownMap(cooldown_windows=1000)
    code_cooldown.record_batched(22, 490)
    server = ValidatorServer()
    server.set_active_batchers({
        "openmathinstruct": _batcher(cooldown_map=math_cooldown),
        "opencodeinstruct": _batcher(
            cooldown_map=code_cooldown,
            env=_CodeEnv(),
        ),
    })
    server.set_current_state(WindowState.OPEN)
    return server


def test_cooldown_bitmap_roundtrip_is_range_bounded():
    encoded, count = encode_cooldown_bitmap({2, 5, 7, 99}, (5, 10))
    assert count == 2
    assert decode_cooldown_bitmap(encoded, (5, 10)) == {5, 7}


def test_cooldown_bitmap_rejects_noncanonical_tail_bits():
    # A five-bit range leaves the upper three bits of its final byte unused.
    with pytest.raises(ValueError, match="out-of-range bits"):
        decode_cooldown_bitmap("gA==", (0, 5))


def test_miner_state_carries_each_environment_once():
    client = TestClient(_server().app)
    response = client.get("/miner-state")
    assert response.status_code == 200
    state = MinerState.model_validate(response.json())
    assert set(state.environments) == {
        "openmathinstruct",
        "opencodeinstruct",
    }
    assert 11 in state.environments["openmathinstruct"].cooldown_prompts()
    assert 22 not in state.environments["openmathinstruct"].cooldown_prompts()
    assert 22 in state.environments["opencodeinstruct"].cooldown_prompts()


def test_miner_state_etag_supports_conditional_polling():
    client = TestClient(_server().app)
    first = client.get("/miner-state")
    etag = first.headers["etag"]
    second = client.get("/miner-state", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == etag


def test_miner_state_does_not_change_legacy_state_bytes():
    client = TestClient(_server().app)
    before = client.get("/state", params={"env": "openmathinstruct"}).content
    assert client.get("/miner-state").status_code == 200
    after = client.get("/state", params={"env": "openmathinstruct"}).content
    assert after == before


def test_concurrent_epoch_requires_lane_aware_state():
    batcher = _batcher()
    server = ValidatorServer()
    server.set_active_epoch_batchers({
        ("fake", batcher.window_start): batcher,
    })
    server.set_current_state(WindowState.OPEN)
    assert TestClient(server.app).get("/miner-state").status_code == 409


@pytest.mark.asyncio
async def test_miner_state_client_reuses_etag_on_304():
    state_body = TestClient(_server().app).get("/miner-state").json()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("If-None-Match") == '"state"':
            return httpx.Response(304, headers={"ETag": '"state"'})
        return httpx.Response(
            200,
            json=state_body,
            headers={"ETag": '"state"'},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        state, etag = await get_miner_state_v1("https://validator", client=client)
        cached, next_etag = await get_miner_state_v1(
            "https://validator",
            client=client,
            etag=etag,
        )

    assert state is not None
    assert cached is None
    assert next_etag == '"state"'
    assert requests[-1].headers["If-None-Match"] == '"state"'
