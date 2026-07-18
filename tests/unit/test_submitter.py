"""Tests for the miner HTTP submitter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.miner.submitter import (
    NoValidatorFoundError,
    SubmissionError,
    discover_validator_url,
    get_runtime_contract_v1,
    get_window_state_v2,
    submit_batch_v2,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
    RolloutSubmission,
    RuntimeContract,
    RuntimeFingerprint,
    WindowState,
)
from reliquary.shared.runtime_fingerprint import collect_runtime_fingerprint


# --------------------- discover_validator_url ---------------------


def test_discover_picks_first_permitted_with_routable_axon() -> None:
    meta = SimpleNamespace(
        validator_permit=[False, True, True],
        axons=[
            SimpleNamespace(ip="1.1.1.1", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=9000),
            SimpleNamespace(ip="3.3.3.3", port=9001),
        ],
    )
    assert discover_validator_url(meta) == "http://2.2.2.2:9000"


def test_discover_skips_unset_axon_ip() -> None:
    meta = SimpleNamespace(
        validator_permit=[True, True],
        axons=[
            SimpleNamespace(ip="0.0.0.0", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=8888),
        ],
    )
    assert discover_validator_url(meta) == "http://2.2.2.2:8888"


def test_discover_falls_back_to_default_port_when_axon_port_zero() -> None:
    meta = SimpleNamespace(
        validator_permit=[True],
        axons=[SimpleNamespace(ip="1.1.1.1", port=0)],
    )
    assert discover_validator_url(meta) == f"http://1.1.1.1:{VALIDATOR_HTTP_PORT}"


def test_discover_raises_when_no_permitted() -> None:
    meta = SimpleNamespace(
        validator_permit=[False, False],
        axons=[
            SimpleNamespace(ip="1.1.1.1", port=8888),
            SimpleNamespace(ip="2.2.2.2", port=8888),
        ],
    )
    with pytest.raises(NoValidatorFoundError):
        discover_validator_url(meta)


def test_discover_raises_when_metagraph_malformed() -> None:
    with pytest.raises(NoValidatorFoundError):
        discover_validator_url(SimpleNamespace())


# ---- v2 submitter tests ----


def _rollouts(k=4):
    out = []
    for i in range(8):
        out.append(
            RolloutSubmission(
                tokens=[1, 2, 3],
                reward=1.0 if i < k else 0.0,
                commit={"tokens": [1, 2, 3], "proof_version": "v7"},
                env_name="openmathinstruct",
            )
        )
    return out


def _v2_request():
    return BatchSubmissionRequest(
        miner_hotkey="hk",
        prompt_idx=42,
        window_start=100,
        merkle_root="00" * 32,
        rollouts=_rollouts(),
        checkpoint_hash="sha256:test",
        protocol_version=2,
    )


@pytest.mark.asyncio
async def test_submit_batch_v2_ok(monkeypatch):
    responses = [
        httpx.Response(
            200,
            json=BatchSubmissionResponse(
                accepted=True, reason=RejectReason.ACCEPTED
            ).model_dump(mode="json"),
        )
    ]

    async def _post(self, url, content=None, headers=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _v2_request(), client=client)
    assert resp.accepted is True
    assert resp.reason == RejectReason.ACCEPTED
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v2_retries_one_idempotent_precommit_then_reveals(
    monkeypatch,
):
    import reliquary.miner.submitter as submitter

    calls = []
    drand_calls = []

    def _sign_envelope(**kwargs):
        return f"{kwargs['drand_round']}:{kwargs['nonce']}".encode()

    def _sign_precommit(**kwargs):
        return f"precommit:{kwargs['drand_round']}:{kwargs['nonce']}".encode()

    def _drand_round():
        drand_calls.append(100)
        return 100

    async def _post(self, url, content=None, headers=None, timeout=None):
        calls.append((url, json.loads(content), headers))
        if len(calls) == 1:
            raise httpx.ConnectError(
                "transient",
                request=httpx.Request("POST", url),
            )
        if url.endswith("/submit/precommit"):
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "reason": RejectReason.ACCEPTED.value,
                    "receipt_id": "receipt-1",
                    "upload_deadline_ts": 123.0,
                },
            )
        return httpx.Response(200, json={
            "accepted": True,
            "reason": RejectReason.SUBMITTED.value,
        })

    monkeypatch.setattr(submitter, "_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(submitter, "sign_envelope", _sign_envelope)
    monkeypatch.setattr(submitter, "sign_precommit", _sign_precommit)
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()

    response = await submit_batch_v2(
        "http://fake",
        _v2_request(),
        client=client,
        wallet=object(),
        randomness="ab" * 32,
        drand_round_fn=_drand_round,
    )

    assert response.accepted is True
    assert drand_calls == [100]
    assert calls[0][0].endswith("/submit/precommit")
    assert calls[1][0].endswith("/submit/precommit")
    assert calls[0][1] == calls[1][1]
    assert calls[2][0].endswith("/submit")
    assert calls[2][1]["drand_round"] == 100
    assert len(calls[2][1]["rollouts"]) == 8
    assert calls[2][2]["X-Reliquary-Precommit"] == "receipt-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v2_reject_reason_propagated(monkeypatch):
    async def _post(self, url, content=None, headers=None, timeout=None):
        return httpx.Response(
            200,
            json=BatchSubmissionResponse(
                accepted=False, reason=RejectReason.PROMPT_IN_COOLDOWN
            ).model_dump(mode="json"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _v2_request(), client=client)
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_IN_COOLDOWN
    await client.aclose()


@pytest.mark.asyncio
async def test_get_window_state_v2(monkeypatch):
    state = GrpoBatchState(
        state=WindowState.OPEN,
        window_n=100,
        anchor_block=1000,
        cooldown_prompts=[42, 7],
        valid_submissions=3,
        checkpoint_n=0,
    )

    async def _get(self, url, timeout=None):
        return httpx.Response(200, json=state.model_dump(mode="json"))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    client = httpx.AsyncClient()
    s = await get_window_state_v2("http://fake", client=client)
    assert s.window_n == 100
    assert set(s.cooldown_prompts) == {42, 7}
    await client.aclose()


@pytest.mark.asyncio
async def test_get_window_state_v2_passes_env_query_param(monkeypatch):
    """Per-env cooldown: the miner must select which env's cooldown it reads
    by passing ``env=`` to ``/state`` (the flat field reflects only one env)."""
    state = GrpoBatchState(
        state=WindowState.OPEN, window_n=100, anchor_block=1000,
        cooldown_prompts=[5], valid_submissions=0, checkpoint_n=0,
    )
    seen = {}

    async def _get(self, url, timeout=None):
        seen["url"] = url
        return httpx.Response(200, json=state.model_dump(mode="json"))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    client = httpx.AsyncClient()
    await get_window_state_v2("http://fake", env="opencode", client=client)
    assert "env=opencode" in seen["url"]
    # No env → no query param (backward compatible).
    await get_window_state_v2("http://fake", client=client)
    assert "?" not in seen["url"]
    await client.aclose()


@pytest.mark.asyncio
async def test_get_runtime_contract_v1_uses_separate_capability_endpoint(
    monkeypatch,
):
    contract = RuntimeContract(
        validator_profile=RuntimeFingerprint.model_validate(
            collect_runtime_fingerprint()
        )
    )
    seen = {}

    async def _get(self, url, timeout=None):
        seen["url"] = url
        return httpx.Response(200, json=contract.model_dump(mode="json"))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    client = httpx.AsyncClient()
    result = await get_runtime_contract_v1("http://fake", client=client)

    assert seen["url"] == "http://fake/runtime-contract"
    assert result.telemetry_version == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v2_503_maps_to_window_not_active(monkeypatch):
    """HTTP 503 from /submit short-circuits to WINDOW_NOT_ACTIVE (no retry)."""
    call_count = {"n": 0}

    async def _post(self, url, content=None, headers=None, timeout=None):
        call_count["n"] += 1
        return httpx.Response(503, json={"detail": "no_active_window"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _v2_request(), client=client)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WINDOW_NOT_ACTIVE
    # Crucially: no retries. One call, not three.
    assert call_count["n"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v2_409_maps_to_window_mismatch(monkeypatch):
    async def _post(self, url, content=None, headers=None, timeout=None):
        return httpx.Response(409, json={"detail": "window_mismatch"})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    client = httpx.AsyncClient()
    resp = await submit_batch_v2("http://fake", _v2_request(), client=client)
    assert resp.accepted is False
    assert resp.reason == RejectReason.WINDOW_MISMATCH
    await client.aclose()
