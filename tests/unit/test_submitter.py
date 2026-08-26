"""Tests for the miner HTTP submitter."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.miner.submitter import (
    NoValidatorFoundError,
    SubmissionError,
    discover_validator_url,
    get_checkpoint_epoch_plan_v1,
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
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    canonical_manifest_bytes,
    generation_contract_sha256,
    manifest_sha256,
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
    assert "generation_profile_id" not in calls[0][1]
    assert calls[2][0].endswith("/submit")
    assert calls[2][1]["drand_round"] == 100
    assert "generation_profile_id" not in calls[2][1]
    assert len(calls[2][1]["rollouts"]) == 8
    assert calls[2][2]["X-Reliquary-Precommit"] == "receipt-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_submit_batch_v3_serializes_profile_in_precommit_and_reveal(
    monkeypatch,
):
    import reliquary.miner.submitter as submitter

    calls = []

    async def _post(self, url, content=None, headers=None, timeout=None):
        calls.append((url, json.loads(content), headers))
        if url.endswith("/submit/precommit"):
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "reason": RejectReason.ACCEPTED.value,
                    "receipt_id": "receipt-v3",
                    "upload_deadline_ts": 123.0,
                },
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "reason": RejectReason.SUBMITTED.value,
            },
        )

    request = _v2_request().model_copy(
        update={
            "protocol_version": 3,
            "generation_profile_id": "qwen35-4b-auction-v3",
        }
    )
    monkeypatch.setattr(submitter, "sign_envelope", lambda **kwargs: b"envelope")
    monkeypatch.setattr(submitter, "sign_precommit", lambda **kwargs: b"precommit")
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    async with httpx.AsyncClient() as client:
        response = await submit_batch_v2(
            "http://fake",
            request,
            client=client,
            wallet=object(),
            randomness="ab" * 32,
            drand_round_fn=lambda: 100,
        )

    assert response.accepted is True
    assert calls[0][1]["generation_profile_id"] == "qwen35-4b-auction-v3"
    assert calls[1][1]["generation_profile_id"] == "qwen35-4b-auction-v3"


@pytest.mark.asyncio
async def test_submit_batch_v2_refreshes_stale_precommit_after_backoff(
    monkeypatch,
):
    import reliquary.miner.submitter as submitter

    events = []
    rounds = iter((100, 101))

    def _drand_round():
        value = next(rounds)
        events.append(("finalize", value))
        return value

    async def _sleep(delay):
        events.append(("sleep", delay))

    async def _post(self, url, content=None, headers=None, timeout=None):
        body = json.loads(content)
        if url.endswith("/submit/precommit"):
            events.append(("precommit", body["drand_round"]))
            if body["drand_round"] == 100:
                return httpx.Response(
                    200,
                    json={
                        "accepted": False,
                        "reason": RejectReason.STALE_ROUND.value,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "reason": RejectReason.ACCEPTED.value,
                    "receipt_id": "receipt-fresh",
                    "upload_deadline_ts": 123.0,
                },
            )
        events.append(("reveal", body["drand_round"]))
        assert headers["X-Reliquary-Precommit"] == "receipt-fresh"
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "reason": RejectReason.SUBMITTED.value,
            },
        )

    monkeypatch.setattr(submitter, "_RETRY_DELAYS", (1.0, 2.0))
    monkeypatch.setattr(submitter.asyncio, "sleep", _sleep)
    monkeypatch.setattr(submitter, "sign_envelope", lambda **kwargs: b"envelope")
    monkeypatch.setattr(submitter, "sign_precommit", lambda **kwargs: b"precommit")
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    async with httpx.AsyncClient() as client:
        response = await submit_batch_v2(
            "http://fake",
            _v2_request(),
            client=client,
            wallet=object(),
            randomness="ab" * 32,
            drand_round_fn=_drand_round,
        )

    assert response.accepted is True
    assert events == [
        ("finalize", 100),
        ("precommit", 100),
        ("sleep", 1.0),
        ("finalize", 101),
        ("precommit", 101),
        ("reveal", 101),
    ]


@pytest.mark.asyncio
async def test_submit_batch_v2_retries_transient_precommit_capacity(
    monkeypatch,
):
    import reliquary.miner.submitter as submitter

    events = []
    precommit_bodies = []

    async def _sleep(delay):
        events.append(("sleep", delay))

    async def _post(self, url, content=None, headers=None, timeout=None):
        body = json.loads(content)
        if url.endswith("/submit/precommit"):
            precommit_bodies.append(body)
            events.append(("precommit", len(precommit_bodies)))
            if len(precommit_bodies) == 1:
                return httpx.Response(
                    200,
                    json={
                        "accepted": False,
                        "reason": RejectReason.BATCH_FILLED.value,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "reason": RejectReason.ACCEPTED.value,
                    "receipt_id": "receipt-after-capacity-recycled",
                    "upload_deadline_ts": 123.0,
                },
            )
        events.append(("reveal", headers["X-Reliquary-Precommit"]))
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "reason": RejectReason.SUBMITTED.value,
            },
        )

    monkeypatch.setattr(submitter, "_RETRY_DELAYS", (1.0, 2.0))
    monkeypatch.setattr(submitter.asyncio, "sleep", _sleep)
    monkeypatch.setattr(submitter, "sign_envelope", lambda **kwargs: b"envelope")
    monkeypatch.setattr(submitter, "sign_precommit", lambda **kwargs: b"precommit")
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    async with httpx.AsyncClient() as client:
        response = await submit_batch_v2(
            "http://fake",
            _v2_request(),
            client=client,
            wallet=object(),
            randomness="ab" * 32,
            drand_round_fn=lambda: 100,
        )

    assert response.accepted is True
    assert events == [
        ("precommit", 1),
        ("sleep", 1.0),
        ("precommit", 2),
        ("reveal", "receipt-after-capacity-recycled"),
    ]
    assert precommit_bodies[0] == precommit_bodies[1]


@pytest.mark.asyncio
async def test_submit_batch_v2_waits_out_unsafe_drand_boundary(monkeypatch):
    import reliquary.miner.submitter as submitter

    now = [2.5]
    sleeps = []
    submitted_rounds = []

    async def _sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    async def _post(self, url, content=None, headers=None, timeout=None):
        body = json.loads(content)
        submitted_rounds.append(body["drand_round"])
        if url.endswith("/submit/precommit"):
            return httpx.Response(
                200,
                json={
                    "accepted": True,
                    "reason": RejectReason.ACCEPTED.value,
                    "receipt_id": "receipt-safe-round",
                    "upload_deadline_ts": 30.0,
                },
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "reason": RejectReason.SUBMITTED.value,
            },
        )

    monkeypatch.setattr(submitter.time, "time", lambda: now[0])
    monkeypatch.setattr(submitter.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        "reliquary.infrastructure.drand.get_current_chain",
        lambda: {"genesis_time": 0, "period": 3},
    )
    monkeypatch.setattr(submitter, "sign_envelope", lambda **kwargs: b"envelope")
    monkeypatch.setattr(submitter, "sign_precommit", lambda **kwargs: b"precommit")
    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    async with httpx.AsyncClient() as client:
        response = await submit_batch_v2(
            "http://fake",
            _v2_request(),
            client=client,
            wallet=object(),
            randomness="ab" * 32,
        )

    assert response.accepted is True
    assert sleeps == [pytest.approx(0.55)]
    assert submitted_rounds == [2, 2]


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
    await get_window_state_v2(
        "http://fake",
        env="opencode",
        window=116,
        client=client,
    )
    assert "env=opencode" in seen["url"]
    assert "window=116" in seen["url"]
    # No env → no query param (backward compatible).
    await get_window_state_v2("http://fake", client=client)
    assert "?" not in seen["url"]
    with pytest.raises(ValueError, match="window requires env"):
        await get_window_state_v2("http://fake", window=116, client=client)
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


def _checkpoint_epoch_plan_fixture():
    return build_epoch_plan(
        protocol=ProtocolBinding(
            profile_id="experimental-fixture",
            protocol_version=600,
            generation_contract_sha256=generation_contract_sha256(
                {"fixture": True}
            ),
        ),
        checkpoint=CheckpointBinding(
            number=4,
            repo_id="example/checkpoint",
            revision="a" * 40,
            commit_observed_round=100,
        ),
        epoch_beacon=BeaconBinding(
            source="drand",
            chain="quicknet",
            chain_hash="b" * 64,
            round=101,
            randomness="c" * 64,
        ),
        beacon_delay_rounds=1,
        first_window=80,
        window_count=2,
        warmup_rounds=3,
        window_schedule=WindowSchedule(
            mode="concurrent_checkpoint_epoch",
            collection_seconds=60.0,
            timeout_seconds=7200,
        ),
        training_mode="sequential_steps",
        target_groups_per_environment_lane=16,
        candidate_limit_per_environment_lane=24,
        environment_universes={"math": 100},
        prompt_range_size=8,
    )


@pytest.mark.asyncio
async def test_get_checkpoint_epoch_plan_binds_etag_state_and_canonical_body(
    monkeypatch,
):
    import reliquary.miner.submitter as submitter_module

    plan = _checkpoint_epoch_plan_fixture()
    digest = manifest_sha256(plan)
    seen = {}

    async def _get(self, url, timeout=None):
        seen["url"] = url
        return httpx.Response(
            200,
            content=canonical_manifest_bytes(plan),
            headers={"ETag": f'"{digest}"'},
        )

    async def _verify_public_beacon(fetched_plan):
        assert fetched_plan == plan

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    monkeypatch.setattr(
        submitter_module,
        "_verify_checkpoint_epoch_public_beacon",
        _verify_public_beacon,
    )
    client = httpx.AsyncClient()
    fetched = await get_checkpoint_epoch_plan_v1(
        "http://fake",
        expected_manifest_sha256=digest,
        client=client,
    )

    assert seen["url"] == "http://fake/checkpoint-epoch"
    assert fetched == plan
    await client.aclose()


@pytest.mark.asyncio
async def test_get_checkpoint_epoch_plan_can_discover_during_warmup(monkeypatch):
    import reliquary.miner.submitter as submitter_module

    plan = _checkpoint_epoch_plan_fixture()
    digest = manifest_sha256(plan)

    async def _get(self, url, timeout=None):
        return httpx.Response(
            200,
            content=canonical_manifest_bytes(plan),
            headers={"ETag": f'"{digest}"'},
        )

    async def _verify_public_beacon(_plan):
        return None

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    monkeypatch.setattr(
        submitter_module,
        "_verify_checkpoint_epoch_public_beacon",
        _verify_public_beacon,
    )
    client = httpx.AsyncClient()

    assert await get_checkpoint_epoch_plan_v1(
        "http://fake",
        client=client,
    ) == plan
    await client.aclose()


@pytest.mark.asyncio
async def test_checkpoint_epoch_public_beacon_must_match_independent_drand(
    monkeypatch,
):
    import reliquary.infrastructure.drand as drand
    import reliquary.miner.submitter as submitter_module

    plan = _checkpoint_epoch_plan_fixture()
    monkeypatch.setattr(
        drand,
        "get_current_chain",
        lambda: {
            "name": plan.epoch_beacon.chain,
            "hash": plan.epoch_beacon.chain_hash,
        },
    )
    monkeypatch.setattr(
        drand,
        "get_beacon",
        lambda **_kwargs: {
            "source": "drand",
            "chain": plan.epoch_beacon.chain,
            "chain_hash": plan.epoch_beacon.chain_hash,
            "round": plan.epoch_beacon.round,
            "randomness": "d" * 64,
            "signature": "e" * 192,
        },
    )

    with pytest.raises(SubmissionError, match="public beacon"):
        await submitter_module._verify_checkpoint_epoch_public_beacon(plan)


@pytest.mark.asyncio
async def test_checkpoint_epoch_public_beacon_verifier_is_time_bounded(
    monkeypatch,
):
    import hashlib
    import time

    import reliquary.infrastructure.drand as drand
    import reliquary.miner.submitter as submitter_module

    plan = _checkpoint_epoch_plan_fixture()
    signature = "aa"
    randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
    plan = replace(
        plan,
        epoch_beacon=replace(plan.epoch_beacon, randomness=randomness),
    )
    monkeypatch.setattr(
        drand,
        "get_current_chain",
        lambda: {
            "name": plan.epoch_beacon.chain,
            "hash": plan.epoch_beacon.chain_hash,
        },
    )
    monkeypatch.setattr(
        drand,
        "get_beacon",
        lambda **_kwargs: {
            "source": "drand",
            "chain": plan.epoch_beacon.chain,
            "chain_hash": plan.epoch_beacon.chain_hash,
            "round": plan.epoch_beacon.round,
            "randomness": randomness,
            "signature": signature,
        },
    )
    monkeypatch.setattr(
        drand,
        "verify_beacon_signature",
        lambda *_args: time.sleep(0.1) or True,
    )
    monkeypatch.setattr(
        submitter_module,
        "_CHECKPOINT_EPOCH_BEACON_VERIFY_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(SubmissionError, match="public beacon"):
        await submitter_module._verify_checkpoint_epoch_public_beacon(plan)


@pytest.mark.asyncio
async def test_get_checkpoint_epoch_plan_rejects_etag_equivocation(monkeypatch):
    plan = _checkpoint_epoch_plan_fixture()

    async def _get(self, url, timeout=None):
        return httpx.Response(
            200,
            content=canonical_manifest_bytes(plan),
            headers={"ETag": f'"{"d" * 64}"'},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)
    client = httpx.AsyncClient()
    with pytest.raises(SubmissionError, match="ETag"):
        await get_checkpoint_epoch_plan_v1(
            "http://fake",
            expected_manifest_sha256=manifest_sha256(plan),
            client=client,
        )
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
