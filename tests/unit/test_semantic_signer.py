"""Security and integration contract for the isolated semantic signer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from reliquary.signer.backend import ChainResult
from reliquary.signer.journal import JournalReplay, JournalUncertain, SignerJournal
from reliquary.signer.protocol import (
    CheckpointSignRequest,
    SetWeightsRequest,
    axon_operation_id,
    checkpoint_operation_id,
    checkpoint_payload,
    weights_operation_id,
)
from reliquary.signer.service import SignerPolicy, create_signer_app


HOTKEY = "5FakeSignerHotkeyAddressLongEnoughForContract"
REPO = "aivolutionedge/reliquary-sn"
REVISION = "a" * 40


@dataclass
class FakeBackend:
    hotkey_address: str = HOTKEY
    checkpoint_calls: int = 0
    weight_calls: int = 0
    axon_calls: int = 0
    chain_error: Exception | None = None

    def sign_checkpoint(self, payload: bytes) -> bytes:
        self.checkpoint_calls += 1
        checkpoint, separator, revision = payload.partition(b"|")
        assert checkpoint.isdigit()
        assert separator == b"|"
        assert revision == REVISION.encode()
        return b"s" * 64

    async def set_weights(self, **_kwargs) -> ChainResult:
        self.weight_calls += 1
        if self.chain_error is not None:
            raise self.chain_error
        return ChainResult(accepted=True, message="included")

    async def serve_axon(self, **_kwargs) -> ChainResult:
        self.axon_calls += 1
        if self.chain_error is not None:
            raise self.chain_error
        return ChainResult(accepted=True, message="included")


@pytest.fixture
def signer(tmp_path):
    backend = FakeBackend()
    journal = SignerJournal(str(tmp_path / "journal" / "signer.sqlite3"))
    policy = SignerPolicy(
        network="finney",
        netuid=81,
        repo_id=REPO,
        axon_ip="203.0.113.81",
        axon_port=8443,
    )
    app = create_signer_app(policy=policy, backend=backend, journal=journal)
    with TestClient(app) as client:
        yield client, backend, journal
    journal.close()


def _checkpoint_request(checkpoint_n: int = 1) -> dict:
    operation = checkpoint_operation_id(
        netuid=81,
        checkpoint_n=checkpoint_n,
        repo_id=REPO,
        revision=REVISION,
    )
    return CheckpointSignRequest(
        operation_id=operation,
        netuid=81,
        checkpoint_n=checkpoint_n,
        repo_id=REPO,
        revision=REVISION,
    ).model_dump()


def _weight_request(epoch_id: int = 100) -> dict:
    uids = [1, 7]
    weights = [0.25, 0.75]
    operation = weights_operation_id(
        netuid=81,
        epoch_id=epoch_id,
        uids=uids,
        weights=weights,
    )
    return SetWeightsRequest(
        operation_id=operation,
        netuid=81,
        epoch_id=epoch_id,
        uids=uids,
        weights=weights,
    ).model_dump()


def test_checkpoint_signature_is_content_bound_and_idempotent(signer):
    client, backend, _journal = signer
    request = _checkpoint_request()

    first = client.post("/v1/checkpoints/sign", json=request)
    second = client.post("/v1/checkpoints/sign", json=request)

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert backend.checkpoint_calls == 1
    assert bytes.fromhex(first.json()["signature_hex"]) == b"s" * 64
    assert checkpoint_payload(1, REVISION) == f"1|{REVISION}".encode()


def test_checkpoint_rejects_tampering_and_wrong_policy(signer):
    client, backend, _journal = signer
    tampered = _checkpoint_request()
    tampered["revision"] = "b" * 40
    assert client.post("/v1/checkpoints/sign", json=tampered).status_code == 422

    wrong_netuid = _checkpoint_request()
    wrong_netuid["netuid"] = 82
    wrong_netuid["operation_id"] = checkpoint_operation_id(
        netuid=82, checkpoint_n=1, repo_id=REPO, revision=REVISION
    )
    assert client.post("/v1/checkpoints/sign", json=wrong_netuid).status_code == 403

    wrong_repo = _checkpoint_request()
    wrong_repo["repo_id"] = "attacker/repository"
    wrong_repo["operation_id"] = checkpoint_operation_id(
        netuid=81,
        checkpoint_n=1,
        repo_id=wrong_repo["repo_id"],
        revision=REVISION,
    )
    assert client.post("/v1/checkpoints/sign", json=wrong_repo).status_code == 403
    assert backend.checkpoint_calls == 0


def test_checkpoint_counter_is_monotonic(signer):
    client, backend, _journal = signer
    assert (
        client.post("/v1/checkpoints/sign", json=_checkpoint_request(2)).status_code
        == 200
    )
    response = client.post("/v1/checkpoints/sign", json=_checkpoint_request(1))
    assert response.status_code == 409
    assert response.json()["detail"] == "monotonic_replay_denied"
    assert backend.checkpoint_calls == 1


def test_weights_are_content_bound_monotonic_and_idempotent(signer):
    client, backend, _journal = signer
    request = _weight_request(100)
    assert client.post("/v1/weights/set", json=request).status_code == 200
    cached = client.post("/v1/weights/set", json=request)
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert backend.weight_calls == 1

    older = client.post("/v1/weights/set", json=_weight_request(99))
    assert older.status_code == 409
    assert older.json()["detail"] == "monotonic_replay_denied"

    tampered = _weight_request(101)
    tampered["weights"] = [0.5, 0.5]
    assert client.post("/v1/weights/set", json=tampered).status_code == 422
    assert backend.weight_calls == 1


def test_uncertain_chain_outcome_is_never_replayed(signer):
    client, backend, _journal = signer
    backend.chain_error = TimeoutError("inclusion response lost")
    request = _weight_request(100)
    first = client.post("/v1/weights/set", json=request)
    assert first.status_code == 503
    backend.chain_error = None
    second = client.post("/v1/weights/set", json=request)
    assert second.status_code == 409
    assert second.json()["detail"] == "operation_outcome_uncertain"
    assert backend.weight_calls == 1


def test_serve_axon_is_fixed_by_policy_and_idempotent(signer):
    client, backend, _journal = signer
    operation = axon_operation_id(netuid=81, ip="203.0.113.81", port=8443)
    request = {
        "protocol_version": 1,
        "operation_id": operation,
        "netuid": 81,
        "ip": "203.0.113.81",
        "port": 8443,
    }
    assert client.post("/v1/axon/serve", json=request).status_code == 200
    assert client.post("/v1/axon/serve", json=request).json()["cached"] is True
    assert backend.axon_calls == 1

    denied = dict(request)
    denied["port"] = 9443
    denied["operation_id"] = axon_operation_id(netuid=81, ip="203.0.113.81", port=9443)
    assert client.post("/v1/axon/serve", json=denied).status_code == 403
    assert backend.axon_calls == 1


def test_unknown_fields_and_arbitrary_signing_routes_are_closed(signer):
    client, _backend, _journal = signer
    request = _checkpoint_request()
    request["arbitrary_payload"] = "sign me"
    assert client.post("/v1/checkpoints/sign", json=request).status_code == 422
    assert client.post("/v1/sign", json={"payload": "00"}).status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_journal_replay_state_survives_process_restart(tmp_path):
    path = str(tmp_path / "journal.sqlite3")
    first = SignerJournal(path)
    first.reserve(
        operation_id="a" * 64,
        kind="set-weights",
        payload_digest="a" * 64,
        cursor_name="weight_epoch",
        cursor_value=100,
    )
    first.mark_uncertain("a" * 64, "TimeoutError")
    first.close()

    second = SignerJournal(path)
    with pytest.raises(JournalUncertain):
        second.reserve(
            operation_id="a" * 64,
            kind="set-weights",
            payload_digest="a" * 64,
        )
    with pytest.raises(JournalReplay):
        second.reserve(
            operation_id="b" * 64,
            kind="set-weights",
            payload_digest="b" * 64,
            cursor_name="weight_epoch",
            cursor_value=100,
        )
    second.close()


def test_bittensor_backend_loads_and_signs_with_hotkey_only(tmp_path):
    import bittensor as bt

    from reliquary.signer.backend import BittensorSignerBackend

    wallet = bt.Wallet(name="validator", hotkey="test", path=str(tmp_path))
    wallet.create_new_hotkey(
        n_words=12,
        use_password=False,
        overwrite=True,
        suppress=True,
    )
    assert not (tmp_path / "validator" / "coldkey").exists()
    assert not (tmp_path / "validator" / "coldkeypub.txt").exists()

    backend = BittensorSignerBackend(
        wallet_name="validator",
        hotkey_name="test",
        wallet_path=str(tmp_path),
        network="finney",
        expected_hotkey=wallet.hotkey.ss58_address,
    )
    payload = checkpoint_payload(1, REVISION)
    signature = backend.sign_checkpoint(payload)
    assert bt.Keypair(ss58_address=backend.hotkey_address).verify(
        data=payload,
        signature=signature,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        BittensorSignerBackend(
            wallet_name="validator",
            hotkey_name="test",
            wallet_path=str(tmp_path),
            network="finney",
            expected_hotkey="5WrongSignerHotkeyAddressLongEnoughForContract",
        )
