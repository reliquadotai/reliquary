"""The signer's async RL checkpoint index operation: a validated index under its own domain, never arbitrary
bytes; the run fixed by policy; the latest version never going back; the client and the checkpoint board over it."""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from reliquary.shared.ckpt_index import (
    SIGNING_DOMAIN, anchor_key, delta_key, index_key, index_signing_payload, verify_index_signature,
)
from reliquary.signer.client import RemoteSignerClient
from reliquary.signer.journal import SignerJournal
from reliquary.signer.protocol import RlIndexSignRequest, rl_index_operation_id
from reliquary.signer.service import SignerPolicy, create_signer_app

RUN = "async-run-1"
SHA = "ab" * 32


def _index(version: int = 2, run: str = RUN, **overrides) -> dict:
    index = {
        "format": "reliquary-ckpt-index/1", "run": run,
        "latest": {"version": version, "sha256": SHA},
        "anchors": [{"version": 0, "sha256": SHA,
                     "files": [{"key": anchor_key(run, 0, "shard-0.safetensors"), "size": 10, "sha256": SHA}],
                     "model_files": [{"name": "config.json", "key": f"rl/{run}/anchors/00000000/model/config.json",
                                      "size": 3, "sha256": SHA}]}],
        "deltas": [{"from": v, "to": v + 1, "key": delta_key(run, v, v + 1), "size": 5, "sha256": SHA,
                    "target_sha256": SHA} for v in range(version)],
        "published_at": 1760000000.125,
    }
    index.update(overrides)
    return index


def _request(index: dict, netuid: int = 81) -> dict:
    return RlIndexSignRequest(operation_id=rl_index_operation_id(netuid=netuid, index=index), netuid=netuid,
                              index=index).model_dump()


@dataclass
class KeypairBackend:
    keypair: object
    payloads: list = field(default_factory=list)

    @property
    def hotkey_address(self) -> str:
        return self.keypair.ss58_address

    def sign_rl_index(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        return bytes(self.keypair.sign(payload))


@pytest.fixture
def keypair():
    import bittensor as bt

    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _app(tmp_path, keypair, *, rl_run: str | None = RUN):
    backend = KeypairBackend(keypair)
    journal = SignerJournal(str(tmp_path / "journal" / "signer.sqlite3"))
    app = create_signer_app(policy=SignerPolicy(network="finney", netuid=81, repo_id="o/r", rl_run=rl_run),
                            backend=backend, journal=journal)
    return app, backend, journal


@pytest.fixture
def signer(tmp_path, keypair):
    app, backend, journal = _app(tmp_path, keypair)
    with TestClient(app) as client:
        yield client, backend
    journal.close()


def test_the_index_is_signed_under_its_domain_bound_and_idempotent(signer, keypair):
    client, backend = signer
    index = _index()
    first = client.post("/v1/rl/index/sign", json=_request(index))
    again = client.post("/v1/rl/index/sign", json=_request(index))
    assert first.status_code == 200 and first.json()["cached"] is False
    assert again.status_code == 200 and again.json()["cached"] is True
    assert backend.payloads == [index_signing_payload(index)]
    assert backend.payloads[0].startswith(SIGNING_DOMAIN)
    assert verify_index_signature(index, first.json()["signature_hex"], keypair.ss58_address)


@pytest.mark.parametrize("tamper", [
    lambda i: i.update(extra="sign me"),
    lambda i: i["anchors"][0].update(note="x"),
    lambda i: i.update(format="reliquary-ckpt-index/2"),
    lambda i: i["deltas"][0].update(key="rl/other-run/deltas/x.rqd"),
    lambda i: i["deltas"][0].update(key=f"rl/{RUN}/../other/x.rqd"),
    lambda i: i["deltas"][0].update(**{"from": 1, "to": 1}),
    lambda i: i["deltas"][0].update(to=9),
    lambda i: i["latest"].update(sha256="ZZ" * 32),
    lambda i: i["deltas"][0].update(size=True),
    lambda i: i.update(run="a/b"),
])
def test_anything_but_a_well_formed_index_of_the_run_is_refused(signer, tamper):
    client, backend = signer
    index = _index()
    tamper(index)
    body = {"protocol_version": 1, "operation_id": rl_index_operation_id(netuid=81, index=index), "netuid": 81,
            "index": index}
    assert client.post("/v1/rl/index/sign", json=body).status_code == 422
    assert backend.payloads == []


def test_the_operation_id_binds_the_index(signer):
    client, _ = signer
    body = _request(_index())
    body["index"] = _index(version=3)
    assert client.post("/v1/rl/index/sign", json=body).status_code == 422


def test_the_run_and_netuid_are_fixed_by_policy(signer, tmp_path, keypair):
    client, _ = signer
    assert client.post("/v1/rl/index/sign", json=_request(_index(run="other"))).json()["detail"] == "rl_run_denied"
    assert client.post("/v1/rl/index/sign", json=_request(_index(), netuid=82)).status_code == 403
    app, _, journal = _app(tmp_path / "off", keypair, rl_run=None)
    with TestClient(app) as off:
        refused = off.post("/v1/rl/index/sign", json=_request(_index()))
    journal.close()
    assert (refused.status_code, refused.json()["detail"]) == (403, "rl_index_operation_disabled")


def test_the_latest_version_may_repeat_but_never_goes_back(signer):
    client, _ = signer
    assert client.post("/v1/rl/index/sign", json=_request(_index(version=3))).status_code == 200
    pruned = _index(version=3, published_at=1760000001.5)
    assert client.post("/v1/rl/index/sign", json=_request(pruned)).status_code == 200
    back = client.post("/v1/rl/index/sign", json=_request(_index(version=2)))
    assert (back.status_code, back.json()["detail"]) == (409, "monotonic_replay_denied")


def test_a_signature_lost_to_a_restart_can_be_retried(tmp_path, keypair):
    path = str(tmp_path / "journal.sqlite3")
    index = _index()
    journal = SignerJournal(path)
    operation = rl_index_operation_id(netuid=81, index=index)
    journal.reserve(operation_id=operation, kind="rl-index-sign", payload_digest=operation,
                    cursor_name="rl_index_version", cursor_value=2, cursor_strict=False)
    journal.close()
    journal = SignerJournal(path)
    app = create_signer_app(policy=SignerPolicy(network="finney", netuid=81, repo_id="o/r", rl_run=RUN),
                            backend=KeypairBackend(keypair), journal=journal)
    with TestClient(app) as client:
        assert client.post("/v1/rl/index/sign", json=_request(index)).status_code == 200
    journal.close()


class _Routed:
    """``httpx.Client`` stand-in that sends the client's requests to the signer app."""

    def __init__(self, app):
        self.app = app

    def __call__(self, **_kwargs):
        return self

    def __enter__(self):
        self.client = TestClient(self.app).__enter__()
        return self

    def __exit__(self, *exc):
        self.client.__exit__(*exc)

    def post(self, url, json):
        return self.client.post(url.removeprefix("https://signer.test"), json=json)


def _remote_client(monkeypatch, app, hotkey: str) -> RemoteSignerClient:
    import reliquary.signer.client as client_module

    monkeypatch.setattr(client_module.httpx, "Client", _Routed(app))
    client = RemoteSignerClient.__new__(RemoteSignerClient)
    client.base_url, client._tls_context, client.netuid = "https://signer.test", None, 81
    client.expected_hotkey, client.repo_id = hotkey, "o/r"
    return client


def test_the_client_and_the_checkpoint_board_sign_through_the_signer(monkeypatch, tmp_path, keypair):
    from reliquary.validator.rl_async_service import index_signer_for
    from reliquary.validator.rl_checkpoint import RlCheckpointBoard

    app, backend, journal = _app(tmp_path, keypair)
    client = _remote_client(monkeypatch, app, keypair.ss58_address)
    index = _index()
    assert verify_index_signature(index, client.sign_rl_index(index).hex(), keypair.ss58_address)

    published = {index_key(RUN): json.dumps(_index(version=3)).encode()}
    board = RlCheckpointBoard(load_index=lambda: published[index_key(RUN)],
                              index_signer=index_signer_for(None, client),
                              validator_hotkey=keypair.ss58_address, presign=lambda key: f"https://r2/{key}")
    answer = asyncio.run(board.answer())
    assert answer["index"]["latest"]["version"] == 3
    assert verify_index_signature(answer["index"], answer["signature"], keypair.ss58_address)
    assert len(backend.payloads) == 2
    journal.close()


def test_the_client_refuses_a_signature_from_another_key(monkeypatch, tmp_path, keypair):
    import bittensor as bt

    other = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    app, _, journal = _app(tmp_path, other)
    client = _remote_client(monkeypatch, app, keypair.ss58_address)
    with pytest.raises(RuntimeError, match="unexpected hotkey"):
        client.sign_rl_index(copy.deepcopy(_index()))
    journal.close()
