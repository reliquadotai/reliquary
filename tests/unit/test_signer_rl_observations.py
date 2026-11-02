"""The signer's async RL scan log operation (``POST /v1/rl/observations/sign``): a validated head or closed page of
``rl/<run>/observations/`` under the observation index domain, never arbitrary bytes; the run fixed by policy, the
document naming the signer's hotkey, heads and pages never going back; the client and the scan log publisher over it
end to end with a real keypair."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

bt = pytest.importorskip("bittensor")

from reliquary.services import publication as pub  # noqa: E402
from reliquary.shared.ckpt_index import observations_root  # noqa: E402
from reliquary.signer.client import RemoteSignerClient  # noqa: E402
from reliquary.signer.journal import SignerJournal  # noqa: E402
from reliquary.signer.protocol import (  # noqa: E402
    RlIndexSignRequest, RlObservationSignRequest, rl_index_operation_id, rl_observation_operation_id,
)
from reliquary.signer.service import SignerPolicy, create_signer_app  # noqa: E402

RUN = "async-run-1"
ROOT = observations_root(RUN)
SHA = "cd" * 32
NOW = 1_760_000_000.0


def _segments(first: int, count: int) -> list[dict]:
    return [{"number": n, "first_seq": 10 * n, "last_seq": 10 * n + 9, "sha256": SHA, "size": 100 + n}
            for n in range(first, first + count)]


def _page(validator: str, k: int = 0, run: str = RUN) -> dict:
    return pub.build_page(run_id=run, validator=validator, root=observations_root(run),
                          segments=_segments(k * pub.PAGE_SEGMENTS + 1, pub.PAGE_SEGMENTS))


def _head(validator: str, last: int = 3, run: str = RUN, generated_at: float = NOW - 5, pages: int = 0) -> dict:
    first = pages * pub.PAGE_SEGMENTS + 1
    refs = [{"key": pub.page_key(run, k * pub.PAGE_SEGMENTS + 1, (k + 1) * pub.PAGE_SEGMENTS,
                                 root=observations_root(run)),
             "first_number": k * pub.PAGE_SEGMENTS + 1, "last_number": (k + 1) * pub.PAGE_SEGMENTS,
             "sha256": SHA, "size": 999} for k in range(pages)]
    return pub.build_head(run_id=run, order_sha256=SHA, validator=validator, segments=_segments(first, last - first + 1),
                          pages=refs, last_number=last, generated_at=generated_at, root=observations_root(run))


def _request(document: dict, netuid: int = 81) -> dict:
    return RlObservationSignRequest(operation_id=rl_observation_operation_id(netuid=netuid, document=document),
                                    netuid=netuid, document=document).model_dump()


def _raw(document: dict, netuid: int = 81) -> dict:
    """The request as a client could send it, unvalidated (the operation id still binds the document)."""
    return {"protocol_version": 1, "operation_id": rl_observation_operation_id(netuid=netuid, document=document),
            "netuid": netuid, "document": document}


@dataclass
class KeypairBackend:
    keypair: object
    payloads: list = field(default_factory=list)

    @property
    def hotkey_address(self) -> str:
        return self.keypair.ss58_address

    def sign_rl_observation(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        return bytes(self.keypair.sign(payload))

    def sign_rl_index(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        return bytes(self.keypair.sign(payload))


@pytest.fixture
def keypair():
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _app(tmp_path, keypair, *, rl_run: str | None = RUN):
    backend = KeypairBackend(keypair)
    journal = SignerJournal(str(tmp_path / "journal" / "signer.sqlite3"))
    app = create_signer_app(policy=SignerPolicy(network="finney", netuid=81, repo_id="o/r", rl_run=rl_run),
                            backend=backend, journal=journal, clock=lambda: NOW)
    return app, backend, journal


@pytest.fixture
def signer(tmp_path, keypair):
    app, backend, journal = _app(tmp_path, keypair)
    with TestClient(app) as client:
        yield client, backend
    journal.close()


def _signed(document: dict, signature_hex: str) -> bytes:
    return pub.canonical_json_bytes({**document, "signature": signature_hex})


def test_heads_and_pages_are_signed_under_the_observation_domain_and_verify(signer, keypair):
    client, backend = signer
    head, page = _head(keypair.ss58_address), _page(keypair.ss58_address)
    first = client.post("/v1/rl/observations/sign", json=_request(head))
    again = client.post("/v1/rl/observations/sign", json=_request(head))
    assert first.status_code == 200 and first.json()["cached"] is False
    assert again.status_code == 200 and again.json()["cached"] is True
    signed_page = client.post("/v1/rl/observations/sign", json=_request(page))
    assert signed_page.status_code == 200
    assert backend.payloads == [pub.document_signing_payload(head), pub.document_signing_payload(page)]
    assert all(p.startswith(pub.INDEX_DOMAIN) for p in backend.payloads)
    read = pub.verify_index(_signed(head, first.json()["signature_hex"]), keypair.ss58_address,
                            expected_run_id=RUN, root=ROOT)
    assert read["kind"] == "head" and read["last_number"] == 3
    read = pub.verify_index(_signed(page, signed_page.json()["signature_hex"]), keypair.ss58_address,
                            expected_run_id=RUN, root=ROOT)
    assert read["kind"] == "page"


def _set_key(document, key):
    document["segments"][0]["key"] = key


@pytest.mark.parametrize("tamper", [
    lambda d: d.update(extra="sign me"),
    lambda d: d.update(signature="00"),                                        # never re-sign a signed document
    lambda d: d["segments"][0].update(note="x"),
    lambda d: d.update(schema="service-observation-index/v2"),
    lambda d: d.update(schema=pub.PAGE_SCHEMA),                                # a head is not a page
    lambda d: _set_key(d, pub.segment_key(RUN, 1)),                            # the window log's root
    lambda d: _set_key(d, f"rl/other/observations/seg-000001.jsonl.gz"),
    lambda d: _set_key(d, f"{ROOT}/../../x/seg-000001.jsonl.gz"),
    lambda d: d["segments"][1].update(number=5),                               # not contiguous
    lambda d: d["segments"][1].update(first_seq=12),
    lambda d: d.update(last_number=4),                                         # does not match its segments
    lambda d: d.update(last_number=True),
    lambda d: d["segments"][0].update(size=True),
    lambda d: d["segments"][0].update(sha256="ZZ" * 32),
    lambda d: d.update(order_sha256="nope"),
    lambda d: d.update(generated_at=float("nan")),
    lambda d: d.update(run_id="a/b"),
    lambda d: d.update(segments=[]),                                           # last_number 3 without segments
    lambda d: d.update(pages=[{"key": f"{ROOT}/index-000001-000200.json", "first_number": 1, "last_number": 200,
                               "sha256": SHA, "size": 1}]),                    # segments do not continue the page
])
def test_anything_but_a_well_formed_head_of_the_run_is_refused(signer, keypair, tamper):
    client, backend = signer
    head = _head(keypair.ss58_address)
    tamper(head)
    try:
        body = _raw(head)
    except ValueError:                       # not canonical JSON (nan): no operation id binds it, no request holds it
        with pytest.raises(ValueError):
            RlObservationSignRequest.model_validate({"protocol_version": 1, "operation_id": "0" * 64,
                                                     "netuid": 81, "document": head})
        return
    assert client.post("/v1/rl/observations/sign", json=body).status_code == 422
    assert backend.payloads == []


@pytest.mark.parametrize("tamper", [
    lambda d: d.update(segments=d["segments"][:-1], last_number=pub.PAGE_SEGMENTS - 1),   # a page is whole
    lambda d: d.update(first_number=2),
    lambda d: _set_key(d, pub.segment_key(RUN, 1)),
    lambda d: d.update(generated_at=NOW),
])
def test_anything_but_a_whole_page_of_the_run_is_refused(signer, keypair, tamper):
    client, backend = signer
    page = _page(keypair.ss58_address)
    tamper(page)
    assert client.post("/v1/rl/observations/sign", json=_raw(page)).status_code == 422
    assert backend.payloads == []


def test_the_operation_id_binds_the_document_and_the_index_operation_does_not_take_it(signer, keypair):
    client, _ = signer
    body = _request(_head(keypair.ss58_address))
    body["document"] = _head(keypair.ss58_address, last=4)
    assert client.post("/v1/rl/observations/sign", json=body).status_code == 422
    head = _head(keypair.ss58_address)                    # another operation's route refuses an observation head
    smuggled = {"protocol_version": 1, "netuid": 81, "index": head,
                "operation_id": rl_index_operation_id(netuid=81, index=head)}
    assert client.post("/v1/rl/index/sign", json=smuggled).status_code == 422
    with pytest.raises(ValueError):
        RlIndexSignRequest.model_validate(smuggled)


def test_run_validator_netuid_and_time_are_fixed_by_policy(signer, tmp_path, keypair):
    client, backend = signer
    other = client.post("/v1/rl/observations/sign", json=_request(_head(keypair.ss58_address, run="other")))
    assert (other.status_code, other.json()["detail"]) == (403, "rl_run_denied")
    named = client.post("/v1/rl/observations/sign", json=_request(_head("5" + "F" * 47)))
    assert (named.status_code, named.json()["detail"]) == (403, "validator_denied")
    ahead = client.post("/v1/rl/observations/sign",
                        json=_request(_head(keypair.ss58_address, generated_at=NOW + 3600)))
    assert (ahead.status_code, ahead.json()["detail"]) == (403, "head_from_the_future")
    assert client.post("/v1/rl/observations/sign", json=_request(_head(keypair.ss58_address), netuid=82)
                       ).status_code == 403
    assert backend.payloads == []
    app, _, journal = _app(tmp_path / "off", keypair, rl_run=None)
    with TestClient(app) as off:
        refused = off.post("/v1/rl/observations/sign", json=_request(_head(keypair.ss58_address)))
    journal.close()
    assert (refused.status_code, refused.json()["detail"]) == (403, "rl_observation_operation_disabled")


def test_heads_and_pages_may_repeat_but_never_go_back(signer, keypair):
    client, _ = signer
    me = keypair.ss58_address
    assert client.post("/v1/rl/observations/sign", json=_request(_head(me, last=5))).status_code == 200
    rewritten = _head(me, last=5, generated_at=NOW - 1)              # a restart rewrites the same head
    assert client.post("/v1/rl/observations/sign", json=_request(rewritten)).status_code == 200
    back = client.post("/v1/rl/observations/sign", json=_request(_head(me, last=4)))
    assert (back.status_code, back.json()["detail"]) == (409, "monotonic_replay_denied")
    assert client.post("/v1/rl/observations/sign", json=_request(_page(me, k=1))).status_code == 200
    assert client.post("/v1/rl/observations/sign", json=_request(_page(me, k=0))).status_code == 409
    # The page cursor is its own: a head after page 1 (last 401) still signs.
    assert client.post("/v1/rl/observations/sign", json=_request(_head(me, last=401, pages=2))).status_code == 200


def test_a_signature_lost_to_a_restart_can_be_retried(tmp_path, keypair):
    path = str(tmp_path / "journal.sqlite3")
    head = _head(keypair.ss58_address)
    journal = SignerJournal(path)
    operation = rl_observation_operation_id(netuid=81, document=head)
    journal.reserve(operation_id=operation, kind="rl-observation-sign", payload_digest=operation,
                    cursor_name="rl_observation_head", cursor_value=3, cursor_strict=False)
    journal.close()
    journal = SignerJournal(path)
    app = create_signer_app(policy=SignerPolicy(network="finney", netuid=81, repo_id="o/r", rl_run=RUN),
                            backend=KeypairBackend(keypair), journal=journal, clock=lambda: NOW)
    with TestClient(app) as client:
        assert client.post("/v1/rl/observations/sign", json=_request(head)).status_code == 200
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


def test_the_client_refuses_a_signature_from_another_key(monkeypatch, tmp_path, keypair):
    other = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    app, _, journal = _app(tmp_path, other)
    client = _remote_client(monkeypatch, app, keypair.ss58_address)
    with pytest.raises(RuntimeError, match="unexpected hotkey"):
        client.sign_rl_observation(copy.deepcopy(_head(other.ss58_address)))
    journal.close()


async def test_the_scan_log_publisher_signs_heads_and_pages_through_the_signer(monkeypatch, tmp_path, keypair):
    """End to end: the lease book's scans, published through ``scan_publisher(signer_client=)`` with the signer's
    public identity only, read back and verified by the miner's reader; past a page boundary."""
    from reliquary.protocol.rl_scan_log import read_scan_log
    from reliquary.signer.client import PublicWalletIdentity
    from reliquary.validator.rl_lease_quota import QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig
    from reliquary.validator.rl_observations import ObservationBoard, scan_publisher

    app, backend, journal = _app(tmp_path, keypair)
    client = _remote_client(monkeypatch, app, keypair.ss58_address)
    book = LeaseBook(tmp_path / "rl.sqlite3", LeaseBookConfig(expiry_s={"math": 600.0}, cooldown_of=lambda env: 2),
                     quota=QuotaController(), clock=lambda: 1000.0)
    objects: dict[str, bytes] = {}

    async def put(key, body, content_type, cache_control):
        objects[key] = body

    async def get(key):
        return objects.get(key)

    wallet = PublicWalletIdentity.from_address(keypair.ss58_address)    # no private key on the controller
    publisher = scan_publisher(book, run=RUN, task_id="rl-async", wallet=wallet, put=put, get=get,
                               clock=lambda: NOW, signer_client=client)
    segments = pub.PAGE_SEGMENTS + 1
    for prompt in range(segments):                                       # one scan, one segment per flush
        book.record_outcome("math", prompt, 0, [1.0] * 8 + [0.0] * 8, "in_zone", now=1500.0 + prompt)
        while await publisher.flush() is not None:
            pass
    assert f"{ROOT}/index-000001-000200.json" in objects
    head = pub.verify_index(objects[f"{ROOT}/index.json"], keypair.ss58_address, expected_run_id=RUN, root=ROOT)
    assert head["last_number"] == segments and len(head["pages"]) == 1
    board = ObservationBoard(publisher, presign=lambda key: f"mem://{key}")

    async def ask(after):
        return board.answer(after)

    async def fetch(url):
        return objects[url.removeprefix("mem://")]
    events, last = await read_scan_log(ask, fetch, keypair.ss58_address)
    assert last == segments and len(events) == segments
    kinds = [p[: len(pub.INDEX_DOMAIN)] for p in backend.payloads]
    assert set(kinds) == {pub.INDEX_DOMAIN} and len(backend.payloads) >= segments + 1   # every head + the page
    await publisher.close()
    book.close()
    journal.close()


async def test_the_async_validator_signs_both_public_boards_through_the_signer(monkeypatch, tmp_path, keypair):
    """``_public_boards`` with a signer client: the checkpoint index and the scan log are both signed by signer-01
    (the wallet is the signer's public identity, holding no key)."""
    import json as _json

    import reliquary.infrastructure.storage as storage
    from reliquary.shared.ckpt_index import index_key, verify_index_signature
    from reliquary.signer.client import PublicWalletIdentity
    from reliquary.validator.rl_async_service import _public_boards
    from reliquary.validator.rl_lease_quota import QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig
    from tests.unit.test_signer_rl_index import _index

    app, backend, journal = _app(tmp_path, keypair)
    client = _remote_client(monkeypatch, app, keypair.ss58_address)
    objects = {index_key(RUN): _json.dumps(_index(run=RUN)).encode()}

    class Bucket:
        exceptions = SimpleNamespace(NoSuchKey=KeyError)

        def get_object(self, Bucket, Key):
            if Key not in objects:
                raise KeyError(Key)
            return {"Body": SimpleNamespace(read=lambda: objects[Key])}

        def put_object(self, Bucket, Key, Body, **_kw):
            objects[Key] = Body

        def generate_presigned_url(self, op, Params, ExpiresIn):
            return f"mem://{Params['Key']}"

    monkeypatch.setattr(storage, "public_checkpoint_client", lambda: Bucket())
    monkeypatch.setattr(storage, "public_checkpoint_bucket", lambda: "public")
    book = LeaseBook(tmp_path / "rl.sqlite3", LeaseBookConfig(expiry_s={"math": 600.0}, cooldown_of=lambda env: 2),
                     quota=QuotaController(), clock=lambda: 1000.0)
    rl = SimpleNamespace(book=book, settings=SimpleNamespace(run=RUN, task_id="rl-async"), board=None,
                         observations=None)
    wallet = PublicWalletIdentity.from_address(keypair.ss58_address)
    publisher = _public_boards(rl, wallet, keypair.ss58_address, client)
    publisher.clock = lambda: NOW
    answer = await rl.board.answer()
    assert verify_index_signature(answer["index"], answer["signature"], keypair.ss58_address)
    book.record_outcome("math", 1, 0, [1.0] * 8 + [0.0] * 8, "in_zone", now=1500.0)
    assert await publisher.flush() is not None
    assert rl.observations.answer()["index"]["last_number"] == 1
    assert len(backend.payloads) == 2 and backend.payloads[1].startswith(pub.INDEX_DOMAIN)
    await publisher.close()
    book.close()
    journal.close()
