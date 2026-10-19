import json

import bittensor as bt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.protocol.gpu_attestation import gpu_request_binding
from reliquary.validator.corpus_registration import NOT_REGISTERED
from reliquary.validator.gpu_attestation import VerifierUnavailable
from reliquary.validator.gpu_registry import GpuRegistry
from reliquary.validator.rl_gpu_routes import HEADER, build_gpu_router
from reliquary.validator.rl_lease_routes import build_rl_lease_router
from .conftest import NOW

# Public development identities generated in memory; never construct a Wallet.
MINER = bt.Keypair.create_from_uri("//Alice")
RIVAL = bt.Keypair.create_from_uri("//Charlie")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
RUN = "gpu-admission-test"


@pytest.fixture
def served(tmp_path, authority, now):
    registry = GpuRegistry(tmp_path / "gpu.sqlite3", validator=VALIDATOR.ss58_address, run=RUN, mode="required",
                            verifier=authority.verifier, clock=lambda: now[0])
    async def registration(hotkey):
        return None if hotkey == MINER.ss58_address else NOT_REGISTERED
    app = FastAPI()
    app.include_router(build_gpu_router(registry, registration=registration))
    with TestClient(app) as client:
        yield registry, client
    registry.close()


def signed(path, *, key=MINER, validator=VALIDATOR.ss58_address, run=RUN, **fields):
    payload = {"miner_hotkey": key.ss58_address, "at": NOW, **fields}
    payload["signature"] = key.sign(gpu_request_binding(payload, validator_hotkey=validator, run=run, path=path)).hex()
    return payload


def post(client, path, **kwargs):
    return client.post(path, json=signed(path, **kwargs), headers={HEADER: MINER.ss58_address})


def attest(client, authority):
    c = post(client, "/rl/gpu/challenge").json()
    return post(client, "/rl/gpu/attest", challenge_id=c["challenge_id"], bundle=authority.bundle(c["nonce"]),
                declared={"gpu_name": "declared H100", "vram_bytes": 80 * 1024**3})


def test_full_authenticated_hardware_flow_and_private_status(served, authority):
    registry, client = served
    assert not post(client, "/rl/gpu/status").json()["attested"]
    accepted = attest(client, authority)
    assert accepted.status_code == 200 and accepted.json()["attested"]
    assert accepted.headers["cache-control"] == "no-store"
    assert post(client, "/rl/gpu/status").json()["hardware"]["driver"] == "580.105.08"
    assert client.get("/rl/gpu/fleet").json()["active_attested_devices"] == 1
    assert client.post("/rl/gpu/status", json={"miner_hotkey": MINER.ss58_address}).status_code == 401


@pytest.mark.parametrize("change", [{"validator": "other-validator"}, {"run": "other-run"}, {"key": RIVAL}])
def test_wrong_identity_audience_and_run_fail_before_state_mutation(served, change):
    registry, client = served
    response = post(client, "/rl/gpu/challenge", **change)
    assert response.status_code == 403
    assert registry._db.execute("SELECT count(*) FROM gpu_challenges").fetchone()[0] == 0


def test_signatures_cover_exact_route_evidence_and_declarations(served, authority):
    registry, client = served
    payload = signed("/rl/gpu/status")
    assert client.post("/rl/gpu/challenge", json=payload, headers={HEADER: MINER.ss58_address}).status_code == 403
    c = post(client, "/rl/gpu/challenge").json()
    payload = signed("/rl/gpu/attest", challenge_id=c["challenge_id"], bundle=authority.bundle(c["nonce"]),
                     declared={"gpu_name": "H100"})
    payload["declared"]["gpu_name"] = "H200"
    assert client.post("/rl/gpu/attest", json=payload, headers={HEADER: MINER.ss58_address}).status_code == 403
    payload = signed("/rl/gpu/attest", challenge_id=c["challenge_id"], bundle=authority.bundle(c["nonce"]))
    payload["bundle"][1]["GPU-0"] += "tampered"
    assert client.post("/rl/gpu/attest", json=payload, headers={HEADER: MINER.ss58_address}).status_code == 403
    assert not registry.status(MINER.ss58_address)["attested"]


def test_unregistered_miners_are_rejected_before_body_decoding(served):
    _, client = served
    response = client.post("/rl/gpu/attest", content=b"not-json", headers={HEADER: RIVAL.ss58_address})
    assert response.status_code == 403 and response.json()["reason"] == "hotkey_not_registered"


@pytest.mark.parametrize("declared", [{"vram_bytes": -1}, {"vram_bytes": True}, {"utilization_percent": 101},
                                      {"is_verified": True}, {"gpu_name": "x" * 129}])
def test_invalid_or_claim_escalating_declarations_are_rejected(served, authority, declared):
    _, client = served
    c = post(client, "/rl/gpu/challenge").json()
    response = post(client, "/rl/gpu/attest", challenge_id=c["challenge_id"], bundle=authority.bundle(c["nonce"]),
                    declared=declared)
    assert response.status_code == 422


def test_body_limit_strict_json_and_expired_requests(served):
    _, client = served
    for body, expected in ((b"x" * (160 * 1024 + 1), 413), (b'{"a":1,"a":2}', 422)):
        assert client.post("/rl/gpu/attest", content=body, headers={HEADER: MINER.ss58_address}).status_code == expected
    assert post(client, "/rl/gpu/challenge", at=NOW - 121).status_code == 400


def test_nvidia_unavailable_never_grants_admission_or_penalizes_miner(served, authority):
    registry, client = served
    class Down:
        def verify(self, *args, **kwargs):
            raise VerifierUnavailable()
    registry.verifier = Down()
    response = attest(client, authority)
    assert response.status_code == 503
    assert not registry.status(MINER.ss58_address)["attested"]


@pytest.mark.parametrize("mode", ["observe", "required"])
def test_native_rl_admission_leases_and_submissions_survive_attestation_expiry(tmp_path, authority, now, mode):
    from tests.unit.test_rl_lease_routes import body, submit_body
    from tests.unit.test_rl_submit import ENV, MathEnv, Tokenizer, group
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions

    registry = GpuRegistry(tmp_path / "hardware.db", validator=VALIDATOR.ss58_address, run=RUN, mode=mode,
                            verifier=authority.verifier, clock=lambda: now[0])
    book = LeaseBook(tmp_path / "leases.db", LeaseBookConfig(expiry_s={ENV: 600}, prompt_range={ENV: (0, 50)},
                                                               cooldown_of=lambda e: 2),
                     quota=QuotaController(QuotaConfig(initial=4)), clock=lambda: now[0])
    book.adopt_checkpoint(0)
    subs = RlSubmissions(book, GroupGrader({ENV: MathEnv()}, Tokenizer(), eos_token_ids=(1,)), clock=lambda: now[0])
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0),
                                             validator_hotkey=VALIDATOR.ss58_address, submissions=subs,
                                             clock=lambda: now[0], gpu_registry=registry))
    app.include_router(build_gpu_router(registry))
    try:
        with TestClient(app, headers={HEADER: MINER.ss58_address}) as client:
            first = client.post("/rl/lease", json=body(prompts=(1,), env=ENV))
            assert first.status_code == (403 if mode == "required" else 200)
            assert book.in_use(MINER.ss58_address) == (0 if mode == "required" else 1)
            assert attest(client, authority).status_code == 200
            lease = client.post("/rl/lease", json=body(prompts=(2,), env=ENV)).json()["leases"][0]
            lease_id = lease["lease_id"]
            saved = registry.lease_hardware(lease_id)
            assert saved["attested"] and not saved["execution_attested"]
            quota = client.get("/rl/quota", params={"hotkey": MINER.ss58_address}).json()
            assert quota["gpu_admission"]["attested"] and "device_id" not in quota["gpu_admission"]
            now[0] += 301
            response = client.post("/rl/submit", json=submit_body(MINER, lease_id, group(8), at=int(now[0])))
            assert response.status_code == 200 and response.json()["verdict"] == "in_zone"
            assert book.lease(lease_id).state == "submitted"
            again = client.post("/rl/lease", json=body(prompts=(3,), env=ENV, at=int(now[0])))
            assert again.status_code == (403 if mode == "required" else 200)
            assert registry.lease_hardware(lease_id) == saved
    finally:
        book.close()
        registry.close()
