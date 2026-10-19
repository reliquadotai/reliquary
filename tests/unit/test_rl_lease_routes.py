"""The lease routes: ``POST /rl/lease`` (signed by the miner's hotkey for this validator and route, like the other RL
requests), ``GET /rl/quota?hotkey=`` (the explainable quota), ``GET /rl/lease/commitments`` (S_k commitments and
revealed secrets) and ``GET /rl/prompts/state`` (the prompt lifecycle, incremental)."""
from __future__ import annotations

import random

import pytest

bt = pytest.importorskip("bittensor")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reliquary.protocol.signatures import build_rl_lease_binding  # noqa: E402
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController  # noqa: E402
from reliquary.validator.rl_lease_routes import (  # noqa: E402
    LEASE_MAX_PER_MINUTE, build_rl_lease_router, rl_lease_path,
)
from reliquary.validator.rl_leases import (  # noqa: E402
    LeaseBook, LeaseBookConfig, LeaseCollector, lease_nonce, verify_reveal,
)

MINER = bt.Keypair.create_from_uri("//Alice")
RIVAL = bt.Keypair.create_from_uri("//Charlie")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
EVE = bt.Keypair.create_from_uri("//Eve")
NOW = 1_800_000_000
ENV = "math"


@pytest.fixture
def served(tmp_path):
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}, max_lag=1,
                                                                  prompt_range={ENV: (0, 1000)}),
                     quota=QuotaController(QuotaConfig(initial=4)), clock=lambda: float(NOW),
                     rng=random.Random(1))
    book.adopt_checkpoint(0)
    return book


def client_for(book, *, registration=None, clock=lambda: float(NOW), admit=None):
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=VALIDATOR.ss58_address,
                                             registration=registration, clock=clock, admit=admit))
    return TestClient(app)


def body(keypair=MINER, *, prompts=(1, 2), k=0, env=ENV, at=NOW, hotkey=None, validator=VALIDATOR,
         path="/rl/lease"):
    hotkey = hotkey or keypair.ss58_address
    binding = build_rl_lease_binding(hotkey=hotkey, at=at, environment=env, checkpoint=k, prompt_ids=list(prompts),
                                     validator_hotkey=validator.ss58_address, path=path)
    return {"miner_hotkey": hotkey, "at": at, "environment": env, "checkpoint": k, "prompt_ids": list(prompts),
            "signature": keypair.sign(binding).hex()}


def test_a_signed_request_gets_its_leases_with_their_nonces(served):
    client = client_for(served)
    assert rl_lease_path() == "/rl/lease"
    answer = client.post("/rl/lease", json=body())
    assert answer.status_code == 200, answer.json()
    content = answer.json()
    assert [x["prompt_idx"] for x in content["leases"]] == [1, 2] and content["refused"] == []
    assert content["quota"] == 4 and content["in_use"] == 2 and content["latest"] == 0
    first = content["leases"][0]
    assert first["checkpoint"] == 0 and first["expires_at"] == NOW + 600
    assert answer.headers["cache-control"] == "no-store"
    served.adopt_checkpoint(1)
    served.adopt_checkpoint(2)                                        # 0 retires: its secret is revealed
    revealed = client.get("/rl/lease/commitments").json()["revealed"][0]
    secret = bytes.fromhex(revealed["secret"])
    assert verify_reveal(bytes.fromhex(revealed["commitment"]), 0, secret)
    assert lease_nonce(secret, ENV, 1, 0).hex() == first["nonce"]


def test_a_task_the_fleet_cannot_host_is_refused_beside_the_books_leases(served):
    async def admit(environment, prompt):
        return {2: ("task_unservable", {}), 3: ("fleet_unavailable", {})}.get(prompt)

    client = client_for(served, admit=admit)
    content = client.post("/rl/lease", json=body(prompts=(1, 2, 3))).json()
    assert [x["prompt_idx"] for x in content["leases"]] == [1] and content["in_use"] == 1
    assert content["refused"] == [{"prompt_idx": 2, "reason": "task_unservable"},
                                  {"prompt_idx": 3, "reason": "fleet_unavailable"}]
    nothing = client.post("/rl/lease", json=body(prompts=(2,), at=NOW + 1)).json()
    assert nothing["leases"] == [] and nothing["refused"] == [{"prompt_idx": 2, "reason": "task_unservable"}]


def test_a_rival_learns_only_that_the_prompt_is_taken(served):
    client = client_for(served)
    assert client.post("/rl/lease", json=body()).status_code == 200
    answer = client.post("/rl/lease", json=body(RIVAL, prompts=(2, 3)))
    content = answer.json()
    assert answer.status_code == 200
    assert content["refused"] == [{"prompt_idx": 2, "reason": "taken"}]
    assert [x["prompt_idx"] for x in content["leases"]] == [3]
    assert MINER.ss58_address not in answer.text


def test_requests_that_are_not_the_hotkeys_own_are_refused(served):
    client = client_for(served)
    unsigned = client.post("/rl/lease", json={**body(), "signature": ""})
    assert unsigned.status_code == 401 and unsigned.json()["reason"] == "signature_required"
    forged = client.post("/rl/lease", json=body(EVE, hotkey=MINER.ss58_address))
    assert forged.status_code == 403 and forged.json()["reason"] == "bad_signature"
    elsewhere = client.post("/rl/lease", json=body(at=NOW + 1, validator=EVE))
    assert elsewhere.status_code == 403
    tampered = {**body(at=NOW + 2), "prompt_ids": [1, 2, 3]}
    assert client.post("/rl/lease", json=tampered).status_code == 403
    other_route = client.post("/rl/lease", json=body(at=NOW + 3, path="/rl/sessions/allocation"))
    assert other_route.status_code == 403
    stale = client.post("/rl/lease", json=body(at=NOW - 10_000))
    assert stale.status_code == 400 and stale.json()["reason"] == "stale_request"
    assert served.in_use(MINER.ss58_address) == 0                       # nothing above leased anything
    first = body(at=NOW + 4)
    assert client.post("/rl/lease", json=first).status_code == 200
    replay = client.post("/rl/lease", json=first)
    assert replay.status_code == 409 and replay.json()["reason"] == "request_reused"


def test_an_unregistered_hotkey_is_refused_and_each_minute_is_bounded(served):
    from reliquary.validator.corpus_registration import NOT_REGISTERED

    async def gate(hotkey):
        return None if hotkey == MINER.ss58_address else NOT_REGISTERED

    client = client_for(served, registration=gate)
    refused = client.post("/rl/lease", json=body(EVE))
    assert refused.status_code == 403 and refused.json()["reason"] == "hotkey_not_registered"
    codes = [client.post("/rl/lease", json=body(at=NOW + i, prompts=(i,))).status_code
             for i in range(LEASE_MAX_PER_MINUTE + 1)]
    assert codes[:-1] == [200] * LEASE_MAX_PER_MINUTE and codes[-1] == 429


def test_malformed_requests_are_refused_before_any_interval(served):
    client = client_for(served)
    extra = {**body(), "nonce": "00"}
    assert client.post("/rl/lease", json=extra).status_code == 422          # extra=forbid
    for bad in (body(env="unknown"), body(prompts=(5000,), at=NOW + 1), body(prompts=(1, 1), at=NOW + 2)):
        answer = client.post("/rl/lease", json=bad)
        assert answer.status_code == 409 and answer.json()["reason"] == "lease_invalid", answer.json()


def test_the_quota_route_explains_a_hotkeys_state(served):
    client = client_for(served)
    client.post("/rl/lease", json=body())
    answer = client.get("/rl/quota", params={"hotkey": MINER.ss58_address})
    assert answer.status_code == 200
    state = answer.json()
    assert state["quota"] == 4 and state["in_use"] == 2 and state["phase"] == "slow_start"
    assert state["counters"]["leased"] == 2 and state["reason"] and state["max_lag"] == 1
    assert client.get("/rl/quota", params={"hotkey": "nope"}).status_code == 422
    assert client.get("/rl/quota").status_code == 422


def test_the_commitments_route_lists_no_open_lease(served):
    client = client_for(served)
    client.post("/rl/lease", json=body())
    content = client.get("/rl/lease/commitments").json()
    assert content["latest"] == 0 and [c["k"] for c in content["active"]] == [0]
    assert "lease" not in str(content).replace("rl-lease", "")


def test_the_prompt_state_route_serves_the_lifecycle_from_a_cursor(served):
    client = client_for(served)
    assert client.post("/rl/lease", json=body(prompts=(3,))).status_code == 200
    served.record_outcome(ENV, 1, 0, [1.0] * 16, "too_easy")
    served.record_outcome(ENV, 2, 0, [1.0] + [0.0] * 15, "in_zone")
    answer = client.get("/rl/prompts/state")
    assert answer.status_code == 200 and answer.headers["cache-control"] == "no-store"
    content = answer.json()
    assert [(p["prompt_idx"], p["state"]) for p in content["prompts"]] == [(1, "blacklisted"), (2, "cooldown")]
    assert [s["verdict"] for s in content["scans"]] == ["too_easy", "in_zone"] and content["latest"] == 0
    assert MINER.ss58_address not in answer.text
    later = client.get("/rl/prompts/state", params={"since": content["cursor"]}).json()
    assert later["scans"] == [] and later["prompts"] == [] and later["cursor"] == content["cursor"]
    assert client.get("/rl/prompts/state", params={"environment": "code"}).json()["scans"] == []
    refused = client.post("/rl/lease", json=body(prompts=(1, 2, 4), at=NOW + 1)).json()["refused"]
    assert refused == [{"prompt_idx": 1, "reason": "blacklisted"}, {"prompt_idx": 2, "reason": "cooling"}]
    for bad in ("-1", "x", "1" * 20):
        assert client.get("/rl/prompts/state", params={"since": bad}).status_code == 422


def test_the_validator_server_mounts_the_lease_routes(served):
    from reliquary.validator.server import ValidatorServer

    server = ValidatorServer()
    server.mount_rl_lease_router(build_rl_lease_router(
        book=served, collector=LeaseCollector(served, interval_s=0.0), validator_hotkey=VALIDATOR.ss58_address,
        clock=lambda: float(NOW)))
    paths = {getattr(route, "path", "") for route in server.app.routes}
    assert {"/rl/lease", "/rl/quota", "/rl/lease/commitments", "/rl/prompts/state"} <= paths
    assert "/rl/submit" not in paths                                  # mounted only with a submission path


def submit_body(keypair, lease_id, rollouts, *, at=NOW, hotkey=None, path="/rl/submit"):
    from reliquary.protocol.rl_submit import encode_tokens, rollouts_digest
    from reliquary.protocol.signatures import build_rl_submit_binding

    hotkey = hotkey or keypair.ss58_address
    wire = [{"index": i, "tokens": encode_tokens(r.tokens), "force_span": None} for i, r in enumerate(rollouts)]
    digest = rollouts_digest([(w["index"], w["tokens"], None) for w in wire])
    binding = build_rl_submit_binding(hotkey=hotkey, at=at, lease_id=lease_id, rollouts_digest=digest,
                                      validator_hotkey=VALIDATOR.ss58_address, path=path)
    return {"miner_hotkey": hotkey, "at": at, "lease_id": lease_id, "rollouts": wire,
            "signature": keypair.sign(binding).hex()}


def submit_app(tmp_path, registration=None, headers=None):
    from reliquary.protocol.rl_submit import HOTKEY_HEADER
    from tests.unit.test_rl_submit import ENV as MATH, MathEnv, Tokenizer
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions

    book = LeaseBook(tmp_path / "submit.sqlite3", LeaseBookConfig(expiry_s={MATH: 600.0}, cooldown_of=lambda e: 2),
                     quota=QuotaController(QuotaConfig(initial=8)), clock=lambda: float(NOW))
    book.adopt_checkpoint(0)
    subs = RlSubmissions(book, GroupGrader({MATH: MathEnv()}, Tokenizer(), eos_token_ids=(1,)),
                         clock=lambda: float(NOW), rng=random.Random(0))
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=VALIDATOR.ss58_address, clock=lambda: float(NOW),
                                             submissions=subs, registration=registration))
    return book, subs, TestClient(app, headers={HOTKEY_HEADER: MINER.ss58_address} if headers is None else headers)


@pytest.fixture
def submitting(tmp_path):
    return submit_app(tmp_path)


def test_a_submission_names_a_registered_hotkey_before_its_body_is_read(tmp_path):
    from reliquary.protocol.rl_submit import HOTKEY_HEADER
    from reliquary.validator.corpus_registration import NOT_REGISTERED
    from tests.unit.test_rl_submit import ENV as MATH, group
    from reliquary.validator.rl_leases import LeaseRequest

    asked = []

    async def registration(hotkey):
        asked.append(hotkey)
        return None if hotkey == MINER.ss58_address else NOT_REGISTERED

    book, _, client = submit_app(tmp_path, registration, headers={})
    junk = {"Content-Encoding": "gzip"}
    assert client.post("/rl/submit", content=b"x").json()["reason"] == "hotkey_required"
    stranger = client.post("/rl/submit", content=b"\x1f\x8b not gzip",
                           headers={**junk, HOTKEY_HEADER: EVE.ss58_address})
    assert stranger.status_code == 403 and stranger.json()["reason"] == "hotkey_not_registered"   # never inflated
    assert asked == [EVE.ss58_address]
    lease = book.grant([LeaseRequest(MINER.ss58_address, MATH, (1,), 0)])[0].leases[0]
    other = client.post("/rl/submit", json=submit_body(MINER, lease.lease_id, group(8)),
                        headers={HOTKEY_HEADER: RIVAL.ss58_address})
    assert other.status_code == 403 and book.lease(lease.lease_id).state == "open"   # RIVAL is not registered

    async def everyone(hotkey):
        return None

    (tmp_path / "b").mkdir()
    book2, _, client2 = submit_app(tmp_path / "b", everyone, headers={HOTKEY_HEADER: RIVAL.ss58_address})
    lease = book2.grant([LeaseRequest(MINER.ss58_address, MATH, (1,), 0)])[0].leases[0]
    mismatch = client2.post("/rl/submit", json=submit_body(MINER, lease.lease_id, group(8)))
    assert mismatch.status_code == 422 and book2.lease(lease.lease_id).state == "open"


def test_an_on_time_submission_survives_the_expiry_sweep_while_it_is_graded(submitting):
    from tests.unit.test_rl_submit import ENV as MATH, group
    from reliquary.validator.rl_leases import LeaseRequest

    book, subs, client = submitting
    lease = book.grant([LeaseRequest(MINER.ss58_address, MATH, (1,), 0)])[0].leases[0]
    grade, swept = subs.grader.grade, []

    def slow_grade(*args):                                  # the sweep runs past the expiry while the group grades
        swept.append(book.expire_due(now=lease.expires_at + 60))
        return grade(*args)

    subs.grader.grade = slow_grade
    answer = client.post("/rl/submit", json=submit_body(MINER, lease.lease_id, group(8)))
    assert answer.status_code == 200 and answer.json()["verdict"] == "in_zone" and swept == [0]
    assert book.lease(lease.lease_id).state == "submitted" and book._submitting == {}
    other = book.grant([LeaseRequest(MINER.ss58_address, MATH, (2,), 0)])[0].leases[0]
    assert book.expire_due(now=other.expires_at) == 1                  # released: expiry applies again


def test_a_signed_submission_ends_its_lease_and_answers_the_validators_grades(submitting):
    from tests.unit.test_rl_submit import ENV as MATH, group
    from reliquary.validator.rl_leases import LeaseRequest

    book, subs, client = submitting
    lease = book.grant([LeaseRequest(MINER.ss58_address, MATH, (1,), 0)])[0].leases[0]
    rollouts = group(8)
    missing = submit_body(MINER, lease.lease_id, rollouts[:15])
    assert client.post("/rl/submit", json=missing).status_code == 422          # every index required
    forged = submit_body(EVE, lease.lease_id, rollouts, hotkey=MINER.ss58_address, at=NOW + 1)
    assert client.post("/rl/submit", json=forged).status_code == 403
    rival = book.grant([LeaseRequest(RIVAL.ss58_address, MATH, (2,), 0)])[0].leases[0]
    theirs = client.post("/rl/submit", json=submit_body(MINER, rival.lease_id, rollouts, at=NOW + 2))
    assert theirs.status_code == 403 and theirs.json()["reason"] == "lease_not_yours"
    bad = submit_body(MINER, lease.lease_id, rollouts, at=NOW + 3)
    bad["rollouts"][0]["tokens"] = "not-zlib"
    assert client.post("/rl/submit", json=bad).status_code == 403              # the signature covers the tokens
    assert book.lease(lease.lease_id).state == "open"
    answer = client.post("/rl/submit", json=submit_body(MINER, lease.lease_id, rollouts, at=NOW + 4))
    assert answer.status_code == 200, answer.json()
    assert answer.json() == {"lease_id": lease.lease_id, "verdict": "in_zone", "rewards": [1.0] * 8 + [0.0] * 8}
    assert subs.buffer.depth() == {MATH: 1} and book.lease(lease.lease_id).state == "submitted"
    again = client.post("/rl/submit", json={**submit_body(MINER, lease.lease_id, rollouts, at=NOW + 5),
                                            "k_generated": 3})                # informational, unsigned
    assert again.status_code == 200 and again.json() == {**answer.json(), "resent": True}
    assert subs.buffer.depth() == {MATH: 1}
    unknown = client.post("/rl/submit", json=submit_body(MINER, "f" * 32, rollouts, at=NOW + 6))
    assert unknown.status_code == 404 and unknown.json()["reason"] == "lease_unknown"


def test_undecodable_tokens_are_refused_before_the_lease_is_touched(submitting):
    from reliquary.protocol.rl_submit import rollouts_digest
    from reliquary.protocol.signatures import build_rl_submit_binding
    from tests.unit.test_rl_submit import ENV as MATH, group
    from reliquary.validator.rl_leases import LeaseRequest

    book, _, client = submitting
    lease = book.grant([LeaseRequest(MINER.ss58_address, MATH, (3,), 0)])[0].leases[0]
    body = submit_body(MINER, lease.lease_id, group(8))
    body["rollouts"][0]["tokens"] = "AAAA"                                    # base64, not zlib
    digest = rollouts_digest([(w["index"], w["tokens"], None) for w in body["rollouts"]])
    body["signature"] = MINER.sign(build_rl_submit_binding(
        hotkey=MINER.ss58_address, at=NOW, lease_id=lease.lease_id, rollouts_digest=digest,
        validator_hotkey=VALIDATOR.ss58_address, path="/rl/submit")).hex()
    answer = client.post("/rl/submit", json=body)
    assert answer.status_code == 422 and book.lease(lease.lease_id).state == "open"


def test_a_miner_lists_only_its_own_open_leases_with_a_signed_read(served):
    from reliquary.protocol.signatures import build_rl_lease_mine_binding

    client = client_for(served)
    assert client.post("/rl/lease", json=body()).status_code == 200
    assert client.post("/rl/lease", json=body(RIVAL, prompts=(5,))).status_code == 200

    def params(keypair=MINER, at=NOW, path="/rl/lease/mine", signer=None):
        binding = build_rl_lease_mine_binding(hotkey=keypair.ss58_address, at=at,
                                              validator_hotkey=VALIDATOR.ss58_address, path=path)
        return {"hotkey": keypair.ss58_address, "at": str(at), "signature": (signer or keypair).sign(binding).hex()}

    mine = client.get("/rl/lease/mine", params=params())
    assert mine.status_code == 200, mine.json()
    assert sorted(x["prompt_idx"] for x in mine.json()["leases"]) == [1, 2]
    assert mine.json()["lease_expiry_s"] == {ENV: 600.0} and mine.json()["in_use"] == 2
    assert all(x["nonce"] for x in mine.json()["leases"])
    assert client.get("/rl/lease/mine", params=params(signer=EVE)).status_code == 403          # not the hotkey's
    assert client.get("/rl/lease/mine", params=params(path="/rl/quota")).status_code == 403    # another route
    assert client.get("/rl/lease/mine", params=params(at=NOW - 1_000)).status_code == 400      # stale
    replay = params(at=NOW + 1)
    assert client.get("/rl/lease/mine", params=replay).status_code == 200
    assert client.get("/rl/lease/mine", params=replay).json()["reason"] == "request_reused"
    assert client.get("/rl/lease/mine", params={"hotkey": MINER.ss58_address, "at": str(NOW)}).status_code == 401
    quota = client.get("/rl/quota", params={"hotkey": MINER.ss58_address}).json()
    assert quota["lease_expiry_s"] == {ENV: 600.0}
    assert client.post("/rl/lease", json=body(prompts=(9,), at=NOW + 2)).json()["lease_expiry_s"] == 600.0


def test_a_miner_forfeits_its_own_open_lease_at_the_price_of_an_expiry(served):
    from reliquary.protocol.signatures import build_rl_lease_forfeit_binding

    client = client_for(served)
    leases = client.post("/rl/lease", json=body()).json()["leases"]
    rival = client.post("/rl/lease", json=body(RIVAL, prompts=(5,))).json()["leases"][0]
    before = served.quota.quota(MINER.ss58_address, NOW)

    def forfeit(lease_id, keypair=MINER, at=NOW, signer=None):
        binding = build_rl_lease_forfeit_binding(hotkey=keypair.ss58_address, at=at, lease_id=lease_id,
                                                 validator_hotkey=VALIDATOR.ss58_address, path="/rl/lease/forfeit")
        return client.post("/rl/lease/forfeit", json={"miner_hotkey": keypair.ss58_address, "at": at,
                                                      "lease_id": lease_id,
                                                      "signature": (signer or keypair).sign(binding).hex()})

    assert forfeit(leases[0]["lease_id"], signer=EVE).status_code == 403                  # not the hotkey's
    assert forfeit(rival["lease_id"]).status_code == 409                                  # not its lease
    done = forfeit(leases[0]["lease_id"])
    assert done.status_code == 200 and done.json()["state"] == "expired"
    assert served.lease(leases[0]["lease_id"]).state == "expired" and served.in_use(MINER.ss58_address) == 1
    assert served.quota.quota(MINER.ss58_address, NOW) < before                           # charged as an expiry
    assert forfeit(leases[0]["lease_id"], at=NOW + 1).status_code == 409                  # once
