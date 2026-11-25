"""The RL auction routes: ``POST /rl/ask`` (a signed offer per env), ``GET /rl/auction`` (the public view), and the
lease answers and ``GET /rl/quota`` of a task priced by auction (its allowances)."""
from __future__ import annotations

import random

import pytest

bt = pytest.importorskip("bittensor")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reliquary.corpus.token_auction import AuctionParams  # noqa: E402
from reliquary.protocol.signatures import build_rl_ask_binding, build_rl_lease_binding  # noqa: E402
from reliquary.validator.rl_auction import RlAuctions  # noqa: E402
from reliquary.validator.rl_lease_routes import build_rl_lease_router  # noqa: E402
from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
NOW = 1_800_000_000
ENV = "math"


@pytest.fixture
def client(tmp_path):
    auctions = RlAuctions(AuctionParams(ceiling=1e-6, miner_alpha_per_period=1.0), {ENV: 1.0}, batch_groups=8,
                          lag=2, task_id="t", clock=lambda: float(NOW), state_dir_job=lambda job: None)
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}),
                     quota=None, clock=lambda: float(NOW), rng=random.Random(1), auctions=auctions)
    book.adopt_checkpoint(0)
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=VALIDATOR.ss58_address, clock=lambda: float(NOW)))
    return TestClient(app)


def ask(micro=500_000, capacity=4, at=NOW, env=ENV):
    binding = build_rl_ask_binding(hotkey=MINER.ss58_address, at=at, environment=env, micro_alpha_per_mtoken=micro,
                                   capacity=capacity, validator_hotkey=VALIDATOR.ss58_address, path="/rl/ask")
    return {"miner_hotkey": MINER.ss58_address, "at": at, "environment": env, "micro_alpha_per_mtoken": micro,
            "capacity": capacity, "signature": MINER.sign(binding).hex()}


def lease(prompts, at=NOW):
    binding = build_rl_lease_binding(hotkey=MINER.ss58_address, at=at, environment=ENV, checkpoint=0,
                                     prompt_ids=list(prompts), validator_hotkey=VALIDATOR.ss58_address,
                                     path="/rl/lease")
    return {"miner_hotkey": MINER.ss58_address, "at": at, "environment": ENV, "checkpoint": 0,
            "prompt_ids": list(prompts), "signature": MINER.sign(binding).hex()}


def test_a_miner_offers_then_leases_within_its_allowance(client):
    refused = client.post("/rl/lease", json=lease([1])).json()
    assert refused["refused"] == [{"prompt_idx": 1, "reason": "no_ask"}] and refused["quota"] == 0
    offer = ask()
    answer = client.post("/rl/ask", json=offer)
    assert answer.status_code == 200, answer.json()
    assert answer.json()["auction"]["allowance"] == 4
    assert client.post("/rl/ask", json=offer).json()["reason"] == "request_reused"      # single use
    granted = client.post("/rl/lease", json=lease([1, 2, 3, 4, 5], at=NOW + 1)).json()
    assert len(granted["leases"]) == 4 and [r["reason"] for r in granted["refused"]] == ["full"]
    assert granted["quota"] == 4 and granted["in_use"] == 4
    quota = client.get("/rl/quota", params={"hotkey": MINER.ss58_address}).json()
    assert quota["quota"] == 4 and quota["auction"]["environments"][ENV]["held"] == 4


def test_the_public_view_names_no_offer(client):
    client.post("/rl/ask", json=ask())
    view = client.get("/rl/auction").json()
    assert view["lag_target"] == 2 and view["batch_groups"] == 8
    env = view["environments"][ENV]
    assert env["target_groups"] == 16 and env["budget"] == 32      # a prior yield of 1/2 and env["ceiling_alpha_per_mtoken"] == pytest.approx(1)
    assert "allowance" not in env and MINER.ss58_address not in str(view)


def test_a_bad_offer_is_refused(client):
    assert client.post("/rl/ask", json=ask(env="nope")).json()["reason"] == "ask_invalid"
    tampered = ask()
    tampered["micro_alpha_per_mtoken"] = 1
    assert client.post("/rl/ask", json=tampered).json()["reason"] == "bad_signature"


async def test_a_task_path_offer_is_signed_for_its_task(tmp_path):
    import json

    import httpx

    from reliquary.miner.rl_async import HttpRlTransport, RequestRefused
    from reliquary.validator.rl_front import build_front_app

    auctions = RlAuctions(AuctionParams(ceiling=1e-6, miner_alpha_per_period=1.0), {ENV: 1.0}, batch_groups=8,
                          lag=2, task_id="t", clock=lambda: float(NOW), state_dir_job=lambda job: None)
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}),
                     quota=None, clock=lambda: float(NOW), rng=random.Random(1), auctions=auctions)
    book.adopt_checkpoint(0)
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=VALIDATOR.ss58_address, clock=lambda: float(NOW),
                                             task_id="t"))
    spec = tmp_path / "map.json"
    spec.write_text(json.dumps({"default": "t", "tasks": {"t": {"port": 8001}}}))
    front = build_front_app(spec, client_factory=lambda port: httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=front), base_url="http://v") as client:
        def transport(prefix):
            return HttpRlTransport(client, base_url="http://v", hotkey=MINER.ss58_address,
                                   validator_hotkey=VALIDATOR.ss58_address, prefix=prefix, clock=lambda: float(NOW),
                                   sign_binding=lambda binding: MINER.sign(binding).hex())

        answer = await transport("/rl/t").ask(ENV, 500_000, 4)
        assert answer["accepted"] and answer["auction"]["allowance"] == 4
        bare = await transport("/rl").ask(ENV, 500_000, 3)               # the default task, verified as /rl/ask
        assert bare["accepted"] and bare["auction"]["allowance"] == 3
        signed_for_other = ask(at=NOW + 1)
        signed_for_other["signature"] = MINER.sign(build_rl_ask_binding(
            hotkey=MINER.ss58_address, at=NOW + 1, environment=ENV, micro_alpha_per_mtoken=500_000, capacity=4,
            validator_hotkey=VALIDATOR.ss58_address, path="/rl/u/ask")).hex()
        refused = await client.post("/rl/t/ask", json=signed_for_other)
        assert refused.status_code == 403 and refused.json()["reason"] == "bad_signature"
        with pytest.raises(RequestRefused) as wrong:
            await transport("/rl/t").ask("nope", 500_000, 3)
        assert wrong.value.reason == "ask_invalid"
        listed = (await client.get("/rl/tasks")).json()
        assert [t["task_id"] for t in listed] == ["t"] and listed[0]["environments"] == {ENV: 1.0}
