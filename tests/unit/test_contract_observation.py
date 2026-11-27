import json
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.contract_observation import ContractObservation
from reliquary.protocol.signatures import build_corpus_lease_binding
from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM
from reliquary.validator.standing_offers import standing_offers_router


def test_signed_offer_before_opening_changes_only_future_orders(tmp_path):
    market = "test-model|turn"
    observation = ContractObservation(tmp_path / "offers.db", [market])
    now = [10]

    async def registration(hotkey):
        return None if hotkey == "miner" else "not_registered"

    body = {
        "market": market,
        "miner_hotkey": "miner",
        "coldkey": "owner",
        "sequence": 1,
        "micro_alpha_per_mtoken": 3000,
        "tokens_per_second": 50,
        "slots": 2,
    }
    signed = build_corpus_lease_binding(body)
    app = FastAPI()
    app.include_router(
        standing_offers_router(
            observation.book,
            [market],
            registration=registration,
            operator_of=lambda _: "owner",
            verify=lambda row: build_corpus_lease_binding(row) == signed,
            clock=lambda: now[0],
        )
    )
    client = TestClient(app)
    url = "/corpus/standing-offer"
    request = dict(body, signature="test")
    assert client.put(url, json=request).json() == {
        "accepted": True,
        "sequence": 1,
        "mode": "observation",
    }
    assert client.put(url, json=request).status_code == 200
    for field, value in (
        ("micro_alpha_per_mtoken", 0),
        ("tokens_per_second", 100),
        ("coldkey", "other"),
        ("slots", 3),
        ("sequence", 2),
    ):
        assert client.put(url, json={**request, field: value}).status_code == 403
    assert client.get(url).status_code == 405

    marker = {
        "version": 1,
        "mode": "observation",
        "market": market,
        "opened_at": 11,
        "duration_s": 20,
        "target_rate": 50,
        "max_tokens": 1000,
        "budget_rao": 3000,
        "ceiling_micro": 3000,
        "new_rate_cap": 50,
        "audit_cost_micro": 1,
        "evaluation": True,
        "q_min": 0.01,
        "trust_tokens": 1000,
    }
    entry = SimpleNamespace(
        job_id="order-1",
        params={"order_contract": marker, MAX_PRICE_EMISSION_PARAM: 1.0},
    )
    missing = SimpleNamespace(job_id="unobserved", params=entry.params)
    observation.open(missing, None)
    assert observation.book.contracts("unobserved") == []
    observation.open(entry, {"miner": "owner"})
    original = observation.book.contract("order-1", "miner")
    assert original["price_micro"] == 3000 and original["q"] == 1.0
    now[0] = 12
    changed = {**request, "sequence": 2, "micro_alpha_per_mtoken": 1000}
    signed = build_corpus_lease_binding(changed)
    assert client.put(url, json=changed).status_code == 200
    reopened = ContractObservation(tmp_path / "offers.db", [market])
    reopened.open(entry, {})
    assert reopened.book.contract("order-1", "miner") == original
    reopened.close(
        "order-1", 30, {"counts_complete": True, "verified_tokens": 7, "settled": 2}
    )
    assert reopened.book.summary("order-1")["median_micro"] == 3000
    with sqlite3.connect(reopened.book.path) as db:
        assert db.execute(
            "SELECT verified_tokens, settled FROM observed_closures"
        ).fetchone() == (7, 2)


def test_archive_observation_is_private_and_recoverable(tmp_path):
    observation = ContractObservation(tmp_path / "observation.db", [])
    capture = observation.recorder(
        SimpleNamespace(job_id="order", params={MAX_PRICE_EMISSION_PARAM: 1.0})
    )
    archive = {
        "tokens": 7,
        "rewards_by_hotkey": {"a": 0.01},
        "work_period": 10,
        "entry_period": 12,
    }
    rows = [
        {
            "hotkey": "a",
            "token_count": 7,
            "received_at": 100,
            "audited": True,
            "passed": True,
            "lease_price": 0.001,
        },
        {"hotkey": "b", "token_count": 3, "received_at": 101, "passed": False},
    ]
    capture(archive, ["accepted", "voided"], rows)
    capture(archive, ["accepted", "voided"], None)
    capture({**archive, "entry_period": 13}, ["earlier"], None)
    capture(
        {**archive, "entry_period": 14, "max_price_per_token": 0.002},
        ["accepted", "voided"],
        rows,
    )
    assert "deliveries" not in archive
    with sqlite3.connect(observation.book.path) as db:
        first, second, repriced = [
            json.loads(row[0])
            for row in db.execute(
                "SELECT document FROM observed_periods ORDER BY entry_period"
            )
        ]
    assert [r["id"] for r in first["deliveries"]] == ["accepted"]
    assert second["deliveries"] is None
    assert repriced["miner_alpha_per_period"] is None
