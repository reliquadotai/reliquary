import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.corpus.contract_observation import ContractObservation
from reliquary.protocol.signatures import build_corpus_lease_binding
from reliquary.validator.corpus_periods import MAX_PRICE_EMISSION_PARAM, MAX_PRICE_PARAM
from reliquary.validator.corpus_registration import RegisteredHotkeys
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
            observation,
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
    owners = {"miner": "owner"}
    timestamps = [10.5]
    async def load_owners():
        return dict(owners)
    registry = RegisteredHotkeys(load=load_owners, clock=lambda: timestamps[0], keep_owner_history=True)
    asyncio.run(registry.refresh())
    owners["miner"] = "next-owner"
    timestamps[0] = 12
    asyncio.run(registry.refresh())
    assert registry.owners_at(12) == (12, {"miner": "next-owner"})
    assert registry.owners_at(10) is None
    observation.open(entry, registry.owners_at(11))
    too_late = SimpleNamespace(job_id="after-rotation", params=entry.params)
    observation.open(too_late, registry.owners_at(12))
    assert observation.book.contracts("after-rotation") == []

    original = observation.book.contract("order-1", "miner")
    assert original["price_micro"] == 3000 and original["q"] == 1.0
    now[0] = 12
    changed = {**request, "sequence": 2, "micro_alpha_per_mtoken": 1000}
    signed = build_corpus_lease_binding(changed)
    assert client.put(url, json=changed).status_code == 200
    reopened = ContractObservation(tmp_path / "offers.db", [market])
    reopened.open(entry, None)
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
    marker = {"opening_valuation": {MAX_PRICE_PARAM: 0.001, MAX_PRICE_EMISSION_PARAM: 1.0}}
    # After a restart the live registry can carry a different conversion or none.
    capture = observation.recorder(SimpleNamespace(job_id="order", params={
        "order_contract": marker, MAX_PRICE_PARAM: 0.002, MAX_PRICE_EMISSION_PARAM: 9,
    }))
    archive = {
        "tokens": 7,
        "max_price_per_token": 0.001,
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
    assert first["miner_alpha_per_period"] == 1.0 and first["valuation_source"] == "opening_declaration"
    assert [r["id"] for r in first["deliveries"]] == ["accepted"]
    assert second["deliveries"] is None
    assert repriced["miner_alpha_per_period"] is None

@pytest.mark.parametrize("failed_write", ["UPDATE ON orders", "INSERT ON observed_closures"])
def test_recovery_uses_certified_time_after_failed_closure(tmp_path, failed_write):
    obs = ContractObservation(tmp_path / "book.db", [])
    obs.book.open_order("order", "model|turn", opened_at=100, duration_s=100, target_rate=1,
                        max_tokens=2, budget_rao=2, ceiling_micro=1, new_rate_cap=1)
    final = {"observation_closed_at": 103, "counts_complete": True, "verified_tokens": 2, "settled": 1}
    with sqlite3.connect(obs.book.path) as db:
        db.execute(f"CREATE TRIGGER simulate_failure BEFORE {failed_write} "
                   "BEGIN SELECT RAISE(ABORT, 'interrupted'); END")
    with pytest.raises(sqlite3.IntegrityError, match="interrupted"):
        obs.close("order", 103, final)
    with sqlite3.connect(obs.book.path) as db:
        db.execute("DROP TRIGGER simulate_failure")
    async def read_final(job_id):
        assert job_id == "order"
        return final
    restarted = ContractObservation(obs.book.path, [])
    records = SimpleNamespace(read_final_status=read_final)
    asyncio.run(restarted.recover_closures(records))
    asyncio.run(restarted.recover_closures(records))
    assert restarted.book.order("order")["closed_at"] == 103
    with sqlite3.connect(obs.book.path) as db:
        assert db.execute("SELECT verified_tokens,settled FROM observed_closures").fetchall() == [(2, 1)]


def test_busy_local_book_does_not_delay_initialization_and_can_recover(tmp_path, monkeypatch):
    from reliquary.corpus.contract_observation import DB_ENV
    from reliquary.corpus.contract_spec import MARKETS_ENV

    path = tmp_path / "busy.db"
    monkeypatch.setenv(DB_ENV, str(path))
    monkeypatch.setenv(MARKETS_ENV, json.dumps({"model|turn": {"mode": "observation"}}))
    with sqlite3.connect(path) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        observer = ContractObservation.from_env()
        app = FastAPI()
        app.include_router(standing_offers_router(observer, registration=None, operator_of=None))
        async def while_locked():
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                await asyncio.wait_for(observer.recover_closures(None), timeout=1)
        asyncio.run(while_locked())
    assert observer.book.order("absent") is None
