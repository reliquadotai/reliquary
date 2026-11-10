"""A draw job priced by auction, through its routes: a hotkey posts an offer, its leases carry its price, only a
leased slot is accepted, the record carries the lease's terms, an unused lease is paid back, and a period pays
every token at one price."""
from __future__ import annotations

import asyncio
import secrets
import time

import pytest

from reliquary.corpus.token_auction import AuctionParams, TokenAuction
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)
from tests.unit.test_draw_leases import _submit_as

EMISSION = 100.0


class Clock:
    def __init__(self, now=6000.0):
        self.now = now

    def __call__(self):
        return self.now


class _Records:
    def __init__(self):
        self.submissions = {}

    async def write_submission(self, job_id, submission_id, record):
        if submission_id in self.submissions:
            return False
        self.submissions[submission_id] = record
        return True


def _auction(clock, ceiling_alpha=0.5):
    return TokenAuction(AuctionParams(ceiling=ceiling_alpha / 1e6 / EMISSION, miner_alpha_per_period=EMISSION,
                                      initial_leases=8), clock=clock, mean_tokens=100)


def _route(fake_r2, auction, records, *, prompts=2, slots=1):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.validator.corpus_service import build_corpus_router
    from tests.unit.test_corpus_service import (
        _CountingStore, _Environment, _manifest, _Renderer, _SeededJob, _Tokenizer,
    )

    raw = {**_manifest(), "prompt_count": prompts, "slots_per_prompt": slots, "seed": 5,
           "sampling": {**_manifest()["sampling"], "draw": "gumbel_max"}}
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    seeded = _SeededJob(_CountingStore(fake_r2), _Renderer(), _Environment(), raw)
    app = FastAPI()
    router = build_corpus_router(
        job_id="swe-v1", store=seeded.store, tokenizer=_Tokenizer(), renderer=seeded.renderer,
        verify_signature=lambda request: True, verify_lease_signature=lambda request: request.signature != "bad",
        prompt_job_for=seeded.prompt_job_for, records=records, durable_records=False, auction=auction)
    app.include_router(router)
    return TestClient(app), router


def _ask(client, hotkey, alpha, *, signature="ok", capacity=None):
    body = {"job_id": "swe-v1", "miner_hotkey": hotkey, "micro_alpha_per_mtoken": int(alpha * 1e6),
            "issued_at": int(time.time()), "nonce": secrets.token_hex(16), "signature": signature}
    if capacity is not None:
        body["capacity"] = capacity
    return client.post("/corpus/ask", json=body).json()


def _lease(client, hotkey, count=None):
    body = {"job_id": "swe-v1", "miner_hotkey": hotkey, "issued_at": int(time.time()),
            "nonce": secrets.token_hex(16), "signature": "ok"}
    if count is not None:
        body["count"] = count
    return client.post("/corpus/lease", json=body).json()


def test_a_hotkey_leases_only_after_posting_an_ask(fake_r2):
    clock = Clock()
    client, _ = _route(fake_r2, _auction(clock), _Records())
    assert _lease(client, "5A")["reason"] == "no_ask"
    assert _ask(client, "5A", 0.3, signature="bad")["reason"] == "bad_signature"
    answer = _ask(client, "5A", 0.3)
    assert answer["accepted"] is True and "5A" not in repr(answer["detail"])
    lease = _lease(client, "5A")
    assert lease["leased"] is True
    assert lease["detail"]["price_alpha_per_mtoken"] == pytest.approx(0.3)


def test_an_ask_over_the_round_price_gets_no_lease(fake_r2):
    clock = Clock()
    client, _ = _route(fake_r2, _auction(clock, ceiling_alpha=0.5), _Records())
    _ask(client, "5A", 0.9)
    refused = _lease(client, "5A")
    assert refused["reason"] == "ask_above_price"
    assert refused["detail"]["ceiling_alpha_per_mtoken"] == pytest.approx(0.5)


def test_only_a_leased_slot_is_accepted_and_its_record_carries_the_price(fake_r2):
    clock = Clock()
    auction = _auction(clock)
    records = _Records()
    client, _ = _route(fake_r2, auction, records)
    _ask(client, "5A", 0.2)
    assert _submit_as(client, "5A", 0, 0)["reason"] == "lease_required"
    lease = _lease(client, "5A")
    assert _submit_as(client, "5A", lease["prompt_index"], lease["slot"])["accepted"] is True
    (record,) = records.submissions.values()
    assert auction.params.alpha_of(record["lease_price"]) == pytest.approx(0.2)
    assert record["lease_debit"] == 0


def test_an_unused_lease_is_taken_back_from_the_next_accepted_submission(fake_r2):
    clock = Clock()
    auction = _auction(clock)
    records = _Records()
    client, router = _route(fake_r2, auction, records)
    _ask(client, "5A", 0.2)
    first = _lease(client, "5A")
    release = {"job_id": "swe-v1", "miner_hotkey": "5A", "lease_id": first["lease_id"], "signature": "ok"}
    assert client.post("/corpus/lease/release", json=release).json()["reason"] == "released"
    price = auction.params.share_of(0.2)
    auction.lease_ended("5A", price, 1.0)               # as an expiry: 100 tokens owed at 0.2
    second = _lease(client, "5A")
    assert _submit_as(client, "5A", second["prompt_index"], second["slot"])["accepted"] is True
    (record,) = records.submissions.values()
    assert record["lease_debit"] > 0
    assert record["lease_debit"] <= record["lease_price"] * record["token_count"]


def test_a_refused_submission_gives_its_debit_back(fake_r2):
    clock = Clock()
    auction = _auction(clock)
    client, _ = _route(fake_r2, auction, _Records(), prompts=1)
    _ask(client, "5A", 0.2)
    lease = _lease(client, "5A")
    owed = auction.params.share_of(0.2) * 100
    auction.lease_ended("5A", auction.params.share_of(0.2), 1.0)
    assert _submit_as(client, "5A", lease["prompt_index"], lease["slot"])["accepted"] is True
    left = auction.take_debit("5A", 1.0)
    auction.refund_debit("5A", left)
    # The slot is taken: a resend is refused and keeps the debt where it was.
    assert _submit_as(client, "5A", lease["prompt_index"], lease["slot"])["accepted"] is False
    assert auction.take_debit("5A", 1.0) == pytest.approx(left)
    assert left < owed


def test_a_job_not_priced_by_auction_refuses_asks(fake_r2):
    client, _ = _route(fake_r2, None, _Records())
    assert _ask(client, "5A", 0.2)["reason"] == "not_priced_by_auction"
    assert _lease(client, "5A")["leased"] is True


def test_the_settlement_pays_every_token_at_the_periods_price_less_the_debit():
    from reliquary.validator.corpus_settlement import rewards_for

    verdicts = [
        {"passed": True, "hotkey": "a", "token_count": 100, "lease_price": 1e-6, "lease_debit": 0.0},
        {"passed": True, "hotkey": "a", "token_count": 100, "lease_price": 2e-6, "lease_debit": 5e-5},
        {"passed": True, "hotkey": "b", "token_count": 50, "lease_price": 1e-6, "lease_debit": 1.0},
        {"passed": False, "hotkey": "c", "token_count": 100, "lease_price": 1e-6, "lease_debit": 0.0},
    ]
    # Every token at the period's price, the highest lease delivered (2e-6); b's debt passes its pay.
    assert rewards_for(verdicts, 1.0) == {"a": pytest.approx(4e-4 - 5e-5)}
    assert rewards_for(verdicts, 1.0, uniform_price=3e-6) == {"a": pytest.approx(6e-4 - 5e-5)}
    # Over the pool: scaled down to it.
    assert sum(rewards_for(verdicts, 1e-4).values()) == pytest.approx(1e-4)


# ------------------------------------------------------------------ the miner


def test_the_miner_posts_its_ask_before_leasing_and_again_when_the_validator_has_none():
    from tests.unit.test_draw_leases import _Gen, _granted, _LeasingClient, _mine

    class _AskingClient(_LeasingClient):
        def __init__(self, leases):
            super().__init__(leases)
            self.asks = []
            self.order = []

        def ask(self, body):
            self.asks.append(body)
            self.order.append("ask")
            return {"accepted": True, "reason": "accepted"}

        def lease(self, body):
            self.order.append("lease")
            return super().lease(body)

    client = _AskingClient([{"leased": False, "reason": "no_ask"}, _granted(10, 0, "aa" * 16)])
    _mine(client, _Gen(), max_steps=3, sign_lease=lambda body: "sig", ask_alpha_per_mtoken=0.25)
    assert client.order[:3] == ["ask", "lease", "ask"]
    assert client.asks[0]["micro_alpha_per_mtoken"] == 250_000
    assert client.submitted == [(10, 0)]


def test_a_miner_without_an_ask_waits_on_a_job_priced_by_auction():
    from tests.unit.test_draw_leases import _Gen, _LeasingClient, _mine

    client = _LeasingClient([{"leased": False, "reason": "no_ask"}])
    counts = _mine(client, _Gen(), max_steps=1, sign_lease=lambda body: "sig")
    assert counts["lease_no_ask"] == 1 and client.submitted == []


def test_an_ask_is_signed_under_its_own_domain():
    from reliquary.protocol.signatures import CORPUS_ASK_DOMAIN, build_corpus_lease_binding

    lease = {"job_id": "j", "miner_hotkey": "h", "issued_at": 1, "nonce": "ab" * 16}
    ask = {**lease, "micro_alpha_per_mtoken": 5}
    assert build_corpus_lease_binding(ask) != build_corpus_lease_binding(lease)
    assert build_corpus_lease_binding(ask) != build_corpus_lease_binding({**ask, "micro_alpha_per_mtoken": 6})
    assert CORPUS_ASK_DOMAIN == b"reliquary/corpus-ask/v1"
    # A capacity, or a lease count, is bound under a new domain: an older body still verifies as it did.
    assert build_corpus_lease_binding({**ask, "capacity": None}) == build_corpus_lease_binding(ask)
    assert build_corpus_lease_binding({**ask, "capacity": 4}) != build_corpus_lease_binding({**ask, "capacity": 5})
    assert build_corpus_lease_binding({**lease, "count": None}) == build_corpus_lease_binding(lease)
    assert build_corpus_lease_binding({**lease, "count": 2}) != build_corpus_lease_binding(lease)


def test_a_record_carries_both_lease_terms_or_neither():
    from reliquary.infrastructure.corpus_record_store import _record_identity
    from tests.unit.test_corpus_service import EOS

    from reliquary.infrastructure import corpus_record_store as store
    from reliquary.protocol.corpus_submission import CorpusCompletion

    completion = CorpusCompletion(tokens=[7, 7, EOS], text="x").model_dump()
    record = {"submission_id": "ab" * 32, "job_id": "swe-v1",
              "hotkey": "5A", "cursor": 0, "prompt_index": 0, "rendered_prompt": "p", "received_at": 1.0,
              "token_count": 3, "completions": [completion]}

    record["schema"] = store.RECORD_SCHEMA_V1
    _record_identity("swe-v1", "ab" * 32, record)
    _record_identity("swe-v1", "ab" * 32, {**record, "lease_price": 1e-6, "lease_debit": 0.0})
    with pytest.raises(ValueError):
        _record_identity("swe-v1", "ab" * 32, {**record, "lease_price": 1e-6})
    with pytest.raises(ValueError):
        _record_identity("swe-v1", "ab" * 32, {**record, "lease_price": -1.0, "lease_debit": 0.0})


def test_record_metadata_reads_the_lease_terms():
    import json

    from reliquary.infrastructure.corpus_record_store import submission_meta

    tail = json.dumps({"cursor": 0, "hotkey": "5A", "received_at": 2.0, "token_count": 9,
                       "lease_price": 1.5e-06, "lease_debit": 0.0, "completions": []}).replace(" ", "").encode()
    meta = submission_meta(tail)
    assert meta["lease_price"] == pytest.approx(1.5e-6) and meta["lease_debit"] == 0.0


# ------------------------------------------------------------------ audit and settlement


def test_the_verdict_carries_its_records_lease_terms():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    auditor = CorpusAuditor.__new__(CorpusAuditor)
    auditor._meta, auditor._lease_terms, auditor._arrivals = {}, {}, {}
    auditor._judged, auditor._unjudged, auditor._clock = set(), {}, lambda: 5.0
    auditor._remember("s1", {"hotkey": "h", "received_at": 1.0, "token_count": 3,
                             "lease_price": 1e-6, "lease_debit": 2e-7})
    auditor._remember("s2", {"hotkey": "h", "received_at": 1.0, "token_count": 3})
    priced = auditor._verdict("s1", "h", 3, {"passed": True}, None)
    assert (priced["lease_price"], priced["lease_debit"]) == (1e-6, 2e-7)
    assert "lease_price" not in auditor._verdict("s2", "h", 3, {"passed": True}, None)


def test_a_priced_period_pays_its_leases_and_burns_the_rest():
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
    from tests.unit.test_corpus_period_settlement import GENESIS, Archives, Records, at, verdict

    records, archives = Records(), Archives()
    records.verdicts = {
        "a1": {**verdict("a1", "a", 1000, at(3)), "lease_price": 2e-5, "lease_debit": 0.0},
        "b1": {**verdict("b1", "b", 1000, at(3)), "lease_price": 2e-5, "lease_debit": 5e-3},
    }
    settler = CorpusPeriodSettler(task_id="t", job_id="t", cap=0.5, records=records, archives=archives,
                                  oldest_pending=lambda: None, genesis=lambda: GENESIS,
                                  clock=lambda: at(5, 1000), max_price=1e-4)
    asyncio.run(settler.settle_once())
    (doc,) = archives.docs.values()
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(0.02), "b": pytest.approx(0.015)}
    assert records.state["period_paid"][str(3)] == pytest.approx(0.035)
    assert cp.period_budget(0.5, 1e-4, 2000) == pytest.approx(0.2)


def test_one_request_leases_several_slots_up_to_the_offers_capacity(fake_r2):
    clock = Clock()
    auction = _auction(clock)
    client, _ = _route(fake_r2, auction, _Records(), prompts=4)
    _ask(client, "5A", 0.2, capacity=3)
    answer = _lease(client, "5A", count=5)
    assert answer["leased"] is True
    assert len(answer["leases"]) == 3 and answer["detail"]["stopped"] == "allowance_full"
    assert answer["lease_id"] == answer["leases"][0]["lease_id"]
    slots = {(lease["prompt_index"], lease["slot"]) for lease in answer["leases"]}
    assert len(slots) == 3
    refused = _lease(client, "5A")
    assert refused["reason"] == "allowance_full" and refused["detail"]["allowance"] == 3


def test_a_request_without_a_count_answers_as_before(fake_r2):
    client, _ = _route(fake_r2, None, _Records())
    answer = _lease(client, "5A")
    assert answer["leased"] is True and answer["leases"] == []


def test_a_delivery_times_the_hotkeys_next_leases(fake_r2):
    clock = Clock()
    auction = _auction(clock)
    client, _ = _route(fake_r2, auction, _Records(), prompts=4)
    _ask(client, "5A", 0.2)
    for _ in range(3):
        lease = _lease(client, "5A")
        assert _submit_as(client, "5A", lease["prompt_index"], lease["slot"])["accepted"] is True
    assert len(auction._deliveries["5A"]) == 3


def test_a_period_pays_its_tokens_at_the_highest_lease_delivered_in_it():
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
    from tests.unit.test_corpus_period_settlement import GENESIS, Archives, Records, at, verdict

    records, archives = Records(), Archives()
    records.verdicts = {
        "a1": {**verdict("a1", "a", 1000, at(3)), "lease_price": 1e-5, "lease_debit": 0.0},
        "b1": {**verdict("b1", "b", 1000, at(3)), "lease_price": 2e-5, "lease_debit": 0.0},
    }
    settler = CorpusPeriodSettler(task_id="t", job_id="t", cap=0.5, records=records, archives=archives,
                                  oldest_pending=lambda: None, genesis=lambda: GENESIS,
                                  clock=lambda: at(5, 1000), max_price=1e-4)
    asyncio.run(settler.settle_once())
    (doc,) = archives.docs.values()
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(0.02), "b": pytest.approx(0.02)}
    assert records.state["period_price"][str(3)] == pytest.approx(2e-5)
