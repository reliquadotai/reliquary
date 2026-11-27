from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import bittensor as bt
import pytest

from reliquary.corpus.contract_trial import ContractTrial
from reliquary.corpus.order_contracts import ContractBook
from reliquary.protocol.corpus_submission import CorpusLeaseRequest
from reliquary.protocol.signatures import sign_corpus_lease


@pytest.fixture
def trial(tmp_path, request):
    terms = getattr(request, "param", {})
    now = [1_900_000_000]
    keys = [bt.Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob")]
    book = ContractBook(tmp_path / "trial.sqlite")
    for key, price in zip(keys, terms.get("prices", (1000, 2000)), strict=True):
        book.post_offer(
            "model|turn",
            key.ss58_address,
            key.ss58_address,
            price_micro=price,
            rate=terms.get("rate", 1),
            slots=2,
            at=now[0] - 5,
        )
    book.open_order(
        "order",
        "model|turn",
        opened_at=now[0],
        duration_s=100,
        target_rate=terms.get("rate", 1),
        max_tokens=terms.get("volume", 100),
        budget_rao=terms.get("budget", 150),
        ceiling_micro=max(terms.get("prices", (1000, 2000))),
        new_rate_cap=terms.get("rate", 1),
    )
    job = SimpleNamespace(
        job_id="order",
        prompt_start=0,
        prompt_count=4,
        slots_per_prompt=1,
        sampling=SimpleNamespace(max_new_tokens=terms.get("maximum", 30), n=1),
    )

    def start():
        return ContractTrial(ContractBook(book.path), job, clock=lambda: now[0])

    def signed_request(key, nonce):
        body = CorpusLeaseRequest(
            job_id="order", miner_hotkey=key.ss58_address, nonce=f"{nonce:032x}", issued_at=now[0], signature="unsigned"
        )
        body.signature = sign_corpus_lease(SimpleNamespace(hotkey=key), body)
        return body

    return start, signed_request, now, keys, book


def test_grant_receipt_restart_reassign_and_core_rewards(trial):
    start, request, now, (a, b), book = trial
    t = start()
    first_request = request(a, 1)
    first = t.request(first_request)
    assert first.leased and book.contract("order", a.ss58_address)["remaining_tokens"] == 70
    assert start().request(first_request) == first
    assert not t.request(request(b, 2)).leased
    second = t.request(request(a, 3))
    assert t.cancel(a.ss58_address, second.lease_id)
    assert not start().request(request(a, 3)).leased
    assert not t.receive(a.ss58_address, second.lease_id, "cancelled", 10)
    assert t.receive(a.ss58_address, first.lease_id, "receipt-a", 10)
    assert t.receive(a.ss58_address, first.lease_id, "receipt-a", 10)
    assert not t.receive(a.ss58_address, first.lease_id, "different", 10)
    assert not t.cancel(a.ss58_address, first.lease_id)
    now[0] += 5
    book.exit("order", a.ss58_address, at=now[0], reason="announced")
    assert book.contract("order", b.ss58_address)["price_micro"] == 2000
    start().verdict("receipt-a", passed=True)
    t.verdict("receipt-a", passed=True)
    third = t.request(request(b, 4))
    assert third.leased and t.receive(b.ss58_address, third.lease_id, "receipt-b", 30)
    spare = t.request(request(b, 5))
    assert spare.leased
    now[0] = int(spare.expires_at)
    assert not start().receive(b.ss58_address, spare.lease_id, "too-late", 20)
    t.verdict("receipt-b", passed=True)
    assert t.rewards() == pytest.approx({a.ss58_address: 10 / 1e9, b.ss58_address: 60 / 1e9}, abs=1e-18)
    assert book.contract("order", b.ss58_address)["remaining_tokens"] == 30


def test_rollback_rejection_and_concurrent_request(trial, monkeypatch):
    start, request, _clock, (a, _), book = trial
    t = start()
    body = request(a, 10)
    original_response = t._response

    def crash(row):
        raise RuntimeError("before response")

    monkeypatch.setattr(t, "_response", crash)
    with pytest.raises(RuntimeError, match="before response"):
        t.request(body)
    assert book.contract("order", a.ss58_address)["remaining_tokens"] == 100
    monkeypatch.setattr(t, "_response", original_response)
    with ThreadPoolExecutor(2) as pool:
        leases = list(pool.map(lambda _: start().request(body), range(2)))
    first = leases[0]
    assert first == leases[1] and first.leased
    assert book.contract("order", a.ss58_address)["remaining_tokens"] == 70
    with pytest.raises(ValueError, match="invalid trial receipt"):
        t.receive(a.ss58_address, first.lease_id, "over", 31)
    assert t.receive(a.ss58_address, first.lease_id, "invalid-work", 30)
    t.verdict("invalid-work", passed=False)
    assert book.contract("order", a.ss58_address)["remaining_tokens"] == 100
    assert start().rewards() == {}
    with pytest.raises(ValueError, match="conflicting verdict"):
        start().verdict("invalid-work", passed=True)
    assert start().request(request(a, 11)).leased


def test_full_reservation_before_checkout_and_one_concurrent_receipt(trial):
    start, request, _, (a, _), book = trial
    runner = start()
    for nonce in range(3):
        lease = runner.request(request(a, nonce))
        assert lease.leased
        if nonce == 0:
            with ThreadPoolExecutor(2) as pool:
                results = list(
                    pool.map(
                        lambda receipt, lease_id=lease.lease_id: start().receive(a.ss58_address, lease_id, receipt, 30),
                        ("left", "right"),
                    )
                )
            assert sorted(results) == [False, True]
            receipt = ("left", "right")[results.index(True)]
        else:
            receipt = f"accepted-{nonce}"
            assert runner.receive(a.ss58_address, lease.lease_id, receipt, 30)
        runner.verdict(receipt, passed=True)
    assert book.contract("order", a.ss58_address)["remaining_tokens"] == 10
    assert runner.request(request(a, 4)).reason == "no_reservation"
    assert runner.rewards() == pytest.approx({a.ss58_address: 90 / 1e9}, abs=1e-18)


@pytest.mark.parametrize(
    ("trial", "deliveries", "contractual", "valued"),
    [
        ({"prices": (3, 4), "rate": 10, "volume": 1000, "budget": 3, "maximum": 500}, [500, 500], 3, 2),
        ({"prices": (1_000_000, 2_000_000), "budget": 1000, "maximum": 1}, [1], 1000, 999),
    ],
    indirect=["trial"],
)
def test_binary_share_valuation_and_contract_round_once_per_hotkey(trial, deliveries, contractual, valued):
    start, request, _, (a, _), _book = trial
    runner = start()
    for index, tokens in enumerate(deliveries):
        lease = runner.request(request(a, index))
        receipt = str(index)
        assert runner.receive(a.ss58_address, lease.lease_id, receipt, tokens)
        runner.verdict(receipt, passed=True)
    assert runner.rewards() == {a.ss58_address: contractual / 1e9}
    assert runner.rao_breakdown() == {
        a.ss58_address: {"contractual_rao": contractual, "valued_rao": valued, "difference_rao": valued - contractual}
    }
