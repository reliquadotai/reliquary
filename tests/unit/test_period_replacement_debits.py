import pytest

from reliquary.corpus.order_contracts import ContractBook
from reliquary.corpus.replacement_debits import period_debit_rows
from reliquary.validator.corpus_periods import PERIOD_EPOCH


def test_archive_rows_require_end_of_period_owner_and_final_recovery(tmp_path):
    book = ContractBook(tmp_path / "book")
    start = int(PERIOD_EPOCH) + 2
    for miner, owner, bid in (("worker", "owner", 1000), ("other", "second", 3000)):
        book.post_offer(
            "model", miner, owner, price_micro=bid, rate=1, slots=1, at=start - 1
        )
    book.open_order(
        "order",
        "model",
        opened_at=start,
        duration_s=500,
        max_tokens=500,
        target_rate=1,
        budget_rao=1500,
        ceiling_micro=3000,
        new_rate_cap=1,
    )
    book.exit("order", "worker", at=start, reason="announced", event_id="replacement")
    events = book.debt_events("order")
    fields = {
        "order_id": "order",
        "work_period": 0,
        "owners_at_period_end": {"worker": "owner"},
        "recovered_rao": {"replacement": 300},
    }
    rows = period_debit_rows(events + events, **fields)
    assert len(rows) == 1
    assert (
        rows[0]["assessed_rao"] == 1000
        and rows[0]["recovered_at_settlement_rao"] == 300
    )
    assert rows[0]["coldkey_at_earn"] == "owner"
    assert period_debit_rows(events, **fields) == rows
    with pytest.raises(ValueError, match="owner"):
        period_debit_rows(
            events, **{**fields, "owners_at_period_end": {"worker": "different"}}
        )
    with pytest.raises(ValueError, match="recovery"):
        period_debit_rows(events, **{**fields, "recovered_rao": {}})
