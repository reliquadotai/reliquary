import math

import pytest

from reliquary.corpus.audit_policy import AuditParams, MinerState, decision
from reliquary.corpus.contract_pacing import (
    audit_rate,
    contract_lease_seconds,
    decay_counts,
    paced_rate,
    reliability,
    tail_rescue_eligible,
)
from reliquary.corpus.draw_leases import DrawLeases
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401 (fixtures)
from tests.unit.test_draw_leases import _setup, _submit_as
from tests.unit.test_token_auction_routes import (
    _ask,
    _auction,
    _lease,
    _Records,
    _route,
)


def test_coldkey_confidence_does_not_override_compulsory_audits():
    opts = {"q_min": 0.02, "trust_tokens": 100_000}
    assert audit_rate(None, **opts) == audit_rate(0, **opts) == 1
    assert audit_rate(100_000, **opts) == pytest.approx(0.02 + 0.98 / math.e)
    q = audit_rate(1_000_000_000, **opts)
    assert q == 0.02
    assert audit_rate(1_000_000_000, evaluation=True, **opts) == 1
    assert audit_rate(1_000_000_000, required=True, **opts) == 1
    assess = {"params": AuditParams(q=q), "now": 5000, "received_at": 0, "recent_submissions": 1000,
                  "randomness_hex": "ff" * 32, "submission_id": "00" * 32}
    assert decision(MinerState(), **assess) == "audit"
    assert decision(MinerState(audited_passed=200, suspect_until=6000), **assess) == "audit"
    assert decision(MinerState(audited_passed=200, banned_until=6000), **assess) == "void_banned"
    assert decision(MinerState(audited_passed=200), **{**assess, "randomness_hex": None}) == "audit"


def test_reputation_prior_survives_decay_and_excess_work():
    assert reliability(0, 0) == 0.8
    delivered, committed = decay_counts(2_000_000, 10_000_000, 2 * 7 * 86400)
    assert (delivered, committed) == (500_000, 2_500_000)
    assert reliability(delivered, committed) == 8_500_000 / 12_500_000
    assert reliability(10_000_000, 2_000_000) == reliability(2_000_000, 2_000_000)
    with pytest.raises(ValueError):
        decay_counts(1, 1, -10)


def test_audit_pacer_short_and_sustained_slowdowns_and_queue_bound():
    sample = {"card_tokens_per_second": 1000, "queued_audit_tokens": 900_000, "audit_fraction": 0.2,
                  "offered_rate": 10000, "needed_rate": 10000}
    start = paced_rate(None, now=0, **sample)
    assert start.rate == 5000
    short = paced_rate(start, now=30, **{**sample, "card_tokens_per_second": 100})
    assert short.rate == start.rate
    sustained = paced_rate(short, now=301, **{**sample, "card_tokens_per_second": 100})
    assert sustained.rate == 4500
    assert paced_rate(sustained, now=310, **{**sample, "card_tokens_per_second": 100}).rate == 4500
    more = paced_rate(sustained, now=602, **{**sample, "card_tokens_per_second": 100})
    assert more.rate == 4050
    full_queue = paced_rate(more, now=610, **{**sample, "queued_audit_tokens": 2_000_000})
    assert full_queue.rate == 0
    assert paced_rate(full_queue, now=615, **{**sample, "queued_audit_tokens": 0,
                                             "needed_rate": 90, "offered_rate": 50}).rate == 50
    assert paced_rate(None, now=0, **{**sample, "card_tokens_per_second": 0}).rate == 0
    with pytest.raises(ValueError):
        paced_rate(short, now=5, **sample)


def test_short_leases_preserve_signed_episode_floor_and_tail_boundary():
    assert contract_lease_seconds(3600) == 900
    assert contract_lease_seconds(3600, expected_seconds=140) == 280
    assert contract_lease_seconds(7200, expected_seconds=140, episode_floor_seconds=4000) == 4000
    opts = {"total_slots": 100, "expired": True, "reliable": True}
    assert tail_rescue_eligible(remaining_slots=2, **opts)
    assert not tail_rescue_eligible(remaining_slots=3, **opts)
    assert not tail_rescue_eligible(remaining_slots=0, **opts)
    assert not tail_rescue_eligible(remaining_slots=1, **{**opts, "expired": False})
    assert not tail_rescue_eligible(remaining_slots=1, **{**opts, "reliable": False})


@pytest.mark.parametrize("winner", ["original", "replacement"])
def test_eligible_rescue_races_only_after_expiry_and_survives_restart(tmp_path, winner):
    path = tmp_path / "leases.json"
    def eligible(hotkey):
        return hotkey == "replacement"

    job, ledger, _, clock, book, free = _setup(prompts=1, slots=1, state_path=path,
                                             contract_tail_rescue=eligible)
    original = book.grant("original", free, price=1.0)
    clock.now += 99
    assert book.grant("replacement", free) is None
    clock.now += 1
    rescue = book.grant("replacement", free, price=2.0)
    restored = DrawLeases(job, clock=clock, seconds=100, state_path=path, contract_tail_rescue=eligible)
    slot = original.prompt_index, original.slot
    assert restored.held("original", *slot) == original
    assert restored.held("replacement", *slot) == rescue
    assert restored.refusal("original", *slot) is None
    assert restored.refusal("replacement", *slot) is None
    assert restored.refusal("unrelated", *slot) is not None
    assert restored.grant("unrelated", free) is None
    ledger.consume(original.prompt_index)
    restored.settle(*slot, winner)
    assert restored.grant("replacement", free) is None
    assert restored.held("original", *slot) is None and restored.held("replacement", *slot) is None


@pytest.mark.parametrize("rescue_permission", [False, None])
def test_untrusted_or_bulk_replacement_can_always_finish_but_excludes_late_submitter(rescue_permission):
    _, _, _, clock, book, free = _setup(prompts=1, slots=1, contract_tail_rescue=lambda _: rescue_permission)
    original = book.grant("original", free)
    clock.now += 70
    assert book.grant("replacement", free) is None
    clock.now += 30
    assert book.grant("replacement", free) is not None
    assert book.held("original", original.prompt_index, original.slot) is None
    assert book.refusal("original", original.prompt_index, original.slot) is not None


@pytest.mark.parametrize("winner, other", [("5A", "5B"), ("5B", "5A")])
def test_http_submission_after_rescue_records_only_the_winning_lease_price(fake_r2, winner, other):  # noqa: F811
    _, _, _, clock, book, _ = _setup(prompts=1, slots=1, limit_per_hotkey=False,
                                     contract_tail_rescue=lambda hotkey: hotkey == "5B")
    auction, records = _auction(clock), _Records()
    client, router = _route(fake_r2, auction, records, prompts=1)
    router.draw_leases["swe-v1"] = book
    auction.bind_holdings(book.counts)
    for hotkey, price in (("5A", 0.2), ("5B", 0.3)):
        assert _ask(client, hotkey, price)["accepted"]
    assert _lease(client, "5A")["leased"]
    clock.now += 100
    lease = _lease(client, "5B")
    assert lease["leased"]
    slot = lease["prompt_index"], lease["slot"]
    price = book.held(winner, *slot).price
    assert _submit_as(client, winner, *slot)["accepted"]
    assert not _submit_as(client, other, *slot)["accepted"]
    assert [record["lease_price"] for record in records.submissions.values()] == [price]
