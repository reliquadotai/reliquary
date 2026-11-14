"""Draw leases our side failed are never charged: no auction debt, no halved capacity, no unused-lease count.
Only the issuer's own records decide (its refused opens, the session book's finals), never the miner's word."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

pytest.importorskip("reliquary_sandbox")

from reliquary.corpus.draw_leases import DrawLeases, free_draw_slots  # noqa: E402
from reliquary.corpus.slots import SlotLedger
from reliquary.sandbox import corpus_lease_faults as faults
from reliquary.sandbox.sessions import OURS_REFUSALS, Refusal, SessionIssuer
from tests.unit.test_draw_leases import Clock, _job
from tests.unit.test_token_auction import ALLOWANCE_FULL, _auction, _fill


def _leases(fault, *, auction=None, clock=None):
    job = _job(prompts=10, slots=4)
    ledger = SlotLedger(prompt_start=10, prompt_count=10, slots_per_prompt=4)
    clock = clock or Clock()
    kw = {}
    if auction is not None:
        kw = dict(limit_per_hotkey=False,
                  on_unused=lambda lease, held: auction.lease_ended(lease.hotkey, lease.price, held))
    leases = DrawLeases(job, clock=clock, rng=random.Random(0), seconds=100.0, validator_fault=fault, **kw)
    return leases, free_draw_slots(job, ledger, lambda p, s: False), clock


def test_a_lease_our_side_failed_is_released_and_expires_uncharged():
    ended = []
    ours = {"a"}
    job = _job(prompts=10, slots=4)
    ledger = SlotLedger(prompt_start=10, prompt_count=10, slots_per_prompt=4)
    clock = Clock()
    leases = DrawLeases(job, clock=clock, rng=random.Random(0), seconds=100.0,
                        on_unused=lambda lease, held: ended.append((lease.hotkey, held)),
                        validator_fault=lambda lease: ("box_failed", "x") if lease.hotkey in ours else None)
    free = free_draw_slots(job, ledger, lambda p, s: False)
    mine, theirs = leases.grant("a", free), leases.grant("b", free)
    clock.now += 80
    assert leases.release("a", mine.lease_id) and leases.release("b", theirs.lease_id)
    assert ended == [("b", pytest.approx(0.8))]
    for hotkey in ("a", "a", "b"):
        leases.grant(hotkey, free)
    clock.now += 101
    leases.expire()
    assert ended[1:] == [("b", 1.0)]
    assert leases.cap("a") == 10 and leases.cap("b") < 10     # only b's expiry counts against it


def test_a_fault_check_that_raises_charges_as_before():
    def broken(lease):
        raise RuntimeError("book unreadable")

    ended = []
    job = _job(prompts=10, slots=4)
    ledger = SlotLedger(prompt_start=10, prompt_count=10, slots_per_prompt=4)
    leases = DrawLeases(job, clock=Clock(), rng=random.Random(0), seconds=100.0, validator_fault=broken,
                        on_unused=lambda lease, held: ended.append(lease.hotkey))
    lease = leases.grant("a", free_draw_slots(job, ledger, lambda p, s: False))
    leases.release("a", lease.lease_id)
    assert ended == ["a"]


def test_an_auction_lease_our_sandbox_failed_neither_owes_nor_halves_the_capacity():
    auction, clock, book = _auction(budget=100)
    auction.post_ask("5A", 0.2, 8)
    leases, free, lease_clock = _leases(lambda lease: ("open_refused_ours", "fleet full"), auction=auction)
    lease = leases.grant("5A", free, price=auction.params.share_of(0.2))
    lease_clock.now += 80                                # a late release: a miss, were it the miner's
    leases.release("5A", lease.lease_id)
    assert auction.debt_alpha("5A") == 0
    assert _fill(auction, book, "5A") == (8, ALLOWANCE_FULL)
    charged, _, charged_book = _auction(budget=100)
    charged.post_ask("5A", 0.2, 8)
    leases, free, lease_clock = _leases(None, auction=charged)
    lease = leases.grant("5A", free, price=charged.params.share_of(0.2))
    lease_clock.now += 80
    leases.release("5A", lease.lease_id)
    assert charged.debt_alpha("5A") > 0
    charged.take_debit("5A", 1.0)
    assert _fill(charged, charged_book, "5A") == (4, ALLOWANCE_FULL)     # the same release by the miner: halved


# ------------------------------------------------------------------ the evidence


def _record(session_id, *, prompt=12, issued_at=1000, state="closed", closed_status=None, ended_status=None,
            fault=None, job_id="job-1"):
    return SimpleNamespace(session_id=session_id, job_id=job_id, prompt_index=prompt, issued_at=issued_at,
                           state=state, closed_status=closed_status, ended_status=ended_status, fault=fault,
                           machine_id="m1")


class FakeIssuer:
    def __init__(self, records=(), refused=None):
        self.book = SimpleNamespace(of_hotkey=lambda hotkey: tuple(records))
        self._refused = refused

    def refused_ours(self, hotkey, job_id, prompt_index, since):
        return self._refused if self._refused is not None and self._refused[0] >= since else None


def _lease(prompt=12, granted_at=1000.0, hotkey="5A"):
    return SimpleNamespace(lease_id="f" * 32, hotkey=hotkey, prompt_index=prompt, granted_at=granted_at)


def _judge(records=(), refused=None, budget=None):
    return faults.corpus_lease_fault(FakeIssuer(records, refused), "job-1",
                                     budget=budget or faults.FaultBudget())(_lease())


def test_the_session_book_and_our_refusals_are_the_evidence():
    assert _judge(refused=(1005, "sandbox_saturated"))[0] == "open_refused_ours"
    box = _record("s2", issued_at=1010, closed_status="box_failed")
    assert _judge([box])[0] == "box_failed"
    assert _judge([_record("s3", issued_at=1010, state="voided")])[0] == "machine_drained"
    # A graded final last, a session before the grant, another prompt or job: the miner's.
    graded = _record("s4", issued_at=1020, closed_status="graded")
    assert _judge([box, graded]) is None
    assert _judge([_record("s5", issued_at=900, closed_status="box_failed")]) is None
    assert _judge([_record("s6", prompt=13, closed_status="box_failed")]) is None
    assert _judge([_record("s7", job_id="other", closed_status="box_failed")]) is None
    assert _judge() is None


def test_a_session_after_our_refusal_is_what_decides():
    """Refused once, then opened: the session that ran is judged, not the refusal (review of aa7adcd2)."""
    refused = (1005, "sandbox_saturated")
    expired = _record("s1", issued_at=1010, closed_status="expired")
    unsubmitted = _record("s2", issued_at=1010, state="closed_graded", closed_status="graded")
    assert _judge([expired], refused) is None
    assert _judge([unsubmitted], refused) is None
    assert _judge([_record("s3", issued_at=1010, closed_status="box_failed")], refused)[0] == "box_failed"
    # A session before the refusal, then nothing ran: the refusal stands.
    assert _judge([_record("s4", issued_at=1001, closed_status="expired")], refused)[0] == "open_refused_ours"
    # A refusal before the grant is no excuse; a lease that simply expired with nothing on record is the miner's.
    assert _judge([], (900, "sandbox_saturated")) is None


def test_only_the_grading_pool_abort_is_ours():
    pool = _record("s1", issued_at=1010, state="aborted", closed_status="aborted", fault="grading_pool")
    other = _record("s2", issued_at=1010, state="aborted", closed_status="aborted")
    assert _judge([pool])[0] == "aborted"
    assert _judge([other]) is None


def test_box_failed_is_free_a_bounded_number_of_times_a_day_per_hotkey_across_jobs(tmp_path):
    clock = Clock()
    box = _record("s2", issued_at=1010, closed_status="box_failed")
    budget = faults.FaultBudget(2, path=tmp_path / "budget.json", clock=clock)
    issuer = FakeIssuer([box])
    one = faults.corpus_lease_fault(issuer, "job-1", budget=budget)
    other = faults.corpus_lease_fault(FakeIssuer([_record("s9", job_id="job-2", issued_at=1010,
                                                          closed_status="box_failed")]), "job-2", budget=budget)
    assert one(_lease()) is not None and other(_lease()) is not None
    assert one(_lease()) is None                        # the hotkey's day is spent, whichever job
    assert one(_lease(hotkey="5B")) is not None
    # A restart reads the day back.
    again = faults.FaultBudget(2, path=tmp_path / "budget.json", clock=clock)
    assert faults.corpus_lease_fault(issuer, "job-1", budget=again)(_lease()) is None
    clock.now += 86401
    assert faults.corpus_lease_fault(issuer, "job-1", budget=again)(_lease()) is not None
    # A refused open is ours, never budgeted.
    assert _judge(refused=(1005, "sandbox_capacity"), budget=faults.FaultBudget(0)) is not None


def _issuer(*, self_caused=False):
    issuer = SessionIssuer.__new__(SessionIssuer)
    issuer._ours_refused, issuer._clock = {}, lambda: 1000.0
    issuer._self_caused = lambda hotkey, terms, refusal: self_caused
    return issuer


ENGAGEMENT = {"kind": "corpus", "job_id": "job-1", "prompt_index": 12}


def test_the_issuer_remembers_the_corpus_opens_it_refused_on_our_side():
    issuer = _issuer()
    issuer._note_ours("5A", ENGAGEMENT, Refusal("prompt_unavailable", {}))
    assert issuer.refused_ours("5A", "job-1", 12, since=0) is None           # the miner's own limits
    issuer._note_ours("5A", ENGAGEMENT, Refusal("sandbox_saturated", {}), terms=object())
    assert issuer.refused_ours("5A", "job-1", 12, since=0) == (1000, "sandbox_saturated")
    assert issuer.refused_ours("5A", "job-1", 12, since=1001) is None        # before the lease
    assert issuer.refused_ours("5B", "job-1", 12, since=0) is None
    issuer._note_ours("5A", {"kind": "rl_lease", "job_id": "job-1", "prompt_index": 13},
                      Refusal("sandbox_saturated", {}))
    assert issuer.refused_ours("5A", "job-1", 13, since=0) is None
    assert {"sandbox_saturated", "task_unplaceable", "store_unavailable"} <= OURS_REFUSALS
    assert "prompt_unavailable" not in OURS_REFUSALS and "live_cap" not in OURS_REFUSALS


def test_a_saturation_of_the_miners_own_making_is_not_ours():
    issuer = _issuer(self_caused=True)
    issuer._note_ours("5A", ENGAGEMENT, Refusal("sandbox_saturated", {}), terms=object())
    assert issuer.refused_ours("5A", "job-1", 12, since=0) is None


def test_a_quota_refusal_is_ours_only_within_the_hotkeys_own_allowance():
    """A hotkey over its own allowance while the fleet is fully allocated leases past its share: never free."""
    issuer = _issuer()
    issuer._note_ours("5A", ENGAGEMENT, Refusal("quota_live", {"live": 4, "allowance": 4, "max": 4}))
    issuer._note_ours("5A", ENGAGEMENT, Refusal("quota_pool", {"pool": "newcomer"}))
    assert issuer.refused_ours("5A", "job-1", 12, since=0) is None
    issuer._note_ours("5A", ENGAGEMENT, Refusal("quota_live", {"why": "fleet", "live": 1, "allowance": 4}))
    assert issuer.refused_ours("5A", "job-1", 12, since=0) == (1000, "quota_live")
