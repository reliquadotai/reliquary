"""The async RL submission path (``rl_submit``): grade-all, lease closing, buffer intake, out-of-zone store and
sampled audit with graded sanctions, pay-eligibility events, quota accounting."""
from __future__ import annotations

import random

import pytest

from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.services.exploration import AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import BANNED, LeaseBook, LeaseBookConfig, LeaseRequest
from reliquary.validator.rl_submit import (
    BAN, FIRST_SCAN, FIRST_SCAN_FORFEITED, TRAINED, UNTRAINED, AuditResult, GroupGrader, RlSubmissions, Rollout,
    SubmitConfig,
)

ENV = "openmathinstruct"           # math_bft termination, boxed answers, binary rewards
NOW = 1_800_000_000.0
EOS = 1
CAP = 64                           # the env's token cap in these tests (truncation = prompt + completion >= CAP)


class Tokenizer:
    """One token per character (offset 10); ids below 10 are special (EOS = 1). No chat template: the rendered
    prompt is the problem's text."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i - 10) for i in ids if i >= 10)


class MathEnv:
    name = ENV

    def get_problem(self, idx):
        return {"prompt": f"Q{idx}", "ground_truth": "4"}


def text(s: str) -> tuple[int, ...]:
    return tuple(Tokenizer().encode(s))


RIGHT = Rollout(text("\\boxed{4}") + (EOS,))
WRONG = Rollout(text("\\boxed{5}") + (EOS,))
CUT = Rollout(text("x" * (CAP - 2)))   # no EOS, reaches the cap with the 2-token prompt "Qn"


def group(right: int, *, cut: int = 0) -> list[Rollout]:
    return [RIGHT] * right + [CUT] * cut + [WRONG] * (16 - right - cut)


@pytest.fixture(autouse=True)
def small_cap(monkeypatch):
    monkeypatch.setattr("reliquary.validator.admission.max_new_tokens_for_environment", lambda env: CAP)
    monkeypatch.setattr("reliquary.validator.rl_submit.max_new_tokens_for_environment", lambda env: CAP)


@pytest.fixture
def book(tmp_path):
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}, cooldown_of=lambda env: 2),
                     quota=QuotaController(QuotaConfig(initial=64, ceiling=64)), clock=lambda: NOW)
    book.adopt_checkpoint(0)
    yield book
    book.close()


def service(book, **config) -> RlSubmissions:
    grader = GroupGrader({ENV: MathEnv()}, Tokenizer(), eos_token_ids=(EOS,))
    return RlSubmissions(book, grader, config=SubmitConfig(**config), clock=lambda: NOW, rng=random.Random(0))


def lease(book, prompt, hotkey="A"):
    return book.grant([LeaseRequest(hotkey, ENV, (prompt,), 0)])[0].leases[0]


def kinds(subs):
    return [(e["lease_id"], e["kind"]) for e in subs.events()]


def test_an_in_zone_group_is_graded_by_the_validator_and_buffered_for_the_trainer(book):
    subs = service(book)
    held = lease(book, 7)
    answer = subs.submit(held.lease_id, "A", group(8), arrived_at=NOW + 5)
    assert answer == {"lease_id": held.lease_id, "verdict": "in_zone", "rewards": [1.0] * 8 + [0.0] * 8}
    assert book.lease(held.lease_id).state == "submitted"
    queued = subs.buffer.take(ENV, 10)
    assert len(queued) == 1
    g = queued[0]
    assert (g.lease_id, g.hotkey, g.checkpoint, g.nonce, g.arrived_at, g.prompt_idx) == (
        held.lease_id, "A", 0, held.nonce, NOW + 5, 7)
    assert g.prompt_tokens == text("Q7") and g.completions[0] == RIGHT.tokens and g.rewards[0] == 1.0
    stored = subs.group(held.lease_id)
    assert stored["verdict"] == "in_zone" and stored["training"] == "queued" and stored["audit"] is None
    assert stored["first_scan"] == 1 and stored["completion_tokens"] == 16 * len(RIGHT.tokens)
    assert book.quota.state("A")["counters"]["in_zone"] == 1 and book.in_use("A") == 0
    assert [p["state"] for p in book.prompt_state()["prompts"]] == ["cooldown"]
    assert subs.events() == [] and subs.audit_jobs(10) == []


@pytest.mark.parametrize(("rollouts", "verdict", "state"), [
    (group(16), "too_easy", "blacklisted"),
    (group(0), "too_hard", "cooldown"),
    (group(8, cut=2), "truncated", "cooldown"),
])
def test_out_of_zone_groups_are_stored_audited_in_probation_and_count_as_delivered(book, rollouts, verdict, state):
    subs = service(book)
    held = lease(book, 3)
    answer = subs.submit(held.lease_id, "A", rollouts)
    assert answer["verdict"] == verdict
    stored = subs.group(held.lease_id)
    assert stored["verdict"] == verdict and stored["audit"] == "pending" and stored["first_scan"] == 1
    assert subs.buffer.take(ENV, 10) == []
    assert book.quota.state("A")["counters"]["out_of_zone"] == 1
    assert book.prompt_state()["prompts"][0]["state"] == state
    if verdict == "truncated":
        assert stored["truncated"] == "[false, false, false, false, false, false, false, false, true, true, " \
                                      "false, false, false, false, false, false]"
    [job] = subs.audit_jobs(10)
    assert job.lease_id == held.lease_id and job.completions == tuple(r.tokens for r in rollouts)
    assert subs.events() == []                                   # drawn: payable only after the audit
    subs.report_audit(held.lease_id, AuditResult(passed=True))
    assert kinds(subs) == [(held.lease_id, FIRST_SCAN)] and subs.events()[0]["detail"]["audited"] is True
    assert subs.audit_jobs(10) == [] and subs.group(held.lease_id)["audit"] == "passed"


def test_a_lease_is_submitted_once_by_its_holder_before_expiry_with_every_index_and_resends_are_idempotent(book):
    subs = service(book)
    held = lease(book, 1)
    with pytest.raises(PermissionError):
        subs.submit(held.lease_id, "B", group(8))
    with pytest.raises(ValueError):
        subs.submit(held.lease_id, "A", group(8)[:15])             # missing index: refused, lease stays open
    with pytest.raises(LookupError, match="unknown"):
        subs.submit("nope", "A", group(8))
    assert book.lease(held.lease_id).state == "open"
    first = subs.submit(held.lease_id, "A", group(8))
    # a resend (lost answer) gets the original verdict back, and nothing is graded or recorded twice
    assert subs.submit(held.lease_id, "A", group(0)) == {**first, "resent": True}
    with pytest.raises(PermissionError):
        subs.submit(held.lease_id, "B", group(8))
    assert subs.buffer.depth() == {ENV: 1} and len(book.prompt_state()["scans"]) == 1
    late = lease(book, 2)
    with pytest.raises(LookupError, match="expired"):
        subs.submit(late.lease_id, "A", group(8), arrived_at=late.expires_at)
    assert book.lease(late.lease_id).state == "expired" and book.quota.state("A")["counters"]["expired"] == 1
    assert subs.group(late.lease_id) is None and len(subs.buffer.take(ENV, 10)) == 1


def test_a_group_refused_for_integrity_closes_the_lease_records_no_scan_and_halves_the_quota(book):
    subs = service(book)
    held = lease(book, 4)
    before = book.quota.quota("A", NOW)
    bad = group(8)
    bad[3] = Rollout(text("\\boxed{4}") + (EOS, EOS))                # EOS twice: bad termination
    answer = subs.submit(held.lease_id, "A", bad)
    assert answer == {"lease_id": held.lease_id, "verdict": "rejected", "rewards": [], "reason": "bad_termination"}
    assert book.lease(held.lease_id).state == "submitted"
    assert book.prompt_state()["scans"] == [] and subs.group(held.lease_id)["reason"] == "bad_termination"
    state = book.quota.state("A", NOW)
    assert state["counters"]["rejected"] == 1 and state["quota"] == before // 2
    stray = lease(book, 5)
    spanned = group(8)
    spanned[0] = Rollout(CUT.tokens, (0, 99))                         # a force span the BFT rule refuses
    assert subs.submit(stray.lease_id, "A", spanned)["reason"] == "tampered"


def test_a_grader_outage_leaves_the_lease_open_for_a_resend(book, monkeypatch):
    subs = service(book)
    held = lease(book, 6)

    def down(*args, **kwargs):
        raise GraderInfrastructureError("grader down")

    monkeypatch.setattr("reliquary.environment.registry.EnvironmentSpec.score_many", down)
    with pytest.raises(GraderInfrastructureError):
        subs.submit(held.lease_id, "A", group(8))
    assert book.lease(held.lease_id).state == "open" and subs.group(held.lease_id) is None


def test_probation_audits_every_group_then_the_rate_draws_after_submission(book):
    subs = service(book, probation_groups=2, audit_rate=0.0)
    first = [lease(book, p) for p in (10, 11)]
    for held in first:
        subs.submit(held.lease_id, "A", group(0))
    assert [subs.group(h.lease_id)["audit"] for h in first] == ["pending", "pending"]
    assert subs.in_probation("A")
    for held in first:
        subs.report_audit(held.lease_id, AuditResult(passed=True))
    assert not subs.in_probation("A")
    free = lease(book, 12)
    subs.submit(free.lease_id, "A", group(16))
    assert subs.group(free.lease_id)["audit"] == "not_drawn"
    assert kinds(subs)[-1] == (free.lease_id, FIRST_SCAN)            # not drawn: payable at once
    assert subs.events()[-1]["detail"] == {"audited": False, "fraction": 0.25}
    assert subs.in_probation("B")                                      # another hotkey is still new
    sampled = service(book, probation_groups=0, audit_rate=1.0)
    always = lease(book, 13)
    sampled.submit(always.lease_id, "A", group(0))
    assert sampled.group(always.lease_id)["audit"] == "pending"
    with sampled.book.lock:
        forced = book.db.execute("SELECT forced FROM rl_audits WHERE lease_id=?", (always.lease_id,)).fetchone()[0]
    assert forced == 0


def test_a_rescan_out_of_zone_is_never_payable(book):
    subs = service(book, probation_groups=0, audit_rate=0.0)
    first = lease(book, 20)
    subs.submit(first.lease_id, "A", group(0))
    book.adopt_checkpoint(1)
    book.adopt_checkpoint(2)                                           # the 0/16 cooldown (2 checkpoints) is over
    again = book.grant([LeaseRequest("A", ENV, (20,), 2)])[0].leases[0]
    subs.submit(again.lease_id, "A", group(0))
    assert subs.group(again.lease_id)["first_scan"] == 0
    assert kinds(subs) == [(first.lease_id, FIRST_SCAN)]


def test_a_deterministic_failure_bans_and_forfeits(book):
    subs = service(book)
    held = lease(book, 30)
    subs.submit(held.lease_id, "A", group(16))
    other = lease(book, 31)
    subs.report_audit(held.lease_id, AuditResult(passed=False, failure_class=AUDIT_FAILURE_DETERMINISTIC))
    assert kinds(subs) == [(held.lease_id, BAN), (held.lease_id, FIRST_SCAN_FORFEITED)]
    assert book.banned_until("A") == NOW + 86400
    assert book.grant([LeaseRequest("A", ENV, (32,), 0)])[0].refused == {32: BANNED}
    with pytest.raises(PermissionError, match="banned"):
        subs.submit(other.lease_id, "A", group(8))
    with pytest.raises(LookupError):
        subs.report_audit(held.lease_id, AuditResult(passed=True))     # concluded once
    classless = lease(book, 33, "B")
    subs.submit(classless.lease_id, "B", group(0))
    with pytest.raises(ValueError):
        subs.report_audit(classless.lease_id, AuditResult(passed=False))   # a failure needs its class


def test_a_statistical_failure_reopens_probation_and_recidivism_bans(book):
    subs = service(book, probation_groups=1, reprobation_passes=2, audit_rate=1.0, statistical_max=1)
    leases = [lease(book, p) for p in range(40, 45)]
    subs.submit(leases[0].lease_id, "A", group(0))
    subs.report_audit(leases[0].lease_id, AuditResult(passed=True))                 # probation done
    assert not subs.in_probation("A")
    subs.submit(leases[1].lease_id, "A", group(0))
    subs.report_audit(leases[1].lease_id, AuditResult(passed=False, failure_class=AUDIT_FAILURE_STATISTICAL))
    assert subs.in_probation("A") and book.banned_until("A") is None                 # re-probation, no ban
    assert kinds(subs) == [(leases[0].lease_id, FIRST_SCAN), (leases[1].lease_id, FIRST_SCAN_FORFEITED)]
    for held in leases[2:4]:
        subs.submit(held.lease_id, "A", group(0))
        subs.report_audit(held.lease_id, AuditResult(passed=True))
    assert not subs.in_probation("A")
    subs.submit(leases[4].lease_id, "A", group(0))
    subs.report_audit(leases[4].lease_id, AuditResult(passed=False, failure_class=AUDIT_FAILURE_STATISTICAL))
    assert book.banned_until("A") == NOW + 86400                                     # 2 > 1 in the window
    assert kinds(subs)[-2][1] == BAN


def test_an_inconclusive_audit_is_unpaid_and_unsanctioned(book):
    subs = service(book)
    held = lease(book, 50)
    subs.submit(held.lease_id, "A", group(0))
    subs.report_audit(held.lease_id, AuditResult(passed=None))
    assert kinds(subs) == [(held.lease_id, FIRST_SCAN_FORFEITED)] and book.banned_until("A") is None
    assert subs.in_probation("A") and subs.group(held.lease_id)["audit"] == "inconclusive"


def test_audits_run_through_the_proof_forward_past_probation_first(book):
    subs = service(book, probation_groups=1, audit_rate=1.0)
    veteran = lease(book, 60, "V")
    subs.submit(veteran.lease_id, "V", group(0))
    subs.report_audit(veteran.lease_id, AuditResult(passed=True))
    newcomer, sampled = lease(book, 61, "N"), lease(book, 62, "V")
    subs.submit(newcomer.lease_id, "N", group(0))
    subs.submit(sampled.lease_id, "V", group(0))
    assert [g.lease_id for g in subs.audit_jobs(10)] == [sampled.lease_id, newcomer.lease_id]
    seen = []

    def forward(g):
        seen.append(g.lease_id)
        return AuditResult(passed=True, rollout_match=(0.99,) * 16)

    assert subs.run_audits(forward, 10) == 2 and seen == [sampled.lease_id, newcomer.lease_id]
    assert subs.audit_jobs(10) == []


def test_a_raising_audit_forward_neither_stalls_the_queue_nor_runs_forever(book):
    subs = service(book)
    poison, honest = lease(book, 63), lease(book, 64)
    subs.submit(poison.lease_id, "A", group(0))
    subs.submit(honest.lease_id, "A", group(0))

    def forward(g):
        if g.lease_id == poison.lease_id:
            raise RuntimeError("CUDA out of memory")
        return AuditResult(passed=True, rollout_match=(0.99,) * 16)

    assert subs.run_audits(forward, 10, max_attempts=2) == 1                          # the other one still ran
    assert [g.lease_id for g in subs.audit_jobs(10)] == [poison.lease_id]
    assert subs.run_audits(forward, 10, max_attempts=2) == 1                          # second failure: concluded
    assert subs.audit_jobs(10) == [] and subs.group(poison.lease_id)["audit"] == "inconclusive"
    assert book.banned_until("A") is None


def test_the_trainer_reports_match_and_verdict_back_for_pay_and_sanctions(book):
    subs = service(book, probation_groups=0)
    good, stale, forged = (lease(book, p) for p in (70, 71, 72))
    for held in (good, stale, forged):
        subs.submit(held.lease_id, "A", group(8))
    taken = subs.buffer.take(ENV, 2)
    assert {g.lease_id for g in taken} <= {good.lease_id, stale.lease_id, forged.lease_id}
    with pytest.raises(ValueError):
        subs.report_training(good.lease_id, rollout_match=[0.99] * 15, accepted=True)
    subs.report_training(good.lease_id, rollout_match=[0.99] * 16, accepted=True)
    subs.report_training(stale.lease_id, rollout_match=[0.9] * 16, accepted=False)            # no evidence
    subs.report_training(forged.lease_id, rollout_match=[0.5] * 16, accepted=False,
                         failure_class=AUDIT_FAILURE_STATISTICAL)
    events = subs.events()
    assert [(e["lease_id"], e["kind"]) for e in events] == [
        (good.lease_id, TRAINED), (stale.lease_id, UNTRAINED), (forged.lease_id, UNTRAINED)]
    assert events[0]["detail"]["rollout_match"] == [0.99] * 16
    assert events[0]["detail"]["completion_tokens"] == [len(RIGHT.tokens)] * 16
    assert subs.group(good.lease_id)["training"] == "accepted"
    assert subs.group(forged.lease_id)["training"] == "rejected"
    assert subs.in_probation("A")                                     # the statistical failure re-opens it
    assert subs.buffer.depth() == {} and subs.buffer.take(ENV, 10) == []
    with pytest.raises(LookupError):
        subs.report_training(good.lease_id, rollout_match=[0.99] * 16, accepted=True)


def test_the_buffer_hands_out_freshest_first_and_survives_a_restart(book, tmp_path):
    subs = service(book)
    old = lease(book, 80)
    subs.submit(old.lease_id, "A", group(8), arrived_at=NOW)
    book.adopt_checkpoint(1)
    new = book.grant([LeaseRequest("A", ENV, (81,), 1)])[0].leases[0]
    subs.submit(new.lease_id, "A", group(8), arrived_at=NOW + 1)
    assert subs.buffer.depth() == {ENV: 2}
    assert [g.lease_id for g in subs.buffer.take(ENV, 1)] == [new.lease_id]
    subs.buffer.release([new.lease_id])
    assert [g.lease_id for g in subs.buffer.take(ENV, 5)] == [new.lease_id, old.lease_id]
    book.close()
    reopened = LeaseBook(tmp_path / "leases.sqlite3", book.config, quota=QuotaController(), clock=lambda: NOW)
    again = service(reopened)
    assert again.buffer.depth() == {ENV: 2}                          # taken groups come back at reopen
    reopened.close()


def test_the_token_codec_round_trips_and_bounds_what_it_inflates():
    import base64
    import zlib

    from reliquary.protocol.rl_submit import decode_tokens, encode_tokens, rollouts_digest

    tokens = (0, 1, 151_643, 2**32 - 1)
    assert decode_tokens(encode_tokens(tokens), max_tokens=4) == tokens
    with pytest.raises(ValueError):
        decode_tokens(encode_tokens(tokens), max_tokens=3)
    bomb = base64.b64encode(zlib.compress(b"\0" * 4_000_000)).decode()
    with pytest.raises(ValueError):
        decode_tokens(bomb, max_tokens=1000)
    for bad in ("", "!!", base64.b64encode(zlib.compress(b"abc")).decode()):
        with pytest.raises(ValueError):
            decode_tokens(bad, max_tokens=10)
    one = rollouts_digest([(0, encode_tokens(tokens), None)])
    assert one != rollouts_digest([(0, encode_tokens(tokens), (0, 1))]) != rollouts_digest([(1, encode_tokens(tokens), None)])
