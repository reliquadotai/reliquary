"""Proof deferral: a submission is graded and scored at admission, but not proven
until it is ranked high enough to win. Scoring + ranking reuse the merged
``difficulty_auction`` module (v2), and the same-prompt winner is resolved at
seal.
"""
import hashlib

from reliquary.constants import DIFFICULTY_AUCTION_DELTA
from reliquary.validator.batcher import PendingSubmission


def _pending(hotkey="a", prompt_idx=1, k=2, m=8, drand_round=1):
    return PendingSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        request=None,
        rewards=[1.0] * k + [0.0] * (m - k),
        drand_round=drand_round,
        merkle_root=hotkey.encode().ljust(32, b"\x00"),
        selection_digest=hotkey.encode().ljust(32, b"\x00"),
        prompt_content_sha256=hashlib.sha256(
            f"prompt:{prompt_idx}".encode()
        ).hexdigest(),
        target_content_sha256=hashlib.sha256(b"target").hexdigest(),
    )


def test_pending_submission_is_scored_at_admission():
    """Scoring is cheap (it only needs the rewards), so it happens before the GPU
    ever sees the submission — that is what lets us rank before proving. Under the
    difficulty score v(k)=std·(1-mean)^δ the peak is at k=2."""
    assert _pending(k=2).value > _pending(k=6).value


def test_pending_submission_ranks_in_the_auction():
    """It must satisfy the duck-type the difficulty-auction ``_rank_key`` consumes
    (hotkey, prompt_idx, drand_round, selection_digest), so the same ranking code
    works on unproven candidates."""
    from reliquary.validator.difficulty_auction import _rank_key, difficulty_score

    hard = _pending(hotkey="hard", prompt_idx=1, k=2)
    easy = _pending(hotkey="easy", prompt_idx=2, k=6)

    ranked = sorted(
        (
            (p, difficulty_score(p.rewards, delta=DIFFICULTY_AUCTION_DELTA))
            for p in (easy, hard)
        ),
        key=_rank_key,
    )

    assert [p.hotkey for p, _ in ranked] == ["hard", "easy"]


def test_accept_does_not_touch_the_gpu():
    """Admission must be proof-free. If the GPU is called during accept, the
    whole design collapses — we would be proving ~69 submissions per window."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    calls = []

    def _exploding_proof(*a, **kw):
        calls.append(1)
        raise AssertionError("GRAIL must not run during admission")

    b = _make_batcher(verify_commitment_proofs_fn=_exploding_proof)
    resp = b.accept_submission(_request(prompt_idx=7, hotkey="miner"))

    assert resp.accepted is True
    assert calls == []
    assert len(b.pending_submissions()) == 1
    assert b.pending_submissions()[0].value > 0.0   # graded + scored


def test_verify_expensive_runs_the_proof_and_returns_a_valid_submission():
    import torch
    from reliquary.validator.verifier import ProofResult
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    calls = []

    def _counting_grail(*a, **kw):
        calls.append(1)
        return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))

    b = _make_batcher(verify_commitment_proofs_fn=_counting_grail)
    b.accept_submission(_request(prompt_idx=7, hotkey="miner"))
    pending = b.pending_submissions()[0]

    proven = b._verify_expensive(pending)

    assert proven is not None
    assert proven.hotkey == "miner"
    assert proven.prompt_idx == pending.prompt_idx
    # The GPU proof actually ran — once per rollout. A _verify_expensive that
    # silently skipped the proof would still satisfy the assertions above.
    assert len(calls) == len(pending.request.rollouts)


def test_verify_expensive_reject_charges_debt_archives_and_redacts_sketch():
    """The task's core invariant: a submission rejected INSIDE _verify_expensive
    still charges per-hotkey proof-failure debt, still lands in
    rejected_submissions, and still has sketch_diff_max redacted to None. The
    proof runs at seal now, but the reject bookkeeping must be unchanged."""
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _make_batcher, _request,
    )

    b = _make_batcher(verify_commitment_proofs_fn=_always_false_grail)
    b.accept_submission(_request(prompt_idx=7, hotkey="cheater"))
    pending = b.pending_submissions()[0]

    assert b._verify_expensive(pending) is None
    assert b.proof_failure_debt("cheater") == 1
    assert len(b.rejected_submissions) == 1
    rejected = b.rejected_submissions[0]
    assert rejected.hotkey == "cheater"
    assert rejected.reason == "grail_fail"
    # Anti-tuning: the GRAIL sketch diff is never surfaced to miners.
    assert rejected.sketch_diff_max is None


def test_admitted_submissions_land_in_pending_not_valid():
    """Admission grades and scores into the PENDING pool; proofs run at seal, so
    _valid stays empty during the window. Reaching B distinct prompts does NOT
    seal — the window is time-boxed on the collection deadline."""
    from reliquary.constants import B_BATCH
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"hk{i}"))

    # No count-based seal: proofs are deferred and the window stays open.
    assert b.is_sealed() is False
    assert b._valid == []
    assert len(b.pending_submissions()) == B_BATCH


def test_valid_submissions_at_decision_reports_pending_not_valid():
    """Miners read this telemetry to know how many submissions were admitted by
    decision time. Proofs run at seal, so valid_count is 0 during the window;
    reporting it would lie. The field must ride the pending count."""
    from reliquary.validator.observability import SubmitTelemetry
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=7, hotkey="a"))
    b.accept_submission(_request(prompt_idx=8, hotkey="b"))

    tel = SubmitTelemetry.from_request(
        _request(prompt_idx=9, hotkey="c"), t_arrival=0.0
    )
    tel.refresh_from_batcher(b, at_decision=True)

    assert b.valid_count == 0            # nothing proven mid-window
    assert tel.valid_submissions_at_decision == 2


def test_valid_submissions_at_arrival_reports_pending_not_valid():
    """The arrival branch has the same flaw as the decision branch: valid_count
    is 0 for the whole window under deferred proving, so the arrival count would
    log a permanent 0. It must also ride the pending (graded, unproven) count."""
    from reliquary.validator.observability import SubmitTelemetry
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=7, hotkey="a"))
    b.accept_submission(_request(prompt_idx=8, hotkey="b"))

    tel = SubmitTelemetry.from_request(
        _request(prompt_idx=9, hotkey="c"), t_arrival=0.0
    )
    tel.refresh_from_batcher(b, at_decision=False)

    assert b.valid_count == 0            # nothing proven mid-window
    assert tel.valid_submissions_at_arrival == 2


def test_state_reports_admitted_submissions_not_proven_ones():
    """Miners poll /state and act on ``valid_submissions``. Proofs now run at
    seal, so reading ``_valid`` would report 0 for the whole window."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=7, hotkey="a"))
    b.accept_submission(_request(prompt_idx=8, hotkey="b"))

    state = b.get_state()

    assert b._valid == []                    # nothing proven yet
    assert state.valid_submissions == 2      # but /state must not lie


def test_state_wire_contract_is_unchanged():
    """GrpoBatchState is extra="forbid" with a strict miner-side parse: the
    pending count must ride the EXISTING field, not a new one."""
    from reliquary.protocol.submission import GrpoBatchState
    from tests.unit.test_grpo_window_batcher import _make_batcher

    fields = set(GrpoBatchState.model_fields)

    assert "valid_submissions" in fields
    assert not fields & {"pending_submissions", "pending_count"}
    assert set(_make_batcher().get_state().model_dump()) == fields


def test_decision_ts_is_stamped_at_admission_not_at_proof():
    """The pre-generation forensic metric is arrival_ts - (decision_ts -
    response_time). A seal-time decision_ts silently breaks it."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    clock = [1000.0]
    b = _make_batcher(wall_clock_fn=lambda: clock[0])
    b.accept_submission(_request(prompt_idx=7, hotkey="miner"))
    pending = b.pending_submissions()[0]

    clock[0] = 1600.0  # the proof runs minutes later, at seal
    proven = b._verify_expensive(pending)

    assert pending.decision_ts == 1000.0
    assert proven.decision_ts == 1000.0


def test_decision_ts_of_a_proof_stage_reject_is_the_admission_instant():
    """Rejected submissions are archived with the same forensic fields, so a
    gate that now fires at seal must still report when the miner was seen."""
    from reliquary.constants import CHALLENGE_K
    from tests.unit.test_grpo_window_batcher import (
        _grail_with_logits, _make_batcher, _ModelStubWithVocab, _request,
    )

    seq_len = CHALLENGE_K + 4
    clock = [1000.0]
    b = _make_batcher(
        model=_ModelStubWithVocab(),
        verify_commitment_proofs_fn=_grail_with_logits(seq_len),
        wall_clock_fn=lambda: clock[0],
    )
    req = _request()
    # Last token != 99 (EOS) and no cap hit → BAD_TERMINATION, a proof-stage gate.
    req.rollouts[0].commit["tokens"] = list(range(seq_len))
    req.rollouts[0].tokens = req.rollouts[0].commit["tokens"]
    b.accept_submission(req)

    clock[0] = 1600.0
    assert b._verify_expensive(b.pending_submissions()[0]) is None

    rejected = b.rejected_submissions[0]
    assert rejected.reason == "bad_termination"
    assert rejected.decision_ts == 1000.0


# --------------------------- prove top-down ---------------------------


def test_proving_stops_once_b_submissions_pass():
    """The GPU saving. Distinct arrival tiers: we must not prove candidate 9
    when 8 earlier-tier candidates have already passed. (Candidates in ONE
    tier are all proven — the fair-split pays them all — so this test gives
    each candidate its own arrival round.)"""
    from reliquary.constants import B_BATCH, M_ROLLOUTS
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_proof(commit, model, randomness):
        proofs.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_proof)
    for i in range(12):
        _accept(b, _request(prompt_idx=i, hotkey=f"m{i}"), arrival_round=100 + i)

    b.seal_batch()

    assert len(b.valid_submissions()) == B_BATCH
    assert b.proof_attempts == B_BATCH                     # 8 candidates, NOT 12
    # The GRAIL proof runs once per rollout, so the GPU bill is the 8 winners'
    # rollouts only — candidates 9..12 never reach the model. The forensic
    # sample is disabled for now (FORENSIC_SAMPLE_PER_WINDOW=0), so no extras.
    assert len(proofs) == B_BATCH * M_ROLLOUTS


def test_failed_proof_promotes_the_next_ranked():
    """Promote-on-failure: a fabricated group tops the ranking (it names its own
    score), fails the proof, and the honest submission behind it takes the slot."""
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher,
        _request_with_prompt_unique_tokens,
    )

    faker_prompt, honest_prompt = 1, 2

    def _fail_only_the_faker(commit, model, randomness):
        # The commit carries no hotkey (CommitModel is extra="forbid"), but this
        # helper keys every token on prompt_idx, so the group is identifiable.
        if commit["tokens"][0] // 100 == faker_prompt:
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_only_the_faker)
    # k=2 is the peak of v(k): the faker hand-writes the top-ranked reward vector
    # and so is proven FIRST, ahead of the honest k=4 group.
    b.accept_submission(_request_with_prompt_unique_tokens(
        prompt_idx=faker_prompt, hotkey="faker",
        rewards=[1.0, 1.0] + [0.0] * 6,
    ))
    b.accept_submission(_request_with_prompt_unique_tokens(
        prompt_idx=honest_prompt, hotkey="honest",
        rewards=[1.0] * 4 + [0.0] * 4,
    ))
    assert (
        b.pending_submissions()[0].value > b.pending_submissions()[1].value
    ), "the fabricated group must outrank the honest one for this test to bite"

    b.seal_batch()

    assert [s.hotkey for s in b.valid_submissions()] == ["honest"]
    assert b.proof_failure_debt("faker") == 1


def test_fabricated_groups_do_not_starve_honest_fill():
    """Many fabricated k=2 groups (the score peak) from DISTINCT hotkeys rank
    above the honest k=4 groups and each fails GRAIL. Promote-on-failure keeps
    going past them and the honest groups still fill the batch. The v2 global
    proof budget is the graded-pool ceiling (96), well above this pool, so it
    does not cut the loop off before the honest groups are reached."""
    from reliquary.constants import B_BATCH
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher,
        _request_with_prompt_unique_tokens,
    )

    n_fakes = 20
    # Prompt indices stay small so prompt_idx*100 tokens fit the test model vocab.
    honest_prompts = set(range(50, 50 + B_BATCH))

    def _fail_the_fakes(commit, model, randomness):
        prompt_idx = commit["tokens"][0] // 100
        if prompt_idx in honest_prompts:
            return _always_true_grail(commit, model, randomness)
        return _always_false_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_the_fakes)
    for i in range(n_fakes):  # distinct hotkeys, so the per-hotkey cap never bites
        b.accept_submission(_request_with_prompt_unique_tokens(
            prompt_idx=i, hotkey=f"fake{i}", rewards=[1.0, 1.0] + [0.0] * 6,
        ))
    for p in honest_prompts:
        b.accept_submission(_request_with_prompt_unique_tokens(
            prompt_idx=p, hotkey=f"honest{p}", rewards=[1.0] * 4 + [0.0] * 4,
        ))

    b.seal_batch()

    winners = {s.hotkey for s in b.valid_submissions()}
    assert winners == {f"honest{p}" for p in honest_prompts}


def test_single_hotkey_griefer_is_capped_by_per_hotkey_failures():
    """The per-hotkey half of the griefer bound. One hotkey flooding fabricated
    distinct-prompt groups (each ranks at the top by construction) is proven at
    most MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW times. The debt the
    failed proofs charge locks the hotkey out of the remaining attempts, so the
    pool size does not matter."""
    from reliquary.constants import (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW,
    )
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_false_grail(commit, model, randomness):
        proofs.append(1)
        return _always_false_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_false_grail)
    # Many more distinct-prompt groups than the per-hotkey cap, all one hotkey.
    for i in range(20):
        b.accept_submission(_request(prompt_idx=i, hotkey="griefer"))

    b.seal_batch()

    assert b.valid_submissions() == []
    # The ranked pass stops at the per-hotkey debt cap; the forensic sample is
    # disabled (FORENSIC_SAMPLE_PER_WINDOW=0), so no extra proofs.
    assert len(proofs) == MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
    assert b.proof_failure_debt("griefer") == (
        MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
    )


def test_global_proof_budget_bounds_a_multi_hotkey_flood():
    """The v2 global proof budget (spec §2.3): proving cannot exceed the graded
    pool ceiling. With the ceiling monkeypatched below the fabricated population,
    the ranked pass stops and the window advances with a shortfall rather than
    proving the whole flood."""
    import reliquary.validator.batcher as batcher_mod
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_false_grail(commit, model, randomness):
        proofs.append(1)
        return _always_false_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_false_grail)
    # Distinct hotkeys so the per-hotkey cap never bites; only the global
    # ceiling can stop the flood.
    for i in range(20):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"fake{i}"))

    original = batcher_mod.MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    batcher_mod.MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 5
    try:
        b.seal_batch()
    finally:
        batcher_mod.MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = original

    assert b.valid_submissions() == []
    assert b.proof_attempts == 5           # stopped at the global ceiling
    assert len(proofs) == 5


def test_forensic_sample_disabled_without_seal_randomness():
    """No post-deadline entropy (mock / no-drand) → no sample: only the 8 winners'
    rollouts are proven and forensic_sample stays empty."""
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting(commit, model, randomness):
        proofs.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting)
    assert b.seal_randomness == ""              # unset by default
    # Distinct arrival tiers, so exactly 8 winners exist (one shared tier
    # would be boundary-fair-split and proven in full).
    for i in range(20):
        _accept(b, _request(prompt_idx=i, hotkey=f"m{i}"), arrival_round=100 + i)

    b.seal_batch()

    assert len(b.valid_submissions()) == 8
    assert len(proofs) == 8 * M_ROLLOUTS       # only the 8 winners' rollouts
    assert b.forensic_sample == []


def test_forensic_sample_watches_next_ranked_non_winners():
    """The next-ranked content-unique groups are the research counterfactuals."""
    from reliquary.constants import FORENSIC_SAMPLE_PER_WINDOW
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    def _make(seal_rand):
        b = _make_batcher(verify_commitment_proofs_fn=_always_true_grail)
        b.seal_randomness = seal_rand
        for i in range(20):
            _accept(b, _request(prompt_idx=i, hotkey=f"m{i}"),
                    arrival_round=100 + i)
        b.seal_batch()
        return b

    b = _make("beacon-round-777")
    winners = {s.hotkey for s in b.valid_submissions()}
    assert len(b.forensic_sample) == FORENSIC_SAMPLE_PER_WINDOW
    watched = {r.hotkey for r in b.forensic_sample}
    assert watched.isdisjoint(winners)          # only non-winners are sampled

    # With no exact score/arrival tie in this fixture, different seal entropy
    # leaves the same next-ranked research shortlist.
    other = {r.hotkey for r in _make("beacon-round-999").forensic_sample}
    assert watched == other == {"m8", "m9"}


def test_forensic_sample_failure_cannot_abort_sealing():
    """Observational proofs fail open after ranked winners are established."""
    from reliquary.constants import B_BATCH, FORENSIC_SAMPLE_PER_WINDOW
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    b = _make_batcher(verify_commitment_proofs_fn=_always_true_grail)
    b.seal_randomness = "beacon-round-failure"
    for i in range(B_BATCH + FORENSIC_SAMPLE_PER_WINDOW + 4):
        _accept(b, _request(prompt_idx=i, hotkey=f"m{i}"), arrival_round=100 + i)

    original_verify = b._verify_expensive
    calls = 0

    def _fail_after_ranked_winners(pending):
        nonlocal calls
        calls += 1
        if calls > B_BATCH:
            raise RuntimeError("synthetic forensic failure")
        return original_verify(pending)

    b._verify_expensive = _fail_after_ranked_winners
    batch, _ = b.seal_batch()

    assert len(batch) == B_BATCH
    assert len(b.valid_submissions()) == B_BATCH
    assert len(b.forensic_sample) == FORENSIC_SAMPLE_PER_WINDOW
    assert all(result.passed is None for result in b.forensic_sample)
    assert b.forensic_proof_errors_by_type == {
        "RuntimeError": FORENSIC_SAMPLE_PER_WINDOW,
    }


def test_score_ranks_only_inside_the_calibrated_sigma_band():
    """The auction ranks eligible groups but does not pay k=1 until an
    independent grader clears the false-negative floor. The calibrated steady
    state band remains k=2..6; k=1, k=7, and unanimous groups are out of zone."""
    from reliquary.validator.batcher import RejectReason
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    r1 = b.accept_submission(
        _request(prompt_idx=1, hotkey="k1", rewards=[1.0] + [0.0] * 7)
    )
    assert r1.accepted is False and r1.reason == RejectReason.OUT_OF_ZONE
    r7 = b.accept_submission(
        _request(prompt_idx=2, hotkey="k7", rewards=[1.0] * 7 + [0.0])
    )
    assert r7.accepted is False and r7.reason == RejectReason.OUT_OF_ZONE

    assert b.accept_submission(
        _request(prompt_idx=5, hotkey="k2", rewards=[1.0] * 2 + [0.0] * 6)
    ).accepted
    assert b.accept_submission(
        _request(prompt_idx=6, hotkey="k6", rewards=[1.0] * 6 + [0.0] * 2)
    ).accepted
    assert len(b.pending_submissions()) == 2

    r8 = b.accept_submission(_request(prompt_idx=3, hotkey="k8", rewards=[1.0] * 8))
    assert r8.accepted is False and r8.reason == RejectReason.OUT_OF_ZONE
    r0 = b.accept_submission(_request(prompt_idx=4, hotkey="k0", rewards=[0.0] * 8))
    assert r0.accepted is False and r0.reason == RejectReason.OUT_OF_ZONE


# ---------------- speed-ranked auction (score, arrival_round) ----------------

import pytest

from reliquary.constants import B_BATCH


def _telemetry(prompt_idx, hotkey, arrival_round, arrival_ts=0.0):
    from reliquary.validator.observability import SubmitTelemetry
    return SubmitTelemetry(
        window_n=500, prompt_idx=prompt_idx, hotkey=hotkey,
        merkle_root="00" * 32, protocol_version=2,
        submitted_drand_round=arrival_round, t_arrival=arrival_ts,
        prompt_hash_lead="", merkle_root_lead="",
        precommit_arrival_ts=arrival_ts,
        arrival_drand_round=arrival_round,
    )


def _accept(b, req, arrival_round, arrival_ts=0.0):
    resp = b.accept_submission(
        req,
        telemetry=_telemetry(
            req.prompt_idx, req.miner_hotkey, arrival_round, arrival_ts
        ),
    )
    assert resp.accepted is True


def _shift_tokens(req, offset):
    """Mark a request's rollouts so a test proof fn can target it."""
    for rollout in req.rollouts:
        tokens = [t + offset for t in rollout.tokens]
        rollout.tokens = tokens
        rollout.commit["tokens"] = tokens
    return req


def test_equal_score_earlier_arrival_ranks_first():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    _accept(b, _request(prompt_idx=1, hotkey="slow"), arrival_round=105)
    _accept(b, _request(prompt_idx=2, hotkey="fast"), arrival_round=103)
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["fast"]["rank"] < rows["slow"]["rank"]
    assert rows["fast"]["tier"] < rows["slow"]["tier"]
    assert rows["fast"]["arrival_round_source"] == "arrival"


def test_score_dominates_speed():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    _accept(b, _request(prompt_idx=1, hotkey="fast-easy",
                        rewards=[1.0] * 6 + [0.0] * 2), arrival_round=100)
    _accept(b, _request(prompt_idx=2, hotkey="slow-hard",
                        rewards=[1.0] * 2 + [0.0] * 6), arrival_round=199)
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["slow-hard"]["rank"] == 1


def test_speed_decides_the_last_slot_between_equal_scores():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH):
        _accept(b, _request(prompt_idx=i, hotkey=f"fast{i}"), arrival_round=103)
    _accept(b, _request(prompt_idx=99, hotkey="slow"), arrival_round=104)
    _batch, rewards = b.seal_batch()

    assert "slow" not in rewards
    assert b.proof_attempts == B_BATCH        # the losing tier is never proven
    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["slow"]["status"] == "not_needed"
    assert rows["slow"]["proof_attempted"] is False


def test_no_telemetry_falls_back_to_submitted_round():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    slow = _request(prompt_idx=1, hotkey="slow")
    slow.drand_round = 105
    fast = _request(prompt_idx=2, hotkey="fast")
    fast.drand_round = 103
    assert b.accept_submission(slow).accepted
    assert b.accept_submission(fast).accepted
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["fast"]["rank"] < rows["slow"]["rank"]
    assert rows["fast"]["arrival_round_source"] == "submitted_fallback"


def test_boundary_tier_proves_only_candidates_that_can_win():
    """Two remaining slots in a three-candidate tier produce two winners."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH - 2):
        _accept(b, _request(prompt_idx=i, hotkey=f"a{i}"), arrival_round=100)
    for j in range(3):
        _accept(b, _request(prompt_idx=10 + j, hotkey=f"b{j}"), arrival_round=101)
    _accept(b, _request(prompt_idx=20, hotkey="late"), arrival_round=102)
    b.seal_batch()

    assert b.proof_attempts == B_BATCH
    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["late"]["proof_attempted"] is False
    boundary = [rows[f"b{j}"] for j in range(3)]
    assert sum(row["selected"] for row in boundary) == 2
    assert sum(row["proof_attempted"] for row in boundary) == 2
    assert sum(row["status"] == "not_needed" for row in boundary) == 1


def test_prompt_falls_to_next_tier_when_winning_tier_fails():
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    def _fail_marked(commit, model, randomness):
        if commit["tokens"][0] >= 1000:
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_marked)
    _accept(b, _shift_tokens(_request(prompt_idx=5, hotkey="faker"), 1000),
            arrival_round=100)
    _accept(b, _request(prompt_idx=5, hotkey="honest"), arrival_round=101)
    _batch, rewards = b.seal_batch()

    assert [s.hotkey for s in b.valid_submissions()] == ["honest"]
    assert rewards == {"honest": pytest.approx(1.0 / B_BATCH)}
    assert b.proof_failure_debt("faker") == 1


def test_same_prompt_same_tier_has_one_full_slot_winner():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for hk in ("op-a", "op-b"):
        _accept(b, _request(prompt_idx=7, hotkey=hk), arrival_round=100)
    batch, rewards = b.seal_batch()

    share = 1.0 / B_BATCH
    assert len(rewards) == 1
    assert next(iter(rewards.values())) == pytest.approx(share)
    assert len(b.valid_submissions()) == 1
    assert len(batch) == 1
    rows = {row["hotkey"]: row for row in b.auction_candidates}
    assert sum(row["selected"] for row in rows.values()) == 1
    assert sum(
        row["status"] == "same_prompt_superseded" for row in rows.values()
    ) == 1


def test_same_prompt_split_survivor_takes_full_share():
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    def _fail_marked(commit, model, randomness):
        if commit["tokens"][0] >= 1000:
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_marked)
    _accept(b, _request(prompt_idx=7, hotkey="honest"), arrival_round=100)
    _accept(b, _shift_tokens(_request(prompt_idx=7, hotkey="cheat"), 1000),
            arrival_round=100)
    _batch, rewards = b.seal_batch()

    assert rewards == {"honest": pytest.approx(1.0 / B_BATCH)}


def test_same_content_different_prompt_indices_share_one_slot():
    from tests.unit.test_grpo_window_batcher import FakeEnv, _make_batcher, _request

    class DuplicateContentEnv(FakeEnv):
        def get_problem(self, idx):
            prompt = "same canonical problem" if idx in {7, 8} else f"p{idx}"
            return {"prompt": prompt, "ground_truth": "a", "id": f"pid-{idx}"}

    b = _make_batcher(env=DuplicateContentEnv())
    _accept(b, _request(prompt_idx=7, hotkey="first"), arrival_round=100)
    _accept(b, _request(prompt_idx=8, hotkey="second"), arrival_round=100)
    batch, rewards = b.seal_batch()

    assert len(batch) == 1
    assert len(rewards) == 1
    assert next(iter(rewards.values())) == pytest.approx(1.0 / B_BATCH)
    rows = {row["hotkey"]: row for row in b.auction_candidates}
    assert sum(row["selected"] for row in rows.values()) == 1
    assert sum(
        row["status"] == "same_content_superseded"
        for row in rows.values()
    ) == 1
    assert b.content_selection["content_alignment_ok"] is True


def test_content_cooldown_blocks_duplicate_under_new_prompt_index():
    from reliquary.validator.cooldown import ContentCooldownMap
    from tests.unit.test_grpo_window_batcher import FakeEnv, _make_batcher, _request

    class DuplicateContentEnv(FakeEnv):
        def get_problem(self, idx):
            return {"prompt": "same canonical problem", "ground_truth": "a"}

    shared = ContentCooldownMap(cooldown_windows=1_000_000)
    first = _make_batcher(
        env=DuplicateContentEnv(), content_cooldown_map=shared
    )
    _accept(first, _request(prompt_idx=7, hotkey="winner"), arrival_round=100)
    first.seal_batch()

    second = _make_batcher(
        env=DuplicateContentEnv(),
        content_cooldown_map=shared,
        window_start=501,
    )
    _accept(second, _request(
        prompt_idx=8, hotkey="duplicate", window_start=501
    ), arrival_round=101)
    batch, rewards = second.seal_batch()

    assert batch == []
    assert rewards == {}
    assert second.auction_candidates[0]["status"] == "content_in_cooldown"


def test_boundary_tier_pays_only_selected_training_groups():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH - 2):
        _accept(b, _request(prompt_idx=i, hotkey=f"a{i}"), arrival_round=100)
    for j in range(3):
        _accept(b, _request(prompt_idx=10 + j, hotkey=f"b{j}"), arrival_round=101)
    batch, rewards = b.seal_batch()

    share = 1.0 / B_BATCH
    for i in range(B_BATCH - 2):
        assert rewards[f"a{i}"] == pytest.approx(share)
    boundary_rewards = {
        hotkey: amount
        for hotkey, amount in rewards.items()
        if hotkey.startswith("b")
    }
    assert len(boundary_rewards) == 2
    assert all(amount == pytest.approx(share) for amount in boundary_rewards.values())
    assert sum(rewards.values()) == pytest.approx(1.0)
    assert len(batch) == B_BATCH
    assert b.rewarded_but_not_selected_by_hotkey == {}
    assert b.reward_alignment == {
        "selected_groups": B_BATCH,
        "rewarded_groups": B_BATCH,
        "paid_unselected_groups": 0,
        "selected_unrewarded_groups": 0,
        "reward_alignment_ok": True,
        "slot_share": pytest.approx(share),
        "distributed_reward": pytest.approx(1.0),
        "expected_distributed_reward": pytest.approx(1.0),
    }


def _sealed_equal_score_winners(order, *, seal_randomness, hotkey_prefix):
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    operators = {
        f"{hotkey_prefix}{i}": f"operator-{i}" for i in range(20)
    }
    batcher = _make_batcher(operator_by_hotkey=operators)
    exact_k2 = [1.0, 1.0] + [0.0] * 6
    for i in order:
        _accept(
            batcher,
            _request(
                prompt_idx=i,
                hotkey=f"{hotkey_prefix}{i}",
                rewards=exact_k2,
            ),
            arrival_round=100,
            arrival_ts=1_000.0,
        )
    batcher.current_checkpoint_hash = "a" * 40
    batcher.seal_randomness = seal_randomness
    batch, rewards = batcher.seal_batch()
    ranked_prompts = [
        row["prompt_idx"]
        for row in sorted(batcher.auction_candidates, key=lambda row: row["rank"])
    ]
    return batcher, [submission.prompt_idx for submission in batch], ranked_prompts, rewards


def test_sealed_tiebreak_is_order_independent_and_not_hotkey_bound():
    forward = _sealed_equal_score_winners(
        range(20), seal_randomness="beacon-a", hotkey_prefix="alpha-"
    )
    reversed_input = _sealed_equal_score_winners(
        reversed(range(20)),
        seal_randomness="beacon-a",
        hotkey_prefix="alpha-",
    )
    renamed_hotkeys = _sealed_equal_score_winners(
        range(20), seal_randomness="beacon-a", hotkey_prefix="beta-"
    )

    assert forward[1] == reversed_input[1] == renamed_hotkeys[1]
    assert forward[2] == reversed_input[2] == renamed_hotkeys[2]
    assert len(forward[1]) == B_BATCH
    assert len(forward[3]) <= B_BATCH
    assert all(
        row["rank_entropy_source"] == "seal_drand"
        for row in forward[0].auction_candidates
    )


def test_changing_seal_randomness_changes_exact_tie_winners():
    first = _sealed_equal_score_winners(
        range(20), seal_randomness="beacon-a", hotkey_prefix="alpha-"
    )
    second = _sealed_equal_score_winners(
        range(20), seal_randomness="beacon-b", hotkey_prefix="alpha-"
    )

    assert set(first[1]) != set(second[1])


def test_missing_seal_randomness_uses_validator_precommit_arrival():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    batcher = _make_batcher()
    _accept(
        batcher,
        _request(prompt_idx=1, hotkey="later"),
        arrival_round=100,
        arrival_ts=1_000.2,
    )
    _accept(
        batcher,
        _request(prompt_idx=2, hotkey="earlier"),
        arrival_round=100,
        arrival_ts=1_000.1,
    )
    batcher.seal_batch()

    rows = {row["hotkey"]: row for row in batcher.auction_candidates}
    assert rows["earlier"]["rank"] < rows["later"]["rank"]
    assert rows["earlier"]["rank_entropy_source"] == (
        "validator_arrival_fallback"
    )
