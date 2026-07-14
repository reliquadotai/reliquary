"""Proof deferral: a submission is graded and scored at admission, but not proven
until it is ranked high enough to win. See
docs/superpowers/specs/2026-07-14-difficulty-auction-design.md §7
"""
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
    )


def test_pending_submission_is_scored_at_admission():
    """Scoring is cheap (it only needs the rewards), so it happens before the GPU
    ever sees the submission — that is what lets us rank before proving."""
    assert _pending(k=2).value > _pending(k=6).value


def test_pending_submission_ranks_in_the_auction():
    """It must satisfy the duck-type select_batch_auction consumes, so the same
    ranking code works on unproven candidates."""
    from reliquary.validator.batch_auction import select_batch_auction
    from reliquary.validator.cooldown import CooldownMap

    hard = _pending(hotkey="hard", prompt_idx=1, k=2)
    easy = _pending(hotkey="easy", prompt_idx=2, k=6)

    batch, _ = select_batch_auction(
        [easy, hard], b=1,
        cooldown_map=CooldownMap(cooldown_windows=0), current_window=1, pool=1.0,
    )

    assert [s.hotkey for s in batch] == ["hard"]


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
    assert proven.value == pending.value
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


def test_seal_trigger_counts_pending_not_valid():
    """The seal trigger must fire on the B-th distinct PENDING prompt. Proofs
    run at seal, so _valid stays empty for the whole window — a trigger reading
    _valid would never fire."""
    from reliquary.constants import B_BATCH
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    assert b._seal_trigger_round is None
    for i in range(B_BATCH):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"hk{i}"))

    # The trigger armed off the pending pool, not off _valid.
    assert b._seal_trigger_round is not None
    assert b.is_sealed()
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
