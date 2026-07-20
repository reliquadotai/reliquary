"""Proven-dominance early close: mid-window proofs cached and reused at seal,
budgets spanning the window, and the close trigger itself.

Spec: docs/superpowers/specs/2026-07-20-auction-v2-proven-dominance-close-design.md
"""
from reliquary.constants import B_BATCH
from tests.unit.test_grpo_window_batcher import (
    _always_false_grail,
    _make_batcher,
    _request,
)


def _auction_batcher(**overrides):
    b = _make_batcher(**overrides)
    assert b.difficulty_auction_enabled  # FakeEnv is the production Math lane
    return b


def _accept(b, prompt_idx, hotkey, k=2, drand_round=10):
    """Admit one candidate with a binary k-of-8 reward profile (k=2 is the
    difficulty peak). Arrival ranking uses the submitted-round fallback in
    mock mode, so drand_round orders the tiers."""
    req = _request(
        prompt_idx=prompt_idx,
        hotkey=hotkey,
        rewards=[1.0] * k + [0.0] * (8 - k),
    )
    req.drand_round = drand_round
    resp = b.accept_submission(req)
    assert resp.accepted, resp.reason
    return b.pending_submissions()[-1]


def test_prove_ranked_reuses_cached_fail_without_touching_the_gpu():
    calls = []

    def _grail(commit, model, randomness):
        calls.append(1)
        return _always_false_grail(commit, model, randomness)

    b = _auction_batcher(verify_commitment_proofs_fn=_grail)
    p = _accept(b, prompt_idx=1, hotkey="hk1")
    proven = b._verify_expensive(p)          # simulate the mid-window prover
    assert proven is None and calls          # our fake grail always fails
    calls.clear()
    b._early_proof_results[id(p)] = None
    b.early_close_proof_attempts = 1
    b.early_close_proof_failures = 1

    b.force_seal("test")
    b.seal_batch(pool=1.0)

    assert calls == []                       # cache hit: GPU untouched at seal
    row = next(r for r in b.auction_candidates if r["hotkey"] == "hk1")
    assert row["proof_phase"] == "midwindow"
    assert row["proof_passed"] is False
    # attempts telemetry includes the mid-window attempt exactly once
    assert b.proof_attempts == 1


def test_cached_pass_is_selected_and_debt_gates_do_not_reevaluate_it():
    """Sequential semantics: a candidate proven mid-window (before its operator
    hit the failure cap) stays selected at seal even if later mid-window
    failures pushed the operator over the cap."""
    b = _auction_batcher()
    p = _accept(b, prompt_idx=1, hotkey="hk1")
    sub = b._verify_expensive(p)             # default grail passes
    assert sub is not None
    b._early_proof_results[id(p)] = sub
    b.early_close_proof_attempts = 1
    # operator over the cap AFTER the proof happened (unmapped -> hotkey)
    b._expensive_proof_failures_by_operator["hk1"] = 10_000

    b.force_seal("test")
    b.seal_batch(pool=1.0)

    row = next(r for r in b.auction_candidates if r["hotkey"] == "hk1")
    assert row["status"] == "selected"
    assert row["proof_phase"] == "midwindow"
    assert [s.hotkey for s in b.valid_submissions()] == ["hk1"]


def test_midwindow_wall_seconds_count_into_the_seal_wall_budget():
    from reliquary.constants import MAX_PROOF_WALL_SECONDS

    b = _auction_batcher()
    _accept(b, prompt_idx=1, hotkey="hk1")
    b.early_close_proof_wall_seconds = MAX_PROOF_WALL_SECONDS  # spent it all
    b.force_seal("test")
    b.seal_batch(pool=1.0)

    assert b.proof_wall_exhausted is True
    assert b.valid_submissions() == []
