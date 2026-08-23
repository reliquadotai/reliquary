"""All-token authenticity enforcement for code rollouts.

Covers the detector semantics the batcher now enforces on
``opencodeinstruct``: a confident-position token outside the sampling
support is flagged, an honest confident token passes, and the
forced-termination span stays exempt (so it is never a false positive).
"""

import importlib

from reliquary.validator.verifier import ProofResult, evaluate_all_token_auth_shadow


def _proof(chosen, argmax):
    return ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=chosen,
        completion_argmax_probs=argmax,
    )


def test_flags_out_of_nucleus_token():
    # Position 2: chosen prob is near-zero while the model was ~certain of a
    # different argmax -> not samplable at protocol top_p, so it is flagged.
    proof = _proof([0.99, 0.99, 1e-7, 0.99], [0.99, 0.99, 0.999, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is False
    assert metrics["findings"] == 1


def test_honest_confident_tokens_pass():
    proof = _proof([0.99, 0.98, 0.97, 0.99], [0.99, 0.98, 0.97, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is True
    assert metrics["findings"] == 0


def test_low_prob_at_uncertain_position_not_flagged():
    # chosen below threshold but the model was NOT confident (argmax < conf):
    # a genuine decision point, legitimately sampled -> no finding.
    proof = _proof([0.99, 1e-7, 0.99], [0.99, 0.30, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is True
    assert metrics["findings"] == 0


def test_forced_span_is_exempt():
    # Same injected-looking token, but the position is inside the exempted
    # forced-termination span -> must not count as a false positive.
    proof = _proof([0.99, 0.99, 1e-7, 0.99], [0.99, 0.99, 0.999, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof, exempt_positions={2})
    assert ok is True
    assert metrics["findings"] == 0


def test_enforce_flag_cannot_be_disabled_by_env(monkeypatch):
    import reliquary.constants as constants

    monkeypatch.setenv("RELIQUARY_ALL_TOKEN_AUTH_ENFORCE", "0")
    reloaded = importlib.reload(constants)
    try:
        assert reloaded.ALL_TOKEN_AUTH_ENFORCE is True
    finally:
        monkeypatch.delenv("RELIQUARY_ALL_TOKEN_AUTH_ENFORCE", raising=False)
        importlib.reload(constants)
