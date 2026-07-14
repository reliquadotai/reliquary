from types import SimpleNamespace

from reliquary.validator.verifier import (
    evaluate_all_token_auth_shadow,
    evaluate_token_authenticity,
    validate_force_span,
)

# In these fixtures: atomic </think> = 777, canonical FORCE ids = [777, 7, 8],
# prompt_length = 2, thinking budget = 2 (the thinking part is tokens[2:4]).
_FORCE = [777, 7, 8]
_CLOSE = {777}


def test_validate_force_span_accepts_byte_exact_span():
    # prompt[0,1] + thinking[5,6] + force[777,7,8] + answer[55,99]
    tokens = [0, 1, 5, 6, 777, 7, 8, 55, 99]
    meta = {"forced": True, "force_span": (4, 7)}
    ok, exempt = validate_force_span(
        tokens, meta, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is True
    assert exempt == {2, 3, 4}  # span (4,7) shifted by prompt_length 2


def test_validate_force_span_rejects_noncanonical_tokens():
    tokens = [0, 1, 5, 6, 777, 42, 8, 55, 99]  # 42 differs from canonical 7
    meta = {"forced": True, "force_span": (4, 7)}
    ok, exempt = validate_force_span(
        tokens, meta, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is False
    assert exempt == set()


def test_validate_force_span_rejects_split_think_close():
    # split-tokenised </think> = [510,26003,29] instead of the atomic id 777
    tokens = [0, 1, 5, 6, 510, 26003, 29, 7, 8, 55, 99]
    meta = {"forced": True, "force_span": (4, 9)}
    ok, _ = validate_force_span(
        tokens, meta, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is False


def test_validate_force_span_rejects_early_force():
    # byte-exact span, but injected before the thinking budget (budget=10 ≠ 2)
    tokens = [0, 1, 5, 6, 777, 7, 8, 55, 99]
    meta = {"forced": True, "force_span": (4, 7)}
    ok, _ = validate_force_span(
        tokens, meta, _FORCE, 2, thinking_budget=10, think_close_ids=_CLOSE,
    )
    assert ok is False


def test_validate_force_span_rejects_think_close_before_force():
    # model already emitted </think> (777) in the thinking part → must not force
    tokens = [0, 1, 777, 6, 777, 7, 8, 55, 99]
    meta = {"forced": True, "force_span": (4, 7)}
    ok, _ = validate_force_span(
        tokens, meta, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is False


def test_validate_force_span_non_forced_is_noop():
    ok, exempt = validate_force_span(
        [0, 1, 5, 6], {"forced": False}, _FORCE, 2,
        thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is True
    assert exempt == set()


def test_validate_force_span_rejects_out_of_range():
    ok, _ = validate_force_span(
        [0, 1, 5, 6, 777], {"forced": True, "force_span": (4, 99)}, _FORCE, 2,
        thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is False


def test_validate_force_span_rejects_non_numeric_bounds_without_raising():
    ok, exempt = validate_force_span(
        [0, 1, 5, 6, 777, 7, 8],
        {"forced": True, "force_span": ("not-an-index", object())},
        _FORCE,
        2,
        thinking_budget=2,
        think_close_ids=_CLOSE,
    )
    assert ok is False
    assert exempt == set()


def test_auth_exempts_force_positions():
    # completion positions 2,3,4 are injected force tokens (prob ~0)
    proof = SimpleNamespace(
        completion_chosen_probs=[0.9, 0.8, 1e-12, 1e-12, 1e-12, 0.7],
        completion_argmax_probs=[0.9, 0.8, 0.99, 0.99, 0.99, 0.7],
        completion_argmax_ids=[1, 2, 3, 4, 5, 6],
    )
    ok, _ = evaluate_token_authenticity(proof, threshold=1e-6)
    assert ok is False
    ok2, _ = evaluate_token_authenticity(proof, threshold=1e-6, exempt_positions={2, 3, 4})
    assert ok2 is True


def test_all_token_shadow_exempts_force_positions():
    proof = SimpleNamespace(
        completion_chosen_probs=[0.9, 0.8, 1e-12, 1e-12, 1e-12, 0.7],
        completion_argmax_probs=[0.9, 0.8, 0.99, 0.99, 0.99, 0.7],
        completion_argmax_ids=[1, 2, 3, 4, 5, 6],
    )
    ok, metrics = evaluate_all_token_auth_shadow(
        proof, threshold=1e-6, exempt_positions={2, 3, 4},
    )
    assert ok is True
    assert metrics["findings"] == 0
