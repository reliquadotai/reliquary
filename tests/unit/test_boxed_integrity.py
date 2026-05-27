from reliquary.validator.boxed_integrity import (
    extract_boxed_spans,
    has_malformed_final_answer,
)


def test_extract_wellformed_and_malformed():
    spans = extract_boxed_spans(r"answer \boxed{121} then \boxed{")
    assert len(spans) == 2
    assert spans[0].content == "121" and spans[0].well_formed is True
    assert spans[1].well_formed is False  # unclosed


def test_extract_special_token_box_is_malformed():
    spans = extract_boxed_spans(r"\boxed{<|im_end|>}")
    assert spans[0].well_formed is False


def test_extract_empty_box_is_malformed():
    spans = extract_boxed_spans(r"\boxed{}")
    assert spans[0].well_formed is False


def test_rewarded_rollout_never_flagged():
    # A scored-correct rollout has a well-formed final box by definition.
    flagged, reason = has_malformed_final_answer(1.0, r"\boxed{a}")
    assert flagged is False and reason is None


def test_honest_no_box_not_flagged():
    # No boxed at all = clean give-up (env falls back to trailing number).
    flagged, _ = has_malformed_final_answer(0.0, "I am not sure, maybe 7")
    assert flagged is False


def test_genuine_wrong_wellformed_box_not_flagged():
    # Model cleanly boxes a wrong value -> legitimate negative, must count.
    flagged, _ = has_malformed_final_answer(0.0, r"so \boxed{9}")
    assert flagged is False


def test_wellformed_final_after_earlier_box_not_flagged():
    # "boxed something earlier then a well-formed (wrong) final" is
    # indistinguishable from honest self-correction -> NOT flagged here
    # (the boxed-answer probability check covers forced wrong answers).
    flagged, _ = has_malformed_final_answer(0.0, r"\boxed{a} ... actually \boxed{9}")
    assert flagged is False


def test_dangling_special_token_final_box_flagged():
    text = r"answer is \boxed{9} then $$\boxed{<|im_end|>"
    flagged, reason = has_malformed_final_answer(0.0, text)
    assert flagged is True and reason == "malformed_final_boxed"


def test_empty_final_box_flagged():
    flagged, reason = has_malformed_final_answer(0.0, r"answer \boxed{9} ... \boxed{}")
    assert flagged is True and reason == "malformed_final_boxed"


def test_unclosed_final_box_flagged():
    # Repetition-to-cap proxy: last box cut off / unclosed.
    flagged, reason = has_malformed_final_answer(0.0, r"\boxed{8}\boxed{8}\boxed{")
    assert flagged is True and reason == "malformed_final_boxed"


def test_cap_truncated_malformed_final_is_deferred():
    # Near the token cap -> budget exhaustion, governed by the termination
    # guard, not flagged here.
    flagged, _ = has_malformed_final_answer(
        0.0, r"\boxed{8}\boxed{", completion_length=8100, cap=8192
    )
    assert flagged is False


def test_non_cap_malformed_final_is_flagged():
    # Far from the cap -> deliberate stop with a junk final box.
    flagged, reason = has_malformed_final_answer(
        0.0, r"\boxed{8}\boxed{", completion_length=1025, cap=8192
    )
    assert flagged is True and reason == "malformed_final_boxed"


def test_reject_reason_member_exists():
    from reliquary.protocol.submission import RejectReason
    assert RejectReason.MALFORMED_FINAL_ANSWER.value == "malformed_final_answer"
