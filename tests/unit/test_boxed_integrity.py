from reliquary.validator.boxed_integrity import (
    extract_boxed_spans,
    is_reward_manipulated,
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
    flagged, reason = is_reward_manipulated(1.0, r"\boxed{a}", "a")
    assert flagged is False and reason is None


def test_honest_failure_no_box_not_flagged():
    flagged, _ = is_reward_manipulated(0.0, "I am not sure, maybe 7", "a")
    assert flagged is False


def test_genuine_wrong_single_box_not_flagged():
    flagged, _ = is_reward_manipulated(0.0, r"so \boxed{9}", "a")  # 9 != GT a
    assert flagged is False


def test_flip_boxed_gt_then_wrong_is_flagged():
    text = r"work \boxed{a} ... actually \boxed{9}"
    flagged, reason = is_reward_manipulated(0.0, text, "a")
    assert flagged is True and reason == "boxed_gt_earlier"


def test_dangling_special_token_after_valid_box_is_flagged():
    text = r"answer is \boxed{9} then $$\boxed{<|im_end|>"
    flagged, reason = is_reward_manipulated(0.0, text, "a")
    assert flagged is True and reason == "malformed_final"


def test_empty_ground_truth_not_flagged():
    flagged, _ = is_reward_manipulated(0.0, r"\boxed{a}", "")
    assert flagged is False


def test_normalization_parity_with_reward_fn():
    # "3.0" boxed must match GT "3" the same way the reward fn would.
    flagged, reason = is_reward_manipulated(0.0, r"\boxed{3.0} ... \boxed{x}", "3")
    assert flagged is True and reason == "boxed_gt_earlier"


def test_reject_reason_member_exists():
    from reliquary.protocol.submission import RejectReason
    assert RejectReason.REWARD_MANIPULATION.value == "reward_manipulation"
