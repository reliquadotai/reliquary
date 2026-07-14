from reliquary.validator.rollout_telemetry import (
    classify_bft_termination,
    token_degeneracy_metrics,
)


def test_token_degeneracy_metrics_locates_repetition_onset():
    metrics = token_degeneracy_metrics(
        [1, 2, 3, 4, 8, 1, 2, 3, 4, 9] + [7] * 8,
        tail_size=8,
    )

    assert metrics["token_count"] == 18
    assert metrics["repeated_ngram_fraction"] > 0
    assert metrics["first_repeated_ngram_offset"] == 5
    assert metrics["max_same_token_run"] == 8
    assert metrics["first_same_token_run_offset"] == 10
    assert metrics["tail_repeated_ngram_fraction"] > 0


def test_token_degeneracy_metrics_handles_short_completion():
    metrics = token_degeneracy_metrics([1, 2, 3])

    assert metrics["repeated_ngram_fraction"] == 0.0
    assert metrics["first_repeated_ngram_offset"] is None
    assert metrics["max_same_token_run"] == 1


def test_classify_bft_termination_uses_only_validated_force_span():
    tokens = [10, 11, 5, 6, 200, 201, 55, 99]

    forced = classify_bft_termination(
        tokens,
        prompt_length=2,
        completion_length=6,
        eos_ids={99},
        think_close_ids={200},
        validated_force_span=(4, 6),
        thinking_budget=2,
        answer_budget=4,
    )
    untrusted = classify_bft_termination(
        tokens,
        prompt_length=2,
        completion_length=6,
        eos_ids={99},
        think_close_ids={200},
        validated_force_span=None,
        thinking_budget=2,
        answer_budget=4,
    )

    assert forced == "forced_phase2_eos"
    assert untrusted == "natural_phase2_eos"


def test_classify_bft_termination_distinguishes_caps():
    assert classify_bft_termination(
        [10, 11, 5, 6, 200, 201, 7, 7, 7],
        prompt_length=2,
        completion_length=7,
        eos_ids={99},
        think_close_ids={200},
        validated_force_span=(4, 6),
        thinking_budget=2,
        answer_budget=3,
    ) == "forced_phase2_cap"
    assert classify_bft_termination(
        [10, 11, 5, 200, 7, 7, 7],
        prompt_length=2,
        completion_length=5,
        eos_ids={99},
        think_close_ids={200},
        validated_force_span=None,
        thinking_budget=8,
        answer_budget=3,
    ) == "natural_phase2_cap"
