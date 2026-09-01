import random

from scripts.simulate_checkpoint_epoch import (
    Candidate,
    _economic_sensitivity,
    _population,
    _rank_epoch_lane,
    _run_policy,
)


def _candidate(*, operator: int, prompt: int, difficulty: float) -> Candidate:
    return Candidate(
        environment="math",
        lane=0,
        operator=operator,
        prompt=prompt,
        difficulty=difficulty,
        tokens=32_000,
        gpu_seconds=40.0,
        prepared_at_open=True,
        valid=True,
        stale=False,
    )


def test_synthetic_epoch_rank_preserves_utility_then_operator_rounds():
    candidates = [
        _candidate(operator=0, prompt=0, difficulty=0.9),
        _candidate(operator=0, prompt=1, difficulty=0.9),
        _candidate(operator=1, prompt=2, difficulty=0.9),
        _candidate(operator=1, prompt=3, difficulty=0.8),
    ]

    ranked = _rank_epoch_lane(19_871, candidates)

    assert [candidate.difficulty for candidate in ranked] == [0.9, 0.9, 0.9, 0.8]
    assert {candidate.operator for candidate in ranked[:2]} == {0, 1}
    assert ranked[2].operator == 0


def test_economic_sensitivity_is_deterministic_and_marks_scope():
    first = _economic_sensitivity()
    second = _economic_sensitivity()

    assert first == second
    assert "not production telemetry" in first["scope"]
    baseline = first["length_rows"][2]
    assert baseline["mean_completion_tokens"] == 2_000
    assert baseline["unbounded_generation_capacity_index"] == 1.0
    assert baseline["ticketed_epoch_paid_group_index"] == 1.0
    assert baseline["gross_token_capacity_index"] == 1.0
    assert baseline["flat_slot_return_per_gpu_second_index"] == 1.0
    assert baseline["unilateral_token_share_return_per_gpu_second_index"] == 1.0
    assert baseline["first_offer_rate_index"] == 1.0
    assert all(
        row["fourth_sequential_offer_vs_parallel"] == 0.25
        for row in first["length_rows"]
    )
    assert (
        first["length_rows"][0]["flat_slot_return_per_gpu_second_index"]
        > baseline["flat_slot_return_per_gpu_second_index"]
    )
    assert first["synthetic_compute_waste_envelope"] == {
        "unbounded_64_for_16": 0.75,
        "ticketed_32_for_16_worst_case": 0.5,
        "ticketed_20_for_16_example": 0.2,
        "adaptive_fill_at_90_percent_validity": 0.1,
        "note": (
            "group-count envelope only; it does not price utility, proof "
            "cost, retries, or real participant behavior"
        ),
    }
    assert first["symmetric_gross_token_contest"][-1] == {
        "operators": 32,
        "pool_fraction_dissipated_as_linear_cost": 0.96875,
    }


def test_synthetic_comparison_keeps_rate_selection_and_reward_distinct():
    population = _population(
        rng=random.Random(19_871),
        horizon=4,
        candidate_limit=64,
    )
    fill = _run_policy(
        name="rate_paced_fill",
        seed=19_871,
        population=population,
        horizon=4,
        target=16,
        candidate_supply=32,
        reveal_limit=32,
        mode="rate_fill",
    )
    epoch = _run_policy(
        name="checkpoint_epoch",
        seed=19_871,
        population=population,
        horizon=4,
        target=16,
        candidate_supply=64,
        reveal_limit=32,
        mode="ticketed_epoch",
    )

    assert fill["selection_mode"] == "rate_fill"
    assert fill["reward_unit"] == "eos_completion_tokens"
    assert epoch["selection_mode"] == "ticketed_epoch"
    assert epoch["reward_unit"] == "selected_group"
    assert epoch["selected_mean_difficulty"] > fill["selected_mean_difficulty"]
    assert epoch["generation_intents"] == 512
    assert fill["generation_intents"] is None
