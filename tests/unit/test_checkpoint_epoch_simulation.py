import random

from scripts.simulate_checkpoint_epoch import (
    Candidate,
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
