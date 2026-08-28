from scripts.simulate_checkpoint_epoch import (
    Candidate,
    _economic_sensitivity,
    _rank_epoch_lane,
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
    assert first["symmetric_gross_token_contest"][-1] == {
        "operators": 32,
        "pool_fraction_dissipated_as_linear_cost": 0.96875,
    }
