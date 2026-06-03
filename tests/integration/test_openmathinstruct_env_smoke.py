"""End-to-end smoke test for the OpenMathInstruct-2 environment.

Requires network + ~1 GB of disk (downloads the first parquet shard from
HuggingFace). Skipped if the dataset cannot be reached.
"""

import pytest


def test_default_environments_includes_openmathinstruct():
    from reliquary.constants import ENVIRONMENT_MIX
    assert any(name == "openmathinstruct" for name, _ in ENVIRONMENT_MIX)


def test_default_environment_loads_and_scores():
    """Load the smallest possible shard set and run reward on a real row."""
    import os
    os.environ.setdefault("RELIQUARY_OMI_SHARDS", "1")
    from reliquary.environment import load_environment, Environment

    try:
        env = load_environment("openmathinstruct")
    except Exception as exc:
        pytest.skip(f"Could not load default environment: {exc}")

    assert isinstance(env, Environment)
    assert len(env) > 0

    problem = env.get_problem(0)
    assert "prompt" in problem and "ground_truth" in problem and "id" in problem
    assert len(problem["id"]) == 16
    # The boxed-answer instruction must be appended so reward extraction works
    # regardless of the base model's default style (validator + miner share this).
    assert r"\boxed{}" in problem["prompt"]

    # Correct answer in boxed form
    assert env.compute_reward(
        problem, r"\boxed{" + problem["ground_truth"] + "}"
    ) == 1.0
    # Obviously wrong
    assert env.compute_reward(problem, "definitely wrong") == 0.0


def test_get_problem_is_deterministic():
    """Same index always returns the same problem (cross-miner consistency)."""
    import os
    os.environ.setdefault("RELIQUARY_OMI_SHARDS", "1")
    from reliquary.environment import load_environment
    try:
        env = load_environment("openmathinstruct")
    except Exception as exc:
        pytest.skip(f"Could not load environment: {exc}")
    p1 = env.get_problem(123)
    p2 = env.get_problem(123)
    assert p1["id"] == p2["id"]
    assert p1["prompt"] == p2["prompt"]
    assert p1["ground_truth"] == p2["ground_truth"]


def test_modulo_wrap_for_out_of_range_index():
    """Indices beyond len(env) wrap modulo without raising."""
    import os
    os.environ.setdefault("RELIQUARY_OMI_SHARDS", "1")
    from reliquary.environment import load_environment
    try:
        env = load_environment("openmathinstruct")
    except Exception as exc:
        pytest.skip(f"Could not load environment: {exc}")
    n = len(env)
    p1 = env.get_problem(0)
    p2 = env.get_problem(n)
    assert p1["id"] == p2["id"]
