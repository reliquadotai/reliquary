"""Smoke test for the OpenCodeInstruct environment.

Loads the deterministic subset from HF Hub (or a local override
via RELIQUARY_OCI_SUBSET_REPO) and exercises get_problem +
compute_reward against an in-process fake grader (no runsc needed
for this smoke).

Skipped if the dataset can't be loaded.
"""

import pytest


def test_load_env_and_get_problem_shape():
    from reliquary.environment import load_environment, Environment

    try:
        env = load_environment("opencodeinstruct")
    except Exception as exc:
        pytest.skip(f"could not load opencodeinstruct env: {exc}")

    assert isinstance(env, Environment)
    assert len(env) > 0

    p = env.get_problem(0)
    assert "prompt" in p and "ground_truth" in p and "id" in p
    assert len(p["id"]) == 16

    import json
    tests = json.loads(p["ground_truth"])
    assert isinstance(tests, list) and len(tests) > 0
    assert all(isinstance(t, str) for t in tests)


def test_compute_reward_zero_when_grader_unreachable(monkeypatch):
    """Without a running grader, compute_reward returns 0.0 (never raises)."""
    from reliquary.environment import load_environment

    try:
        env = load_environment("opencodeinstruct")
    except Exception as exc:
        pytest.skip(f"could not load opencodeinstruct env: {exc}")

    # Point the grader client at a definitely-missing socket.
    monkeypatch.setattr(env._grader, "socket_path", "/tmp/definitely-not-a-real-socket.sock")
    p = env.get_problem(0)
    r = env.compute_reward(p, "```python\ndef anything(): pass\n```")
    assert r == 0.0
