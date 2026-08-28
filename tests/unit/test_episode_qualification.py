from __future__ import annotations

from scripts.qualify_episode_suite import (
    qualify_adversarial,
    qualify_cpu,
    summarize_model_environment,
)


def _row(reward: float, *, error=None, exact_replay=True):
    return {
        "reward": reward,
        "error": error,
        "exact_replay": exact_replay,
        "invalid_actions": 0,
        "elapsed_seconds": 1.0,
    }


def test_cpu_and_adversarial_qualification_pass_all_episode_environments():
    names = (
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    )
    assert all(value["passed"] for value in qualify_cpu(names, 2).values())
    assert all(value["passed"] for value in qualify_adversarial(names).values())


def test_model_gate_rejects_uniform_success_or_uniform_failure():
    for reward in (0.0, 1.0):
        rows = [_row(reward), _row(reward)]
        summary = summarize_model_environment(
            rows,
            {0: [reward, reward]},
            sigma_min=0.24,
        )
        assert summary["passed"] is False
        assert summary["grpo_eligible_groups"] == 0


def test_model_gate_requires_exact_replay_and_training_frontier():
    rows = [_row(0.0), _row(1.0)]
    assert summarize_model_environment(
        rows,
        {0: [0.0, 1.0]},
        sigma_min=0.24,
    )["passed"] is True
    rows[1]["exact_replay"] = False
    assert summarize_model_environment(
        rows,
        {0: [0.0, 1.0]},
        sigma_min=0.24,
    )["passed"] is False
