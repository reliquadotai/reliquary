from __future__ import annotations

from types import SimpleNamespace

from reliquary.constants import CHALLENGE_K
from reliquary.protocol.profiles import resolve_protocol_profile
from reliquary.protocol.signatures import build_episode_commit_binding
from reliquary.protocol.submission import CommitModel
from reliquary.validator.training import _policy_token_positions
from reliquary.validator.verifier import policy_token_positions


def _episode_commit() -> dict:
    length = max(12, CHALLENGE_K + 4)
    tokens = list(range(length))
    episode = {
        "schema_version": "reliquary/episode/v1",
        "renderer_id": "reliquary-jsonl-tools-v1",
        "task_id": "task-1",
        "seed": 7,
        "actions": [{"tool": "lookup", "arguments": {}}, {"final": "done"}],
        "assistant_spans": [[2, 4], [7, 9]],
        "observation_digests": ["0" * 64, "1" * 64],
        "termination_reason": "finished",
        "state_digest": "2" * 64,
        "trace_digest": "3" * 64,
    }
    return {
        "tokens": tokens,
        "commitments": [{} for _ in tokens],
        "proof_version": "v8",
        "model": {"name": "model", "layer_index": 0},
        "signature": "aa",
        "beacon": {"randomness": "aa"},
        "rollout": {
            "prompt_length": 2,
            "completion_length": length - 2,
            "success": False,
            "total_reward": 0.0,
            "advantage": 0.0,
            "token_logprobs": [0.0] * 4,
            "episode": episode,
        },
    }


def test_v8_schema_and_policy_positions():
    commit = _episode_commit()
    parsed = CommitModel.model_validate(commit)
    assert parsed.rollout.episode is not None
    expected = [2, 3, 7, 8]
    assert policy_token_positions(commit["tokens"], commit["rollout"]) == expected

    rollout = SimpleNamespace(commit=commit)
    rollout._validated_assistant_spans = ((2, 4), (7, 9))
    assert _policy_token_positions(rollout) == expected


def test_v8_rejects_single_turn_proof_version():
    commit = _episode_commit()
    commit["proof_version"] = "v7"
    try:
        CommitModel.model_validate(commit)
    except ValueError as exc:
        assert "v8" in str(exc)
    else:
        raise AssertionError("episode accepted without proof v8")


def test_episode_profile_is_opt_in_and_binds_limits():
    profile = resolve_protocol_profile("qwen3-4b-reliquary-episode-v7-dev1")
    contract = profile.to_generation_contract()
    assert set(contract["environments"]) == {
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    }
    episode = contract["environments"]["reliquary_stateful_tools_v1"]["episode"]
    assert episode["max_action_tokens"] == 1024
    assert episode["max_episode_tokens"] == 16384


def test_v8_signature_binding_changes_with_episode_state():
    commit = _episode_commit()
    kwargs = dict(
        tokens=commit["tokens"],
        randomness_hex="aa",
        model_name="model",
        layer_index=0,
        commitments=commit["commitments"],
    )
    first = build_episode_commit_binding(
        **kwargs, episode=commit["rollout"]["episode"]
    )
    changed = dict(commit["rollout"]["episode"])
    changed["state_digest"] = "f" * 64
    second = build_episode_commit_binding(**kwargs, episode=changed)
    assert first != second
