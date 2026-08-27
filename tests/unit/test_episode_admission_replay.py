from __future__ import annotations

import time

from reliquary.constants import M_ROLLOUTS
from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.registry import get_environment_spec
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator import admission
from reliquary.validator.admission import (
    AdmissionContext,
    AdmissionRuntimeMaterials,
    ParsedSubmission,
    score_and_finalize_submission,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return _Encoding(list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return bytes(ids).decode("utf-8")


def _request(*, tamper_observation: bool = False):
    tokenizer = _ByteTokenizer()
    spec = get_environment_spec("reliquary_retrieval_tools_v1")
    env = spec.create()
    task = env.get_task(0)
    trace = EpisodeRunner(
        renderer=CanonicalEpisodeRenderer(lambda text: tokenizer.encode(text).ids)
    ).run(
        env,
        task,
        seed=17,
        policy=ScriptedPolicy(task.private["reference_actions"]),
    )
    assert trace.reward is not None
    episode = {
        "schema_version": trace.schema,
        "renderer_id": "reliquary-jsonl-tools-v1",
        "task_id": trace.task_id,
        "seed": trace.seed,
        "actions": [action.to_wire() for action in trace.actions],
        "assistant_spans": [list(span) for span in trace.assistant_spans],
        "observation_digests": list(trace.observation_digests),
        "termination_reason": trace.termination_reason,
        "state_digest": trace.reward.state_digest,
        "trace_digest": trace.trace_digest,
    }
    rollouts = []
    for index in range(M_ROLLOUTS):
        tokens = list(trace.tokens)
        if tamper_observation and index == 0:
            tokens[trace.assistant_spans[0][1] + 1] ^= 1
        commit = {
            "tokens": tokens,
            "rollout": {
                "prompt_length": trace.assistant_spans[0][0],
                "completion_length": len(tokens) - trace.assistant_spans[0][0],
                "success": False,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": [0.0] * sum(
                    end - start for start, end in trace.assistant_spans
                ),
                "episode": episode,
            },
        }
        rollouts.append(RolloutSubmission(
            tokens=tokens,
            reward=0.0,
            commit=commit,
            env_name=spec.name,
        ))
    request = BatchSubmissionRequest(
        miner_hotkey="miner",
        prompt_idx=0,
        window_start=1,
        merkle_root="0" * 64,
        rollouts=rollouts,
        checkpoint_hash="checkpoint",
    )
    return tokenizer, env, trace, request


def _score(monkeypatch, *, tamper_observation=False):
    tokenizer, env, trace, request = _request(
        tamper_observation=tamper_observation
    )
    monkeypatch.setattr(admission, "_WORKER_TOKENIZER", tokenizer)
    monkeypatch.setattr(
        admission,
        "episode_limits_for_environment",
        lambda _environment: (1024, 16384, 65536),
    )
    result = score_and_finalize_submission(
        ParsedSubmission(
            request=request,
            rollout_hashes=[bytes([index]) for index in range(M_ROLLOUTS)],
            selection_digest=b"digest",
        ),
        AdmissionRuntimeMaterials(
            canonical_prompt_tokens=list(
                trace.tokens[:trace.assistant_spans[0][0]]
            ),
            problem=env.get_problem(0),
            completion_texts=[""] * M_ROLLOUTS,
        ),
        AdmissionContext(
            randomness="aa",
            environment=env.name,
            vocab_size=None,
            max_sequence_length=20000,
            eos_token_ids=(),
            canonical_force_ids=(),
            think_close_ids=(),
            bootstrap=False,
            enforce_envelope_signature=False,
            enforce_legacy_merkle=False,
        ),
        time.monotonic() + 10,
    )
    return request, trace, result


def test_admission_replays_before_trusting_assistant_spans(monkeypatch):
    request, trace, result = _score(monkeypatch)
    # Every reference rollout succeeds, so the group has no GRPO variance and
    # is correctly rejected only after replay/reward assignment.
    assert result.reject_reason is RejectReason.OUT_OF_ZONE
    for rollout in request.rollouts:
        assert rollout.reward == 1.0
        assert rollout._validated_assistant_spans == trace.assistant_spans
        assert rollout._validated_episode_trace_digest == trace.trace_digest


def test_admission_rejects_tampered_tool_observation(monkeypatch):
    _request_value, _trace, result = _score(
        monkeypatch, tamper_observation=True
    )
    assert result.reject_reason is RejectReason.REWARD_MISMATCH
    assert result.reject_stage == "reward"
