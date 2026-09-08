from __future__ import annotations

import copy
from types import SimpleNamespace
import time

import pytest
import torch

from reliquary.constants import M_ROLLOUTS
from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction
from reliquary.environment.agentic.suite import BUILTIN_EPISODE_ENVIRONMENTS
from reliquary.environment.registry import get_environment_spec
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)
from reliquary.shared.training_payload import (
    decode_training_payload,
    encode_training_payload,
)
from reliquary.validator import admission
from reliquary.validator.admission import (
    AdmissionContext,
    AdmissionRuntimeMaterials,
    ParsedSubmission,
    score_and_finalize_submission,
)
from reliquary.validator.training import (
    _policy_token_positions,
    _selected_logprobs_for_tokens,
    reset_training_state,
    train_step,
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


class _TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 12)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyBase()
        self.lm_head = torch.nn.Linear(12, 256, bias=False)


def _episode_metadata(trace):
    assert trace.reward is not None
    return {
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


def _mixed_request(environment: str):
    tokenizer = _ByteTokenizer()
    spec = get_environment_spec(environment)
    task = spec.create().get_task(0)
    rollouts = []
    traces = []
    for index in range(M_ROLLOUTS):
        actions = (
            task.private["reference_actions"]
            if index % 2 == 0
            else [AssistantAction.final("incorrect")]
        )
        trace = EpisodeRunner(
            renderer=CanonicalEpisodeRenderer(
                lambda text: tokenizer.encode(text).ids
            )
        ).run(
            spec.create(),
            task,
            seed=index,
            policy=ScriptedPolicy(actions),
        )
        traces.append(trace)
        assistant_tokens = sum(
            end - start for start, end in trace.assistant_spans
        )
        prompt_length = trace.assistant_spans[0][0]
        rollouts.append(RolloutSubmission(
            tokens=list(trace.tokens),
            reward=0.0,
            env_name=environment,
            commit={
                "tokens": list(trace.tokens),
                "rollout": {
                    "prompt_length": prompt_length,
                    "completion_length": len(trace.tokens) - prompt_length,
                    "success": False,
                    "total_reward": 0.0,
                    "advantage": 0.0,
                    "token_logprobs": [0.0] * assistant_tokens,
                    "episode": _episode_metadata(trace),
                },
            },
        ))
    request = BatchSubmissionRequest(
        miner_hotkey="episode-integration-miner",
        prompt_idx=0,
        window_start=1,
        merkle_root="0" * 64,
        rollouts=rollouts,
        checkpoint_hash="checkpoint",
    )
    return tokenizer, task, traces, request


@pytest.mark.parametrize("environment", BUILTIN_EPISODE_ENVIRONMENTS)
def test_mixed_episode_reaches_payload_and_optimizer(
    monkeypatch,
    environment: str,
):
    from reliquary import constants

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 7)
    tokenizer, task, traces, request = _mixed_request(environment)
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
            selection_digest=b"episode-integration",
        ),
        AdmissionRuntimeMaterials(
            canonical_prompt_tokens=list(
                traces[0].tokens[:traces[0].assistant_spans[0][0]]
            ),
            problem=get_environment_spec(environment).create().get_problem(0),
            completion_texts=[""] * M_ROLLOUTS,
        ),
        AdmissionContext(
            randomness="aa",
            environment=environment,
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
    assert result.reject_reason is None
    assert [rollout.reward for rollout in request.rollouts] == [
        float(index % 2 == 0) for index in range(M_ROLLOUTS)
    ]

    model = _TinyPolicy()
    with torch.no_grad():
        for rollout in request.rollouts:
            tokens = torch.tensor([rollout.commit["tokens"]])
            selected = _selected_logprobs_for_tokens(
                model,
                tokens,
                tokens[0, 1:],
            )
            positions = _policy_token_positions(rollout)
            values = [float(selected[position - 1]) for position in positions]
            rollout.commit["rollout"]["token_logprobs"] = values
            rollout._validated_completion_logprobs = values

    group = SimpleNamespace(rollouts=request.rollouts, prompt_idx=0)
    payload = encode_training_payload(
        {environment: [group]},
        window_start=1,
        checkpoint_revision="episode-integration",
        env_order=[environment],
        env_targets={environment: 16},
        window_quarantine={"quarantined": False, "reasons": []},
    )
    decoded = decode_training_payload(payload).batches()[environment]
    assert all(
        rollout._validated_assistant_spans is not None
        for rollout in decoded[0].rollouts
    )

    reset_training_state()
    before = next(model.parameters()).detach().clone()
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    assert train_step(model, [decoded], ref_model=reference) is model
    after = next(model.parameters()).detach()
    assert not torch.equal(before, after)
