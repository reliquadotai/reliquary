"""``reliquarylogic_v1`` through mix resolution, admission, seal and trainer.

The unit tests prove the generators and checkers are correct in isolation.
This proves the environment actually fits the chain: that its manifest and
contract resolve, that admission overwrites the miner's claimed reward, that
a sealed batch carries the right provenance into the archive, and that the
training payload round-trips.

It deliberately drives one equality-checked family and one
constraint-checked family through the same path, because those take
different branches inside ``check_answer`` and only the first was ever
exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from reliquary import constants as C
from reliquary.environment.logic_tasks import generate_logic_task
from reliquary.environment.registry import (
    get_environment_spec, resolve_environment_mix,
)
from reliquary.protocol.profiles import resolve_protocol_profile
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)
from reliquary.shared.training_payload import (
    decode_training_payload,
    encode_training_payload,
)
from reliquary.trainer.train_runner import TrainRunner
from reliquary.validator.admission import (
    AdmissionContext,
    AdmissionRuntimeMaterials,
    ParsedSubmission,
    score_and_finalize_submission,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.dedup import compute_rollout_hash
from reliquary.validator.prompt_content import (
    prompt_content_sha256,
    target_content_sha256,
)
from reliquary.validator.selection_digest import (
    compute_rollouts_selection_digest,
)
from reliquary.validator.service import ValidationService
from reliquary.validator.verifier import ProofResult


ENVIRONMENT = "reliquarylogic_v1"
PROFILE = "qwen3-4b-reliquary-logic-v8-dev1"


class _Wallet:
    class hotkey:
        ss58_address = "validator-test"

        @staticmethod
        def sign(data: bytes) -> bytes:
            return hashlib.sha256(data).digest()


class _Tokenizer:
    eos_token_id = None

    def decode(self, _tokens):
        return ""


class _Model:
    class config:
        vocab_size = 100_000
        max_position_embeddings = 4096

    def eval(self):
        return self

    def parameters(self):
        return []

    def gradient_checkpointing_enable(self):
        return None


def _index_for(check: str) -> int:
    """First index whose family uses the given reward check."""
    for index in range(20_000):
        if generate_logic_task(index).check == check:
            return index
    raise AssertionError(f"no active family uses {check!r}")


def _request(environment, prompt_idx: int, hotkey: str):
    problem = environment.get_problem(prompt_idx)
    # Constraint-checked specs carry no answer on the wire, so the reference
    # comes from the generator rather than from the ground truth.
    expected = generate_logic_task(prompt_idx).expected
    completions = []
    rollouts = []
    for rollout_index in range(C.M_ROLLOUTS):
        correct = rollout_index < C.M_ROLLOUTS // 2
        completions.append(
            f"reasoning-{rollout_index}\n```json\n"
            + json.dumps(
                {"result": expected if correct else None},
                separators=(",", ":"),
            )
            + "\n```"
        )
        tokens = [1] + [100 + rollout_index] * C.CHALLENGE_K
        commit = {
            "tokens": tokens,
            "commitments": [{"sketch": 0} for _token in tokens],
            "proof_version": "v7",
            "model": {"name": "test-model", "layer_index": 1},
            "signature": "ab" * 32,
            "beacon": {"randomness": "cd" * 16},
            "rollout": {
                "prompt_length": 1,
                "completion_length": C.CHALLENGE_K,
                # Deliberately false: authoritative admission must overwrite it.
                "success": True,
                "total_reward": 1.0,
                "advantage": 0.0,
                "token_logprobs": [0.0] * len(tokens),
            },
        }
        rollouts.append(
            RolloutSubmission(
                tokens=tokens,
                reward=1.0,
                commit=commit,
                env_name=ENVIRONMENT,
            )
        )
    request = BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=50,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="",
        drand_round=1,
        protocol_version=0,
    )
    return problem, completions, request


def _prepare(environment, prompt_idx: int, hotkey: str):
    problem, completions, request = _request(environment, prompt_idx, hotkey)
    hashes = [
        compute_rollout_hash(rollout.tokens) for rollout in request.rollouts
    ]
    parsed = ParsedSubmission(
        request=request,
        rollout_hashes=hashes,
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
    )
    prepared = score_and_finalize_submission(
        parsed,
        AdmissionRuntimeMaterials(
            canonical_prompt_tokens=None,
            problem=problem,
            completion_texts=completions,
        ),
        AdmissionContext(
            randomness="cd" * 16,
            environment=ENVIRONMENT,
            vocab_size=100_000,
            max_sequence_length=4096,
            eos_token_ids=(),
            canonical_force_ids=(),
            think_close_ids=(),
            bootstrap=False,
            enforce_envelope_signature=False,
            enforce_legacy_merkle=False,
        ),
        time.monotonic() + 5.0,
    )
    prepared.prompt_content_sha256 = prompt_content_sha256(
        ENVIRONMENT, problem["prompt"]
    )
    prepared.target_content_sha256 = target_content_sha256(
        ENVIRONMENT, problem
    )
    prepared.task_family = problem["task_family"]
    prepared.generator_version = problem["generator_version"]
    prepared.operation_id = problem["operation_id"]
    prepared.difficulty = problem["difficulty"]
    return prepared


def test_profile_declares_the_environment_on_the_cpu_lane():
    profile = resolve_protocol_profile(PROFILE)
    mix = resolve_environment_mix(
        [ENVIRONMENT],
        profile_environments=profile.environments,
        default_batch_target=C.B_BATCH,
    )
    assert mix == [(ENVIRONMENT, 16)]
    spec = get_environment_spec(ENVIRONMENT)
    # A logic puzzle must never start the code grader.
    assert spec.admission_resource_class == "cpu"
    assert spec.validator_authoritative_reward is True


@pytest.mark.asyncio
async def test_logic_environment_admission_seal_and_trainer(monkeypatch):
    service = ValidationService(
        wallet=_Wallet(),
        model=_Model(),
        tokenizer=_Tokenizer(),
        env_mix=[(ENVIRONMENT, 2)],
        use_drand=False,
    )
    assert service.env_targets == {ENVIRONMENT: 2}
    environment = service.envs[ENVIRONMENT]

    batcher = GrpoWindowBatcher(
        window_start=50,
        env=environment,
        model=_Model(),
        tokenizer=_Tokenizer(),
        completion_text_fn=lambda rollout: "accepted"
        if rollout.reward > 0.5
        else "rejected",
        verify_commitment_proofs_fn=lambda *_args, **_kwargs: ProofResult(
            all_passed=True,
            passed=1,
            checked=1,
            logits=torch.empty(0),
        ),
        verify_signature_fn=lambda *_args, **_kwargs: True,
        drand_round_check_enabled=False,
        batch_target=2,
    )
    batcher.difficulty_auction_enabled = True
    batcher.randomness = "cd" * 16
    batcher.seal_randomness = "ef" * 16

    # One family per reward branch: equality and constraint satisfaction.
    indices = [_index_for("equality"), _index_for("numbrix_path")]
    families = {generate_logic_task(index).family for index in indices}
    assert len(families) == 2

    for position, prompt_idx in enumerate(indices):
        prepared = _prepare(environment, prompt_idx, f"miner-{position}")
        assert prepared.reject_reason is None
        # The miner claimed 1.0 on every rollout; admission is authoritative.
        assert prepared.rewards == [1.0] * (C.M_ROLLOUTS // 2) + [0.0] * (
            C.M_ROLLOUTS // 2
        )
        assert prepared.task_family == "logic_v1"
        reserved, reason, stage = batcher.reserve_prepared_identity(
            prepared.request, prepared.rollout_hashes
        )
        assert (reserved, reason, stage) == (True, None, None)
        assert batcher.accept_prepared_submission(prepared).accepted

    selected, payouts = batcher.seal_batch()
    assert len(selected) == 2
    assert payouts == {"miner-0": 0.5, "miner-1": 0.5}
    assert all(group.task_family == "logic_v1" for group in selected)

    archived = {}
    service._utility_telemetry.write_window = MagicMock()
    import reliquary.infrastructure.archive_queue as archive_queue

    monkeypatch.setattr(
        archive_queue,
        "get_archive_queue",
        lambda: SimpleNamespace(
            enqueue=lambda _window, payload: archived.update(payload)
        ),
    )
    await service._archive_window(
        {ENVIRONMENT: batcher},
        {ENVIRONMENT: (selected, payouts)},
    )
    assert archived["batch_targets"] == {ENVIRONMENT: 2}
    # Read the pin rather than repeating it: that the profile and registry
    # agree is asserted in the unit tests, and repeating it here would make
    # every regeneration break an unrelated file.
    assert archived["environment_manifest_sha256_by_environment"] == {
        ENVIRONMENT: get_environment_spec(ENVIRONMENT).environment_manifest_sha256
    }
    assert archived["batch"][0]["task_family"] == "logic_v1"
    assert archived["batch"][0]["target_content_sha256"]

    monkeypatch.setattr(C, "PROTOCOL_VERSION", 8)
    encoded = encode_training_payload(
        {ENVIRONMENT: selected},
        window_start=50,
        checkpoint_revision="candidate-base",
        env_order=[ENVIRONMENT],
        env_targets={ENVIRONMENT: 2},
        window_quarantine={"quarantined": False, "reasons": []},
    )
    decoded = decode_training_payload(encoded)
    assert decoded.env_targets == {ENVIRONMENT: 2}

    trained = []
    runner = TrainRunner(
        object(),
        ref_model=object(),
        env_targets={ENVIRONMENT: 2},
        env_order=[ENVIRONMENT],
        assess_fn=lambda *_args, **_kwargs: SimpleNamespace(quarantined=False),
        train_step_fn=lambda model, batches, **_kwargs: (
            trained.append(len(batches[0])) or model
        ),
    )
    assert runner.step(decoded) is True
    assert trained == [2]
