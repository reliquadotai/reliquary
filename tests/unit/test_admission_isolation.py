from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
import multiprocessing
import os
import time
from types import SimpleNamespace

import bittensor as bt
import pytest
from tokenizers import Tokenizer, models

from reliquary.constants import CHALLENGE_K
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
)
from reliquary.validator.admission import (
    AdmissionContext,
    AdmissionProblemMaterials,
    AdmissionReceiptBinding,
    AdmissionRuntimeMaterials,
    ParsedSubmission,
    initialize_admission_worker,
    materialize_and_score_submission,
    parse_and_validate_submission,
    prepare_submission,
    score_and_finalize_submission,
)
from reliquary.protocol.signatures import sign_commit_binding
from reliquary.validator.selection_digest import (
    compute_rollouts_selection_digest,
)
from reliquary.validator.server import ValidatorServer


def _tokenizer_json() -> str:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "prompt": 1,
                r"\boxed{4}": 2,
                r"A completely different incorrect derivation. \boxed{5}": 3,
            },
            unk_token="[UNK]",
        )
    )
    return tokenizer.to_str()


_TEST_KEYPAIR = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


class _TestWallet:
    hotkey = _TEST_KEYPAIR


def _crash_worker() -> None:
    os._exit(17)


def _sleep_worker(seconds: float) -> str:
    time.sleep(seconds)
    return "late"


def _echo_worker(value: str) -> str:
    return value


def _server_with_admission_pool() -> tuple[ValidatorServer, str]:
    environment = "openmathinstruct"
    server = ValidatorServer()
    server._active_batchers = {
        environment: SimpleNamespace(
            env=SimpleNamespace(name=environment),
            tokenizer=SimpleNamespace(
                backend_tokenizer=Tokenizer.from_str(_tokenizer_json())
            ),
        )
    }
    server._admission_process_pools[environment] = (
        server._new_admission_pool(environment)
    )
    return server, environment


def _request() -> BatchSubmissionRequest:
    rollouts = []
    for index in range(8):
        answer_token = 2 if index < 4 else 3
        tokens = [1] + [answer_token] * (CHALLENGE_K - 1)
        tokens[2 + index] = 0
        reward = 1.0 if index < 4 else 0.0
        commitments = [{"sketch": 0} for _ in tokens]
        signature = sign_commit_binding(
            tokens=tokens,
            randomness_hex="cd" * 16,
            model_name="test-model",
            layer_index=1,
            commitments=commitments,
            wallet=_TestWallet,
        ).hex()
        commit = {
            "tokens": tokens,
            "commitments": commitments,
            "proof_version": "v7",
            "model": {"name": "test-model", "layer_index": 1},
            "signature": signature,
            "beacon": {"randomness": "cd" * 16},
            "rollout": {
                "prompt_length": 1,
                "completion_length": len(tokens) - 1,
                "success": reward > 0.5,
                "total_reward": reward,
                "advantage": 0.0,
                "token_logprobs": [0.0] * len(tokens),
            },
        }
        rollouts.append(
            RolloutSubmission(
                tokens=tokens,
                reward=reward,
                commit=commit,
                env_name="openmathinstruct",
            )
        )
    return BatchSubmissionRequest(
        miner_hotkey=_TEST_KEYPAIR.ss58_address,
        prompt_idx=7,
        window_start=11,
        merkle_root="00" * 32,
        rollouts=rollouts,
        checkpoint_hash="checkpoint",
        drand_round=3,
        protocol_version=2,
        nonce="nonce",
    )


def _context() -> AdmissionContext:
    return AdmissionContext(
        randomness="cd" * 16,
        environment="openmathinstruct",
        vocab_size=4,
        max_sequence_length=4096,
        eos_token_ids=(),
        canonical_force_ids=(),
        think_close_ids=(),
        bootstrap=False,
        enforce_envelope_signature=False,
        enforce_legacy_merkle=False,
    )


def _binding(request: BatchSubmissionRequest, payload_bytes: int):
    return AdmissionReceiptBinding(
        miner_hotkey=request.miner_hotkey,
        prompt_idx=request.prompt_idx,
        window_start=request.window_start,
        merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash,
        environment="openmathinstruct",
        payload_bytes=payload_bytes,
        drand_round=request.drand_round,
        protocol_version=request.protocol_version,
        nonce=request.nonce,
    )


def test_raw_preparation_matches_receipt_and_rejects_wrong_randomness(
    monkeypatch,
):
    monkeypatch.setattr(
        "reliquary.validator.admission.verify_commit_signature",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "reliquary.validator.admission.legacy_submission_merkle_matches",
        lambda _request: (True, "00" * 32),
    )
    request = _request()
    raw_body = request.model_dump_json().encode()
    parsed = parse_and_validate_submission(
        raw_body,
        _binding(request, len(raw_body)),
        _context(),
        time.monotonic() + 5.0,
    )

    assert parsed.reject_reason is None
    assert len(parsed.rollout_hashes) == 8
    assert parsed.selection_digest is not None

    request.rollouts[0].commit["beacon"]["randomness"] = "ef" * 16
    tampered = request.model_dump_json().encode()
    rejected = parse_and_validate_submission(
        tampered,
        _binding(request, len(tampered)),
        _context(),
        time.monotonic() + 5.0,
    )
    assert rejected.reject_reason is RejectReason.WRONG_RANDOMNESS
    assert rejected.reject_stage == "randomness"


def test_spawned_worker_decodes_scores_and_returns_picklable_request():
    request = _request()
    raw_body = request.model_dump_json().encode()
    materials = AdmissionProblemMaterials(
        problem={"prompt": "prompt", "ground_truth": "4", "id": "p"},
        rendered_prompt="prompt",
    )
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_admission_worker,
        initargs=(_tokenizer_json(),),
    ) as executor:
        prepared = executor.submit(
            prepare_submission,
            raw_body,
            _binding(request, len(raw_body)),
            materials,
            _context(),
            time.monotonic() + 10.0,
        ).result(timeout=15.0)

    assert prepared.reject_reason is None
    assert prepared.rewards == [1.0] * 4 + [0.0] * 4
    assert prepared.request is not None
    assert prepared.request.prompt_idx == request.prompt_idx
    assert prepared.legacy_merkle_status in {"match", "mismatch"}


def test_spawned_worker_deadline_is_terminal():
    request = _request()
    parsed = ParsedSubmission(
        request=request,
        rollout_hashes=[],
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
    )
    materials = AdmissionProblemMaterials(
        problem={"prompt": "prompt", "ground_truth": "4", "id": "p"},
        rendered_prompt="prompt",
    )
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_admission_worker,
        initargs=(_tokenizer_json(),),
    ) as executor:
        prepared = executor.submit(
            materialize_and_score_submission,
            parsed,
            materials,
            _context(),
            time.monotonic() - 1.0,
        ).result(timeout=5.0)

    assert prepared.reject_reason is RejectReason.WORKER_DROPPED
    assert prepared.reject_stage == "admission_timeout"
    assert prepared.timed_out is True


def test_authenticated_termination_reject_keeps_identity_artifacts():
    request = _request()
    raw_body = request.model_dump_json().encode()

    prepared = parse_and_validate_submission(
        raw_body,
        _binding(request, len(raw_body)),
        replace(_context(), eos_token_ids=(3,)),
        time.monotonic() + 5.0,
    )

    assert prepared.reject_reason is RejectReason.BAD_TERMINATION
    assert len(prepared.rollout_hashes) == len(request.rollouts)
    assert prepared.selection_digest is not None


def test_prepared_reason_parity_for_prompt_mismatch():
    request = _request()
    parsed = ParsedSubmission(
        request=request,
        rollout_hashes=[bytes([index]) * 32 for index in range(8)],
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
    )
    result = score_and_finalize_submission(
        parsed,
        AdmissionRuntimeMaterials(
            canonical_prompt_tokens=[3],
            problem={"prompt": "prompt", "ground_truth": "4", "id": "p"},
            completion_texts=[r"\boxed{4}"] * 4 + [r"\boxed{5}"] * 4,
        ),
        _context(),
        time.monotonic() + 5.0,
    )

    assert result.reject_reason is RejectReason.PROMPT_MISMATCH
    assert result.reject_stage == "prompt_binding"


def test_prepared_reason_parity_for_nonfinite_reward(monkeypatch):
    request = _request()
    parsed = ParsedSubmission(
        request=request,
        rollout_hashes=[bytes([index]) * 32 for index in range(8)],
        selection_digest=compute_rollouts_selection_digest(request.rollouts),
    )
    monkeypatch.setattr(
        "reliquary.validator.admission._compute_omi_reward",
        lambda *_args: float("nan"),
    )
    result = score_and_finalize_submission(
        parsed,
        AdmissionRuntimeMaterials(
            canonical_prompt_tokens=[1],
            problem={"prompt": "prompt", "ground_truth": "4", "id": "p"},
            completion_texts=[r"\boxed{4}"] * 4 + [r"\boxed{5}"] * 4,
        ),
        _context(),
        time.monotonic() + 5.0,
    )

    assert result.reject_reason is RejectReason.REWARD_MISMATCH
    assert result.reject_stage == "reward"


@pytest.mark.asyncio
async def test_admission_pool_recovers_after_worker_crash():
    server, environment = _server_with_admission_pool()
    failed_pool = server._admission_process_pools[environment]
    try:
        with pytest.raises(BrokenProcessPool):
            await server._run_admission_process(
                environment,
                _crash_worker,
                wall_seconds=2.0,
            )

        assert server._admission_process_pools[environment] is not failed_pool
        assert server._admission_worker_restarts[environment] == 1
        assert await server._run_admission_process(
            environment,
            _echo_worker,
            "healthy",
            wall_seconds=2.0,
        ) == "healthy"
    finally:
        server._terminate_admission_pool(
            server._admission_process_pools[environment]
        )


@pytest.mark.asyncio
async def test_admission_pool_recovers_after_external_timeout():
    server, environment = _server_with_admission_pool()
    failed_pool = server._admission_process_pools[environment]
    try:
        with pytest.raises(TimeoutError):
            await server._run_admission_process(
                environment,
                _sleep_worker,
                5.0,
                wall_seconds=0.0,
            )

        assert server._admission_process_pools[environment] is not failed_pool
        assert server._admission_worker_restarts[environment] == 1
        assert server._admission_timeouts[environment] == 1
        assert await server._run_admission_process(
            environment,
            _echo_worker,
            "healthy",
            wall_seconds=2.0,
        ) == "healthy"
    finally:
        server._terminate_admission_pool(
            server._admission_process_pools[environment]
        )
