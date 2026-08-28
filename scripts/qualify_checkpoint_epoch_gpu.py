#!/usr/bin/env python3
"""Offline full-model qualification for one checkpoint-epoch group.

The script never opens a network listener or submission connection. It loads
the pinned V5 base model, generates one forced-seed group, builds the normal
miner proof payload, finalizes its compact epoch commitment, and verifies every
rollout locally with the validator proof path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from types import SimpleNamespace

import bittensor as bt
import httpx
from pydantic import ValidationError
import torch

from reliquary.constants import (
    ACTIVE_PROTOCOL_PROFILE,
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    FORCED_SEED_CDF_ENFORCE,
)
from reliquary.environment.forced_sampling import u_at
from reliquary.miner.engine import MiningEngine
from reliquary.miner.submitter import (
    finalize_checkpoint_epoch_commitment_v1,
    finalize_checkpoint_epoch_generation_intent_v1,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    CommitModel,
    WindowState,
)
from reliquary.protocol.profiles import resolve_protocol_profile
from reliquary.protocol.tokens import encode_prompt
from reliquary.shared.checkpoint_epoch import (
    BeaconBinding,
    CheckpointBinding,
    ProtocolBinding,
    WindowSchedule,
    build_epoch_plan,
    generation_contract_sha256,
    manifest_sha256,
)
from reliquary.shared.checkpoint_epoch_market import (
    SignedGenerationIntentSet,
    generation_intent_set_sha256,
    generation_intent_set_signing_bytes,
)
from reliquary.shared.modeling import load_text_generation_model, load_tokenizer
from reliquary.validator.batcher import (
    GrpoWindowBatcher,
    _forced_seed_rollout_reject,
    _forced_seed_verdict,
    _verify_logprobs_for_training,
    _verify_short_logprob_claim,
)
from reliquary.validator.server import ValidatorServer
from reliquary.validator.prompt_content import (
    prompt_content_sha256,
    render_canonical_prompt,
)
from reliquary.validator.verifier import (
    verify_commitment_proofs,
    verify_logprobs_claim,
)


PROFILE_ID = "qwen3-4b-base-dapo-reasoning-v5"
ENVIRONMENT = "openmathinstruct"


class _Environment:
    validator_authoritative_reward = True

    def __init__(self, name: str, problem: dict) -> None:
        self.name = name
        self.problem = problem
        self.successful_completions: set[str] = set()

    def __len__(self) -> int:
        return 10_000

    def get_problem(self, _prompt_idx: int) -> dict:
        return self.problem

    def compute_reward(self, _problem, completion: str) -> float:
        return float(completion in self.successful_completions)


def _wallet():
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    hotkey = SimpleNamespace(
        ss58_address=keypair.ss58_address,
        sign=keypair.sign,
    )
    return SimpleNamespace(hotkey=hotkey)


def _plan(profile, environment: str):
    contract = profile.to_generation_contract()
    return build_epoch_plan(
        protocol=ProtocolBinding(
            profile_id=profile.profile_id,
            protocol_version=profile.protocol_version,
            generation_contract_sha256=generation_contract_sha256(contract),
        ),
        checkpoint=CheckpointBinding(
            number=1,
            repo_id=profile.model_id,
            revision=profile.model_revision,
            commit_observed_round=1_000,
        ),
        epoch_beacon=BeaconBinding(
            source="drand",
            chain="quicknet",
            chain_hash="1" * 64,
            round=1_001,
            randomness="2" * 64,
        ),
        beacon_delay_rounds=1,
        first_window=500,
        window_count=16,
        warmup_rounds=20,
        window_schedule=WindowSchedule(
            mode="concurrent_checkpoint_epoch",
            collection_seconds=1_600.0,
            timeout_seconds=7_200,
        ),
        training_mode="sequential_steps",
        target_groups_per_environment_lane=16,
        candidate_limit_per_environment_lane=32,
        environment_universes={environment: 10_000},
        prompt_range_size=100,
    )


def _completion_text(tokenizer, rollout) -> str:
    metadata = rollout.commit["rollout"]
    return tokenizer.decode(
        rollout.commit["tokens"][int(metadata["prompt_length"]):]
    )


def _qualify_http_lane(
    *,
    plan,
    profile,
    model,
    tokenizer,
    wallet,
    environment,
    finalized,
) -> dict:
    """Exercise the real local HTTP admission and post-seal proof path."""
    if environment.name != ENVIRONMENT:
        raise ValueError("unexpected HTTP lane environment")
    epoch_window = plan.windows[0]
    prompt_slice = epoch_window.prompt_slices[0]
    operator = "offline-qualification-operator"
    batcher = GrpoWindowBatcher(
        window_start=epoch_window.window_number,
        env=environment,
        model=model,
        tokenizer=tokenizer,
        completion_text_fn=lambda rollout: _completion_text(tokenizer, rollout),
        canonical_prompt_tokens_fn=lambda prompt_idx: encode_prompt(
            tokenizer, environment.get_problem(prompt_idx)["prompt"]
        ),
        drand_round_check_enabled=False,
        operator_by_hotkey={wallet.hotkey.ss58_address: operator},
        experimental_epoch_ranking=True,
        experimental_prompt_range=(prompt_slice.start, prompt_slice.stop),
        collection_seconds=plan.window_schedule.collection_seconds,
        max_productive_candidates=plan.candidate_limit_per_environment_lane,
        max_ranked_proof_attempts=plan.candidate_limit_per_environment_lane,
    )
    batcher.current_checkpoint_hash = profile.model_revision
    batcher.randomness = epoch_window.generation_randomness
    batcher.checkpoint_epoch_generation_randomness = (
        epoch_window.generation_randomness
    )
    batcher.set_prompt_range()

    server = ValidatorServer()
    server._auction_admission_enabled = True
    server.set_current_checkpoint(SimpleNamespace(
        checkpoint_n=plan.checkpoint.number,
        repo_id=plan.checkpoint.repo_id,
        revision=plan.checkpoint.revision,
        signature="offline-qualification",
    ))
    server.set_checkpoint_epoch_plan(plan)
    server.set_active_epoch_batchers({
        (environment.name, epoch_window.window_number): batcher,
    })
    server.set_registered_hotkeys(
        {wallet.hotkey.ss58_address},
        operator_by_hotkey={wallet.hotkey.ss58_address: operator},
    )
    server.set_checkpoint_epoch_phase("intent")
    server.set_current_state(WindowState.OPEN)

    async def exercise_http():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://offline-qualification",
        ) as client:
            intent = finalize_checkpoint_epoch_generation_intent_v1(
                wallet=wallet,
                operator_id=operator,
                plan=plan,
                window_start=epoch_window.window_number,
                environment=environment.name,
                prompt_idx=finalized.precommit.prompt_idx,
                prompt_content_sha256=prompt_content_sha256(
                    environment.name,
                    render_canonical_prompt(
                        tokenizer,
                        str(environment.problem["prompt"]),
                    ),
                ),
            )
            intention = (
                await client.post(
                    "/checkpoint-epoch/generation-intents",
                    content=intent.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )
            ).json()
            if not intention.get("accepted"):
                raise RuntimeError(f"generation intent failed: {intention}")
            server.set_checkpoint_epoch_phase("selection")
            frozen = server.freeze_checkpoint_epoch_generation_intent_set(
                intent_close_round=close_round,
                validator_hotkey=wallet.hotkey.ss58_address,
            )
            server.install_checkpoint_epoch_generation_intent_set(
                SignedGenerationIntentSet(
                    intent_set=frozen,
                    intent_set_sha256=generation_intent_set_sha256(frozen),
                    validator_signature=wallet.hotkey.sign(
                        generation_intent_set_signing_bytes(frozen)
                    ).hex(),
                )
            )
            selected = server.select_checkpoint_epoch_generation_tickets(
                intent_close_round=close_round,
                admission_beacon=BeaconBinding(
                    source=plan.epoch_beacon.source,
                    chain=plan.epoch_beacon.chain,
                    chain_hash=plan.epoch_beacon.chain_hash,
                    round=close_round + 1,
                    randomness="e" * 64,
                ),
                generation_deadline_ts=time.time() + 120.0,
            )
            committed = (
                await client.post(
                    "/submit/precommit",
                    content=finalized.precommit.model_dump_json(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Reliquary-Epoch-Intent": intention["intent_id"],
                    },
                )
            ).json()
            if not committed.get("accepted"):
                raise RuntimeError(f"precommit failed: {committed}")
            batcher.window_opened_wall_ts = (
                time.time() - plan.window_schedule.collection_seconds - 1.0
            )
            admission_started = time.perf_counter()
            revealed = (
                await client.post(
                    "/submit",
                    content=finalized.payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Reliquary-Precommit": committed["receipt_id"],
                    },
                )
            ).json()
            reveal_seconds = time.perf_counter() - admission_started
            queue = server._submission_queue_for_environment(environment.name)
            queued = queue.get_nowait()
            request = BatchSubmissionRequest.model_validate_json(queued.raw_body)
            started, start_reason = batcher.start_revealed_admission(
                queued.receipt.receipt_id,
                request,
            )
            if not started:
                raise RuntimeError(f"candidate admission failed: {start_reason}")
            try:
                admission = batcher.accept_submission(request)
            finally:
                batcher.finish_proof_admission(request)
            server._complete_upload_receipt(queued.receipt, admission)
            admission_seconds = time.perf_counter() - admission_started
            return (
                intention,
                committed,
                selected,
                revealed,
                admission,
                reveal_seconds,
                admission_seconds,
            )

    close_round = plan.epoch_beacon.round + 9
    (
        intention,
        committed,
        selected,
        revealed,
        admission,
        reveal_seconds,
        admission_seconds,
    ) = asyncio.run(exercise_http())

    if not revealed.get("accepted"):
        raise RuntimeError(f"reveal failed: {revealed}")
    if not admission.accepted:
        raise RuntimeError(f"candidate admission failed: {admission.reason.value}")
    if batcher.pending_count != 1:
        raise RuntimeError(
            "revealed payload did not enter the candidate pool: "
            f"response={revealed} rejects={batcher.reject_counts}"
        )
    pending_before_seal = batcher.pending_count
    batcher.collection_close_drand_round = close_round + 1
    batcher.seal_beacon_round = close_round + 2
    batcher.seal_randomness = "f" * 64
    batcher.force_seal("offline_qualification")
    winners, rewards = batcher.seal_batch(commit_side_effects=False)
    candidate = batcher.auction_candidates[0]
    return {
        "intent_accepted": intention["accepted"],
        "payload_precommit_accepted": committed["accepted"],
        "selected_primary": selected[(
            environment.name,
            epoch_window.window_number,
        )]["primary"],
        "reveal_accepted": bool(revealed["accepted"]),
        "candidate_admitted": admission.accepted,
        "pending_before_seal": pending_before_seal,
        "reveal_seconds": round(reveal_seconds, 3),
        "admission_seconds": round(admission_seconds, 3),
        "proof_attempts": batcher.proof_attempts,
        "proven_winners": len(winners),
        "rewarded_hotkeys": len(rewards),
        "rank_entropy_source": candidate["rank_entropy_source"],
        "candidate_status": candidate["status"],
        "candidate_proof_passed": candidate["proof_passed"],
        "reject_counts": dict(batcher.reject_counts),
        "arrival_fields_in_rank_key": False,
        "seal_beacon_round": batcher.seal_beacon_round,
        "collection_close_drand_round": batcher.collection_close_drand_round,
    }


def _synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--http-lane", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    profile = resolve_protocol_profile(PROFILE_ID)
    if ACTIVE_PROTOCOL_PROFILE.profile_id != profile.profile_id:
        raise SystemExit(
            f"set RELIQUARY_PROTOCOL_PROFILE={profile.profile_id}"
        )
    plan = _plan(profile, ENVIRONMENT)
    window = plan.windows[0]
    prompt_slice = window.prompt_slices[0]
    prompt_idx = prompt_slice.start
    problem = {
        "id": "offline-qualification",
        "prompt": (
            "Solve this arithmetic problem carefully and show the intermediate "
            "calculation so another reader can verify it. Compute 17 * 23, "
            "then place the final numerical answer in a boxed expression."
        ),
        "ground_truth": "391",
    }

    load_started = time.perf_counter()
    tokenizer = load_tokenizer(
        profile.model_id,
        revision=profile.model_revision,
    )
    if len(encode_prompt(tokenizer, problem["prompt"])) < CHALLENGE_K:
        raise SystemExit("qualification prompt must cover the proof challenge")
    model = load_text_generation_model(
        profile.model_id,
        revision=profile.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()
    _synchronize()
    load_seconds = time.perf_counter() - load_started

    wallet = _wallet()
    environment = _Environment(ENVIRONMENT, problem)
    engine = MiningEngine(
        model,
        model,
        tokenizer,
        wallet,
        env=environment,
        vllm_gpu=0,
        proof_gpu=0,
        max_new_tokens=args.max_new_tokens,
    )
    torch.cuda.reset_peak_memory_stats()

    generation_started = time.perf_counter()
    generations = engine._generate_m_rollouts(
        problem,
        window.generation_randomness,
        env_name=environment.name,
        prompt_idx=prompt_idx,
        checkpoint_hash=profile.model_revision,
    )
    _synchronize()
    generation_seconds = time.perf_counter() - generation_started

    if args.http_lane:
        decoded = [
            tokenizer.decode(
                item["tokens"][int(item["prompt_length"]):]
            )
            for item in generations
        ]
        environment.successful_completions = set(decoded[: len(decoded) // 2])
        if sum(environment.compute_reward(problem, text) for text in decoded) != 8:
            raise SystemExit("HTTP reward fixture requires eight unique winners")

    proof_build_started = time.perf_counter()
    request = engine.build_batch_request_from_generations(
        generations=generations,
        problem=problem,
        environment=environment,
        randomness=window.generation_randomness,
        prompt_idx=prompt_idx,
        window_number=window.window_number,
        checkpoint_revision=profile.model_revision,
    )
    for rollout_index, rollout in enumerate(request.rollouts):
        try:
            CommitModel.model_validate(rollout.commit)
        except ValidationError as exc:
            raise RuntimeError(
                f"rollout {rollout_index} commit schema failed: {exc}"
            ) from exc
    finalized = finalize_checkpoint_epoch_commitment_v1(
        request,
        plan=plan,
        wallet=wallet,
        drand_round=plan.activation_not_before_round,
    )
    _synchronize()
    proof_build_seconds = time.perf_counter() - proof_build_started

    validation_started = time.perf_counter()
    proof_passed = 0
    logprob_passed = 0
    seed_stochastic = 0
    seed_exact_matches = 0
    seed_per_rollout = []
    seed_hard_mismatches = 0
    sketch_diff_max = 0
    for rollout_index, rollout in enumerate(request.rollouts):
        metadata = rollout.commit["rollout"]
        completion_length = int(metadata["completion_length"])
        uniforms = [
            u_at(
                window.generation_randomness,
                prompt_idx,
                profile.model_revision,
                rollout_index,
                offset,
            )
            for offset in range(completion_length)
        ]
        proof = verify_commitment_proofs(
            rollout.commit,
            model,
            window.generation_randomness,
            tokenizer=tokenizer,
            seed_u_values=uniforms,
        )
        proof_passed += int(proof.all_passed)
        seed_stochastic += proof.seed_n_stochastic
        seed_exact_matches += proof.seed_n_match
        seed_per_rollout.append((proof.seed_n_stochastic, proof.seed_n_match))
        seed_hard_mismatches += proof.seed_n_hard_mismatch
        sketch_diff_max = max(sketch_diff_max, proof.sketch_diff_max)
        logprob_ok, _ = verify_logprobs_claim(
            rollout.tokens,
            int(metadata["prompt_length"]),
            completion_length,
            metadata["token_logprobs"],
            proof,
        )
        if not logprob_ok and completion_length < CHALLENGE_K:
            logprob_ok, _ = _verify_short_logprob_claim(
                _verify_logprobs_for_training(proof, completion_length),
                rollout.tokens,
                int(metadata["prompt_length"]),
                completion_length,
                metadata["token_logprobs"],
            )
        logprob_passed += int(logprob_ok)
    _synchronize()
    validation_seconds = time.perf_counter() - validation_started

    completion_lengths = [
        len(item["tokens"]) - int(item["prompt_length"])
        for item in generations
    ]
    completion_tokens = sum(completion_lengths)
    prepare_seconds = generation_seconds + proof_build_seconds
    seed_ratio_gate_passed = not (
        _forced_seed_verdict(seed_stochastic, seed_exact_matches, True)
        or _forced_seed_rollout_reject(seed_per_rollout, True)
    )
    result = {
        "scope": "offline single-group qualification; no submission",
        "device": torch.cuda.get_device_name(0),
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "profile_id": profile.profile_id,
        "model_revision": profile.model_revision,
        "manifest_sha256": manifest_sha256(plan),
        "window_seed": window.generation_randomness,
        "rollouts": len(generations),
        "max_new_tokens": args.max_new_tokens,
        "completion_tokens": completion_tokens,
        "completion_length_min": min(completion_lengths),
        "completion_length_median": statistics.median(completion_lengths),
        "completion_length_max": max(completion_lengths),
        "completion_cap_hits": sum(
            length == args.max_new_tokens for length in completion_lengths
        ),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "proof_build_seconds": round(proof_build_seconds, 3),
        "local_validation_seconds": round(validation_seconds, 3),
        "prepare_seconds": round(prepare_seconds, 3),
        "generation_tokens_per_second": round(
            completion_tokens / generation_seconds, 3
        ),
        "groups_per_1600_seconds_at_observed_length": math.floor(
            1_600.0 / prepare_seconds
        ),
        "payload_bytes": len(finalized.payload),
        "proofs_passed": proof_passed,
        "logprob_checks_passed": logprob_passed,
        "seed_hard_mismatches": seed_hard_mismatches,
        "seed_exact_match_ratio": round(
            seed_exact_matches / max(seed_stochastic, 1), 6
        ),
        "seed_ratio_gate_passed": seed_ratio_gate_passed,
        "exact_cdf_gate_enforced": FORCED_SEED_CDF_ENFORCE,
        "sketch_diff_max": sketch_diff_max,
        "peak_cuda_memory_gib": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        ),
    }
    if args.http_lane:
        result["http_epoch_lane"] = _qualify_http_lane(
            plan=plan,
            profile=profile,
            model=model,
            tokenizer=tokenizer,
            wallet=wallet,
            environment=environment,
            finalized=finalized,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    expected = profile.sampling.rollouts
    if (
        proof_passed != expected
        or logprob_passed != expected
        or not seed_ratio_gate_passed
        or (
            args.http_lane
            and result["http_epoch_lane"]["proven_winners"] != 1
        )
    ):
        raise SystemExit("full-model qualification failed")


if __name__ == "__main__":
    main()
