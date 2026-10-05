from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

from bittensor_wallet import Keypair
import pytest

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
from reliquary.miner.engine import _compute_merkle_root
from reliquary.protocol import signatures
from reliquary.protocol.seed_pool import SeedPool
from reliquary.protocol.service_contract import ServiceContract
from reliquary.protocol.service_submission import ServiceBinding
from reliquary.protocol.submission import BatchSubmissionRequest, RejectReason, RolloutSubmission
from reliquary.services.runtime import SUPPORTED_SERVICE_CAPABILITIES, validate_submission_policy
from reliquary.validator.admission import (
    AdmissionContext, AdmissionReceiptBinding, AdmissionRuntimeMaterials,
    parse_and_validate_submission, score_and_finalize_submission,
)


def _contract(pool=True):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["environment"]["id"] = "openmathinstruct"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    value["policies"]["reward"] = {"kind": "exploration-discount/v1", "divisor": 4,
                                   "budget_bps": 1000, "refresh_windows": 2, "max_tokens_per_group": 1000000}
    if pool:
        value["policies"]["sampling"] = {"kind": "public-group-pool/v1", "group_size": M_ROLLOUTS,
                                         "pool_groups": 3, "renewal_windows": 2}
    return ServiceContract.from_dict(value)


@pytest.fixture
def signed_request(monkeypatch):
    monkeypatch.setattr(signatures, "bt", SimpleNamespace(Keypair=Keypair))
    key = Keypair.create_from_seed("0x" + "02" * 32)
    wallet = SimpleNamespace(hotkey=key)
    def make(*, pool=True, purpose="exploration", legacy=False):
        contract = _contract(pool)
        announcement = {"contract": contract.to_dict(), "supported_capabilities": sorted(SUPPORTED_SERVICE_CAPABILITIES),
                        "pool_epoch": 5, "pool_randomness": "ab" * 32}
        seed_pool = SeedPool.from_contract(contract, prompt_idx=7, checkpoint_hash="d" * 40,
                                          pool_epoch=5, randomness="ab" * 32) if pool and not legacy else None
        intent = ServiceBinding(contract.sha256, purpose)
        rollouts = []
        for index in range(M_ROLLOUTS):
            tokens = [1] + [2] * CHALLENGE_K
            tokens[5] = 10 + index
            tokens[-1] = 99
            metadata = {"prompt_length": 1, "completion_length": len(tokens) - 1,
                        "success": False, "total_reward": 0.0, "advantage": 0.0,
                        "token_logprobs": [0.0] * (len(tokens) - 1)}
            if not legacy:
                metadata["service_binding"] = intent.rollout_binding(index)
            if seed_pool is not None:
                metadata["seed_pool"] = seed_pool.selection(1).rollout_binding(index)
            commitments = [{"sketch": 0}] * len(tokens)
            if legacy:
                signature = signatures.sign_commit_binding(tokens, "cd" * 16, "test", 1, commitments, wallet)
            else:
                signature = signatures.sign_service_commit_binding(tokens, "cd" * 16, "test", 1,
                    commitments, metadata["service_binding"], wallet, seed_pool=metadata.get("seed_pool"))
            commit = {"tokens": tokens, "commitments": commitments,
                      "proof_version": "v7" if legacy else "public-group-proof/v1" if pool else "service-group-proof/v1",
                      "model": {"name": "test", "layer_index": 1}, "signature": signature.hex(),
                      "beacon": {"randomness": "cd" * 16}, "rollout": metadata}
            rollouts.append(RolloutSubmission(tokens=tokens, reward=0.0, commit=commit, env_name="openmathinstruct"))
        request = BatchSubmissionRequest(miner_hotkey=key.ss58_address, prompt_idx=7, window_start=11,
                                         merkle_root=_compute_merkle_root(rollouts), rollouts=rollouts,
                                         checkpoint_hash="d" * 40, drand_round=3, protocol_version=2, nonce="fresh",
                                         service_binding=intent.to_dict() if not legacy else None,
                                         pool_selection=seed_pool.selection(1).to_dict() if seed_pool else None)
        _sign_envelope(request, wallet)
        return request, announcement, wallet
    return make


def _sign_envelope(request, wallet):
    request.merkle_root = _compute_merkle_root(request.rollouts)
    request.envelope_signature = signatures.sign_envelope(wallet=wallet, miner_hotkey=request.miner_hotkey,
        prompt_idx=request.prompt_idx, window_start=request.window_start, merkle_root=request.merkle_root,
        checkpoint_hash=request.checkpoint_hash, drand_round=request.drand_round, randomness="cd" * 16,
        nonce=request.nonce, protocol_version=request.protocol_version, generation_profile_id=request.generation_profile_id,
        service_binding=request.service_binding, pool_selection=request.pool_selection).hex()


def _parse(request, announcement):
    raw = request.model_dump_json().encode()
    context = AdmissionContext(randomness="cd" * 16, environment="openmathinstruct", vocab_size=128,
                               max_sequence_length=4096, eos_token_ids=(99,), canonical_force_ids=(), think_close_ids=(),
                               bootstrap=False, enforce_envelope_signature=True, enforce_legacy_merkle=True,
                               service_policy=announcement)
    binding = AdmissionReceiptBinding(miner_hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx,
        window_start=request.window_start, merkle_root=request.merkle_root, checkpoint_hash=request.checkpoint_hash,
        environment="openmathinstruct", payload_bytes=len(raw), drand_round=request.drand_round,
        protocol_version=request.protocol_version, nonce=request.nonce)
    return parse_and_validate_submission(raw, binding, context, time.monotonic() + 5), context


@pytest.mark.parametrize("pool", [False, True])
def test_actual_parser_accepts_service_intent_with_real_signatures_and_merkle(signed_request, pool):
    request, announcement, _ = signed_request(pool=pool)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is None
    assert parsed.legacy_merkle_status == "match"
    assert len(parsed.rollout_hashes) == M_ROLLOUTS


def test_uniform_exploration_still_requires_truthful_reward_and_valid_answers(signed_request):
    request, announcement, _ = signed_request()
    parsed, context = _parse(request, announcement)
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"},
        completion_texts=[f"Derivation number {i}. \\boxed{{5}}" for i in range(M_ROLLOUTS)])
    prepared = score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5)
    assert prepared.reject_reason is None
    assert prepared.rewards == [0.0] * M_ROLLOUTS
    bad = replace(materials, completion_texts=["\\boxed{}"] * M_ROLLOUTS)
    assert score_and_finalize_submission(parsed, bad, context, time.monotonic() + 5).reject_reason is not None


def test_same_uniform_group_cannot_fill_training_path(signed_request):
    request, announcement, _ = signed_request(purpose="training")
    parsed, context = _parse(request, announcement)
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"},
        completion_texts=[f"Derivation number {i}. \\boxed{{5}}" for i in range(M_ROLLOUTS)])
    assert score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5).reject_reason is RejectReason.OUT_OF_ZONE


def test_training_uses_the_contract_threshold_in_the_actual_grader(signed_request):
    request, announcement, wallet = signed_request(purpose="training")
    request.rollouts[0].reward = 1.0
    _sign_envelope(request, wallet)
    parsed, context = _parse(request, announcement)
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"},
        completion_texts=[f"Derivation number {i}. \\boxed{{{4 if i == 0 else 5}}}" for i in range(M_ROLLOUTS)])
    prepared = score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5)
    assert prepared.reject_reason is None
    assert prepared.rewards == [1.0] + [0.0] * (M_ROLLOUTS - 1)


@pytest.mark.parametrize("purpose", ["training", "exploration"])
def test_service_grading_refuses_unboxed_outcomes_on_both_lanes(signed_request, purpose):
    request, announcement, wallet = signed_request(purpose=purpose)
    correct = M_ROLLOUTS // 2 if purpose == "training" else 0
    for index, rollout in enumerate(request.rollouts):
        rollout.reward = float(index < correct)
    _sign_envelope(request, wallet)
    parsed, context = _parse(request, announcement)
    texts = [f"Derivation number {i}. \\boxed{{{4 if i < correct else 5}}}" for i in range(M_ROLLOUTS)]
    texts[-1] = "Unfinished derivation without a final answer."
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"}, completion_texts=texts)
    assert score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5).reject_reason is RejectReason.OUT_OF_ZONE


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("threshold,eligible", [(2400, True), (2600, False)])
def test_arrival_gate_reuses_the_service_threshold_before_reserving(bootstrap, threshold, eligible):
    from reliquary.validator.batcher import FillState
    from tests.unit.test_grpo_window_batcher import _make_batcher
    from tests.unit.test_prove_on_arrival import _pending_stub

    contract = _contract(pool=False).to_dict()
    contract["scoring"]["sigma_min_bps"] = threshold
    batcher = _make_batcher()
    batcher.bootstrap = bootstrap
    batcher.service_policy = {"contract": contract}
    batcher.fill_state = FillState(budgets={"openmathinstruct": 4}, picks_target=16)
    extended = []
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    pending = _pending_stub(1, rewards=[0.0] * (M_ROLLOUTS // 2) + [0.5] * (M_ROLLOUTS // 2))
    pending.request = SimpleNamespace(service_binding={"purpose": "training"}, rollouts=[None] * M_ROLLOUTS)
    batcher._submit_arrival_proof(pending)
    assert bool(extended) is eligible
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == int(eligible)


@pytest.mark.parametrize("purpose", ["training", "exploration"])
@pytest.mark.parametrize("uncertain_field", ["truncated_count", "unboxed_count"])
def test_uncertain_service_arrivals_never_reserve_capacity(purpose, uncertain_field):
    from reliquary.validator.batcher import FillState
    from tests.unit.test_grpo_window_batcher import _make_batcher
    from tests.unit.test_prove_on_arrival import _pending_stub

    batcher = _make_batcher()
    batcher.service_policy = {"contract": _contract(pool=False).to_dict()}
    batcher.fill_state = FillState(budgets={"openmathinstruct": 4}, picks_target=16)
    batcher._extend_proof_plan = lambda candidates: pytest.fail("uncertain group dispatched")
    rewards = [0.0, 1.0] * (M_ROLLOUTS // 2) if purpose == "training" else [0.0] * M_ROLLOUTS
    pending = _pending_stub(1, rewards=rewards)
    pending.request = SimpleNamespace(service_binding={"purpose": purpose}, rollouts=[None] * M_ROLLOUTS)
    setattr(pending, uncertain_field, 1)
    batcher._submit_arrival_proof(pending)
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 0


@pytest.mark.parametrize("purpose", ["training", "exploration"])
def test_deep_uncertainty_cannot_be_journaled_as_a_verified_signal(signed_request, purpose):
    from reliquary.services.runtime import ServicePolicyLimit
    from tests.unit.test_grpo_window_batcher import _make_batcher

    request, announcement, _ = signed_request(purpose=purpose)
    batcher = _make_batcher()
    batcher.service_policy = announcement
    batcher.service_runtime = SimpleNamespace(contract=ServiceContract.from_dict(announcement["contract"]))
    pending = SimpleNamespace(request=request)
    verified = SimpleNamespace(truncated_count=1, unboxed_count=0)
    with pytest.raises(ServicePolicyLimit, match="uncertain outcomes"):
        batcher._record_service_proof(pending, verified)


def test_only_server_pool_beacon_and_epoch_are_authoritative(signed_request):
    request, announcement, _ = signed_request()
    for change in ({"pool_epoch": 6}, {"pool_randomness": "ef" * 32}):
        parsed, _ = _parse(request, {**announcement, **change})
        assert parsed.reject_reason is RejectReason.GENERATION_CONTRACT_MISMATCH
        assert parsed.reject_stage == "service_contract"


def test_resigned_envelope_cannot_hide_modified_candidate_proof(signed_request):
    request, announcement, wallet = signed_request()
    request.pool_selection["candidate_id"] = 2
    for rollout in request.rollouts:
        rollout.commit["rollout"]["seed_pool"]["candidate_id"] = 2
    _sign_envelope(request, wallet)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is RejectReason.BAD_SIGNATURE
    assert parsed.reject_stage == "rollout_signature"


def test_legacy_valid_request_keeps_old_wire_and_refuses_hidden_service_metadata(signed_request):
    request, _, wallet = signed_request(pool=False, legacy=True)
    parsed, _ = _parse(request, None)
    assert parsed.reject_reason is None
    assert "service_binding" not in request.model_dump()
    assert "pool_selection" not in request.model_dump()
    request.rollouts[0].commit["rollout"]["service_binding"] = ServiceBinding("aa" * 32, "training").rollout_binding(0)
    _sign_envelope(request, wallet)
    parsed, _ = _parse(request, None)
    assert parsed.reject_reason is RejectReason.GENERATION_CONTRACT_MISMATCH


def test_legacy_sampling_contract_cannot_accept_hidden_pool_metadata(signed_request):
    request, announcement, _ = signed_request(pool=False)
    request.rollouts[0].commit["rollout"]["seed_pool"] = {"schema": "public-group-rollout/v1",
        "pool_sha256": "aa" * 32, "candidate_id": 1, "rollout_index": 0}
    with pytest.raises(ValueError, match="not allowed"):
        validate_submission_policy(request, announcement)


def test_announced_capability_cannot_enable_unsupported_draw_variant(signed_request):
    request, announcement, _ = signed_request(pool=False)
    value = announcement["contract"]
    value["policies"]["sampling"] = {"kind": "public-draw-pool/v1", "group_size": M_ROLLOUTS,
                                     "pool_draws": M_ROLLOUTS + 1, "renewal_windows": 2}
    contract = ServiceContract.from_dict(value)
    request.service_binding = ServiceBinding(contract.sha256, "exploration").to_dict()
    for index, rollout in enumerate(request.rollouts):
        rollout.commit["rollout"]["service_binding"] = ServiceBinding(contract.sha256, "exploration").rollout_binding(index)
    announcement["supported_capabilities"].append("public-draw-pool/v1")
    with pytest.raises(ValueError, match="runtime lacks capabilities"):
        validate_submission_policy(request, announcement)


def _deep_service_batcher(request, announcement, verify):
    from reliquary.constants import PROTOCOL_PROFILE_ID, PROTOCOL_VERSION
    from reliquary.protocol.signatures import verify_commit_signature
    from tests.unit.test_grpo_window_batcher import _make_batcher

    request.protocol_version = PROTOCOL_VERSION
    request.generation_profile_id = PROTOCOL_PROFILE_ID
    batcher = _make_batcher(window_start=request.window_start, verify_commitment_proofs_fn=verify,
                            verify_signature_fn=verify_commit_signature)
    batcher.current_checkpoint_hash = request.checkpoint_hash
    batcher.service_policy = announcement
    batcher.tokenizer.decode = lambda ids, **kwargs: "".join(
        "\\boxed{0}" if token == 2 else "" if token == 99 else "x" for token in ids
    )
    return batcher


def _service_proof(commit, **changes):
    from reliquary.validator.verifier import ProofResult

    meta = commit["rollout"]
    size = meta["completion_length"]
    values = dict(all_passed=True, passed=1, checked=1, has_sparse_outputs=True,
                  p_stop=1.0, terminal_pick_ok=True,
                  challenge_lp_indices=list(range(meta["prompt_length"], len(commit["tokens"]))),
                  challenge_lp_values=[0.0] * size,
                  completion_chosen_probs=[1.0] * size,
                  completion_argmax_probs=[1.0] * size,
                  completion_argmax_ids=commit["tokens"][meta["prompt_length"]:],
                  seed_n_stochastic=size, seed_n_match=size, seed_n_positions=size,
                  seed_n_boundary_match=size, **changes)
    return ProofResult(**values)


@pytest.mark.parametrize("purpose", ["training", "exploration"])
def test_service_cap_discovered_by_proof_cannot_become_a_scheduler_pass(signed_request, monkeypatch, purpose):
    from reliquary.validator import batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import _prove_one

    request, announcement, wallet = signed_request(purpose=purpose)
    if purpose == "training":
        for index, rollout in enumerate(request.rollouts):
            rollout.reward = float(index < M_ROLLOUTS // 2)
    _sign_envelope(request, wallet)
    def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        return _service_proof(commit)
    batcher = _deep_service_batcher(request, announcement, verifier)
    def decode(ids, **kwargs):
        index = next((i for i in range(M_ROLLOUTS) if 10 + i in ids), None)
        if index is None:
            return "".join("\\boxed{0}" if token == 2 else "" if token == 99 else "x" for token in ids)
        correct = purpose == "training" and index < M_ROLLOUTS // 2
        return f"Derivation number {index} {'CORRECT' if correct else 'wrong'}. \\boxed{{0}}"
    batcher.tokenizer.decode = decode
    # Simulate cap status learned only from the expensive proof. Legacy allows
    # one such truncation; the service must reject before returning PASSED.
    monkeypatch.setattr(batcher_module, "is_cap_truncation", lambda commit, *args, **kwargs: commit is request.rollouts[-1].commit)
    assert _prove_one(batcher, request) is None
    assert batcher.reject_counts[RejectReason.OUT_OF_ZONE.value] == 1


@pytest.mark.parametrize("pool", [False, True])
def test_deep_service_refuses_verifier_without_seed_interface(signed_request, pool):
    from tests.unit.test_grpo_window_batcher import _prove_one

    request, announcement, _ = signed_request(pool=pool)
    def old_verifier(commit, model, randomness):
        return _service_proof(commit)
    batcher = _deep_service_batcher(request, announcement, old_verifier)
    assert _prove_one(batcher, request) is None
    assert batcher.reject_counts[RejectReason.GENERATION_CONTRACT_MISMATCH.value] == 1


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("missing", ["sparse", "positions"])
def test_deep_service_refuses_missing_sampling_evidence(signed_request, pool, missing):
    from tests.unit.test_grpo_window_batcher import _prove_one

    request, announcement, _ = signed_request(pool=pool)
    def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        proof = _service_proof(commit)
        if missing == "sparse":
            proof.has_sparse_outputs = False
        else:
            proof.seed_n_positions = 0
        return proof
    batcher = _deep_service_batcher(request, announcement, verifier)
    assert _prove_one(batcher, request) is None
    reason = RejectReason.GENERATION_CONTRACT_MISMATCH if missing == "sparse" else RejectReason.SEED_MISMATCH
    assert batcher.reject_counts[reason.value] == 1


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("mismatch", ["ratio", "cdf", None])
def test_deep_service_enforces_seed_gates_when_legacy_flags_are_shadow(signed_request, monkeypatch, pool, mismatch):
    from reliquary.validator import batcher as batcher_module
    from tests.unit.test_grpo_window_batcher import _prove_one

    monkeypatch.setattr(batcher_module, "FORCED_SEED_ENFORCE", False)
    monkeypatch.setattr(batcher_module, "FORCED_SEED_CDF_ENFORCE", False)
    request, announcement, _ = signed_request(pool=pool)
    seen = []
    def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        seen.append(seed_u_values)
        proof = _service_proof(commit)
        if mismatch == "ratio":
            proof.seed_n_match = 0
        elif mismatch == "cdf":
            proof.seed_n_hard_mismatch = 1
            proof.seed_max_cdf_miss = 0.2
        return proof
    batcher = _deep_service_batcher(request, announcement, verifier)
    result = _prove_one(batcher, request)
    assert len(seen) == M_ROLLOUTS
    assert all(len(values) == CHALLENGE_K for values in seen)
    if mismatch:
        assert result is None
        assert batcher.reject_counts[RejectReason.SEED_MISMATCH.value] == 1
    else:
        assert result is not None
