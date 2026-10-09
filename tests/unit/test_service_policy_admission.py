from dataclasses import replace
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
from reliquary.protocol.service_contract import SUPPORTED_V2_CAPABILITIES
from reliquary.protocol.service_schedule import initial_schedule
from reliquary.services.admission_policy import validate_submission_policy
from tests.unit.service_v2_fixtures import CODE, contract_v2_dict
from reliquary.validator.admission import (
    AdmissionContext, AdmissionReceiptBinding, AdmissionRuntimeMaterials,
    parse_and_validate_submission, score_and_finalize_submission,
)


OMI = "openmathinstruct"
POOL_SEEDS = 2 * M_ROLLOUTS
SEEDS = tuple(range(1, POOL_SEEDS, 2))           # the default subset of these tests: every odd seed
OTHER = tuple(range(M_ROLLOUTS))                 # another valid subset of the same pool


def _contract(pool=True, missing_box="uncertain"):
    value = contract_v2_dict(envs=(OMI, CODE), missing_box=missing_box)
    if not pool:
        for env in value["environments"].values():
            env["sampling"] = {"kind": "legacy/v1"}
    return ServiceContract.from_dict(value)


def _announcement(contract, **changes):
    return {"contract": contract.to_dict(), "schedule": initial_schedule(contract).to_dict(),
            "checkpoint": {"checkpoint_n": 3, "repo": "models/test", "revision": "d" * 40, "sha256": "e" * 64},
            "supported_capabilities": sorted(SUPPORTED_V2_CAPABILITIES),
            "pool_epoch": 5, "pool_randomness": "ab" * 32, **changes}


@pytest.fixture
def signed_request(monkeypatch):
    monkeypatch.setattr(signatures, "bt", SimpleNamespace(Keypair=Keypair))
    key = Keypair.create_from_seed("0x" + "02" * 32)
    wallet = SimpleNamespace(hotkey=key)
    def make(*, pool=True, purpose="exploration", legacy=False, seeds=SEEDS, missing_box="uncertain"):
        contract = _contract(pool, missing_box)
        announcement = _announcement(contract)
        seed_pool = SeedPool.from_contract(contract, environment=OMI, prompt_idx=7, checkpoint_hash="d" * 40,
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
                metadata["seed_pool"] = seed_pool.selection(seeds).rollout_binding(index)
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
                                         pool_selection=seed_pool.selection(seeds).to_dict() if seed_pool else None)
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


def _parse(request, announcement, max_sequence_length=4096):
    raw = request.model_dump_json().encode()
    context = AdmissionContext(randomness="cd" * 16, environment="openmathinstruct", vocab_size=128,
                               max_sequence_length=max_sequence_length, eos_token_ids=(99,), canonical_force_ids=(), think_close_ids=(),
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


def _lane_of(prepared, announcement=None, *, request=None, contract=None):
    from reliquary.services.admission_policy import service_lane

    return service_lane(prepared.request, contract or _contract(), prepared.rewards,
                        truncated_indices=prepared.truncated_indices, uncertain_indices=prepared.uncertain_indices,
                        attainable_rewards=prepared.attainable_rewards)


def test_a_uniform_group_declared_for_training_is_admitted_as_an_exploration_observation(signed_request):
    """The worker no longer refuses on the declared purpose: the VECTOR decides the lane, so a uniform
    group cannot fill the training path whatever it declares."""
    request, announcement, _ = signed_request(purpose="training")
    parsed, context = _parse(request, announcement)
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"},
        completion_texts=[f"Derivation number {i}. \\boxed{{5}}" for i in range(M_ROLLOUTS)])
    prepared = score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5)
    assert prepared.reject_reason is None
    assert _lane_of(prepared).lane == "exploration"


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


def _graded(signed_request, *, purpose, correct, missing_box="uncertain", unboxed=(M_ROLLOUTS - 1,)):
    """Grade a group in the real admission worker: ``correct`` right answers first, the rest
    wrong, and the rollouts in ``unboxed`` properly terminated without any box."""
    request, announcement, wallet = signed_request(purpose=purpose, missing_box=missing_box)
    for index, rollout in enumerate(request.rollouts):
        rollout.reward = float(index < correct and index not in unboxed)
    _sign_envelope(request, wallet)
    parsed, context = _parse(request, announcement)
    assert parsed.reject_reason is None
    texts = [f"Derivation number {i}. \\boxed{{{4 if i < correct else 5}}}" for i in range(M_ROLLOUTS)]
    for index in unboxed:
        texts[index] = f"Unfinished derivation {index} without a final answer."
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"}, completion_texts=texts)
    return score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5)


def test_training_admits_an_in_zone_math_group_with_one_unboxed_rollout(signed_request):
    prepared = _graded(signed_request, purpose="training", correct=M_ROLLOUTS // 2)
    assert prepared.reject_reason is None
    assert prepared.uncertain_indices == (M_ROLLOUTS - 1,)
    assert prepared.truncated_indices == ()
    assert prepared.unboxed_count == 1
    assert prepared.attainable_rewards == (0.0, 1.0)
    assert prepared.rewards[-1] == 0.0


def test_training_refuses_a_math_group_whose_only_failure_is_unboxed(signed_request):
    """All right but one, and that one has no box: it may have been right, i.e. a uniform group."""
    prepared = _graded(signed_request, purpose="training", correct=M_ROLLOUTS)
    assert prepared.rewards == [1.0] * (M_ROLLOUTS - 1) + [0.0]
    assert prepared.reject_reason is None
    # In zone as observed but not robust (R23): never a training group, never exploration.
    lane = _lane_of(prepared)
    assert (lane.lane, lane.reason) == ("unproven", "not_robust")


def test_the_same_group_where_a_missing_box_is_a_plain_zero_is_in_zone(signed_request):
    """Science rule (``missing_box: graded``): same texts, same rewards, no uncertainty."""
    prepared = _graded(signed_request, purpose="training", correct=M_ROLLOUTS, missing_box="graded")
    assert prepared.rewards == [1.0] * (M_ROLLOUTS - 1) + [0.0]
    assert prepared.reject_reason is None
    assert prepared.uncertain_indices == ()
    assert prepared.unboxed_count == 0


@pytest.mark.parametrize("missing_box", ["uncertain", "graded"])
def test_a_length_capped_rollout_is_uncertain_in_every_environment(signed_request, monkeypatch, missing_box):
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "truncated_rollout_indices", lambda request, context: (M_ROLLOUTS - 1,))
    wrong = _graded(signed_request, purpose="training", correct=M_ROLLOUTS - 1, missing_box=missing_box, unboxed=())
    assert wrong.rewards == [1.0] * (M_ROLLOUTS - 1) + [0.0]
    assert wrong.reject_reason is None
    assert _lane_of(wrong).lane == "unproven"                  # the capped failure may have been a success (R23)
    half = _graded(signed_request, purpose="training", correct=M_ROLLOUTS // 2, missing_box=missing_box, unboxed=())
    assert half.reject_reason is None
    assert _lane_of(half).lane == "training"
    assert half.truncated_indices == (M_ROLLOUTS - 1,)
    assert half.uncertain_indices == (M_ROLLOUTS - 1,)
    assert half.truncated_count == 1


def test_exploration_keeps_an_observation_with_an_unboxed_rollout(signed_request):
    prepared = _graded(signed_request, purpose="exploration", correct=0)
    assert prepared.reject_reason is None
    assert prepared.rewards == [0.0] * M_ROLLOUTS
    assert prepared.uncertain_indices == (M_ROLLOUTS - 1,)
    assert prepared.truncated_indices == ()
    assert _lane_of(prepared).lane == "exploration"
    # A miner cannot route an in-zone vector to exploration: the declared purpose plays no part.
    in_zone = _graded(signed_request, purpose="exploration", correct=M_ROLLOUTS // 2)
    assert in_zone.reject_reason is None
    assert _lane_of(in_zone).lane == "training"


def test_the_worker_derives_the_lattice_without_the_legacy_flag(signed_request, monkeypatch):
    """The service rule must not depend on a legacy profile switch to know the lattice."""
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "ROBUST_TRUNCATION_UTILITY_ENABLED", False)
    prepared = _graded(signed_request, purpose="training", correct=M_ROLLOUTS // 2)
    assert prepared.reject_reason is None
    assert prepared.attainable_rewards == (0.0, 1.0)
    assert prepared.robust_utility is None


def test_the_service_path_never_carries_the_legacy_threshold_utility(signed_request, monkeypatch):
    """The legacy SIGMA_MIN utility must not leak into a service group (the lane uses the contract's
    threshold), even when the legacy switch is on."""
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "ROBUST_TRUNCATION_UTILITY_ENABLED", True)
    prepared = _graded(signed_request, purpose="training", correct=M_ROLLOUTS // 2)
    assert prepared.uncertain_indices == (M_ROLLOUTS - 1,)
    assert prepared.robust_utility is None and prepared.reject_reason is None


def test_legacy_grading_never_reaches_the_service_rule(signed_request, monkeypatch):
    import reliquary.services.admission_policy as policy
    import reliquary.validator.admission as admission

    def unreachable(*args, **kwargs):
        raise AssertionError("service policy code reached without a service policy")

    for name in ("service_lane", "missing_box_is_uncertain", "uncertain_rollout_indices",
                 "exploration_pay_entitlement"):
        monkeypatch.setattr(policy, name, unreachable)
    request, _, wallet = signed_request(pool=False, legacy=True)
    for index, rollout in enumerate(request.rollouts):
        rollout.reward = float(index < M_ROLLOUTS // 2)
    _sign_envelope(request, wallet)
    parsed, context = _parse(request, None)
    assert parsed.reject_reason is None
    texts = [f"Derivation number {i}. \\boxed{{{4 if i < M_ROLLOUTS // 2 else 5}}}" for i in range(M_ROLLOUTS)]
    texts[-1] = "Unfinished derivation without a final answer."
    materials = AdmissionRuntimeMaterials(canonical_prompt_tokens=[1], problem={"ground_truth": "4"}, completion_texts=texts)
    seen = {}
    for answer_format in ("boxed", "boxed_or_trailing_number"):
        monkeypatch.setattr(admission, "MATH_ANSWER_FORMAT", answer_format)
        prepared = score_and_finalize_submission(parsed, materials, context, time.monotonic() + 5)
        assert prepared.reject_reason is None
        assert prepared.uncertain_indices == () and prepared.truncated_indices == ()   # service-only fields
        seen[answer_format] = prepared.unboxed_count
    assert seen == {"boxed": 1, "boxed_or_trailing_number": 0}                         # the rule of main


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


def _arrival(purpose, rewards, *, service=True, **fields):
    from reliquary.validator.batcher import FillState
    from tests.unit.test_grpo_window_batcher import _make_batcher
    from tests.unit.test_prove_on_arrival import _pending_stub

    batcher = _make_batcher()
    if service:
        batcher.service_policy = {"contract": _contract(pool=False).to_dict()}
    batcher.fill_state = FillState(budgets={"openmathinstruct": 4}, picks_target=16)
    extended = []
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)
    pending = _pending_stub(1, rewards=rewards)
    pending.request = SimpleNamespace(service_binding={"purpose": purpose}, rollouts=[None] * M_ROLLOUTS)
    for name, value in fields.items():
        setattr(pending, name, value)
    batcher._submit_arrival_proof(pending)
    reserved = batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"]
    assert reserved == int(bool(extended))
    return reserved


HALF = [1.0] * (M_ROLLOUTS // 2) + [0.0] * (M_ROLLOUTS - M_ROLLOUTS // 2)
BINARY = (0.0, 1.0)


@pytest.mark.parametrize("counts", [{"truncated_count": 1}, {"unboxed_count": 1}])
def test_a_training_arrival_with_one_uncertain_rollout_reserves_capacity(counts):
    assert _arrival("training", HALF, uncertain_indices=(M_ROLLOUTS - 1,), attainable_rewards=BINARY, **counts) == 1


def test_a_training_arrival_the_uncertainty_could_collapse_reserves_nothing():
    only_success_uncertain = [1.0] + [0.0] * (M_ROLLOUTS - 1)
    assert _arrival("training", only_success_uncertain, uncertain_indices=(0,), attainable_rewards=BINARY,
                    truncated_count=1) == 0
    assert _arrival("training", only_success_uncertain) == 1


def test_the_arrival_gate_uses_the_real_uncertain_rollouts_not_a_placeholder():
    """Rollout 0 is the only success and is certain; the uncertain one is a failure elsewhere.
    A gate that marked index 0 uncertain whenever a count was set would refuse this group."""
    rewards = [1.0] + [0.0] * (M_ROLLOUTS - 1)
    assert _arrival("training", rewards, uncertain_indices=(5,), attainable_rewards=BINARY,
                    truncated_count=1, unboxed_count=1) == 1
    # counts alone carry no index: nothing is uncertain for the rule
    assert _arrival("training", rewards, truncated_count=1, unboxed_count=1) == 1
    # the legacy single index is not what the service rule reads
    assert _arrival("training", rewards, truncated_index=0, attainable_rewards=BINARY) == 1


def test_a_training_arrival_without_a_lattice_cannot_carry_uncertainty():
    assert _arrival("training", HALF, uncertain_indices=(M_ROLLOUTS - 1,)) == 0


def test_an_exploration_arrival_with_an_uncertain_rollout_is_still_an_observation_and_reserves_nothing():
    # Out of zone: an exploration observation, which never reserves fill-closed capacity.
    assert _arrival("exploration", [0.0] * M_ROLLOUTS, uncertain_indices=(3,), truncated_indices=(3,),
                    attainable_rewards=BINARY, truncated_count=1) == 0
    # The same vector declared for training is the same lane.
    assert _arrival("training", [0.0] * M_ROLLOUTS, uncertain_indices=(3,), truncated_indices=(3,),
                    attainable_rewards=BINARY, truncated_count=1) == 0
    # An in-zone vector is a training group whatever it declares.
    assert _arrival("exploration", HALF, uncertain_indices=(3,), attainable_rewards=BINARY) == 1


def test_the_worker_result_reaches_the_pending_group_with_its_indices():
    from reliquary.validator.admission import PreparedSubmission
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    batcher = _make_batcher()
    request = _request(prompt_idx=21, hotkey="hk")
    prepared = PreparedSubmission(request=request, completion_texts=[], rewards=[r.reward for r in request.rollouts],
                                  rollout_hashes=[], selection_digest=None, prompt_content_sha256="a" * 64,
                                  target_content_sha256="b" * 64, truncated_count=1, truncated_indices=(2,),
                                  uncertain_indices=(2, 5), attainable_rewards=BINARY)
    assert batcher.accept_prepared_submission(prepared).accepted
    pending = batcher.pending_submissions()[-1]
    assert (pending.truncated_indices, pending.uncertain_indices) == ((2,), (2, 5))
    assert pending.attainable_rewards == BINARY
    plain = PreparedSubmission(request=_request(prompt_idx=22, hotkey="hk2"), completion_texts=[],
                               rewards=[r.reward for r in request.rollouts], rollout_hashes=[], selection_digest=None,
                               prompt_content_sha256="c" * 64, target_content_sha256="b" * 64)
    assert batcher.accept_prepared_submission(plain).accepted
    pending = batcher.pending_submissions()[-1]
    assert (pending.truncated_indices, pending.uncertain_indices) == ((), ())


def test_the_legacy_arrival_gate_never_reaches_the_service_rule(monkeypatch):
    import reliquary.services.admission_policy as policy
    import reliquary.validator.batcher as batcher_module

    def unreachable(*args, **kwargs):
        raise AssertionError("service policy code reached without a service policy")

    for name in ("service_lane", "missing_box_is_uncertain", "uncertain_rollout_indices",
                 "exploration_pay_entitlement"):
        monkeypatch.setattr(policy, name, unreachable)
    calls = []
    real = batcher_module.robust_utility_admits

    def spy(rewards, **kwargs):
        calls.append(kwargs)
        return real(rewards, **kwargs)

    monkeypatch.setattr(batcher_module, "robust_utility_admits", spy)
    from reliquary.constants import SIGMA_MIN
    # legacy reads its own single index and ignores the service-only fields
    assert _arrival("training", HALF, service=False, truncated_index=M_ROLLOUTS - 1, truncated_count=1,
                    uncertain_indices=(0, 1, 2)) == 1
    assert calls == [{"sigma_min": SIGMA_MIN, "truncated_indices": (M_ROLLOUTS - 1,),
                      "attainable_rewards": (0.0, 1.0)}]


def test_only_server_pool_beacon_and_epoch_are_authoritative(signed_request):
    request, announcement, _ = signed_request()
    for change in ({"pool_epoch": 6}, {"pool_randomness": "ef" * 32}):
        parsed, _ = _parse(request, {**announcement, **change})
        assert parsed.reject_reason is RejectReason.GENERATION_CONTRACT_MISMATCH
        assert parsed.reject_stage == "service_contract"


def test_resigned_envelope_cannot_hide_reassigned_seeds(signed_request):
    """Rollouts signed for one subset cannot be passed off as another subset's draws."""
    request, announcement, wallet = signed_request()
    request.pool_selection["seeds"] = list(OTHER)
    for index, rollout in enumerate(request.rollouts):
        rollout.commit["rollout"]["seed_pool"]["seed_index"] = OTHER[index]
    _sign_envelope(request, wallet)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is RejectReason.BAD_SIGNATURE
    assert parsed.reject_stage == "rollout_signature"


@pytest.mark.parametrize("seeds", [SEEDS, OTHER, tuple(range(M_ROLLOUTS, POOL_SEEDS)),
                                   (0, *range(M_ROLLOUTS + 1, POOL_SEEDS))])
def test_any_m_distinct_seeds_of_the_pool_are_admitted(signed_request, seeds):
    request, announcement, _ = signed_request(seeds=seeds)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is None
    assert request.pool_selection["seeds"] == list(seeds)


def test_the_validated_selection_is_the_signed_one(signed_request):
    """A consistent request for one subset under an envelope signed for another is refused."""
    request, announcement, wallet = signed_request(seeds=OTHER)
    genuine = dict(request.pool_selection)
    request.pool_selection = {**genuine, "seeds": list(SEEDS)}
    _sign_envelope(request, wallet)                      # the envelope now attests SEEDS
    request.pool_selection = genuine                     # ...but the request carries OTHER
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is RejectReason.BAD_ENVELOPE_SIGNATURE


@pytest.mark.parametrize("forge", ["duplicate", "unsorted", "out_of_range", "short", "long", "bool", "old_schema"])
def test_non_canonical_selections_never_reach_the_proof_stage(signed_request, forge):
    request, announcement, wallet = signed_request()
    # positive control: the same request, untouched, passes the very same parse path
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is None
    seeds = list(SEEDS)
    if forge == "duplicate":
        seeds[1] = seeds[0]
    elif forge == "unsorted":
        seeds[0], seeds[1] = seeds[1], seeds[0]
    elif forge == "out_of_range":
        seeds[-1] = POOL_SEEDS
    elif forge == "short":
        seeds = seeds[:-1]
    elif forge == "long":
        seeds = sorted(set(seeds) | {0})
    elif forge == "bool":
        seeds = [False, True, *seeds[2:]] if SEEDS[0] > 1 else [True, *seeds[1:]]
    request.pool_selection = ({"schema": "public-group-selection/v1", "pool_sha256": request.pool_selection["pool_sha256"],
                               "candidate_id": 1} if forge == "old_schema" else {**request.pool_selection, "seeds": seeds})
    for index, rollout in enumerate(request.rollouts):
        if forge != "old_schema" and index < len(seeds) and type(seeds[index]) is int and 0 <= seeds[index] < 128:
            rollout.commit["rollout"]["seed_pool"]["seed_index"] = seeds[index]
    with pytest.raises(ValueError):
        validate_submission_policy(request, announcement)
    with pytest.raises(ValueError):                       # the wire schema refuses it as well
        BatchSubmissionRequest.model_validate(request.model_dump())
    # the real admission parse path
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is RejectReason.BAD_SCHEMA
    assert parsed.reject_stage == "schema"                # at the schema stage, long before signatures and proofs


def test_two_rollouts_cannot_claim_the_same_seed(signed_request):
    request, announcement, wallet = signed_request()
    first = request.rollouts[0].commit["rollout"]["seed_pool"]["seed_index"]
    request.rollouts[1].commit["rollout"]["seed_pool"]["seed_index"] = first
    _sign_envelope(request, wallet)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is RejectReason.GENERATION_CONTRACT_MISMATCH
    assert parsed.reject_stage == "service_contract"       # before signatures, long before proofs


def test_validator_draws_each_rollout_from_its_seed_not_its_rank(signed_request):
    draws = {}
    for seeds in (SEEDS, (0, *SEEDS[:-1])):               # SEEDS[k] sits at rank k, then at rank k + 1
        request, announcement, _ = signed_request(seeds=seeds)
        seen = []
        def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
            seen.append(seed_u_values)
            return _service_proof(commit)
        batcher = _deep_service_batcher(request, announcement, verifier)
        assert _prove_one(batcher, request) is not None
        pool = SeedPool.from_contract(_contract(), environment=OMI, prompt_idx=7, checkpoint_hash="d" * 40,
                                      pool_epoch=5, randomness="ab" * 32)
        assert seen == [[pool.uniform(seed, j) for j in range(CHALLENGE_K)] for seed in seeds]
        draws[seeds] = dict(zip(seeds, seen))
    for seed in SEEDS[:-1]:
        assert draws[SEEDS][seed] == draws[(0, *SEEDS[:-1])][seed]


def test_service_group_id_is_the_selection_digest_not_the_miner(signed_request):
    from reliquary.protocol.seed_pool import PoolSelection
    from reliquary.validator.batcher import GrpoWindowBatcher

    def group_id(request):
        return GrpoWindowBatcher._service_group_id(SimpleNamespace(request=request, selection_digest=b"\x01" * 32))
    a, _, _ = signed_request(seeds=SEEDS)
    b, _, _ = signed_request(seeds=SEEDS)
    b.miner_hotkey, b.nonce, b.window_start = "another-miner", "another-nonce", 12
    for index, rollout in enumerate(b.rollouts):
        rollout.commit["tokens"][5] = 50 + index            # another miner's completions for the same seeds
    c, _, _ = signed_request(seeds=OTHER)
    assert group_id(a) == group_id(b) == PoolSelection.from_dict(a.pool_selection).sha256
    assert group_id(c) != group_id(a)
    legacy, _, _ = signed_request(pool=False)
    assert group_id(legacy) not in (group_id(a), group_id(c))


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
    request.rollouts[0].commit["rollout"]["seed_pool"] = {"schema": "public-seed-rollout/v2",
        "pool_sha256": "aa" * 32, "seed_index": 1, "rollout_index": 0}
    with pytest.raises(ValueError, match="not allowed"):
        validate_submission_policy(request, announcement)


def test_announced_capability_cannot_enable_unsupported_draw_variant(signed_request):
    from reliquary.protocol.service_contract import ServiceContractError

    request, announcement, _ = signed_request(pool=False)
    value = announcement["contract"]
    value["environments"][OMI]["sampling"] = {"kind": "public-draw-pool/v1", "group_size": M_ROLLOUTS,
                                              "pool_draws": M_ROLLOUTS + 1, "renewal_windows": 2}
    announcement["supported_capabilities"].append("public-draw-pool/v1")
    with pytest.raises(ServiceContractError):
        validate_submission_policy(request, announcement)


def _prove_one(batcher, request, audit=True, admission_caps=()):
    """The lane-independent proof checks (seed gates, capability, termination, logprobs) in audit mode:
    what an exploration audit runs. These tests send a uniform group, which is no training group."""
    if not batcher.accept_submission(request).accepted:
        return None
    pending = batcher.pending_submissions()[-1]
    if admission_caps:   # caps the (worker) admission already saw, before the proof finds its own
        pending.truncated_indices = tuple(admission_caps)
        pending.uncertain_indices = tuple(sorted({*pending.uncertain_indices, *admission_caps}))
    return batcher._verify_expensive(pending, audit=audit)


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
    batcher.service_environment = request.rollouts[0].env_name
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
                  seed_n_boundary_match=size)
    values.update(changes)
    return ProofResult(**values)


def _proved_with_a_cap(signed_request, monkeypatch, *, purpose, correct, capped=True, unboxed=(),
                       missing_box="uncertain", admission_caps=()):
    """Prove a group through the in-process path. ``capped``: the proof alone finds the last
    rollout cut by the length cap. ``unboxed``: rollouts that terminated without any box."""
    from reliquary.validator import batcher as batcher_module
    request, announcement, wallet = signed_request(purpose=purpose, missing_box=missing_box)
    for index, rollout in enumerate(request.rollouts):
        rollout.reward = float(index < correct and index not in unboxed)
    _sign_envelope(request, wallet)
    def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        return _service_proof(commit)
    batcher = _deep_service_batcher(request, announcement, verifier)
    def decode(ids, **kwargs):
        index = next((i for i in range(M_ROLLOUTS) if 10 + i in ids), None)
        if index is None:
            return "".join("\\boxed{0}" if token == 2 else "" if token == 99 else "x" for token in ids)
        return f"Derivation number {index} {'CORRECT' if index < correct else 'wrong'}. \\boxed{{0}}"
    batcher.tokenizer.decode = decode
    graded_text = batcher._completion_text
    def completion_text(rollout):
        index = next(i for i, candidate in enumerate(request.rollouts) if candidate is rollout)
        return f"wrong {index}, and no final answer" if index in unboxed else graded_text(rollout)
    batcher._completion_text = completion_text
    # ``capped``: True = the last rollout; a tuple = those rollout positions; False = none.
    cut = (M_ROLLOUTS - 1,) if capped is True else tuple(capped or ())
    cut_commits = [request.rollouts[index].commit for index in cut]
    monkeypatch.setattr(batcher_module, "is_cap_truncation",
                        lambda commit, *args, **kwargs: any(commit is c for c in cut_commits))
    result = _prove_one(batcher, request, audit=purpose == "exploration", admission_caps=admission_caps)
    pending = batcher.pending_submissions()[-1] if batcher.pending_submissions() else None
    return result, batcher, pending


def test_a_cap_found_by_the_proof_no_longer_sinks_a_robust_training_group(signed_request, monkeypatch):
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                                  correct=M_ROLLOUTS // 2)
    assert result is not None
    assert batcher.reject_counts.get(RejectReason.OUT_OF_ZONE.value, 0) == 0
    assert pending.truncated_indices == (M_ROLLOUTS - 1,)
    assert pending.uncertain_indices == (M_ROLLOUTS - 1,)
    assert result.truncated_count == 1


def test_a_cap_found_by_the_proof_refuses_a_training_group_it_could_collapse(signed_request, monkeypatch):
    """All right but the last, which the proof finds cut: it may have been right too."""
    result, batcher, _ = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                            correct=M_ROLLOUTS - 1)
    assert result is None
    assert batcher.reject_counts[RejectReason.OUT_OF_ZONE.value] == 1


def test_two_capped_rollouts_are_no_longer_a_bad_termination_on_the_service_path(signed_request, monkeypatch):
    """R21: the legacy count limit (more than N capped rollouts) does not apply; the robust rule decides."""
    from reliquary.constants import MAX_TRUNCATED_PER_SUBMISSION
    two = (M_ROLLOUTS - 2, M_ROLLOUTS - 1)
    assert len(two) > MAX_TRUNCATED_PER_SUBMISSION        # the legacy limit WOULD reject this group
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                                  correct=M_ROLLOUTS // 2, capped=two)
    assert result is not None and result.truncated_count == 2
    assert batcher.reject_counts.get(RejectReason.BAD_TERMINATION.value, 0) == 0
    assert pending.truncated_indices == two and pending.uncertain_indices == two
    # the robust rule still prices both: all right but the two cut -> they may both have been right
    result, batcher, _ = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                            correct=M_ROLLOUTS - 2, capped=two)
    assert result is None
    assert batcher.reject_counts[RejectReason.OUT_OF_ZONE.value] == 1
    assert batcher.reject_counts.get(RejectReason.BAD_TERMINATION.value, 0) == 0


def test_the_caps_admission_saw_and_the_caps_the_proof_finds_are_one_union(signed_request, monkeypatch):
    """Two successes only: with rollout 0 uncertain the group is robust (it may have been one success),
    with rollout 1 uncertain too (found by the proof) both may have failed, i.e. a uniform group."""
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training", correct=2,
                                                  capped=False, admission_caps=(0,))
    assert result is not None and pending.truncated_indices == (0,)                  # admission's cap alone: robust
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training", correct=2,
                                                  capped=(1,), admission_caps=(0,))
    assert result is None and batcher.reject_counts[RejectReason.OUT_OF_ZONE.value] == 1
    assert pending.truncated_indices == (0, 1) and pending.uncertain_indices == (0, 1)   # the union, no duplicates
    result, _, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training", correct=M_ROLLOUTS // 2,
                                            capped=(0, M_ROLLOUTS - 1), admission_caps=(0,))
    assert result is not None and pending.truncated_indices == (0, M_ROLLOUTS - 1)    # a cap seen twice counts once


def test_in_process_missing_box_follows_the_contract(signed_request, monkeypatch):
    """Direct (non worker) path: all right but one, which terminated without a box."""
    last = (M_ROLLOUTS - 1,)
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                                  correct=M_ROLLOUTS, capped=False, unboxed=last)
    assert result is None                                         # maths: it may have been right
    # R23: no longer refused at admission (it is an unproven observation); a TRAINING proof of it refuses
    assert pending.proof_reject_stage == "service_signal"
    assert batcher.reject_counts[RejectReason.OUT_OF_ZONE.value] == 1
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                                  correct=M_ROLLOUTS, capped=False, unboxed=last,
                                                  missing_box="graded")
    assert result is not None                                     # plain zero: 15/16 is in zone
    assert pending.uncertain_indices == () and pending.unboxed_count == 0
    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="training",
                                                  correct=M_ROLLOUTS // 2, capped=False, unboxed=last)
    assert result is not None                                     # maths, robust to that rollout
    assert pending.uncertain_indices == last and pending.truncated_indices == ()
    assert pending.attainable_rewards == (0.0, 1.0)


def test_a_cap_found_by_the_proof_keeps_the_exploration_observation_and_marks_it(signed_request, monkeypatch):
    """Published, and recognisable as unpaid: the cut rollout is recorded for the pay decision."""
    from reliquary.services.admission_policy import exploration_pay_entitlement

    result, batcher, pending = _proved_with_a_cap(signed_request, monkeypatch, purpose="exploration", correct=0)
    assert result is not None
    assert batcher.reject_counts.get(RejectReason.OUT_OF_ZONE.value, 0) == 0
    assert pending.truncated_indices == (M_ROLLOUTS - 1,)
    pay = exploration_pay_entitlement(pending.rewards, truncated_indices=pending.truncated_indices,
                                      uncertain_indices=pending.uncertain_indices)
    assert (pay.entitled, pay.reason) == (False, "truncated")


@pytest.mark.parametrize("pool", [False, True])
def test_deep_service_refuses_verifier_without_seed_interface(signed_request, pool):
    request, announcement, _ = signed_request(pool=pool)
    def old_verifier(commit, model, randomness):
        return _service_proof(commit)
    batcher = _deep_service_batcher(request, announcement, old_verifier)
    assert _prove_one(batcher, request) is None
    assert batcher.reject_counts[RejectReason.GENERATION_CONTRACT_MISMATCH.value] == 1


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("missing", ["sparse", "positions"])
def test_deep_service_refuses_missing_sampling_evidence(signed_request, pool, missing):
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


# ---- O1(a): the rollout length is bounded at admission on the service path (every lane) -------------------
# The schema already ties completion_length to the largest per-environment cap; the environment's own bound can
# be smaller, which is what a long-completion payload would exploit. Shrink it to make the group oversize.

@pytest.fixture
def small_bound(monkeypatch):
    from reliquary import constants
    monkeypatch.setitem(constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV, OMI, CHALLENGE_K - 1)


@pytest.mark.parametrize("purpose", ["training", "exploration"])
def test_o1_a_service_group_with_a_rollout_longer_than_the_environment_bound_is_refused_at_admission(
        signed_request, small_bound, purpose):
    request, announcement, _ = signed_request(purpose=purpose)
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason == RejectReason.BAD_TOKENS
    assert parsed.reject_stage == "service_length"


def test_o1_a_a_rollout_at_the_bound_is_not_refused_for_its_length(signed_request, monkeypatch):
    from reliquary import constants
    monkeypatch.setitem(constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV, OMI, CHALLENGE_K)
    request, announcement, _ = signed_request()
    parsed, _ = _parse(request, announcement)
    assert parsed.reject_reason is None


def test_o1_a_the_length_rule_is_the_environments_bound_on_the_completion(signed_request, small_bound):
    from reliquary.validator.admission import service_length_valid
    request, announcement, _ = signed_request()
    rollout = request.rollouts[0]
    assert not service_length_valid(rollout.commit["tokens"], rollout.commit["rollout"], OMI)


def test_service_length_valid_episode_branch_uses_the_episode_bound_not_the_completion_bound(monkeypatch):
    from reliquary import constants
    from reliquary.validator import admission
    from reliquary.validator.admission import service_length_valid
    monkeypatch.setattr(admission, "episode_limits_for_environment", lambda environment: (4, 10, 1000))
    monkeypatch.setitem(constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV, OMI, 1)
    meta = {"prompt_length": 2, "episode": {"turns": []}, "completion_length": 8}
    assert service_length_valid(list(range(10)), meta, OMI)          # 10 tokens fit the episode bound (the cap of 1 is not used)
    assert not service_length_valid(list(range(11)), meta, OMI)
    monkeypatch.setattr(admission, "episode_limits_for_environment", lambda environment: None)
    assert not service_length_valid(list(range(10)), meta, OMI)       # no episode profile: falls back to the completion bound


def test_service_length_valid_missing_prompt_length_is_zero_and_a_bad_one_is_refused(monkeypatch):
    from reliquary import constants
    from reliquary.validator.admission import service_length_valid
    monkeypatch.setitem(constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV, OMI, 5)
    assert service_length_valid([1, 2, 3, 4, 5], {}, OMI)            # prompt_length defaults to 0: 5 completion tokens
    assert not service_length_valid([1, 2, 3, 4, 5, 6], {}, OMI)
    for bad in (-1, 6, "x", None, [1], 10 ** 400):
        assert not service_length_valid([1, 2, 3, 4, 5], {"prompt_length": bad}, OMI), bad


def test_service_length_valid_completion_length_claim_is_parsed_strictly(monkeypatch):
    from reliquary import constants
    from reliquary.validator.admission import service_length_valid
    monkeypatch.setitem(constants.MAX_NEW_TOKENS_PROTOCOL_CAP_BY_ENV, OMI, 5)
    tokens = [1, 2, 3, 4, 5]
    assert service_length_valid(tokens, {"prompt_length": 0, "completion_length": "4"}, OMI)   # a numeric string is read as a number
    assert not service_length_valid(tokens, {"prompt_length": 0, "completion_length": "6"}, OMI)
    for bad in ("abc", None, [3], {"a": 1}, 10 ** 400):
        assert not service_length_valid(tokens, {"prompt_length": 0, "completion_length": bad}, OMI), bad


# ---------------------------------------------------------------- R31 amended (F1): the classification pass

def _two_pass(signed_request, monkeypatch, *, logprob=(), distribution=(), authenticity=(), toploc=(), proof=None):
    """A real ``_verify_expensive`` audit of a service exploration group whose gates fail on the named rollouts
    (rollout ``i`` carries token ``10 + i`` at position 5), then its classification pass."""
    from reliquary.validator import batcher as batcher_module

    request, announcement, _ = signed_request(purpose="exploration")
    index = lambda tokens: tokens[5] - 10
    def verifier(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        i = index(commit["tokens"])
        changes = dict((proof or {}).get(i, {}))
        if i in toploc:
            changes.update(toploc_checked=True, toploc_passed=False)
        return _service_proof(commit, **changes)
    batcher = _deep_service_batcher(request, announcement, verifier)
    monkeypatch.setattr(batcher_module, "proof_rejection",
                        lambda profile, *, grail_passed, toploc_passed:
                        "toploc_fail" if toploc_passed is False else None if grail_passed else "grail_fail")
    monkeypatch.setattr(batcher_module, "verify_logprobs_claim",
                        lambda **kw: (False, 9.0) if index(kw["tokens"]) in logprob else (True, 0.0))
    monkeypatch.setattr(batcher_module, "evaluate_token_distribution",
                        lambda **kw: (False, {}) if index(kw["tokens"]) in distribution else (True, {}))
    monkeypatch.setattr(batcher_module, "evaluate_token_authenticity",
                        lambda proof, **kw: (False, {}) if index(kw["tokens"]) in authenticity else (True, {}))
    shadows = []
    real_shadow = batcher_module.record_forced_seed_shadow
    monkeypatch.setattr(batcher_module, "record_forced_seed_shadow",
                        lambda *a, **k: (shadows.append(1), real_shadow(*a, **k))[1])
    assert _prove_one(batcher, request, audit=True) is None                     # the audit fails
    pending = batcher.pending_submissions()[-1]
    first = (pending.proof_reject_stage, pending.proof_reject_scope)
    before = (dict(batcher.reject_counts), batcher.proof_failure_debt(request.miner_hotkey), pending.reject_response,
              pending.truncated_indices, pending.uncertain_indices, len(shadows))
    pending.service_observation_id = "ab" * 32
    klass = batcher._classify_audit_failure(pending, model=None)
    # the classification pass changes no verdict, no count, no debt, no forensic record, no lane
    assert (dict(batcher.reject_counts), batcher.proof_failure_debt(request.miner_hotkey), pending.reject_response,
            pending.truncated_indices, pending.uncertain_indices, len(shadows)) == before
    assert (pending.proof_reject_stage, pending.proof_reject_scope) == first
    return first, (pending.proof_classify_stage, pending.proof_classify_scope), klass


def test_r31_f1_rollout_0_fails_logprob_and_rollout_1_fails_toploc_deterministic(signed_request, monkeypatch):
    first, second, klass = _two_pass(signed_request, monkeypatch, logprob={0}, toploc={1})
    assert first == ("logprob", None)                                     # the miner chose what fails first
    assert second == ("toploc", None) and klass == "deterministic"


def test_r31_f1_a_forged_token_where_distribution_runs_before_authenticity_is_deterministic(signed_request, monkeypatch):
    first, second, klass = _two_pass(signed_request, monkeypatch, distribution={2}, authenticity={2})
    assert first == ("distribution", None)
    assert second == ("token_authenticity", None) and klass == "deterministic"


def test_r31_f1_an_honest_group_with_one_statistical_failure_is_statistical(signed_request, monkeypatch):
    first, second, klass = _two_pass(signed_request, monkeypatch, logprob={0})
    assert first == ("logprob", None)
    assert second == (None, None) and klass == "statistical"


def test_r31_f1_the_post_loop_hard_cdf_mismatch_is_checked_in_the_classification_pass(signed_request, monkeypatch):
    first, second, klass = _two_pass(signed_request, monkeypatch, logprob={0},
                                     proof={3: {"seed_n_hard_mismatch": 1}})
    assert first == ("logprob", None)                                     # the audit never reached the seed gate
    assert second == ("forced_seed", "cdf_hard_mismatch") and klass == "deterministic"


def test_r31_f1_the_agreement_floors_are_off_in_the_classification_pass(signed_request, monkeypatch):
    first, second, klass = _two_pass(signed_request, monkeypatch, logprob={0}, proof={3: {"seed_n_match": 0}})
    assert first == ("logprob", None)
    assert second == (None, None) and klass == "statistical"
    # and the floor alone is a statistical audit failure that the classification pass confirms as such
    first, second, klass = _two_pass(signed_request, monkeypatch, proof={3: {"seed_n_match": 0}})
    assert first[0] == "forced_seed" and first[1] in ("group", "rollout")
    assert second == (None, None) and klass == "statistical"
