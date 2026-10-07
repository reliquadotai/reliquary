import copy
from types import SimpleNamespace
import pytest

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.service_submission import ServiceBinding, validate_service_rollout_bindings
from reliquary.protocol.seed_pool import PoolSelection
from reliquary.protocol.signatures import build_commit_binding, build_envelope_binding, build_service_commit_binding
from reliquary.protocol.submission import BatchSubmissionRequest, CommitModel, GrpoBatchState, MinerState, RolloutSubmission, WindowState


def _commit(index=0, *, service=True, pool=False):
    size = CHALLENGE_K + 1
    metadata = {"prompt_length": 1, "completion_length": size - 1, "success": False,
                "total_reward": 0.0, "advantage": 0.0, "token_logprobs": [0.0] * (size - 1)}
    if service:
        metadata["service_binding"] = ServiceBinding("aa" * 32, "exploration").rollout_binding(index)
    if pool:
        metadata["seed_pool"] = PoolSelection("bb" * 32, 1).rollout_binding(index)
    return {"tokens": list(range(size)), "commitments": [{}] * size,
            "proof_version": "public-group-proof/v1" if pool else "service-group-proof/v1" if service else "v7",
            "model": {"name": "test", "layer_index": 1}, "signature": "aa" * 64,
            "beacon": {"randomness": "ab" * 32}, "rollout": metadata}


def _envelope(**changes):
    return build_envelope_binding(miner_hotkey="miner", window_start=4, prompt_idx=7,
                                  merkle_root="aa" * 32, checkpoint_hash="d" * 40,
                                  drand_round=9, randomness="ab" * 32, nonce="fresh", **changes)


def test_service_and_pool_intent_changes_both_signed_domains():
    intent = ServiceBinding("aa" * 32, "exploration")
    pool = PoolSelection("bb" * 32, 1)
    base = _envelope()
    bound = _envelope(service_binding=intent.to_dict())
    both = _envelope(service_binding=intent.to_dict(), pool_selection=pool.to_dict())
    assert len({base, bound, both, _envelope(pool_selection=pool.to_dict())}) == 4
    for change in (ServiceBinding("cc" * 32, "exploration"), ServiceBinding("aa" * 32, "training")):
        assert _envelope(service_binding=change.to_dict(), pool_selection=pool.to_dict()) != both
    parts = ([1, 2], "ab" * 32, "test", 1, [{}, {}])
    signed = build_service_commit_binding(*parts, intent.rollout_binding(0), pool.rollout_binding(0))
    assert signed != build_commit_binding(*parts)
    for change in (intent.rollout_binding(1), ServiceBinding("aa" * 32, "training").rollout_binding(0)):
        if change["rollout_index"] == 1:
            with pytest.raises(ValueError):
                build_service_commit_binding(*parts, change, pool.rollout_binding(0))
        else:
            assert build_service_commit_binding(*parts, change, pool.rollout_binding(0)) != signed


def test_schema_refuses_unsigned_service_fields_and_reindexed_pool():
    for pool in (False, True):
        value = _commit(pool=pool)
        CommitModel.model_validate(value)
        for version in ("v7", "v8"):
            with pytest.raises(ValueError):
                CommitModel.model_validate({**value, "proof_version": version})
    invalid = _commit(pool=True)
    invalid["rollout"]["seed_pool"]["rollout_index"] = 1
    with pytest.raises(ValueError, match="indices differ"):
        CommitModel.model_validate(invalid)


def test_every_service_rollout_matches_envelope_in_original_order():
    binding = ServiceBinding("aa" * 32, "exploration")
    commits = [_commit(i) for i in range(M_ROLLOUTS)]
    validate_service_rollout_bindings(binding, commits)
    for invalid in (commits[::-1], [{**commits[0], "rollout": {}}] + commits[1:]):
        with pytest.raises(ValueError):
            validate_service_rollout_bindings(binding, invalid)
    altered = copy.deepcopy(commits)
    altered[0]["rollout"]["service_binding"]["purpose"] = "training"
    with pytest.raises(ValueError):
        validate_service_rollout_bindings(binding, altered)
    altered[0]["rollout"]["service_binding"]["rollout_index"] = False
    with pytest.raises(ValueError):
        validate_service_rollout_bindings(binding, altered)


def test_optional_fields_remain_absent_in_legacy_serialization():
    commit = CommitModel.model_validate(_commit(service=False))
    assert "seed_pool" not in commit.model_dump()["rollout"]
    assert "service_binding" not in commit.model_dump()["rollout"]
    rollouts = [RolloutSubmission(tokens=commit.tokens, reward=0, commit=commit.model_dump(), env_name="math")
                for _ in range(M_ROLLOUTS)]
    request = BatchSubmissionRequest(miner_hotkey="hk", prompt_idx=7, window_start=4,
                                     merkle_root="aa" * 32, rollouts=rollouts, checkpoint_hash="d" * 40)
    assert "pool_selection" not in request.model_dump()
    assert "service_binding" not in request.model_dump()
    state = GrpoBatchState(state=WindowState.OPEN, window_n=1, anchor_block=1, valid_submissions=0)
    miner = MinerState(state=WindowState.OPEN, window_n=1, anchor_block=1, environments={})
    assert "service_policy" not in state.model_dump()
    assert "service_policy" not in miner.model_dump()


@pytest.mark.parametrize("pool", [False, True])
def test_real_sr25519_service_proof_and_envelope_reject_mutated_intent(monkeypatch, pool):
    from bittensor_wallet import Keypair
    from reliquary.protocol import signatures

    # Fixed test-only key; never an operational wallet.
    key = Keypair.create_from_seed("0x" + "01" * 32)
    wallet = SimpleNamespace(hotkey=key)
    monkeypatch.setattr(signatures, "bt", SimpleNamespace(Keypair=Keypair))
    commit = _commit(pool=pool)
    metadata = commit["rollout"]
    commit["signature"] = signatures.sign_service_commit_binding(
        commit["tokens"], commit["beacon"]["randomness"], "test", 1,
        commit["commitments"], metadata["service_binding"], wallet,
        seed_pool=metadata.get("seed_pool"),
    ).hex()
    assert signatures.verify_commit_signature(commit, key.ss58_address)
    for field, value in (("purpose", "training"), ("contract_sha256", "cc" * 32), ("rollout_index", 1)):
        altered = copy.deepcopy(commit)
        altered["rollout"]["service_binding"][field] = value
        assert not signatures.verify_commit_signature(altered, key.ss58_address)
    assert not signatures.verify_commit_signature({**commit, "proof_version": "v7"}, key.ss58_address)
    if pool:
        altered = copy.deepcopy(commit)
        altered["rollout"]["seed_pool"]["candidate_id"] = 2
        assert not signatures.verify_commit_signature(altered, key.ss58_address)
    fields = dict(miner_hotkey=key.ss58_address, window_start=4, prompt_idx=7,
                  merkle_root="aa" * 32, checkpoint_hash="d" * 40, drand_round=9,
                  randomness="ab" * 32, nonce="fresh",
                  service_binding=ServiceBinding("aa" * 32, "exploration").to_dict(),
                  pool_selection=PoolSelection("bb" * 32, 1).to_dict() if pool else None)
    sig = signatures.sign_envelope(wallet=wallet, **fields).hex()
    assert signatures.verify_envelope_signature(envelope_signature=sig, **fields)
    assert not signatures.verify_envelope_signature(envelope_signature=sig,
        **{**fields, "service_binding": ServiceBinding("aa" * 32, "training").to_dict()})
    assert not signatures.verify_envelope_signature(envelope_signature=sig,
        **{**fields, "service_binding": None})
