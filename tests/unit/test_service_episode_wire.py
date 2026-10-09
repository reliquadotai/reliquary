"""The signed-episode wire: precommit, engagement, commit metadata, bindings."""
import copy
import json
from types import SimpleNamespace

import pytest

from reliquary.protocol.seed_pool import SeedPoolError, validate_rollout_selection
from reliquary.protocol.service_episode import (
    EpisodePrecommit, EpisodeWireError, episode_commit_material, episode_without_transcript,
    is_signed_episode, parse_rl_engagement, rl_engagement,
)
from reliquary.protocol.service_submission import ServiceBinding, validate_service_rollout_bindings
from reliquary.protocol.signatures import build_service_episode_commit_binding
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA, CommitModel
from reliquary.services.admission_policy import validate_submission_policy
from tests.unit.episode_v2_fixtures import (
    EPISODE, REVISION, TASK, WINDOW_BEACON, episode_block, episode_contract, episode_contract_dict,
    episode_pool, episode_precommit,
    episode_runtime, signed_episode_commit, signed_episode_metadata,
)
from tests.unit.service_v2_fixtures import MATH

TRANSCRIPT = {"token": {"claims": {}}, "records": [{"body": {"i": 0}}]}
TOKENS = list(range(10, 50))          # CommitModel needs at least CHALLENGE_K tokens
SPANS = [(5, 15), (20, 40)]           # 30 model tokens, an observation segment in between


def _group_commits():
    contract = episode_contract()
    pool = episode_pool(contract)
    selection = pool.selection(list(range(pool.group_size)))
    commits = [signed_episode_commit(
        tokens=TOKENS, spans=SPANS, selection=selection, index=index, contract=contract,
        episode=signed_episode_metadata(precommit_sha256="e" * 64, seed_index=seed, spans=SPANS,
                                        transcript=copy.deepcopy(TRANSCRIPT)))
        for index, seed in enumerate(selection.seeds)]
    return contract, pool, selection, commits


def test_a_precommit_round_trips_and_its_digest_is_stable():
    contract = episode_contract()
    precommit = episode_precommit(contract, hotkey="5Hot")
    again = EpisodePrecommit.from_dict(json.loads(json.dumps(precommit.to_dict())))
    assert again == precommit and again.sha256 == precommit.sha256
    assert precommit.to_dict()["schema"] == "reliquary/episode-precommit/v1"
    assert precommit.pool_sha256 == episode_pool(contract).sha256


@pytest.mark.parametrize("field, value", [
    ("order", "x"), ("window", -1), ("window", True), ("environment", "Bad-Env"), ("task_index", 1.0),
    ("checkpoint", "d" * 64), ("pool_sha256", "A" * 64), ("hotkey", ""), ("schema", "other"),
])
def test_a_malformed_precommit_is_refused(field, value):
    value_dict = {**episode_precommit(episode_contract(), hotkey="5Hot").to_dict(), field: value}
    with pytest.raises(EpisodeWireError):
        EpisodePrecommit.from_dict(value_dict)


def test_a_precommit_with_an_extra_field_is_refused():
    value = {**episode_precommit(episode_contract(), hotkey="5Hot").to_dict(), "nonce": 1}
    with pytest.raises(EpisodeWireError):
        EpisodePrecommit.from_dict(value)


def test_the_engagement_names_window_precommit_and_seed():
    sha = "e" * 64
    assert rl_engagement(7, sha, 12) == f"rl:7:{sha}:12"
    assert parse_rl_engagement(rl_engagement(7, sha, 12)) == (7, sha, 12)


@pytest.mark.parametrize("text", [
    "corpus:job:1", "rl:7:" + "e" * 64, "rl:07:" + "e" * 64 + ":1", "rl:7:" + "E" * 64 + ":1",
    "rl:7:" + "e" * 64 + ":-1", None,
])
def test_a_malformed_engagement_is_refused(text):
    with pytest.raises(EpisodeWireError):
        parse_rl_engagement(text)


def test_a_signed_episode_commit_validates():
    _, _, _, commits = _group_commits()
    parsed = CommitModel.model_validate(commits[0])
    assert parsed.rollout.episode.schema_version == SIGNED_EPISODE_SCHEMA
    assert parsed.rollout.episode.assistant_token_count == 30


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["rollout"].pop("seed_pool"), "public seed pool"),
    (lambda c: c.update(proof_version="v8"), "proof version"),
    (lambda c: c["rollout"]["episode"].update(seed_index=c["rollout"]["episode"]["seed_index"] + 1),
     "own pool seed"),
    (lambda c: c["rollout"]["episode"].update(assistant_spans=[[6, 15], [20, 40]]), "right after the prompt"),
    (lambda c: c["rollout"]["episode"].update(assistant_spans=[[5, 15], [20, 39]]), "ends the tokens"),
    (lambda c: c["rollout"]["episode"].update(assistant_spans=[[5, 15], [12, 40]]), "sorted"),
    (lambda c: c["rollout"].update(token_logprobs=[-1.0] * 7), "token_logprobs"),
    (lambda c: c["rollout"].update(forced=True), "forced span"),
])
def test_a_signed_episode_commit_that_breaks_a_binding_is_refused(mutate, message):
    _, _, _, commits = _group_commits()
    commit = commits[0]
    mutate(commit)
    with pytest.raises(ValueError, match=message):
        CommitModel.model_validate(commit)


def test_a_legacy_episode_v1_is_unchanged_and_still_refused_with_a_seed_pool():
    from tests.unit.test_episode_protocol import _episode_commit

    commit = _episode_commit()
    CommitModel.model_validate(commit)
    commit["rollout"]["seed_pool"] = {"schema": "public-seed-rollout/v2", "pool_sha256": "e" * 64,
                                      "seed_index": 0, "rollout_index": 0}
    with pytest.raises(ValueError, match="service group proofs do not support episodes"):
        CommitModel.model_validate(commit)


def test_the_pool_takes_signed_episodes_only_for_an_episode_environment():
    _, pool, selection, commits = _group_commits()
    validate_rollout_selection(pool, selection, commits, signed_episodes=True)
    with pytest.raises(SeedPoolError, match="single-turn"):
        validate_rollout_selection(pool, selection, commits)
    swapped = copy.deepcopy(commits)
    swapped[0]["rollout"]["episode"]["seed_index"] = selection.seeds[1]
    with pytest.raises(SeedPoolError, match="chosen seed"):
        validate_rollout_selection(pool, selection, swapped, signed_episodes=True)
    plain = copy.deepcopy(commits)
    for commit in plain:
        del commit["rollout"]["episode"]
    with pytest.raises(SeedPoolError, match="signed episodes"):
        validate_rollout_selection(pool, selection, plain, signed_episodes=True)


def test_the_service_binding_takes_signed_episodes_only_when_asked():
    contract, _, _, commits = _group_commits()
    binding = ServiceBinding(contract.sha256, "training")
    validate_service_rollout_bindings(binding, commits, signed_episodes=True)
    with pytest.raises(ValueError, match="single-turn"):
        validate_service_rollout_bindings(binding, commits)


def _request(contract, selection, commits, *, env=EPISODE):
    return SimpleNamespace(service_binding=ServiceBinding(contract.sha256, "training").to_dict(),
                           pool_selection=selection.to_dict(), checkpoint_hash=REVISION, prompt_idx=TASK,
                           rollouts=[SimpleNamespace(commit=commit, env_name=env) for commit in commits])


def test_the_submission_policy_routes_signed_episodes_to_episode_environments_only(tmp_path):
    rt = episode_runtime(tmp_path)
    announcement = rt.announcement(window=1, randomness=WINDOW_BEACON)
    contract, _, selection, commits = _group_commits()
    assert validate_submission_policy(_request(contract, selection, commits), announcement).sha256 == contract.sha256
    with pytest.raises(ValueError):
        validate_submission_policy(_request(contract, selection, commits, env=MATH), announcement)
    legacy = SimpleNamespace(service_binding=None, pool_selection=None, rollouts=[
        SimpleNamespace(commit={"rollout": {"episode": commit["rollout"]["episode"]}}, env_name=EPISODE)
        for commit in commits])
    with pytest.raises(ValueError, match="active service task"):
        validate_submission_policy(legacy, None)


def test_the_commit_binding_covers_the_episode_and_its_transcript():
    _, _, _, commits = _group_commits()
    commit = commits[0]
    meta = commit["rollout"]

    def binding(episode):
        return build_service_episode_commit_binding(commit["tokens"], "cd" * 32, "model", -1, commit["commitments"],
                                                    meta["service_binding"], meta["seed_pool"], episode)

    base = binding(meta["episode"])
    other = copy.deepcopy(meta["episode"])
    other["transcript"]["records"][0]["body"]["i"] = 1
    assert binding(other) != base
    moved = copy.deepcopy(meta["episode"])
    moved["stop"] = "max_turns"
    assert binding(moved) != base
    assert episode_commit_material(meta["episode"]) == episode_commit_material(copy.deepcopy(meta["episode"]))


def test_a_real_signature_verifies_and_breaks_with_the_transcript():
    bt = pytest.importorskip("bittensor")
    from reliquary.protocol.signatures import sign_service_episode_commit_binding, verify_commit_signature

    keypair = bt.Keypair.create_from_uri("//Alice")
    _, _, _, commits = _group_commits()
    commit = commits[0]
    meta = commit["rollout"]
    commit["signature"] = sign_service_episode_commit_binding(
        commit["tokens"], commit["beacon"]["randomness"], commit["model"]["name"], commit["model"]["layer_index"],
        commit["commitments"], meta["service_binding"], meta["seed_pool"], meta["episode"],
        SimpleNamespace(hotkey=keypair)).hex()
    assert verify_commit_signature(commit, keypair.ss58_address)
    legacy_shaped = copy.deepcopy(commit)
    legacy_shaped["rollout"]["episode"]["schema_version"] = "reliquary/episode/v1"
    assert not verify_commit_signature(legacy_shaped, keypair.ss58_address)
    commit["rollout"]["episode"]["transcript"]["records"][0]["body"]["i"] = 9
    assert not verify_commit_signature(commit, keypair.ss58_address)


def test_the_helpers_detect_and_strip_the_transcript():
    _, _, _, commits = _group_commits()
    meta = commits[0]["rollout"]
    assert is_signed_episode(meta)
    assert not is_signed_episode({"episode": {"schema_version": "reliquary/episode/v1"}})
    stripped = episode_without_transcript(meta)
    assert "transcript" not in stripped["episode"] and "transcript" in meta["episode"]
    assert stripped["episode"]["assistant_spans"] == meta["episode"]["assistant_spans"]


# --- Strictness: canonical encodings, domain separation, legacy bytes ---------------------------------

@pytest.mark.parametrize("mutate", [
    lambda e: e.update(seed_index=True),
    lambda e: e.update(seed_index=1.0),
    lambda e: e.update(assistant_spans=[["5", 15], [20, 40]]),
    lambda e: e.update(assistant_spans=[[5.0, 15], [20, 40]]),
    lambda e: e.update(assistant_spans=[[True, 15], [20, 40]]),
    lambda e: e.update(assistant_spans=[[5, 15, 16], [20, 40]]),
    lambda e: e.update(precommit_sha256="E" * 64),
    lambda e: e.update(nonce=1),
    lambda e: e["transcript"].update(x=float("nan")),
])
def test_a_signed_episode_has_one_canonical_encoding(mutate):
    _, _, _, commits = _group_commits()
    mutate(commits[0]["rollout"]["episode"])
    with pytest.raises(ValueError):
        CommitModel.model_validate(commits[0])


def test_a_signed_episode_seed_must_be_an_integer_in_the_group_validators():
    contract, pool, selection, commits = _group_commits()
    commits[1]["rollout"]["episode"]["seed_index"] = float(selection.seeds[1])
    with pytest.raises(SeedPoolError, match="chosen seed"):
        validate_rollout_selection(pool, selection, commits, signed_episodes=True)


def test_a_legacy_episode_is_refused_where_signed_episodes_are_required():
    contract, pool, selection, commits = _group_commits()
    for commit in commits:
        commit["rollout"]["episode"]["schema_version"] = "reliquary/episode/v1"
    with pytest.raises(SeedPoolError, match="signed episodes"):
        validate_rollout_selection(pool, selection, commits, signed_episodes=True)
    with pytest.raises(ValueError, match="signed episodes"):
        validate_service_rollout_bindings(ServiceBinding(contract.sha256, "training"), commits,
                                          signed_episodes=True)


def test_single_turn_groups_are_refused_where_signed_episodes_are_required():
    contract, _, _, commits = _group_commits()
    for commit in commits:
        del commit["rollout"]["episode"]
    with pytest.raises(ValueError, match="signed episodes"):
        validate_service_rollout_bindings(ServiceBinding(contract.sha256, "training"), commits,
                                          signed_episodes=True)


def test_every_signing_domain_is_prefix_free():
    """Each signed object hashes ``domain || len-prefixed parts``: prefix-free domains keep every
    signature valid for exactly one kind of object."""
    from reliquary.protocol import signatures

    domains = {name: value for name, value in vars(signatures).items()
               if name.endswith("_DOMAIN") or name.endswith("_DOMAIN_V3")}
    assert domains["SERVICE_EPISODE_COMMIT_DOMAIN"] == b"service-episode-commit/v1"
    assert domains["EPISODE_PRECOMMIT_DOMAIN"] == b"reliquary/episode-precommit/v1"
    assert len(set(domains.values())) == len(domains)
    for name, value in domains.items():
        for other_name, other in domains.items():
            if name != other_name:
                assert not other.startswith(value), (name, other_name)


def test_the_episode_commit_binding_differs_from_the_single_turn_one():
    from reliquary.protocol.signatures import build_service_commit_binding

    _, _, _, commits = _group_commits()
    commit = commits[0]
    meta = commit["rollout"]
    args = (commit["tokens"], "cd" * 32, "model", -1, commit["commitments"], meta["service_binding"],
            meta["seed_pool"])
    assert build_service_episode_commit_binding(*args, meta["episode"]) != build_service_commit_binding(*args)


def test_an_episode_without_its_transcript_cannot_be_bound():
    _, _, _, commits = _group_commits()
    with pytest.raises(EpisodeWireError):
        episode_commit_material(episode_without_transcript(commits[0]["rollout"])["episode"])


def test_the_engagement_refuses_non_canonical_inputs():
    with pytest.raises(EpisodeWireError):
        rl_engagement(True, "e" * 64, 1)
    with pytest.raises(EpisodeWireError):
        rl_engagement(1, "e" * 64, -1)
    with pytest.raises(EpisodeWireError):
        parse_rl_engagement("rl:7:" + "e" * 64 + ":1:2")
    with pytest.raises(EpisodeWireError):
        parse_rl_engagement("rl:+7:" + "e" * 64 + ":1")


def test_the_precommit_binding_is_canonical_and_audience_bound():
    from reliquary.protocol.signatures import build_episode_precommit_binding

    precommit = episode_precommit(episode_contract(), hotkey="5Hot")
    base = build_episode_precommit_binding(precommit.to_dict(), at=100, validator_hotkey="5Val", path="/rl/p")
    assert build_episode_precommit_binding(precommit, at=100, validator_hotkey="5Val", path="/rl/p") == base
    assert build_episode_precommit_binding(precommit.to_dict(), at=101, validator_hotkey="5Val",
                                           path="/rl/p") != base
    assert build_episode_precommit_binding(precommit.to_dict(), at=100, validator_hotkey="5Other",
                                           path="/rl/p") != base
    assert build_episode_precommit_binding(precommit.to_dict(), at=100, validator_hotkey="5Val",
                                           path="/rl/q") != base
    for bad in ({**precommit.to_dict(), "nonce": 1}, {**precommit.to_dict(), "window": 1.0}):
        with pytest.raises(EpisodeWireError):
            build_episode_precommit_binding(bad, at=100, validator_hotkey="5Val", path="/rl/p")


def test_a_real_precommit_signature_verifies_only_for_its_own_hotkey_and_object():
    bt = pytest.importorskip("bittensor")
    from reliquary.protocol.signatures import (
        build_episode_precommit_binding, verify_episode_precommit_signature, verify_hotkey_signature,
    )

    miner = bt.Keypair.create_from_uri("//Alice")
    other = bt.Keypair.create_from_uri("//Dave")
    precommit = episode_precommit(episode_contract(), hotkey=miner.ss58_address).to_dict()
    kwargs = {"at": 100, "validator_hotkey": "5Val", "path": "/rl/episodes/precommit"}
    signature = miner.sign(build_episode_precommit_binding(precommit, **kwargs)).hex()
    assert verify_episode_precommit_signature(miner.ss58_address, precommit, signature=signature, **kwargs)
    assert not verify_episode_precommit_signature(miner.ss58_address, {**precommit, "window": 2},
                                                  signature=signature, **kwargs)
    assert not verify_episode_precommit_signature(miner.ss58_address, {**precommit, "x": 1},
                                                  signature=signature, **kwargs)
    assert not verify_episode_precommit_signature(miner.ss58_address, precommit, signature=signature,
                                                  **{**kwargs, "at": -1})
    # A precommit naming another hotkey never verifies, even signed by its signer.
    foreign = episode_precommit(episode_contract(), hotkey=other.ss58_address).to_dict()
    foreign_sig = miner.sign(build_episode_precommit_binding(foreign, **kwargs)).hex()
    assert verify_hotkey_signature(miner.ss58_address, build_episode_precommit_binding(foreign, **kwargs),
                                   foreign_sig)
    assert not verify_episode_precommit_signature(miner.ss58_address, foreign, signature=foreign_sig, **kwargs)


def test_a_signed_episode_never_verifies_under_another_proof_version():
    bt = pytest.importorskip("bittensor")
    from reliquary.protocol.signatures import (
        sign_episode_commit_binding, sign_service_episode_commit_binding, verify_commit_signature,
    )

    keypair = bt.Keypair.create_from_uri("//Alice")
    wallet = SimpleNamespace(hotkey=keypair)
    _, _, _, commits = _group_commits()
    commit = commits[0]
    meta = commit["rollout"]
    args = (commit["tokens"], commit["beacon"]["randomness"], "model", -1, commit["commitments"])
    signed = copy.deepcopy(commit)
    signed["signature"] = sign_service_episode_commit_binding(
        *args, meta["service_binding"], meta["seed_pool"], meta["episode"], wallet).hex()
    assert verify_commit_signature(signed, keypair.ss58_address)
    for version in ("v7", "v8", "service-group-proof/v1"):
        assert not verify_commit_signature({**signed, "proof_version": version}, keypair.ss58_address)
    no_pool = copy.deepcopy(signed)
    del no_pool["rollout"]["seed_pool"]
    assert not verify_commit_signature(no_pool, keypair.ss58_address)
    # A legacy v8 signature over the same episode dict, stripped of its pool, is not a signed episode's.
    legacy = copy.deepcopy(commit)
    legacy["proof_version"] = "v8"
    del legacy["rollout"]["seed_pool"]
    del legacy["rollout"]["service_binding"]
    legacy["signature"] = sign_episode_commit_binding(*args, meta["episode"], wallet).hex()
    assert not verify_commit_signature(legacy, keypair.ss58_address)
    moved = {**legacy, "proof_version": "public-group-proof/v1",
             "rollout": {**legacy["rollout"], "seed_pool": meta["seed_pool"],
                         "service_binding": meta["service_binding"]}}
    assert not verify_commit_signature(moved, keypair.ss58_address)


# Fixed inputs, digests computed on the tree before signed episodes (5dd82b56): legacy and single-turn v2
# commits must keep binding to exactly these bytes.
_SEED_POOL = {"schema": "public-seed-rollout/v2",
              "pool_sha256": "2b1728b34ff4a702376e6d1515f159ecd007c6a98e0268225a4829e3faa57cb6",
              "seed_index": 1, "rollout_index": 1}
_SERVICE = {"schema": "service-rollout/v1",
            "contract_sha256": "6f81d8fd5e1494dd5f543a18a1b10c2d26768a726ba2a8c87b405d57666c2359",
            "purpose": "training", "rollout_index": 1}


def test_legacy_and_single_turn_commit_bindings_are_byte_identical():
    from reliquary.protocol.signatures import (
        build_commit_binding, build_episode_commit_binding, build_public_group_commit_binding,
        build_service_commit_binding,
    )
    from tests.unit.test_episode_protocol import _episode_commit

    tokens = list(range(10, 50))
    commitments = [{"i": i} for i in tokens]
    args = (tokens, "cd" * 32, "model", -1, commitments)
    assert build_commit_binding(*args).hex() == (
        "0f43c7ce5033fc5221c17b0c85396cdcc288eebdcb4ac49e32de2462242512b8")
    assert build_public_group_commit_binding(*args, _SEED_POOL).hex() == (
        "2e9bb17c828664ffc4f0d7a664945ada55b488281c7845e747405d897b42fc9c")
    assert build_service_commit_binding(*args, _SERVICE).hex() == (
        "f8cc1c69e70b4bbc3783fbd0d49882975a41d88ce213e34be0a8eab46bbdd9b6")
    assert build_service_commit_binding(*args, _SERVICE, _SEED_POOL).hex() == (
        "9ae2961c19c86706dca04beaaae68275ea8ffd0de9059192d6e2f796af00695d")
    legacy = _episode_commit()
    assert build_episode_commit_binding(legacy["tokens"], "aa", "model", 0, legacy["commitments"],
                                        legacy["rollout"]["episode"]).hex() == (
        "9dc31f6e7336e0b9ac53aad51cf0fcb20111f78c584b40cc34444cda685adad0")


def _dump_sha(commit) -> str:
    import hashlib

    dumped = CommitModel.model_validate(commit).model_dump(mode="json")
    return hashlib.sha256(json.dumps(dumped, sort_keys=True).encode()).hexdigest()


def test_legacy_and_single_turn_commits_serialise_as_before():
    from tests.unit.test_episode_protocol import _episode_commit

    assert _dump_sha(_episode_commit()) == "c6ab29ba911d744c54c7eded98cab61cc53fa2115571114f963c8bb47fdae1dd"
    tokens = list(range(10, 50))
    single = {"tokens": tokens, "commitments": [{"i": i} for i in tokens], "proof_version": "public-group-proof/v1",
              "model": {"name": "model", "layer_index": -1}, "signature": "aa", "beacon": {"randomness": "cd" * 32},
              "rollout": {"prompt_length": 5, "completion_length": 35, "success": True, "total_reward": 1.0,
                          "advantage": 0.0, "token_logprobs": [-1.0] * 35, "seed_pool": _SEED_POOL,
                          "service_binding": _SERVICE}}
    assert _dump_sha(single) == "205e29d03d6b61373f6795bf6287f9e4d57f9de71997f47174e88f1b510b162c"


@pytest.mark.parametrize("hotkey", ["5Hot ", "5H0t", "5Hél", "5" * 65])
def test_a_precommit_hotkey_is_a_bounded_base58_string(hotkey):
    with pytest.raises(EpisodeWireError):
        episode_precommit(episode_contract(), hotkey=hotkey)


def test_only_a_signed_episode_has_commit_material():
    _, _, _, commits = _group_commits()
    legacy_shaped = {**commits[0]["rollout"]["episode"], "schema_version": "reliquary/episode/v1"}
    with pytest.raises(EpisodeWireError):
        episode_commit_material(legacy_shaped)


def _long_commit(contract, completion):
    """A signed-episode commit with ``completion`` tokens after a 5-token prompt (stub proofs)."""
    pool = episode_pool(contract)
    selection = pool.selection(list(range(pool.group_size)))
    tokens = list(range(5 + completion))
    spans = [(5, len(tokens))]
    commits = [signed_episode_commit(
        tokens=tokens, spans=spans, selection=selection, index=index, contract=contract,
        episode=signed_episode_metadata(precommit_sha256="e" * 64, seed_index=seed, spans=spans,
                                        transcript=copy.deepcopy(TRANSCRIPT)))
        for index, seed in enumerate(selection.seeds)]
    return selection, commits


def _long_contract():
    from reliquary.protocol.service_contract import ServiceContract

    return ServiceContract.from_dict(episode_contract_dict(episode=episode_block(max_episode_tokens=40000)))


def test_a_long_signed_episode_is_accepted_under_a_contract_allowing_it(tmp_path):
    contract = _long_contract()
    selection, commits = _long_commit(contract, 40000)
    assert CommitModel.model_validate(commits[0]).rollout.completion_length == 40000
    announcement = episode_runtime(tmp_path, contract).announcement(window=1, randomness=WINDOW_BEACON)
    assert validate_submission_policy(_request(contract, selection, commits), announcement).sha256 == contract.sha256


def test_a_signed_episode_over_the_contract_limit_is_refused(tmp_path):
    contract = _long_contract()
    selection, commits = _long_commit(contract, 40001)
    CommitModel.model_validate(commits[0])          # the wire allows it, the order does not
    announcement = episode_runtime(tmp_path, contract).announcement(window=1, randomness=WINDOW_BEACON)
    with pytest.raises(ValueError, match="max_episode_tokens"):
        validate_submission_policy(_request(contract, selection, commits), announcement)


def test_a_signed_episode_is_bounded_by_the_trajectory_cap_on_the_wire():
    contract = _long_contract()
    _, commits = _long_commit(contract, 60001)
    with pytest.raises(ValueError, match="completion_length"):
        CommitModel.model_validate(commits[0])


def test_a_legacy_rollout_keeps_its_completion_cap():
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    contract = _long_contract()
    for completion, ok in ((MAX_NEW_TOKENS_PROTOCOL_CAP, True), (MAX_NEW_TOKENS_PROTOCOL_CAP + 1, False)):
        _, commits = _long_commit(contract, completion)
        commit = commits[0]
        for key in ("episode", "seed_pool", "service_binding"):
            commit["rollout"].pop(key)
        commit["proof_version"] = "v7"
        commit["rollout"]["token_logprobs"] = [-1.0] * len(commit["tokens"])
        if ok:
            CommitModel.model_validate(commit)
        else:
            with pytest.raises(ValueError, match="completion_length"):
                CommitModel.model_validate(commit)


def _math_group():
    """Episode commits built on MATH's own pool and selection: only the env gate can refuse them."""
    from reliquary.protocol.seed_pool import SeedPool

    contract = episode_contract()
    pool = SeedPool.from_contract(contract, environment=MATH, prompt_idx=TASK, checkpoint_hash=REVISION,
                                  pool_epoch=1, randomness=WINDOW_BEACON)
    selection = pool.selection(list(range(pool.group_size)))
    commits = [signed_episode_commit(
        tokens=TOKENS, spans=SPANS, selection=selection, index=index, contract=contract,
        episode=signed_episode_metadata(precommit_sha256="e" * 64, seed_index=seed, spans=SPANS,
                                        transcript=copy.deepcopy(TRANSCRIPT)))
        for index, seed in enumerate(selection.seeds)]
    return contract, selection, commits


def test_a_single_turn_environment_refuses_signed_episodes_through_the_env_gate(tmp_path):
    contract, selection, commits = _math_group()
    announcement = episode_runtime(tmp_path, contract).announcement(window=1, randomness=WINDOW_BEACON)
    with pytest.raises(ValueError, match="single-turn"):
        validate_submission_policy(_request(contract, selection, commits, env=MATH), announcement)


def test_a_v1_contract_does_not_touch_the_environments_block():
    from reliquary.protocol.service_submission import ServiceBinding as Binding

    class V1:
        version = 1
        sha256 = "ab" * 32

        @property
        def environments(self):
            raise AssertionError("environments read on a v1 contract")

        def environment(self, name):
            raise ValueError("reached the per-environment policy")

    _, _, selection, commits = _group_commits()
    for index, commit in enumerate(commits):
        del commit["rollout"]["episode"]
        commit["rollout"]["service_binding"] = Binding(V1.sha256, "training").rollout_binding(index)
    request = _request(SimpleNamespace(sha256=V1.sha256), selection, commits)
    request.service_binding = Binding(V1.sha256, "training").to_dict()
    schedule = SimpleNamespace(active_environments=lambda: {EPISODE})
    announcement = {"checkpoint": {"revision": REVISION}}
    with pytest.raises(ValueError, match="reached the per-environment policy"):
        validate_submission_policy(request, announcement, parsed=(V1(), schedule))


def test_the_engagement_bounds_are_the_same_both_ways():
    sha = "e" * 64
    assert parse_rl_engagement(rl_engagement(0, sha, 127)) == (0, sha, 127)
    with pytest.raises(EpisodeWireError):
        rl_engagement(1, sha, 128)
    with pytest.raises(EpisodeWireError):
        parse_rl_engagement(f"rl:1:{sha}:128")
    with pytest.raises(EpisodeWireError):
        parse_rl_engagement(f"rl:{2**53}:{sha}:1")
    assert parse_rl_engagement(f"rl:{2**53 - 1}:{sha}:1")[0] == 2**53 - 1


@pytest.mark.parametrize("stop", ["agent\x00", "agent\n", "Agent", "agent-completed", "a" * 33, "", "agent completed"])
def test_the_episode_stop_condition_is_a_bounded_lowercase_token(stop):
    _, _, _, commits = _group_commits()
    commits[0]["rollout"]["episode"]["stop"] = stop
    with pytest.raises(ValueError):
        CommitModel.model_validate(commits[0])
