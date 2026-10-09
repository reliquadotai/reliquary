"""Admission wiring for signed-episode groups (plan 2C, Task 8)."""
import asyncio
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.constants import M_ROLLOUTS  # noqa: E402
from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.validator import admission, episode_intake  # noqa: E402
from reliquary.validator.admission import (  # noqa: E402
    AdmissionContext, AdmissionProblemMaterials, ParsedSubmission, PreparedSubmission,
)
from reliquary.validator.episode_admission import EpisodeGroupChecker  # noqa: E402
from reliquary.validator.episode_intake import EpisodeClaim, EpisodeGroupIntake  # noqa: E402
from reliquary.validator.server import ValidatorServer, _episode_admission_fields  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, WINDOW_BEACON, FixedSource, batch_request, episode_contract, episode_group, episode_runtime,
    episode_signers, register_episode_env,
)
from tests.unit.sandbox_fixtures import NOW, directory  # noqa: E402
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_trajectory_parse import FakeRenderer  # noqa: E402

CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)


class Sessions:
    """The issuer's surface the intake uses; ``claim_all`` is all-or-none like ``SessionIssuer.claim_all``."""

    def __init__(self, refuse=None):
        self.claimed, self.released, self.paid, self.held = [], [], [], set()
        self.refuse = dict(refuse or {})
        self.fail_release = False
        self.group_keys = []

    async def claim_all(self, session_ids, *, hotkey, received=None, precommit_sha256=None):
        self.group_keys.append(precommit_sha256)
        for session_id in session_ids:
            if session_id in self.refuse:
                return session_id, SimpleNamespace(reason=self.refuse[session_id], retry_after=None, detail={})
        self.claimed.extend(session_ids)
        self.held.update(session_ids)
        return None

    async def release_claim(self, session_id):
        if self.fail_release:
            raise RuntimeError("store down")
        self.released.append(session_id)
        self.held.discard(session_id)

    async def submitted(self, session_id):
        self.paid.append(session_id)
        self.held.discard(session_id)


class Outcomes:
    def __init__(self):
        self.calls = []

    def group_settled(self, **kwargs):
        self.calls.append(kwargs)


def world(tmp_path, *, refuse=None, directory_ready=True, precommits=None, verifier_keys=None, seen=frozenset,
          intake_kwargs=None, **group_kwargs):
    validator, machine = episode_signers(tmp_path)
    group = episode_group(CONTRACT, validator=validator, machine=machine, **group_kwargs)
    sessions, outcomes = Sessions(refuse), Outcomes()
    known = {group.precommit.sha256: group.precommit} if precommits is None else precommits
    intake = EpisodeGroupIntake(
        checkers={EPISODE: EpisodeGroupChecker(policy=POLICY, renderer=FakeRenderer(), source=FixedSource(),
                                               chunk_tokens=32)},
        precommits=known.get, directory=lambda now: directory(machine) if directory_ready else None,
        token_verifier=attest.Ed25519TokenVerifier(
            verifier_keys if verifier_keys is not None else {validator.key_id: validator.public_key_b64}),
        sessions=sessions, seen=seen, outcomes=outcomes, **(intake_kwargs or {}))
    request = batch_request(group)
    prepared = PreparedSubmission(request=request, completion_texts=[], rewards=[], rollout_hashes=[],
                                  selection_digest=b"d" * 32, episode_pending=True)
    return SimpleNamespace(group=group, intake=intake, sessions=sessions, outcomes=outcomes, prepared=prepared)


def admit(w, environment=EPISODE):
    return asyncio.run(w.intake.admit(environment=environment, prepared=w.prepared, received=NOW + 100,
                                      contract=CONTRACT))


def test_an_honest_group_is_completed_and_its_sessions_claimed(tmp_path):
    w = world(tmp_path)
    prepared, claim = admit(w)
    assert prepared.reject_reason is None and prepared.episode_pending is False
    assert prepared.rewards == w.group.rewards
    assert claim == EpisodeClaim("5Hot", w.group.precommit.sha256,
                                 tuple(f"s-{seed}" for seed in w.group.selection.seeds))
    assert w.sessions.claimed == list(claim.session_ids)
    assert w.sessions.group_keys == [w.group.precommit.sha256]      # the precommit is taken with them
    asyncio.run(w.intake.settle(claim, accepted=True))
    assert w.sessions.paid == list(claim.session_ids) and w.sessions.released == []
    assert w.outcomes.calls[-1] == {"hotkey": "5Hot", "precommit_sha256": w.group.precommit.sha256,
                                    "session_ids": claim.session_ids, "accepted": True}


def test_a_refused_decision_releases_every_claim(tmp_path):
    w = world(tmp_path)
    _, claim = admit(w)
    asyncio.run(w.intake.settle(claim, accepted=False))
    assert w.sessions.released == list(claim.session_ids) and w.sessions.paid == []
    assert w.outcomes.calls[-1]["accepted"] is False


def test_a_group_the_checker_refuses_claims_nothing(tmp_path):
    w = world(tmp_path, precommits={})
    prepared, claim = admit(w)
    assert claim is None and w.sessions.claimed == []
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.PRECOMMIT_INVALID, "episode_precommit")
    assert w.outcomes.calls == [{"hotkey": "5Hot", "precommit_sha256": w.group.precommit.sha256,
                                 "session_ids": (), "accepted": False}]


@pytest.mark.parametrize("claim_reason, reason, stage", [
    ("session_submitted", RejectReason.HASH_DUPLICATE, "episode_session_reused"),
    ("session_claimed", RejectReason.WORKER_DROPPED, "episode_session_busy"),
    ("session_busy", RejectReason.WORKER_DROPPED, "episode_session_busy"),
    ("session_expired", RejectReason.PRECOMMIT_EXPIRED, "episode_deadline"),
    ("session_not_submittable", RejectReason.REWARD_MISMATCH, "episode_transcript"),
    ("session_unknown", RejectReason.REWARD_MISMATCH, "episode_transcript"),
    ("precommit_submitted", RejectReason.HASH_DUPLICATE, "episode_precommit_used"),
    ("precommit_claimed", RejectReason.WORKER_DROPPED, "episode_session_busy"),
])
def test_a_session_that_cannot_be_claimed_refuses_the_group_holding_nothing(tmp_path, claim_reason, reason,
                                                                            stage):
    w = world(tmp_path)
    third = f"s-{w.group.selection.seeds[2]}"
    w.sessions.refuse[third] = claim_reason
    prepared, claim = admit(w)
    assert claim is None
    assert (prepared.reject_reason, prepared.reject_stage) == (reason, stage)
    assert w.sessions.held == set() and w.sessions.paid == []
    assert w.outcomes.calls[-1]["accepted"] is False and len(w.outcomes.calls[-1]["session_ids"]) == M_ROLLOUTS


def test_a_stale_directory_is_retryable(tmp_path):
    w = world(tmp_path, directory_ready=False)
    prepared, claim = admit(w)
    assert claim is None
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.WORKER_DROPPED, "episode_directory")


def test_an_environment_this_intake_does_not_serve_is_refused(tmp_path):
    w = world(tmp_path)
    prepared, claim = admit(w, environment=MATH)
    assert claim is None
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.GENERATION_CONTRACT_MISMATCH,
                                                               "episode_unserved")


def test_a_lane_refusal_after_the_claim_releases_every_session(tmp_path, monkeypatch):
    w = world(tmp_path)

    def out_of_zone(prepared, facts, contract):
        prepared.reject_reason, prepared.reject_stage = RejectReason.OUT_OF_ZONE, "zone"

    monkeypatch.setattr(episode_intake, "finish_prepared", out_of_zone)
    prepared, claim = admit(w)
    assert claim is None and prepared.reject_reason is RejectReason.OUT_OF_ZONE
    assert w.sessions.held == set() and sorted(w.sessions.released) == sorted(w.sessions.claimed)
    assert len(w.sessions.claimed) == M_ROLLOUTS and w.outcomes.calls[-1]["accepted"] is False


def test_an_exception_after_the_claim_releases_every_session_and_propagates(tmp_path, monkeypatch):
    w = world(tmp_path)

    def broken(prepared, facts, contract):
        raise RuntimeError("boom")

    monkeypatch.setattr(episode_intake, "finish_prepared", broken)
    with pytest.raises(RuntimeError, match="boom"):
        admit(w)
    assert w.sessions.held == set() and len(w.sessions.released) == M_ROLLOUTS


def test_a_failing_release_does_not_stop_the_others_or_raise(tmp_path):
    w = world(tmp_path)
    _, claim = admit(w)
    w.sessions.fail_release = True
    asyncio.run(w.intake.settle(claim, accepted=False))
    assert w.outcomes.calls[-1]["accepted"] is False


def test_the_admission_child_parses_and_leaves_the_transcripts_to_the_parent(tmp_path, monkeypatch):
    register_episode_env(monkeypatch)
    w = world(tmp_path)
    request = w.prepared.request
    assert admission.submission_interaction_matches(request, EPISODE)
    assert not admission.submission_interaction_matches(request, MATH)
    context = AdmissionContext(randomness="cd" * 32, environment=EPISODE, vocab_size=None, max_sequence_length=10**6,
                               eos_token_ids=(), canonical_force_ids=(), think_close_ids=(), bootstrap=False,
                               enforce_envelope_signature=False, enforce_legacy_merkle=False,
                               service_policy={"contract": CONTRACT.to_dict()}, signed_episode=True,
                               episode_max_tokens=POLICY.max_episode_tokens)
    parsed = ParsedSubmission(request=request, rollout_hashes=[b"h"] * M_ROLLOUTS, selection_digest=b"d" * 32)
    prepared = admission.materialize_and_score_submission(
        parsed, AdmissionProblemMaterials(problem={}, rendered_prompt=""), context, deadline_monotonic=10**12)
    assert prepared.episode_pending is True and prepared.reject_reason is None and prepared.rewards == []
    # Without the episode context the same group is never pending (the legacy grading path runs).
    plain = dataclasses.replace(context, signed_episode=False)
    assert admission.materialize_and_score_submission(
        parsed, AdmissionProblemMaterials(problem={}, rendered_prompt=""), plain,
        deadline_monotonic=10**12).episode_pending is False


def test_an_episode_v1_env_refuses_a_signed_episode_and_the_reverse(tmp_path, monkeypatch):
    from reliquary.environment import registry

    register_episode_env(monkeypatch)
    w = world(tmp_path)
    request = w.prepared.request
    v1_name = next((name for name, spec in registry.ENVIRONMENT_SPECS.items()
                    if spec.interaction_mode == "episode"), None)
    if v1_name is not None:
        assert not admission.submission_interaction_matches(request, v1_name)
    for rollout in request.rollouts:
        rollout.commit["rollout"]["episode"] = {"schema_version": "episode/v1"}
    assert not admission.submission_interaction_matches(request, EPISODE)


def test_the_length_bound_of_a_signed_episode_is_its_contracts(tmp_path):
    w = world(tmp_path)
    commit = w.prepared.request.rollouts[0].commit
    tokens, meta = list(commit["tokens"]), commit["rollout"]
    assert admission.service_length_valid(tokens, meta, EPISODE, episode_max_tokens=len(tokens))
    assert not admission.service_length_valid(tokens, meta, EPISODE, episode_max_tokens=len(tokens) - 1)
    assert not admission.service_length_valid(tokens, meta, EPISODE)


def test_only_an_episode_environment_of_a_v2_service_window_gets_the_episode_context(tmp_path):
    from tests.unit.test_service_contract import example as v1_contract_dict

    rt = episode_runtime(tmp_path)
    episode_batcher = SimpleNamespace(env=SimpleNamespace(name=EPISODE),
                                      service_policy=rt.announcement(window=1, randomness=WINDOW_BEACON))
    math_batcher = SimpleNamespace(env=SimpleNamespace(name=MATH), service_policy=episode_batcher.service_policy)
    other_batcher = SimpleNamespace(env=SimpleNamespace(name="not_in_order"),
                                    service_policy=episode_batcher.service_policy)
    legacy_batcher = SimpleNamespace(env=SimpleNamespace(name=EPISODE), service_policy=None)
    v1_batcher = SimpleNamespace(env=SimpleNamespace(name=EPISODE),
                                 service_policy={"contract": v1_contract_dict()})
    broken_batcher = SimpleNamespace(env=SimpleNamespace(name=EPISODE), service_policy={"contract": {"x": 1}})
    assert _episode_admission_fields(episode_batcher) == {"signed_episode": True,
                                                          "episode_max_tokens": POLICY.max_episode_tokens}
    assert _episode_admission_fields(math_batcher) == {}
    assert _episode_admission_fields(other_batcher) == {}
    assert _episode_admission_fields(legacy_batcher) == {}
    assert _episode_admission_fields(v1_batcher) == {}
    assert _episode_admission_fields(broken_batcher) == {}


def server_and_batcher(tmp_path, **batcher_kwargs):
    rt = episode_runtime(tmp_path / "rt")
    server = ValidatorServer.__new__(ValidatorServer)
    server._episode_intake = None
    values = {"service_policy": rt.announcement(window=1, randomness=WINDOW_BEACON),
              "_service_window_contract": lambda: CONTRACT, "window_start": 1, "seal_snapshot_started": False,
              "env": SimpleNamespace(name=EPISODE), **batcher_kwargs}
    batcher = SimpleNamespace(**values)
    server._active_batchers = {EPISODE: batcher}
    return server, batcher


def run_server_admit(server, batcher, w):
    return asyncio.run(server._admit_episode_group(batcher, SimpleNamespace(environment=EPISODE), w.prepared,
                                                   SimpleNamespace(t_body_completed=NOW + 100)))


def test_the_server_refuses_an_episode_group_it_has_no_intake_for_and_delegates_otherwise(tmp_path, monkeypatch):
    register_episode_env(monkeypatch)
    w = world(tmp_path)
    server, batcher = server_and_batcher(tmp_path)
    receipt = SimpleNamespace(environment=EPISODE)
    prepared, claim = asyncio.run(server._admit_episode_group(batcher, receipt, w.prepared,
                                                              SimpleNamespace(t_body_completed=NOW + 100)))
    assert claim is None and prepared.reject_stage == "episode_unserved"
    fresh = world(tmp_path / "again")
    server._episode_intake = fresh.intake
    prepared, claim = asyncio.run(server._admit_episode_group(batcher, receipt, fresh.prepared,
                                                              SimpleNamespace(t_body_completed=NOW + 100)))
    assert prepared.reject_reason is None and claim is not None
    asyncio.run(server._settle_episode_claim(claim, accepted=True))
    assert fresh.sessions.paid == list(claim.session_ids)


def test_the_auction_admission_runs_the_episode_intake_and_settles_its_claims():
    source = inspect.getsource(ValidatorServer._process_auction_submission)
    assert 'if getattr(prepared, "episode_pending", False) is True:' in source
    assert "await self._admit_episode_group(" in source
    # Only the batcher's own acceptance pays the sessions, settled in ``finally`` (every exit).
    assert "episode_accepted = bool(response.accepted)" in source
    assert source.index("episode_accepted = bool(response.accepted)") > source.index(
        "response = batcher.accept_prepared_submission(")
    finally_block = source[source.index("        finally:\n            if episode_claim is not None:"):]
    assert finally_block.index("await self._settle_episode_claim(episode_claim, accepted=episode_accepted)") < 300
    assert source.index("await self._admit_episode_group(") < source.index("batcher.reserve_prepared_identity(")


# -- the issuer's all-or-none group claim ------------------------------------------------------------

def _issuer_with_sessions(tmp_path, count):
    from tests.unit.test_sandbox_sessions import build, open_

    env = build(tmp_path)
    grant = open_(env)
    record = env.book.get(grant.session_id)
    ids = [grant.session_id]
    for n in range(1, count):
        clone = dataclasses.replace(record, session_id=f"g-{n}", request_id=f"{n:032d}")
        env.book.add(clone)
        ids.append(clone.session_id)
    return env, ids


def test_the_issuer_claims_a_group_all_or_none(tmp_path):
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 3)
    assert asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW)) is None
    assert all(env.book.is_claimed(session_id) for session_id in ids)
    for session_id in ids:
        asyncio.run(env.issuer.release_claim(session_id))
    # One session already paid in another group: nothing of this group stays claimed.
    env.book.add(dataclasses.replace(env.book.get(ids[2]), state=SUBMITTED))
    refused = asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW))
    assert refused[0] == ids[2] and refused[1].reason == "session_submitted"
    assert not any(env.book.is_claimed(session_id) for session_id in ids)


def test_a_session_claimed_by_another_group_in_flight_refuses_the_whole_group(tmp_path):
    env, ids = _issuer_with_sessions(tmp_path, 3)
    assert asyncio.run(env.issuer.claim(ids[1], hotkey="5Hot", received=NOW)) is None
    refused = asyncio.run(env.issuer.claim_all(ids, hotkey="5Hot", received=NOW))
    assert refused[0] == ids[1] and refused[1].reason == "session_claimed"
    assert not env.book.is_claimed(ids[0]) and not env.book.is_claimed(ids[2])
    assert env.book.is_claimed(ids[1])           # the other group's claim is untouched
    twice = asyncio.run(env.issuer.claim_all([ids[0], ids[0]], hotkey="5Hot", received=NOW))
    assert twice[1].reason == "session_claimed" and not env.book.is_claimed(ids[0])


@pytest.mark.parametrize("change, reason, stage", [
    ("inactive", RejectReason.WINDOW_NOT_ACTIVE, "episode_window"),
    ("other_window", RejectReason.WINDOW_MISMATCH, "episode_window"),
    ("sealed", RejectReason.BATCH_FILLED, "episode_window_sealed"),
    ("stale_checkpoint", RejectReason.GENERATION_CONTRACT_MISMATCH, "service_contract"),
    ("no_announcement", RejectReason.GENERATION_CONTRACT_MISMATCH, "episode_unserved"),
])
def test_the_server_never_verifies_a_group_outside_the_current_window_or_policy(tmp_path, monkeypatch, change,
                                                                               reason, stage):
    register_episode_env(monkeypatch)
    w = world(tmp_path)
    server, batcher = server_and_batcher(tmp_path)
    server._episode_intake = w.intake
    if change == "inactive":
        server._active_batchers = {}
    elif change == "other_window":
        batcher.window_start = 2
    elif change == "sealed":
        batcher.seal_snapshot_started = True
    elif change == "stale_checkpoint":
        batcher.service_policy = {**batcher.service_policy,
                                  "checkpoint": {**batcher.service_policy["checkpoint"], "revision": "e" * 40}}
    else:
        batcher.service_policy = None
    calls = []
    monkeypatch.setattr(w.intake, "admit", lambda **kw: calls.append(kw))
    prepared, claim = run_server_admit(server, batcher, w)
    assert claim is None and calls == [] and w.sessions.claimed == []
    assert (prepared.reject_reason, prepared.reject_stage, prepared.episode_pending) == (reason, stage, False)


def test_a_child_refusal_is_never_pending(tmp_path):
    w = world(tmp_path)
    context = AdmissionContext(randomness="cd" * 32, environment=EPISODE, vocab_size=None, max_sequence_length=10**6,
                               eos_token_ids=(), canonical_force_ids=(), think_close_ids=(), bootstrap=False,
                               enforce_envelope_signature=False, enforce_legacy_merkle=False,
                               service_policy={"contract": CONTRACT.to_dict()}, signed_episode=True,
                               episode_max_tokens=POLICY.max_episode_tokens)
    parsed = ParsedSubmission(request=w.prepared.request, rollout_hashes=[], selection_digest=None,
                              reject_reason=RejectReason.GENERATION_CONTRACT_MISMATCH, reject_stage="service_contract")
    prepared = admission.materialize_and_score_submission(
        parsed, AdmissionProblemMaterials(problem={}, rendered_prompt=""), context, deadline_monotonic=10**12)
    assert prepared.episode_pending is False
    assert prepared.reject_reason is RejectReason.GENERATION_CONTRACT_MISMATCH


def test_a_token_signed_by_another_validator_is_refused(tmp_path):
    from tests.unit.sandbox_fixtures import signer

    (tmp_path / "other").mkdir(parents=True)
    other = signer(tmp_path / "other", "o", "v1")
    w = world(tmp_path, verifier_keys={other.key_id: other.public_key_b64})
    prepared, claim = admit(w)
    assert claim is None and w.sessions.claimed == []
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_a_session_in_the_paid_set_is_refused_before_any_claim(tmp_path):
    validator, machine = episode_signers(tmp_path / "probe")
    probe = episode_group(CONTRACT, validator=validator, machine=machine)
    paid = frozenset({f"s-{probe.selection.seeds[0]}"})
    w = world(tmp_path, seen=lambda: paid)
    prepared, claim = admit(w)
    assert claim is None and w.sessions.claimed == []
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.HASH_DUPLICATE, "episode_session_reused")


def test_a_checker_of_another_contract_refuses_until_rebuilt(tmp_path):
    w = world(tmp_path)
    stale = EpisodeGroupChecker(policy=dataclasses.replace(POLICY, max_turns=POLICY.max_turns + 1),
                                renderer=FakeRenderer(), source=FixedSource(), chunk_tokens=32)
    good = w.intake._checkers[EPISODE]
    w.intake.set_checkers({EPISODE: stale})
    prepared, claim = admit(w)
    assert claim is None and w.sessions.claimed == []
    assert (prepared.reject_reason, prepared.reject_stage) == (RejectReason.GENERATION_CONTRACT_MISMATCH,
                                                               "episode_policy_stale")
    w.intake.set_checkers({EPISODE: good})
    w.prepared.reject_reason = w.prepared.reject_stage = None
    prepared, claim = admit(w)
    assert prepared.reject_reason is None and claim is not None


def test_transcript_checks_are_rate_limited_per_hotkey(tmp_path):
    now = {"t": 0.0}
    w = world(tmp_path, intake_kwargs={"max_checks_per_minute": 1, "clock": lambda: now["t"]})
    _, claim = admit(w)
    assert claim is not None
    asyncio.run(w.intake.settle(claim, accepted=False))
    prepared, claim = admit(w)
    assert claim is None and (prepared.reject_reason, prepared.reject_stage) == (RejectReason.RATE_LIMITED,
                                                                                "episode_rate")
    now["t"] = 61.0
    w.prepared.reject_reason = w.prepared.reject_stage = None
    assert admit(w)[1] is not None


def test_checks_in_flight_are_bounded_per_hotkey_and_run_off_the_loop(tmp_path, monkeypatch):
    import threading

    w = world(tmp_path, intake_kwargs={"max_checks_in_flight": 1})
    checker = w.intake._checkers[EPISODE]
    real, gate, threads = checker.check, threading.Event(), []

    def slow(*args, **kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(checker, "check", slow)
    second = dataclasses.replace(w.prepared)

    async def both():
        first = asyncio.create_task(w.intake.admit(environment=EPISODE, prepared=w.prepared, received=NOW + 100,
                                                   contract=CONTRACT))
        await asyncio.sleep(0.05)
        refused = await w.intake.admit(environment=EPISODE, prepared=second, received=NOW + 100, contract=CONTRACT)
        gate.set()
        return await first, refused

    (prepared, claim), (refused, none) = asyncio.run(both())
    assert claim is not None and none is None and threads == [False]
    assert (refused.reject_reason, refused.reject_stage) == (RejectReason.RATE_LIMITED, "episode_rate")


def test_the_checker_builder_renders_the_contracts_tools(monkeypatch):
    from reliquary.environment import agentic_swe
    from reliquary.validator.episode_intake import build_episode_checker

    seen = {}
    monkeypatch.setattr(agentic_swe, "load_turn_renderer",
                        lambda path, tools: seen.update(path=path, tools=tools) or FakeRenderer())
    checker = build_episode_checker(POLICY, checkpoint_dir="/ck", source=FixedSource(), chunk_tokens=32)
    assert seen == {"path": "/ck", "tools": tuple(POLICY.tools)} and checker.policy == POLICY


def test_one_paid_group_per_precommit_even_on_disjoint_seeds(tmp_path):
    from reliquary.protocol.service_episode import rl_engagement
    from reliquary.sandbox.sessions import SUBMITTED

    env, ids = _issuer_with_sessions(tmp_path, 4)
    sha = "a" * 64
    for n, session_id in enumerate(ids):
        env.book.add(dataclasses.replace(env.book.get(session_id), engagement=rl_engagement(1, sha, n),
                                         kind="rl_precommit"))
    first, second = ids[:2], ids[2:]
    assert asyncio.run(env.issuer.claim_all(first, hotkey="5Hot", received=NOW, precommit_sha256=sha)) is None
    busy = asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256=sha))
    assert busy[1].reason == "precommit_claimed" and busy[1].retry_after
    assert not any(env.book.is_claimed(session_id) for session_id in second)
    env.book.add(dataclasses.replace(env.book.get(first[0]), state=SUBMITTED))
    for session_id in first:
        asyncio.run(env.issuer.release_claim(session_id))
    used = asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256=sha))
    assert used[1].reason == "precommit_submitted"
    assert not any(env.book.is_claimed(session_id) for session_id in second)
    # Another precommit is not affected.
    assert asyncio.run(env.issuer.claim_all(second, hotkey="5Hot", received=NOW, precommit_sha256="b" * 64)) is None
