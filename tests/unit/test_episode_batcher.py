"""The batcher proves, audits and releases signed-episode groups (plan 2C, Task 10)."""
import dataclasses
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, TOPLOC_DEPLOYED_DEFAULTS  # noqa: E402
from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.validator import batcher as batcher_module  # noqa: E402
from reliquary.validator.batcher import (  # noqa: E402
    GrpoWindowBatcher, PendingSubmission, _proof_commit, signed_episode_proof_refusal,
)
from reliquary.validator.verifier import ProofResult, policy_token_positions  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, PROMPT_TEXT, TASK, WINDOW_BEACON, REVISION, batch_request, episode_block, episode_contract,
    episode_group, episode_runtime, episode_signers, make_test_episode_env,
)
from tests.unit.test_grpo_window_batcher import _make_batcher  # noqa: E402
from tests.unit.test_trajectory_parse import TERM, TEXT  # noqa: E402

SHADOW_TOPLOC = dataclasses.replace(TOPLOC_DEPLOYED_DEFAULTS, mode="shadow")
TOPLOC_PROFILE = dataclasses.replace(ACTIVE_PROTOCOL_PROFILE,
                                     proofs=(*ACTIVE_PROTOCOL_PROFILE.proofs, SHADOW_TOPLOC))


@pytest.fixture(autouse=True)
def toploc_profile(monkeypatch):
    """The validator's profile names TOPLOC (shadow mode): a signed episode needs it whatever the mode."""
    monkeypatch.setattr(batcher_module, "_active_profile", lambda: TOPLOC_PROFILE)


class EpisodeEnv:
    name = EPISODE
    interaction_mode = "signed_episode"

    def __len__(self):
        return 1000

    def get_problem(self, idx):
        return {"prompt": "p", "ground_truth": "", "id": "x"}

    def compute_reward(self, problem, completion):
        raise TypeError("never scored from text")


def _stub(*, toploc_checked=True, toploc_passed=True, stops=True, seen=None, uniforms=None, cdf_miss=None,
          seed_short=0):
    def verify(commit, model, randomness, *, tokenizer=None, seed_u_values=None):
        if seen is not None:
            seen.append(commit)
        if uniforms is not None:
            uniforms.append(list(seed_u_values))
        count = len(policy_token_positions(commit["tokens"], commit["rollout"]))
        return ProofResult(all_passed=True, passed=1, checked=1, has_sparse_outputs=True,
                           toploc_checked=toploc_checked, toploc_passed=toploc_passed,
                           episode_stop_picks_ok=stops, seed_n_positions=count - seed_short,
                           episode_stop_first_bad_turn=0 if stops is False else None,
                           episode_stop_cdf_miss=cdf_miss,
                           completion_chosen_probs=[0.5] * count)
    return verify


def world(tmp_path, verify, *, contract=None):
    rt = episode_runtime(tmp_path / "rt", contract)
    validator, machine = episode_signers(tmp_path / "keys")
    group = episode_group(rt.contract, validator=validator, machine=machine)
    request = batch_request(group)
    for rollout in request.rollouts:
        rollout._validated_assistant_spans = tuple(
            tuple(span) for span in rollout.commit["rollout"]["episode"]["assistant_spans"])
    b = _make_batcher(window_start=1, env=EpisodeEnv(), verify_commitment_proofs_fn=verify)
    b.service_runtime, b.service_environment = rt, EPISODE
    b.service_policy = rt.announcement(window=1, randomness=WINDOW_BEACON, environment=EPISODE)
    b.current_checkpoint_hash = REVISION
    pending = PendingSubmission(hotkey="5Hot", prompt_idx=TASK, request=request, rewards=list(group.rewards),
                                drand_round=0, merkle_root=b"\0" * 32, selection_digest=b"\0" * 32)
    return SimpleNamespace(rt=rt, b=b, pending=pending, request=request, group=group)


def _eos(monkeypatch, ids):
    monkeypatch.setattr("reliquary.shared.modeling.resolve_eos_token_ids", lambda model, tokenizer: set(ids))


def test_the_proof_never_receives_the_transcript_nor_a_carve_out(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    seen = []
    w = world(tmp_path, _stub(seen=seen))
    first = w.request.rollouts[0].commit["rollout"]
    first["forced"], first["force_span"] = True, [first["prompt_length"], first["prompt_length"] + 4]
    w.b._verify_expensive(w.pending)
    assert seen                                # rollout 0 at least (a later phase 1 gate may stop the rest)
    for commit in seen:
        assert "transcript" not in commit["rollout"]["episode"]
        assert "forced" not in commit["rollout"] and "force_span" not in commit["rollout"]
        assert commit["toploc_spec"] == SHADOW_TOPLOC.to_contract()
        assert commit["rollout"]["episode"]["assistant_spans"]
    # The miner's commit is never mutated: admission and the payload release still read it.
    assert all("transcript" in rollout.commit["rollout"]["episode"] for rollout in w.request.rollouts)
    assert first["forced"] is True
    legacy = {"tokens": [1, 2], "rollout": {"prompt_length": 1, "forced": True}}
    assert _proof_commit(legacy, ACTIVE_PROTOCOL_PROFILE) is legacy
    v1 = {"tokens": [1, 2], "rollout": {"prompt_length": 1, "episode": {"transcript": "kept", "x": 1}}}
    assert _proof_commit(v1, ACTIVE_PROTOCOL_PROFILE) is v1


@pytest.mark.parametrize("stub, eos, stage, reason", [
    (dict(toploc_checked=False), {TERM}, "service_proof_capability", RejectReason.GENERATION_CONTRACT_MISMATCH),
    (dict(toploc_passed=False), {TERM}, "toploc", RejectReason.TOPLOC_FAIL),
    (dict(stops=False), {TERM}, "termination", RejectReason.BAD_TERMINATION),
    (dict(stops=None), {TERM}, "service_proof_capability", RejectReason.GENERATION_CONTRACT_MISMATCH),
    # The turns end on TERM, a stop the proof does not check, and not at their cap: no verdict.
    (dict(stops=None), {99}, "service_proof_capability", RejectReason.GENERATION_CONTRACT_MISMATCH),
])
def test_a_signed_episode_proof_needs_toploc_and_its_stop_picks(tmp_path, monkeypatch, stub, eos, stage, reason):
    _eos(monkeypatch, eos)
    w = world(tmp_path, _stub(**stub))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == stage
    assert w.b.reject_counts[reason.value] >= 1


def test_a_proof_with_toploc_and_stops_goes_on_to_the_phase_one_gates(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    w = world(tmp_path, _stub())
    w.b._verify_expensive(w.pending)
    assert w.pending.proof_reject_stage not in {
        "service_proof_capability", "toploc", "termination", "episode_replay_binding", "service_contract"}


def test_without_a_toploc_contract_a_signed_episode_is_no_verdict_before_the_forward(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})                  # every turn's stop is checkable: only the missing TOPLOC refuses
    monkeypatch.setattr(batcher_module, "_active_profile", lambda: ACTIVE_PROTOCOL_PROFILE)
    assert batcher_module.toploc_proof(ACTIVE_PROTOCOL_PROFILE) is None
    seen = []
    w = world(tmp_path, _stub(seen=seen))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "service_proof_capability"
    assert seen == []


def test_the_proof_proves_exactly_the_spans_admission_checked(tmp_path):
    seen = []
    w = world(tmp_path, _stub(seen=seen))
    rollout = w.request.rollouts[0]
    start, end = rollout._validated_assistant_spans[-1]
    rollout._validated_assistant_spans = (*rollout._validated_assistant_spans[:-1], (start, end - 1))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "episode_replay_binding"
    assert seen == []                          # refused before the forward


def test_episode_uniforms_follow_the_model_token_offset_across_turns(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    uniforms = []
    w = world(tmp_path, _stub(uniforms=uniforms))
    w.b._verify_expensive(w.pending)
    assert uniforms
    for index, values in enumerate(uniforms):
        spans = w.request.rollouts[index]._validated_assistant_spans
        model_tokens = sum(end - start for start, end in spans)
        seed = w.group.selection.seeds[index]
        assert values == [w.group.pool.uniform(seed, j) for j in range(model_tokens)]
        assert len(spans) == 2                 # two turns: the second turn continues the offsets


def test_the_refusal_table():
    spans = [(2, 5), (7, 9)]
    tokens = [0, 0, 5, 5, TERM, 0, 0, 5, 6]
    ok = SimpleNamespace(toploc_checked=True, toploc_passed=True, episode_stop_picks_ok=True)
    assert signed_episode_proof_refusal(ok, tokens, spans, {TERM}) is None
    unknown = SimpleNamespace(toploc_checked=True, toploc_passed=True, episode_stop_picks_ok=None)
    assert signed_episode_proof_refusal(unknown, tokens, spans, set()) is None          # no stop expected
    assert signed_episode_proof_refusal(unknown, tokens, spans, {TERM}) == (
        RejectReason.GENERATION_CONTRACT_MISMATCH, "service_proof_capability")
    # With the contract's limits: the last turn (2 tokens, no stop) is not at its cap -> no verdict;
    # at its cap (per-turn cap 2) it is a normal capped turn; the episode cap counts from the absolute start.
    capped = dict(max_tokens_per_turn=2, max_episode_tokens=100)
    assert signed_episode_proof_refusal(ok, tokens, spans, {TERM}, max_tokens_per_turn=8,
                                        max_episode_tokens=100) == (
        RejectReason.GENERATION_CONTRACT_MISMATCH, "service_proof_capability")
    assert signed_episode_proof_refusal(ok, tokens, [(2, 5), (7, 9)], {TERM}, **capped) is None
    assert signed_episode_proof_refusal(ok, tokens, spans, {TERM}, max_tokens_per_turn=8,
                                        max_episode_tokens=9) is None
    failed = SimpleNamespace(toploc_checked=True, toploc_passed=False, episode_stop_picks_ok=False)
    assert signed_episode_proof_refusal(failed, tokens, spans, {TERM}) == (RejectReason.TOPLOC_FAIL, "toploc")


def test_the_pre_forward_guard_bounds_an_episode_by_its_contract(tmp_path):
    small = episode_contract(episode=episode_block(max_tokens_per_turn=16, max_episode_tokens=40))
    w = world(tmp_path, _stub(), contract=small)
    with pytest.raises(ValueError, match="bound"):
        w.b._service_pre_forward_guard(w.request, w.b.model)
    roomy = world(tmp_path / "roomy", _stub())
    roomy.b._service_pre_forward_guard(roomy.request, roomy.b.model)


def test_an_oversize_episode_is_refused_before_any_forward(tmp_path):
    seen = []
    small = episode_contract(episode=episode_block(max_tokens_per_turn=16, max_episode_tokens=40))
    w = world(tmp_path, _stub(seen=seen), contract=small)
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "service_length" and seen == []


def test_the_reward_policy_counts_model_tokens_only(tmp_path):
    w = world(tmp_path, _stub())
    model = sum(end - start for r in w.request.rollouts for start, end in r._validated_assistant_spans)
    assert GrpoWindowBatcher._service_token_count(w.request) == model
    assert model < sum(len(r.tokens) - r.commit["rollout"]["prompt_length"] for r in w.request.rollouts)
    for rollout in w.request.rollouts:
        rollout._validated_assistant_spans = None      # never validated: counted whole, never fewer
    assert GrpoWindowBatcher._service_token_count(w.request) == sum(
        len(r.tokens) - r.commit["rollout"]["prompt_length"] for r in w.request.rollouts)


def test_an_unpaid_exploration_episode_group_drops_its_transcripts(tmp_path):
    w = world(tmp_path, _stub())
    w.pending.service_lane = "exploration"
    w.b._release_observation_payload(w.pending)
    assert w.request.rollouts == []


def test_an_inconclusive_proof_hands_the_group_back_a_failed_one_does_not(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    calls = []
    inconclusive = world(tmp_path / "a", _stub(stops=None))
    inconclusive.b.episode_proof_inconclusive = lambda pending, stage: calls.append((pending, stage))
    for _ in range(2):
        inconclusive.b._execute_scheduled_proof(inconclusive.pending, model=inconclusive.b.model,
                                                count_operator_debt=False)
    assert calls == [(inconclusive.pending, "service_proof_capability")]     # once per group
    forged = world(tmp_path / "b", _stub(stops=False))
    forged.b.episode_proof_inconclusive = lambda pending, stage: calls.append((pending, stage))
    forged.b._execute_scheduled_proof(forged.pending, model=forged.b.model, count_operator_debt=False)
    assert forged.pending.proof_reject_stage == "termination"
    assert len(calls) == 1                                                    # sessions stay consumed


def test_an_inconclusive_exploration_audit_hands_the_group_back(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    w = world(tmp_path, _stub(toploc_checked=False))
    calls, lost = [], []
    w.b.episode_proof_inconclusive = lambda pending, stage: calls.append(stage)
    monkeypatch.setattr(w.b, "_mark_validator_lost", lambda identity: lost.append(identity))
    w.pending.service_observation_id = "obs-1"
    w.b._conclude_exploration_audit(w.pending, w.b._verify_expensive(w.pending, audit=True), ())
    assert calls == ["service_proof_capability"] and lost == ["obs-1"]


def test_a_legacy_batcher_never_calls_the_hook(tmp_path):
    b = _make_batcher()
    calls = []
    b.episode_proof_inconclusive = lambda pending, stage: calls.append(stage)
    b._episode_proof_inconclusive(SimpleNamespace(hotkey="h"), "service_proof_capability")
    assert calls == [] and GrpoWindowBatcher.episode_proof_inconclusive is None


class ChatTokenizer:
    """A tokenizer whose chat template would change the prompt (whatever the profile's prompt encoding)."""
    chat_template = "chat"

    def apply_chat_template(self, messages, **kwargs):
        return "<chat>" + messages[0]["content"]


def test_a_signed_episode_prompt_is_the_task_sources_verbatim(monkeypatch):
    monkeypatch.setattr("reliquary.constants.RAW_COMPLETION_PROMPTS", False)
    env = make_test_episode_env()
    assert batcher_module._render_environment_prompt(env, ChatTokenizer(), TASK) == PROMPT_TEXT


def test_every_stage_the_episode_check_rejects_at_is_classified():
    from reliquary.services.exploration import AUDIT_FAILURE_CLASS_BY_STAGE

    assert AUDIT_FAILURE_CLASS_BY_STAGE["toploc"] == AUDIT_FAILURE_CLASS_BY_STAGE["termination"] == "deterministic"
    assert {"service_proof_capability", "episode_replay_binding", "service_contract"} <= (
        GrpoWindowBatcher._AUDIT_INCONCLUSIVE_STAGES)
    assert batcher_module.signed_episode_proof_refusal is signed_episode_proof_refusal
    assert TEXT != TERM


# --- Task 10 carries (binding list): pre-forward stop set, telemetry, texts, debt, seed coverage. ---------

def test_a_turn_ending_on_a_stop_the_proof_does_not_check_is_no_verdict_before_the_forward(tmp_path, monkeypatch):
    _eos(monkeypatch, {99})                    # the turns end on TERM, outside the proof's stop set
    seen = []
    w = world(tmp_path, _stub(seen=seen))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "service_proof_capability"
    assert seen == []                          # no forward spent on a pick nobody can judge
    _eos(monkeypatch, set())                   # no stop set at all: the same, never a pass
    w = world(tmp_path / "none", _stub(seen=seen))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "service_proof_capability" and seen == []


def test_the_unchecked_turn_helper():
    tokens = [0, 0, 5, 5, TERM, 0, 0, 5, 6]
    limits = dict(max_tokens_per_turn=8, max_episode_tokens=100)
    assert batcher_module.signed_episode_unchecked_turn(tokens, [(2, 5), (7, 9)], {TERM}, **limits) == 1
    assert batcher_module.signed_episode_unchecked_turn(tokens, [(2, 5), (7, 9)], {TERM}, max_tokens_per_turn=2,
                                                        max_episode_tokens=100) is None
    assert batcher_module.signed_episode_unchecked_turn(tokens, [(2, 5)], set(), **limits) == 0


def test_a_failed_stop_pick_logs_its_turn_and_cdf_distance(tmp_path, monkeypatch, caplog):
    _eos(monkeypatch, {TERM})
    w = world(tmp_path, _stub(stops=False, cdf_miss=0.25))
    with caplog.at_level("WARNING"):
        assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "termination"
    assert any("first_bad_turn=0 cdf_miss=0.25" in record.getMessage() for record in caplog.records)


def test_a_signed_episode_is_never_decoded_into_a_completion_text(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    w = world(tmp_path, _stub())
    decoded = []
    w.b._completion_text = lambda rollout: decoded.append(rollout) or "decoded"
    w.b._verify_expensive(w.pending)
    assert decoded == []


def test_an_inconclusive_episode_proof_charges_no_operator_debt_a_forged_one_does(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    inconclusive = world(tmp_path / "a", _stub(stops=None, toploc_checked=False))
    inconclusive.b._execute_scheduled_proof(inconclusive.pending, model=inconclusive.b.model,
                                            count_operator_debt=True)
    assert inconclusive.pending.proof_reject_stage == "service_proof_capability"
    assert inconclusive.b._operator_proof_failure_debt_for_hotkey("5Hot") == 0
    assert inconclusive.b.expensive_proof_failures_by_hotkey.get("5Hot", 0) == 0
    forged = world(tmp_path / "b", _stub(stops=False))
    forged.b._execute_scheduled_proof(forged.pending, model=forged.b.model, count_operator_debt=True)
    assert forged.pending.proof_reject_stage == "termination"
    assert forged.b._operator_proof_failure_debt_for_hotkey("5Hot") == 1


def test_the_seed_check_must_cover_every_model_position(tmp_path, monkeypatch):
    _eos(monkeypatch, {TERM})
    w = world(tmp_path, _stub(seed_short=1))
    assert w.b._verify_expensive(w.pending) is None
    assert w.pending.proof_reject_stage == "service_seed_coverage"
    assert w.b.reject_counts[RejectReason.SEED_MISMATCH.value] >= 1


def test_the_restored_content_cooldown_digests_a_signed_episode_prompt_verbatim(monkeypatch):
    from reliquary.validator.prompt_content import prompt_content_sha256
    from reliquary.validator.service import ValidationService

    class Prompts:
        def export_state(self):
            return {TASK: 7}

    class Contents:
        def __init__(self):
            self.recorded = {}

        def export_state(self):
            return {}

        def record_selected(self, digest, window):
            self.recorded[digest] = window

    env = make_test_episode_env()
    contents = Contents()
    fake = SimpleNamespace(_cooldown_per_env={EPISODE: Prompts()}, envs={EPISODE: env},
                           _content_cooldown_per_env={EPISODE: contents}, tokenizer=ChatTokenizer())
    monkeypatch.setattr("reliquary.constants.RAW_COMPLETION_PROMPTS", False)
    assert ValidationService._top_up_content_cooldown_from_prompt_state(fake, 3) == 1
    assert contents.recorded == {prompt_content_sha256(EPISODE, PROMPT_TEXT): 7}
    assert prompt_content_sha256(EPISODE, PROMPT_TEXT) == prompt_content_sha256(
        EPISODE, batcher_module._render_environment_prompt(env, ChatTokenizer(), TASK))
