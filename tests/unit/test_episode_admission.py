"""Admission of a signed-episode group, no GPU."""
import copy
import dataclasses
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.validator.episode_admission import (  # noqa: E402
    EpisodeGroupChecker, EpisodeGroupFacts, EpisodeRefusal, finish_prepared,
)
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, FIRST_TURN_TEXT, FixedSource, batch_request, episode_contract, episode_group, episode_precommit, episode_signers,
    play_episode,
)
from tests.unit.sandbox_fixtures import NOW, directory  # noqa: E402
from tests.unit.test_trajectory_parse import TEXT, FakeRenderer  # noqa: E402

CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)
RECEIVED = NOW + 100


def world(tmp_path, **group_kwargs):
    validator, machine = episode_signers(tmp_path)
    group = episode_group(CONTRACT, validator=validator, machine=machine, **group_kwargs)
    verifier = attest.Ed25519TokenVerifier({validator.key_id: validator.public_key_b64})

    def check(*, request=None, precommit=group.precommit, seen=(), received=RECEIVED, policy=POLICY,
              source=None, renderer=None):
        checker = EpisodeGroupChecker(policy=policy, renderer=renderer or FakeRenderer(),
                                      source=source or FixedSource(),
                                      chunk_tokens=32)
        return checker.check(group.request if request is None else request, precommit=precommit,
                             directory=directory(machine), token_verifier=verifier, seen=frozenset(seen),
                             received=received)

    return SimpleNamespace(group=group, check=check, validator=validator, machine=machine)


def refused(outcome, reason, stage):
    assert isinstance(outcome, EpisodeRefusal), outcome
    assert (outcome.reason, outcome.stage) == (reason, stage), outcome
    return outcome


def test_an_honest_group_gives_the_final_records_rewards(tmp_path):
    w = world(tmp_path)
    facts = w.check()
    assert isinstance(facts, EpisodeGroupFacts), facts
    assert list(facts.rewards) == w.group.rewards
    assert facts.session_ids == tuple(f"s-{seed}" for seed in w.group.selection.seeds)
    assert facts.spans == tuple(tuple(tuple(s) for s in r.commit["rollout"]["episode"]["assistant_spans"])
                                for r in w.group.request.rollouts)
    assert facts.model_tokens == len(w.group.request.rollouts) * (FIRST_TURN_TEXT + 2 + 10)


def test_an_episode_cut_by_a_limit_is_a_normal_episode(tmp_path):
    # The last turn runs exactly to the EPISODE cap with no stop token: the harness stopped it.
    w = world(tmp_path, last=[TEXT] * 11, stop="context_length")
    tokens, _, _ = _last_turn(w)
    facts = w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=len(tokens)))
    assert isinstance(facts, EpisodeGroupFacts), facts
    assert list(facts.rewards) == w.group.rewards


def test_the_max_turns_stop_needs_exactly_max_turns(tmp_path):
    w = world(tmp_path, stop="max_turns")
    assert isinstance(w.check(policy=dataclasses.replace(POLICY, max_turns=2)), EpisodeGroupFacts)
    refused(w.check(policy=dataclasses.replace(POLICY, max_turns=3)), RejectReason.BAD_TOKENS, "episode_parse")


def test_a_seed_swapped_between_sessions_is_refused(tmp_path):
    w = world(tmp_path)
    rollouts = w.group.request.rollouts
    first, second = rollouts[0].commit["rollout"]["episode"], rollouts[1].commit["rollout"]["episode"]
    first["transcript"], second["transcript"] = second["transcript"], first["transcript"]
    refused(w.check(), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_a_transcript_of_another_hotkey_or_precommit_is_refused(tmp_path):
    w = world(tmp_path)
    for other in (episode_precommit(CONTRACT, hotkey="5Another"), episode_precommit(CONTRACT, hotkey="5Hot", window=2)):
        request = copy.deepcopy(w.group.request)
        _, _, foreign = play_episode(validator=w.validator, machine=w.machine, precommit=other,
                                     seed=w.group.selection.seeds[0], session_id="s-x", reward=1.0)
        request.rollouts[0].commit["rollout"]["episode"]["transcript"] = foreign
        refused(w.check(request=request), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_a_modified_tool_output_is_refused(tmp_path):
    w = world(tmp_path)
    records = w.group.request.rollouts[0].commit["rollout"]["episode"]["transcript"]["records"]
    records[1]["body"]["output"] = "forged"
    refused(w.check(), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_a_forged_observation_token_is_refused(tmp_path):
    w = world(tmp_path)
    commit = w.group.request.rollouts[0].commit
    first_end = commit["rollout"]["episode"]["assistant_spans"][0][1]
    commit["tokens"][first_end + 2] += 1                  # inside the tool-output segment
    refused(w.check(), RejectReason.BAD_TOKENS, "episode_parse")


def test_an_old_checkpoint_is_refused_without_a_proof_stage(tmp_path):
    w = world(tmp_path)
    request = copy.deepcopy(w.group.request)
    request.checkpoint_hash = "e" * 40
    refused(w.check(request=request), RejectReason.WRONG_CHECKPOINT, "episode_checkpoint")
    # A precommit recorded at another checkpoint than the group's.
    stale = dataclasses.replace(w.group.precommit, checkpoint="e" * 40)
    for rollout in w.group.request.rollouts:
        rollout.commit["rollout"]["episode"]["precommit_sha256"] = stale.sha256
    refused(w.check(precommit=stale), RejectReason.WRONG_CHECKPOINT, "episode_checkpoint")


def test_a_paid_session_cannot_be_paid_again(tmp_path):
    w = world(tmp_path)
    refused(w.check(seen={"s-0"}), RejectReason.HASH_DUPLICATE, "episode_session_reused")


def test_a_late_group_is_refused(tmp_path):
    w = world(tmp_path)
    refused(w.check(received=NOW + 4500 + attest.GRADING_GRACE_S + 1), RejectReason.PRECOMMIT_EXPIRED,
            "episode_deadline")


def test_record_zero_must_name_the_contracts_env_package_and_tools(tmp_path):
    w = world(tmp_path)
    refused(w.check(policy=dataclasses.replace(POLICY, env_package="reliquary-swe==9.9.9")),
            RejectReason.REWARD_MISMATCH, "episode_record0")
    refused(w.check(policy=dataclasses.replace(POLICY, tools=("bash",))), RejectReason.REWARD_MISMATCH,
            "episode_record0")


def test_the_prompt_is_the_validators_own_render(tmp_path):
    w = world(tmp_path)
    refused(w.check(source=FixedSource(text="Another task.")), RejectReason.PROMPT_MISMATCH, "episode_prompt")
    # Same length, other content: the tokens themselves are compared, not only the prompt's length.
    refused(w.check(source=FixedSource(text="Write 43 to /work/answer.txt.")), RejectReason.PROMPT_MISMATCH,
            "episode_prompt")


def test_the_contracts_episode_length_binds(tmp_path):
    w = world(tmp_path)
    refused(w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=40)), RejectReason.BAD_TOKENS,
            "episode_length")


def test_an_episode_too_short_for_the_logprob_challenge_is_refused(tmp_path, monkeypatch):
    # Below CHALLENGE_K model tokens the proof's log-prob check cannot run, and away from temperature 1 it
    # is required: the episode would fail at proof, so admission refuses it on its shape (the miner too).
    from reliquary.constants import CHALLENGE_K
    from reliquary.validator import episode_admission
    from tests.unit.test_trajectory_parse import TERM

    first = FIRST_TURN_TEXT + 1 + 1                     # the fixture's first turn: text, one call, TERM
    short = world(tmp_path / "short", last=[TEXT] * (CHALLENGE_K - first - 2) + [TERM])
    outcome = refused(short.check(), RejectReason.BAD_TOKENS, "episode_length")
    assert outcome.detail["check"] == "too_few_model_tokens", outcome
    monkeypatch.setattr(episode_admission, "T_PROTO", 1.0)
    assert isinstance(short.check(), EpisodeGroupFacts)
    monkeypatch.undo()
    exact = world(tmp_path / "exact", last=[TEXT] * (CHALLENGE_K - first - 1) + [TERM])
    assert isinstance(exact.check(), EpisodeGroupFacts)


def test_the_precommit_must_be_this_groups(tmp_path):
    w = world(tmp_path)
    refused(w.check(precommit=None), RejectReason.PRECOMMIT_INVALID, "episode_precommit")
    other_task = episode_precommit(CONTRACT, hotkey="5Hot", task=4)
    refused(w.check(precommit=other_task), RejectReason.PRECOMMIT_INVALID, "episode_precommit")
    request = copy.deepcopy(w.group.request)
    request.rollouts[0].commit["rollout"]["episode"]["seed_index"] += 1
    refused(w.check(request=request), RejectReason.PRECOMMIT_INVALID, "episode_precommit")


def test_an_ungraded_final_is_never_in_a_group(tmp_path):
    from tests.unit.sandbox_fixtures import transcript

    w = world(tmp_path)
    rollout = w.group.request.rollouts[0]
    claims = attest.SessionClaims.from_dict(rollout.commit["rollout"]["episode"]["transcript"]["token"]["claims"])
    rollout.commit["rollout"]["episode"]["transcript"] = transcript(w.validator, w.machine, claims,
                                                                     status="box_failed")
    refused(w.check(), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_one_proof_per_span_chunk(tmp_path):
    w = world(tmp_path)
    w.group.request.rollouts[0].commit["toploc_proofs"].append("AAAA")
    refused(w.check(), RejectReason.BAD_SCHEMA, "episode_proof_shape")


def test_finishing_sets_rewards_spans_and_the_lane(tmp_path):
    w = world(tmp_path)
    request = batch_request(w.group)
    facts = w.check(request=request)
    prepared = SimpleNamespace(request=request, rewards=[], completion_texts=[], episode_pending=True,
                               reject_reason=None, reject_stage=None, truncated_indices=(1,),
                               uncertain_indices=(1,), attainable_rewards=(0.5,))
    finish_prepared(prepared, facts, CONTRACT)
    assert prepared.rewards == w.group.rewards and prepared.episode_pending is False
    assert prepared.reject_reason is None
    assert (prepared.truncated_indices, prepared.uncertain_indices) == ((), ())
    for rollout, spans, reward in zip(request.rollouts, facts.spans, w.group.rewards):
        assert rollout._validated_assistant_spans == spans and rollout.reward == reward
        assert rollout.commit["rollout"]["total_reward"] == reward


def test_a_uniform_group_is_exploration_not_a_refusal(tmp_path):
    from reliquary.constants import M_ROLLOUTS

    w = world(tmp_path, rewards=[0.0] * M_ROLLOUTS)
    request = batch_request(w.group)
    prepared = SimpleNamespace(request=request, rewards=[], completion_texts=[], episode_pending=True,
                               reject_reason=None, reject_stage=None)
    finish_prepared(prepared, w.check(request=request), CONTRACT)
    assert prepared.reject_reason is None and prepared.rewards == [0.0] * M_ROLLOUTS


def test_a_group_mixing_two_precommits_is_refused(tmp_path):
    # An episode of another precommit of the same miner (another task) inside this group.
    w = world(tmp_path)
    other = episode_precommit(CONTRACT, hotkey="5Hot", task=4)
    request = copy.deepcopy(w.group.request)
    seed = w.group.selection.seeds[1]
    _, _, foreign = play_episode(validator=w.validator, machine=w.machine, precommit=other, seed=seed,
                                 session_id="s-other", reward=1.0)
    episode = request.rollouts[1].commit["rollout"]["episode"]
    episode["precommit_sha256"], episode["transcript"] = other.sha256, foreign
    refused(w.check(request=request), RejectReason.PRECOMMIT_INVALID, "episode_precommit")
    # Naming the group's precommit with the other precommit's transcript fails the engagement binding.
    request.rollouts[1].commit["rollout"]["episode"]["precommit_sha256"] = w.group.precommit.sha256
    refused(w.check(request=request), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_the_same_session_twice_in_a_group_is_refused(tmp_path):
    w = world(tmp_path)
    request = copy.deepcopy(w.group.request)
    request.rollouts[1].commit["rollout"]["episode"]["transcript"] = copy.deepcopy(
        request.rollouts[0].commit["rollout"]["episode"]["transcript"])
    refused(w.check(request=request), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_the_precommit_must_be_of_this_order_and_env(tmp_path):
    w = world(tmp_path)
    request = copy.deepcopy(w.group.request)
    request.service_binding = {**request.service_binding, "contract_sha256": "0" * 64}
    refused(w.check(request=request), RejectReason.PRECOMMIT_INVALID, "episode_precommit")
    request = copy.deepcopy(w.group.request)
    request.rollouts[2].env_name = "openmathinstruct"
    refused(w.check(request=request), RejectReason.PRECOMMIT_INVALID, "episode_precommit")


def test_declared_lengths_must_be_the_episodes(tmp_path):
    w = world(tmp_path)
    for field_name, delta in (("prompt_length", 1), ("completion_length", -1)):
        request = copy.deepcopy(w.group.request)
        request.rollouts[0].commit["rollout"][field_name] += delta
        refused(w.check(request=request), RejectReason.BAD_TOKENS, "episode_length")


def test_a_forged_transcript_never_reads_as_a_stale_checkpoint(tmp_path):
    # Another checkpoint AND a tampered record: the forgery decides the reason, not the milder one.
    from reliquary.protocol.service_episode import rl_engagement
    from tests.unit.episode_v2_fixtures import ENV_PACKAGE, SPLIT
    from tests.unit.sandbox_fixtures import claims, transcript

    w = world(tmp_path)
    precommit, seed = w.group.precommit, w.group.selection.seeds[0]
    session = claims(session_id="s-0", hotkey=precommit.hotkey, engagement=rl_engagement(1, precommit.sha256, seed),
                     split=SPLIT, index=precommit.task_index, checkpoint="e" * 40, issued_at=NOW,
                     expires_at=NOW + 4500)
    foreign = transcript(w.validator, w.machine, session, reward=1.0, env_package=ENV_PACKAGE,
                         calls=[{"turn": 0, "k": 0, "arguments": {"command": "c0"}, "output": "ok"}])
    request = copy.deepcopy(w.group.request)
    request.rollouts[0].commit["rollout"]["episode"]["transcript"] = copy.deepcopy(foreign)
    refused(w.check(request=request), RejectReason.WRONG_CHECKPOINT, "episode_checkpoint")
    foreign["records"][1]["body"]["output"] = "forged"
    request.rollouts[0].commit["rollout"]["episode"]["transcript"] = foreign
    refused(w.check(request=request), RejectReason.REWARD_MISMATCH, "episode_transcript")


def test_a_renderer_error_on_miner_tokens_is_a_refusal(tmp_path):
    class Raising(FakeRenderer):
        def tool_calls(self, completion_ids):
            raise ValueError("undecodable")

    w = world(tmp_path)
    validator, machine = w.validator, w.machine
    checker = EpisodeGroupChecker(policy=POLICY, renderer=Raising(), source=FixedSource(), chunk_tokens=32)
    outcome = checker.check(w.group.request, precommit=w.group.precommit, directory=directory(machine),
                            token_verifier=attest.Ed25519TokenVerifier({validator.key_id: validator.public_key_b64}),
                            seen=frozenset(), received=RECEIVED)
    refused(outcome, RejectReason.BAD_TOKENS, "episode_parse")


# --- group admission edge cases ------------------------------------------------------------

def _last_turn(w, index=0):
    """(tokens, the last span's absolute start, the last completion) of rollout ``index``."""
    commit = w.group.request.rollouts[index].commit
    start, end = commit["rollout"]["episode"]["assistant_spans"][-1]
    return commit["tokens"], start, commit["tokens"][start:end]


def test_an_early_cut_claimed_as_context_length_is_refused(tmp_path):
    # H1: a normally ended last turn whose calls ran, stopped by the miner, not by a limit.
    w = world(tmp_path, last_calls=1, stop="context_length")
    refused(w.check(), RejectReason.BAD_TERMINATION, "episode_termination")
    # Without calls too: a turn ending on its terminator is no context limit unless the next prompt overflows.
    w = world(tmp_path / "plain", stop="context_length")
    refused(w.check(), RejectReason.BAD_TERMINATION, "episode_termination")


def test_an_honest_cap_hit_after_calls_is_admitted(tmp_path):
    # H1 (a): the last turn ran exactly to the episode cap, with a call in it the gateway executed.
    from tests.unit.test_trajectory_parse import CALL

    w = world(tmp_path, last=[TEXT] * 10 + [CALL], last_calls=1, stop="context_length")
    tokens, _, _ = _last_turn(w)
    assert isinstance(w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=len(tokens))),
                      EpisodeGroupFacts)
    # One token short of the episode cap, with no terminator: a cut, not a limit.
    refused(w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=len(tokens) + 1)),
            RejectReason.BAD_TERMINATION, "episode_termination")


def test_a_per_turn_cap_hit_far_from_the_episode_cap_is_refused(tmp_path):
    # A per-turn cap is no episode limit: an honest harness goes on to the next turn.
    from tests.unit.test_trajectory_parse import CALL

    cap = FIRST_TURN_TEXT + 2                           # the first turn's length: only the last turn is capped
    w = world(tmp_path, last=[TEXT] * (cap - 1) + [CALL], last_calls=1, stop="context_length")
    refused(w.check(policy=dataclasses.replace(POLICY, max_tokens_per_turn=cap)), RejectReason.BAD_TERMINATION,
            "episode_termination")
    w = world(tmp_path / "plain", last=[TEXT] * cap, stop="context_length")
    refused(w.check(policy=dataclasses.replace(POLICY, max_tokens_per_turn=cap)), RejectReason.BAD_TERMINATION,
            "episode_termination")


def test_an_honest_observation_overflow_is_admitted(tmp_path):
    # H1 (b): the last turn ended on its terminator and its calls' observations overflow the episode.
    from reliquary_sandbox.observation import render_observation

    from reliquary.corpus.signed_parse import signed_records

    w = world(tmp_path, last_calls=1, stop="context_length")
    tokens, start, last = _last_turn(w)
    signed = w.group.request.rollouts[0].commit["rollout"]["episode"]["transcript"]
    observation = render_observation(signed_records(signed).calls[-1].to_dict())
    next_prompt = len(FakeRenderer().next_prompt(tokens[:start], last, [observation]))
    assert next_prompt > len(tokens)
    facts = w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=next_prompt - 1))
    assert isinstance(facts, EpisodeGroupFacts), facts
    # A next prompt that still fits: the harness would have gone on, so the stop is a lie.
    refused(w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=next_prompt)),
            RejectReason.BAD_TERMINATION, "episode_termination")


def test_a_last_turn_without_calls_is_never_a_context_stop(tmp_path):
    # A turn with no call ends the episode (agent_completed) however long a next prompt would be.
    w = world(tmp_path / "plain", stop="context_length")
    tokens, start, last = _last_turn(w)
    next_prompt = len(FakeRenderer().next_prompt(tokens[:start], last, []))
    refused(w.check(policy=dataclasses.replace(POLICY, max_episode_tokens=next_prompt - 1)),
            RejectReason.BAD_TERMINATION, "episode_termination")


def test_a_max_turns_stop_after_calls_needs_the_turn_limit(tmp_path):
    w = world(tmp_path, last_calls=1, stop="max_turns")
    assert isinstance(w.check(policy=dataclasses.replace(POLICY, max_turns=2)), EpisodeGroupFacts)
    outcome = w.check(policy=dataclasses.replace(POLICY, max_turns=3))
    assert isinstance(outcome, EpisodeRefusal) and outcome.reason in (RejectReason.BAD_TOKENS,
                                                                      RejectReason.BAD_TERMINATION), outcome


def test_the_selection_is_exactly_the_contracts_group_size(tmp_path):
    # M1: a group of fewer seeds than group_size, consistent with its own selection, is refused.
    from reliquary.constants import M_ROLLOUTS
    from reliquary.protocol.seed_pool import PoolSelection

    assert POLICY.pool_seeds == 2 * M_ROLLOUTS
    w = world(tmp_path)
    request = copy.deepcopy(w.group.request)
    request.rollouts = request.rollouts[:-1]
    request.pool_selection = PoolSelection(w.group.selection.pool_sha256, w.group.selection.seeds[:-1]).to_dict()
    refused(w.check(request=request), RejectReason.BAD_SCHEMA, "episode_selection")


class _SpyRenderer(FakeRenderer):
    def __init__(self):
        self.parses = 0

    def span_is_canonical(self, prompt_ids, completion_ids):
        self.parses += 1
        return super().span_is_canonical(prompt_ids, completion_ids)

    def tool_calls(self, completion_ids):
        self.parses += 1
        return super().tool_calls(completion_ids)

    def next_prompt(self, prompt_ids, completion_ids, observations):
        self.parses += 1
        return super().next_prompt(prompt_ids, completion_ids, observations)


def test_every_transcript_is_verified_before_any_parse(tmp_path):
    # M3: a forged transcript in the LAST rollout is refused before the renderer parses anything.
    w = world(tmp_path)
    last = len(w.group.request.rollouts) - 1
    records = w.group.request.rollouts[last].commit["rollout"]["episode"]["transcript"]["records"]
    records[1]["body"]["output"] = "forged"
    spy = _SpyRenderer()
    outcome = refused(w.check(renderer=spy), RejectReason.REWARD_MISMATCH, "episode_transcript")
    assert outcome.detail["rollout"] == last
    assert spy.parses == 0
    honest = world(tmp_path / "honest")
    spy = _SpyRenderer()
    assert isinstance(honest.check(renderer=spy), EpisodeGroupFacts) and spy.parses > 0


def test_a_seed_index_must_be_a_true_int(tmp_path):
    # L4: True == 1 and 1.0 == 1 in Python; neither names a seed.
    w = world(tmp_path)
    index = list(w.group.selection.seeds).index(1)
    for forged in (True, 1.0):
        request = copy.deepcopy(w.group.request)
        request.rollouts[index].commit["rollout"]["episode"]["seed_index"] = forged
        refused(w.check(request=request), RejectReason.PRECOMMIT_INVALID, "episode_precommit")


class _VersionedSource(FixedSource):
    version = 1


def test_the_prompt_cache_follows_the_source_version(tmp_path):
    # L1: a cached render of an older source version is never compared with the trajectory.
    w = world(tmp_path)
    source = _VersionedSource()
    checker = EpisodeGroupChecker(policy=POLICY, renderer=FakeRenderer(), source=source, chunk_tokens=32)

    def check():
        return checker.check(w.group.request, precommit=w.group.precommit, directory=directory(w.machine),
                             token_verifier=attest.Ed25519TokenVerifier({w.validator.key_id:
                                                                         w.validator.public_key_b64}),
                             seen=frozenset(), received=RECEIVED)

    assert isinstance(check(), EpisodeGroupFacts)
    source.text, source.version = "Another task.", 2
    refused(check(), RejectReason.PROMPT_MISMATCH, "episode_prompt")


@pytest.mark.parametrize("bad", ["float", "string", "bool", "before_prompt", "not_a_list", "empty"])
def test_assistant_spans_are_read_as_the_proof_reads_them(tmp_path, bad):
    # int() coercion let [2.0, 12], ["2", 12], [True, ..] through admission that the proof refuses.
    w = world(tmp_path)
    episode = w.group.request.rollouts[0].commit["rollout"]["episode"]
    first = list(episode["assistant_spans"][0])
    episode["assistant_spans"][0] = {
        "float": [float(first[0]), first[1]], "string": [str(first[0]), first[1]],
        "bool": [True, first[1]], "before_prompt": [first[0] - 1, first[1]],
    }.get(bad, first)
    if bad == "not_a_list":
        episode["assistant_spans"] = "2,12"
    if bad == "empty":
        episode["assistant_spans"] = []
    outcome = w.check()
    assert isinstance(outcome, EpisodeRefusal) and outcome.stage in ("episode_schema", "episode_prompt")
