"""The miner side of RL episodes: routes, runner hooks, the stop, the signed-episode commit."""
import asyncio
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

bt = pytest.importorskip("bittensor")
attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO  # noqa: E402
from reliquary.corpus.trajectory import GeneratedTurn  # noqa: E402
from reliquary.environment import forced_sampling as fs  # noqa: E402
from reliquary.miner.episode_commit import (  # noqa: E402
    AdmittedStop, build_signed_episode_commit, episode_metadata, episode_sequence, episode_stop,
    withdraw_inadmissible,
)
from reliquary.miner.rl_episode_client import (  # noqa: E402
    HttpRlEpisodes, RlSignedEpisodeRunner, check_engine_caps, rl_transcript_refusal, signed_precommit_body,
    signed_rl_open_request,
)
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused, sandbox_open_path  # noqa: E402
from reliquary.protocol.service_submission import ServiceBinding  # noqa: E402
from reliquary.protocol.signatures import (  # noqa: E402
    verify_commit_signature, verify_episode_precommit_signature, verify_sandbox_open_signature,
)
from reliquary.protocol.submission import CommitModel  # noqa: E402
from reliquary.sandbox.rl_routes import episode_precommit_path  # noqa: E402
from reliquary.validator.verifier import verify_commitment_proofs  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, episode_contract, episode_pool, episode_precommit, episode_signers, play_episode,
)
from tests.unit.sandbox_fixtures import NOW  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)


def sign(binding: bytes) -> str:
    return MINER.sign(binding).hex()


def test_the_precommit_and_the_rl_open_are_signed_for_this_validator_and_route():
    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    path = episode_precommit_path()
    body = signed_precommit_body(precommit=precommit, sign_binding=sign, now=NOW,
                                 validator_hotkey=VALIDATOR.ss58_address, path=path)
    assert body["miner_hotkey"] == MINER.ss58_address and body["at"] == NOW
    assert verify_episode_precommit_signature(MINER.ss58_address, body["precommit"], at=body["at"],
                                              signature=body["signature"], validator_hotkey=VALIDATOR.ss58_address,
                                              path=path)
    assert not verify_episode_precommit_signature(MINER.ss58_address, body["precommit"], at=body["at"],
                                                  signature=body["signature"],
                                                  validator_hotkey=VALIDATOR.ss58_address, path="/corpus/x")
    open_path = sandbox_open_path("/rl")
    request = signed_rl_open_request(hotkey=MINER.ss58_address, precommit_sha256=precommit.sha256, seed_index=4,
                                     sign_binding=sign, now=NOW, request_id="a" * 32,
                                     validator_hotkey=VALIDATOR.ss58_address, path=open_path)
    assert request["engagement"] == {"kind": "rl_precommit",
                                     "precommit": {"precommit_sha256": precommit.sha256, "seed_index": 4}}
    assert verify_sandbox_open_signature(request, validator_hotkey=VALIDATOR.ss58_address, path=open_path)
    assert not verify_sandbox_open_signature(request, validator_hotkey=VALIDATOR.ss58_address,
                                             path=sandbox_open_path("/corpus"))


class FakeResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code, self._payload, self.headers = status_code, payload, headers or {}

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, response):
        self.response, self.posts = response, []

    def post(self, path, content, headers):
        self.posts.append((path, content))
        return self.response


def test_the_precommit_is_posted_to_the_rl_route_and_its_refusals_carry_their_wait():
    http = FakeHttp(FakeResponse(200, {"precommit_sha256": "e" * 64, "created": True}))
    sessions = HttpRlEpisodes(http, validator_hotkey=VALIDATOR.ss58_address)
    assert sessions.prefix == "/rl" and sessions.precommit_path() == episode_precommit_path("/rl")
    assert sessions.precommit({"x": 1}) == {"precommit_sha256": "e" * 64, "created": True}
    assert http.posts == [("/rl/episodes/precommit", b'{"x":1}')]
    http.response = FakeResponse(429, {"reason": "precommit_rate", "detail": {}}, {"Retry-After": "7"})
    with pytest.raises(SessionRefused) as refused:
        sessions.precommit({"x": 1})
    assert (refused.value.reason, refused.value.status, refused.value.retry_after) == ("precommit_rate", 429, 7.0)


def test_the_miner_runs_the_admissions_transcript_checks_it_can(tmp_path):
    validator, machine = episode_signers(tmp_path)
    precommit = episode_precommit(CONTRACT, hotkey="5Hot")
    _, _, transcript = play_episode(validator=validator, machine=machine, precommit=precommit, seed=2,
                                    session_id="s-2", reward=1.0)
    assert rl_transcript_refusal(transcript, policy=POLICY, precommit=precommit, seed_index=2, now=NOW + 10) is None
    assert rl_transcript_refusal(transcript, policy=POLICY, precommit=precommit, seed_index=3,
                                 now=NOW + 10)[0] == "episode_transcript"
    assert rl_transcript_refusal(transcript, policy=POLICY, precommit=precommit, seed_index=2,
                                 now=NOW + 10**6)[0] == "episode_deadline"
    other = dataclasses.replace(POLICY, env_package="reliquary-swe==9.9.9")
    assert rl_transcript_refusal(transcript, policy=other, precommit=precommit, seed_index=2,
                                 now=NOW + 10)[0] == "episode_record0"
    other = dataclasses.replace(POLICY, tools=("bash",))
    assert rl_transcript_refusal(transcript, policy=other, precommit=precommit, seed_index=2,
                                 now=NOW + 10)[0] == "episode_record0"
    assert rl_transcript_refusal(None, policy=POLICY, precommit=precommit, seed_index=2,
                                 now=NOW + 10)[0] == "malformed_submission"


def _runner(sessions, **kwargs):
    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    return precommit, RlSignedEpisodeRunner(
        policy=POLICY, precommit=precommit, prompt="Write 42.", hotkey=MINER.ss58_address, sign_binding=sign,
        sessions=sessions, model_name="m", renderer_model_dir="/m", generate_url="http://127.0.0.1:1",
        sampling=SimpleNamespace(temperature=1.0, top_p=1.0), **kwargs)


def test_the_rl_runner_opens_its_seed_and_checks_with_the_rl_binding():
    sessions = HttpRlEpisodes(SimpleNamespace(), validator_hotkey=VALIDATOR.ss58_address)
    precommit, runner = _runner(sessions)
    body = runner._open_body(6, "b" * 32)
    assert body["engagement"]["precommit"] == {"precommit_sha256": precommit.sha256, "seed_index": 6}
    assert verify_sandbox_open_signature(body, validator_hotkey=VALIDATOR.ss58_address, path="/rl/sandbox/sessions")
    assert runner.requires_text_state is False
    assert runner._source.prompt(13) == "Write 42."
    assert runner._live._value == POLICY.pool_seeds
    assert runner.deadline(0) > POLICY.budgets_dict()["wall_s"]
    with pytest.raises(ValueError):
        RlSignedEpisodeRunner(
            policy=POLICY, precommit=episode_precommit(CONTRACT, hotkey=VALIDATOR.ss58_address), prompt="p",
            hotkey=MINER.ss58_address, sign_binding=sign, sessions=sessions, model_name="m",
            renderer_model_dir="/m", generate_url="http://127.0.0.1:1",
            sampling=SimpleNamespace(temperature=1.0, top_p=1.0))


class RlSessions:
    """The RL validator's session routes, faked (a final close holds the slot: closed_graded)."""

    prefix = "/rl"
    validator_hotkey = VALIDATOR.ss58_address

    def __init__(self, grant):
        self.grant, self.opened, self.closed = grant, [], []

    def open(self, body):
        self.opened.append(body)
        return self.grant

    def close(self, session_id, body):
        self.closed.append((session_id, body))
        return {"state": "closed_graded" if body["reason"] == "final" else "closed"}


class Harness:
    def __init__(self, transcript, state, status="graded"):
        self.transcript, self.state, self.runs, self.status = transcript, state, [], status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, *, token, prompt, on_trace=None):
        self.runs.append(prompt)
        trace = SimpleNamespace(id="t-1", stop_condition="agent_completed")
        return SimpleNamespace(trace=trace, state=self.state, transcript=self.transcript, refusals=(), error=None,
                               final={"body": {"status": self.status, "reward": 1.0, "reason": None}})


def _play(tmp_path, *, seed, state=b"\xff\xfe not text", run_seed=None, status="graded"):
    validator, machine = episode_signers(tmp_path)
    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    _, _, transcript = play_episode(validator=validator, machine=machine, precommit=precommit, seed=seed,
                                    session_id="s-1", reward=1.0)
    sessions = RlSessions({"session_id": "s-1", "token": transcript["token"], "gateway_url": "http://g:1",
                           "expires_at": NOW + 4500})
    harness = Harness(transcript, state, status)
    _, runner = _runner(sessions, runner_factory=lambda url: harness, clock=lambda: NOW + 10,
                        new_request_id=lambda: "c" * 32)

    async def go():
        async with runner:
            return await runner.run(seed if run_seed is None else run_seed)

    return asyncio.run(go()), sessions, harness, transcript


def test_an_rl_episode_is_kept_whatever_its_graded_state_and_closed_final(tmp_path):
    result, sessions, harness, transcript = _play(tmp_path, seed=2)
    assert result.ok and result.error is None, result.error
    assert result.final_diff == "" and result.transcript == transcript and result.stop == "agent_completed"
    assert harness.runs == ["Write 42."]
    (opened,) = sessions.opened
    assert opened["engagement"]["kind"] == "rl_precommit" and opened["engagement"]["precommit"]["seed_index"] == 2
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-1", "final")]
    asyncio.run(result.release())
    assert [body["reason"] for _, body in sessions.closed] == ["final", "withdraw"]
    assert sessions.closed[1][1]["transcript"] == transcript


def test_an_aborted_episode_carries_its_final_status_and_is_closed_final(tmp_path):
    result, sessions, _, transcript = _play(tmp_path, seed=2, status="aborted")
    assert not result.ok and result.final_status == "aborted"
    assert [(sid, body["reason"]) for sid, body in sessions.closed] == [("s-1", "final")]
    graded, _, _, _ = _play(tmp_path / "graded", seed=2)
    assert graded.final_status is None


def test_an_rl_transcript_of_another_seed_is_withdrawn_at_once(tmp_path):
    result, sessions, _, transcript = _play(tmp_path, seed=2, run_seed=3)
    assert not result.ok and "episode_transcript" in result.error
    assert [(body["reason"], body["transcript"]) for _, body in sessions.closed] == [("withdraw", transcript)]


def test_the_sequence_of_a_session_log_is_its_prompt_turns_and_observations():
    turns = [GeneratedTurn((1, 2, 3), (10, 11), ()), GeneratedTurn((1, 2, 3, 10, 11, 7, 7), (12, 13, 14), ())]
    tokens, spans = episode_sequence(turns)
    assert tokens == [1, 2, 3, 10, 11, 7, 7, 12, 13, 14]
    assert spans == [(3, 5), (7, 10)]
    with pytest.raises(ValueError):
        episode_sequence([GeneratedTurn((1, 2, 3), (10,), ()), GeneratedTurn((9, 9), (12,), ())])
    with pytest.raises(ValueError):          # no observation between two turns
        episode_sequence([GeneratedTurn((1, 2, 3), (10,), ()), GeneratedTurn((1, 2, 3, 10), (12,), ())])
    with pytest.raises(ValueError):
        episode_sequence([GeneratedTurn((1, 2, 3), (), ())])
    with pytest.raises(ValueError):
        episode_sequence([])


STOPS = frozenset({2, 3})


def _stop(raw, tokens, spans, *, turns=4, per_turn=8, total=100):
    return episode_stop(raw, tokens=tokens, spans=spans, max_turns=turns, max_tokens_per_turn=per_turn,
                        max_episode_tokens=total, stop_ids=STOPS)


def test_the_stop_is_the_one_the_validator_admits():
    closed = [9] * 5 + [1, 1, 3] + [9, 9] + [1, 2]          # prompt 5, turns [5, 8) and [10, 12)
    spans = [(5, 8), (10, 12)]
    assert _stop("agent_completed", closed, spans) == "agent_completed"
    assert _stop("context_length", closed, spans) == "context_length"
    assert _stop("max_turns", closed, spans, turns=2) == "max_turns"
    assert _stop("max_turns", closed, spans, turns=3) is None
    assert _stop("error", closed, spans) is None and _stop(None, closed, spans) is None
    # verifiers labels a cut at the EPISODE cap agent_completed: the validator takes it as context_length.
    cut = [9] * 5 + [1, 1, 3] + [9, 9] + [1, 1]
    assert _stop("agent_completed", cut, spans, total=12) == "context_length"
    assert _stop("context_length", cut, spans, total=12) == "context_length"
    assert _stop("max_turns", cut, spans, total=12, turns=2) == "max_turns"
    # A cut at the per-turn cap: only the max_turns-th turn; a cut short of any cap is never admitted.
    assert _stop("agent_completed", cut, spans, per_turn=2, turns=2) == "max_turns"
    assert _stop("agent_completed", cut, spans, per_turn=2) is None
    assert _stop("agent_completed", cut, spans) is None
    assert _stop("agent_completed", cut, spans, turns=2) is None           # short of every cap
    assert _stop("max_turns", cut, spans, turns=2) is None
    assert _stop("agent_completed", cut, spans, total=11) is None          # longer than the episode cap
    assert _stop("agent_completed", cut + [1], spans, total=13) is None    # trailing tokens


def test_the_mapped_stop_passes_the_validators_admission(tmp_path):
    """The validator's own checker on the stop ``episode_stop`` gives for honest limit hits."""
    from reliquary.protocol.submission import RejectReason
    from reliquary.validator.episode_admission import EpisodeGroupFacts, EpisodeRefusal
    from tests.unit.episode_v2_fixtures import FIRST_TURN_TEXT
    from tests.unit.test_episode_admission import world
    from tests.unit.test_trajectory_parse import TEXT, FakeRenderer

    def admitted(w, policy, raw):
        commit = w.group.request.rollouts[0].commit
        stop = episode_stop(raw, tokens=commit["tokens"], spans=commit["rollout"]["episode"]["assistant_spans"],
                            max_turns=policy.max_turns, max_tokens_per_turn=policy.max_tokens_per_turn,
                            max_episode_tokens=policy.max_episode_tokens, stop_ids=FakeRenderer().stop_ids)
        for rollout in w.group.request.rollouts:
            rollout.commit["rollout"]["episode"]["stop"] = stop
        return stop, w.check(policy=policy)

    # Cut at the episode cap, labelled agent_completed by verifiers.
    w = world(tmp_path / "a", last=[TEXT] * 11, stop="agent_completed")
    capped = dataclasses.replace(POLICY, max_episode_tokens=len(w.group.request.rollouts[0].commit["tokens"]))
    raw_outcome = w.check(policy=capped)
    assert isinstance(raw_outcome, EpisodeRefusal), raw_outcome
    stop, outcome = admitted(w, capped, "agent_completed")
    assert stop == "context_length" and isinstance(outcome, EpisodeGroupFacts), outcome
    # Cut at the per-turn cap on the last allowed turn (the cap is the first turn's length).
    cap = FIRST_TURN_TEXT + 2
    w = world(tmp_path / "b", last=[TEXT] * cap, stop="agent_completed")
    limited = dataclasses.replace(POLICY, max_tokens_per_turn=cap, max_turns=2)
    stop, outcome = admitted(w, limited, "agent_completed")
    assert stop == "max_turns" and isinstance(outcome, EpisodeGroupFacts), outcome
    # The same cut before the last allowed turn: nothing the validator admits, so nothing is submitted.
    w = world(tmp_path / "c", last=[TEXT] * cap, stop="agent_completed")
    early = dataclasses.replace(POLICY, max_tokens_per_turn=cap, max_turns=3)
    assert admitted(w, early, "agent_completed")[0] is None
    for label in ("agent_completed", "context_length", "max_turns"):
        for rollout in w.group.request.rollouts:
            rollout.commit["rollout"]["episode"]["stop"] = label
        outcome = w.check(policy=early)
        assert isinstance(outcome, EpisodeRefusal) and outcome.reason in (
            RejectReason.BAD_TOKENS, RejectReason.BAD_TERMINATION), (label, outcome)
    # An honest closed ending keeps its label.
    w = world(tmp_path / "d")
    stop, outcome = admitted(w, POLICY, "agent_completed")
    assert stop == "agent_completed" and isinstance(outcome, EpisodeGroupFacts), outcome


def _tiny():
    config = AutoConfig.for_model(
        "qwen3", vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, eos_token_id=2,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).to(torch.float32).eval()


def _turn(model, context, pool, seed, base_offset, length):
    tokens, out = list(context), []
    for j in range(length):
        with torch.no_grad():
            logits = model(torch.tensor([tokens])).logits[0, -1]
        probs = fs.warp(logits.float(), t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        token = fs.pick(probs, pool.uniform(seed, base_offset + j))
        tokens.append(token)
        out.append(token)
    return out


def _honest_commit():
    from reliquary.protocol.grail_verifier import GRAILVerifier

    model = _tiny()
    pool = episode_pool(CONTRACT)
    selection = pool.selection(list(range(pool.group_size)))
    index, seed = 2, selection.seeds[2]
    prompt = list(range(20, 30))
    first = _turn(model, prompt, pool, seed, 0, 10)
    observation = [7, 7, 7]
    second = _turn(model, prompt + first + observation, pool, seed, len(first), 12)
    tokens, spans = episode_sequence([GeneratedTurn(tuple(prompt), tuple(first), ()),
                                      GeneratedTurn(tuple(prompt + first + observation), tuple(second), ())])
    stop = episode_stop("max_turns", tokens=tokens, spans=spans, max_turns=2, max_tokens_per_turn=12,
                        max_episode_tokens=100, stop_ids=frozenset())
    episode = episode_metadata(precommit_sha256="e" * 64, seed_index=seed, spans=spans, stop=stop,
                               transcript={"token": {}, "records": []})
    randomness = "cd" * 32
    kwargs = dict(model=model, verifier=GRAILVerifier(hidden_dim=model.config.hidden_size), tokens=tokens,
                  spans=spans, episode=episode,
                  service_binding=ServiceBinding(CONTRACT.sha256, "training").rollout_binding(index),
                  seed_pool=selection.rollout_binding(index), randomness=randomness,
                  wallet=SimpleNamespace(hotkey=MINER), toploc=PROOF)
    commit = build_signed_episode_commit(**kwargs)
    positions = [t for start, end in spans for t in range(start, end)]
    return SimpleNamespace(model=model, commit=commit, randomness=randomness, spans=spans, kwargs=kwargs,
                           uniforms=[pool.uniform(seed, j) for j in range(len(positions))])


def test_an_honest_miners_commit_passes_the_validators_proof_on_cpu():
    honest = _honest_commit()
    commit = honest.commit
    CommitModel.model_validate(commit)
    assert verify_commit_signature(commit, MINER.ss58_address)
    assert len(commit["rollout"]["token_logprobs"]) == sum(end - start for start, end in honest.spans)
    result = verify_commitment_proofs({**commit, "toploc_spec": PROOF.to_contract()}, honest.model,
                                      honest.randomness, seed_u_values=honest.uniforms)
    assert result.toploc_checked and result.toploc_passed, result.toploc_reason
    assert result.seed_n_stochastic > 0 and result.seed_n_match == result.seed_n_stochastic
    assert result.all_passed
    # The episode is bound: another stop (or transcript) breaks the signature.
    forged = {**commit, "rollout": {**commit["rollout"], "episode": {**commit["rollout"]["episode"],
                                                                     "stop": "agent_completed"}}}
    assert not verify_commit_signature(forged, MINER.ss58_address)
    # Logprobs of each model token under the row that drew it.
    with torch.no_grad():
        logits = honest.model(torch.tensor([commit["tokens"]])).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = [log_probs[t - 1, commit["tokens"][t]].item() for start, end in honest.spans for t in range(start, end)]
    assert commit["rollout"]["token_logprobs"] == pytest.approx(expected, abs=1e-4)
    with pytest.raises(ValueError):
        build_signed_episode_commit(**{**honest.kwargs, "spans": honest.spans[:1]})
    other = {**honest.kwargs["episode"], "assistant_spans": [[s, e - 1] for s, e in honest.spans[:-1]]
             + [list(honest.spans[-1])]}
    with pytest.raises(ValueError):
        build_signed_episode_commit(**{**honest.kwargs, "episode": other})


# -- metadata and stop admission ----------------------------------------------------------------------


def test_the_metadata_takes_only_a_stop_episode_stop_admitted():
    closed = [9] * 5 + [1, 1, 3] + [9, 9] + [1, 2]
    spans = [(5, 8), (10, 12)]
    stop = _stop("agent_completed", closed, spans)
    assert isinstance(stop, AdmittedStop)
    episode = episode_metadata(precommit_sha256="e" * 64, seed_index=1, spans=spans, stop=stop, transcript={})
    assert episode["stop"] == "agent_completed" and type(episode["stop"]) is str
    for raw in ("agent_completed", "context_length", "max_turns", "error", None, AdmittedStop("episode_closed")):
        with pytest.raises(ValueError):
            episode_metadata(precommit_sha256="e" * 64, seed_index=1, spans=spans, stop=raw, transcript={})


def test_the_engine_caps_must_be_the_policys():
    caps = dict(max_total_tokens=POLICY.max_episode_tokens, max_tokens_per_turn=POLICY.max_tokens_per_turn,
                max_model_len=POLICY.max_episode_tokens)
    check_engine_caps(POLICY, **caps)
    for name in caps:
        for wrong in (caps[name] - 1, caps[name] + 1):
            with pytest.raises(ValueError, match=name):
                check_engine_caps(POLICY, **{**caps, name: wrong})
    with pytest.raises(ValueError):
        check_engine_caps(dataclasses.replace(POLICY, max_tokens_per_turn=1), **{**caps, "max_tokens_per_turn": True})


# Episodes of any turn shape, in the fake renderer's ids, and the validator's check of a group of them.

def _play_turns(validator, machine, precommit, seed, *, turns, reward):
    """``turns``: [(text tokens, calls, last token)]; each call is signed by the gateway."""
    from reliquary_sandbox.observation import render_observation

    from reliquary.corpus.signed_parse import signed_records
    from reliquary.protocol.service_episode import rl_engagement
    from tests.unit.episode_v2_fixtures import ENV_PACKAGE, PROMPT_TEXT, SPLIT
    from tests.unit.sandbox_fixtures import claims, transcript
    from tests.unit.test_trajectory_parse import CALL, TEXT, FakeRenderer

    renderer = FakeRenderer()
    session = claims(session_id=f"s-{seed}", hotkey=precommit.hotkey,
                     engagement=rl_engagement(precommit.window, precommit.sha256, seed), split=SPLIT,
                     index=precommit.task_index, checkpoint=precommit.checkpoint, issued_at=NOW,
                     expires_at=NOW + 4500)
    signed = transcript(validator, machine, session, status="graded", reward=float(reward),
                        env_package=ENV_PACKAGE,
                        calls=[{"turn": t, "k": k, "arguments": {"command": f"c{k}"}, "output": "ok"}
                               for t, (_, calls, _) in enumerate(turns) for k in range(calls)])
    bodies = iter(signed_records(signed).calls)
    tokens, spans = renderer.initial_ids(PROMPT_TEXT), []
    for t, (text, calls, last) in enumerate(turns):
        completion = [TEXT] * text + [CALL] * calls + [last]
        start = len(tokens)
        spans.append((start, start + len(completion)))
        if t + 1 < len(turns):
            observations = [render_observation(next(bodies).to_dict()) for _ in range(calls)]
            tokens = renderer.next_prompt(tokens, completion, observations)
        else:
            tokens = tokens + completion
    return tokens, spans, signed


def _group(tmp_path, turns, *, stop="agent_completed"):
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.episode_v2_fixtures import (
        episode_pool, half_rewards, signed_episode_commit, signed_episode_metadata,
    )

    validator, machine = episode_signers(tmp_path)
    pool = episode_pool(CONTRACT)
    precommit = episode_precommit(CONTRACT, hotkey="5Hot")
    selection = pool.selection(list(range(M_ROLLOUTS)))
    rewards, rollouts = half_rewards(), []
    for index, seed in enumerate(selection.seeds):
        tokens, spans, signed = _play_turns(validator, machine, precommit, seed, turns=turns, reward=rewards[index])
        episode = signed_episode_metadata(precommit_sha256=precommit.sha256, seed_index=seed, spans=spans,
                                          transcript=signed, stop=stop)
        commit = signed_episode_commit(tokens=tokens, spans=spans, episode=episode, selection=selection,
                                       index=index, contract=CONTRACT)
        rollouts.append(SimpleNamespace(tokens=tokens, reward=0.0, commit=commit, env_name=EPISODE))
    request = SimpleNamespace(
        miner_hotkey="5Hot", prompt_idx=precommit.task_index, window_start=precommit.window,
        checkpoint_hash=precommit.checkpoint, pool_selection=selection.to_dict(), rollouts=rollouts,
        service_binding=ServiceBinding(CONTRACT.sha256, "training").to_dict())
    return SimpleNamespace(request=request, precommit=precommit, validator=validator, machine=machine)


def _validator_check(group, *, policy=POLICY, prompt=None):
    from reliquary.validator.episode_admission import EpisodeGroupChecker
    from tests.unit.episode_v2_fixtures import FixedSource
    from tests.unit.sandbox_fixtures import directory
    from tests.unit.test_trajectory_parse import FakeRenderer

    source = FixedSource() if prompt is None else FixedSource(text=prompt)
    checker = EpisodeGroupChecker(policy=policy, renderer=FakeRenderer(), source=source, chunk_tokens=32)
    verifier = attest.Ed25519TokenVerifier({group.validator.key_id: group.validator.public_key_b64})
    return checker.check(group.request, precommit=group.precommit, directory=directory(group.machine),
                         token_verifier=verifier, seen=frozenset(), received=NOW + 100)


def _miner_screen(group, *, raw, policy=POLICY, prompt=None):
    """The miner's screen of the group's episodes, as ``mine_task`` holds them: (kept seeds, released seeds)."""
    from tests.unit.episode_v2_fixtures import PROMPT_TEXT
    from tests.unit.test_trajectory_parse import FakeRenderer

    released, outcomes = [], []
    for rollout in group.request.rollouts:
        episode = rollout.commit["rollout"]["episode"]
        tokens, spans = rollout.commit["tokens"], episode["assistant_spans"]
        session = SimpleNamespace(turns=[GeneratedTurn(tuple(tokens[:start]), tuple(tokens[start:end]), ())
                                         for start, end in spans])

        async def release(seed=episode["seed_index"]):
            released.append(seed)

        result = SimpleNamespace(ok=True, transcript=episode["transcript"], stop=raw, release=release)
        outcomes.append(SimpleNamespace(seed_index=episode["seed_index"], result=result, session=session))
    kept = asyncio.run(withdraw_inadmissible(outcomes, policy=policy, renderer=FakeRenderer(),
                                             prompt=PROMPT_TEXT if prompt is None else prompt))
    return kept, released


HONEST = [(20, 1, 1), (9, 0, 1)]         # (text, calls, last token): TERM = 1, EOT = 2; CHALLENGE_K model tokens


def test_an_honest_episode_passes_the_miners_screen_and_the_validator(tmp_path):
    from reliquary.validator.episode_admission import EpisodeGroupFacts

    group = _group(tmp_path, HONEST)
    assert isinstance(_validator_check(group), EpisodeGroupFacts)
    kept, released = _miner_screen(group, raw="agent_completed")
    assert released == [] and len(kept) == len(group.request.rollouts)
    outcome, admissible = kept[0]
    assert admissible.stop == "agent_completed" and isinstance(admissible.stop, AdmittedStop)
    assert admissible.tokens == group.request.rollouts[0].commit["tokens"]


def _refused_by_both(group, *, raw, policy=POLICY, miner_prompt=None, validator_prompt=None):
    from reliquary.validator.episode_admission import EpisodeRefusal

    outcome = _validator_check(group, policy=policy, prompt=validator_prompt)
    assert isinstance(outcome, EpisodeRefusal), outcome
    kept, released = _miner_screen(group, raw=raw, policy=policy, prompt=miner_prompt)
    assert kept == [] and sorted(released) == sorted(r.commit["rollout"]["episode"]["seed_index"]
                                                     for r in group.request.rollouts)
    return outcome


def test_an_episode_the_validator_refuses_is_withdrawn_before_the_choice(tmp_path):
    from reliquary.protocol.submission import RejectReason

    # The prompt is not the validator's render of the task.
    group = _group(tmp_path / "prompt", HONEST)
    assert _refused_by_both(group, raw="agent_completed", miner_prompt="Another task.",
                            validator_prompt="Another task.").stage == "episode_prompt"
    # Same length, other content: the prompt tokens themselves are compared.
    same_length = "Write 43 to /work/answer.txt."
    assert _refused_by_both(group, raw="agent_completed", miner_prompt=same_length,
                            validator_prompt=same_length).stage == "episode_prompt"
    # An observation token that is not the re-render of its signed call record.
    group = _group(tmp_path / "observation", HONEST)
    for rollout in group.request.rollouts:
        first_end = rollout.commit["rollout"]["episode"]["assistant_spans"][0][1]
        rollout.commit["tokens"][first_end + 2] += 1
    assert _refused_by_both(group, raw="agent_completed").stage == "episode_parse"
    # Three short turns.
    group = _group(tmp_path / "short", [(2, 1, 1), (2, 1, 1), (2, 1, 1), (9, 0, 1)])
    outcome = _refused_by_both(group, raw="agent_completed")
    assert (outcome.reason, outcome.stage, outcome.detail["check"]) == (
        RejectReason.BAD_TOKENS, "episode_turns", "short_turns"), outcome
    # An earlier turn ended on eos, not on the turn terminator.
    group = _group(tmp_path / "eos", [(9, 1, 2), (9, 0, 1)])
    outcome = _refused_by_both(group, raw="agent_completed")
    assert (outcome.stage, outcome.detail["check"]) == ("episode_termination", "bad_termination"), outcome
    # A closed last turn labelled context_length whose next prompt still fits under the cap.
    group = _group(tmp_path / "context", [(9, 1, 1), (9, 1, 1)], stop="context_length")
    outcome = _refused_by_both(group, raw="context_length")
    assert (outcome.stage, outcome.detail["check"]) == ("episode_termination", "bad_stop")
    # An honest episode with fewer model tokens than the proof's log-prob challenge needs.
    group = _group(tmp_path / "few", [(9, 1, 1), (9, 0, 1)])
    outcome = _refused_by_both(group, raw="agent_completed")
    assert (outcome.reason, outcome.stage, outcome.detail["check"]) == (
        RejectReason.BAD_TOKENS, "episode_length", "too_few_model_tokens"), outcome


def _next_prompt_length(group):
    from reliquary_sandbox.observation import render_observation

    from reliquary.corpus.signed_parse import signed_records
    from tests.unit.test_trajectory_parse import FakeRenderer

    commit = group.request.rollouts[0].commit
    tokens = commit["tokens"]
    start, end = commit["rollout"]["episode"]["assistant_spans"][-1]
    observation = render_observation(signed_records(commit["rollout"]["episode"]["transcript"]).calls[-1].to_dict())
    return len(FakeRenderer().next_prompt(tokens[:start], tokens[start:end], [observation]))


def test_a_next_prompt_exactly_at_the_cap_withdraws_the_episode(tmp_path):
    """The endpoint refuses a prompt that leaves no room with a 400 whose text verifiers does not read as a
    context-length error (its pre-flight only rejects a prompt LONGER than max_model_len), so the harness
    ends in an error: whatever the label, the miner never submits the episode, and the validator would
    refuse it as context_length (the next prompt fits)."""
    import httpx
    import openai
    from verifiers.legacy.clients.openai_chat_completions_client import handle_openai_overlong_prompt
    from verifiers.legacy.errors import OverlongPromptError

    from reliquary.miner.corpus_generate_server import GenerateEngine
    from reliquary.validator.episode_admission import EpisodeGroupFacts

    group = _group(tmp_path, [(20, 1, 1), (9, 1, 1)], stop="context_length")
    at_cap = _next_prompt_length(group)
    engine = GenerateEngine(None, max_total_tokens=at_cap, max_tokens_per_turn=POLICY.max_tokens_per_turn)
    with pytest.raises(ValueError) as error:
        asyncio.run(engine.generate("s-0", [7] * at_cap, None))

    @handle_openai_overlong_prompt
    async def endpoint():
        request = httpx.Request("POST", "http://127.0.0.1:1/inference/v1/generate")
        response = httpx.Response(400, json={"detail": str(error.value)}, request=request)
        raise openai.BadRequestError(str(error.value), response=response, body=None)

    with pytest.raises(openai.BadRequestError) as raised:
        asyncio.run(endpoint())
    assert not isinstance(raised.value, OverlongPromptError)
    capped = dataclasses.replace(POLICY, max_episode_tokens=at_cap)
    for raw in ("error", "context_length", "agent_completed"):
        kept, released = _miner_screen(group, raw=raw, policy=capped)
        assert kept == [] and len(released) == len(group.request.rollouts), raw
    _refused_by_both(group, raw="context_length", policy=capped)
    # One token more and the next prompt overflows: verifiers' context_length, admitted by both.
    over = dataclasses.replace(POLICY, max_episode_tokens=at_cap - 1)
    assert isinstance(_validator_check(group, policy=over), EpisodeGroupFacts)
    kept, released = _miner_screen(group, raw="context_length", policy=over)
    assert released == [] and [a.stop for _, a in kept] == ["context_length"] * len(kept)


# The RL open: a throttled seed is reopened (same seed) after its wait, within the window.

class ScriptedSessions(RlSessions):
    def __init__(self, grant, script):
        super().__init__(grant)
        self.script = list(script)

    def open(self, body):
        self.opened.append(body)
        if self.script:
            raise self.script.pop(0)
        return self.grant


def _scripted(tmp_path, script, **kwargs):
    validator, machine = episode_signers(tmp_path)
    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    _, _, transcript = play_episode(validator=validator, machine=machine, precommit=precommit, seed=2,
                                    session_id="s-1", reward=1.0)
    sessions = ScriptedSessions({"session_id": "s-1", "token": transcript["token"], "gateway_url": "http://g:1",
                                 "expires_at": NOW + 4500}, script)
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds

    _, runner = _runner(sessions, runner_factory=lambda url: Harness(transcript, b""),
                        clock=lambda: NOW + 10 + now[0], monotonic=lambda: now[0], sleep=sleep,
                        new_request_id=lambda: "c" * 32, **kwargs)
    return sessions, runner, now


def test_a_throttled_seed_is_reopened_after_its_retry_after(tmp_path):
    script = [SessionRefused("job_live_cap", retry_after=7.0, status=429),
              SessionRefused("open_busy", retry_after=5.0, status=503),
              SessionRefused("environment_not_served", retry_after=3.0, status=503),
              SessionRefused("sandbox_capacity", status=503)]
    sessions, runner, now = _scripted(tmp_path, script)

    async def go():
        async with runner:
            return await runner.run(2)

    result = asyncio.run(go())
    assert result.ok, result.error
    assert [body["engagement"]["precommit"]["seed_index"] for body in sessions.opened] == [2] * 5
    assert now[0] >= 7 + 5 + 3 and runner._unavailable == 0


def test_the_rl_named_throttles_never_stop_the_hotkey(tmp_path):
    script = [SessionRefused("environment_not_served", retry_after=1.0, status=503) for _ in range(12)]
    sessions, runner, _ = _scripted(tmp_path, script)

    async def go():
        async with runner:
            return await runner.run(2)

    assert asyncio.run(go()).ok and len(sessions.opened) == 13


@pytest.mark.parametrize("reason", ["precommit_unknown", "precommit_stale", "seed_out_of_pool", "engagement_taken",
                                    "session_too_long"])
def test_a_final_refusal_ends_the_seed_and_holds_no_other_open(tmp_path, reason):
    sessions, runner, now = _scripted(tmp_path, [SessionRefused(reason, status=409)])

    async def go():
        async with runner:
            with pytest.raises(SessionRefused) as refused:
                await runner.run(2)
            assert refused.value.reason == reason
            assert runner._not_before <= now[0]               # no exponential hold on the next seed's open
            return await runner.run(2)

    assert asyncio.run(go()).ok
    assert len(sessions.opened) == 2 and now[0] == 0.0


def test_reopening_is_bounded_by_the_window(tmp_path):
    script = [SessionRefused("open_busy", retry_after=100.0, status=503) for _ in range(10)]
    sessions, runner, now = _scripted(tmp_path, script, open_until=NOW + 10 + 250)

    async def go():
        async with runner:
            with pytest.raises(SessionRefused) as refused:
                await runner.run(2)
            return refused.value

    assert asyncio.run(go()).reason == "open_busy"
    assert len(sessions.opened) == 3 and now[0] == 200.0
