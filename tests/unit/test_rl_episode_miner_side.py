"""The miner side of RL episodes: routes, runner hooks, the stop, the signed-episode commit (plan 2C, Task 14)."""
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
    build_signed_episode_commit, episode_metadata, episode_sequence, episode_stop,
)
from reliquary.miner.rl_episode_client import (  # noqa: E402
    HttpRlEpisodes, RlSignedEpisodeRunner, rl_transcript_refusal, signed_precommit_body, signed_rl_open_request,
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
    def __init__(self, transcript, state):
        self.transcript, self.state, self.runs = transcript, state, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, *, token, prompt, on_trace=None):
        self.runs.append(prompt)
        trace = SimpleNamespace(id="t-1", stop_condition="agent_completed")
        return SimpleNamespace(trace=trace, state=self.state, transcript=self.transcript, refusals=(), error=None,
                               final={"body": {"status": "graded", "reward": 1.0, "reason": None}})


def _play(tmp_path, *, seed, state=b"\xff\xfe not text", run_seed=None):
    validator, machine = episode_signers(tmp_path)
    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    _, _, transcript = play_episode(validator=validator, machine=machine, precommit=precommit, seed=seed,
                                    session_id="s-1", reward=1.0)
    sessions = RlSessions({"session_id": "s-1", "token": transcript["token"], "gateway_url": "http://g:1",
                           "expires_at": NOW + 4500})
    harness = Harness(transcript, state)
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
    # Cut at the per-turn cap on the last allowed turn.
    w = world(tmp_path / "b", last=[TEXT] * 11, stop="agent_completed")
    limited = dataclasses.replace(POLICY, max_tokens_per_turn=11, max_turns=2)
    stop, outcome = admitted(w, limited, "agent_completed")
    assert stop == "max_turns" and isinstance(outcome, EpisodeGroupFacts), outcome
    # The same cut before the last allowed turn: nothing the validator admits, so nothing is submitted.
    w = world(tmp_path / "c", last=[TEXT] * 11, stop="agent_completed")
    early = dataclasses.replace(POLICY, max_tokens_per_turn=11, max_turns=3)
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
    episode = episode_metadata(precommit_sha256="e" * 64, seed_index=seed, spans=spans, stop="max_turns",
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
