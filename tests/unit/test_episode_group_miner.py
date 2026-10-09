"""The episode-group miner (plan 2C, Task 15)."""
import asyncio
import inspect
from types import SimpleNamespace

import pytest

bt = pytest.importorskip("bittensor")
attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.constants import M_ROLLOUTS  # noqa: E402
from reliquary.corpus.trajectory import GeneratedTurn  # noqa: E402
from reliquary.miner import engine as engine_module  # noqa: E402
from reliquary.miner.agentic_episode import EpisodeResult  # noqa: E402
from reliquary.miner.corpus_generate_server import SessionLog  # noqa: E402
from reliquary.miner.episode_group_miner import EpisodeGroupMiner  # noqa: E402
from reliquary.miner.forced_draw import ForcedDraws  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.protocol.service_episode import EpisodePrecommit  # noqa: E402
from reliquary.protocol.signatures import verify_episode_precommit_signature  # noqa: E402
from reliquary.protocol.submission import RejectReason  # noqa: E402
from reliquary.protocol.toploc import span_chunk_count  # noqa: E402
from reliquary.sandbox.rl_routes import episode_precommit_path  # noqa: E402
from tests.unit.episode_v2_fixtures import (  # noqa: E402
    EPISODE, PROMPT_TEXT, TASK, WINDOW_BEACON, episode_runtime, episode_signers, play_episode,
)
from tests.unit.sandbox_fixtures import NOW  # noqa: E402
from tests.unit.test_trajectory_parse import FakeRenderer  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
CAPS = {"max_total_tokens": 4096, "max_tokens_per_turn": 512, "max_model_len": 4096}   # episode_block's
CHUNK = 32


class Sessions:
    validator_hotkey, prefix = VALIDATOR.ss58_address, "/rl"

    def __init__(self, refuse=()):
        self.bodies, self.refuse = [], list(refuse)

    def precommit(self, body):
        self.bodies.append(body)
        if self.refuse:
            raise self.refuse.pop(0)
        return {"precommit_sha256": EpisodePrecommit.from_dict(body["precommit"]).sha256, "created": True}


class Runner:
    """A played pool: ``rewards[seed]`` is the graded reward (None: the episode did not end graded); a seed in
    ``stops`` ends with that raw stop; one in ``refused`` gets no session; one in ``crash`` raises mid-episode."""

    def __init__(self, keys, rewards, *, refused=(), stops=None, crash=(), submit_by=None):
        self.keys, self.rewards, self.refused, self.crash = keys, rewards, set(refused), set(crash)
        self.stops, self.submit_by = dict(stops or {}), submit_by
        self.released, self.submitted, self.played, self.entered = [], [], {}, []
        self.precommit = self.prompt = self.open_until = None

    async def __aenter__(self):
        self.entered.append("in")
        return self

    async def __aexit__(self, *exc):
        self.entered.append("out")

    async def run(self, seed, on_session=None):
        if seed in self.refused:
            raise SessionRefused("sandbox_capacity")
        trace = f"trace-{seed}"
        on_session(trace)
        if seed in self.crash:
            raise RuntimeError("the harness broke")
        reward = self.rewards.get(seed)
        if reward is None:
            return EpisodeResult(trace, "", None, False, None, error="box_failed")
        validator, machine = self.keys
        self.played[trace] = play_episode(validator=validator, machine=machine, precommit=self.precommit, seed=seed,
                                          session_id=f"s-{seed}", reward=reward)

        async def release():
            self.released.append(seed)

        return EpisodeResult(trace, "", self.stops.get(seed, "agent_completed"), True, reward,
                             transcript=self.played[trace][2], release=release,
                             submitted=lambda: self.submitted.append(seed), submit_by=self.submit_by)


class Engine:
    def __init__(self, runner):
        self.runner, self.taken, self.dropped = runner, [], []

    def take_session(self, trace):
        self.taken.append(trace)
        if trace not in self.runner.played:
            return SessionLog()
        tokens, spans, _ = self.runner.played[trace]
        return SessionLog(turns=[GeneratedTurn(tuple(tokens[:start]), tuple(tokens[start:end]), ())
                                 for start, end in spans])

    def drop_session(self, trace):
        self.dropped.append(trace)


class Draws(ForcedDraws):
    def __init__(self):
        super().__init__()
        self.history = []

    def bind(self, session_id, binding):
        self.history.append((session_id, binding.seed_index))
        super().bind(session_id, binding)


def prove(generation, randomness):
    """A commit shaped like ``build_signed_episode_commit``'s (stub proofs, one per span chunk)."""
    spans = generation["spans"]
    model_tokens = sum(end - start for start, end in spans)
    return {"tokens": list(generation["tokens"]), "commitments": [{} for _ in generation["tokens"]],
            "proof_version": "public-group-proof/v1", "model": {"name": "model", "layer_index": -1},
            "signature": "aa", "beacon": {"randomness": randomness},
            "rollout": {"prompt_length": spans[0][0], "completion_length": len(generation["tokens"]) - spans[0][0],
                        "success": False, "total_reward": 0.0, "advantage": 0.0,
                        "token_logprobs": [-1.0] * model_tokens, "episode": generation["episode"],
                        "seed_pool": generation["seed_pool"], "service_binding": generation["service_binding"]},
            "toploc_proofs": ["AAAA"] * sum(span_chunk_count(end - start, CHUNK) for start, end in spans)}


def world(tmp_path, rewards, *, verdicts=("accepted",), sessions=None, miner_class=EpisodeGroupMiner, caps=CAPS,
          clock=None, **runner_kwargs):
    rt = episode_runtime(tmp_path / "rt")
    keys = episode_signers(tmp_path / "keys")
    runner = Runner(keys, rewards, **runner_kwargs)
    engine, draws, submitted, slept = Engine(runner), Draws(), [], []
    verdicts = list(verdicts)

    async def submit(request):
        submitted.append(request)
        reason = RejectReason(verdicts.pop(0) if len(verdicts) > 1 else verdicts[0])
        return SimpleNamespace(accepted=reason is RejectReason.ACCEPTED, reason=reason)

    async def sleep(seconds):
        slept.append(seconds)

    def runner_factory(policy, precommit, *, prompt, open_until):
        runner.precommit, runner.prompt, runner.open_until = precommit, prompt, open_until
        return runner

    kwargs = {} if clock is None else {"clock": clock}
    miner = miner_class(hotkey=MINER.ss58_address, sign_binding=lambda b: MINER.sign(b).hex(),
                        sessions=sessions or Sessions(), runner_factory=runner_factory, engine=engine,
                        draws=draws, prove=prove, submit=submit, renderer_for=lambda policy: FakeRenderer(),
                        task_prompt=lambda environment, task: PROMPT_TEXT, engine_caps=caps, sleep=sleep, **kwargs)
    announcement = rt.announcement(window=1, randomness=WINDOW_BEACON)

    def mine(open_until=None):
        return asyncio.run(miner.mine_task(announcement=announcement, randomness="cd" * 32, window=1,
                                           environment=EPISODE, task_index=TASK, open_until=open_until))

    return SimpleNamespace(rt=rt, keys=keys, runner=runner, engine=engine, draws=draws, submitted=submitted,
                           slept=slept, mine=mine, miner=miner,
                           pool=rt.seed_pool(environment=EPISODE, prompt_idx=TASK, window=1))


def alternating(count):
    return {seed: float(seed % 2) for seed in range(count)}


def validator_check(w, request):
    """The validator's own episode admission on the miner's group."""
    from reliquary.validator.episode_admission import EpisodeGroupChecker
    from tests.unit.episode_v2_fixtures import FixedSource
    from tests.unit.sandbox_fixtures import directory

    validator, machine = w.keys
    policy = w.rt.contract.episode_policy(EPISODE)
    checker = EpisodeGroupChecker(policy=policy, renderer=FakeRenderer(), source=FixedSource(), chunk_tokens=CHUNK)
    verifier = attest.Ed25519TokenVerifier({validator.key_id: validator.public_key_b64})
    return checker.check(request, precommit=w.runner.precommit, directory=directory(machine),
                         token_verifier=verifier, seen=frozenset(), received=NOW + 100)


def test_an_honest_pool_is_precommitted_played_chosen_submitted_and_the_rest_withdrawn(tmp_path):
    from reliquary.validator.episode_admission import EpisodeGroupFacts

    w = world(tmp_path, alternating(2 * M_ROLLOUTS))
    response = w.mine(open_until=123.0)
    assert response.accepted
    (body,) = w.miner._sessions.bodies
    precommit = EpisodePrecommit.from_dict(body["precommit"])
    assert (precommit.environment, precommit.task_index, precommit.window) == (EPISODE, TASK, 1)
    assert precommit.pool_sha256 == w.pool.sha256 and precommit.hotkey == MINER.ss58_address
    assert precommit == w.runner.precommit and w.runner.prompt == PROMPT_TEXT and w.runner.open_until == 123.0
    assert verify_episode_precommit_signature(MINER.ss58_address, body["precommit"], at=body["at"],
                                              signature=body["signature"], validator_hotkey=VALIDATOR.ss58_address,
                                              path=episode_precommit_path("/rl"))
    # Every seed's session was bound to its own draw, and no binding is left once the episodes ended.
    assert sorted(w.draws.history) == sorted((f"trace-{s}", s) for s in range(2 * M_ROLLOUTS))
    assert all(w.draws.get(f"trace-{s}") is None for s in range(2 * M_ROLLOUTS))
    assert sorted(w.engine.taken) == sorted(f"trace-{s}" for s in range(2 * M_ROLLOUTS)) and w.engine.dropped == []
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(M_ROLLOUTS))
    episodes = [r.commit["rollout"]["episode"] for r in request.rollouts]
    assert [e["seed_index"] for e in episodes] == list(range(M_ROLLOUTS))
    assert all(e["precommit_sha256"] == precommit.sha256 and e["stop"] == "agent_completed" for e in episodes)
    tokens, spans, signed = w.runner.played["trace-0"]
    assert request.rollouts[0].tokens == tokens and episodes[0]["assistant_spans"] == [list(s) for s in spans]
    assert episodes[0]["transcript"] == signed
    assert [r.reward for r in request.rollouts] == [float(s % 2) for s in range(M_ROLLOUTS)]
    assert request.service_binding["purpose"] == "training"
    assert w.runner.submitted == list(range(M_ROLLOUTS))
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))
    assert w.runner.entered == ["in", "out"]
    # What the miner sends is what the validator's own admission takes.
    assert isinstance(validator_check(w, request), EpisodeGroupFacts)


def test_a_refused_group_withdraws_every_episode_and_is_never_sent_again(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=("out_of_zone",))
    assert not w.mine().accepted
    assert len(w.submitted) == 1
    assert w.runner.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_a_group_the_validator_was_too_busy_for_is_sent_again(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=("worker_dropped", "accepted"))
    assert w.mine().accepted
    assert len(w.submitted) == 2 and len(w.slept) == 1
    assert w.runner.submitted == list(range(M_ROLLOUTS))


def test_a_group_past_its_grading_deadline_is_not_sent(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), submit_by=NOW, clock=lambda: NOW + 1)
    assert w.mine() is None
    assert w.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_too_few_graded_episodes_submit_nothing(tmp_path):
    w = world(tmp_path, alternating(M_ROLLOUTS - 1))
    assert w.mine() is None
    assert w.submitted == [] and sorted(w.runner.released) == list(range(M_ROLLOUTS - 1))


def test_an_episode_the_validator_would_refuse_is_withdrawn_before_the_choice(tmp_path):
    seen = []

    class Recording(EpisodeGroupMiner):
        def choose_episodes(self, outcomes, *, group_size):
            seen.extend(outcome.seed_index for outcome in outcomes)
            return super().choose_episodes(outcomes, group_size=group_size)

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), stops={0: "error", 1: "max_total_tokens"},
              miner_class=Recording)
    assert w.mine().accepted
    assert 0 not in seen and 1 not in seen and len(seen) == 2 * M_ROLLOUTS - 2
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(2, M_ROLLOUTS + 2))
    assert sorted(w.runner.released) == [0, 1] + list(range(M_ROLLOUTS + 2, 2 * M_ROLLOUTS))


def test_a_refused_seed_is_skipped_and_a_crashed_one_drops_its_session(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), refused={0}, crash={1})
    w.mine()
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(2, M_ROLLOUTS + 2))
    assert w.engine.dropped == ["trace-1"] and w.draws.get("trace-1") is None


def test_a_miner_may_cherry_pick_any_seeds(tmp_path):
    class HighSeeds(EpisodeGroupMiner):
        def choose_episodes(self, outcomes, *, group_size):
            return sorted(outcomes, key=lambda o: -o.seed_index)[:group_size]

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), miner_class=HighSeeds)
    w.mine()
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS))


def test_a_chooser_that_breaks_the_rules_releases_everything(tmp_path):
    class Twice(EpisodeGroupMiner):
        def choose_episodes(self, outcomes, *, group_size):
            return [outcomes[0]] * group_size

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), miner_class=Twice)
    with pytest.raises(ValueError, match="distinct seeds"):
        w.mine()
    assert w.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_a_failing_submit_releases_everything(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS))

    async def broken(request):
        raise RuntimeError("all retries failed")

    w.miner._submit = broken
    with pytest.raises(RuntimeError):
        w.mine()
    assert w.runner.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_a_uniform_group_is_sent_as_exploration(tmp_path):
    w = world(tmp_path, {seed: 0.0 for seed in range(2 * M_ROLLOUTS)})
    w.mine()
    assert w.submitted[0].service_binding["purpose"] == "exploration"


def test_a_precommit_already_recorded_is_reused_another_is_an_error(tmp_path):
    rt = episode_runtime(tmp_path / "probe")
    pool = rt.seed_pool(environment=EPISODE, prompt_idx=TASK, window=1)
    mine = EpisodePrecommit(order=rt.contract.sha256, window=1, environment=EPISODE, task_index=TASK,
                            checkpoint=rt.envelope(1)["checkpoint"]["revision"], pool_sha256=pool.sha256,
                            hotkey=MINER.ss58_address)
    same = Sessions(refuse=[SessionRefused("precommit_exists", {"precommit_sha256": mine.sha256}, status=409)])
    assert world(tmp_path / "same", alternating(2 * M_ROLLOUTS), sessions=same).mine().accepted
    other = Sessions(refuse=[SessionRefused("precommit_exists", {"precommit_sha256": "f" * 64}, status=409)])
    w = world(tmp_path / "other", alternating(2 * M_ROLLOUTS), sessions=other)
    with pytest.raises(SessionRefused):
        w.mine()
    assert w.runner.entered == [] and w.draws.history == []


def test_a_throttled_precommit_is_sent_again_within_the_window(tmp_path):
    busy = [SessionRefused("precommit_rate", status=429, retry_after=7.0)]
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), sessions=Sessions(refuse=list(busy)))
    assert w.mine().accepted
    assert len(w.miner._sessions.bodies) == 2 and w.slept == [7.0]
    late = world(tmp_path / "late", alternating(2 * M_ROLLOUTS), sessions=Sessions(refuse=list(busy)),
                 clock=lambda: 100.0)
    with pytest.raises(SessionRefused):
        late.mine(open_until=105.0)
    assert late.slept == [] and late.runner.entered == []


def test_an_engine_whose_caps_are_not_the_policys_mines_nothing(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), caps=dict(CAPS, max_tokens_per_turn=1024))
    with pytest.raises(ValueError, match="max_tokens_per_turn"):
        w.mine()
    assert w.miner._sessions.bodies == []


def test_the_single_turn_engine_refuses_a_signed_episode_environment(monkeypatch):
    from tests.unit.episode_v2_fixtures import register_episode_env
    from tests.unit.service_v2_fixtures import MATH

    register_episode_env(monkeypatch)
    with pytest.raises(ValueError, match="signed-episode environment"):
        engine_module._single_turn_mined_spec(EPISODE)
    assert engine_module._single_turn_mined_spec(MATH).name == MATH
    source = inspect.getsource(engine_module.MiningEngine.mine_window)
    assert source.index("_single_turn_mined_spec(env_name)") < source.index("env.get_problem(prompt_idx)")
