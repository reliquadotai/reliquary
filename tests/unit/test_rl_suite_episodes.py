"""Replayed episodes (deterministic tool worlds) on the async RL path: the miner plays a lease's 16 EnvScaler
episodes on the forced episode engine, the validator replays every one through the world and grades it, the group
is buffered and reaches the trainer with its model spans. Over the two-world EnvScaler corpus of
``envscaler_v2_fixtures`` (real world processes, the real runner, ChatML renderer and replay) and a byte
tokenizer whose ChatML markers are one token each; the engine is scripted (no GPU)."""
from __future__ import annotations

import asyncio
import dataclasses
import random
from types import SimpleNamespace

import pytest

from reliquary.corpus.suite_episode import SuiteReplayOutcome, TokenizerCodec
from reliquary.environment.agentic.chatml import ChatMLEpisodeRenderer
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import EnvScalerToolsV2Environment
from reliquary.environment.agentic.types import AssistantAction
from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.miner.forced_draw import ForcedDraws, LeaseDrawBinding
from reliquary.miner.rl_async_episodes import EpisodeLeaseFailed
from reliquary.miner.rl_suite_episodes import SuiteLeasePlayer
from reliquary.protocol.profiles import PROFILES
from reliquary.validator.rl_submit import Rollout
from reliquary.validator.rl_suite_episodes import (
    InlineLeaseReplayer, LeaseSuiteChecker, LeaseSuiteReplayer, is_replayed_env, lease_episode_seed,
    suite_lease_policy,
)
from tests.unit import envscaler_v2_fixtures as fixtures
from tests.unit.test_suite_episode_miner import IM_END, ByteTokenizer

ENV = "envscaler_tools_v2"
V10 = PROFILES["teutonic-9b-reliquary-suite-v10-gumbel-dev1"]
NONCE = "ab" * 32
CODEC = TokenizerCodec(ByteTokenizer())
ADD_APPLE = AssistantAction(kind="tool", tool="add_item", arguments={"name": "apple", "qty": 1})
FINAL = AssistantAction.final("done")


def _turn(action) -> list[int]:
    return CODEC.encode(ChatMLEpisodeRenderer.action_text(action, reasoning="Think.")) + [IM_END]


class ScriptedEngine:
    """The forced engine as the player sees it: a session must be bound to the lease's draw before its first
    turn; each turn is capped as ``GenerateEngine`` caps it (per turn, what is left of the episode budget).
    Rollout ``i`` plays ``scripts(i)``."""

    def __init__(self, draws, scripts, *, per_turn=8192, total=65536):
        self.draws, self.scripts, self.per_turn, self.total = draws, scripts, per_turn, total
        self.turns: dict[str, int] = {}
        self.bindings: dict[str, LeaseDrawBinding] = {}
        self.taken: list[str] = []

    async def generate(self, session_id, prompt_ids, max_tokens):
        binding = self.draws.get(session_id)
        assert isinstance(binding, LeaseDrawBinding), "an unbound session"
        self.bindings[session_id] = binding
        turn = self.turns.get(session_id, 0)
        self.turns[session_id] = turn + 1
        ids = _turn(self.scripts(binding.rollout_index)[turn])
        cap = min(self.per_turn, self.total - len(prompt_ids))
        return {"choices": [{"token_ids": ids[:cap]}]}

    def take_session(self, session_id):
        self.taken.append(session_id)


def _environment():
    return EnvScalerToolsV2Environment(fixtures.corpus())


def _lease(**kw):
    values = dict(lease_id="lease-suite-1", hotkey="5Hot", environment=ENV, prompt_idx=fixtures.SHOP, checkpoint=3,
                  nonce=NONCE)
    values.update(kw)
    return SimpleNamespace(**values)


def _solves(index):
    return [ADD_APPLE, FINAL] if index % 2 == 0 else [FINAL]


def _setting(policy=None, scripts=_solves):
    environment = _environment()
    policy = policy or suite_lease_policy(ENV, profile=V10, environment=environment)
    draws = ForcedDraws()
    engine = ScriptedEngine(draws, scripts, per_turn=policy.max_tokens_per_turn, total=policy.max_episode_tokens)
    player = SuiteLeasePlayer(policy=policy, environment=environment, codec=CODEC, engine=engine, draws=draws)
    checker = LeaseSuiteChecker(policy=policy, environment=environment, codec=CODEC,
                                replayer=InlineLeaseReplayer(environment, encode=CODEC.encode, decode=CODEC.decode))
    return SimpleNamespace(policy=policy, engine=engine, player=player, checker=checker, draws=draws)


def _rollouts(played):
    return [Rollout(r.tokens, None, r.wire()) for r in played]


def test_envscaler_is_a_replayed_env_of_the_v10_profile(real_rl_prompt_ranges):
    from reliquary.protocol.rl_prompt_ranges import rl_prompt_range

    assert is_replayed_env(ENV) and not is_replayed_env("reliquary_dapo_math_v1")
    policy = suite_lease_policy(ENV, profile=V10)
    assert (policy.max_turns, policy.max_tokens_per_turn, policy.max_episode_tokens) == (25, 8192, 65536)
    assert policy.renderer_id == "reliquary-chatml-tools-v1"
    assert rl_prompt_range(ENV, 7825) == (0, 1296)
    with pytest.raises(ValueError):
        suite_lease_policy("reliquary_dapo_math_v1", profile=V10)


def test_a_lease_of_replayed_episodes_is_played_bound_to_its_draw_and_graded_by_the_replay():
    s = _setting()
    lease = _lease()
    played = asyncio.run(s.player.play(lease))
    assert len(played) == 16 and all(r.transcript == {} for r in played)
    assert [r.stop for r in played] == ["agent_completed"] * 16
    assert [len(r.spans) for r in played] == [2, 1] * 8
    # every session bound to the lease's draw, one per rollout, and its log released
    bindings = sorted(s.engine.bindings.values(), key=lambda b: b.rollout_index)
    assert [b.rollout_index for b in bindings] == list(range(16))
    assert {(b.nonce, b.prompt_idx, b.checkpoint) for b in bindings} == {(NONCE, fixtures.SHOP, 3)}
    assert sorted(s.engine.taken) == sorted(s.engine.bindings) and s.draws.get(s.engine.taken[0]) is None
    graded = s.checker.grade(lease, _rollouts(played), received=0.0)
    assert graded.reject is None and graded.rewards == (1.0, 0.0) * 8 and graded.truncated == (False,) * 16
    assert graded.prompt_tokens == tuple(s.player.prompt_ids(fixtures.SHOP))
    assert graded.spans == tuple(r.spans for r in played)
    assert lease_episode_seed(lease) == lease_episode_seed(_lease(lease_id="other"))   # keyed by nonce and task
    assert lease_episode_seed(lease) != lease_episode_seed(_lease(nonce="cd" * 32))


def test_a_forged_observation_or_claim_is_refused():
    s = _setting()
    lease = _lease()
    played = asyncio.run(s.player.play(lease))
    rollouts = _rollouts(played)
    first = rollouts[0]
    (_, end), (start, _) = first.spans
    observation = list(first.tokens)
    at = (end + start) // 2                                  # inside the tool's answer
    observation[at] = ord("X") if observation[at] != ord("X") else ord("Y")
    forged = Rollout(tuple(observation), None, played[0].wire())
    graded = s.checker.grade(lease, [forged, *rollouts[1:]], received=0.0)
    assert graded.reject == "bad_observation"
    # a stop the replay does not end on
    claimed = Rollout(first.tokens, None, {**played[0].wire(), "stop": "max_turns"})
    assert s.checker.grade(lease, [claimed, *rollouts[1:]], received=0.0).reject == "bad_stop"
    # a replayed episode carries no transcript
    signed = Rollout(first.tokens, None, {**played[0].wire(), "transcript": {"records": []}})
    assert s.checker.grade(lease, [signed, *rollouts[1:]], received=0.0).reject == "episode_schema"
    # episodes played on another world's task do not replay on this one
    assert s.checker.grade(_lease(prompt_idx=fixtures.HOSTILE), rollouts, received=0.0).reject == "bad_observation"


def test_a_cut_episode_is_scored_zero_and_an_early_cut_is_refused():
    environment = _environment()
    prompt = len(SuiteLeasePlayer(policy=suite_lease_policy(ENV, profile=V10), environment=environment,
                                  codec=CODEC, engine=None, draws=None).prompt_ids(fixtures.SHOP))
    turn = len(_turn(ADD_APPLE))
    # room for the first turn, not for the observation that answers it
    policy = dataclasses.replace(suite_lease_policy(ENV, profile=V10), max_episode_tokens=prompt + turn + 20)
    s = _setting(policy=policy, scripts=lambda index: [ADD_APPLE, FINAL] if index < 8 else [FINAL])
    lease = _lease()
    played = asyncio.run(s.player.play(lease))
    assert [r.stop for r in played] == ["context_length"] * 8 + ["agent_completed"] * 8
    graded = s.checker.grade(lease, _rollouts(played), received=0.0)
    assert graded.reject is None and graded.rewards == (0.0,) * 16
    assert graded.truncated == (True,) * 8 + (False,) * 8
    # a solved episode cut after its first turn while it had room: refused, never scored
    honest = _setting()
    full = asyncio.run(honest.player.play(lease))
    (first_end,) = [full[0].spans[0][1]]
    early = Rollout(full[0].tokens[:first_end], None,
                    {"spans": [list(full[0].spans[0])], "stop": "context_length", "transcript": {}})
    graded = honest.checker.grade(lease, [early, *_rollouts(full)[1:]], received=0.0)
    assert graded.reject == "bad_stop"


def test_the_miner_gives_up_a_lease_whose_episode_failed():
    def scripts(index):
        if index == 5:
            raise RuntimeError("engine step failed")
        return _solves(index)

    s = _setting(scripts=scripts)
    with pytest.raises(EpisodeLeaseFailed, match="rollout 5"):
        asyncio.run(s.player.play(_lease()))


class _Future:
    def __init__(self, outcome=None, error=None):
        self.outcome, self.error = outcome, error

    def result(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.outcome

    def running(self):
        return True

    def cancel(self):
        pass


class _Pool:
    """``PoolSuiteReplayer``'s surface the lease replayer drives."""

    def __init__(self, futures):
        self.futures, self.dropped = list(futures), []
        self.workers = 2

    def _executor(self):
        return self

    def submit(self, *args, **kwargs):
        return self.futures.pop(0)

    def _settled(self, executor, outcome):
        return outcome

    def _drop(self, executor, why):
        self.dropped.append(why)

    def _timed_out(self, executor, future):
        return RuntimeError("no replay answer")


def _replayer(futures):
    replayer = LeaseSuiteReplayer.__new__(LeaseSuiteReplayer)
    replayer.pool, replayer.deadline_seconds, replayer.wait_seconds = _Pool(futures), 1.0, 1.0
    return replayer


def test_a_replay_the_pool_cannot_answer_is_the_validators_outage():
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool

    ok = SuiteReplayOutcome(None, reward=1.0, termination_reason="finished")
    assert _replayer([_Future(ok)]).replay(ENV) is ok
    with pytest.raises(GraderInfrastructureError, match="deadline"):
        _replayer([_Future(SuiteReplayOutcome(("bad_observation", {}), deadline=True))]).replay(ENV)
    with pytest.raises(GraderInfrastructureError):
        _replayer([_Future(error=FutureTimeout())]).replay(ENV)
    broken = _replayer([_Future(error=BrokenProcessPool()), _Future(ok)])
    assert broken.replay(ENV) is ok and len(broken.pool.dropped) == 1
    with pytest.raises(GraderInfrastructureError, match="twice"):
        _replayer([_Future(error=BrokenProcessPool()), _Future(error=BrokenProcessPool())]).replay(ENV)
    # through the checker: an outage raises (the lease stays open), never a refusal
    s = _setting()
    lease = _lease()
    rollouts = _rollouts(asyncio.run(s.player.play(lease)))
    s.checker.replayer = _replayer([_Future(error=FutureTimeout())] * 16)
    with pytest.raises(GraderInfrastructureError):
        s.checker.grade(lease, rollouts, received=0.0)


def test_a_lease_of_replayed_episodes_is_submitted_buffered_and_trained_on_model_tokens(tmp_path):
    from reliquary.trainer.async_trainer import training_group
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseRequest
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions, SubmitConfig

    s = _setting()
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}),
                     quota=QuotaController(QuotaConfig(initial=8, ceiling=8)), clock=lambda: 100.0)
    book.adopt_checkpoint(3)
    granted = book.grant([LeaseRequest("5Hot", ENV, (fixtures.SHOP,), 3)])[0].leases[0]
    grader = GroupGrader({}, SimpleNamespace(), eos_token_ids=(IM_END,), episodes={ENV: s.checker})
    subs = RlSubmissions(book, grader, config=SubmitConfig(), clock=lambda: 100.0, rng=random.Random(0))
    lease = _lease(lease_id=granted.lease_id, nonce=granted.nonce, prompt_idx=granted.prompt_idx,
                   checkpoint=granted.checkpoint)
    played = asyncio.run(s.player.play(lease))
    rollouts = _rollouts(played)
    answer = subs.submit(granted.lease_id, "5Hot", rollouts)
    assert answer["verdict"] == "in_zone" and answer["rewards"] == [1.0, 0.0] * 8
    (buffered,) = subs.buffer.take(4)
    assert buffered.spans == tuple(r.spans for r in rollouts)
    group = training_group(buffered, slot0=0, index=0)
    p = len(buffered.prompt_tokens)
    for rollout, sent in zip(group.rollouts, rollouts):
        assert rollout._validated_assistant_spans == tuple((p + a, p + b) for a, b in sent.spans)
        assert len(rollout.commit["rollout"]["token_logprobs"]) == sum(b - a for a, b in sent.spans)
    model = sum(b - a for a, b in rollouts[0].spans)
    assert subs.group(granted.lease_id)["completion_tokens"] == sum(sum(b - a for a, b in r.spans) for r in rollouts)
    assert model < len(rollouts[0].tokens)                    # observations are no model work
    book.close()


def test_the_episode_stack_plays_replayed_envs_on_its_engine(tmp_path, monkeypatch):
    import reliquary.protocol.profiles as profiles
    from reliquary.miner.rl_async_episodes import EpisodeEngineHost, build_episode_generator
    from tests.unit.test_rl_async_episode_service import MINER, VALIDATOR, base_model, fake_deps, sign

    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", V10)
    made = []
    generator = build_episode_generator(
        {}, validator_url="http://v", validator_hotkey=VALIDATOR.ss58_address, hotkey=MINER.ss58_address,
        sign_binding=sign, base_model_dir=str(base_model(tmp_path)), work_dir=str(tmp_path / "work"),
        deps=fake_deps(made, [], []), suites=(ENV,), load_tokenizer=lambda directory: ByteTokenizer(),
        suite_environments={ENV: _environment()})
    assert generator.environments == {ENV} and not generator.policies
    player = generator.suites[ENV]
    assert isinstance(player, SuiteLeasePlayer) and player.policy.max_turns == 25
    host = generator.host
    assert isinstance(host, EpisodeEngineHost)
    assert (host.max_total_tokens, host.max_tokens_per_turn) == (65536, 8192)
    assert set(host._stop_ids()) == {IM_END, IM_END + 1}           # <|im_end|>, <|endoftext|>


async def test_mine_async_routes_a_replayed_env_to_the_episode_stack(tmp_path):
    from reliquary.miner.rl_async import AsyncMinerConfig, miner_generator
    from reliquary.miner.rl_async_episodes import MixedGenerator
    from tests.unit.test_rl_async_episode_service import MINER

    built = {}

    class Single:
        def __init__(self, **kw):
            built["single"] = kw

    class Episodes:
        def __init__(self, blocks, **kw):
            built["episodes"] = (blocks, kw)
            self.environments = frozenset(kw.get("suites", ()))

    generator = miner_generator(
        AsyncMinerConfig(environments=(ENV, "reliquary_dapo_math_v1")), blocks={}, base_model_dir=str(tmp_path),
        work_dir=str(tmp_path), wallet=SimpleNamespace(hotkey=MINER), build_single=Single, build_episodes=Episodes,
        validator_url="http://v", validator_hotkey="5V")
    assert isinstance(generator, MixedGenerator) and "single" in built
    blocks, kw = built["episodes"]
    assert blocks == {} and kw["suites"] == (ENV,)
    assert generator._episode_names() == {ENV}
