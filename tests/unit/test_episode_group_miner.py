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

    def __init__(self, keys, rewards, *, refused=(), stops=None, crash=(), submit_by=None, slow=(), aborted=None):
        self.keys, self.rewards, self.refused, self.crash = keys, rewards, set(refused), set(crash)
        self.stops, self.submit_by = dict(stops or {}), submit_by
        self.slow, self.aborted = set(slow), dict(aborted or {})    # aborted: seed -> plays ending aborted
        self.released, self.submitted, self.played, self.entered, self.runs = [], [], {}, [], []
        self.precommit = self.prompt = self.open_until = None

    async def __aenter__(self):
        self.entered.append("in")
        return self

    async def __aexit__(self, *exc):
        self.entered.append("out")

    async def run(self, seed, on_session=None):
        if seed in self.refused:
            raise SessionRefused("sandbox_capacity")
        plays = self.runs.count(seed)
        self.runs.append(seed)
        trace = f"trace-{seed}" if plays == 0 else f"trace-{seed}-{plays}"
        on_session(trace)
        if seed in self.crash:
            raise RuntimeError("the harness broke")
        if seed in self.slow:
            await asyncio.sleep(20)
        if plays < self.aborted.get(seed, 0):
            return EpisodeResult(trace, "", None, False, None, error="the sandbox episode ended aborted",
                                 final_status="aborted")
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


class Validator:
    """``/submit`` and the verdict route, faked. Each send takes the next item of ``verdicts``: ``"reason"``
    or ``"reason/stage"`` is answered by a queue receipt (SUBMITTED) and recorded as that verdict after
    ``pending`` polls (until then the route shows the previous record, as the real one does for a resent
    group of the same Merkle root); ``"sync:reason"`` is answered at once (no record)."""

    def __init__(self, verdicts, pending=0, interim=False):
        self.items, self.pending, self.interim = list(verdicts), pending, interim
        self.requests, self.records, self.polls, self.waiting = [], [], [], None

    async def submit(self, request):
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if item.startswith("sync:"):
            reason = RejectReason(item[len("sync:"):])
            response = SimpleNamespace(accepted=reason is RejectReason.ACCEPTED, reason=reason)
            response._retry_after_seconds = 9.0 if reason is RejectReason.WINDOW_NOT_ACTIVE else None
            return response
        reason, _, stage = item.partition("/")
        accepted = reason == "accepted"
        self.waiting = [self.pending, {"merkle_root": request.merkle_root, "window_n": request.window_start,
                                       "accepted": accepted, "reason": reason, "reject_stage": stage or None,
                                       "is_final": not accepted, "ts": float(len(self.records) + 1)}]
        return SimpleNamespace(accepted=True, reason=RejectReason.SUBMITTED)

    async def verdict(self, window, merkle_root):
        self.polls.append((window, merkle_root))
        if self.waiting is not None:
            if self.waiting[0] <= 0:
                self.records.append(self.waiting[1])
                self.waiting = None
            else:
                self.waiting[0] -= 1
                if self.interim:      # an admission record not decided yet
                    return {"status": "pending", "verdict": {"accepted": False, "is_final": False,
                                                             "reason": "submitted", "ts": 0.5}}
        if not self.records:
            return {"status": "not_found", "verdict": None}
        last = self.records[-1]
        return {"status": "found" if last["is_final"] else "pending", "source": "memory", "verdict": dict(last)}


class Clock:
    """Wall clock the fake sleep advances."""

    def __init__(self, now=NOW):
        self.now = float(now)

    def __call__(self):
        return self.now


def world(tmp_path, rewards, *, verdicts=("accepted",), pending=0, interim=False, sessions=None, miner_class=EpisodeGroupMiner,
          caps=CAPS, clock=None, proof_budget_s=None, **runner_kwargs):
    rt = episode_runtime(tmp_path / "rt")
    keys = episode_signers(tmp_path / "keys")
    runner = Runner(keys, rewards, **runner_kwargs)
    engine, draws, slept = Engine(runner), Draws(), []
    validator = Validator(verdicts, pending, interim)

    async def sleep(seconds):
        slept.append(seconds)
        if isinstance(clock, Clock):
            clock.now += seconds

    def runner_factory(policy, precommit, *, prompt, open_until):
        runner.precommit, runner.prompt, runner.open_until = precommit, prompt, open_until
        return runner

    kwargs = {} if clock is None else {"clock": clock}
    if proof_budget_s is not None:
        kwargs["proof_budget_s"] = proof_budget_s
    miner = miner_class(hotkey=MINER.ss58_address, sign_binding=lambda b: MINER.sign(b).hex(),
                        sessions=sessions or Sessions(), runner_factory=runner_factory, engine=engine,
                        draws=draws, prove=prove, submit=validator.submit, verdicts=validator.verdict,
                        renderer_for=lambda policy: FakeRenderer(),
                        task_prompt=lambda environment, task: PROMPT_TEXT, engine_caps=caps, sleep=sleep, **kwargs)
    announcement = rt.announcement(window=1, randomness=WINDOW_BEACON)

    def mine(open_until=None):
        return asyncio.run(miner.mine_task(announcement=announcement, randomness="cd" * 32, window=1,
                                           environment=EPISODE, task_index=TASK, open_until=open_until))

    return SimpleNamespace(rt=rt, keys=keys, runner=runner, engine=engine, draws=draws,
                           submitted=validator.requests, validator=validator, slept=slept, mine=mine, miner=miner,
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

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), clock=Clock())
    response = w.mine(open_until=NOW + 3600.0)
    assert response.accepted
    (body,) = w.miner._sessions.bodies
    precommit = EpisodePrecommit.from_dict(body["precommit"])
    assert (precommit.environment, precommit.task_index, precommit.window) == (EPISODE, TASK, 1)
    assert precommit.pool_sha256 == w.pool.sha256 and precommit.hotkey == MINER.ss58_address
    assert precommit == w.runner.precommit and w.runner.prompt == PROMPT_TEXT and w.runner.open_until == NOW + 3600.0
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
        def choose_episodes(self, outcomes, *, group_size, in_zone=None):
            seen.extend(outcome.seed_index for outcome in outcomes)
            return super().choose_episodes(outcomes, group_size=group_size, in_zone=in_zone)

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
        def choose_episodes(self, outcomes, *, group_size, in_zone=None):
            return sorted(outcomes, key=lambda o: -o.seed_index)[:group_size]

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), miner_class=HighSeeds)
    w.mine()
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS))


def test_a_chooser_that_breaks_the_rules_releases_everything(tmp_path):
    class Twice(EpisodeGroupMiner):
        def choose_episodes(self, outcomes, *, group_size, in_zone=None):
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


def test_the_kept_episodes_stay_held_until_the_verdict_behind_the_queue_receipt(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), pending=3, interim=True)
    seen = []
    poll = w.validator.verdict

    async def watched(window, merkle_root):
        seen.append((list(w.runner.submitted), list(w.runner.released)))
        return await poll(window, merkle_root)

    w.miner._verdicts = watched
    response = w.mine()
    assert response.accepted and response.verdict["reason"] == "accepted"
    (request,) = w.submitted
    assert w.validator.polls == [(1, request.merkle_root)] * 4 and w.slept == [2.0, 3.0, 4.5]   # backing off
    assert seen == [([], [])] * 4            # nothing submitted nor withdrawn while the verdict was pending
    assert w.runner.submitted == list(range(M_ROLLOUTS))
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))


def test_a_refused_verdict_behind_the_queue_receipt_withdraws_everything(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=("out_of_zone",), pending=2)
    response = w.mine()
    assert not response.accepted and response.reason == "out_of_zone"
    assert len(w.submitted) == 1 and w.runner.submitted == []
    assert sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


RESENT = ["worker_dropped/episode_session_busy", "worker_dropped/episode_directory",
          "worker_dropped/episode_checker_busy", "worker_dropped/episode_persist_failed",
          "worker_dropped/admission_timeout", "worker_dropped/prompt_source", "worker_dropped/admission_worker",
          "rate_limited/episode_group_in_flight", "rate_limited/episode_checks_in_flight", "sync:window_not_active",
          "sync:worker_dropped"]
FINAL = ["rate_limited/episode_rate", "rate_limited/episode_timeout", "out_of_zone/service_exploration_banned",
         "bad_schema/episode_transcript", "hash_duplicate/episode_session_reused",
         "hash_duplicate/episode_precommit_used", "precommit_expired/episode_deadline",
         "window_mismatch/episode_window", "window_mismatch/episode_window_sealed", "bad_prompt_idx/episode_unserved",
         "window_mismatch/episode_policy_stale", "bad_schema/service_contract", "out_of_zone/out_of_zone",
         "window_mismatch/window_mismatch", "bad_prompt_idx/bad_prompt_idx", "grail_fail/proof",
         "worker_dropped/code_grader_crash", "batch_filled/admission_queue", "sync:rate_limited",
         "sync:window_mismatch", "sync:bad_prompt_idx", "sync:batch_filled"]


def test_the_resent_stages_are_the_shared_retryable_set():
    from reliquary.protocol.episode_retry import RETRYABLE_STAGES

    assert RETRYABLE_STAGES <= {item.partition("/")[2] for item in RESENT}


@pytest.mark.parametrize("item", RESENT)
def test_a_resendable_refusal_sends_the_group_again(tmp_path, item):
    # pending=1: right after the resend the route still shows the first refusal; it must not be read as
    # the second send's verdict.
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=(item, "accepted"), pending=1)
    response = w.mine()
    assert response.accepted, response
    assert len(w.submitted) == 2 and w.runner.submitted == list(range(M_ROLLOUTS))
    backoff = 9.0 if item == "sync:window_not_active" else 5.0
    assert backoff in w.slept


@pytest.mark.parametrize("item", FINAL)
def test_a_final_refusal_is_never_resent_and_withdraws_everything(tmp_path, item):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=(item, "accepted"))
    response = w.mine()
    assert not response.accepted and len(w.submitted) == 1
    assert w.runner.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_resending_stops_at_the_grading_deadline(tmp_path):
    clock = Clock()
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), verdicts=("worker_dropped/episode_checker_busy",),
              submit_by=NOW + 12, clock=clock, proof_budget_s=0.0)
    response = w.mine()
    assert not response.accepted and response.stage == "episode_checker_busy"
    assert len(w.submitted) == 2 and w.slept == [5.0, 10.0]     # at NOW, NOW + 5; NOW + 15 is past it
    assert w.runner.submitted == [] and sorted(w.runner.released) == list(range(2 * M_ROLLOUTS))


def test_a_verdict_that_never_comes_is_waited_for_until_the_grading_deadline_and_the_group_left_held(tmp_path):
    from reliquary.miner.episode_group_miner import VERDICT_POLL_MAX_S
    from reliquary.miner.signed_episode import SUBMIT_TRANSIT_S

    clock = Clock()
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), pending=10 ** 6, submit_by=NOW + 3000, clock=clock,
              proof_budget_s=0.0)
    response = w.mine(open_until=NOW + 200)
    assert not response.accepted and response.reason == "submitted"
    assert len(w.submitted) == 1 and clock.now == NOW + 3000 + SUBMIT_TRANSIT_S     # submit_by, not the window end
    assert max(w.slept) == VERDICT_POLL_MAX_S and w.slept[:3] == [2.0, 3.0, 4.5]
    # The poll timed out: the chosen group is neither submitted nor withdrawn (it lapses on its own);
    # only the episodes not sent are withdrawn.
    assert w.runner.submitted == [] and sorted(w.runner.released) == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))


def test_a_group_graded_past_the_window_end_plus_transit_still_gets_its_verdict(tmp_path):
    from reliquary.miner.signed_episode import SUBMIT_TRANSIT_S

    clock = Clock()
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), pending=25, submit_by=NOW + 3000, clock=clock,
              proof_budget_s=0.0)
    response = w.mine(open_until=NOW + 200)
    assert response.accepted, response
    assert clock.now > NOW + 200 + SUBMIT_TRANSIT_S                 # the verdict came after the window end + transit
    assert w.runner.submitted == list(range(M_ROLLOUTS))
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))


def test_a_group_cancelled_while_its_verdict_is_pending_is_not_withdrawn(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), pending=10 ** 6)

    async def cancelled(seconds):
        raise asyncio.CancelledError

    w.miner._sleep = cancelled
    with pytest.raises(asyncio.CancelledError):
        w.mine()
    assert len(w.submitted) == 1 and w.runner.submitted == []
    assert sorted(w.runner.released) == list(range(M_ROLLOUTS, 2 * M_ROLLOUTS))


def test_seeds_still_playing_are_cut_before_the_grading_deadline(tmp_path):
    import time

    slow = set(range(M_ROLLOUTS, 2 * M_ROLLOUTS))
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), slow=slow, submit_by=time.time() + 100.3,
              proof_budget_s=100.0)
    started = time.monotonic()
    response = w.mine()
    assert time.monotonic() - started < 10 and response.accepted
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == list(range(M_ROLLOUTS))
    assert sorted(w.engine.dropped) == sorted(f"trace-{s}" for s in slow)
    assert all(w.draws.get(f"trace-{s}") is None for s in slow)


def test_seeds_still_playing_are_cut_at_the_window_end(tmp_path):
    import time

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), slow={0})
    started = time.monotonic()
    assert w.mine(open_until=time.time() + 0.3) is None       # the window is over: nothing sent
    assert time.monotonic() - started < 10
    assert w.engine.dropped == ["trace-0"] and w.submitted == []
    assert sorted(w.runner.released) == list(range(1, 2 * M_ROLLOUTS))


def test_a_seed_aborted_by_its_machine_is_played_once_more(tmp_path):
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), aborted={0: 1, 1: 5})
    assert w.mine().accepted
    assert w.runner.runs.count(0) == 2 and w.runner.runs.count(1) == 2
    assert ("trace-0-1", 0) in w.draws.history and w.draws.get("trace-0-1") is None
    (request,) = w.submitted
    assert request.pool_selection["seeds"] == [0] + list(range(2, M_ROLLOUTS + 1))


def test_a_throttled_precommit_is_sent_again_until_the_window_end_never_at_once(tmp_path):
    from reliquary.miner.episode_group_miner import PRECOMMIT_ATTEMPTS, PRECOMMIT_MIN_BACKOFF_S

    busy = [SessionRefused("validator_busy", status=503) for _ in range(PRECOMMIT_ATTEMPTS + 2)]
    clock = Clock()
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), sessions=Sessions(refuse=list(busy)), clock=clock)
    assert w.mine(open_until=NOW + 3600).accepted
    assert len(w.miner._sessions.bodies) == PRECOMMIT_ATTEMPTS + 3
    assert w.slept[:PRECOMMIT_ATTEMPTS + 2] == [PRECOMMIT_MIN_BACKOFF_S] * (PRECOMMIT_ATTEMPTS + 2)
    bounded = world(tmp_path / "bounded", alternating(2 * M_ROLLOUTS), sessions=Sessions(refuse=list(busy) * 3),
                    clock=Clock())
    with pytest.raises(SessionRefused):
        bounded.mine(open_until=NOW + 3 * PRECOMMIT_MIN_BACKOFF_S + 1)
    assert len(bounded.miner._sessions.bodies) == 4 and bounded.runner.entered == []


def test_the_default_chooser_prefers_an_in_zone_mix_of_rewards(tmp_path):
    rewards = {seed: 0.0 if seed < M_ROLLOUTS else 1.0 for seed in range(2 * M_ROLLOUTS)}
    w = world(tmp_path, rewards)
    assert w.mine().accepted
    (request,) = w.submitted
    half = M_ROLLOUTS // 2
    assert request.pool_selection["seeds"] == list(range(half)) + list(range(2 * M_ROLLOUTS - half, 2 * M_ROLLOUTS))
    assert request.service_binding["purpose"] == "training"


def test_the_http_verdict_route_is_the_validators(tmp_path):
    import httpx

    from reliquary.miner.episode_group_miner import http_verdicts

    asked = []

    def handler(request):
        asked.append(request.url.path)
        if request.url.path.endswith("a" * 64):
            return httpx.Response(200, json={"status": "pending", "verdict": None})
        return httpx.Response(503)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fetch = http_verdicts("http://v", MINER.ss58_address, client=client)
            return await fetch(4, "a" * 64), await fetch(4, "b" * 64)

    assert asyncio.run(go()) == ({"status": "pending", "verdict": None}, None)
    assert asked[0] == f"/miner-verdicts/{MINER.ss58_address}/4/{'a' * 64}"


def test_the_single_turn_engine_leaves_signed_episode_environments_out(monkeypatch):
    from tests.unit.episode_v2_fixtures import register_episode_env
    from tests.unit.service_v2_fixtures import MATH

    register_episode_env(monkeypatch)
    math, episode = SimpleNamespace(name=MATH), SimpleNamespace(name=EPISODE)
    envs, mix = engine_module._single_turn_envs({MATH: math, EPISODE: episode}, [(MATH, 3), (EPISODE, 1)])
    assert envs == {MATH: math} and mix == [(MATH, 3)]
    with pytest.raises(ValueError, match="no single-turn environment"):
        engine_module._single_turn_envs({EPISODE: episode}, [(EPISODE, 1)])
    # The engine filters at construction; mining a window never meets a signed-episode env.
    init = inspect.getsource(engine_module.MiningEngine.__init__)
    assert init.count("_single_turn_envs(") == 2 and init.index("_single_turn_envs(") < init.index("self._cooldown_per_env")
    assert "_single_turn_mined_spec" not in inspect.getsource(engine_module)
