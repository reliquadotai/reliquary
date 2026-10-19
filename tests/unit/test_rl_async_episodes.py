"""Episodes (multi-turn, signed sandboxes) on the async RL path (rl-async-design.md §4.10): the submission wire,
the draw keyed per model-token position across turns (miner == validator == trainer), the validator's lease-bound
verification of the signed transcripts and the token structure (the window path's checks), the trainer's mask
(only model tokens get loss, substitution and match), the audit forward, the lease engagement of the sandbox
issuer and the miner's episode generator."""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.miner.forced_draw import LeaseDrawBinding
from reliquary.miner.rl_async import EpisodeRollout, HeldLease, encode_submission, submission_digest
from reliquary.miner.vllm_generation import FORCED_SEED_KEY, _forced_state, _uniform
from reliquary.protocol.rl_draw import lease_uniforms, model_positions
from reliquary.protocol.rl_submit import episode_spans, rollouts_digest
from reliquary.protocol.service_episode import EpisodeWireError, rl_lease_engagement
from reliquary.validator.rl_buffer import SubmittedGroup

NONCE = "cd" * 32
SPANS = ((0, 6), (9, 13), (16, 21))                 # three model turns, two observations (3 tokens each)
COMPLETION = 21


# --- wire --------------------------------------------------------------------------------------------------------
def episode(spans=SPANS, stop="agent_completed", transcript=None) -> dict:
    return {"spans": [list(s) for s in spans], "stop": stop, "transcript": transcript or {"token": {}, "records": []}}


def test_episode_spans_are_read_strictly():
    assert episode_spans(episode(), COMPLETION) == SPANS
    for bad in (((1, 6), (9, 21)),                     # the completion starts on a model turn
                ((0, 6), (9, 20)),                     # ...and ends on one
                ((0, 6), (5, 21)),                     # overlapping
                ((0, 6), (6, 6), (9, 21)),             # empty
                ((0, 6), (9, 22))):                    # outside
        with pytest.raises(ValueError):
            episode_spans(episode(bad), COMPLETION)
    with pytest.raises(ValueError):
        episode_spans({**episode(), "extra": 1}, COMPLETION)
    with pytest.raises(ValueError):
        episode_spans({**episode(), "spans": [[0, 6.0], [9, 21]]}, COMPLETION)


def test_the_digest_binds_the_episode_and_keeps_single_turn_digests():
    single = [(0, "AAAA", None), (1, "BBBB", (1, 2))]
    assert rollouts_digest(single) == rollouts_digest([(*r, None) for r in single])
    one = rollouts_digest([(0, "AAAA", None, episode())])
    assert one != rollouts_digest([(0, "AAAA", None)])
    assert one != rollouts_digest([(0, "AAAA", None, episode(stop="max_turns"))])
    assert one != rollouts_digest([(0, "AAAA", None, episode(transcript={"token": {"x": 1}, "records": []}))])


def test_the_lease_engagement_names_the_lease_and_the_rollout():
    assert rl_lease_engagement("ab12", 3) == "rl-lease:ab12:3"
    for lease_id, index in (("a:b", 0), ("", 0), ("a" * 65, 0), ("ab", -1), ("ab", 128), ("ab", True)):
        with pytest.raises(EpisodeWireError):
            rl_lease_engagement(lease_id, index)


def test_the_miner_signs_what_the_route_verifies():
    from reliquary.validator.rl_lease_routes import RlSubmitRequestBody, submitted_rollouts

    lease = HeldLease("L1", "env", 3, 2, NONCE, 1e12)
    rollouts = [EpisodeRollout(tuple(range(COMPLETION)), SPANS, "agent_completed", {"token": {"i": i}})
                for i in range(16)]
    payload = encode_submission(lease, rollouts)
    body = RlSubmitRequestBody(miner_hotkey="h", at=0, signature="", **payload)
    digest = rollouts_digest([(r.index, r.tokens, r.force_span, r.episode) for r in body.rollouts])
    assert digest == submission_digest(payload)
    group = submitted_rollouts(body, max_tokens=4096)
    assert [r.spans for r in group] == [SPANS] * 16 and group[3].episode["transcript"] == {"token": {"i": 3}}
    mixed = dict(payload, rollouts=[dict(payload["rollouts"][0], episode=None)] + payload["rollouts"][1:])
    with pytest.raises(ValueError):
        submitted_rollouts(RlSubmitRequestBody(miner_hotkey="h", at=0, signature="", **mixed), max_tokens=4096)


# --- the draw across turns ---------------------------------------------------------------------------------------
def test_miner_validator_and_trainer_key_an_episode_by_model_token_position():
    """Each turn's request starts its draw at the model tokens before it; the validator and the trainer read the
    same stream over the model spans only (observations take no position)."""
    from reliquary.validator.training import _forced_draw_uniforms

    group = SubmittedGroup("L1", "A", "env", 9, 4, NONCE, 0.0, (0.0,) * 2, (False,) * 2, (1, 2),
                           (tuple(range(COMPLETION)),) * 2, (None, None), (SPANS, SPANS))
    group = SubmittedGroup.unpack(group.pack())                       # as the buffer hands it over
    assert group.spans == (SPANS, SPANS)
    for index in range(2):
        binding, miner, before = LeaseDrawBinding(NONCE, 9, 4, index), [], 0
        for start, end in SPANS:                                       # what the forced engine sends per turn
            state = _forced_state(binding.extra_args(before)[FORCED_SEED_KEY])
            miner += [_uniform(state, state["offset"] + j) for j in range(end - start)]
            before += end - start
        validator = group.draw_uniforms(index)
        trainer = _forced_draw_uniforms(SimpleNamespace(_forced_draw_uniforms=validator), before)
        assert miner == validator == trainer == lease_uniforms(NONCE, 9, 4, index, 15)
    assert group.model_positions(0) == model_positions(COMPLETION, SPANS) == [*range(6), *range(9, 13),
                                                                              *range(16, 21)]
    with pytest.raises(ValueError):
        LeaseDrawBinding("XY", 9, 4, 0)


# --- trainer -----------------------------------------------------------------------------------------------------
def _episode_completion(model_pick, uniforms, observation, spans=SPANS, forge=False, gen=None):
    """A completion whose model turns follow ``model_pick(last token, u)`` and whose observations are ``observation``
    tokens (anything: the env's, never drawn)."""
    tokens, last, t = [], 3, 0
    for k, (start, end) in enumerate(spans):
        if k:
            gap = start - len(tokens)
            tokens += observation[:gap]
            last = tokens[-1]
        for _ in range(end - start):
            pick = int(torch.randint(0, 32, (1,), generator=gen)) if forge else model_pick(last, uniforms[t])
            tokens.append(pick)
            last, t = pick, t + 1
    return tuple(tokens)


def test_the_trainer_masks_observations_and_verifies_model_tokens(monkeypatch):
    from reliquary.trainer.async_trainer import DrawGate, training_group
    from reliquary.validator import training, verifier
    from tests.unit.test_substitution_loss import _GUMBEL, _VOCAB, _Model, _logits_at

    monkeypatch.setattr(training, "KL_BETA", 0.0)
    monkeypatch.setattr("reliquary.constants.TRAINING_LOSS", "substitution")
    monkeypatch.setattr(verifier, "draw_settings", lambda: _GUMBEL)
    training.reset_training_state()
    model, gen = _Model(), torch.Generator().manual_seed(1)

    def pick(last, u):
        return int(fs.pick_rows(_logits_at(model, last), [u], draw=fs.DRAW_GUMBEL_MAX)[0])

    spans = ((0, 30), (34, 64), (68, 98))
    n = 4
    base = SubmittedGroup("ep", "A", "env", 7, 3, NONCE, 0.0, (1.0, 0.0, 1.0, 0.0), (False,) * n, (1, 2, 3),
                          ((0,) * 98,) * n, (None,) * n, (spans,) * n)
    completions = []
    for i in range(n):
        observation = [int(t) for t in torch.randint(0, _VOCAB, (8,), generator=gen)]   # the env's text
        completions.append(_episode_completion(pick, base.draw_uniforms(i), observation, spans, forge=(i == 3),
                                               gen=gen))
    group = training_group(SubmittedGroup(**{**base.__dict__, "completions": tuple(completions)}), slot0=0, index=0)
    rollout = group.rollouts[0]
    assert rollout._validated_assistant_spans == tuple((3 + s, 3 + e) for s, e in spans)
    assert len(rollout._forced_draw_uniforms) == len(rollout.commit["rollout"]["token_logprobs"]) == 90
    assert training._policy_token_positions(rollout) == [3 + p for p in model_positions(98, spans)]
    gate = DrawGate([[0, 1, 2, 3]], group_floor=0.7)
    plan = [(group, [1.0, -1.0, 1.0, -1.0], 1.0 / (n * 90))]
    for p in model.parameters():
        p.grad = None
    training._accumulate_grouped_grads(model, None, plan, torch.device("cpu"), budget=10_000, atomic=False,
                                       kl_stats=training._new_kl_stats(), gate=gate)
    gate.judge(plan)
    # random observations cost nothing; the forged model turns of rollout 3 leave the loss and, under the forgery
    # floor, reject the group
    assert gate.admitted == [1, 1, 1, 0] and gate.accepted == [False] and gate.pooled(0) >= 0.7
    assert [gate.rate(s) for s in range(3)] == [1.0] * 3 and gate.rate(3) < 0.5
    assert [gate.tallies[s][2] for s in range(4)] == [90] * 4 and gate.tallies[0][3] == 90   # model tokens only
    training.reset_training_state()


# --- audit -------------------------------------------------------------------------------------------------------
def test_the_audit_judges_model_positions_only(monkeypatch):
    from reliquary.services.exploration import AUDIT_FAILURE_STATISTICAL
    from reliquary.validator import rl_audit, verifier
    from tests.unit.test_rl_audit import PROMPT, TinyLM, _transitions

    monkeypatch.setattr(verifier, "draw_settings", lambda: verifier.DrawSettings(
        draw=fs.DRAW_GUMBEL_MAX, t=1.0, top_k=0, top_p=1.0, stochastic_threshold=0.99, boundary_epsilon=0.05))
    monkeypatch.setattr(verifier, "lm_head_fp32", lambda: True)
    table = _transitions()

    def pick(last, u):
        return int(fs.pick_rows(table[last:last + 1], [u], draw=fs.DRAW_GUMBEL_MAX)[0])

    spans = ((0, 40), (44, 84), (88, 128))
    rng = random.Random(0)

    def group(nonce):
        base = SubmittedGroup("L", "A", "env", 9, 3, NONCE, 0.0, (0.0,) * 16, (False,) * 16, PROMPT,
                              ((0,) * 128,) * 16, (None,) * 16, (spans,) * 16)
        completions = []
        for i in range(16):
            uniforms = lease_uniforms(nonce, 9, 3, i, 120)
            completions.append(_episode_completion_from(pick, uniforms, spans, rng))
        return SubmittedGroup(**{**base.__dict__, "completions": tuple(completions)})

    honest = group(NONCE)
    assert rl_audit.audit_forward(TinyLM(table))(honest).passed is True
    diagnostics = rl_audit.rollout_draw_diagnostics(TinyLM(table), honest, 0)
    assert diagnostics.n_positions == 120                     # observations never judged
    as_single_turn = SubmittedGroup(**{**honest.__dict__, "spans": None})
    assert rl_audit.audit_forward(TinyLM(table))(as_single_turn).passed is False
    other = rl_audit.audit_forward(TinyLM(table))(group("ab" * 32))
    assert other.passed is False and other.failure_class == AUDIT_FAILURE_STATISTICAL


def _episode_completion_from(pick, uniforms, spans, rng):
    tokens, last, t = [], 20, 0
    for k, (start, end) in enumerate(spans):
        if k:
            tokens += [rng.randrange(32) for _ in range(start - len(tokens))]
            last = tokens[-1]
        for _ in range(end - start):
            last = pick(last, uniforms[t])
            tokens.append(last)
            t += 1
    return tuple(tokens)


# --- the validator: signed transcripts bound to the lease --------------------------------------------------------
LEASE_K = 3


def _lease(**kw):
    from tests.unit.episode_v2_fixtures import EPISODE, TASK

    values = dict(lease_id="lease1", hotkey="5Hot", environment=EPISODE, prompt_idx=TASK, checkpoint=LEASE_K,
                  nonce=NONCE, issued_at=0.0, expires_at=1e12, state="open")
    values.update(kw)
    return SimpleNamespace(**values)


def play_lease_episode(*, validator, machine, lease, index, reward, session_id=None, engagement=None,
                       output="ok", last=None, stop="agent_completed"):
    """One honest two-turn episode (``episode_v2_fixtures.play_episode``'s), its session token naming the lease's
    rollout: ``(rollout, prompt length)``."""
    from reliquary_sandbox.observation import render_observation

    from reliquary.corpus.signed_parse import signed_records
    from reliquary.validator.rl_submit import Rollout
    from tests.unit.episode_v2_fixtures import ENV_PACKAGE, FIRST_TURN_TEXT, PROMPT_TEXT, SPLIT
    from tests.unit.sandbox_fixtures import NOW, claims, transcript
    from tests.unit.test_trajectory_parse import CALL, TERM, TEXT, FakeRenderer

    renderer = FakeRenderer()
    prompt = renderer.initial_ids(PROMPT_TEXT)
    first = [TEXT] * FIRST_TURN_TEXT + [CALL, TERM]
    session = claims(session_id=session_id or f"s-{lease.lease_id}-{index}", hotkey=lease.hotkey,
                     engagement=engagement or rl_lease_engagement(lease.lease_id, index), split=SPLIT,
                     index=lease.prompt_idx, checkpoint=f"k{lease.checkpoint}", issued_at=NOW,
                     expires_at=NOW + 4500)
    signed = transcript(validator, machine, session, status="graded", reward=float(reward), env_package=ENV_PACKAGE,
                        calls=[{"turn": 0, "k": 0, "arguments": {"command": "c0"}, "output": output}])
    observations = [render_observation(body.to_dict()) for body in signed_records(signed).calls[:1]]
    full = renderer.next_prompt(prompt, first, observations)
    tokens = full + (list(last) if last is not None else [TEXT] * 9 + [TERM])
    p = len(prompt)
    spans = [[0, len(first)], [len(full) - p, len(tokens) - p]]
    return Rollout(tuple(tokens[p:]), None, {"spans": spans, "stop": stop, "transcript": signed}), p


@pytest.fixture
def world(tmp_path):
    attest = pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.validator.rl_episode import LeaseEpisodeChecker
    from tests.unit.episode_v2_fixtures import EPISODE, FixedSource, episode_contract, episode_signers
    from tests.unit.sandbox_fixtures import NOW, directory
    from tests.unit.test_trajectory_parse import FakeRenderer

    validator, machine = episode_signers(tmp_path / "keys")
    policy = episode_contract().episode_policy(EPISODE)
    checker = LeaseEpisodeChecker(policy=policy, renderer=FakeRenderer(), source=FixedSource(),
                                  directory=lambda now: directory(machine),
                                  token_verifier=attest.Ed25519TokenVerifier({validator.key_id:
                                                                              validator.public_key_b64}))

    def rollouts(lease, rewards=None, **kw):
        rewards = rewards or [1.0, 0.0] * 8
        return [play_lease_episode(validator=validator, machine=machine, lease=lease, index=i, reward=rewards[i],
                                   **kw)[0] for i in range(16)]

    return SimpleNamespace(checker=checker, rollouts=rollouts, received=NOW + 100, validator=validator,
                           machine=machine, policy=policy)


def test_honest_episodes_are_graded_by_their_final_records(world):
    lease = _lease()
    rollouts = world.rollouts(lease)
    graded = world.checker.grade(lease, rollouts, received=world.received)
    assert graded.reject is None and graded.rewards == (1.0, 0.0) * 8 and graded.truncated == (False,) * 16
    assert graded.spans == tuple(r.spans for r in rollouts)
    assert graded.prompt_tokens == world.checker.prompt_ids(lease.prompt_idx)


def test_a_transcript_of_another_rollout_or_lease_or_checkpoint_is_refused(world):
    lease = _lease()
    rollouts = world.rollouts(lease)
    swapped = [rollouts[1], rollouts[0], *rollouts[2:]]
    assert world.checker.grade(lease, swapped, received=world.received).reject == "episode_transcript"
    other = world.rollouts(_lease(lease_id="lease2"))
    assert world.checker.grade(lease, other, received=world.received).reject == "episode_transcript"
    newer = world.rollouts(_lease(checkpoint=LEASE_K + 1))
    assert world.checker.grade(lease, newer, received=world.received).reject == "episode_checkpoint"
    assert world.checker.grade(_lease(hotkey="5Other"), rollouts,
                               received=world.received).reject == "episode_transcript"


def test_tampered_observations_or_model_tokens_are_refused(world):
    from reliquary.validator.rl_submit import Rollout

    lease = _lease()
    rollouts = world.rollouts(lease)
    tokens = list(rollouts[4].tokens)
    observation = rollouts[4].spans[0][1] + 2                           # inside the first observation
    tokens[observation] += 1
    tampered = Rollout(tuple(tokens), None, rollouts[4].episode)
    graded = world.checker.grade(lease, [*rollouts[:4], tampered, *rollouts[5:]], received=world.received)
    assert graded.reject == "episode_parse"
    early = Rollout(rollouts[2].tokens, None, {**rollouts[2].episode, "stop": "max_turns"})
    graded = world.checker.grade(lease, [*rollouts[:2], early, *rollouts[3:]], received=world.received)
    assert graded.reject in ("episode_parse", "episode_termination")     # a limit stop the harness never hit
    late = world.checker.grade(lease, rollouts, received=world.received + 10**6)
    assert late.reject == "episode_deadline"


def test_a_last_turn_cut_at_the_per_turn_cap_is_admitted_for_a_lease(world):
    """A lease takes all its episodes: the draw that ran a turn to the per-turn cap ends the episode there."""
    from tests.unit.test_trajectory_parse import TEXT

    lease = _lease()
    cap = world.policy.max_tokens_per_turn
    rollouts = world.rollouts(lease, last=[TEXT] * cap, stop="context_length")
    graded = world.checker.grade(lease, rollouts, received=world.received)
    assert graded.reject is None and graded.rewards == (1.0, 0.0) * 8
    short = world.rollouts(lease, last=[TEXT] * (cap - 1), stop="context_length")      # no cap reached
    assert world.checker.grade(lease, short, received=world.received).reject is not None


def test_a_turn_that_would_not_export_still_trains_for_a_lease(world):
    """A lease's episode is never exported: a span that does not round-trip through the template (a turn cut in
    its reasoning, a stray reasoning close) is judged by its calls, records and draw, not refused."""
    world.checker.renderer.canonical = False
    world.checker.renderer.unclosed = True
    lease = _lease()
    graded = world.checker.grade(lease, world.rollouts(lease), received=world.received)
    assert graded.reject is None


def test_a_stale_directory_is_the_validators_outage(world):
    from reliquary.environment.grader_client import GraderInfrastructureError

    world.checker.directory = lambda now: None
    with pytest.raises(GraderInfrastructureError):
        world.checker.grade(_lease(), world.rollouts(_lease()), received=world.received)


def test_a_lease_of_episodes_is_submitted_buffered_and_reported_on_model_tokens(world, tmp_path, monkeypatch):
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseRequest
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions, SubmitConfig
    from tests.unit.episode_v2_fixtures import EPISODE, TASK, register_episode_env
    from tests.unit.sandbox_fixtures import NOW

    register_episode_env(monkeypatch)
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={EPISODE: 600.0}),
                     quota=QuotaController(QuotaConfig(initial=8, ceiling=8)), clock=lambda: float(NOW + 100))
    book.adopt_checkpoint(LEASE_K)
    granted = book.grant([LeaseRequest("5Hot", EPISODE, (TASK,), LEASE_K)])[0].leases[0]
    grader = GroupGrader({}, SimpleNamespace(), eos_token_ids=(1,), episodes={EPISODE: world.checker})
    subs = RlSubmissions(book, grader, config=SubmitConfig(), clock=lambda: float(NOW + 100),
                         rng=random.Random(0))
    rollouts = world.rollouts(_lease(lease_id=granted.lease_id, nonce=granted.nonce))
    answer = subs.submit(granted.lease_id, "5Hot", rollouts)
    assert answer["verdict"] == "in_zone" and answer["rewards"] == [1.0, 0.0] * 8
    (buffered,) = subs.buffer.take(4)
    assert buffered.spans == tuple(r.spans for r in rollouts)
    model = sum(e - s for s, e in rollouts[0].spans)
    assert subs.group(granted.lease_id)["completion_tokens"] == 16 * model < sum(len(r.tokens) for r in rollouts)
    with pytest.raises(ValueError):                                     # a count above the model tokens
        subs.report_training(granted.lease_id, rollout_match=[1.0] * 16, accepted=True,
                             matched_tokens=[model + 1] * 16)
    subs.report_training(granted.lease_id, rollout_match=[1.0] * 16, accepted=True, matched_tokens=[model] * 16)
    (trained,) = [e for e in subs.events() if e["kind"] == "trained"]
    assert trained["detail"]["completion_tokens"] == [model] * 16
    # a single-turn body for an episode lease is refused for integrity
    granted2 = book.grant([LeaseRequest("5Hot", EPISODE, (TASK + 1,), LEASE_K)])[0].leases[0]
    from reliquary.validator.rl_submit import Rollout

    answer = subs.submit(granted2.lease_id, "5Hot", [Rollout(r.tokens) for r in rollouts])
    assert answer["verdict"] == "rejected" and answer["reason"] == "episode_schema"
    book.close()


# --- the sandbox issuer's lease engagement -----------------------------------------------------------------------
def test_the_lease_engagement_grants_one_session_per_open_lease_rollout(tmp_path):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.sandbox.rl_engagements import EpisodeEnvironmentView, RlLeaseEngagements
    from reliquary.sandbox.sessions import EngagementTerms, Refusal
    from reliquary.sandbox.tasks import ResolvedTask
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseRequest
    from tests.unit.episode_v2_fixtures import EPISODE, episode_contract

    now = 1_800_000_000.0
    leases = LeaseBook(tmp_path / "l.sqlite3", LeaseBookConfig(expiry_s={EPISODE: 600.0}),
                       quota=QuotaController(QuotaConfig(initial=8, ceiling=8)), clock=lambda: now)
    leases.adopt_checkpoint(2)
    lease = leases.grant([LeaseRequest("5Hot", EPISODE, (11,), 2)])[0].leases[0]
    policy = episode_contract().episode_policy(EPISODE)

    async def resolve(index):
        return ResolvedTask("img@sha256:" + "a" * 64, {})

    held = set()
    sessions = SimpleNamespace(policy=SimpleNamespace(io_timeout_s=5.0, retry_after_s=1.0, open_window_s=600),
                               engagement_held=lambda name: name in held, open_refusal=lambda hotkey, at: None)
    book = RlLeaseEngagements(book=leases, sessions=sessions, clock=lambda: now,
                              environments=lambda env: EpisodeEnvironmentView(policy, resolve)
                              if env == EPISODE else None)

    def terms(hotkey, lease_id, index):
        return asyncio.run(book.terms(hotkey, {"kind": "rl_lease",
                                               "lease": {"lease_id": lease_id, "rollout_index": index}}))

    got = terms("5Hot", lease.lease_id, 5)
    assert isinstance(got, EngagementTerms) and got.exclusive
    assert (got.engagement, got.checkpoint, got.index, got.env) == (
        f"rl-lease:{lease.lease_id}:5", "k2", 11, policy.sandbox_env)
    assert got.still_valid() is None
    held.add(got.engagement)
    assert terms("5Hot", lease.lease_id, 5).reason == "engagement_taken"
    assert terms("5Other", lease.lease_id, 6).reason == "lease_not_yours"
    assert terms("5Hot", lease.lease_id, 16).reason == "seed_out_of_pool"
    assert terms("5Hot", "nope", 0).reason == "lease_closed"
    assert isinstance(asyncio.run(book.terms("5Hot", {"kind": "rl_lease", "lease": {"lease_id": lease.lease_id}})),
                      Refusal)
    leases.submit(lease.lease_id, "5Hot", in_zone=False, now=now)
    assert terms("5Hot", lease.lease_id, 7).reason == "lease_closed"
    assert got.still_valid().reason == "lease_closed"                   # checked again under the issuer lock
    leases.close()


# --- the miner's episode generator -------------------------------------------------------------------------------
class _Result:
    def __init__(self, index, ok=True, final_status=None):
        self.session_id, self.ok, self.final_status = f"s{index}", ok, final_status
        self.transcript = {"token": {"i": index}}
        self.stop, self.reward = "agent_completed", 1.0
        self.released = self.submitted_called = False

    async def release(self):
        self.released = True

    def submitted(self):
        self.submitted_called = True


class _Runner:
    def __init__(self, fail=None, abort_once=()):
        self.fail, self.abort_once, self.runs, self.results = fail, set(abort_once), [], []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, index, on_session=None):
        on_session(f"s{index}")
        self.runs.append(index)
        if index in self.abort_once:
            self.abort_once.discard(index)
            return _Result(index, ok=False, final_status="aborted")
        result = _Result(index)
        self.results.append(result)
        return result


def test_the_episode_generator_plays_every_rollout_bound_to_the_lease(monkeypatch):
    from reliquary.miner import episode_commit
    from reliquary.miner.forced_draw import ForcedDraws
    from reliquary.miner.rl_async_episodes import EpisodeLeaseFailed, EpisodeLeaseGenerator

    lease = HeldLease("lease9", "ep", 4, 2, NONCE, 1e12)
    bound = {}

    class Draws(ForcedDraws):
        def bind(self, session_id, binding):
            bound[session_id] = binding
            super().bind(session_id, binding)

    class Engine:
        def take_session(self, session_id):
            return SimpleNamespace(session_id=session_id)

    prompt = [7, 7, 7]

    def admissible(outcome, *, policy, renderer, prompt, lease=False):
        assert lease                                                     # the async lease rules
        if outcome.seed_index == 13 and policy.refuse:
            return None, "episode_parse: bad"
        tokens = [7, 7, 7] + [outcome.seed_index] * 4 + [99] * 2 + [5]
        return episode_commit.AdmissibleEpisode(tokens, [(3, 7), (9, 10)], episode_commit.AdmittedStop(
            "agent_completed")), None

    monkeypatch.setattr(episode_commit, "episode_admissible", admissible)
    runners = []

    def factory(policy, held, text):
        runners.append(_Runner(abort_once={2}))
        return runners[-1]

    policy = SimpleNamespace(environment="ep", refuse=False)
    generator = EpisodeLeaseGenerator(policies={"ep": policy}, runner_factory=factory, engine=Engine(),
                                      draws=Draws(), renderer_for=lambda p: SimpleNamespace(initial_ids=lambda t: prompt),
                                      task_prompt=lambda env, idx: "task")
    (group,) = asyncio.run(generator.agenerate([lease]))
    assert len(group) == 16 and all(isinstance(r, EpisodeRollout) for r in group)
    assert group[5].tokens == (5, 5, 5, 5, 99, 99, 5) and group[5].spans == ((0, 4), (6, 7))
    assert group[5].transcript == {"token": {"i": 5}}
    assert sorted(runners[0].runs).count(2) == 2                        # a machine-signed abort is played again
    assert bound["s3"] == LeaseDrawBinding(NONCE, 4, 2, 3)
    assert all(r.submitted_called for r in runners[0].results)
    payload = encode_submission(lease, group)
    assert payload["rollouts"][5]["episode"]["spans"] == [[0, 4], [6, 7]]
    policy.refuse = True                                                 # one rollout unusable: the lease fails
    (failed,) = asyncio.run(generator.agenerate([lease]))
    assert isinstance(failed, EpisodeLeaseFailed)
    assert all(r.released for r in runners[1].results)


# --- miner throughput: one vLLM call for a batch of leases -------------------------------------------------------
def test_a_batch_of_leases_is_one_engine_call_each_request_keyed_by_its_own_lease(monkeypatch):
    """The fake engine drives the runner-keyed processor with every request of the batch interleaved in its rows
    (as vLLM's persistent batch does): each request still draws its own lease's stream."""
    from reliquary.miner import vllm_generation
    from reliquary.miner.rl_async import ROLLOUTS, VllmGroupGenerator
    from tests.unit.test_vllm_forced_seed import _Engine

    vocab, steps = 64, 6
    table = torch.randn(vocab, vocab, generator=torch.Generator().manual_seed(3))
    monkeypatch.setattr(vllm_generation, "default_draw", lambda: fs.DRAW_GUMBEL_MAX)
    monkeypatch.setattr(vllm_generation, "T_PROTO", 1.0)
    monkeypatch.setattr(vllm_generation, "TOP_K_PROTO", 0)
    monkeypatch.setattr(vllm_generation, "TOP_P_PROTO", 1.0)

    class LLM:
        def __init__(self):
            self.calls = []

        def generate(self, requests, sampling, use_tqdm=False):
            self.calls.append(len(requests))
            engine = _Engine(monkeypatch, capacity=len(requests))
            order = list(range(len(requests)))
            random.Random(len(requests)).shuffle(order)               # rows in no particular order
            for row, request in enumerate(order):
                engine.add(f"req{request}", row, sampling[request])
            outputs = [[] for _ in requests]
            last = [requests[r]["prompt_token_ids"][-1] for r in order]
            for _ in range(steps):
                picks = engine.step(table[torch.tensor(last)])
                for row, request in enumerate(order):
                    outputs[request].append(int(picks[row]))
                last = [int(p) for p in picks]
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=o)]) for o in outputs]

    llm = LLM()
    generator = VllmGroupGenerator.__new__(VllmGroupGenerator)
    generator._engine = vllm_generation.VLLMRolloutGenerator("m", engine=llm, sampling_params_class=SimpleNamespace)
    generator.eos_ids = [10**6]
    generator._prompt = lambda env, idx: [idx % vocab, (idx * 3) % vocab]
    monkeypatch.setattr("reliquary.constants.max_new_tokens_for_environment", lambda env: steps)
    leases = [HeldLease(f"L{i}", "math", 10 + i, 2, f"{i:02x}" * 32, 1e12) for i in range(3)]
    groups = generator.generate(leases)
    assert llm.calls == [3 * ROLLOUTS]                                  # one call for the whole batch
    for lease, group in zip(leases, groups):
        assert len(group) == ROLLOUTS
        for index, completion in enumerate(group):
            uniforms, last, want = lease_uniforms(lease.nonce, lease.prompt_idx, 2, index, steps), \
                generator._prompt("math", lease.prompt_idx)[-1], []
            for u in uniforms:
                last = int(fs.pick_rows(table[last:last + 1], [u], draw=fs.DRAW_GUMBEL_MAX)[0])
                want.append(last)
            assert completion == want
