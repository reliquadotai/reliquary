"""Forced-draw suite episodes (an order on a suite set): the miner draws every turn of an episode on one
binding, submits however the episode ends (a ``context_length`` cut included), and the validator's intake and
grader check a cut as the RL lease path does."""
from __future__ import annotations

import asyncio
import copy

import pytest

from reliquary.corpus.job import parse_job
from reliquary.miner.agentic_miner import Identity, harness_key, mine_agentic
from reliquary.miner.corpus_generate_server import Finished, GenerateEngine
from reliquary.miner.forced_draw import ForcedDraws
from reliquary.miner.suite_episode import SuiteEpisodeRunner
from tests.unit.test_suite_episode_miner import (
    CODEC,
    GOLDEN_INDEX,
    Client,
    _call,
    _golden,
    _manifest,
    _telecom_or_skip,
    _text,
    _Records,
)
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)


def _drawn_manifest(**episode):
    raw = _manifest(episode=episode, prompt_start=GOLDEN_INDEX, prompt_count=1, seed=11)
    raw["sampling"] = {**raw["sampling"], "draw": "gumbel_max"}
    return raw


def _drawn_job(**episode):
    return parse_job(_drawn_manifest(**episode))


class ForcedScriptedCore:
    """The scripted core of a forced engine: records each turn's draw, returns no proof."""

    def __init__(self, texts, codec=CODEC):
        self._script = [list(codec.encode(t)) for t in texts]
        self._done = list(codec.encode(_call("done")))
        self._queue = []
        self.draws = []

    def add(self, request_id, prompt_ids, max_tokens, draw=None):
        assert draw is not None, "a forced engine sends every turn with its draw"
        from reliquary.miner.vllm_generation import FORCED_SEED_KEY

        self.draws.append(dict(draw[FORCED_SEED_KEY]))
        self._queue.append((request_id, max_tokens))

    def has_unfinished(self):
        return bool(self._queue)

    def step(self):
        out = []
        for request_id, cap in self._queue:
            tokens = tuple((self._script.pop(0) if self._script else self._done)[:cap])
            out.append(Finished(request_id, tokens, tuple(-0.1 for _ in tokens), "stop", ()))
        self._queue = []
        return out


def _mine_drawn(texts, job, *, client=None):
    core = ForcedScriptedCore(texts)
    engine = GenerateEngine(core, max_total_tokens=job.episode.max_total_tokens,
                            max_tokens_per_turn=job.episode.max_tokens_per_turn, draws=ForcedDraws())
    engine.start()
    try:
        runner = SuiteEpisodeRunner(job=job, engine=engine, encode=CODEC.encode, decode=CODEC.decode,
                                    chunk_tokens=32)
        client = client or Client()
        counts = asyncio.run(mine_agentic(
            job=job, identities=[Identity("5Hot", sign=lambda body: "sig", episodes=1)],
            client=client, engine=engine, runners={harness_key(None): runner},
            decode=CODEC.decode, concurrency=1, stop_ids=runner.stop_ids))
    finally:
        engine.stop()
    return counts["5Hot"], client.bodies, core, runner


def _cut_budget(job, texts):
    """A budget with room for the first turn, not for the observation after it."""
    probe = SuiteEpisodeRunner(job=job, engine=None, encode=CODEC.encode, decode=CODEC.decode)
    return len(probe.initial_ids(GOLDEN_INDEX)) + len(CODEC.encode(texts[0])) + 5


def test_a_drawn_suite_job_parses():
    job = _drawn_job()
    assert job.episode.is_suite and job.sampling.draw == "gumbel_max" and job.seed == 11
    assert parse_job(job.to_contract()) == job


def test_every_turn_of_a_drawn_episode_follows_its_slots_draw():
    from reliquary.corpus.eval_draw import draw_context

    _telecom_or_skip()
    actions, _ = _golden()
    texts = [_text(action, turn=turn) for turn, action in enumerate(actions)]
    job = _drawn_job(max_turns=40)
    counts, bodies, core, _ = _mine_drawn(texts, job)
    assert counts["accepted"] == 1, counts
    (body,) = bodies
    trajectory = body["trajectory"]
    assert trajectory["stop"] == "agent_completed"
    assert all(turn["proofs"] == [] for turn in trajectory["turns"])
    # One binding for the whole episode: the slot's stream, each turn starting at the model tokens before it.
    slot = body["cursor"]
    expected = draw_context(job, GOLDEN_INDEX, slot)
    offsets, done = [], 0
    for turn in trajectory["turns"]:
        offsets.append(done)
        done += turn["end"] - turn["start"]
    assert len(core.draws) == len(trajectory["turns"])
    for draw, offset in zip(core.draws, offsets):
        assert {k: draw[k] for k in ("randomness", "prompt_idx", "checkpoint_hash", "rollout_index")} == {
            k: expected[k] for k in ("randomness", "prompt_idx", "checkpoint_hash", "rollout_index")}
        assert draw["base_offset"] == offset


def test_a_drawn_episode_with_no_room_left_is_submitted_as_a_cut():
    _telecom_or_skip()
    texts = [_call("check_status_bar"), _call("done", turn=1)]
    job = _drawn_job()
    job = _drawn_job(max_total_tokens=_cut_budget(job, texts))
    counts, bodies, _, runner = _mine_drawn(texts, job)
    assert counts["accepted"] == 1, counts
    (body,) = bodies
    assert body["trajectory"]["stop"] == "context_length"
    assert len(body["trajectory"]["turns"]) == 1


def test_a_cut_claimed_with_room_left_fails_the_precheck():
    from reliquary.corpus.trajectory import BuiltTrajectory

    _telecom_or_skip()
    texts = [_call("check_status_bar"), _call("done", turn=1)]
    job = _drawn_job()
    _, (body,), _, runner = _mine_drawn(texts, job)
    trajectory = body["trajectory"]
    built = BuiltTrajectory(
        prompt_ids=tuple(runner.initial_ids(GOLDEN_INDEX)), tokens=tuple(trajectory["tokens"]),
        spans=tuple((t["start"], t["end"]) for t in trajectory["turns"]),
        proofs=tuple(() for _ in trajectory["turns"]), final_diff="", stop="agent_completed")
    assert runner.precheck(built, GOLDEN_INDEX) is None
    # Cut after the first turn although the budget had room: refused, as the validator would.
    first = trajectory["turns"][0]
    early = BuiltTrajectory(prompt_ids=built.prompt_ids, tokens=built.tokens[:first["end"]],
                            spans=built.spans[:1], proofs=((),), final_diff="", stop="context_length")
    reason, detail = runner.precheck(early, GOLDEN_INDEX)
    assert reason == "bad_stop" and "room" in detail["why"], detail


def test_replay_kwargs_name_the_budget_only_for_a_drawn_job():
    from reliquary.corpus.suite_episode import suite_source
    from reliquary.validator.suite_intake import job_suite_stops, replay_kwargs

    _telecom_or_skip()
    drawn = _drawn_job()
    free = parse_job(_manifest(prompt_start=GOLDEN_INDEX, prompt_count=1))
    for job, budget in ((drawn, drawn.episode.max_total_tokens), (free, None)):
        kwargs = replay_kwargs(job, suite_source(job.episode), prompt_index=GOLDEN_INDEX, prompt_ids=[1],
                               tokens=[2], spans=[(0, 1)], stop="agent_completed")
        assert kwargs.get("max_episode_tokens") == budget
    assert "context_length" in job_suite_stops(drawn)
    assert "context_length" not in job_suite_stops(free)


def _route(job, manifest, fake_r2, *, records):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.corpus.suite_episode import suite_source
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.validator.corpus_service import build_corpus_router
    from reliquary.validator.suite_intake import InlineSuiteReplayer, SuiteEpisodeIntake
    from tests.unit.test_corpus_service import _CountingStore
    from tests.unit.test_suite_episode_miner import ByteTokenizer

    asyncio.run(job_store.write_job(manifest, None, **fake_r2))
    source = suite_source(job.episode)
    replayer = InlineSuiteReplayer(encode=CODEC.encode, decode=CODEC.decode)
    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id=job.job_id, store=_CountingStore(fake_r2), tokenizer=ByteTokenizer(), renderer=None,
        verify_signature=lambda request: True, records=records, on_accepted=lambda sid: None,
        proof_chunk_tokens=None, job=job, episode_intake=SuiteEpisodeIntake(
            job=job, source=source, tokenizer=ByteTokenizer(), vocab_size=None, chunk_tokens=32,
            replayer=replayer)))
    return TestClient(app), source, replayer


@pytest.mark.parametrize("cut", [False, True])
def test_a_drawn_episode_is_accepted_and_graded_by_the_validator(fake_r2, cut):
    from reliquary.validator.suite_grading import LocalEpisodeGrader

    _telecom_or_skip()
    actions, golden = _golden()
    texts = [_text(action, turn=turn) for turn, action in enumerate(actions)]
    episode = {"max_turns": 40}
    if cut:
        episode["max_total_tokens"] = _cut_budget(_drawn_job(**episode), texts)
    manifest = _drawn_manifest(**copy.deepcopy(episode))
    job = parse_job(manifest)
    records = _Records()
    http, source, replayer = _route(job, manifest, fake_r2, records=records)
    counts, bodies, _, _ = _mine_drawn(texts, job, client=Client(
        lambda body: http.post("/corpus/submit", json=body).json()))
    assert counts["accepted"] == 1, counts
    (body,) = bodies
    (submission_id,) = records.written
    grade = asyncio.run(LocalEpisodeGrader(job=job, records=records, replayer=replayer,
                                           source=source).grade_one(submission_id))
    assert grade["status"] == "ok" and grade["replay_certified"] is True, grade
    if cut:
        assert body["trajectory"]["stop"] == "context_length"
        assert grade["graded_success"] is False and grade["grade"]["reward"] == 0.0
        assert grade["replay"]["termination_reason"] == "context_length"
    else:
        assert grade["graded_success"] is True and grade["instance_id"] == golden["task_id"]


def test_a_free_suite_job_still_refuses_a_cut(fake_r2):
    from reliquary.validator.suite_intake import SuiteEpisodeIntake

    _telecom_or_skip()
    free = parse_job(_manifest(prompt_start=GOLDEN_INDEX, prompt_count=1))
    intake = SuiteEpisodeIntake.__new__(SuiteEpisodeIntake)
    intake._job = free
    trajectory = type("T", (), {"transcript": None, "final_diff": "", "stop": "context_length"})()
    request = type("R", (), {"trajectory": trajectory, "completions": [], "prompt_index": GOLDEN_INDEX})()
    refusal = SuiteEpisodeIntake.check(intake, request)
    assert refusal.reason == "bad_stop", refusal
