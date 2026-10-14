"""envscaler_tools_v2 as a corpus suite-episode source, end to end.

`jobs create` on its graded SFT range, a miner's episode solving one real
scenario, the intake's replay in the spawned replay pool (2 GiB RLIMIT_AS,
30 s deadline), the local grader, and `jobs export --sft`, whose tokens must
be the chat template's rendering.

The miner here is the RL episode runner over a scripted policy, which is what
the suite miner (`reliquary.miner.suite_episode`, on its own branch) drives
its engine through; the trajectory it submits is cut the same way.

Needs the fetched corpus (RELIQUARY_ENVSCALER_DATA, see
`scripts/fetch_envscaler_corpus.py`) and a Qwen3.5 tokenizer; skipped without
either.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import time

import pytest

from reliquary.corpus.job import parse_job
from reliquary.corpus.suite_episode import (
    TokenizerCodec,
    episode_seed,
    replay_stop,
    suite_span_refusal,
)
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID, ChatMLEpisodeRenderer
from reliquary.environment.agentic.envs.envscaler_tools_v2 import environment as v2
from reliquary.environment.agentic.envs.envscaler_tools_v2.corpus import DATA_ENV_VAR
from reliquary.environment.agentic.runner import EpisodeRunner
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, span_chunk_count
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)

ENV = "envscaler_tools_v2"
CHUNK = 32
PROOF = "A" * 200
QWEN35_TOKENIZER = Path(os.environ.get(
    "RELIQUARY_TEST_QWEN35_TOKENIZER", "/home/ubuntu/bench-rl-throughput/dryrun/tiny-qwen35"))

# env_170_rl-task_1, in the corpus range (and at 1310 in the graded SFT
# range, which is in reserve): switch a patient from Amlodipine to Lisinopril
# and correct one drug-interaction record. Seven checks, all false at reset;
# the script below makes every one true.
SCENARIO_INDEX = 7231
SCENARIO_TASK = "env_170_rl-task_1"


def _call(tool: str, **arguments) -> AssistantAction:
    return AssistantAction(kind="tool", tool=tool, arguments=arguments)


SOLUTION = [
    _call("update_user_medication_status", user_id="USR5", medication_id="MEDC3",
          active_status=True),
    _call("update_user_medication_status", user_id="USR5", medication_id="MEDF6",
          active_status=False),
    _call("list_active_user_medications", user_id="USR5"),
    _call("get_drug_interactions_all_pairs", medication_ids=["MEDC3", "MEDE5"]),
    _call("remove_drug_interaction", medication_id_1="MEDE5", medication_id_2="MEDF6"),
    _call("add_drug_interaction", medication_id_1="MEDE5", medication_id_2="MEDF6",
          interaction_type="No clinically significant interaction",
          interaction_description=(
              "No significant interaction is known; concomitant use is generally safe."),
          severity="low"),
    AssistantAction.final("USR5 now takes Lisinopril; the interaction record is corrected."),
]


@pytest.fixture(scope="module")
def setting():
    if not os.environ.get(DATA_ENV_VAR):
        pytest.skip(f"{DATA_ENV_VAR} not set: the EnvScaler corpus is fetched, not vendored")
    if not QWEN35_TOKENIZER.is_dir():
        pytest.skip("no Qwen3.5 tokenizer on this host")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN35_TOKENIZER))
    environment = ENVIRONMENT_SPECS[ENV].create()
    assert environment.get_task(SCENARIO_INDEX).id == SCENARIO_TASK
    return tokenizer, TokenizerCodec(tokenizer), environment


def _episode(**overrides) -> dict:
    episode = {"env": {"package": "reliquary-suite", "name": ENV,
                       "version": ENVIRONMENT_SPECS[ENV].environment_manifest_sha256,
                       "split": "train"},
               "harness": "suite", "renderer": CHATML_RENDERER_ID, "max_turns": 16,
               "max_tokens_per_turn": 3072, "max_total_tokens": 60000}
    episode.update(overrides)
    return episode


def _declare(prompt_start: int, prompt_count: int):
    from reliquary.cli.main import prepare_corpus_job

    return prepare_corpus_job(
        job_id="envscaler-v2-sft", task_id=None, model="Org/Model", model_revision="r" * 40,
        model_architecture="qwen3_5", checkpoint_sha256="c" * 64, from_profile=None,
        prompt_encoding=None, prompt_source=ENV, prompt_count=prompt_count,
        prompt_start=prompt_start, renderer_id=CHATML_RENDERER_ID, eos_token_id=248046,
        slots_per_prompt=2, max_new_tokens=3072, cap=0.05, min_incentive_share=0.5,
        audit_params={}, episode=copy.deepcopy(_episode()))


class _ScriptedModel:
    """The policy a miner's engine stands for: each turn as a ChatML-trained
    model writes it, stopped on its own <|im_end|>, tokenized by the miner."""

    def __init__(self, actions, codec) -> None:
        self._actions, self._codec, self._turn = iter(actions), codec, 0

    def generate(self, **_):
        text = ChatMLEpisodeRenderer.action_text(
            next(self._actions), reasoning=f"Step {self._turn}: act on what I know.",
        ) + "<|im_end|>"
        self._turn += 1
        return GeneratedAction(text=text, tokens=tuple(self._codec.encode(text)))


def _mine(environment, job, codec, actions):
    """What a suite miner submits for the job's one prompt."""
    index = job.prompt_start
    task = environment.get_task(index)
    renderer = ChatMLEpisodeRenderer(codec.encode)
    trace = EpisodeRunner(
        renderer=renderer, max_turns=min(job.episode.max_turns, environment.max_turns),
    ).run(environment, task, seed=episode_seed(job.job_id, index),
          policy=_ScriptedModel(actions, codec))
    offset = len(renderer.encode_initial(task))
    tokens = list(trace.tokens[offset:trace.assistant_spans[-1][1]])
    spans = [(s - offset, e - offset) for s, e in trace.assistant_spans]
    assert suite_span_refusal(tokens, spans, encode=codec.encode, decode=codec.decode) is None
    request = CorpusSubmissionRequest(
        job_id=job.job_id, miner_hotkey="5Hot", cursor=0, prompt_index=index,
        checkpoint_sha256="c" * 64, rendered_prompt=renderer.initial_text(task),
        trajectory={"tokens": tokens, "turns": [
            {"start": s, "end": e,
             "proofs": [PROOF] * span_chunk_count(e - s, CHUNK, MIN_CHUNK_TOKENS)}
            for s, e in spans], "final_diff": "", "stop": replay_stop(trace)},
        signature="ok")
    return request, trace


class _Records:
    """The route's submissions, passed audits, and the grades written."""

    def __init__(self):
        self.written, self.verdicts, self.grades = {}, {}, {}

    async def write_submission(self, job_id, submission_id, record):
        self.written[submission_id] = record
        self.verdicts[submission_id] = {"passed": True}
        return True

    async def read_submission(self, job_id, sid):
        return self.written.get(sid)

    async def list_submission_ids(self, job_id):
        return list(self.written)

    async def list_verdict_ids(self, job_id):
        return list(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def list_voided_ids(self, job_id):
        return []

    async def read_regrade(self, job_id, sid):
        return None

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def write_grade(self, job_id, sid, document):
        self.grades[sid] = document

    async def list_grade_ids(self, job_id):
        return list(self.grades)


def test_a_job_is_declared_on_the_corpus_range_only(setting) -> None:
    start, end = v2.SFT_CORPUS_RANGE
    manifest, entry = _declare(start, end - start)
    job = parse_job(manifest)

    assert job.episode.is_suite
    contract = entry.contract["environments"][ENV]
    assert contract["episode"]["renderer_id"] == CHATML_RENDERER_ID
    for prompt_start, count in ((0, 10), (start - 1, 2), (v2.SFT_GRADED_START, 1),
                                (v2.SFT_UNGRADED_START, 1)):
        with pytest.raises(ValueError, match="corpus ranges"):
            _declare(prompt_start, count)


def test_a_mined_envscaler_episode_is_accepted_graded_and_exported(setting, fake_r2) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.corpus.delivery import episode_rows
    from reliquary.corpus.suite_episode import suite_source
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.validator.corpus_service import build_corpus_router
    from reliquary.validator.suite_grading import LocalEpisodeGrader
    from reliquary.validator.suite_intake import PoolSuiteReplayer, SuiteEpisodeIntake
    from tests.unit.test_corpus_service import _CountingStore

    tokenizer, codec, environment = setting

    # 1. `jobs create`: one prompt, the scenario the script solves.
    manifest, _ = _declare(SCENARIO_INDEX, 1)
    job = parse_job(manifest)
    asyncio.run(job_store.write_job(manifest, None, **fake_r2))
    source = suite_source(job.episode)

    # 2. The miner's episode: the world runs in its own subprocess here too.
    request, mined = _mine(environment, job, codec, SOLUTION)
    assert mined.reward.success is True and mined.reward.reward == 1.0, mined.reward
    assert request.trajectory.stop == "agent_completed"

    # 3. The route, replaying in the spawned pool, as a serving validator does:
    # a worker under RLIMIT_AS 2 GiB that loads the corpus and starts the
    # world's subprocess, inside the worker-side deadline.
    replayer = PoolSuiteReplayer.for_tokenizer(tokenizer, workers=1)
    try:
        records, accepted = _Records(), []
        app = FastAPI()
        app.include_router(build_corpus_router(
            job_id=job.job_id, store=_CountingStore(fake_r2), tokenizer=tokenizer,
            renderer=None, verify_signature=lambda r: True, records=records,
            on_accepted=accepted.append, proof_chunk_tokens=CHUNK,
            episode_intake=SuiteEpisodeIntake(
                job=job, source=source, tokenizer=tokenizer, vocab_size=None,
                chunk_tokens=CHUNK, replayer=replayer)))
        started = time.monotonic()
        body = TestClient(app).post("/corpus/submit", json=request.model_dump()).json()
        first_replay = time.monotonic() - started
        assert body["accepted"] is True, body
        # The first replay in a fresh worker loads the corpus: well inside 30 s.
        assert first_replay < 15.0, first_replay
        (submission_id,) = records.written

        # 4. The local grader replays it again, in the same pool.
        grade = asyncio.run(LocalEpisodeGrader(
            job=job, records=records, replayer=replayer).grade_one(submission_id))
        assert grade["status"] == "ok" and grade["graded_success"] is True, grade
        assert grade["replay_certified"] is True and grade["instance_id"] == SCENARIO_TASK
    finally:
        replayer.close()

    # 5. `jobs export --sft`: the row is the chat template's transcript.
    async def export():
        counts: dict = {}
        rows = [row async for row in episode_rows(
            job=job, records=records, renderer=None, source=source, counts=counts,
            sft_only=True, quarantined=[], codec=codec)]
        return rows, counts

    (row,), counts = asyncio.run(export())
    assert counts["sft_rows"] == 1, counts
    messages, tools = json.loads(row["messages"]), json.loads(row["tools"])
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    assert row["tokens"] == codec.encode(rendered)
    assert row["graded_success"] is True and row["replay_certified"] is True
    assert sum(row["assistant_mask"]) == sum(e - s for s, e in (
        (t.start, t.end) for t in request.trajectory.turns))
    # Typed arguments survive the round trip: a boolean, a list.
    calls = [m["tool_calls"][0]["function"] for m in messages if m.get("tool_calls")]
    assert calls[0]["arguments"]["active_status"] is True
    assert calls[3]["arguments"]["medication_ids"] == ["MEDC3", "MEDE5"]


def test_an_unsolved_episode_is_accepted_and_graded_as_a_failure(setting) -> None:
    from reliquary.corpus.suite_episode import suite_source
    from reliquary.validator.agentic_intake import IntakeFacts
    from reliquary.validator.suite_intake import InlineSuiteReplayer, SuiteEpisodeIntake

    tokenizer, codec, environment = setting
    job = parse_job(_declare(SCENARIO_INDEX, 1)[0])
    request, mined = _mine(environment, job, codec, SOLUTION[:2] + SOLUTION[-1:])
    assert mined.reward.reward == 0.0
    intake = SuiteEpisodeIntake(
        job=job, source=suite_source(job.episode), tokenizer=tokenizer, vocab_size=None,
        chunk_tokens=CHUNK, replayer=InlineSuiteReplayer(encode=codec.encode,
                                                         decode=codec.decode))

    facts = intake.check(request)

    assert isinstance(facts, IntakeFacts), facts
    assert facts.success is False and facts.reward == 0.0


def test_a_forged_observation_is_refused_by_the_pool_replay(setting) -> None:
    from reliquary.corpus.suite_episode import suite_source
    from reliquary.validator.suite_intake import PoolSuiteReplayer, SuiteEpisodeIntake

    tokenizer, codec, environment = setting
    job = parse_job(_declare(SCENARIO_INDEX, 1)[0])
    request, _ = _mine(environment, job, codec, SOLUTION)
    trajectory = request.trajectory.model_dump()
    first_turn_end = trajectory["turns"][0]["end"]
    trajectory["tokens"][first_turn_end + 12] = codec.encode("X")[0]
    forged = CorpusSubmissionRequest.model_validate(
        {**request.model_dump(), "trajectory": trajectory})
    replayer = PoolSuiteReplayer.for_tokenizer(tokenizer, workers=1)
    try:
        intake = SuiteEpisodeIntake(job=job, source=suite_source(job.episode),
                                    tokenizer=tokenizer, vocab_size=None, chunk_tokens=CHUNK,
                                    replayer=replayer)
        refusal = intake.check(forged)
    finally:
        replayer.close()

    assert refusal.reason == "bad_observation", refusal


def test_a_replay_past_its_deadline_kills_its_world(setting) -> None:
    """The worker's SIGALRM lands while the replay waits on the world's pipe:
    the deadline is the worker's, never an observation, and the world process
    does not outlive the replay."""
    from reliquary.validator import suite_intake

    tokenizer, codec, environment = setting
    job = parse_job(_declare(SCENARIO_INDEX, 1)[0])
    request, _ = _mine(environment, job, codec, SOLUTION)
    kwargs = suite_intake.replay_kwargs(
        job, prompt_index=SCENARIO_INDEX,
        prompt_ids=ChatMLEpisodeRenderer(codec.encode).encode_initial(
            environment.get_task(SCENARIO_INDEX)),
        tokens=request.trajectory.tokens,
        spans=[(t.start, t.end) for t in request.trajectory.turns],
        stop=request.trajectory.stop)
    suite_intake._WORKER_TOKENIZER = tokenizer.backend_tokenizer
    before = _world_processes()
    try:
        outcome = suite_intake.replay_in_worker(ENV, 0.05, **kwargs)
    finally:
        suite_intake._WORKER_TOKENIZER = None

    assert outcome.deadline is True
    assert _world_processes() <= before


def _world_processes() -> set[int]:
    from reliquary.environment.agentic.envs.envscaler_tools_v2.sandbox import WORKER_PATH

    found = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if str(WORKER_PATH).encode() in command:
            found.add(int(entry.name))
    return found
