"""Suite episode jobs through the row map: an eval job's row ``i`` plays the
task its set's ``i``-th reference names, at the set's split, on every side
(the miner, the validator's intake, the replay grader, the export). A
training job's rows stay the environment's train tasks, by index.

The environment is a fake episode environment whose task ``i`` has id
``t-{i}``, behind a fake registry spec that records the splits it builds.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from reliquary.corpus import suite_episode
from reliquary.corpus.job import parse_job
from reliquary.corpus.suite_episode import TokenizerCodec, episode_seed, job_suite_source
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID
from reliquary.environment.agentic.types import EpisodeTask
from reliquary.eval import prompt_source as ps
from reliquary.eval import rows as eval_rows
from reliquary.eval.prompt_source import eval_source_for
from tests.unit.test_corpus_suite_episode import ByteTokenizer, _GradeRecords

ENV = "fake_suite_v1"
ENV_SHA = "a" * 64
SPLIT = "benchmark"
CODEC = TokenizerCodec(ByteTokenizer())


class FakeEpisodeEnv:
    def __init__(self, split: str, ids: dict[int, str]) -> None:
        self.name, self.split, self.max_turns, self._ids = ENV, split, 8, ids

    def __len__(self) -> int:
        return 100

    def get_task(self, index: int) -> EpisodeTask:
        return EpisodeTask(id=self._ids.get(index, f"t-{index}"),
                           prompt=f"Task {index}: say done.", tools=())


class FakeSpec:
    name = ENV
    interaction_mode = "episode"
    environment_manifest_sha256 = ENV_SHA
    renderer_id = CHATML_RENDERER_ID
    external_distribution = None

    def __init__(self, ids=None) -> None:
        self.ids = dict(ids or {})
        self.builds: list[str] = []

    def create(self, split: str = "train") -> FakeEpisodeEnv:
        self.builds.append(split)
        return FakeEpisodeEnv(split, self.ids)


@pytest.fixture
def spec(monkeypatch):
    """The fake environment as the only registry entry; no eval rows kept."""
    fake = FakeSpec()
    monkeypatch.setattr(suite_episode, "_registry", lambda given: given or {ENV: fake})
    monkeypatch.setattr(ps, "_loaded", {})
    eval_rows._row_map.cache_clear()
    yield fake
    eval_rows._row_map.cache_clear()


def _set_rows(count=3):
    return [{"problem_id": f"p{i}", "set_id": "fake-bench", "env": ENV, "interaction": "episode",
             "split": SPLIT, "source_index": 42 + i, "instance_id": f"t-{42 + i}"}
            for i in range(count)]


def _body(rows):
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _eval_source(monkeypatch, rows=None) -> str:
    body = _body(rows or _set_rows())
    monkeypatch.setattr(ps, "FETCHERS", [lambda source: body])
    return eval_source_for("fake-bench", body, 3).name


def _manifest(prompt_source=ENV, split="train", prompt_count=50, **overrides):
    raw = {"schema": "reliquary/corpus-job/v1", "job_id": "suite-eval-rows",
           "checkpoint_repo": "Org/Model", "checkpoint_revision": "rev1",
           "checkpoint_sha256": "c" * 64, "prompt_source": prompt_source,
           "prompt_count": prompt_count, "renderer_id": CHATML_RENDERER_ID, "eos_token_id": 257,
           "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                        "max_new_tokens": 3072, "n": 1},
           "slots_per_prompt": 2, "filter": None, "prompt_order": "free",
           "deadline_round": None,
           "episode": {"env": {"package": "reliquary-suite", "name": ENV, "version": ENV_SHA,
                               "split": split},
                       "harness": "suite", "renderer": CHATML_RENDERER_ID, "max_turns": 8,
                       "max_tokens_per_turn": 3072, "max_total_tokens": 60000}}
    raw.update(overrides)
    return raw


def _eval_job(monkeypatch):
    return parse_job(_manifest(prompt_source=_eval_source(monkeypatch), split=SPLIT,
                               prompt_count=3))


def _kwargs(job, source, prompt_index):
    from reliquary.validator.suite_intake import replay_kwargs

    return replay_kwargs(job, source, prompt_index=prompt_index, prompt_ids=[1, 2], tokens=[3],
                         spans=[(0, 1)], stop="agent_completed")


def test_eval_suite_source_maps_rows(spec, monkeypatch):
    source = job_suite_source(_eval_job(monkeypatch))
    assert source.task(0).id == "t-42" and source.instance_id(2) == "t-44"
    assert len(source) == 3 and source.rows.split == SPLIT
    assert "Task 43" in source.initial_text(1)
    # One environment per source, built at the set's split.
    assert spec.builds == [SPLIT]
    with pytest.raises(IndexError):
        source.task(3)


def test_replay_kwargs_maps_task_index_and_split(spec, monkeypatch):
    job = _eval_job(monkeypatch)
    kwargs = _kwargs(job, job_suite_source(job), 0)
    assert kwargs["task_index"] == 42 and kwargs["split"] == SPLIT
    assert kwargs["seed"] == episode_seed(job.job_id, 0)


def test_non_eval_suite_job_unchanged(spec):
    job = parse_job(_manifest(prompt_start=10, prompt_count=5))
    source = job_suite_source(job)
    environment = FakeEpisodeEnv("train", {})
    # The whole train environment, by index, as before.
    assert len(source) == len(environment) and spec.builds == ["train"]
    for i in (0, 12, 99):
        assert source.task(i) == environment.get_task(i)
    assert source.instance_id(12) == "t-12"
    assert _kwargs(job, source, 12) == {
        "renderer_id": CHATML_RENDERER_ID, "max_turns": 8, "task_index": 12,
        "seed": episode_seed(job.job_id, 12), "prompt_ids": [1, 2], "tokens": [3],
        "spans": [(0, 1)], "stop": "agent_completed", "split": "train"}
    assert suite_episode.suite_source(job.episode).task(7).id == "t-7"


def test_suite_source_refuses_stale_instance_ids(spec, monkeypatch):
    spec.ids[42] = "other"
    with pytest.raises(ValueError, match="row 0"):
        job_suite_source(_eval_job(monkeypatch))


def test_resolve_suite_accepts_eval_source(spec, monkeypatch):
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError, SuiteSourceSpec, _resolve_suite,
    )

    job = _eval_job(monkeypatch)
    resolved = _resolve_suite(job.prompt_source, CHATML_RENDERER_ID, job.episode)
    assert isinstance(resolved, SuiteSourceSpec)
    source = resolved.create(job)
    assert len(source) == 3 and source.task(1).id == "t-43" and spec.builds == [SPLIT]
    train = parse_job(_manifest())
    assert len(_resolve_suite(ENV, None, train.episode).create(train)) == 100
    with pytest.raises(CorpusPromptSourceError, match="draws its prompts"):
        _resolve_suite("reliquary_retrieval_tools_v1", None, train.episode)


def test_build_suite_intake_serves_the_eval_rows(spec, monkeypatch):
    from reliquary.validator.suite_intake import build_suite_intake

    intake = build_suite_intake(_eval_job(monkeypatch), tokenizer=ByteTokenizer(),
                                vocab_size=None, chunk_tokens=32, replayer=None)
    assert intake.source.task(0).id == "t-42" and len(intake.source) == 3


class _RecordingReplayer:
    workers = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def areplay(self, env_name, **kwargs):
        from reliquary.corpus.suite_episode import SuiteReplayOutcome

        self.calls.append((env_name, kwargs))
        return SuiteReplayOutcome(None, task_id="t-42", reward=1.0, success=True,
                                  termination_reason="final", trace_digest="d" * 64)


def _record(job, prompt_index=0):
    return {"schema": "reliquary/corpus-submission-record/v2", "submission_id": "s1",
            "job_id": job.job_id, "hotkey": "5Hot", "cursor": 0, "prompt_index": prompt_index,
            "rendered_prompt": "", "received_at": 1.0, "token_count": 1,
            "completions": [{"prompt_tokens": [1, 2], "tokens": [3, 4],
                             "turns": [{"start": 0, "end": 2}], "stop": "agent_completed"}]}


def test_suite_grader_replays_mapped_task(spec, monkeypatch):
    from reliquary.validator.suite_grading import LocalEpisodeGrader

    job = _eval_job(monkeypatch)
    replayer = _RecordingReplayer()
    grader = LocalEpisodeGrader(job=job, records=_GradeRecords({"s1": _record(job)}),
                                replayer=replayer, source=job_suite_source(job))
    document = asyncio.run(grader.grade_one("s1"))
    (env_name, kwargs), = replayer.calls
    assert env_name == ENV and kwargs["task_index"] == 42 and kwargs["split"] == SPLIT
    assert kwargs["seed"] == episode_seed(job.job_id, 0)
    assert document["status"] == "ok" and document["instance_id"] == "t-42"


def test_replay_builds_the_environment_at_its_split(spec, monkeypatch):
    from reliquary.environment import registry
    from reliquary.validator import suite_intake

    built = []
    monkeypatch.setattr(registry, "get_environment_spec", lambda name: spec)
    monkeypatch.setattr(suite_intake, "replay_outcome",
                        lambda environment, **kw: built.append((environment.split, kw)))
    suite_intake._replay(ENV, CODEC.encode, CODEC.decode, split=SPLIT, task_index=42)
    suite_intake._replay(ENV, CODEC.encode, CODEC.decode, task_index=7)
    assert [split for split, _ in built] == [SPLIT, "train"]
    assert all("split" not in kw for _, kw in built)


class _Replayed(Exception):
    pass


def test_export_replays_mapped_task(spec, monkeypatch):
    from reliquary.corpus.delivery import suite_episode_row

    job = _eval_job(monkeypatch)
    source = job_suite_source(job)
    seen = {}

    def replay_suite(environment, **kwargs):
        seen.update(kwargs, split=environment.split)
        raise _Replayed

    monkeypatch.setattr(suite_episode, "replay_suite", replay_suite)
    record = _record(job)
    record["completions"][0]["prompt_tokens"] = list(
        source.renderer(CODEC.encode).encode_initial(source.task(0)))
    with pytest.raises(_Replayed):
        suite_episode_row(job=job, submission_id="s1", record=record,
                          grade={"instance_id": "t-42"}, source=source, encode=CODEC.encode,
                          decode=CODEC.decode)
    assert seen["task_index"] == 42 and seen["split"] == SPLIT
    assert seen["seed"] == episode_seed(job.job_id, 0)
