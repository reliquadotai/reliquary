"""Only verified work leaves; the filter annotates, it never pays."""

import asyncio
import json

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.corpus.export import export_rows


class _Records:
    def __init__(self):
        self.subs = {
            "1" * 64: {"hotkey": "A", "prompt_index": 3, "rendered_prompt": "q3",
                       "completions": [{"text": "yes", "tokens": [1]}, {"text": "no", "tokens": [2]}]},
            "2" * 64: {"hotkey": "B", "prompt_index": 4, "rendered_prompt": "q4",
                       "completions": [{"text": "x", "tokens": [3]}]},
        }
        self.verdicts = {"1" * 64: {"passed": True}, "2" * 64: {"passed": False}}

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_submission(self, job_id, sid):
        return self.subs[sid]


class _Job:
    job_id = "math-v1"


def _collect(**kw):
    async def go():
        return [row async for row in export_rows(job=_Job(), records=_Records(), **kw)]
    return asyncio.run(go())


def test_only_passed_submissions_are_exported_one_row_per_completion():
    rows = _collect()
    assert [(r["prompt"], r["completion"], r["hotkey"]) for r in rows] == [("q3", "yes", "A"), ("q3", "no", "A")]


def test_the_filter_annotates_every_row():
    rows = _collect(grade=lambda prompt_index, text: (text == "yes", 1.0 if text == "yes" else 0.0))
    assert [(r["completion"], r["accepted"], r["score"]) for r in rows] == [("yes", True, 1.0), ("no", False, 0.0)]


def test_a_passed_verdict_with_no_record_is_skipped_with_a_warning(caplog):
    """R2 is not transactional: a verdict can be visible before its submission
    object is. Exporting pays nothing, so skipping that row is safe -- the
    alternative is a bare crash that blanks the whole run over one bad row."""
    import logging

    class _RecordsWithAGap(_Records):
        async def read_submission(self, job_id, sid):
            if sid == "1" * 64:
                return None
            return await super().read_submission(job_id, sid)

    records = _RecordsWithAGap()
    records.verdicts = {"1" * 64: {"passed": True}, "2" * 64: {"passed": True}}

    async def go():
        with caplog.at_level(logging.WARNING, logger="reliquary.corpus.export"):
            return [row async for row in export_rows(job=_Job(), records=records)]

    rows = asyncio.run(go())

    assert [(r["prompt"], r["completion"]) for r in rows] == [("q4", "x")]
    assert any("1" * 12 in message for message in caplog.messages)


# --- `jobs export`: the CLI wiring around `export_rows`. A fake job store, a
# fake record store, and a fake environment spec so the filter path never
# touches a real bucket or a real dataset. ---


class _FakeRecords:
    def __init__(self):
        self.subs: dict = {}
        self.verdicts: dict = {}

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_submission(self, job_id, sid):
        return self.subs[sid]


class _FakeEnvironment:
    """`compute_reward` accepts iff the text is `"yes"`, so a threshold of 0.5
    splits rows deterministically without any real grader."""

    name = "fake-env"

    def get_problem(self, index):
        return {"id": index}

    def compute_reward(self, problem, completion):
        return 1.0 if completion == "yes" else 0.0


class _FakeSpec:
    interaction_mode = "single_turn"

    def create(self):
        return _FakeEnvironment()


class _FakeEpisodeSpec:
    interaction_mode = "episode"

    def create(self):  # pragma: no cover - must never run
        raise AssertionError("an episode-mode source must never be built to grade text")


def _job_spec(*, job_id="math-v1", prompt_source="fake-env", filter_=None, prompt_count=10):
    from reliquary.corpus.job import JobSpec, Sampling

    return JobSpec(
        job_id=job_id,
        checkpoint_repo="org/Frozen",
        checkpoint_revision="abc123",
        checkpoint_sha256="a" * 64,
        prompt_source=prompt_source,
        prompt_count=prompt_count,
        renderer_id="reliquary-external-prompt-v1",
        eos_token_id=151645,
        sampling=Sampling(
            temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64, n=1
        ),
        slots_per_prompt=1,
        filter=filter_,
        prompt_order="free",
        deadline_round=None,
    )


@pytest.fixture
def jobs(monkeypatch):
    """The job store, as a dict the test can seed; `jobs export` only reads."""
    from reliquary.infrastructure import corpus_job_store as job_store

    stored: dict = {}

    async def _read_job(job_id, **kwargs):
        job = stored.get(job_id)
        return (job, '"etag"') if job is not None else (None, None)

    monkeypatch.setattr(job_store, "read_job", _read_job)
    return stored


@pytest.fixture
def records(monkeypatch):
    from reliquary.infrastructure import corpus_record_store

    fake = _FakeRecords()
    monkeypatch.setattr(corpus_record_store, "BucketRecordStore", lambda **kw: fake)
    return fake


@pytest.fixture
def env_specs(monkeypatch):
    from reliquary.environment import registry

    # `corpus_service` binds its own `ENVIRONMENT_SPECS` name at MODULE import
    # time (`from ... import ENVIRONMENT_SPECS`). Importing it here, before the
    # patch below, makes sure that binding captures the real catalog rather
    # than -- if this were this process's first import of `corpus_service`,
    # e.g. triggered lazily by `jobs export` itself -- freezing onto our fake
    # one for the rest of the test session, past this fixture's teardown.
    import reliquary.validator.corpus_service  # noqa: F401

    specs = {"fake-env": _FakeSpec(), "fake-episode-env": _FakeEpisodeSpec()}
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", specs)
    return specs


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jobs_export_writes_one_row_per_completion_of_passed_submissions(
    jobs, records, tmp_path
):
    jobs["math-v1"] = _job_spec()
    records.subs = {
        "1" * 64: {"hotkey": "A", "prompt_index": 3, "rendered_prompt": "q3",
                   "completions": [{"text": "yes", "tokens": [1]}, {"text": "no", "tokens": [2]}]},
        "2" * 64: {"hotkey": "B", "prompt_index": 4, "rendered_prompt": "q4",
                   "completions": [{"text": "x", "tokens": [3]}]},
    }
    records.verdicts = {"1" * 64: {"passed": True}, "2" * 64: {"passed": False}}
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(app, ["jobs", "export", "math-v1", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert "2 rows written" in result.output
    rows = _read_lines(out)
    assert [(r["prompt"], r["completion"], r["hotkey"]) for r in rows] == [
        ("q3", "yes", "A"), ("q3", "no", "A")
    ]
    assert "accepted" not in rows[0]


def test_jobs_export_with_apply_filter_annotates_every_row(jobs, records, env_specs, tmp_path):
    from reliquary.corpus.job import Filter

    jobs["math-v1"] = _job_spec(filter_=Filter(grader_id="fake-grader", threshold=0.5))
    records.subs = {
        "1" * 64: {"hotkey": "A", "prompt_index": 0, "rendered_prompt": "q0",
                   "completions": [{"text": "yes", "tokens": [1]}, {"text": "no", "tokens": [2]}]},
    }
    records.verdicts = {"1" * 64: {"passed": True}}
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(
        app, ["jobs", "export", "math-v1", "--out", str(out), "--apply-filter"]
    )

    assert result.exit_code == 0, result.output
    rows = _read_lines(out)
    assert [(r["completion"], r["accepted"], r["score"]) for r in rows] == [
        ("yes", True, 1.0), ("no", False, 0.0)
    ]


def test_jobs_export_only_accepted_drops_the_rejected_rows(jobs, records, env_specs, tmp_path):
    from reliquary.corpus.job import Filter

    jobs["math-v1"] = _job_spec(filter_=Filter(grader_id="fake-grader", threshold=0.5))
    records.subs = {
        "1" * 64: {"hotkey": "A", "prompt_index": 0, "rendered_prompt": "q0",
                   "completions": [{"text": "yes", "tokens": [1]}, {"text": "no", "tokens": [2]}]},
    }
    records.verdicts = {"1" * 64: {"passed": True}}
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(
        app,
        ["jobs", "export", "math-v1", "--out", str(out), "--apply-filter", "--only-accepted"],
    )

    assert result.exit_code == 0, result.output
    assert "1 rows written" in result.output
    rows = _read_lines(out)
    assert [r["completion"] for r in rows] == ["yes"]


def test_jobs_export_apply_filter_without_a_job_filter_is_refused(jobs, records, tmp_path):
    jobs["math-v1"] = _job_spec(filter_=None)
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(
        app, ["jobs", "export", "math-v1", "--out", str(out), "--apply-filter"]
    )

    assert result.exit_code != 0
    assert "filter" in result.output
    assert not out.exists()


def test_jobs_export_apply_filter_on_an_episode_source_is_refused(jobs, records, env_specs, tmp_path):
    """Episode-mode environments have no single-completion `compute_reward`
    path; grading one anyway would score the wrong thing silently."""
    from reliquary.corpus.job import Filter

    jobs["math-v1"] = _job_spec(
        prompt_source="fake-episode-env", filter_=Filter(grader_id="fake-grader", threshold=0.5)
    )
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(
        app, ["jobs", "export", "math-v1", "--out", str(out), "--apply-filter"]
    )

    assert result.exit_code != 0
    assert "episode" in result.output.lower()
    assert not out.exists()


def test_jobs_export_refuses_an_unknown_job(jobs, records, tmp_path):
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(app, ["jobs", "export", "ghost", "--out", str(out)])

    assert result.exit_code != 0
    assert "ghost" in result.output
    assert not out.exists()


def test_jobs_export_mid_stream_failure_leaves_no_file_or_temp_file_behind(
    jobs, records, tmp_path
):
    """A truncated, valid-looking dataset at `--out` is worse than no file at
    all: a downstream trainer cannot tell it apart from a completed export."""
    jobs["math-v1"] = _job_spec()
    records.subs = {
        "1" * 64: {"hotkey": "A", "prompt_index": 3, "rendered_prompt": "q3",
                   "completions": [{"text": "yes", "tokens": [1]}]},
        "2" * 64: {"hotkey": "B", "prompt_index": 4, "rendered_prompt": "q4",
                   "completions": [{"text": "x", "tokens": [3]}]},
    }
    records.verdicts = {"1" * 64: {"passed": True}, "2" * 64: {"passed": True}}
    real_read_submission = records.read_submission

    async def _raising(job_id, sid):
        if sid == "2" * 64:
            raise RuntimeError("record store exploded mid-stream")
        return await real_read_submission(job_id, sid)

    records.read_submission = _raising
    out = tmp_path / "dataset.jsonl"

    result = CliRunner().invoke(app, ["jobs", "export", "math-v1", "--out", str(out)])

    assert result.exit_code != 0
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
