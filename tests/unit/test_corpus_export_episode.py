"""Export v2: one row per certified trajectory, the SFT set is the certified successes."""

import asyncio
import json

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from reliquary.corpus.delivery import (
    EPISODE_ROW_FIELDS,
    LocalDirectorySink,
    episode_rows,
    export_delivery,
)
from reliquary.corpus.job import parse_job
from reliquary.environment.agentic_swe import BASH_SYSTEM_PROMPT, SweSource
from tests.unit.test_corpus_grading import _record
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_trajectory_parse import FakeRenderer


class RoundTripRenderer(FakeRenderer):
    """The fake renderer, able to render a whole conversation back to ids: an
    assistant message carries its span's ids (the fake parser keeps nothing
    else), and tool messages are bridged as the turn renderer bridges them."""

    def assistant_message(self, completion_ids):
        return {**super().assistant_message(completion_ids),
                "content": json.dumps([int(t) for t in completion_ids])}

    def render_messages(self, messages):
        ids = self.initial_ids(messages[1]["content"])
        k = 2
        while k < len(messages):
            completion = json.loads(messages[k]["content"])
            k += 1
            observations = []
            while k < len(messages) and messages[k]["role"] == "tool":
                observations.append(messages[k]["content"])
                k += 1
            ids = (self.next_prompt(ids, completion, observations) if observations
                   else ids + completion)
        return ids

    def whitespace_free(self, ids):
        return tuple(ids)


R = RoundTripRenderer()

JOB = parse_job(_manifest(prompt_count=3))
SOURCE = SweSource([("i0", "p"), ("repo__x.1", "fix it"), ("i2", "q")])
# The validator's own render of the task's prompt (what intake stores).
PROMPT = R.initial_ids(SOURCE.prompt(1))
NAMES = ("certified", "failed", "uncertified", "disputed", "voided", "ungraded", "audit", "held")
IDS = {name: f"{k:x}" * 64 for k, name in enumerate(NAMES, start=1)}
OK = {"instance_id": "repo__x.1", "status": "ok", "graded_by": ["e1"],
      "replay": {"status": "ok", "certified": True, "graded_by": ["e1", "e2"]}}


def _episode_record():
    record = _record()
    record["completions"][0]["prompt_tokens"] = list(PROMPT)
    return record


class _Records:
    def __init__(self):
        self.verdicts = {sid: {"passed": name != "audit"} for name, sid in IDS.items()}
        self.submissions = {sid: _episode_record() for sid in IDS.values()}
        self.grades = {
            IDS["certified"]: {**OK, "graded_success": True, "replay_certified": True},
            # A failing grade drawn for replay and certified: real observations, tests failed.
            IDS["failed"]: {**OK, "graded_success": False, "replay_certified": True},
            IDS["uncertified"]: {**OK, "graded_success": True, "replay_certified": False,
                                 "replay": {"status": "ok", "certified": False}},
            IDS["disputed"]: {**OK, "graded_success": True, "replay_certified": False,
                              "replay": {"status": "disputed", "certified": False}},
            IDS["voided"]: {**OK, "graded_success": True, "replay_certified": True},
            IDS["held"]: {**OK, "graded_by": ["q1"], "graded_success": True,
                          "replay_certified": True},
        }
        self.regrades = {}
        self.voided = {IDS["voided"]: {"reason": "replay_failed"}}

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_submission(self, job_id, sid):
        return self.submissions.get(sid)

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def read_regrade(self, job_id, sid):
        return self.regrades.get(sid)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_voided(self, job_id, sid):
        return self.voided.get(sid)


async def _rows(records, *, sft_only, quarantined=("q1",), counts=None):
    counts = {} if counts is None else counts
    return [row async for row in episode_rows(job=JOB, records=records, renderer=R, source=SOURCE,
                                              counts=counts, sft_only=sft_only,
                                              quarantined=quarantined)]


def _collect(sft_only, records=None):
    counts = {}
    rows = asyncio.run(_rows(records or _Records(), sft_only=sft_only, counts=counts))
    return rows, counts


def test_rows_hold_messages_tokens_and_the_assistant_mask():
    rows, counts = _collect(False)
    assert sorted(r["submission_id"] for r in rows) == sorted([IDS["certified"], IDS["failed"]])
    assert (counts["voided"], counts["ungraded"], counts["held"], counts["uncertified"]) == (1, 1, 1, 2)
    row = next(r for r in rows if r["submission_id"] == IDS["certified"])
    assert set(row) == set(EPISODE_ROW_FIELDS)
    messages = json.loads(row["messages"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == BASH_SYSTEM_PROMPT
    assert messages[1]["content"] == SOURCE.prompt(1)
    assert messages[3] == {"role": "tool", "tool_call_id": "call_0_0", "content": "a.py"}
    assert row["tokens"][:len(PROMPT)] == PROMPT
    assert len(row["assistant_mask"]) == len(row["tokens"])
    spans = json.loads(row["turns"])
    assert sum(row["assistant_mask"]) == sum(e - s for s, e in spans)
    assert all(row["assistant_mask"][len(PROMPT) + s:len(PROMPT) + e] == [1] * (e - s)
               for s, e in spans)
    assert row["task_id"] == "repo__x.1" and row["stop"] == "agent_completed"
    assert row["graded_success"] is True and row["replay_certified"] is True


def test_the_sft_set_is_the_certified_successes():
    rows, counts = _collect(True)
    assert [r["submission_id"] for r in rows] == [IDS["certified"]]
    assert counts["sft_rows"] == 1 and counts["rows"] == 1


def test_a_regrade_supersedes_the_grade():
    records = _Records()
    records.regrades[IDS["certified"]] = {**records.grades[IDS["certified"]],
                                          "replay_certified": False}
    assert asyncio.run(_rows(records, sft_only=True)) == []


def test_a_regrade_releases_what_a_quarantined_executor_decided_alone():
    records = _Records()
    records.regrades[IDS["held"]] = {**records.grades[IDS["held"]], "graded_by": ["e3"],
                                     "generation": 1, "regraded_for": ["q1"]}
    ids = [r["submission_id"] for r in asyncio.run(_rows(records, sft_only=True))]
    assert sorted(ids) == sorted([IDS["certified"], IDS["held"]])


def test_a_replay_decided_alone_by_a_quarantined_executor_is_held():
    records = _Records()
    records.grades[IDS["certified"]]["replay"] = {"status": "ok", "certified": True,
                                                  "graded_by": ["q1"]}
    assert asyncio.run(_rows(records, sft_only=True)) == []


def test_a_void_read_directly_is_never_exported():
    records = _Records()
    # Voided after the listing: the per-row read still sees it.
    listed = dict(records.voided)
    records.voided[IDS["certified"]] = {"reason": "replay_failed"}

    async def list_voided_ids(job_id):
        return sorted(listed)

    records.list_voided_ids = list_voided_ids
    assert asyncio.run(_rows(records, sft_only=True)) == []


def test_a_record_whose_prompt_is_not_the_tasks_is_not_exported():
    records = _Records()
    records.submissions[IDS["certified"]]["completions"][0]["prompt_tokens"] = R.initial_ids("other")
    counts = {}
    assert asyncio.run(_rows(records, sft_only=True, counts=counts)) == []
    assert counts["prompt_mismatch"] == 1


def test_the_quarantined_executors_must_be_named():
    with pytest.raises(ValueError, match="quarantined"):
        asyncio.run(_rows(_Records(), sft_only=True, quarantined=None))


def test_parquet_delivery_carries_the_episode_columns(tmp_path):
    sink = LocalDirectorySink(tmp_path / "bucket")
    manifest = asyncio.run(export_delivery(job=JOB, records=_Records(), sink=sink, delivery_id="d1",
                                           renderer=R, source=SOURCE, sft_only=True,
                                           quarantined=("q1",), work_dir=tmp_path / "work"))
    assert manifest["columns"] == list(EPISODE_ROW_FIELDS) and manifest["rows"] == 1
    table = pq.read_table(tmp_path / "bucket" / manifest["shards"][0]["key"])
    assert table.column("replay_certified").to_pylist() == [True]
    report = json.loads((tmp_path / "bucket" / manifest["report"]).read_text())
    assert report["filter"]["applied"] is True


def test_an_episode_delivery_needs_its_renderer_source_and_quarantines(tmp_path):
    sink = LocalDirectorySink(tmp_path / "bucket")
    with pytest.raises(ValueError):
        asyncio.run(export_delivery(job=JOB, records=_Records(), sink=sink, delivery_id="d1",
                                    work_dir=tmp_path / "work"))


# -- the CLI ------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    from reliquary.cli import main as cli_main
    from reliquary.environment import agentic_swe
    from reliquary.infrastructure import corpus_executor_store, corpus_job_store
    from reliquary.infrastructure import corpus_record_store

    records = _Records()

    async def read_job(job_id, **kw):
        return (JOB, '"e"') if job_id == JOB.job_id else (None, None)

    async def list_executors(**kw):
        return [{"executor_id": "q1", "scope": "grade", "status": "quarantined"},
                {"executor_id": "e1", "scope": "grade", "status": "active"}]

    monkeypatch.setattr(corpus_job_store, "read_job", read_job)
    monkeypatch.setattr(corpus_record_store, "BucketRecordStore", lambda **kw: records)
    monkeypatch.setattr(corpus_executor_store, "list_executors", list_executors)
    monkeypatch.setattr(agentic_swe, "load_turn_renderer", lambda directory: R)
    monkeypatch.setattr(agentic_swe, "load_swe_source", lambda num_images: SOURCE)
    monkeypatch.setattr(cli_main, "_episode_tokenizer_dir", lambda job: "/nowhere")
    _drained(monkeypatch, True)
    return cli_main.app


def test_cli_exports_the_sft_set_as_json_lines(cli, tmp_path):
    out = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out), "--sft"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["submission_id"] for r in rows] == [IDS["certified"]]
    assert rows[0]["messages"][0]["role"] == "system" and isinstance(rows[0]["turns"], list)


def test_cli_without_sft_exports_every_certified_row(cli, tmp_path):
    out = tmp_path / "all.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert len(out.read_text().splitlines()) == 2


def test_cli_refuses_the_text_filter_on_an_episode_job(cli, tmp_path):
    out = tmp_path / "x.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out),
                                      "--apply-filter"])
    assert result.exit_code != 0 and not out.exists()


# -- the admin route ------------------------------------------------------------


def test_admin_refuses_an_episode_delivery(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
    from reliquary.admin.service import create_admin_app
    from reliquary.infrastructure import corpus_job_store

    job = parse_job(_manifest(prompt_count=3, job_id="math-swe"))

    async def read_job(job_id, **kw):
        return (job, '"e"') if job_id == "math-swe" else (None, None)

    monkeypatch.setattr(corpus_job_store, "read_job", read_job)
    app = create_admin_app(secret=b"s", pool_max=0.3, models={}, records=_Records(),
                           task_prefix="math-", deliveries=LocalDirectorySink(tmp_path / "p"),
                           current_round=lambda: 1, work_dir=tmp_path / "w")
    import time

    path, body = "/admin/v1/jobs/math-swe/deliveries", b"{}"
    stamp, nonce = str(int(time.time())), "ab" * 16
    headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
               SIGNATURE_HEADER: sign_request(b"s", stamp, nonce, "POST", path, body),
               "content-type": "application/json"}
    with TestClient(app) as client:
        response = client.request("POST", path, content=body, headers=headers)
    assert response.status_code == 422 and "--sft" in response.json()["detail"]


# -- fix round 1: the messages are proven, the task cross-checked, the export says what it left out


class _DropsAToken(RoundTripRenderer):
    """A parser that loses a token: its messages no longer render to the tokens."""

    def assistant_message(self, completion_ids):
        return super().assistant_message(list(completion_ids)[1:])


def test_a_row_whose_messages_do_not_render_to_its_tokens_is_skipped():
    counts = {}
    rows = asyncio.run(_rows_with(_DropsAToken(), counts))
    assert rows == [] and counts["unrendered"] == 2


async def _rows_with(renderer, counts, records=None):
    return [row async for row in episode_rows(job=JOB, records=records or _Records(),
                                              renderer=renderer, source=SOURCE, counts=counts,
                                              quarantined=("q1",))]


def test_tool_call_ids_are_qualified_by_turn():
    rows, _ = _collect(True)
    messages = json.loads(rows[0]["messages"])
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["id"] == "call_0_0"
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_0_0"]


def test_a_grade_naming_another_task_is_not_exported():
    records = _Records()
    records.grades[IDS["certified"]]["instance_id"] = "i2"
    counts = {}
    rows = asyncio.run(_rows_with(R, counts, records))
    assert IDS["certified"] not in [r["submission_id"] for r in rows]
    assert counts["task_mismatch"] == 1


def _drained(monkeypatch, drained):
    from reliquary.validator import corpus_job_status

    async def stored_job_counts(records, job_id):
        return {"drained": drained}

    monkeypatch.setattr(corpus_job_status, "stored_job_counts", stored_job_counts)


def test_cli_refuses_a_job_still_moving(cli, monkeypatch, tmp_path):
    _drained(monkeypatch, False)
    out = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out), "--sft"])
    assert result.exit_code != 0 and "drained" in result.output and not out.exists()


def test_cli_exports_a_moving_job_when_told_to_and_says_so(cli, monkeypatch, tmp_path):
    _drained(monkeypatch, False)
    out = tmp_path / "sft.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out), "--sft",
                                      "--allow-incomplete"])
    assert result.exit_code == 0, result.output
    sidecar = json.loads((tmp_path / "sft.jsonl.counts.json").read_text())
    assert sidecar["drained"] is False and sidecar["exported"] == 1


def test_cli_writes_the_counts_beside_the_rows(cli, monkeypatch, tmp_path):
    _drained(monkeypatch, True)
    out = tmp_path / "all.jsonl"
    result = CliRunner().invoke(cli, ["jobs", "export", JOB.job_id, "--out", str(out)])
    assert result.exit_code == 0, result.output
    sidecar = json.loads((tmp_path / "all.jsonl.counts.json").read_text())
    assert sidecar["drained"] is True and sidecar["sft_only"] is False
    assert (sidecar["exported"], sidecar["ungraded"], sidecar["held"], sidecar["voided"],
            sidecar["uncertified"], sidecar["unparseable"]) == (2, 1, 1, 1, 2, 0)
    assert sidecar["quarantined_executors"] == ["q1"]
    assert isinstance(sidecar["exported_at"], float)
    assert sidecar["counts"]["rows"] == 2
