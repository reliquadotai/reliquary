"""Delivery v3 of an agentic (signed-sandbox) order: one row per certified trajectory, its full transcript and its
signed grade; annotate delivers every one, drop the successes only."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

pytest.importorskip("reliquary_sandbox.attest")

import pyarrow.parquet as pq  # noqa: E402

from reliquary.corpus.delivery import (  # noqa: E402
    DELIVERY_SCHEMA_V3, V3_EPISODE_ROW_FIELDS, LocalDirectorySink, export_delivery,
)
from reliquary.corpus.job import parse_job  # noqa: E402
from tests.unit.test_corpus_export_signed import build  # noqa: E402
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import signed_episode  # noqa: E402


def _job():
    job = parse_job(_manifest(episode=signed_episode(), job_id="order-gen-swe1", prompt_count=3))
    return replace(job, seed=7, sampling=replace(job.sampling, draw="gumbel_max"))


class _Source:
    def instance_id(self, index):
        return f"repo__{index}"

    def prompt(self, index):
        return "Fix task 0."


class _Records:
    """Prompt 0: a success (audited) and a failure; prompt 1: a success and an uncertified grade; prompt 2:
    nothing. One verdict failed its audit."""

    def __init__(self, trajectory):
        self.subs, self.verdicts, self.grades = {}, {}, {}
        plan = [(0, 0, 1.0, True), (0, 1, 0.0, True), (1, 0, 1.0, True), (1, 1, 1.0, False)]
        for k, (prompt, slot, reward, certified) in enumerate(plan):
            sid = f"{k:064x}"
            self.subs[sid] = {"prompt_index": prompt, "cursor": slot, "hotkey": "hk",
                              "completions": [trajectory]}
            self.verdicts[sid] = {"passed": True, "audited": k == 0, "token_count": 1,
                                  **({"forced_draw": {"match": 0.995}} if k == 0 else {})}
            self.grades[sid] = {"status": "ok" if certified else "error", "replay_certified": certified,
                                "replay": {"status": "ok", "certified": certified}, "grade": {"reward": reward},
                                "graded_success": reward == 1.0, "instance_id": f"repo__{prompt}"}
        self.verdicts[f"{9:064x}"] = {"passed": False, "audited": True, "reason": "draw_mismatch"}

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def list_voided_ids(self, job_id):
        return []

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def read_regrade(self, job_id, sid):
        return None


def _deliver(tmp_path, *, filter_mode="annotate", output_format="parquet", **kw):
    trajectory, renderer = build(tmp_path)
    sink = LocalDirectorySink(tmp_path / "sink")
    manifest = asyncio.run(export_delivery(
        job=_job(), records=_Records(trajectory), sink=sink, delivery_id="d1", schema_version=3,
        filter_mode=filter_mode, output_format=output_format, renderer=renderer, source=_Source(),
        quarantined=[], work_dir=tmp_path / "work", clock=lambda: 100.0, **kw))
    report = json.loads((tmp_path / "sink" / "deliveries" / "d1" / "report.json").read_text())
    return manifest, report, tmp_path / "sink" / "deliveries" / "d1"


def test_annotate_delivers_every_certified_trajectory_with_its_grade(tmp_path):
    manifest, report, root = _deliver(tmp_path)
    assert manifest["schema"] == DELIVERY_SCHEMA_V3 and manifest["columns"] == list(V3_EPISODE_ROW_FIELDS)
    rows = pq.read_table(root / "part-00000.parquet").to_pylist()
    assert [(r["prompt_index"], r["reward"], r["graded_success"]) for r in rows] == [
        (0, 1.0, True), (0, 0.0, False), (1, 1.0, True)]
    first = rows[0]
    assert first["certified"] and first["audited"] and first["draw_match"] == 0.995
    assert first["draw_slot"] == 0 and first["seed"] == 7 and first["task_id"] == "repo__0"
    messages = json.loads(first["messages"])
    assert [m["role"] for m in messages][:3] == ["system", "user", "assistant"]
    assert any(m["role"] == "tool" for m in messages) and messages[2]["tool_calls"]
    assert first["completion_tokens"] == sum(e - s for s, e in json.loads(first["turns"]))
    assert report["filter"]["mode"] == "annotate" and report["verification"]["grade"] == "signed-sandbox"
    assert report["counts"]["episodes"]["uncertified"] == 1 and report["counts"]["verdicts"] == 5
    assert report["billing"]["completion_tokens"] == sum(r["completion_tokens"] for r in rows)
    assert report["billing"]["audited_completion_tokens"] == first["completion_tokens"]
    assert "sandbox time" in report["billing"]["note"]
    # Prompt 1 is one sample short (its uncertified one), prompt 2 has none.
    errors = [json.loads(line) for line in (root / "errors-00000.jsonl").read_text().splitlines()]
    assert [(e["prompt_index"], e["error"]["code"]) for e in errors] == [(1, "missing_samples"),
                                                                         (2, "missing_samples")]


def test_drop_delivers_the_successes_only(tmp_path):
    _, report, root = _deliver(tmp_path, filter_mode="drop")
    rows = pq.read_table(root / "part-00000.parquet").to_pylist()
    assert [r["graded_success"] for r in rows] == [True, True]
    assert report["counts"]["rejected_dropped"] == 1
    # A dropped failure was generated: its prompt is not short of it.
    errors = (root / "errors-00000.jsonl").read_text()
    assert '"prompt_index":0' not in errors


def test_openai_lines_carry_the_transcript_and_grade(tmp_path):
    _, _, root = _deliver(tmp_path, output_format="openai_jsonl")
    lines = [json.loads(line) for line in (root / "output-00000.jsonl").read_text().splitlines()]
    assert [line["custom_id"] for line in lines] == ["prompt-0", "prompt-1"]
    choices = lines[0]["response"]["body"]["choices"]
    assert len(choices) == 2 and choices[0]["message"]["role"] == "assistant"
    assert choices[0]["messages"][0]["role"] == "system"
    assert choices[0]["reliquary"]["reward"] == 1.0 and choices[1]["reliquary"]["graded_success"] is False


@pytest.mark.parametrize("kw,word", [
    ({"filter_mode": "none"}, "always graded"),
    ({"dedupe": True}, "deduplicated"),
])
def test_an_episode_delivery_is_always_graded_and_never_deduped(tmp_path, kw, word):
    with pytest.raises(ValueError, match=word):
        _deliver(tmp_path, **kw)
