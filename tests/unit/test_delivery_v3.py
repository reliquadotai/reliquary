"""Delivery v3: the one delivery of every order (batch on a file, batch on an env range, eval)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from reliquary.corpus.delivery import (
    DELIVERY_SCHEMA_V3,
    V3_ROW_FIELDS,
    LocalDirectorySink,
    delivery_rows,
    delivery_rows_v3,
    export_delivery,
    set_rows_for_job,
)
from reliquary.corpus.encoding import token_ids_text

EOS = 99


def _sid(i: int) -> str:
    return f"{i:064x}"


def _job(*, draw="gumbel_max", prompt_count=4, slots=2, filter=None, prompt_source="math"):
    sampling = SimpleNamespace(draw=draw, n=1, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
                               max_new_tokens=8)
    return SimpleNamespace(
        job_id="order-gen-x", filter=filter, eos_token_id=EOS, sampling=sampling, seed=7 if draw else None,
        slots_per_prompt=slots, prompt_start=0, prompt_count=prompt_count, prompt_end=prompt_count,
        checkpoint_sha256="c" * 64, checkpoint_revision="r" * 40, checkpoint_repo="org/model",
        judge=None, prompt_source=prompt_source, episode=None,
        to_contract=lambda: {"job_id": "order-gen-x"})


class _Records:
    """Prompt p has submissions at slots 0 and 1 (prompt 3: none). Slot 1 of prompt 0 is cut at the budget,
    prompt 2 slot 1's verdict failed, prompt 1 slot 0 is voided after the listing."""

    def __init__(self):
        self.subs, self.verdicts = {}, {}
        self.late_voided = set()
        k = 0
        for p in range(3):
            for slot in range(2):
                sid = _sid(k)
                cut = p == 0 and slot == 1
                failed = p == 2 and slot == 1
                self.verdicts[sid] = ({"passed": False, "audited": True, "reason": "draw_mismatch"} if failed
                                      else {"passed": True, "audited": slot == 0,
                                            **({"forced_draw": {"match": 0.99}} if slot == 0 else {})})
                self.subs[sid] = {
                    "hotkey": f"HK{k}", "prompt_index": p, "cursor": slot,
                    "rendered_prompt": token_ids_text([5, 6, 7 + p]),
                    "completions": [{"text": f"p{p}s{slot}", "tokens": [1, 2] if cut else [1, 2, EOS]}]}
                k += 1
        self.late_voided.add(_sid(2))

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)

    async def list_voided_ids(self, job_id):
        return []

    async def read_voided(self, job_id, sid):
        return {} if sid in self.late_voided else None


SET_ROWS = tuple({"problem_id": f"p{i}", "custom_id": f"req-{i}", "prompt_token_ids": [5, 6, 7 + i],
                  "url": "/v1/completions"} for i in range(4))


class _Slots:
    def __init__(self, exhausted=()):
        self.exhausted = set(exhausted)

    def prompt_state(self, index):
        return "exhausted" if index in self.exhausted else "open"


def _v3_rows(job, records, **kw):
    async def go():
        counts = {}
        rows = [r async for r in delivery_rows_v3(job=job, records=records, counts=counts, **kw)]
        return rows, counts

    return asyncio.run(go())


def test_v3_rows_name_the_draw_sample_and_keep_a_length_cut_answer():
    rows, counts = _v3_rows(_job(), _Records(), set_rows=SET_ROWS)
    assert all(set(r) == set(V3_ROW_FIELDS) for r in rows)
    # Prompt 1 slot 0 voided late, prompt 2 slot 1 failed: 4 rows.
    assert [(r["prompt_index"], r["draw_slot"]) for r in rows] == [(0, 0), (0, 1), (1, 1), (2, 0)]
    cut = rows[1]
    assert cut["finish_reason"] == "length" and cut["completion_token_ids"] == [1, 2]
    first = rows[0]
    assert (first["custom_id"], first["problem_id"], first["seed"], first["sample_index"]) == ("req-0", "p0", 7, 0)
    assert first["prompt_tokens"] == 3 and first["audited"] is True and first["draw_match"] == 0.99
    assert rows[1]["audited"] is False and rows[1]["draw_match"] is None
    assert first["prompt_sha256"] == hashlib.sha256(token_ids_text([5, 6, 7]).encode()).hexdigest()
    assert counts["finish_reason"] == {"stop": 3, "length": 1} and counts["voided"] == 1
    assert counts["completion_tokens"] == 11 and counts["audited_completion_tokens"] == 6
    assert counts["prompt_tokens"] == 12


def test_v3_can_leave_out_length_cut_answers():
    rows, counts = _v3_rows(_job(), _Records(), include_truncated=False)
    assert all(r["finish_reason"] == "stop" for r in rows) and counts["truncated_dropped"] == 1


def test_a_job_without_a_draw_names_no_slot_or_seed():
    rows, _ = _v3_rows(_job(draw=None), _Records())
    assert all(r["draw_slot"] is None and r["seed"] is None and r["sample_index"] is None for r in rows)
    assert all(r["custom_id"] is None and r["problem_id"] is None for r in rows)


def test_the_filter_annotates_or_drops():
    grade = lambda index, text: (index == 0, 1.0 if index == 0 else 0.0)  # noqa: E731
    annotated, counts = _v3_rows(_job(), _Records(), grade=grade)
    assert len(annotated) == 4 and counts["rows_accepted"] == 2
    assert {(r["prompt_index"], r["accepted"]) for r in annotated} == {(0, True), (1, False), (2, False)}
    dropped, counts = _v3_rows(_job(), _Records(), grade=grade, drop_rejected=True)
    assert [r["prompt_index"] for r in dropped] == [0, 0] and counts["rejected_dropped"] == 2


def _export(tmp_path, job, records, **kw):
    sink = LocalDirectorySink(tmp_path / "platform")
    manifest = asyncio.run(export_delivery(job=job, records=records, sink=sink, delivery_id="d-1",
                                           work_dir=tmp_path / "work", schema_version=3, clock=lambda: 5.0,
                                           **kw))
    return manifest, tmp_path / "platform" / "deliveries" / "d-1"


def test_v3_parquet_delivery_writes_its_rows_report_and_errors(tmp_path):
    manifest, root = _export(tmp_path, _job(), _Records(), set_rows=SET_ROWS, slots=_Slots(exhausted={3}))
    assert manifest["schema"] == DELIVERY_SCHEMA_V3 and manifest["format"] == "parquet"
    assert manifest["columns"] == list(V3_ROW_FIELDS) and manifest["rows"] == 4
    table = pq.read_table(root / manifest["shards"][0]["name"])
    assert table.column_names == list(V3_ROW_FIELDS)
    errors = [json.loads(line) for line in (root / "errors-00000.jsonl").read_text().splitlines()]
    # Prompt 1 holds 1 of 2 samples (one voided), prompt 2 one (one failed), prompt 3 none and is exhausted.
    assert [(e["custom_id"], e["error"]["code"]) for e in errors] == [
        ("req-1", "missing_samples"), ("req-2", "missing_samples"), ("req-3", "draw_unfillable")]
    report = json.loads((root / "report.json").read_text())
    verification = report["verification"]
    assert verification["scheme"] == "forced-draw" and verification["draw"]["seed"] == 7
    assert verification["refusals"] == {"draw_mismatch": 1} and verification["audited"] == 4
    assert verification["judge"]["floor_rate"] == 0.98
    assert report["billing"]["completion_tokens"] == 11 and report["counts"]["prompts_short"] == 3
    # Stored: returned as it is.
    again, _ = _export(tmp_path, _job(), _Records(), set_rows=SET_ROWS)
    assert again == manifest


def test_v3_openai_jsonl_holds_one_line_per_prompt_with_its_samples(tmp_path):
    manifest, root = _export(tmp_path, _job(), _Records(), set_rows=SET_ROWS, output_format="openai_jsonl")
    assert manifest["columns"] is None
    lines = [json.loads(line) for line in (root / "output-00000.jsonl").read_text().splitlines()]
    assert [line["custom_id"] for line in lines] == ["req-0", "req-1", "req-2"]
    body = lines[0]["response"]["body"]
    assert lines[0]["error"] is None and lines[0]["response"]["status_code"] == 200
    assert body["object"] == "text_completion" and body["model"] == "org/model"
    assert [(c["index"], c["text"], c["finish_reason"]) for c in body["choices"]] == [
        (0, "p0s0", "stop"), (1, "p0s1", "length")]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_v3_openai_jsonl_answers_a_chat_request_with_a_chat_completion(tmp_path):
    rows = tuple({**r, "url": "/v1/chat/completions"} for r in SET_ROWS)
    _, root = _export(tmp_path, _job(), _Records(), set_rows=rows, output_format="openai_jsonl")
    body = json.loads((root / "output-00000.jsonl").read_text().splitlines()[0])["response"]["body"]
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "p0s0"}


def test_v3_refuses_a_filter_without_a_grader_and_v2_options_stay_v2(tmp_path):
    with pytest.raises(ValueError, match="grader"):
        _export(tmp_path, _job(), _Records(), filter_mode="annotate")
    with pytest.raises(ValueError, match="annotate"):
        _export(tmp_path, _job(), _Records(), keep_rejected=True)
    with pytest.raises(ValueError, match="2 or 3"):
        asyncio.run(export_delivery(job=_job(), records=_Records(), sink=LocalDirectorySink(tmp_path),
                                    delivery_id="d", schema_version=4))


def test_set_rows_come_from_the_jobs_frozen_set(monkeypatch):
    from reliquary.eval import prompt_source

    body = "".join(json.dumps({"problem_id": f"p{i}", "custom_id": f"req-{i}", "prompt_token_ids": [5, 6, i + 1]})
                   + "\n" for i in range(2)).encode()
    source = prompt_source.eval_source_for("upload-abc", body, 2)
    monkeypatch.setattr(prompt_source, "FETCHERS", [lambda s: body])
    prompt_source.forget_eval_prompts(source)
    rows = set_rows_for_job(_job(prompt_source=source.name))
    assert [r["custom_id"] for r in rows] == ["req-0", "req-1"]
    assert set_rows_for_job(_job()) is None


def test_the_eval_grading_and_the_delivery_read_the_same_submissions():
    from reliquary.eval.grading import collect_job_records

    job, records = _job(), _Records()
    collected = asyncio.run(collect_job_records(job, records))

    async def delivered():
        counts = {}
        return {r["submission_id"] async for r in delivery_rows(job=job, records=records, counts=counts)}

    # A late void (prompt 1 slot 0) is out of both; the cut answer is unfinished for v2 only.
    assert {sid for sid, _ in collected["passing"]} == {_sid(0), _sid(1), _sid(3), _sid(4)}
    assert asyncio.run(delivered()) == {_sid(0), _sid(3), _sid(4)}
    assert collected["verdicts"] == 6 and collected["audited"] == 4


def test_a_sample_left_out_on_purpose_is_not_reported_missing(tmp_path):
    manifest, root = _export(tmp_path, _job(), _Records(), set_rows=SET_ROWS, slots=_Slots(exhausted={3}),
                             include_truncated=False)
    errors = [json.loads(line) for line in (root / "errors-00000.jsonl").read_text().splitlines()]
    # Prompt 0's second sample was generated and verified, then cut at the budget and left out: no error for it.
    assert [e["custom_id"] for e in errors] == ["req-1", "req-2", "req-3"]
    report = json.loads((root / "report.json").read_text())
    assert report["counts"]["truncated_dropped"] == 1 and report["counts"]["prompts_delivered"] == 3
