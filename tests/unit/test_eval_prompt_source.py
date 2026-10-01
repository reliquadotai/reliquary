"""An eval set as a corpus job's prompt source (design v2, item 3)."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from reliquary.eval import prompt_source as ps
from reliquary.eval.prompt_source import (
    EvalSetSpec,
    eval_source_for,
    job_prompt_lines,
    load_eval_rows,
    parse_eval_source,
    register_eval_prompts,
)
from reliquary.eval.sets import build_set
from tests.unit.test_eval_sets import opener


@pytest.fixture
def built(tmp_path, monkeypatch):
    build_set("logic", count=6, seed=3, out=tmp_path / "logic-eval-s3-n6",
              open_environment=opener(), clock=lambda: 1.0)
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [ps._from_directory])
    monkeypatch.setenv(ps.SETS_DIR_ENV, str(tmp_path))
    return tmp_path / "logic-eval-s3-n6"


def test_a_source_names_the_first_n_problems_and_their_hash(built):
    body = (built / "prompts.jsonl").read_bytes()
    source = eval_source_for("logic-eval-s3-n6", body, 4)
    head = b"".join(body.splitlines(keepends=True)[:4])
    assert source.sha256 == hashlib.sha256(head).hexdigest()
    assert parse_eval_source(source.name) == source
    for bad in ("eval-set:x", "eval-set:A:1:" + "0" * 64, "eval-set:x:0:" + "0" * 64,
                "openmathinstruct"):
        with pytest.raises(ValueError):
            parse_eval_source(bad)


def test_rows_are_loaded_checked_and_served_byte_for_byte(built):
    body = (built / "prompts.jsonl").read_bytes()
    source = eval_source_for("logic-eval-s3-n6", body, 4)
    rows = load_eval_rows(source)
    assert len(rows) == 4 and rows[0]["problem_id"] == "logic-eval-s3-n6-000000"
    assert job_prompt_lines(source) == b"".join(body.splitlines(keepends=True)[:4])
    environment = EvalSetSpec(source.name).create()
    assert len(environment) == 4
    problem = environment.get_problem(2)
    assert problem["prompt"] == rows[2]["messages"][-1]["content"]
    assert environment.problem_id(2) == "logic-eval-s3-n6-000002"
    assert "source_index" not in str(rows) and "ground_truth" not in str(rows)


def test_tampered_prompts_are_refused(built):
    body = (built / "prompts.jsonl").read_bytes()
    source = eval_source_for("logic-eval-s3-n6", body, 4)
    with pytest.raises(ValueError, match="hash"):
        register_eval_prompts(source, body.replace(b"logic", b"LOGIC", 1))
    with pytest.raises(ValueError, match="fewer"):
        register_eval_prompts(source, b"".join(body.splitlines(keepends=True)[:2]))


def test_an_unreadable_set_is_named(built, monkeypatch):
    monkeypatch.setenv(ps.SETS_DIR_ENV, str(built.parent / "nowhere"))
    with pytest.raises(ValueError, match="not readable"):
        load_eval_rows(parse_eval_source("eval-set:logic-eval-s3-n6:4:" + "a" * 64))


def _job(source: str, renderer: str = "chat-template-v1", count: int = 4):
    from tests.unit.test_corpus_export import _job_spec
    from dataclasses import replace

    return replace(_job_spec(job_id="order-eval-1", prompt_source=source, prompt_count=count),
                   renderer_id=renderer)


def test_a_corpus_job_reads_its_prompts_from_the_set(built):
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        prompt_job_for_spec,
        resolve_prompt_source,
    )

    body = (built / "prompts.jsonl").read_bytes()
    source = eval_source_for("logic-eval-s3-n6", body, 4)
    job = _job(source.name)
    task = prompt_job_for_spec(job).task_for(3)
    assert task.prompt == load_eval_rows(source)[3]["messages"][-1]["content"]
    with pytest.raises(CorpusPromptSourceError):
        prompt_job_for_spec(job).task_for(4)  # not the job's
    with pytest.raises(CorpusPromptSourceError, match="chat template"):
        resolve_prompt_source(source.name, renderer_id="reliquary-external-prompt-v1")
    # A job claiming more rows than its source names is refused.
    with pytest.raises(CorpusPromptSourceError):
        prompt_job_for_spec(_job(source.name, count=5))


def test_the_miner_reads_an_eval_jobs_prompts_and_submits_on_its_scoped_path():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    seen = []

    def handle(request):
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/eval-prompts"):
            return httpx.Response(200, content=b'{"x":1}\n')
        return httpx.Response(200, json={"accepted": True, "reason": "accepted"})

    http = httpx.Client(base_url="http://v", transport=httpx.MockTransport(handle))
    client = HttpCorpusClient(http, job_id="order-eval-7")
    assert client.eval_prompts() == b'{"x":1}\n'
    client.scoped_submit = True  # the miner sets it once the manifest names an eval set
    client.submit({"a": 1})
    assert ("POST", "/corpus/jobs/order-eval-7/submit") in seen
    plain = HttpCorpusClient(http, job_id="code-v1")
    plain.submit({"a": 1})
    assert ("POST", "/corpus/submit") in seen


def test_the_corpus_control_never_wires_an_eval_job():
    from types import SimpleNamespace

    from reliquary.validator.corpus_hot_jobs import OTHER_MODEL, hot_job_refusal

    entry = SimpleNamespace(job_id="order-eval-3", contract={"model_id": "x"})
    verdict = hot_job_refusal(entry, None, process_profile=None, process_contract={},
                              fingerprint="f")
    assert verdict[0] == OTHER_MODEL and "eval control" in verdict[1]
