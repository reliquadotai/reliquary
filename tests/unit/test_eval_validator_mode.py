"""Evaluations served by our own corpus validator: the model on its GPU, TOPLOC
local, the environment graded on CPU. No order control, no qualification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.eval.prompt_source import declared_environment
from tests.unit.test_corpus_hot_jobs import ENV, OTHER_MODEL, REFUSED, _hot_entry, _hot_job, _refusal
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401
from tests.unit.test_corpus_validator import seeded_job  # noqa: F401

EVAL = "eval-set:aime26-set:30:" + "a" * 64


def test_a_catalog_job_declares_its_prompt_source():
    assert declared_environment({"environments": {"src": ENV}}, "src") == "src"


def test_an_eval_job_declares_the_one_environment_of_its_contract():
    contract = {"environments": {"reliquary_external_eval_v1": ENV}}
    assert declared_environment(contract, EVAL) == "reliquary_external_eval_v1"
    assert declared_environment({"environments": {"a": ENV, "b": ENV}}, EVAL) is None
    assert declared_environment({"environments": {}}, EVAL) is None


def _eval_entry(env_name="reliquary_external_eval_v1", **kw):
    entry = _hot_entry(**kw)
    entry.contract["environments"] = {env_name: ENV}
    return entry


def test_our_validator_serves_an_eval_job_on_its_model():
    process = {"environments": {"reliquary_external_eval_v1": ENV}, "protocol_version": 5}
    assert _refusal(_eval_entry(), _hot_job(prompt_source=EVAL), process_contract=process) is None


def test_an_eval_job_the_process_was_not_started_for_is_refused():
    process = {"environments": {"src": ENV}, "protocol_version": 5}
    kind, why = _refusal(_eval_entry(), _hot_job(prompt_source=EVAL), process_contract=process)
    assert kind == REFUSED and "reliquary_external_eval_v1" in why


def test_an_eval_job_on_another_model_waits_for_its_own_validator():
    kind, _ = _refusal(_eval_entry(model="org/Other"), _hot_job(prompt_source=EVAL))
    assert kind == OTHER_MODEL


def test_several_eval_jobs_share_one_process():
    from reliquary.validator.corpus_validator import multi_job_refusal

    process = {"environments": {"reliquary_external_eval_v1": ENV}}
    pairs = [(_eval_entry(task_id=f"t{i}", job_id=f"j{i}"),
              _hot_job(job_id=f"j{i}", prompt_source=EVAL)) for i in range(2)]
    assert multi_job_refusal(pairs, proof_of=lambda e: "p", process_contract=process) is None
    process = {"environments": {"other": ENV}}
    assert "differently" in multi_job_refusal(pairs, proof_of=lambda e: "p",
                                              process_contract=process)


def test_the_corpus_control_serves_an_eval_jobs_prompts(seeded_job, monkeypatch):
    import dataclasses
    import json

    from fastapi.testclient import TestClient

    from reliquary.eval import prompt_source as ps
    from reliquary.validator.corpus_validator import build_corpus_app
    from tests.unit.test_corpus_validator import _Tokenizer, _entry

    monkeypatch.setattr(ps, "_loaded", {})
    body = b"".join(json.dumps({"problem_id": f"s-{i}", "env": "x", "set_id": "s",
                                "messages": [{"role": "user", "content": f"q{i}"}]},
                               sort_keys=True).encode() + b"\n" for i in range(3))
    source = ps.eval_source_for("s", body, 2)
    ps.register_eval_prompts(source, body)

    def app_for(job):
        return TestClient(build_corpus_app(
            entry=_entry(), job=job, store=seeded_job.store, records=None,
            tokenizer=_Tokenizer(), renderer=seeded_job.renderer, verify_signature=lambda r: True,
            auditor=SimpleNamespace(enqueue=lambda r: None), proof_chunk_tokens=None,
            prompt_job_for=seeded_job.prompt_job_for))

    eval_job = dataclasses.replace(seeded_job.job, prompt_source=source.name)
    client = app_for(eval_job)
    served = client.get(f"/corpus/jobs/{eval_job.job_id}/eval-prompts")
    assert served.status_code == 200 and served.content == b"".join(body.splitlines(True)[:2])
    assert client.get("/corpus/jobs/nope/eval-prompts").status_code == 404
    catalog = app_for(seeded_job.job)
    refused = catalog.get(f"/corpus/jobs/{seeded_job.job.job_id}/eval-prompts")
    assert refused.status_code == 404 and refused.json()["detail"] == "not_an_eval_job"
