"""Admin: a job grading needs no provenance; a graded evaluation's files are served."""

from __future__ import annotations

import json
import time

import pytest

from tests.unit.test_admin_eval_jobs import (  # noqa: F401
    _eval_job, _qualification, _qualify, admin,
)
from tests.unit.test_jobs_cli import registry  # noqa: F401

SET = "logic-eval-s1-n8"


def _complete_job(admin):  # noqa: F811
    admin("POST", "/admin/v1/qualifications", _qualification())
    _qualify(admin)
    assert admin("POST", "/admin/v1/jobs", _eval_job()).status_code == 201
    grading = [json.loads(line) for line in
               (admin.root / "set" / "grading.jsonl").read_text().splitlines()]
    for prompt in range(5):
        for k in range(4):
            sid = f"{prompt:032x}{k:032x}"
            text = f'```json\n{{"a": "={grading[prompt]["source_index"]}"}}\n```' if k else "x"
            admin.records.subs[sid] = {"prompt_index": prompt, "hotkey": "hk",
                                       "completions": [{"text": text, "tokens": [1, 2]}]}
            admin.records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": "hk"}
            admin.records.settled.append(sid)


def _job_body(**kw):
    return {"source": "job", "job_id": "order-eval-7", "set_ids": [SET],
            "problems_per_set": {SET: 5}, "samples_per_set": {SET: 4}, **kw}


def _graded(admin, body):  # noqa: F811
    for _ in range(300):
        response = admin("POST", "/admin/v1/evaluations/order-eval-7/grade", body)
        if response.status_code != 202:
            return response
        time.sleep(0.02)
    raise AssertionError("grading never finished")


def test_a_job_is_graded_without_a_provenance(admin):  # noqa: F811
    _complete_job(admin)
    response = _graded(admin, _job_body())
    assert response.status_code == 200, response.text
    report = json.loads((admin.root / "p" / "evaluations" / "order-eval-7" / "report.json")
                        .read_text())
    provenance = report["provenance"]
    assert provenance["model"] == "customer/Model-8B" and provenance["sampling"]["top_k"] == 20
    assert report["envs"]["logic"]["pass@1"]["value"] == pytest.approx(0.75)


def test_an_uploads_grading_still_needs_its_provenance():
    from pydantic import ValidationError

    from reliquary.admin.service import GradeEvaluation

    with pytest.raises(ValidationError, match="provenance"):
        GradeEvaluation(set_ids=[SET], completion_keys=["evaluations/x/c.jsonl"],
                        problems_per_set={SET: 1}, samples_per_set={SET: 1})


def test_the_graded_files_are_served(admin):  # noqa: F811
    path = "/admin/v1/evaluations/order-eval-7/files/report.json"
    assert admin("GET", path).status_code == 404
    _complete_job(admin)
    assert _graded(admin, _job_body()).status_code == 200
    report = admin("GET", path)
    assert report.status_code == 200 and json.loads(report.content)["eval_id"] == "order-eval-7"
    manifest = admin("GET", "/admin/v1/evaluations/order-eval-7/files/manifest.json")
    assert json.loads(manifest.content)["complete"] is True
    parquet = admin("GET", "/admin/v1/evaluations/order-eval-7/files/graded.parquet")
    assert parquet.status_code == 200 and parquet.content[:4] == b"PAR1"
    assert admin("GET", "/admin/v1/evaluations/order-eval-7/files/set.json").status_code == 422
    assert admin("GET", "/admin/v1/evaluations/other-1/files/report.json").status_code == 409
