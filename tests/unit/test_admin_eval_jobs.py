"""Admin: qualification requests and evaluation jobs (design v2, item 2)."""

from __future__ import annotations

import asyncio
import json
import secrets
import time

import pytest
from fastapi.testclient import TestClient

from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
from reliquary.admin.service import create_admin_app
from reliquary.eval import prompt_source as ps
from reliquary.eval import qualification as qual
from reliquary.eval.sets import build_set
from reliquary.eval.storage import SubnetEvalStore, publish_set
from reliquary.infrastructure import corpus_executor_store as executors
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
from tests.unit.test_eval_sets import opener
from tests.unit.test_jobs_cli import _rl_entry, registry  # noqa: F401

SECRET = b"admin-secret"
MODEL = "customer/Model-8B"
REVISION = "c" * 40
THRESHOLDS = {"exp_mismatch_threshold": 75, "mant_mean_threshold": 41.5,
              "mant_median_threshold": 40.0}


@pytest.fixture
def admin(tmp_path, monkeypatch, registry):  # noqa: F811
    from reliquary.corpus.delivery import LocalDirectorySink

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    monkeypatch.setattr(executors, "get_s3_client", lambda **kw: fake)
    monkeypatch.setattr(ps, "_loaded", {})
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    build_set("logic", count=8, seed=1, out=tmp_path / "set", open_environment=opener(),
              clock=lambda: 1.0)
    asyncio.run(publish_set(tmp_path / "set", platform=LocalDirectorySink(tmp_path / "p"),
                            subnet=SubnetEvalStore()))
    app = create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=object(),
                           current_round=lambda: 1)
    client = TestClient(app)
    client.__enter__()

    def call(method, path, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return client.request(method, path, content=data, headers=headers)

    call.registry, call.bucket = registry, fake
    yield call
    client.__exit__(None, None, None)


def _qualification(**kw):
    return {"qualification_id": "order-q1", "model": MODEL, "revision": REVISION,
            "set_id": "logic-eval-s1-n8", "problems": 4, "completions": 8,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "max_new_tokens": 512, "thinking": False, **kw}


def _qualify(admin, status=qual.QUALIFIED):
    """Write the control's side of a finished qualification."""
    async def finish():
        store = qual.QualificationStore()
        record, etag = await store.read("order-q1")
        record.update(status=status, result={
            "thresholds": THRESHOLDS, "architecture": "Qwen3ForCausalLM",
            "checkpoint_sha256": "d" * 64, "eos_token_id": 151645,
            "band": {"exp_mismatch": 50, "mant_mean": 27.6, "mant_median": 20.0, "chunks": 90},
            "tokens_per_gpu_hour": 3.6e6})
        await store.write(record, etag)

    asyncio.run(finish())


def test_a_qualification_is_queued_once_and_read_back(admin):
    first = admin("POST", "/admin/v1/qualifications", _qualification())
    assert first.status_code == 201, first.text
    assert first.json()["status"] == "pending"
    again = admin("POST", "/admin/v1/qualifications", _qualification())
    assert again.status_code == 200
    other = admin("POST", "/admin/v1/qualifications", _qualification(max_new_tokens=9))
    assert other.status_code == 409
    read = admin("GET", "/admin/v1/qualifications/order-q1")
    assert read.status_code == 200 and read.json()["set_id"] == "logic-eval-s1-n8"
    assert admin("GET", "/admin/v1/qualifications/order-none").status_code == 404
    assert admin("POST", "/admin/v1/qualifications",
                 _qualification(qualification_id="x-q")).status_code == 409
    assert admin("POST", "/admin/v1/qualifications",
                 _qualification(qualification_id="order-q2",
                                set_id="logic-eval-s9-n9")).status_code == 404


def _eval_job(**kw):
    return {"job_id": "order-eval-7", "model": MODEL, "env": "logic", "prompt_count": 5,
            "samples_per_prompt": 4, "max_new_tokens": 512, "thinking": False,
            "eval_set_id": "logic-eval-s1-n8", "qualification_id": "order-q1", **kw}


def test_an_eval_job_takes_its_prompts_model_and_thresholds_from_set_and_qualification(admin):
    admin("POST", "/admin/v1/qualifications", _qualification())
    unqualified = admin("POST", "/admin/v1/jobs", _eval_job())
    assert unqualified.status_code == 409 and "model_not_qualified" in unqualified.text
    _qualify(admin)
    created = admin("POST", "/admin/v1/jobs", _eval_job())
    assert created.status_code == 201, created.text
    assert created.json()["cap"] == 0.02
    job, _ = asyncio.run(job_store.read_job("order-eval-7"))
    source = ps.parse_eval_source(job.prompt_source)
    assert (source.set_id, source.count) == ("logic-eval-s1-n8", 5)
    assert job.checkpoint_revision == REVISION and job.checkpoint_sha256 == "d" * 64
    assert job.seed is not None and job.slots_per_prompt == 4
    entry = admin.registry["entries"]["order-eval-7"]
    assert entry.params["audit_q"] == 1.0
    toploc = [p for p in entry.contract["proofs"] if p["scheme"] == "toploc-v1"][0]
    assert {k: toploc[k] for k in THRESHOLDS} == THRESHOLDS and toploc["mode"] == "enforce"
    assert list(entry.contract["environments"]) == ["reliquary_logic_v2"]
    # Idempotent.
    assert admin("POST", "/admin/v1/jobs", _eval_job()).status_code == 200


@pytest.mark.parametrize("change,status", [
    ({"job_id": "order-7", "task_id": None}, 422),         # eval set without the eval prefix
    ({"audit_q": 0.5}, 422),
    ({"prompt_count": 9}, 422),                             # more than the set holds
    ({"env": "code"}, 422),
    ({"model": "someone/else"}, 409),
    ({"qualification_id": None}, 422),
    ({"eval_set_id": "logic-eval-s9-n9"}, 404),
])
def test_eval_job_refusals(admin, change, status):
    admin("POST", "/admin/v1/qualifications", _qualification())
    _qualify(admin)
    response = admin("POST", "/admin/v1/jobs", {**_eval_job(), **change})
    assert response.status_code == status, response.text


def test_only_an_eval_job_may_take_the_eval_prefix(admin):
    response = admin("POST", "/admin/v1/jobs", {
        "job_id": "order-eval-9", "model": MODEL, "env": "reliquary_logic_v2",
        "prompt_count": 5, "samples_per_prompt": 1, "cap": 0.01})
    assert response.status_code == 422
    response = admin("POST", "/admin/v1/jobs", {
        "job_id": "order-9", "model": MODEL, "env": "reliquary_logic_v2",
        "prompt_count": 5, "samples_per_prompt": 1, "cap": 0.01, "seed": 3})
    assert response.status_code == 422


def test_thresholds_under_the_floor_are_refused():
    from reliquary.cli.main import _with_enforced_toploc

    with pytest.raises(ValueError, match="floor"):
        _with_enforced_toploc({"proofs": []}, {**THRESHOLDS, "exp_mismatch_threshold": 10})
    contract = _with_enforced_toploc({"proofs": []}, THRESHOLDS)
    assert contract["proofs"][0]["exp_mismatch_threshold"] == 75


def test_the_seed_is_written_only_when_present():
    from dataclasses import replace

    from reliquary.corpus.job import parse_job
    from tests.unit.test_corpus_export import _job_spec

    plain = _job_spec().to_contract()
    assert "seed" not in plain and parse_job(plain).seed is None
    seeded = replace(_job_spec(), seed=12).to_contract()
    assert seeded["seed"] == 12 and parse_job(seeded).seed == 12
    with pytest.raises(ValueError):
        parse_job({**plain, "seed": -1})
