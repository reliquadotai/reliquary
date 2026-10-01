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


class _JobRecords:
    """An eval job's records: submissions, verdicts, voided ids, settlement."""

    def __init__(self):
        self.subs, self.verdicts, self.voided, self.settled = {}, {}, set(), []

    async def list_submission_ids(self, job_id):
        return sorted(self.subs)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)

    async def read_settlement(self, job_id):
        return {"settled": list(self.settled), "pending": None}, None


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
    from tests.unit.test_eval_grading import GradingEnvironment, _plain_scorer

    records = _JobRecords()
    app = create_admin_app(secret=SECRET, pool_max=0.3, models={}, records=records,
                           current_round=lambda: 1,
                           deliveries=LocalDirectorySink(tmp_path / "p"),
                           open_environment=lambda source, split: GradingEnvironment(source),
                           grade_scorer=_plain_scorer, require_sandbox=lambda spec: None,
                           work_dir=tmp_path / "work")
    client = TestClient(app)
    client.__enter__()

    def call(method, path, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(SECRET, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return client.request(method, path, content=data, headers=headers)

    call.registry, call.bucket, call.records, call.root = registry, fake, records, tmp_path
    yield call
    client.__exit__(None, None, None)


SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def _qualification(**kw):
    return {"qualification_id": "order-q1", "model": MODEL, "revision": REVISION,
            "set_id": "logic-eval-s1-n8", "problems": 5, "completions": 8,
            "sampling": SAMPLING,
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
            "clamped": [], "tokens_per_gpu_hour": 3.6e6,
            "measurements": {"e1": {"provider_id": "lium-1", "host": "h1", "gpu": "H100"},
                             "e2": {"provider_id": "lium-2", "host": "h2", "gpu": "H100"}}})
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
            "sampling": SAMPLING, "eval_set_id": "logic-eval-s1-n8",
            "qualification_id": "order-q1", **kw}


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
    # The order's sampling, the one qualification measured.
    assert (job.sampling.temperature, job.sampling.top_p, job.sampling.top_k) == (0.6, 0.95, 20)
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
    ({"sampling": {**SAMPLING, "temperature": 1.0}}, 409),  # not what was qualified
    ({"max_new_tokens": 1024}, 409),
    ({"thinking": True}, 409),
    ({"prompt_count": 4}, 409),
    ({"sampling": None}, 422),
    ({"qualification_id": "x-q1"}, 409),                    # outside the admin scope
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
    # A corpus job still names its cap: no silent default.
    response = admin("POST", "/admin/v1/jobs", {
        "job_id": "order-9", "model": MODEL, "env": "reliquary_logic_v2",
        "prompt_count": 5, "samples_per_prompt": 1})
    assert response.status_code == 422 and "cap_required" in response.text


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


PROVENANCE = {"model": MODEL, "revision": REVISION, "model_sha": REVISION,
              "sampling": SAMPLING, "thinking": False, "max_new_tokens": 512}


def _grade_body(**kw):
    return {"source": "job", "job_id": "order-eval-7", "set_ids": ["logic-eval-s1-n8"],
            "problems_per_set": {"logic-eval-s1-n8": 5},
            "samples_per_set": {"logic-eval-s1-n8": 4}, "provenance": PROVENANCE, **kw}


def test_an_eval_job_is_graded_from_its_passing_records(admin):
    admin("POST", "/admin/v1/qualifications", _qualification())
    _qualify(admin)
    assert admin("POST", "/admin/v1/jobs", _eval_job()).status_code == 201
    grading = [json.loads(l) for l in (admin.root / "set" / "grading.jsonl").read_text().splitlines()]
    index = [g["source_index"] for g in grading]
    right = lambda i: f'```json\n{{"a": "={index[i]}"}}\n```'  # noqa: E731

    def submit(sid, prompt, texts, passed=True, hotkey="hk1", audited=True):
        admin.records.subs[sid] = {"prompt_index": prompt, "hotkey": hotkey, "completions": [
            {"text": t, "tokens": [5, 2]} for t in texts]}
        admin.records.verdicts[sid] = {"passed": passed, "audited": audited, "hotkey": hotkey}
        admin.records.settled.append(sid)

    for k in range(4):
        submit(f"{k:064x}", 0, [right(0)])
    submit("a" * 64, 1, [right(1)], hotkey="hk2")
    submit("b" * 64, 1, ["no json"], hotkey="hk2")
    submit("c" * 64, 1, [right(1)], passed=False)
    submit("d" * 64, 2, [right(2)])
    admin.records.voided.add("d" * 64)
    # Not drained: refused.
    admin.records.subs["e" * 64] = {"prompt_index": 3, "hotkey": "x", "completions": []}
    assert admin("POST", "/admin/v1/evaluations/order-e7/grade",
                 _grade_body()).status_code == 409
    del admin.records.subs["e" * 64]
    wrong = admin("POST", "/admin/v1/evaluations/order-e7/grade",
                  _grade_body(samples_per_set={"logic-eval-s1-n8": 2}))
    assert wrong.status_code == 422 and "grades as" in wrong.text
    lying = admin("POST", "/admin/v1/evaluations/order-e7/grade", _grade_body(
        provenance={**PROVENANCE, "sampling": {"temperature": 0.1}}))
    assert lying.status_code == 422 and "provenance_mismatch" in lying.text
    # Drained but not every prompt holds its samples: refused unless allowed.
    incomplete = admin("POST", "/admin/v1/evaluations/order-e7/grade", _grade_body())
    assert incomplete.status_code == 409 and "job_not_complete" in incomplete.text
    for _ in range(200):
        response = admin("POST", "/admin/v1/evaluations/order-e7/grade",
                         _grade_body(allow_incomplete=True))
        if response.status_code != 202:
            break
        time.sleep(0.02)
    assert response.status_code == 200, response.text
    assert response.json()["complete"] is False
    report = json.loads((admin.root / "p" / "evaluations" / "order-e7" / "report.json").read_text())
    logic = report["envs"]["logic"]
    assert (logic["n_problems"], logic["samples"], logic["expected_rows"]) == (5, 4, 20)
    assert (logic["graded_rows"], logic["missing_rows"]) == (6, 14)
    # c/n: 4/4, 1/4, then three problems with nothing.
    assert logic["pass@1"]["value"] == pytest.approx((1 + 0.25) / 5)
    provenance = report["provenance"]
    assert provenance["generation"] == "sn81-miners" and provenance["miner_hotkeys"] == 2
    assert provenance["audited_fraction"] == 1.0 and provenance["sampling_verified"] is False
    assert "sampling not verified" in provenance["note"]
    # From the job and its qualification, not from the request.
    assert provenance["checkpoint_sha256"] == "d" * 64 and provenance["sampling"] == SAMPLING
    assert provenance["job_complete"] is False and provenance["allow_incomplete"] is True
    verification = provenance["verification"]
    assert verification["thresholds"] == THRESHOLDS and verification["qualification_id"] == "order-q1"
    assert set(verification["qualification"]["qualifiers"]) == {"e1", "e2"}
    assert "vllm_version" not in provenance


def test_a_job_with_no_submission_is_never_complete(admin):
    from reliquary.eval.grading import job_complete

    admin("POST", "/admin/v1/qualifications", _qualification())
    _qualify(admin)
    admin("POST", "/admin/v1/jobs", _eval_job())
    job, _ = asyncio.run(job_store.read_job("order-eval-7"))
    assert job_complete(job, {"samples_by_prompt": {}}, 4) is False
    assert job_complete(job, {"samples_by_prompt": {i: 4 for i in range(5)}}, 4) is True
    assert job_complete(job, {"samples_by_prompt": {i: 4 for i in range(4)}}, 4) is False
    response = admin("POST", "/admin/v1/evaluations/order-e7/grade", _grade_body())
    assert response.status_code == 409 and "job_not_complete" in response.text


def test_a_job_grading_needs_a_job_and_an_uploads_grading_its_pod():
    from reliquary.admin.service import GradeEvaluation

    with pytest.raises(ValueError):
        GradeEvaluation.model_validate(_grade_body(job_id=None))
    with pytest.raises(ValueError):
        GradeEvaluation.model_validate({**_grade_body(), "source": "uploads",
                                        "job_id": None, "completion_keys": ["k"]})


def test_the_eval_control_status_is_readable_by_the_platform(admin):
    from reliquary.validator.eval_control import write_control_status

    assert admin("GET", "/admin/v1/eval-control/status").status_code == 404
    document = {"schema": "reliquary/eval-control-status/v1", "updated_at": 1.0,
                "models": {f"{MODEL}@{REVISION}": {"executors_needed": 1, "live_executors": 2,
                                                   "live_providers": 2, "waiting_batches": 3,
                                                   "awaiting_third_scorer": 1}},
                "jobs": {}, "quarantined": [], "stats": {}}
    asyncio.run(write_control_status(document))
    asyncio.run(write_control_status({**document, "updated_at": 2.0}))
    read = admin("GET", "/admin/v1/eval-control/status")
    assert read.status_code == 200 and read.json()["updated_at"] == 2.0
    assert read.json()["models"][f"{MODEL}@{REVISION}"]["executors_needed"] == 1
