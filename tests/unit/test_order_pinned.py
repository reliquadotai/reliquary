"""Forced-draw orders without a qualification: one order object for eval and generation, on a frozen set or an
env range, its model pinned from its own files (``orders.models``) and checked by the order control against the
order record the admin wrote."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace

import pytest

from reliquary.eval import qualification as qual
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.orders import models
from tests.unit.test_admin_eval_jobs import MODEL, REVISION, admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_any_model import GEN_ENV, stub_catalog_env

SET_ID = "logic-eval-s1-n8"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
PIN_SHA = "f" * 64


def _pin(model=MODEL, revision=REVISION, **kw):
    return models.ModelPin(**{"model": model, "revision": revision, "architecture": "Qwen3ForCausalLM",
                              "eos_token_id": 151645, "checkpoint_sha256": PIN_SHA, "checkpoint_bytes": 16 * 10**9,
                              "max_position_embeddings": 40960, **kw})


@pytest.fixture
def pins(monkeypatch):
    """What the admin pins per model: a pin, or the error pin_model raises."""
    table: dict = {}

    def pin(repo, revision):
        found = table.get(repo, _pin(repo, revision))
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(models, "pin_model", pin)
    return table


def _order(kind="eval", **kw):
    base = {"job_id": f"order-{kind}-p1", "model": MODEL, "revision": REVISION, "env": "logic",
            "prompt_count": 5, "samples_per_prompt": 4, "max_new_tokens": 512, "thinking": False,
            "sampling": SAMPLING, "eval_set_id": SET_ID, "draw": "gumbel_max"}
    return {**base, **kw}


# -- the pin -----------------------------------------------------------------------


def test_the_hub_fingerprint_is_the_one_a_gpu_computes_over_the_shards(tmp_path):
    from reliquary.corpus.encoding import checkpoint_fingerprint

    files = {"model-00002-of-00002.safetensors": b"second", "model-00001-of-00002.safetensors": b"first"}
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    hub = [(name, hashlib.sha256(body).hexdigest()) for name, body in files.items()]
    assert models.fingerprint_of(hub) == checkpoint_fingerprint(tmp_path)


def _pin_with(facts=None, shards=None, config=None, revision=REVISION):
    return models.pin_model(
        MODEL, revision, facts=lambda r, v: facts or {"architecture": "Qwen3ForCausalLM", "eos_token_id": 7},
        shards=lambda r, v: shards or [("model.safetensors", "a" * 64, 1000)],
        config=lambda r, v: config if config is not None else {"text_config": {"max_position_embeddings": 32768}})


def test_a_pin_reads_architecture_eos_fingerprint_and_context():
    # Only the top-level shards are hashed, as a GPU hashes the downloaded directory.
    extra = _pin_with(shards=[("model.safetensors", "a" * 64, 1000), ("sub/model.safetensors", None, 5),
                              ("README.md", None, 7)])
    assert extra.checkpoint_sha256 == models.fingerprint_of([("model.safetensors", "a" * 64)])
    pin = _pin_with()
    assert (pin.architecture, pin.eos_token_id, pin.checkpoint_bytes) == ("Qwen3ForCausalLM", 7, 1000)
    assert pin.checkpoint_sha256 == models.fingerprint_of([("model.safetensors", "a" * 64)])
    assert pin.max_position_embeddings == 32768
    assert pin.to_dict()["schema"] == models.PIN_SCHEMA


@pytest.mark.parametrize("kw,word", [
    ({"facts": {"architecture": "MambaForCausalLM", "eos_token_id": 1}}, "architecture_unsupported"),
    ({"shards": [("model.safetensors", "a" * 64, models.MAX_CHECKPOINT_BYTES + 1)]}, "model_too_large"),
    ({"revision": "main"}, "40-hex"),
    ({"shards": [("model.safetensors", "a" * 64, 1000), ("original/consolidated.pth", None, models.MAX_REPO_BYTES)]},
     "repo_too_large"),
])
def test_a_pin_outside_the_preset_is_refused(kw, word):
    with pytest.raises(ValueError, match=word):
        _pin_with(**kw)


# -- the admin: one path for eval and generation -------------------------------------


def test_an_eval_order_needs_no_qualification(admin, pins):  # noqa: F811
    created = admin("POST", "/admin/v1/jobs", _order())
    assert created.status_code == 201, created.text
    job, _ = asyncio.run(job_store.read_job("order-eval-p1"))
    assert (job.checkpoint_revision, job.checkpoint_sha256, job.eos_token_id) == (REVISION, PIN_SHA, 151645)
    assert job.sampling.draw == "gumbel_max" and job.seed is not None
    assert admin.registry["entries"]["order-eval-p1"].params["audit_q"] == 1.0
    record = asyncio.run(qual.EvalJobStore().read("order-eval-p1"))
    assert record["schema"] == qual.ORDER_JOB_SCHEMA_V2 and record["kind"] == qual.EVALUATION
    assert record["pin"]["checkpoint_sha256"] == PIN_SHA and "qualification_id" not in record
    assert record["source"] == {"set_id": SET_ID, "problems": 5}
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 200


def test_a_batch_on_a_frozen_set_is_the_same_order(admin, pins):  # noqa: F811
    created = admin("POST", "/admin/v1/jobs", _on_upload(admin, seed=11))
    assert created.status_code == 201, created.text
    job, _ = asyncio.run(job_store.read_job("order-gen-p1"))
    assert admin("POST", "/admin/v1/jobs", _on_upload(admin, "eval", seed=11)).status_code == 201
    evaluation, _ = asyncio.run(job_store.read_job("order-eval-p1"))
    # The same prompts, model, sampling and draw key: the same samples; only audit share, route and grading differ.
    same = ("prompt_source", "checkpoint_sha256", "seed", "slots_per_prompt", "renderer_id")
    assert {k: getattr(job, k) for k in same} == {k: getattr(evaluation, k) for k in same}
    assert job.sampling == evaluation.sampling
    assert job.submit == "scoped"
    assert admin.registry["entries"]["order-gen-p1"].params["audit_q"] == 0.15
    record = asyncio.run(qual.EvalJobStore().read("order-gen-p1"))
    assert record["kind"] == qual.GENERATION and record["seed"] == 11


def test_a_batch_never_draws_from_our_held_out_sets(admin, pins):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order("gen"))
    assert response.status_code == 422 and "upload" in response.text


def test_an_upload_longer_than_the_job_allows_is_refused(admin, pins):  # noqa: F811
    set_id = _upload_set(admin, graded=False, prompt_tokens_max=20_000)
    body = _order("gen", eval_set_id=set_id, prompt_count=3, env="upload")
    refused = admin("POST", "/admin/v1/jobs", body)
    assert refused.status_code == 422 and "max_prompt_tokens_too_small" in refused.text
    assert admin("POST", "/admin/v1/jobs", {**body, "max_prompt_tokens": 20_000}).status_code == 201


def test_a_batch_on_an_env_range(admin, pins, monkeypatch):  # noqa: F811
    stub_catalog_env(monkeypatch)
    body = _order("gen", env=GEN_ENV, prompt_start=100, prompt_count=50, eval_set_id=None)
    created = admin("POST", "/admin/v1/jobs", body)
    assert created.status_code == 201, created.text
    job, _ = asyncio.run(job_store.read_job("order-gen-p1"))
    assert (job.prompt_source, job.prompt_start, job.prompt_count) == (GEN_ENV, 100, 50)
    record = asyncio.run(qual.EvalJobStore().read("order-gen-p1"))
    assert record["source"]["env"] == GEN_ENV and record["source"]["environment_manifest_sha256"]


@pytest.mark.parametrize("change,status,word", [
    ({"revision": None}, 422, "revision"),
    ({"audit_q": 0.5}, 422, "audit_q"),
    ({"eval_set_id": None}, 422, "eval_set_id"),
    ({"prompt_count": 9}, 422, "holds"),
    ({"prompt_start": 1}, 422, "first problem"),
    ({"eval_set_id": "logic-eval-s9-n9"}, 404, "set_unknown"),
    ({"max_new_tokens": 40000, "max_prompt_tokens": 2000}, 422, "context_too_long"),
    ({"max_new_tokens": 40000}, 422, "context_too_long"),
    ({"sampling": None}, 422, "sampling"),
])
def test_pinned_order_refusals(admin, pins, change, status, word):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order(**change))
    assert response.status_code == status and word in response.text, response.text


def test_a_model_outside_the_preset_is_refused(admin, pins):  # noqa: F811
    pins[MODEL] = ValueError("architecture_unsupported: 'MambaForCausalLM'")
    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 422 and "architecture_unsupported" in response.text
    pins[MODEL] = ConnectionError("hub down")
    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 503


def test_a_revision_is_only_for_a_pinned_order(admin, pins):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order(draw=None))
    assert response.status_code == 422


def _upload_set(admin, *, graded, model=MODEL, revision=REVISION, sampling=None, prompt_tokens_max=3):
    from reliquary.eval.storage import SubnetEvalStore, subnet_key

    rows = [{"problem_id": f"r-{i}", "custom_id": f"r-{i}", "prompt_token_ids": [1, 2, 3 + i]} for i in range(3)]
    body = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)
    card = {"schema": "reliquary/eval-set/v1", "set_id": "up-0123456789abcdef01234567", "source_kind": "upload",
            "source": "upload", "env": None, "count": 3, "model": model, "revision": revision, "graded": graded,
            "sampling": sampling or {}, "prompt_tokens_max": prompt_tokens_max}
    store = SubnetEvalStore()

    async def put():
        await store.put_bytes(subnet_key(card["set_id"], "prompts.jsonl"), body)
        await store.put_bytes(subnet_key(card["set_id"], "set.json"), json.dumps(card).encode())

    asyncio.run(put())
    return card["set_id"]


def test_an_upload_set_orders_as_any_set(admin, pins):  # noqa: F811
    from reliquary.protocol.external_eval import EXTERNAL_EVAL_ENVIRONMENT

    set_id = _upload_set(admin, graded=False)
    body = _order("gen", eval_set_id=set_id, prompt_count=3, env="upload")
    created = admin("POST", "/admin/v1/jobs", body)
    assert created.status_code == 201, created.text
    assert list(admin.registry["entries"]["order-gen-p1"].contract["environments"]) == [EXTERNAL_EVAL_ENVIRONMENT]
    # An evaluation needs references; another model's rendering is refused.
    refused = admin("POST", "/admin/v1/jobs", {**body, "job_id": "order-eval-p2"})
    assert refused.status_code == 422 and "references" in refused.text
    pins["other/Model"] = _pin("other/Model")
    other = admin("POST", "/admin/v1/jobs", {**body, "job_id": "order-gen-p3", "model": "other/Model"})
    assert other.status_code == 422 and "another model" in other.text


def _on_upload(admin, kind="gen", **kw):  # noqa: F811
    set_id = _upload_set(admin, graded=True)
    return _order(kind, eval_set_id=set_id, prompt_count=3, env="upload", **kw)


# -- the order control checks the job against the record ------------------------------


def _served(job_id):
    job, _ = asyncio.run(job_store.read_job(job_id))
    declared = asyncio.run(qual.EvalJobStore().read(job_id))
    return job, declared


@pytest.mark.parametrize("kind", ["eval", "gen"])
def test_the_order_control_serves_a_pinned_order(admin, pins, kind):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal, qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _on_upload(admin, kind)).status_code == 201
    job_id = f"order-{kind}-p1"
    job, declared = _served(job_id)
    entry = admin.registry["entries"][job_id]
    assert order_job_refusal(entry, job) is None
    assert qualification_refusal(entry, job, declared, None) is None


@pytest.mark.parametrize("change,word", [
    ({"max_prompt_tokens": 131072}, "max_prompt_tokens"),
    ({"prompt_start": 1}, "source"),
    ({"checkpoint_sha256": "0" * 64}, "fingerprint"),
    ({"eos_token_id": 2}, "eos"),
    ({"seed": 99}, "seed"),
    ({"slots_per_prompt": 8}, "samples"),
    ({"checkpoint_revision": "0" * 40}, "another model"),
])
def test_the_order_control_refuses_a_job_unlike_its_record(admin, pins, change, word):  # noqa: F811
    from reliquary.validator.eval_control import qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _on_upload(admin)).status_code == 201
    job, declared = _served("order-gen-p1")
    entry = admin.registry["entries"]["order-gen-p1"]
    assert word in qualification_refusal(entry, replace(job, **change), declared, None)


def test_the_order_control_refuses_a_changed_sampling_or_kind(admin, pins):  # noqa: F811
    from reliquary.validator.eval_control import qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _on_upload(admin)).status_code == 201
    job, declared = _served("order-gen-p1")
    entry = admin.registry["entries"]["order-gen-p1"]
    hotter = replace(job, sampling=replace(job.sampling, temperature=1.0))
    assert "sampling" in qualification_refusal(entry, hotter, declared, None)
    assert "kind" in qualification_refusal(entry, job, {**declared, "kind": qual.EVALUATION}, None)
    free = replace(job, sampling=replace(job.sampling, draw=None))
    assert "forced-draw" in qualification_refusal(entry, free, declared, None)


def test_a_generation_order_on_a_set_needs_a_draw(admin, pins):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal

    assert admin("POST", "/admin/v1/jobs", _on_upload(admin)).status_code == 201
    job, _ = _served("order-gen-p1")
    entry = admin.registry["entries"]["order-gen-p1"]
    free = replace(job, sampling=replace(job.sampling, draw=None))
    assert "forced draw" in order_job_refusal(entry, free)


# -- grading reads the order, not a qualification --------------------------------------


def test_a_pinned_eval_is_graded_with_its_draw_as_verification(admin, pins):  # noqa: F811
    import time

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    grading = [json.loads(l) for l in (admin.root / "set" / "grading.jsonl").read_text().splitlines()]
    index = [g["source_index"] for g in grading]
    for prompt in range(5):
        for k in range(4):
            sid = f"{prompt * 4 + k:064x}"
            admin.records.subs[sid] = {"prompt_index": prompt, "hotkey": "hk1", "completions": [
                {"text": f'```json\n{{"a": "={index[prompt]}"}}\n```', "tokens": [5, 2]}]}
            admin.records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": "hk1"}
            admin.records.settled.append(sid)
    body = {"source": "job", "job_id": "order-eval-p1", "set_ids": [SET_ID], "problems_per_set": {SET_ID: 5},
            "samples_per_set": {SET_ID: 4}, "allow_incomplete": True}
    for _ in range(200):
        response = admin("POST", "/admin/v1/evaluations/order-ep1/grade", body)
        if response.status_code != 202:
            break
        time.sleep(0.02)
    assert response.status_code == 200, response.text
    report = json.loads((admin.root / "p" / "evaluations" / "order-ep1" / "report.json").read_text())
    verification = report["provenance"]["verification"]
    assert verification["scheme"] == "gumbel_max" and verification["sampling_verified"] is True
    assert verification["source"] == "order" and verification["pin"]["checkpoint_sha256"] == PIN_SHA
    assert verification["judge"]["floor_rate"] == 0.98


def test_an_order_on_an_upload_group_runs_its_lines_sampling(admin, pins):  # noqa: F811
    asked = {**SAMPLING, "min_p": 0.0, "max_tokens": 512, "seed": 11, "n": 4}
    set_id = _upload_set(admin, graded=False, sampling=asked)
    body = _order("gen", eval_set_id=set_id, prompt_count=3, env="upload", seed=11)
    assert admin("POST", "/admin/v1/jobs", body).status_code == 201
    for change, key in (({"seed": 12}, "seed"), ({"samples_per_prompt": 2}, "n"),
                        ({"max_new_tokens": 256}, "max_tokens"),
                        ({"sampling": {**SAMPLING, "temperature": 1.0}}, "temperature")):
        refused = admin("POST", "/admin/v1/jobs", {**body, "job_id": "order-gen-p9", **change})
        assert refused.status_code == 422 and "upload_sampling_differs" in refused.text and key in refused.text


def test_an_eval_on_a_customer_file_is_graded_by_its_references(admin, pins):  # noqa: F811
    import time

    from reliquary.eval.storage import SubnetEvalStore, subnet_key
    from reliquary.orders.uploads import compile_upload
    from tests.unit.test_order_uploads import TOK, completion

    lines = [{**completion(f"q{i}", f"{i}+{i}=", model=MODEL), "reference": str(2 * i)} for i in range(1, 4)]
    body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    (prompt_set,) = compile_upload(body, tokenizer=TOK, vocab_size=200_000, model=MODEL, revision=REVISION,
                                   purpose="eval", grader="boxed_math_v1").sets
    store = SubnetEvalStore()

    async def publish():
        for name, data in prompt_set.files().items():
            await store.put_bytes(subnet_key(prompt_set.set_id, name), data)

    asyncio.run(publish())
    order = _order(eval_set_id=prompt_set.set_id, env="upload", prompt_count=3, samples_per_prompt=1,
                   sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0})
    created = admin("POST", "/admin/v1/jobs", order)
    assert created.status_code == 201, created.text
    answers = ["\\boxed{2}", "\\boxed{5}", "so \\boxed{6}"]
    for prompt, text in enumerate(answers):
        sid = f"{prompt:064x}"
        admin.records.subs[sid] = {"prompt_index": prompt, "hotkey": "hk1",
                                   "completions": [{"text": text, "tokens": [5, 2]}]}
        admin.records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": "hk1"}
        admin.records.settled.append(sid)
    grade = {"source": "job", "job_id": "order-eval-p1", "set_ids": [prompt_set.set_id],
             "problems_per_set": {prompt_set.set_id: 3}, "samples_per_set": {prompt_set.set_id: 1}}
    for _ in range(200):
        response = admin("POST", "/admin/v1/evaluations/order-ep2/grade", grade)
        if response.status_code != 202:
            break
        time.sleep(0.02)
    assert response.status_code == 200, response.text
    report = json.loads((admin.root / "p" / "evaluations" / "order-ep2" / "report.json").read_text())
    (scores,) = report["envs"].values()
    assert scores["pass@1"]["value"] == pytest.approx(2 / 3)
    assert report["provenance"]["verification"]["sampling_verified"] is True


def test_an_eval_order_on_several_sets_has_one_report(admin, pins):  # noqa: F811
    from reliquary.corpus.delivery import LocalDirectorySink

    platform = LocalDirectorySink(admin.root / "p")

    def part(eval_id, env, value, complete=True):
        report = {"complete": complete, "envs": {env: {"pass@1": {"value": value}, "pass@k": {"1": {"value": value}},
                                                       "no_graded_rows": False}},
                  "provenance": {"job_id": f"order-eval-{eval_id}"}}
        asyncio.run(platform.put_json(f"evaluations/{eval_id}/report.json", report))
        asyncio.run(platform.put_json(f"evaluations/{eval_id}/manifest.json", {"eval_id": eval_id}))

    part("order-e1", "math", 0.5)
    part("order-e8", "math", 0.5)
    report8 = json.loads((admin.root / "p" / "evaluations" / "order-e8" / "report.json").read_text())
    report8["provenance"]["checkpoint_sha256"] = "0" * 64
    asyncio.run(platform.put_json("evaluations/order-e8/report.json", report8))
    mixed = admin("POST", "/admin/v1/evaluations/order-mix/combine", {"parts": ["order-e1", "order-e8"]})
    assert mixed.status_code == 409 and "different models" in mixed.text
    part("order-e2", "logic", 0.25)
    part("order-e3", "math", 1.0, complete=False)
    assert admin("POST", "/admin/v1/evaluations/order-all/combine",
                 {"parts": ["order-e1", "order-x9"]}).status_code == 409
    response = admin("POST", "/admin/v1/evaluations/order-all/combine",
                     {"parts": ["order-e1", "order-e2", "order-e3"]})
    assert response.status_code == 200, response.text
    assert response.json()["complete"] is False
    report = json.loads((admin.root / "p" / "evaluations" / "order-all" / "report.json").read_text())
    assert set(report["envs"]) == {"math@order-e1", "logic", "math@order-e3"}
    assert report["macro"]["pass@1"] == pytest.approx((0.5 + 0.25 + 1.0) / 3)
    assert [p["eval_id"] for p in report["parts"]] == ["order-e1", "order-e2", "order-e3"]
    # The same parts again: as stored; others under the same id: refused.
    assert admin("POST", "/admin/v1/evaluations/order-all/combine",
                 {"parts": ["order-e1", "order-e2", "order-e3"]}).status_code == 200
    assert admin("POST", "/admin/v1/evaluations/order-all/combine", {"parts": ["order-e1"]}).status_code == 409
    assert admin("POST", "/admin/v1/evaluations/order-e1/combine", {"parts": ["order-e1"]}).status_code == 422


def test_the_order_control_compares_the_whole_judge_and_audit_share(admin, pins):  # noqa: F811
    from reliquary.corpus.job import DrawJudge
    from reliquary.validator.eval_control import qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _on_upload(admin, "eval")).status_code == 201
    job, declared = _served("order-eval-p1")
    entry = admin.registry["entries"]["order-eval-p1"]
    assert declared["judge"]["floor_rate"] == 0.98
    lax = replace(job, judge=DrawJudge(0.5, 64, 64.0, 64.0, 0.5))
    assert "judge" in qualification_refusal(entry, lax, declared, None)
    many = replace(job, sampling=replace(job.sampling, n=8))
    assert "n" in qualification_refusal(entry, many, declared, None)
    sampled = replace(entry, params={**entry.params, "audit_q": 0.15})
    assert "audit_q" in qualification_refusal(sampled, job, declared, None)
    assert "every submission" in qualification_refusal(sampled, job, {**declared, "audit_q": 0.15}, None)
    elsewhere = replace(entry, contract={**entry.contract, "environments": {"reliquary_logic_v2": {}}})
    assert "environments" in qualification_refusal(elsewhere, job, declared, None)
