"""Agentic eval orders: a forced-draw eval order on a SWE (or TMax) set carries the signed-sandbox episode that
plays its references; the admin checks it against the set and the model's renderer and records it, and the order
control serves the job only when its episode is the recorded one."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from reliquary.eval import qualification as qual
from reliquary.eval.storage import SubnetEvalStore, subnet_key
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import SAMPLING, _pin, pins  # noqa: F401

SWE_SET = "swe-eval-r0-n4"
QWEN35 = "Qwen3_5ForConditionalGeneration"


def _swe_episode(**over):
    from tests.unit.test_corpus_job_signed_sandbox import signed_episode

    episode = signed_episode()
    episode["env"] = {**episode["env"], "split": "eval"}
    episode["max_tokens_per_turn"] = 1024
    episode.update(over)
    return episode


@pytest.fixture
def swe_set(admin, monkeypatch):  # noqa: F811
    """A published SWE set of 4 references in the subnet bucket, its rows checked against a faked bridge."""
    from reliquary.environment import agentic_swe, bridged_swe
    from reliquary.eval import rows as eval_rows

    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda *a, **k: None)
    monkeypatch.setattr(agentic_swe, "load_swe_source", lambda *a, **k: [None] * 100)
    monkeypatch.setattr(bridged_swe, "row_of", lambda split, index: (
        None, SimpleNamespace(instance_id=f"repo__x-{index}")))
    eval_rows._row_map.cache_clear()
    rows = [{"problem_id": f"p{i}", "env": "reliquary-swe", "interaction": "swe", "split": "eval",
             "source_index": i, "instance_id": f"repo__x-{i}", "set_id": SWE_SET} for i in range(4)]
    body = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    card = {"set_id": SWE_SET, "count": 4, "prompts_sha256": hashlib.sha256(body).hexdigest(),
            "interaction": "swe", "source": "reliquary-swe", "env": "reliquary-swe", "split": "eval",
            "source_kind": "catalog", "renderer_id": None}
    store = SubnetEvalStore()
    asyncio.run(store.put_bytes(subnet_key(SWE_SET, "set.json"), json.dumps(card).encode()))
    asyncio.run(store.put_bytes(subnet_key(SWE_SET, "prompts.jsonl"), body))
    yield card
    eval_rows._row_map.cache_clear()


@pytest.fixture
def qwen35(pins):  # noqa: F811
    from tests.unit.test_admin_eval_jobs import MODEL, REVISION

    pins[MODEL] = _pin(MODEL, REVISION, architecture=QWEN35, max_position_embeddings=262144)
    return pins


def _order(**kw):
    from tests.unit.test_admin_eval_jobs import MODEL, REVISION

    base = {"job_id": "order-eval-swe1", "model": MODEL, "revision": REVISION, "env": "reliquary-swe",
            "prompt_count": 3, "samples_per_prompt": 2, "max_new_tokens": 1024, "sampling": SAMPLING,
            "eval_set_id": SWE_SET, "draw": "gumbel_max", "episode": _swe_episode()}
    return {**base, **kw}


def _served(job_id="order-eval-swe1"):
    job, _ = asyncio.run(job_store.read_job(job_id))
    return job, asyncio.run(qual.EvalJobStore().read(job_id))


# -- the admin --------------------------------------------------------------------------------


def test_an_eval_order_on_a_swe_set_declares_its_signed_episode_job(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT

    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 201, response.text
    job, declared = _served()
    assert is_signed_sandbox(job) and job.renderer_id == _swe_episode()["renderer"]
    assert job.episode.env.split == "eval" and job.prompt_count == 3
    assert declared["episode"] == job.episode.to_contract() and declared["thinking"] is None
    entry = admin.registry["entries"]["order-eval-swe1"]
    assert list(entry.contract["environments"]) == [AGENTIC_SWE_ENVIRONMENT]


@pytest.mark.parametrize("change,word", [
    ({"episode": None}, "pass the episode"),
    ({"episode": _swe_episode(env={"package": "reliquary-swe", "version": "a" * 40, "split": "train",
                                   "num_images": 20})}, "split"),
    ({"episode": {k: v for k, v in _swe_episode().items() if k not in ("execution", "sandbox")}},
     "signed sandbox"),
    ({"thinking": True}, "thinking"),
    ({"max_new_tokens": 2048}, "max_tokens_per_turn"),
    ({"draw": None}, "forced-draw"),
])
def test_an_episode_order_unlike_its_set_is_refused(admin, swe_set, qwen35, change, word):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order(**change))
    assert response.status_code == 422 and word in response.text, response.text
    assert "order-eval-swe1" not in admin.registry["entries"]


def test_an_episode_order_on_a_model_without_a_renderer_is_refused(admin, swe_set, pins):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 422 and "episode_renderer_unavailable" in response.text, response.text


def test_a_single_turn_set_takes_no_episode(admin, qwen35):  # noqa: F811
    body = _order(env="logic", eval_set_id="logic-eval-s1-n8", max_new_tokens=512)
    response = admin("POST", "/admin/v1/jobs", body)
    assert response.status_code == 422 and "single-turn" in response.text, response.text


# -- the order control --------------------------------------------------------------------------


def test_the_order_control_serves_the_recorded_episode_only(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal, qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, declared = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    assert order_job_refusal(entry, job) is None
    assert qualification_refusal(entry, job, declared, None) is None
    longer = replace(job, episode=replace(job.episode, max_turns=job.episode.max_turns + 1))
    assert "episode" in qualification_refusal(entry, longer, declared, None)
    assert "episode" in qualification_refusal(entry, job, {**declared, "episode": None}, None)


def test_the_order_control_refuses_an_unsigned_or_unrendered_episode(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.corpus.job import EXECUTION_REPLAY
    from reliquary.validator.eval_control import order_job_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, _ = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    replayed = replace(job, episode=replace(job.episode, execution=EXECUTION_REPLAY, sandbox=None))
    assert "signed sandbox" in order_job_refusal(entry, replayed)
    other = replace(entry, contract={**entry.contract, "model_architecture": "Qwen3ForCausalLM"})
    assert "episode_renderer_unavailable" in order_job_refusal(other, job)


def test_a_qualified_order_never_runs_an_episode(admin, swe_set, qwen35):  # noqa: F811
    from reliquary.validator.eval_control import qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, declared = _served()
    entry = admin.registry["entries"]["order-eval-swe1"]
    qualified = {**declared, "schema": "reliquary/eval-job/v1", "qualification_id": "q1"}
    assert "forced-draw" in qualification_refusal(entry, job, qualified, {"qualification_id": "q1"})
