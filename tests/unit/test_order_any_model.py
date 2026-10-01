"""Dataset orders on any model (design 2026-10-01): generation jobs
(`${prefix}gen-`) served by the GPU-less order control beside eval jobs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401


# -- 1. the prefixes and the corpus control --------------------------------------


def test_the_generation_prefix_follows_the_admin_task_prefix(monkeypatch):
    from reliquary.eval.prompt_source import (
        gen_job_prefix,
        is_gen_job_id,
        is_order_job_id,
    )

    monkeypatch.delenv("RELIQUARY_ADMIN_TASK_PREFIX", raising=False)
    assert gen_job_prefix() == "order-gen-" and is_gen_job_id("order-gen-1")
    assert is_order_job_id("order-gen-1") and is_order_job_id("order-eval-1")
    assert not is_order_job_id("order-7") and not is_order_job_id("code-qwen38-27b-v1")
    monkeypatch.setenv("RELIQUARY_ADMIN_TASK_PREFIX", "acme-")
    assert gen_job_prefix() == "acme-gen-" and not is_gen_job_id("order-gen-1")
    assert is_order_job_id("acme-gen-2") and is_order_job_id("acme-eval-2")
    assert gen_job_prefix("x-") == "x-gen-"


@pytest.mark.parametrize("job_id", ["order-gen-1", "order-eval-1"])
def test_the_corpus_control_screens_every_order_entry(job_id):
    from reliquary.validator.corpus_hot_jobs import (
        OTHER_MODEL,
        eval_entry_screen,
        hot_job_refusal,
        order_entry_screen,
    )

    entry = SimpleNamespace(job_id=job_id, contract={})
    assert order_entry_screen(entry)[0] == OTHER_MODEL
    # The old name is the same screen.
    assert eval_entry_screen is order_entry_screen
    refusal = hot_job_refusal(entry, None, process_profile=None, process_contract={},
                              fingerprint="")
    assert refusal is not None and refusal[0] == OTHER_MODEL
    assert order_entry_screen(SimpleNamespace(job_id="code-qwen38-27b-v1")) is None


@pytest.mark.parametrize("job_id", ["order-gen-1", "order-eval-1"])
def test_the_corpus_control_refuses_an_order_task_at_boot(job_id):
    from reliquary.validator.corpus_validator import run_corpus_validator

    entry = SimpleNamespace(task_id=job_id, job_id=job_id, status="active",
                            mechanism="corpus-generation", params={"cap": 0.02}, contract={})
    with pytest.raises(RuntimeError, match="order control"):
        asyncio.run(run_corpus_validator(
            entry=entry, cap=0.02, wallet=None, netuid=0, signer_client=None,
            http_host="127.0.0.1", http_port=0, set_weights=False, registration_gate=False))


def test_slot_reopening_stays_eval_only():
    from reliquary.validator.corpus_service import record_prompt_failure
    from tests.unit.test_corpus_export import _job_spec

    calls = []

    class _Store:
        async def read_ledgers(self, job_id):
            calls.append(job_id)
            return None, None

    gen = _job_spec(job_id="order-gen-1", prompt_source="reliquary_logic_v2")
    with pytest.raises(ValueError, match="eval"):
        asyncio.run(record_prompt_failure(_Store(), gen, 0, "a" * 64))
    assert calls == []


# -- 5. supported architectures --------------------------------------------------


def test_every_supported_architecture_names_evidence_that_exists():
    from pathlib import Path

    from reliquary.constants import SUPPORTED_ARCHITECTURES, SUPPORTED_MODEL_ARCHITECTURES

    root = Path(__file__).resolve().parents[2]
    assert set(SUPPORTED_ARCHITECTURES) == {"Qwen3ForCausalLM",
                                            "Qwen3_5ForConditionalGeneration"}
    # One table: the startup check reads the same names.
    assert SUPPORTED_MODEL_ARCHITECTURES == frozenset(SUPPORTED_ARCHITECTURES)
    for architecture, evidence in SUPPORTED_ARCHITECTURES.items():
        assert evidence, architecture
        for reference in evidence:
            path = reference.split("::")[0]
            assert (root / path).exists(), (architecture, reference)
            if "::" in reference:
                assert reference.split("::")[1] in (root / path).read_text(), reference


def _queue_with(store, facts):
    from tests.unit.test_eval_qualification import _queue

    return _queue(store, [0.0], facts=facts)


def test_an_unsupported_architecture_is_refused_before_any_executor_is_leased(monkeypatch):
    from reliquary.eval import qualification as qual
    from tests.unit.test_eval_qualification import _executor, _request
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from reliquary.infrastructure import corpus_job_store as job_store

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    store = qual.QualificationStore()
    queue = _queue_with(store, {"architecture": "MambaForCausalLM", "eos_token_id": 2})
    _request(store)
    asyncio.run(queue.refresh())
    assert asyncio.run(queue.claim(_executor("e1"))) is None
    record, _ = asyncio.run(store.read("order-q1"))
    assert record["status"] == qual.REFUSED and record["leases"] == {}
    assert record["result"]["refused_reason"] == "architecture_unsupported"
    assert record["result"]["architecture"] == "MambaForCausalLM"
    assert "Qwen3ForCausalLM" in record["result"]["supported_architectures"]


def test_the_admin_serves_the_architecture_table(admin):  # noqa: F811
    from reliquary.constants import SUPPORTED_ARCHITECTURES

    response = admin("GET", "/admin/v1/architectures")
    assert response.status_code == 200
    body = response.json()
    assert body == {"schema": "reliquary/architectures/v1",
                    "architectures": sorted(SUPPORTED_ARCHITECTURES),
                    "evidence": {k: list(v) for k, v in SUPPORTED_ARCHITECTURES.items()}}


def test_a_job_on_a_record_of_an_unsupported_architecture_is_refused(admin):  # noqa: F811
    from reliquary.eval import qualification as qual
    from tests.unit.test_admin_eval_jobs import _eval_job, _qualification, _qualify

    admin("POST", "/admin/v1/qualifications", _qualification())
    _qualify(admin)

    async def rewrite():
        store = qual.QualificationStore()
        record, etag = await store.read("order-q1")
        record["result"]["architecture"] = "MambaForCausalLM"
        await store.write(record, etag)

    asyncio.run(rewrite())
    response = admin("POST", "/admin/v1/jobs", _eval_job())
    assert response.status_code == 409 and "architecture_unsupported" in response.text


# -- 3. generation qualifications --------------------------------------------------

GEN_ENV = "reliquary_logic_v2"
GEN_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}


class _CatalogSpec:
    """A packaged single-turn source whose rows are cheap: the real spec's
    declarations, a fake build (``row k`` reads ``<env> problem k``)."""

    def __init__(self, spec, rows):
        self._spec, self._rows = spec, rows

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def create(self):
        from tests.unit.test_eval_sets import FakeEnvironment

        return FakeEnvironment(self._spec.name, self._rows)


def stub_catalog_env(monkeypatch, env=GEN_ENV, rows=1000):
    from reliquary.validator import corpus_service

    specs = corpus_service.ENVIRONMENT_SPECS
    monkeypatch.setattr(corpus_service, "ENVIRONMENT_SPECS",
                        {**specs, env: _CatalogSpec(specs[env], rows)})


def _gen_qualification(**kw):
    return {"qualification_id": "order-gq1", "model": "customer/Gen-8B",
            "revision": "e" * 40, "env": GEN_ENV, "prompt_start": 100, "problems": 500,
            "sampling": GEN_SAMPLING, "max_new_tokens": 1024, "thinking": True, **kw}


def test_a_generation_qualification_binds_the_env_range_and_the_conditions(admin, monkeypatch):  # noqa: F811
    from reliquary.eval import qualification as qual
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

    stub_catalog_env(monkeypatch)
    created = admin("POST", "/admin/v1/qualifications", _gen_qualification())
    assert created.status_code == 201, created.text
    record = created.json()
    assert record["kind"] == "generation" and "set_id" not in record
    assert (record["env"], record["prompt_start"], record["problems"]) == (GEN_ENV, 100, 500)
    assert record["completions"] == qual.QUALIFY_COMPLETIONS == 32
    assert record["environment_manifest_sha256"] == \
        ENVIRONMENT_CATALOG[GEN_ENV].environment_manifest_sha256
    assert (record["sampling"], record["max_new_tokens"], record["thinking"]) == (
        GEN_SAMPLING, 1024, True)
    again = admin("POST", "/admin/v1/qualifications", _gen_qualification())
    assert again.status_code == 200
    other = admin("POST", "/admin/v1/qualifications", _gen_qualification(prompt_start=0))
    assert other.status_code == 409


@pytest.mark.parametrize("change", [
    {"env": "openmathinstruct"},                 # renders through the process profile
    {"env": "reliquary_stateful_tools_v2"},      # an episode source
    {"env": "nope"},
    {"set_id": "logic-eval-s1-n8"},              # both an eval set and an env
    {"env": None},                               # neither
    {"prompt_start": 2_381_806},                 # code's held-out tail, on code
])
def test_generation_qualification_refusals(admin, monkeypatch, change):  # noqa: F811
    stub_catalog_env(monkeypatch)
    body = _gen_qualification(**change)
    if change.get("prompt_start"):
        body["env"] = "reliquary_code_v1"
    body = {k: v for k, v in body.items() if v is not None}
    response = admin("POST", "/admin/v1/qualifications", body)
    assert response.status_code == 422, response.text


def test_a_generation_qualification_is_measured_on_the_first_prompts_of_its_range(monkeypatch):
    from reliquary.eval import qualification as qual
    from reliquary.infrastructure import corpus_job_store as job_store
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from tests.unit.test_eval_qualification import _executor

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    stub_catalog_env(monkeypatch)
    store = qual.QualificationStore()
    asyncio.run(store.write(qual.new_generation_request(
        qualification_id="order-gq1", model="m", revision="r" * 40, env=GEN_ENV,
        prompt_start=100, problems=500, sampling=GEN_SAMPLING, max_new_tokens=64,
        thinking=False, clock=lambda: 0.0), None))

    async def facts(repo, revision):
        return {"architecture": "Qwen3ForCausalLM", "eos_token_id": 2}

    async def no_set(set_id):
        raise AssertionError("a generation qualification reads no eval set")

    queue = qual.QualificationQueue(store=store, read_prompts=no_set, model_facts=facts,
                                    clock=lambda: 0.0)
    asyncio.run(queue.refresh())
    lease = asyncio.run(queue.claim(_executor("e1", model="m")))
    assert [p["problem_id"] for p in lease["prompts"]] == [
        f"{GEN_ENV}#{k}" for k in range(100, 132)]
    assert lease["prompts"][0]["text"] == f"{GEN_ENV} problem 100"
    record, _ = asyncio.run(store.read("order-gq1"))
    sample = record["sample_sha256"]
    # The second qualifier is leased exactly the same sample.
    second = asyncio.run(queue.claim(_executor("e2", model="m")))
    assert second["prompts"] == lease["prompts"]
    assert asyncio.run(store.read("order-gq1"))[0]["sample_sha256"] == sample
