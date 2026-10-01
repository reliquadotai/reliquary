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
