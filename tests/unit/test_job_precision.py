"""A forced-draw job's precision (design 2026-11-13 verification tiers): bf16 is never written, so every manifest
before the field hashes the same; another precision is written with its own judge and refused by a binary that
does not know it; orders may name only the precisions the operator enabled."""
from __future__ import annotations

import pytest

from reliquary.corpus.job import (DEFAULT_DRAW_JUDGE, PRECISION_JUDGES, JobError, parse_job)
from tests.unit.test_eval_draw import _raw


def test_a_job_without_a_precision_is_bf16_and_writes_nothing():
    job = parse_job(_raw())
    assert job.precision == "bf16" and "precision" not in job.to_contract()


def test_an_fp8_job_round_trips_and_bf16_is_never_written():
    job = parse_job({**_raw(), "precision": "fp8"})
    assert job.precision == "fp8" and job.to_contract()["precision"] == "fp8"
    assert parse_job(job.to_contract()) == job
    for bad in ("bf16", "int4", "fp4", 8):
        with pytest.raises(JobError):
            parse_job({**_raw(), "precision": bad})


def test_a_precision_needs_a_draw():
    raw = {**_raw(), "precision": "fp8"}
    raw["sampling"] = {k: v for k, v in raw["sampling"].items() if k != "draw"}
    raw.pop("seed", None)
    with pytest.raises(JobError):
        parse_job(raw)


def test_the_manifest_of_an_fp8_job_carries_its_precisions_judge(monkeypatch):
    from reliquary.cli import main

    monkeypatch.setattr("reliquary.validator.corpus_service.prompt_job_for_spec", lambda job, profile: None)
    common = dict(job_id="j-1", checkpoint_repo="m/x", checkpoint_revision="r", checkpoint_sha256="a" * 64,
                  prompt_source="math", prompt_count=4, renderer_id="chat-template-v1", eos_token_id=1,
                  slots_per_prompt=1, temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64,
                  n=1, grader_id=None, threshold=None, prompt_order="free", deadline_round=None, seed=3,
                  draw="gumbel_max", profile=object())
    plain = main.build_job_manifest(**common)
    assert "precision" not in plain and "judge" not in plain
    assert main.build_job_manifest(**common, precision="bf16") == plain
    fp8 = main.build_job_manifest(**common, precision="fp8")
    assert fp8["precision"] == "fp8" and fp8["judge"] == PRECISION_JUDGES["fp8"].to_contract()
    tuned = main.build_job_manifest(**common, precision="fp8", judge={"gap_tolerance": 5.0})
    assert tuned["judge"]["gap_tolerance"] == 5.0
    assert tuned["judge"]["floor_rate"] == PRECISION_JUDGES["fp8"].floor_rate
    assert PRECISION_JUDGES["bf16"] == DEFAULT_DRAW_JUDGE


def test_orders_name_only_enabled_precisions():
    from reliquary.admin.service import order_precisions

    assert order_precisions({}) == ("bf16",)
    assert order_precisions({"RELIQUARY_ORDER_PRECISIONS": "bf16, fp8"}) == ("bf16", "fp8")
    with pytest.raises(ValueError):
        order_precisions({"RELIQUARY_ORDER_PRECISIONS": "bf16,int4"})


def test_order_audit_params_default_to_the_sampled_share():
    from types import SimpleNamespace

    from reliquary.admin.service import order_audit_params

    assert order_audit_params(env={})["audit_q"] == 0.05
    assert order_audit_params(env={})["audit_probation_submissions"] == 20
    tuned = order_audit_params(env={"RELIQUARY_ORDER_AUDIT_Q": "0.1", "RELIQUARY_ORDER_AUDIT_PROBATION": "40"})
    assert (tuned["audit_q"], tuned["audit_probation_submissions"]) == (0.1, 40)
    full = order_audit_params(SimpleNamespace(audit_q=1.0, audit_probation_submissions=5), env={})
    assert (full["audit_q"], full["audit_probation_submissions"]) == (1.0, 5)
    for bad in ({"RELIQUARY_ORDER_AUDIT_Q": "0"}, {"RELIQUARY_ORDER_AUDIT_PROBATION": "0"}):
        with pytest.raises(ValueError):
            order_audit_params(env=bad)
