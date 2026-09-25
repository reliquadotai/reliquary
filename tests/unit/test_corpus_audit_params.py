"""A corpus task declares its audit sampling, probation and hold as task
parameters; absent keys mean V0 (audit everything)."""

from dataclasses import replace

import pytest

from reliquary.shared.task_registry import RegistryError, set_cap, validate_entry
from tests.unit.test_task_set_cap import _entry


def test_audit_parameters_are_validated_with_the_entry():
    entry = _entry("default", 0.9)
    validate_entry(replace(entry, params={**entry.params, "audit_q": 0.1}))
    with pytest.raises(RegistryError):
        validate_entry(replace(entry, params={**entry.params, "audit_q": 0.0}))


def test_set_cap_can_change_audit_q():
    entry = _entry("default", 1.0)
    updated = set_cap({"default": entry}, "default", 0.9, audit_q=0.2)
    assert updated["default"].params["audit_q"] == 0.2


def test_a_corpus_task_declares_its_audit_parameters():
    from reliquary.cli.main import build_corpus_task_entry

    entry = build_corpus_task_entry(
        task_id="corpus-x", job_id="job-x", from_profile="teutonic-9b-reliquary-suite-v9-dev1",
        model_id="org/M", model_revision="r", model_architecture="Qwen3_5ForCausalLM",
        prompt_source="reliquary_dapo_math_v1", cap=0.1, overrides={},
        audit_params={"audit_q": 0.1, "audit_probation_submissions": 100},
    )
    assert entry.params["audit_q"] == 0.1
    validate_entry(entry)
