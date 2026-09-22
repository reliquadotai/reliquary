"""A task may carry its generation contract instead of only pinning its hash.
Entries that do not carry one must be untouched, down to the rendered bytes."""

import json

import pytest

from reliquary.shared.task_registry import (
    RegistryError,
    TaskEntry,
    parse_registry,
    render_registry,
    validate_entry,
)

PARAMS = {
    "start": 1.0, "decay": 0.98, "rounds_per_step": 1000, "deadband": 0.80,
    "snap": 1.20, "floor": 0.05, "cap": 0.90, "median_rounds": 4800,
    "last_good_fills": 50,
}
CONTRACT = {
    "profile_id": "glm-run",
    "model_id": "org/GLM",
    "model_revision": "abc",
    "protocol_version": 6,
    "prompt_encoding": "chat_template",
    "throughput_tiebreak": None,
    "collection_seconds": 600,
    "upload_grace_seconds": 30,
    "sampling": {
        "rollouts": 16, "temperature": 1.0, "top_p": 1.0,
        "top_k": 0, "do_sample": True,
    },
    "environments": {},
}


def _entry(**overrides):
    payload = {
        "task_id": "legacy",
        "profile_id": "v6",
        "profile_sha256": "a" * 64,
        "mechanism": "rl-discovered-price",
        "params": PARAMS,
        "status": "active",
        "retired_at": None,
    }
    payload.update(overrides)
    return TaskEntry(**payload)


def test_an_entry_without_a_contract_is_unchanged():
    entry = _entry()
    assert entry.contract is None
    validate_entry(entry)


def test_a_legacy_entry_renders_without_a_contract_key():
    # Registry objects already exist in R2; their bytes are the agreement.
    rendered = json.loads(render_registry({"legacy": _entry()}))
    assert "contract" not in rendered["tasks"]["legacy"]


def test_an_entry_may_carry_a_contract():
    entry = _entry(task_id="glm-run", contract=CONTRACT)
    validate_entry(entry)
    assert entry.contract["model_id"] == "org/GLM"


def test_a_carried_contract_round_trips_through_the_registry():
    entry = _entry(task_id="glm-run", contract=CONTRACT)
    revived = parse_registry(render_registry({"glm-run": entry}))
    assert revived["glm-run"].contract == CONTRACT


def test_a_carried_contract_is_rendered_under_its_own_key():
    rendered = json.loads(
        render_registry({"glm-run": _entry(task_id="glm-run", contract=CONTRACT)})
    )
    assert rendered["tasks"]["glm-run"]["contract"] == CONTRACT


def test_both_kinds_of_entry_coexist():
    entries = {
        "legacy": _entry(),
        "glm-run": _entry(task_id="glm-run", contract=CONTRACT, params={**PARAMS, "cap": 0.10}),
    }
    revived = parse_registry(render_registry(entries))
    assert revived["legacy"].contract is None
    assert revived["glm-run"].contract == CONTRACT


@pytest.mark.parametrize("bad", ["", [], 3])
def test_a_contract_that_is_not_an_object_is_refused(bad):
    with pytest.raises(RegistryError):
        validate_entry(_entry(task_id="glm-run", contract=bad))
