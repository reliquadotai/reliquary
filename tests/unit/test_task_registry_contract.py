"""A task may carry its generation contract instead of only pinning its hash.
Entries that do not carry one must be untouched, down to the rendered bytes."""

import json

import pytest

from reliquary.protocol.release_contract import canonical_sha256
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
        "contract": None,
    }
    payload.update(overrides)
    # An honest entry pins the digest of the contract it carries, so tests that
    # are not about the digest rule do not have to restate it.
    if payload["contract"] is not None and "profile_sha256" not in overrides:
        payload["profile_sha256"] = canonical_sha256(payload["contract"])
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


def test_a_carried_contract_must_digest_to_the_hash_the_entry_pins():
    # Spec section 5: `profile_sha256` keeps its exact meaning -- the digest of
    # the generation contract this task runs. An entry that pins one contract
    # and carries another is attesting work nobody signed for.
    entry = _entry(task_id="glm-run", contract=CONTRACT, profile_sha256="a" * 64)
    with pytest.raises(RegistryError) as caught:
        validate_entry(entry)
    assert "glm-run" in str(caught.value)


def test_a_carried_contract_whose_digest_agrees_is_accepted():
    entry = _entry(task_id="glm-run", contract=CONTRACT)
    assert entry.profile_sha256 == canonical_sha256(CONTRACT)
    validate_entry(entry)


def test_a_contract_that_cannot_be_hashed_is_refused():
    # A non-JSON value never reached R2 as bytes, but it must be refused where
    # the message can name the task rather than as a bare TypeError.
    entry = _entry(
        task_id="glm-run", contract={"profile_id": {1, 2}}, profile_sha256="a" * 64,
    )
    with pytest.raises(RegistryError) as caught:
        validate_entry(entry)
    assert "glm-run" in str(caught.value)


def test_a_parsed_contract_is_the_entrys_own_copy():
    # Money-adjacent attestation: the entry must own the mapping its digest was
    # checked against, not share it with whoever handed it over.
    raw = render_registry({"glm-run": _entry(task_id="glm-run", contract=CONTRACT)})
    entry = parse_registry(raw)["glm-run"]
    entry.contract["model_id"] = "someone-elses/model"
    assert parse_registry(raw)["glm-run"].contract["model_id"] == "org/GLM"
