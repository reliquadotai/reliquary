"""`tasks create` can seed a contract from a compiled profile and override the
model and the environment set — the two things an operator actually chooses."""

import pytest

from reliquary.cli.main import build_contract_task_entry
from reliquary.environment.abi import canonical_sha256
from reliquary.protocol.profiles import PROFILES, profile_from_contract


def _template():
    # A template with at least two environments, so the subset test is real.
    for profile_id in sorted(PROFILES):
        if len(PROFILES[profile_id].environments) >= 2:
            return profile_id
    pytest.skip("no compiled profile declares two environments")


def test_the_entry_carries_a_contract_whose_digest_is_its_own():
    template = _template()
    entry = build_contract_task_entry(
        task_id="glm-run",
        from_profile=template,
        model_id="org/GLM",
        model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=None,
        cap=0.30,
        overrides={},
    )
    assert entry.contract is not None
    assert entry.profile_sha256 == canonical_sha256(entry.contract)
    assert entry.params["cap"] == 0.30


def test_the_model_is_overridden_and_nothing_else_is():
    template = _template()
    source = PROFILES[template]
    entry = build_contract_task_entry(
        task_id="glm-run", from_profile=template,
        model_id="org/GLM", model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=None, cap=0.30, overrides={},
    )
    rebuilt = profile_from_contract(entry.contract)
    assert rebuilt.model_id == "org/GLM"
    assert rebuilt.model_revision == "abc123"
    assert rebuilt.sampling == source.sampling
    assert set(rebuilt.environments) == set(source.environments)


def test_a_subset_of_environments_can_be_selected():
    template = _template()
    keep = sorted(PROFILES[template].environments)[:1]
    entry = build_contract_task_entry(
        task_id="glm-run", from_profile=template,
        model_id="org/GLM", model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=keep, cap=0.30, overrides={},
    )
    assert sorted(profile_from_contract(entry.contract).environments) == keep


def test_an_environment_the_template_does_not_declare_is_refused():
    with pytest.raises(ValueError) as caught:
        build_contract_task_entry(
            task_id="glm-run", from_profile=_template(),
            model_id="org/GLM", model_revision="abc123",
            model_architecture="Qwen3ForCausalLM",
            environments=["not-in-the-template"], cap=0.30, overrides={},
        )
    assert "not-in-the-template" in str(caught.value)


def test_an_empty_environment_selection_is_refused():
    with pytest.raises(ValueError):
        build_contract_task_entry(
            task_id="glm-run", from_profile=_template(),
            model_id="org/GLM", model_revision="abc123",
            model_architecture="Qwen3ForCausalLM",
            environments=[], cap=0.30, overrides={},
        )


def test_the_task_id_becomes_the_contract_profile_id():
    # Two tasks seeded from one template must not share a profile id, or the
    # existing profile_id check stops distinguishing them.
    entry = build_contract_task_entry(
        task_id="glm-run", from_profile=_template(),
        model_id="org/GLM", model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=None, cap=0.30, overrides={},
    )
    assert entry.profile_id == "glm-run"
    assert entry.contract["profile_id"] == "glm-run"


# --- `model_architecture` is now a required, declared field: the CLI cannot
# infer it (that means fetching an arbitrary HF repo's config), so the
# operator states it, and it rides in the contract untouched. ---

def test_the_contract_carries_the_named_architecture():
    entry = build_contract_task_entry(
        task_id="glm-run", from_profile=_template(),
        model_id="org/GLM", model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=None, cap=0.30, overrides={},
    )
    assert entry.contract["model_architecture"] == "Qwen3ForCausalLM"


def test_the_builder_does_not_validate_architecture_against_any_supported_set():
    # That refusal belongs at startup, in `resolve_task_config`, against the
    # image's own capability list -- a builder-side check here would duplicate
    # it in a place that cannot be kept in sync. Pin the builder's silence.
    entry = build_contract_task_entry(
        task_id="glm-run", from_profile=_template(),
        model_id="org/GLM", model_revision="abc123",
        model_architecture="TotallyMadeUpArchitectureForCausalLM",
        environments=None, cap=0.30, overrides={},
    )
    assert (
        entry.contract["model_architecture"]
        == "TotallyMadeUpArchitectureForCausalLM"
    )


def test_a_model_without_an_architecture_is_refused_by_the_command(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import task_registry_store as store

    state = {"entries": {}, "etag": None}

    async def _read(**kwargs):
        return dict(state["entries"]), state["etag"]

    monkeypatch.setattr(store, "read_registry", _read)

    template = _template()
    result = CliRunner().invoke(app, [
        "tasks", "create", "--task-id", "glm-run",
        "--model", "org/GLM", "--model-revision", "abc123",
        "--from-profile", template, "--cap", "0.3",
    ])

    assert result.exit_code != 0
    assert "--model-architecture" in result.output
    assert state["entries"] == {}


def test_a_legacy_task_without_model_still_routes_to_build_task_entry(monkeypatch):
    """`tasks create` without `--model` must build a legacy, contract-less
    entry exactly as before -- the new contract path must not change it."""
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import task_registry_store as store

    state = {"entries": {}, "etag": None}

    async def _read(**kwargs):
        return dict(state["entries"]), state["etag"]

    async def _write(entries, etag, **kwargs):
        from reliquary.shared.task_registry import validate_registry

        validate_registry(entries)
        state["entries"] = dict(entries)
        state["etag"] = '"v2"'
        return state["etag"]

    monkeypatch.setattr(store, "read_registry", _read)
    monkeypatch.setattr(store, "write_registry", _write)

    template = _template()
    result = CliRunner().invoke(app, [
        "tasks", "create", "--task-id", "default",
        "--profile-id", template, "--cap", "0.5",
    ])

    assert result.exit_code == 0, result.output
    entry = state["entries"]["default"]
    assert entry.contract is None
    assert entry.profile_id == template
