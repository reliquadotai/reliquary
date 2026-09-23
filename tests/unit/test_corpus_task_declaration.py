"""Declaring a corpus job is a registry write like any other, plus the two
rules that make V0 safe: the price is pinned and the job is named."""

import pytest

from reliquary.shared.task_registry import (
    MECHANISM_CORPUS_GENERATION,
    RegistryError,
    validate_entry,
)


def _params(**overrides):
    """Shipped controller defaults, then the overrides a case needs."""
    from dataclasses import asdict

    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    params = asdict(PRODUCTION_PRICE_PARAMS)
    params["cap"] = 0.30
    params["floor"] = 0.30
    params.update(overrides)
    return params


@pytest.fixture
def corpus_entry():
    """A valid corpus entry, with a hook for the one field a case breaks."""
    from reliquary.shared.task_registry import TaskEntry

    def _make(*, job_id="swe-v1", **param_overrides):
        return TaskEntry(
            task_id="corpus-run",
            profile_id="corpus-run",
            profile_sha256="a" * 64,
            mechanism=MECHANISM_CORPUS_GENERATION,
            params=_params(**param_overrides),
            status="active",
            retired_at=None,
            job_id=job_id,
        )

    return _make


@pytest.fixture
def rl_entry():
    """The existing shape, which the new rule must leave alone."""
    from reliquary.shared.task_registry import (
        MECHANISM_RL_DISCOVERED_PRICE,
        TaskEntry,
    )

    def _make(*, job_id=None):
        return TaskEntry(
            task_id="default",
            profile_id="qwen3-4b-base-dapo-reliquary-v1",
            profile_sha256="b" * 64,
            mechanism=MECHANISM_RL_DISCOVERED_PRICE,
            params=_params(floor=0.05),
            status="active",
            retired_at=None,
            job_id=job_id,
        )

    return _make


def test_a_corpus_entry_pins_its_price_and_names_its_job(corpus_entry):
    entry = corpus_entry()
    assert entry.mechanism == MECHANISM_CORPUS_GENERATION
    assert entry.params["floor"] == entry.params["cap"]
    assert entry.job_id
    validate_entry(entry)


def test_a_corpus_entry_whose_price_is_not_pinned_is_refused(corpus_entry):
    # An unpinned floor animates advance(); V0 has no price discovery, so the
    # share would drift with nothing driving it.
    entry = corpus_entry(floor=0.01)
    with pytest.raises(RegistryError) as caught:
        validate_entry(entry)
    assert "floor" in str(caught.value)


def test_a_corpus_entry_without_a_job_is_refused(corpus_entry):
    entry = corpus_entry(job_id=None)
    with pytest.raises(RegistryError) as caught:
        validate_entry(entry)
    assert "job" in str(caught.value).lower()


def test_an_rl_entry_is_not_subjected_to_the_corpus_rules(rl_entry):
    # The new rule must bite on the new mechanism only; every existing entry
    # has floor < cap and no job.
    entry = rl_entry()
    assert entry.params["floor"] < entry.params["cap"]
    assert entry.job_id is None
    validate_entry(entry)


def test_a_job_id_on_an_rl_entry_is_refused(rl_entry):
    entry = rl_entry(job_id="swe-v1")
    with pytest.raises(RegistryError):
        validate_entry(entry)


# --- The registry is the object every validator reads: the new field has to
# survive the trip through R2, and a legacy entry has to come back unchanged. ---


def test_the_named_job_survives_a_render_and_parse_round_trip(corpus_entry):
    from reliquary.shared.task_registry import parse_registry, render_registry

    parsed = parse_registry(render_registry({"corpus-run": corpus_entry()}))

    assert parsed["corpus-run"].job_id == "swe-v1"


def test_a_legacy_entry_reads_back_naming_no_job(rl_entry):
    from reliquary.shared.task_registry import parse_registry, render_registry

    parsed = parse_registry(render_registry({"default": rl_entry()}))

    assert parsed["default"].job_id is None


def test_a_job_id_that_is_not_a_name_is_refused_on_the_wire(corpus_entry):
    """A number where a name belongs must not reach the key interpolation in
    the job store, which is the only thing between it and a bucket path."""
    import json

    from reliquary.shared.task_registry import parse_registry, render_registry

    document = json.loads(render_registry({"corpus-run": corpus_entry()}))
    document["tasks"]["corpus-run"]["job_id"] = 17

    with pytest.raises(RegistryError, match="job_id"):
        parse_registry(json.dumps(document).encode())


# --- `build_corpus_task_entry`: the operator-facing constructor. ---


def _template() -> str:
    """A template declaring more than one environment, so R-2's narrowing to a
    single prompt source is actually exercised."""
    from reliquary.protocol.profiles import PROFILES

    for profile_id in sorted(PROFILES):
        if len(PROFILES[profile_id].environments) >= 2:
            return profile_id
    pytest.skip("no compiled profile declares two environments")


def _prompt_source(template: str) -> str:
    from reliquary.protocol.profiles import PROFILES

    return sorted(PROFILES[template].environments)[0]


def _build(**overrides):
    from reliquary.cli.main import build_corpus_task_entry

    template = _template()
    kwargs = dict(
        task_id="corpus-run",
        job_id="swe-v1",
        from_profile=template,
        model_id="org/Frozen",
        model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        prompt_source=_prompt_source(template),
        cap=0.30,
        overrides={},
    )
    kwargs.update(overrides)
    return build_corpus_task_entry(**kwargs)


def test_the_builder_pins_the_price_names_the_job_and_the_mechanism():
    entry = _build()

    assert entry.mechanism == MECHANISM_CORPUS_GENERATION
    assert entry.job_id == "swe-v1"
    assert entry.params["floor"] == entry.params["cap"] == 0.30
    validate_entry(entry)


def test_the_carried_contract_names_the_prompt_source_and_nothing_else():
    # R-2: the contract's one environment IS the job's prompt source, so a
    # validator that boots this task installs exactly what the job reads from.
    template = _template()
    entry = _build()

    assert entry.contract is not None
    assert list(entry.contract["environments"]) == [_prompt_source(template)]


def test_a_prompt_source_the_template_does_not_declare_is_refused():
    with pytest.raises(ValueError, match="not-an-environment"):
        _build(prompt_source="not-an-environment")


def test_an_explicit_floor_that_contradicts_the_pin_is_refused():
    # Accepting it and then overwriting it is the silent-drop trap this branch
    # keeps closing; the price of a corpus task is the cap, by construction.
    with pytest.raises(ValueError, match="floor"):
        _build(overrides={"floor": 0.05})
