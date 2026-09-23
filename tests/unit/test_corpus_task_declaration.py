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
    from dataclasses import replace

    entry = corpus_entry()
    assert entry.mechanism == MECHANISM_CORPUS_GENERATION
    validate_entry(entry)

    # The assertions above only restate the fixture, so they would hold with
    # the rule reverted. Perturbing each half is what proves the accepted
    # shape is the one the rule requires, not merely one it tolerates.
    with pytest.raises(RegistryError):
        validate_entry(replace(entry, params={**entry.params, "floor": 0.01}))
    with pytest.raises(RegistryError):
        validate_entry(replace(entry, job_id=None))


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
    with pytest.raises(RegistryError, match="does not run one"):
        validate_entry(entry)


# --- The registry is the object every validator reads: the new field has to
# survive the trip through R2, and a legacy entry has to come back unchanged. ---


def test_the_named_job_survives_a_render_and_parse_round_trip(corpus_entry):
    from reliquary.shared.task_registry import parse_registry, render_registry

    parsed = parse_registry(render_registry({"corpus-run": corpus_entry()}))

    assert parsed["corpus-run"].job_id == "swe-v1"


def test_a_legacy_entry_reads_back_naming_no_job(rl_entry):
    from reliquary.shared.task_registry import parse_registry, render_registry

    raw = render_registry({"default": rl_entry()})

    # Against the BYTES, not the parsed object: `job_id is None` is also the
    # dataclass default, so it would hold if `render_registry` stopped writing
    # the key at all. The key is written, and written as null.
    assert b'"job_id":null' in raw
    assert parse_registry(raw)["default"].job_id is None


def test_a_job_id_that_is_not_a_name_is_refused_on_the_wire(corpus_entry):
    """A number where a name belongs must not reach the key interpolation in
    the job store, which is the only thing between it and a bucket path."""
    import json

    from reliquary.shared.task_registry import parse_registry, render_registry

    document = json.loads(render_registry({"corpus-run": corpus_entry()}))
    document["tasks"]["corpus-run"]["job_id"] = 17

    with pytest.raises(RegistryError, match="job_id"):
        parse_registry(json.dumps(document).encode())


# --- One job is paid for by one task. Two tasks naming it would each pay
# their own share for the SAME submissions, so this belongs beside the sum of
# caps in `validate_registry`, not in `validate_entry`. ---


def test_two_active_tasks_may_not_name_the_same_job(corpus_entry):
    from dataclasses import replace

    from reliquary.shared.task_registry import validate_registry

    first = corpus_entry()
    second = replace(first, task_id="corpus-run-2", profile_id="corpus-run-2")

    with pytest.raises(RegistryError, match="both name job"):
        validate_registry({"corpus-run": first, "corpus-run-2": second})


def test_two_tasks_naming_DIFFERENT_jobs_are_fine(corpus_entry):
    from dataclasses import replace

    from reliquary.shared.task_registry import validate_registry

    first = corpus_entry()
    second = replace(
        first, task_id="corpus-run-2", profile_id="corpus-run-2", job_id="math-v2"
    )

    validate_registry({"corpus-run": first, "corpus-run-2": second})


def test_a_retired_task_does_not_hold_its_job_against_a_new_one(corpus_entry):
    """The boundary of the chosen rule: a cancelled job must be re-declarable.
    A retired task accepts no submission, so it cannot double-pay for one."""
    from dataclasses import replace

    from reliquary.shared.task_registry import validate_registry

    retired = replace(corpus_entry(), status="retired", retired_at=5_000_000)
    fresh = replace(
        corpus_entry(), task_id="corpus-run-2", profile_id="corpus-run-2"
    )

    validate_registry({"corpus-run": retired, "corpus-run-2": fresh})


def test_the_uniqueness_rule_is_not_an_entry_rule(corpus_entry):
    """`validate_entry` sees one entry and cannot know about the other, so a
    corpus entry on its own must still validate."""
    validate_entry(corpus_entry())


# --- Declaring the first corpus task makes the registry unreadable to any
# validator whose binary predates the mechanism: they refuse the WHOLE
# registry and will not start. ---


def test_declaring_a_corpus_task_is_refused_unless_it_is_acknowledged(
    corpus_entry,
):
    from reliquary.shared.task_registry import require_fleet_knows_corpus_generation

    with pytest.raises(RegistryError, match="corpus-generation"):
        require_fleet_knows_corpus_generation(corpus_entry(), acknowledged=False)


def test_an_acknowledged_corpus_task_is_allowed(corpus_entry):
    from reliquary.shared.task_registry import require_fleet_knows_corpus_generation

    require_fleet_knows_corpus_generation(corpus_entry(), acknowledged=True)


def test_an_rl_task_needs_no_acknowledgement(rl_entry):
    # The hazard is the unknown MECHANISM, so it exists for corpus tasks only.
    from reliquary.shared.task_registry import require_fleet_knows_corpus_generation

    require_fleet_knows_corpus_generation(rl_entry(), acknowledged=False)


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
