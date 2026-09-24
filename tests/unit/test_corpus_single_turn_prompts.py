"""The second fidelity path: the rows of a single-turn environment.

Task 4 built prompt fidelity against episode environments, whose rows render
through an episode renderer the manifest names. Every source a first corpus job
would draw from is single-turn instead, and those render through the ACTIVE
PROFILE's prompt template -- which the manifest does not choose. So two hazards
live here, and both are what this file is for:

* **Two sources of truth for what the miner was asked.** `get_problem` renders
  through `render_active_prompt`; the manifest carries `renderer_id`
  independently. Under task isolation they agree by construction, and nothing
  enforces it. The tests below require a disagreement to be refused, at
  declaration and again where the job is served.
* **`get_problem` wraps its index with modulo.** An index past the job's own
  prompts does not raise: it returns a VALID prompt for a row the job does not
  own. So the boundary tests here CROSS it -- a test that stayed inside would
  pass just as well on an adapter that never bounded anything.

No GPU, no network, no dataset read: the environments are stubs, and the one
test that uses a real installed spec only RESOLVES it, never builds it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.job import parse_job
from reliquary.environment.agentic.types import EpisodeTask
from reliquary.environment.registry import ENVIRONMENT_SPECS, EnvironmentSpec
from reliquary.validator.corpus_service import (
    CorpusPromptSourceError,
    EnvironmentPromptJob,
    PromptFidelity,
    SingleTurnPromptJob,
    SingleTurnPromptRenderer,
    prompt_job_for_spec,
    renderer_for_job,
    resolve_prompt_source,
)
from reliquary.validator.corpus_text import check_prompt_fidelity

# The CLI's own fake bucket and fake registry, so the declaration test below
# runs `jobs create` against the store it is proved against everywhere else.
from tests.unit.test_jobs_cli import bucket, registry  # noqa: F401

SOURCE = "stub-single-turn"
TEMPLATE_ID = "stub-step-by-step-v1"

# The eight installed single-turn environments the design spec names as the
# sources of a first corpus job. Until this path exists, none of them can be
# declared.
FIRST_JOB_SOURCES = (
    "openmathinstruct",
    "opencodeinstruct",
    "reliquary_dapo_math_v1",
    "reliquary_code_v1",
    "reliquary_instruction_following_v1",
    "reliquarylogic_v1",
    "reliquary_logic_v2",
    "reliquaryverifiable_v1",
)


class _SingleTurnEnvironment:
    """One row per index, WITH the modulo wrap every installed single-turn
    environment has: `get_problem` is documented to be safe out of range, which
    is exactly what makes an unbounded adapter look like it works."""

    name = SOURCE

    def __init__(self, rows: int = 4):
        self._rows = rows

    def __len__(self):
        return self._rows

    def get_problem(self, index):
        row = index % self._rows
        return {
            "prompt": f"question {row}",
            "ground_truth": str(row),
            "id": f"row{row:04d}",
        }


class _SingleTurnSpec:
    """Stands in for an `EnvironmentSpec`: the adapter must not build the real
    one, which reads a dataset."""

    interaction_mode = "single_turn"

    def __init__(self, environment=None):
        self._environment = environment or _SingleTurnEnvironment()

    def create(self):
        return self._environment


class _EpisodeEnvironment:
    name = SOURCE

    def __init__(self, rows: int = 4):
        self._rows = rows

    def __len__(self):
        return self._rows

    def get_task(self, index):
        return EpisodeTask(id=f"row-{index}", prompt=f"question {index}", tools=())


class _EpisodeSpec:
    interaction_mode = "episode"

    def __init__(self, environment=None):
        self._environment = environment or _EpisodeEnvironment()

    def create(self):
        return self._environment


def _profile(template_id=TEMPLATE_ID, *, environment=SOURCE, profile_id="stub-v1"):
    """A protocol profile as far as this path reads one: which prompt template
    each environment renders through."""
    template = (
        None
        if template_id is None
        else SimpleNamespace(template_id=template_id, template="$problem")
    )
    return SimpleNamespace(
        profile_id=profile_id,
        environments={environment: SimpleNamespace(prompt_template=template)},
    )


def _job(**overrides):
    raw = {
        "schema": "reliquary/corpus-job/v1",
        "job_id": "swe-v1",
        "checkpoint_repo": "org/Frozen",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": "a" * 64,
        "prompt_source": SOURCE,
        "prompt_count": 4,
        "renderer_id": TEMPLATE_ID,
        "eos_token_id": 151645,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 2,
            "max_new_tokens": 4096,
            "n": 1,
        },
        "slots_per_prompt": 8,
        "filter": None,
        "prompt_order": "free",
        "deadline_round": None,
    }
    raw.update(overrides)
    return parse_job(raw)


def _prompts(job=None, *, spec=None, profile=None):
    return prompt_job_for_spec(
        job or _job(),
        environments={SOURCE: spec or _SingleTurnSpec()},
        profile=profile or _profile(),
    )


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_a_single_turn_source_resolves_instead_of_being_refused():
    """The whole point: a math or code corpus is undeclarable until this
    stops raising."""
    spec = _SingleTurnSpec()

    resolved = resolve_prompt_source(
        SOURCE,
        environments={SOURCE: spec},
        renderer_id=TEMPLATE_ID,
        profile=_profile(),
    )

    assert resolved is spec


def test_every_source_a_first_corpus_job_names_resolves_without_being_built(
    monkeypatch,
):
    """The real installed specs, against the real compiled profiles. Building
    one reads a dataset, so `create` is made fatal here: resolution must be a
    decision about the manifest, never a download."""
    from reliquary.protocol.profiles import PROFILES

    monkeypatch.setattr(
        EnvironmentSpec,
        "create",
        lambda self: pytest.fail(f"resolution built {self.name}"),
    )

    for source in FIRST_JOB_SOURCES:
        declaring = [
            (profile_id, profile.environments[source].prompt_template)
            for profile_id, profile in sorted(PROFILES.items())
            if source in profile.environments
            and profile.environments[source].prompt_template is not None
        ]
        assert declaring, f"no compiled profile pins a prompt template for {source}"
        profile_id, template = declaring[0]

        resolved = resolve_prompt_source(
            source, renderer_id=template.template_id, profile=profile_id
        )

        assert resolved is ENVIRONMENT_SPECS[source]
        assert resolved.interaction_mode == "single_turn"


def test_a_renderer_the_contract_does_not_declare_is_refused():
    """Trap 1's refusing direction: the environment renders through the
    profile's template, so a manifest naming anything else describes a prompt
    nobody will ever be asked."""
    with pytest.raises(CorpusPromptSourceError) as caught:
        resolve_prompt_source(
            SOURCE,
            environments={SOURCE: _SingleTurnSpec()},
            renderer_id="reliquary-jsonl-tools-v1",
            profile=_profile(),
        )

    message = str(caught.value)
    assert TEMPLATE_ID in message and "reliquary-jsonl-tools-v1" in message


def test_a_single_turn_source_cannot_be_resolved_without_saying_how_it_renders():
    """A caller that omitted the renderer would get the active profile's
    rendering with nothing checking it -- the disagreement this task exists to
    close, reintroduced by a default."""
    with pytest.raises(CorpusPromptSourceError) as caught:
        resolve_prompt_source(
            SOURCE, environments={SOURCE: _SingleTurnSpec()}, profile=_profile()
        )

    assert "renderer_id" in str(caught.value)


def test_a_profile_with_no_prompt_template_has_no_rendering_rule_to_pin():
    """A legacy profile leaves the prompt to environment-local code, so there
    is no id for the manifest to name and no agreement to check. Refused, not
    guessed -- and told apart from a disagreement, which an empty id standing
    in for "no template" would otherwise be mistaken for."""
    with pytest.raises(CorpusPromptSourceError) as caught:
        resolve_prompt_source(
            SOURCE,
            environments={SOURCE: _SingleTurnSpec()},
            renderer_id=TEMPLATE_ID,
            profile=_profile(None, profile_id="legacy-v1"),
        )

    message = str(caught.value)
    assert "no prompt template" in message and "legacy-v1" in message


def test_a_profile_that_does_not_declare_the_source_is_refused():
    """This validator would render a prompt the job's own contract never
    described."""
    with pytest.raises(CorpusPromptSourceError) as caught:
        resolve_prompt_source(
            SOURCE,
            environments={SOURCE: _SingleTurnSpec()},
            renderer_id=TEMPLATE_ID,
            profile=_profile(environment="somebody-else"),
        )

    assert SOURCE in str(caught.value)


def test_a_source_that_is_neither_mode_is_still_refused():
    class _Spec:
        interaction_mode = "batched"

    with pytest.raises(CorpusPromptSourceError) as caught:
        resolve_prompt_source(
            SOURCE, environments={SOURCE: _Spec()}, renderer_id=TEMPLATE_ID
        )

    assert "batched" in str(caught.value)


# --------------------------------------------------------------------------
# The rows
# --------------------------------------------------------------------------


def test_the_prompt_for_a_row_is_what_the_environment_rendered_for_it():
    """An off-by-one here pays a miner for answering a different question."""
    prompts = _prompts()
    renderer = SingleTurnPromptRenderer()

    for index in range(4):
        assert renderer.initial_text(prompts.task_for(index)) == f"question {index}"


def test_a_prompt_from_another_index_is_refused():
    prompts = _prompts()
    renderer = SingleTurnPromptRenderer()

    faithful = check_prompt_fidelity(
        "question 2", job=prompts, prompt_index=2, renderer=renderer
    )
    astray = check_prompt_fidelity(
        "question 3", job=prompts, prompt_index=2, renderer=renderer
    )

    assert faithful.ok
    assert not astray.ok
    assert astray.reason == "prompt_not_faithful"


def test_an_index_past_the_job_s_prompts_is_refused_rather_than_wrapped():
    """Trap 2, crossed rather than approached. The environment holds four rows
    and the job owns two of them, so the two indices below BOTH return a valid
    prompt from `get_problem` -- one of a row the job does not own, one of a
    row that only exists by wrapping. "A prompt came back" is therefore no
    evidence at all, which is why this asserts on the refusal."""
    environment = _SingleTurnEnvironment(rows=4)
    prompts = _prompts(_job(prompt_count=2), spec=_SingleTurnSpec(environment))

    assert environment.get_problem(2)["prompt"] == "question 2"
    assert environment.get_problem(4)["prompt"] == "question 0"

    for outside in (2, 4, 40):
        with pytest.raises(CorpusPromptSourceError) as caught:
            prompts.task_for(outside)
        assert "2 prompts" in str(caught.value)

    with pytest.raises(CorpusPromptSourceError):
        prompts.task_for(-1)


def test_a_source_shorter_than_the_manifest_claims_is_refused():
    """Every slot past the last row names a prompt that only exists by
    wrapping onto another row's."""
    with pytest.raises(CorpusPromptSourceError) as caught:
        _prompts(spec=_SingleTurnSpec(_SingleTurnEnvironment(rows=3)))

    assert "3" in str(caught.value)


def test_a_row_with_no_prompt_text_is_named_not_a_crash():
    class _Empty(_SingleTurnEnvironment):
        def get_problem(self, index):
            return {"ground_truth": "42", "id": "row"}

    prompts = _prompts(spec=_SingleTurnSpec(_Empty()))

    with pytest.raises(CorpusPromptSourceError) as caught:
        prompts.task_for(0)

    assert SOURCE in str(caught.value)


# --------------------------------------------------------------------------
# One protocol, two shapes
# --------------------------------------------------------------------------


def test_the_mode_decides_the_shape_and_the_check_never_learns_which():
    """`check_prompt_fidelity` takes a `PromptJob` and a renderer, and both
    modes supply exactly those two. A check that had to ask would be a third
    place for the modes to diverge."""
    single_turn = _prompts()
    episode = prompt_job_for_spec(
        _job(renderer_id="reliquary-jsonl-tools-v1"),
        environments={SOURCE: _EpisodeSpec()},
        profile=_profile(None),
    )

    assert isinstance(single_turn, SingleTurnPromptJob)
    assert isinstance(episode, EnvironmentPromptJob)
    assert episode.task_for(1).id == "row-1"


def test_the_renderer_a_job_renders_through_follows_its_source_s_mode():
    """What the mount hands the endpoint. An episode job keeps the renderer its
    manifest names; a single-turn job's prompt is already rendered, so the only
    faithful renderer is the one that changes nothing."""
    from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer

    single_turn = renderer_for_job(
        _job(),
        lambda text: [],
        environments={SOURCE: _SingleTurnSpec()},
        profile=_profile(),
    )
    episode = renderer_for_job(
        _job(renderer_id="reliquary-jsonl-tools-v1"),
        lambda text: [],
        environments={SOURCE: _EpisodeSpec()},
        profile=_profile(None),
    )

    assert isinstance(single_turn, SingleTurnPromptRenderer)
    assert isinstance(episode, CanonicalEpisodeRenderer)
    assert single_turn.initial_text(
        EpisodeTask(id="row", prompt="question 9", tools=())
    ) == "question 9"


def test_the_fidelity_seam_serves_a_single_turn_job():
    """The endpoint's own seam, unchanged: it resolves a job to prompts and
    compares. Nothing in it knows which mode it just served."""
    fidelity = PromptFidelity(
        renderer=SingleTurnPromptRenderer(),
        prompt_job_for=lambda job: _prompts(job),
    )
    job = _job()

    async def run():
        return (
            await fidelity("question 1", job=job, prompt_index=1),
            await fidelity("question 0", job=job, prompt_index=1),
        )

    faithful, astray = asyncio.run(run())

    assert faithful.ok
    assert not astray.ok
    assert astray.reason == "prompt_not_faithful"


# --------------------------------------------------------------------------
# Declaration and mount
# --------------------------------------------------------------------------


def test_a_manifest_whose_renderer_disagrees_with_the_contract_is_refused(monkeypatch):
    """`jobs create` already refuses a source this binary cannot render; a
    renderer the job's own contract does not declare is the same failure, one
    field over, and it must not reach the bucket."""
    from reliquary.cli.main import build_job_manifest
    from tests.unit.test_jobs_cli import stub_source_rows

    # Declaration counts the source's rows; the rule under test reads the
    # PROFILE, so stubbing the dataset read leaves it untouched.
    stub_source_rows(monkeypatch, "openmathinstruct", 1000)

    def _manifest(**overrides):
        arguments = {
            "job_id": "math-v1",
            "checkpoint_repo": "org/Frozen",
            "checkpoint_revision": "abc123",
            "checkpoint_sha256": "a" * 64,
            "prompt_source": "openmathinstruct",
            "prompt_count": 1000,
            "renderer_id": "openmathinstruct-step-by-step-v1",
            "from_profile": "qwen3-4b-base-dapo-reliquary-v1",
            "eos_token_id": 151645,
            "slots_per_prompt": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 2,
            "max_new_tokens": 4096,
            "n": 1,
            "grader_id": None,
            "threshold": None,
            "prompt_order": "free",
            "deadline_round": None,
        }
        arguments.update(overrides)
        return build_job_manifest(**arguments)

    assert _manifest()["prompt_source"] == "openmathinstruct"

    with pytest.raises(ValueError, match="openmathinstruct"):
        _manifest(renderer_id="opencodeinstruct-step-by-step-v1")

    # The template the job's contract declares is a property of THAT contract:
    # a profile that renders the same environment some other way is a different
    # declaration, and the manifest cannot straddle both.
    with pytest.raises(ValueError, match="openmathinstruct"):
        _manifest(from_profile="qwen35-2b-auction-v2")


CONTRACT_PROFILE = "qwen3-4b-base-dapo-reliquary-v1"


def _declare_math_job(registry, monkeypatch):
    """A corpus job on an installed MATH source, through the real CLI."""
    from typer.testing import CliRunner

    from reliquary.cli.main import app as cli
    from reliquary.infrastructure import corpus_job_store as job_store

    from tests.unit.test_jobs_cli import _create_args, _rl_entry, stub_source_rows

    # Declaration counts the source's rows, so the dataset read is stubbed:
    # the mount below still resolves the real spec's mode and the real profile.
    stub_source_rows(monkeypatch, "openmathinstruct", 1000)
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(
        cli,
        _create_args(
            **{
                "--from-profile": CONTRACT_PROFILE,
                "--prompt-source": "openmathinstruct",
                "--renderer-id": "openmathinstruct-step-by-step-v1",
                "--job-id": "math-v1",
            }
        ),
    )
    assert result.exit_code == 0, result.output
    entry = registry["entries"]["corpus-run"]
    job, _ = asyncio.run(job_store.read_job(entry.job_id))
    assert job.prompt_source == "openmathinstruct"
    return entry


def _mount(entry):
    from reliquary.cli.main import mount_corpus_service
    from reliquary.validator.server import ValidatorServer

    server = ValidatorServer()
    mounted = asyncio.run(
        mount_corpus_service(
            server,
            entry,
            # Only an episode renderer encodes, and this job has none.
            tokenizer=SimpleNamespace(encode=lambda text, **kw: []),
            verify_signature=lambda request: True,
        )
    )
    return server, mounted


def test_a_math_job_declares_and_mounts_end_to_end(bucket, registry, monkeypatch):
    """The path this task opens, over the real CLI, the real store and the real
    mount: a job on an installed MATH source is declared and its route is
    served. Nothing here builds the environment -- the mount only resolves, and
    the dataset read would come with a submission -- so this stays a unit test
    and still crosses every seam a declaration touches.

    The active profile is set to the job's own contract because that is what
    `RELIQUARY_TASK_CONTRACT` does to a validator serving this task."""
    from reliquary.protocol import profiles

    entry = _declare_math_job(registry, monkeypatch)
    monkeypatch.setattr(
        profiles, "ACTIVE_PROTOCOL_PROFILE", profiles.PROFILES[CONTRACT_PROFILE]
    )

    server, mounted = _mount(entry)

    assert mounted is True
    # The OpenAPI paths, not `app.routes`: FastAPI >= 0.141 keeps an included
    # router's routes out of `app.routes`.
    assert "/corpus/submit" in server.app.openapi()["paths"]


def test_a_validator_rendering_these_prompts_differently_refuses_to_serve(
    bucket, registry, monkeypatch
):
    """Trap 1 where it bites: the manifest is fixed and the profile is not.
    A validator whose contract renders this environment another way would
    compare every submission against a prompt no miner was ever given, and it
    would look exactly like a fleet of dishonest miners."""
    from reliquary.protocol import profiles

    entry = _declare_math_job(registry, monkeypatch)
    monkeypatch.setattr(
        profiles,
        "ACTIVE_PROTOCOL_PROFILE",
        profiles.PROFILES["qwen35-2b-auction-v2"],
    )

    with pytest.raises(CorpusPromptSourceError) as caught:
        _mount(entry)

    assert "openmathinstruct" in str(caught.value)
