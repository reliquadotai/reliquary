"""`jobs create` without --from-profile composes the job's contract from the
model flags, the corpus-v1 run policy and the catalog body of its source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app, build_corpus_task_entry
from reliquary.protocol.composition import (
    MODEL_IDENTITY_FIELDS,
    RUN_POLICIES,
    ModelSpec,
    compose_profile,
)
from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
from reliquary.validator.task_config import (
    CORPUS_SHARED_CONTRACT_FIELDS,
    merge_corpus_contracts,
)
from tests.unit.test_jobs_cli import ACK, _rl_entry, bucket, registry  # noqa: F401

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EPISODE_SOURCE = "reliquary_stateful_tools_v1"


def _args(**overrides):
    options = {
        "--job-id": "tools-v1",
        "--task-id": "corpus-tools",
        "--model": MODEL,
        "--model-revision": REVISION,
        "--model-architecture": ARCHITECTURE,
        "--checkpoint-sha256": "a" * 64,
        "--prompt-source": EPISODE_SOURCE,
        "--prompt-count": "100",
        "--renderer-id": "reliquary-external-prompt-v1",
        "--eos-token-id": "151645",
        "--slots-per-prompt": "8",
        "--cap": "0.1",
    }
    options.update(overrides)
    argv = ["jobs", "create", ACK]
    for flag, value in options.items():
        if value is not None:
            argv += [flag, value]
    return argv


def _create(registry, bucket, **overrides):  # noqa: F811
    registry["entries"].setdefault("default", _rl_entry("default", 0.5))
    return CliRunner().invoke(app, _args(**overrides))


def _manifest(bucket, job_id="tools-v1"):  # noqa: F811
    body, _ = bucket.objects[f"reliquary/corpus/jobs/{job_id}.json"]
    return json.loads(body)


@pytest.fixture
def no_source_build(monkeypatch):
    """A single-turn source builds its dataset to count rows; not here."""
    from reliquary.validator import corpus_service

    monkeypatch.setattr(corpus_service, "prompt_job_for_spec", lambda job, **kw: None)


def test_a_composed_job_carries_the_catalog_body_and_an_enforced_toploc(registry, bucket):  # noqa: F811
    result = _create(registry, bucket)
    assert result.exit_code == 0, result.output
    entry = registry["entries"]["corpus-tools"]
    expected = compose_profile(
        profile_id="corpus-tools",
        model=ModelSpec(MODEL, REVISION, ARCHITECTURE, "raw", (TOPLOC_DEPLOYED_DEFAULTS,)),
        run=RUN_POLICIES["corpus-v1"],
        environments=[EPISODE_SOURCE],
    ).to_generation_contract()
    assert entry.contract == expected
    assert entry.mechanism == "corpus-generation"
    assert entry.job_id == "tools-v1"


def test_without_a_budget_the_job_takes_the_catalog_one(registry, bucket):  # noqa: F811
    result = _create(registry, bucket)
    assert result.exit_code == 0, result.output
    assert _manifest(bucket)["sampling"]["max_new_tokens"] == (
        ENVIRONMENT_CATALOG[EPISODE_SOURCE].max_new_tokens
    )


def test_a_declared_budget_stays_in_the_manifest_not_the_contract(registry, bucket):  # noqa: F811
    result = _create(registry, bucket, **{"--max-new-tokens": "2048"})
    assert result.exit_code == 0, result.output
    assert _manifest(bucket)["sampling"]["max_new_tokens"] == 2048
    body = registry["entries"]["corpus-tools"].contract["environments"][EPISODE_SOURCE]
    assert body["max_new_tokens"] == ENVIRONMENT_CATALOG[EPISODE_SOURCE].max_new_tokens


@pytest.mark.parametrize(("renderer", "flag", "encoding"), [
    ("chat-template-v1", None, "chat_template"),
    ("chat-template-thinking-v1", None, "chat_template"),
    ("reliquary-external-prompt-v1", None, "raw"),
    ("reliquary-external-prompt-v1", "chat_template", "chat_template"),
])
def test_the_prompt_encoding_follows_the_renderer_unless_named(
    registry, bucket, renderer, flag, encoding,  # noqa: F811
):
    result = _create(
        registry, bucket, **{"--renderer-id": renderer, "--prompt-encoding": flag},
    )
    assert result.exit_code == 0, result.output
    assert registry["entries"]["corpus-tools"].contract["prompt_encoding"] == encoding


def test_env_is_an_alias_of_prompt_source(registry, bucket):  # noqa: F811
    argv = _args(**{"--prompt-source": None})
    registry["entries"]["default"] = _rl_entry("default", 0.5)
    result = CliRunner().invoke(app, argv + ["--env", EPISODE_SOURCE])
    assert result.exit_code == 0, result.output
    assert list(registry["entries"]["corpus-tools"].contract["environments"]) == [EPISODE_SOURCE]


def test_an_uncatalogued_source_is_refused_before_any_write(registry, bucket):  # noqa: F811
    result = _create(registry, bucket, **{"--prompt-source": "envscaler_tools_v1"})
    assert result.exit_code != 0
    assert "catalog" in result.output
    assert "corpus-tools" not in registry["entries"]
    assert not bucket.objects


def test_prompt_encoding_with_from_profile_is_refused(registry, bucket):  # noqa: F811
    result = _create(registry, bucket, **{
        "--from-profile": "qwen3-4b-reliquary-episode-v7-dev1",
        "--prompt-encoding": "raw",
    })
    assert result.exit_code != 0
    assert "--prompt-encoding" in result.output and "--from-profile" in result.output
    assert not bucket.objects


def test_the_from_profile_path_still_seeds_from_the_template(registry, bucket):  # noqa: F811
    template = "qwen3-4b-reliquary-episode-v7-dev1"
    result = _create(registry, bucket, **{"--from-profile": template})
    assert result.exit_code == 0, result.output
    contract = registry["entries"]["corpus-tools"].contract
    assert contract["protocol_version"] == 7
    assert contract["collection_seconds"] == 300


def test_composing_corpus_code_v1_reproduces_the_live_contract(
    registry, bucket, no_source_build,  # noqa: F811
):
    result = _create(registry, bucket, **{
        "--job-id": "code-qwen38-27b-v1",
        "--task-id": "corpus-code-v1",
        "--prompt-source": "reliquary_code_v1",
        "--renderer-id": "chat-template-v1",
    })
    assert result.exit_code == 0, result.output
    entry = registry["entries"]["corpus-code-v1"]
    printed = json.dumps(entry.contract, sort_keys=True, separators=(",", ":")) + "\n"
    assert printed == (FIXTURES / "corpus_code_v1_contract.json").read_text()
    assert entry.profile_sha256 == (
        "ffa86eaf42a3034f3c7bfa67b8b5ae3658019acc4fb0ee62d5485607a66b818d"
    )


def test_two_composed_jobs_on_one_model_merge(registry, bucket):  # noqa: F811
    assert _create(registry, bucket).exit_code == 0
    result = _create(registry, bucket, **{
        "--job-id": "retrieval-v1", "--task-id": "corpus-retrieval",
        "--prompt-source": "reliquary_retrieval_tools_v1",
    })
    assert result.exit_code == 0, result.output
    entries = registry["entries"]
    merged = merge_corpus_contracts({
        t: entries[t].contract for t in ("corpus-tools", "corpus-retrieval")
    })
    assert set(merged["environments"]) == {EPISODE_SOURCE, "reliquary_retrieval_tools_v1"}


def _composed_entry(task_id, source, **model):
    spec = dict(model_id=MODEL, model_revision=REVISION, model_architecture=ARCHITECTURE,
                prompt_encoding="chat_template")
    spec.update(model)
    base = compose_profile(
        profile_id=task_id, model=ModelSpec(**spec), run=RUN_POLICIES["corpus-v1"],
        environments=[source],
    )
    return build_corpus_task_entry(
        task_id=task_id, job_id=f"{task_id}-job", base=base, model_id=spec["model_id"],
        model_revision=spec["model_revision"], model_architecture=spec["model_architecture"],
        prompt_source=source, cap=0.05, overrides={},
    )


def _template_entry(task_id, source):
    return build_corpus_task_entry(
        task_id=task_id, job_id=f"{task_id}-job",
        from_profile="qwen3-4b-base-dapo-reliquary-v1", model_id=MODEL,
        model_revision=REVISION, model_architecture=ARCHITECTURE,
        prompt_source=source, cap=0.05, overrides={},
    )


def test_a_composed_job_and_a_template_job_on_one_environment_merge():
    composed = _composed_entry("corpus-math-a", "openmathinstruct")
    seeded = _template_entry("corpus-math-b", "openmathinstruct")
    assert composed.contract["environments"] == seeded.contract["environments"]
    merged = merge_corpus_contracts({
        "corpus-math-a": composed.contract, "corpus-math-b": seeded.contract,
    })
    assert list(merged["environments"]) == ["openmathinstruct"]


@pytest.mark.parametrize(("field", "value"), [
    ("model_id", "Qwen/Other"),
    ("model_revision", "e" * 40),
    ("model_architecture", "Qwen3ForCausalLM"),
])
def test_composed_jobs_on_different_models_refuse(field, value):
    a = _composed_entry("corpus-a", EPISODE_SOURCE)
    b = _composed_entry("corpus-b", "reliquary_retrieval_tools_v1", **{field: value})
    with pytest.raises(ValueError, match=field):
        merge_corpus_contracts({"corpus-a": a.contract, "corpus-b": b.contract})


def test_composed_jobs_with_different_proofs_refuse():
    from dataclasses import replace

    a = _composed_entry("corpus-a", EPISODE_SOURCE)
    other = replace(TOPLOC_DEPLOYED_DEFAULTS, exp_mismatch_threshold=70)
    b = _composed_entry("corpus-b", "reliquary_retrieval_tools_v1", proofs=(other,))
    with pytest.raises(ValueError, match="proofs"):
        merge_corpus_contracts({"corpus-a": a.contract, "corpus-b": b.contract})


def test_the_shared_fields_are_the_model_identity():
    old = ("model_id", "model_revision", "model_architecture", "proofs")
    assert CORPUS_SHARED_CONTRACT_FIELDS == MODEL_IDENTITY_FIELDS == old
    assert CORPUS_SHARED_CONTRACT_FIELDS is MODEL_IDENTITY_FIELDS


def test_build_job_manifest_refuses_both_a_profile_and_a_template():
    from reliquary.cli.main import build_job_manifest
    from reliquary.protocol.profiles import PROFILES

    template = "qwen3-4b-reliquary-episode-v7-dev1"
    with pytest.raises(ValueError, match="profile"):
        build_job_manifest(
            job_id="tools-v1", checkpoint_repo=MODEL, checkpoint_revision=REVISION,
            checkpoint_sha256="a" * 64, prompt_source=EPISODE_SOURCE, prompt_count=100,
            renderer_id="reliquary-jsonl-tools-v1", eos_token_id=151645, slots_per_prompt=8,
            temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=4096, n=1,
            grader_id=None, threshold=None, prompt_order="free", deadline_round=None,
            from_profile=template, profile=PROFILES[template],
        )
