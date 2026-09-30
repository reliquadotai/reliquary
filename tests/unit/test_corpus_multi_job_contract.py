"""The contract one corpus validator runs for several jobs built from different
templates: what must agree, what is per job, and how each job renders."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from reliquary.validator.task_config import (
    TaskConfigError,
    merge_corpus_contracts,
    resolve_corpus_task_configs,
)
from tests.unit.test_corpus_service import CHECKPOINT, _manifest, _r2_client, fake_r2  # noqa: F401
from tests.unit.test_jobs_cli import _rl_entry, registry  # noqa: F401

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "f" * 40
ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
# The real case: maths from the only template declaring openmathinstruct, code
# from the Teutonic suite, both re-pointed at one teacher.
MATH_TEMPLATE = "qwen3-4b-base-dapo-reliquary-v1"
CODE_TEMPLATE = "teutonic-9b-reliquary-suite-v9-dev1"


def _entry(task_id, job_id, template, source, **overrides):
    from reliquary.cli.main import build_corpus_task_entry

    fields = dict(model_id=MODEL, model_revision=REVISION, model_architecture=ARCHITECTURE)
    fields.update(overrides)
    return build_corpus_task_entry(
        task_id=task_id, job_id=job_id, from_profile=template, prompt_source=source,
        cap=0.05, overrides={}, **fields,
    )


def _entries(**code_overrides):
    return {
        "corpus-math": _entry("corpus-math", "math-v1", MATH_TEMPLATE, "openmathinstruct"),
        "corpus-code": _entry("corpus-code", "code-v1", CODE_TEMPLATE, "reliquary_code_v1",
                              **code_overrides),
    }


def _contracts(entries):
    return {task_id: entry.contract for task_id, entry in entries.items()}


def _with(entry, **changes):
    """The entry with its contract changed, digest recomputed as a real one would be."""
    from dataclasses import replace

    from reliquary.environment.abi import canonical_sha256

    contract = {**json.loads(json.dumps(entry.contract)), **changes}
    return replace(entry, contract=contract, profile_sha256=canonical_sha256(contract))


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_two_jobs_from_different_templates_merge_to_the_union():
    entries = _entries()
    merged = merge_corpus_contracts(_contracts(entries))

    assert merged["environments"] == {
        **entries["corpus-math"].contract["environments"],
        **entries["corpus-code"].contract["environments"],
    }
    assert (merged["model_id"], merged["model_revision"]) == (MODEL, REVISION)
    assert merged["profile_id"] == "corpus-code+corpus-math"
    # Their templates disagree on these, which the corpus path never reads.
    assert entries["corpus-math"].contract["prompt_encoding"] != entries["corpus-code"].contract["prompt_encoding"]
    assert entries["corpus-math"].contract["protocol_version"] != entries["corpus-code"].contract["protocol_version"]


def test_the_merge_does_not_depend_on_the_order_the_tasks_are_named():
    contracts = _contracts(_entries())
    reversed_order = dict(reversed(list(contracts.items())))
    assert merge_corpus_contracts(contracts) == merge_corpus_contracts(reversed_order)


@pytest.mark.parametrize("field,value", [
    ("model_id", "Qwen/Other"),
    ("model_revision", "e" * 40),
    ("model_architecture", "Qwen3ForCausalLM"),
])
def test_a_different_model_refuses_naming_the_field(field, value):
    with pytest.raises(ValueError) as caught:
        merge_corpus_contracts(_contracts(_entries(**{field: value})))
    assert field in str(caught.value)
    assert "corpus-math" in str(caught.value) and "corpus-code" in str(caught.value)


def test_different_proofs_refuse():
    entries = _entries()
    code = entries["corpus-code"].contract
    proofs = [dict(p) for p in code["proofs"]]
    proofs[0]["exp_mismatch_threshold"] += 1
    entries["corpus-code"] = _with(entries["corpus-code"], proofs=proofs)
    with pytest.raises(ValueError, match="proofs"):
        merge_corpus_contracts(_contracts(entries))


def test_one_environment_declared_two_ways_refuses():
    entries = _entries()
    math_env = entries["corpus-math"].contract["environments"]["openmathinstruct"]
    entries["corpus-code"] = _with(entries["corpus-code"], environments={
        **entries["corpus-code"].contract["environments"],
        "openmathinstruct": {**math_env, "max_new_tokens": math_env["max_new_tokens"] + 1},
    })
    with pytest.raises(ValueError, match="openmathinstruct"):
        merge_corpus_contracts(_contracts(entries))


def test_a_protocol_version_that_changes_a_sources_rows_refuses():
    # openmathinstruct's row set changes at protocol 4 (train shards only), so a
    # maths task declared below it cannot be served under the other task's 9.
    entries = _entries()
    entries["corpus-math"] = _with(entries["corpus-math"], protocol_version=3)
    with pytest.raises(ValueError, match="openmathinstruct"):
        merge_corpus_contracts(_contracts(entries))


# --------------------------------------------------------------------------
# Startup: the registry check and the CLI
# --------------------------------------------------------------------------


def _resolve(entries, process):
    return resolve_corpus_task_configs(
        entries, ("corpus-math", "corpus-code"),
        profile_id=process["profile_id"], generation_contract=process,
    )


def test_compatible_jobs_from_two_templates_resolve_against_the_merge():
    entries = _entries()
    configs = _resolve(entries, merge_corpus_contracts(_contracts(entries)))
    assert [(c.task_id, c.entry.job_id) for c in configs] == [
        ("corpus-math", "math-v1"), ("corpus-code", "code-v1")]


def test_a_process_on_one_tasks_contract_refuses_with_the_remedy():
    entries = _entries()
    with pytest.raises(TaskConfigError) as caught:
        _resolve(entries, entries["corpus-math"].contract)
    assert "environments" in str(caught.value) and "tasks contract" in str(caught.value)


def test_a_different_model_refuses_at_startup():
    entries = _entries(model_revision="e" * 40)
    with pytest.raises(TaskConfigError, match="model_revision"):
        _resolve(entries, entries["corpus-math"].contract)


def _boot(monkeypatch, entries, process):
    import bittensor

    import reliquary.cli.main as cli_module
    import reliquary.constants as constants
    import reliquary.infrastructure.chain as chain
    import reliquary.validator.corpus_validator as corpus_validator

    monkeypatch.setattr(constants, "TASK_IDS", ("corpus-math", "corpus-code"))
    monkeypatch.setattr(constants, "PROTOCOL_PROFILE_ID", process["profile_id"])
    monkeypatch.setattr(constants, "PROTOCOL_GENERATION_CONTRACT", process)
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(corpus_validator, "run_corpus_validator", fake_run)
    return CliRunner().invoke(cli_module.app, ["validate"]), calls


def test_validate_starts_two_jobs_from_two_templates(monkeypatch, registry):  # noqa: F811
    entries = _entries()
    registry["entries"] = entries
    result, calls = _boot(monkeypatch, entries, merge_corpus_contracts(_contracts(entries)))
    assert result.exit_code == 0, (result.output, result.exception)
    assert [(e.task_id, e.job_id) for e, _ in calls[0]["jobs"]] == [
        ("corpus-math", "math-v1"), ("corpus-code", "code-v1")]


def test_validate_refuses_two_jobs_on_different_models(monkeypatch, registry):  # noqa: F811
    entries = _entries(model_id="Qwen/Other")
    registry["entries"] = entries
    result, calls = _boot(monkeypatch, entries, entries["corpus-math"].contract)
    assert result.exit_code == 4, (result.output, result.exception)
    assert calls == []


def test_tasks_contract_prints_the_merge_of_two_templates(registry):  # noqa: F811
    from reliquary.cli.main import app

    entries = _entries()
    registry["entries"] = entries
    result = CliRunner().invoke(app, ["tasks", "contract", "--task-id", "corpus-math",
                                      "--task-id", "corpus-code"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == merge_corpus_contracts(_contracts(entries))


def test_tasks_contract_refuses_an_rl_task_among_several(registry):  # noqa: F811
    from dataclasses import replace

    from reliquary.cli.main import app

    rl = replace(_rl_entry("logic", 0.5), contract=_entries()["corpus-code"].contract)
    registry["entries"] = {"corpus-math": _entries()["corpus-math"], "logic": rl}
    result = CliRunner().invoke(app, ["tasks", "contract", "--task-id", "corpus-math",
                                      "--task-id", "logic"])
    assert result.exit_code == 1
    assert "corpus" in result.output


# --------------------------------------------------------------------------
# Rendering: each job through its own template
# --------------------------------------------------------------------------


def test_each_source_renders_through_its_own_tasks_template_under_the_merge():
    from reliquary.protocol.profiles import profile_from_contract

    entries = _entries()
    merged = profile_from_contract(merge_corpus_contracts(_contracts(entries)))
    for task_id, source in (("corpus-math", "openmathinstruct"), ("corpus-code", "reliquary_code_v1")):
        own = profile_from_contract(entries[task_id].contract)
        assert (merged.environments[source].prompt_template.render(problem="P", contract="")
                == own.environments[source].prompt_template.render(problem="P", contract=""))
    math = merged.environments["openmathinstruct"].prompt_template
    code = merged.environments["reliquary_code_v1"].prompt_template
    assert math.template_id != code.template_id


def _job(job_id, source, renderer_id):
    from reliquary.corpus.job import parse_job

    return parse_job({**_manifest(), "job_id": job_id, "prompt_source": source,
                      "renderer_id": renderer_id})


def test_a_jobs_renderer_is_checked_against_its_own_tasks_template():
    from reliquary.protocol.profiles import profile_from_contract
    from reliquary.validator.corpus_service import CorpusPromptSourceError, renderer_for_job

    entries = _entries()
    math_profile = profile_from_contract(entries["corpus-math"].contract)
    renderer_for_job(_job("math-v1", "openmathinstruct", "openmathinstruct-step-by-step-v1"),
                     lambda text: [], profile=math_profile)
    with pytest.raises(CorpusPromptSourceError, match="openmathinstruct-step-by-step-v1"):
        renderer_for_job(_job("math-v1", "openmathinstruct", "reliquary-external-prompt-v1"),
                         lambda text: [], profile=math_profile)


def test_multi_job_refusal_refuses_a_source_the_process_declares_differently():
    from reliquary.validator.corpus_validator import multi_job_refusal

    entries = _entries()
    process = merge_corpus_contracts(_contracts(entries))
    pairs = [(entries["corpus-math"], SimpleNamespace(job_id="math-v1", prompt_source="openmathinstruct",
                                                      checkpoint_repo=MODEL, checkpoint_revision=REVISION,
                                                      checkpoint_sha256=CHECKPOINT)),
             (entries["corpus-code"], SimpleNamespace(job_id="code-v1", prompt_source="reliquary_code_v1",
                                                      checkpoint_repo=MODEL, checkpoint_revision=REVISION,
                                                      checkpoint_sha256=CHECKPOINT))]
    assert multi_job_refusal(pairs, process_contract=process) is None
    altered = json.loads(json.dumps(process))
    altered["environments"]["reliquary_code_v1"]["max_new_tokens"] += 1
    refusal = multi_job_refusal(pairs, process_contract=altered)
    assert refusal is not None and "corpus-code" in refusal and "reliquary_code_v1" in refusal


class _Stop(Exception):
    pass


def test_the_validator_resolves_each_jobs_renderer_with_its_own_contract(fake_r2, monkeypatch):  # noqa: F811
    import huggingface_hub

    import reliquary.protocol.profiles as profiles
    import reliquary.validator.corpus_service as corpus_service
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.protocol.profiles import profile_from_contract
    from reliquary.validator.corpus_validator import run_corpus_validator

    entries = _entries()
    for job_id, source, renderer in (("math-v1", "openmathinstruct", "chat-template-v1"),
                                     ("code-v1", "reliquary_code_v1", "chat-template-v1")):
        asyncio.run(job_store.write_job(
            {**_manifest(), "job_id": job_id, "prompt_source": source, "renderer_id": renderer,
             "checkpoint_repo": MODEL, "checkpoint_revision": REVISION}, None, **fake_r2))
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE",
                        profile_from_contract(merge_corpus_contracts(_contracts(entries))))
    seen = {}

    def renderer_for_job(job, encode, *, tokenizer=None, profile=None, environments=None):
        seen[job.job_id] = profile
        return SimpleNamespace()

    monkeypatch.setattr(corpus_service, "renderer_for_job", renderer_for_job)

    def stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stop)

    with pytest.raises(_Stop):
        asyncio.run(run_corpus_validator(
            jobs=[(entries["corpus-math"], 0.05), (entries["corpus-code"], 0.05)],
            wallet=None, netuid=0, signer_client=None, http_host="127.0.0.1", http_port=0,
            set_weights=False, registration_gate=False,
        ))
    assert seen["math-v1"] == profile_from_contract(entries["corpus-math"].contract)
    assert seen["code-v1"] == profile_from_contract(entries["corpus-code"].contract)
