"""One job, declared through the CLI and answered through the mounted router.

Every other test in this plan feeds a fixture into ONE side of a seam: the
CLI's manifest is read back by the test that wrote it, and the endpoint is
driven against a manifest the test built by hand. A field one side writes and
the other drops passes both of those and is only visible here, where the same
bytes travel the whole way -- `jobs create`, the job store, the registry, the
mount's mechanism gate, the endpoint, and the ledger object in the bucket.

Nothing here is stubbed except the tokenizer (a real one is a model download)
and the signature: the prompt source is a real installed environment, the
renderer is the real one its manifest names, and the store is the real store
over the CLI's own fake bucket.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from reliquary.cli.main import app as cli, mount_corpus_service
from reliquary.environment.agentic.renderers import renderer_for
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.validator.server import ValidatorServer

# The CLI's own fake bucket, fake registry and argument builder. A second copy
# here would drift from the one `jobs create` is actually proved against, and
# the point of this file is that both sides are the real ones.
from tests.unit.test_jobs_cli import (  # noqa: F401
    ACK,
    _create_args,
    _prompt_source,
    _rl_entry,
    _template,
    bucket,
    registry,
)

TASK_ID = "corpus-run"
CHECKPOINT = "a" * 64
EOS = 151645
SLOTS_PER_PROMPT = 3


class _DigitTokenizer:
    """Decodes an id to its own digits, so the text an honest miner would send
    is computable without downloading a real tokenizer."""

    def decode(self, ids, **kwargs):
        return "".join(str(i) for i in ids)


def _declared_args():
    """`jobs create` for a job whose prompts a validator can really render."""
    source = _prompt_source(_template())
    return _create_args(
        **{
            # The renderer the environment itself declares: a job naming any
            # other one would fail fidelity on every submission.
            "--renderer-id": ENVIRONMENT_SPECS[source].renderer_id,
            "--eos-token-id": str(EOS),
            "--slots-per-prompt": str(SLOTS_PER_PROMPT),
            "--prompt-count": "64",
        }
    )


def _renderer(job):
    # `encode` is never reached: only `initial_text` is, and it is text-only.
    return renderer_for(job.renderer_id, lambda text: [])


def _mount(entry, *, verify_signature=lambda request: True):
    """The production startup path: the store, the job id and the renderer are
    all derived from the declaration, not supplied by the test.

    `verify_signature` is injected because the production default is the real
    `verify_corpus_signature`, which fails closed on a fake hotkey like the
    ones these tests submit -- a test of the rest of the wiring has to get
    past it. One test below pins that default.
    """
    server = ValidatorServer()
    mounted = asyncio.run(
        mount_corpus_service(
            server,
            entry,
            tokenizer=_DigitTokenizer(),
            verify_signature=verify_signature,
        )
    )
    return server, mounted


def _mount_on(server, entry):
    return asyncio.run(
        mount_corpus_service(server, entry, tokenizer=_DigitTokenizer())
    )


def _rendered_prompt(job, prompt_index):
    """What the miner conditioned on, resolved the way a miner would: from the
    manifest's own `prompt_source`, through the manifest's own renderer."""
    environment = ENVIRONMENT_SPECS[job.prompt_source].create()
    return _renderer(job).initial_text(environment.get_task(prompt_index))


def _submit(client, job, *, prompt_index, filler, rendered_for=None, job_id=None):
    tokens = [filler] * 16 + [EOS]
    request = CorpusSubmissionRequest(
        job_id=job.job_id if job_id is None else job_id,
        miner_hotkey="5Hot",
        cursor=0,
        prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT,
        rendered_prompt=_rendered_prompt(
            job, prompt_index if rendered_for is None else rendered_for
        ),
        completions=[
            {
                # The trailing terminator leaves no trace in the text (T3-a).
                "tokens": tokens,
                "text": "".join(str(token) for token in tokens[:-1]),
            }
        ],
        signature="ok",
    )
    return client.post("/corpus/submit", json=request.model_dump()).json()


def test_a_declared_job_accepts_a_submission_and_fills_its_last_slot(
    bucket, registry
):
    """Declare a job through the CLI, mount the router on the task that pays
    for it, and submit until one prompt's slots are gone."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    result = CliRunner().invoke(cli, _declared_args())
    assert result.exit_code == 0, result.output

    # The submissions below name the job the REGISTRY says this task pays for,
    # not the one the test typed: the CLI writes that id twice, into two
    # objects, and a mismatch between them reads as `job_unknown` here.
    entry = registry["entries"][TASK_ID]
    job, _ = asyncio.run(job_store.read_job(entry.job_id))
    assert job is not None, "the CLI wrote no manifest for the task it declared"
    assert job.slots_per_prompt == SLOTS_PER_PROMPT

    server, mounted = _mount(entry)
    assert mounted is True

    with TestClient(server.app) as client:
        filled = [
            _submit(client, job, prompt_index=0, filler=index)
            for index in range(SLOTS_PER_PROMPT)
        ]
        assert [body["accepted"] for body in filled] == [True] * SLOTS_PER_PROMPT
        assert [body["slots_remaining"] for body in filled] == [2, 1, 0]

        exhausted = _submit(client, job, prompt_index=0, filler=99)
        assert exhausted["accepted"] is False
        assert exhausted["reason"] == "prompt_full"
        assert exhausted["slots_remaining"] == 0

        # The same tokens under another prompt: a slot is spent per prompt,
        # and the digest binds work to the question it answers. Without this
        # the test above is satisfied by a job that refuses everything.
        elsewhere = _submit(client, job, prompt_index=1, filler=0)
        assert elsewhere["accepted"] is True
        assert elsewhere["slots_remaining"] == SLOTS_PER_PROMPT - 1

        # The accepted submissions above already prove the validator resolves
        # the manifest's own prompt source; this is the refusing direction, so
        # a mount that renders some other environment cannot pass both.
        astray = _submit(client, job, prompt_index=1, filler=7, rendered_for=2)
        assert astray["accepted"] is False
        assert astray["reason"] == "prompt_not_faithful"

        # The job id the route serves comes from the registry entry, so work
        # declared under another task's cap is refused here, by name.
        foreign = _submit(client, job, prompt_index=2, filler=7, job_id="other-job")
        assert foreign["accepted"] is False
        assert foreign["reason"] == "job_not_served"

    # What the endpoint derived has to have reached the bucket, or the next
    # validator to read these ledgers sells the same slots again.
    raw, _ = bucket.objects[f"reliquary/corpus/jobs/{job.job_id}/ledgers.json"]
    ledgers = json.loads(raw)
    assert ledgers["slots"] == {"0": SLOTS_PER_PROMPT, "1": 1}
    assert len(ledgers["seen"]) == SLOTS_PER_PROMPT + 1


def test_the_startup_path_binds_the_real_corpus_verifier(
    bucket, registry, monkeypatch
):
    """A caller that leaves `verify_signature` unset gets the real
    `verify_corpus_signature`, not a stub -- proven by swapping that name for
    one that accepts and watching the swap reach the mounted route."""
    import reliquary.protocol.signatures as signatures

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]
    job, _ = asyncio.run(job_store.read_job(entry.job_id))

    calls = []
    monkeypatch.setattr(
        signatures,
        "verify_corpus_signature",
        lambda request: calls.append(request) or True,
    )

    server, mounted = _mount(entry, verify_signature=None)
    assert mounted is True

    with TestClient(server.app) as client:
        body = _submit(client, job, prompt_index=0, filler=1)

    assert len(calls) == 1
    assert body["accepted"] is True


def test_a_declared_job_with_no_manifest_refuses_to_start(bucket, registry):
    """The task is declared and would take its share of the pool, so a job the
    store cannot produce is a startup refusal -- not a route that 404s while
    the task keeps its cap."""
    from reliquary.validator.task_config import TaskConfigError

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]
    bucket.objects.pop(f"reliquary/corpus/jobs/{entry.job_id}.json")

    with pytest.raises(TaskConfigError) as caught:
        _mount(entry)

    assert entry.job_id in str(caught.value)


def test_a_validator_on_an_rl_task_exposes_no_corpus_route(bucket, registry):
    """The mount is gated on the resolved task, so an RL validator serves no
    route that sells slots nobody declared -- and neither does the legacy
    fallback, whose task config carries no registry entry at all."""
    server = ValidatorServer()

    assert _mount_on(server, _rl_entry("default", 1.0)) is False
    assert _mount_on(server, None) is False

    assert [
        path
        for path in (getattr(route, "path", "") for route in server.app.routes)
        if "corpus" in path
    ] == []
    with TestClient(server.app) as client:
        assert client.post("/corpus/submit", json={}).status_code == 404


def test_the_server_gate_does_not_depend_on_the_startup_path(bucket, registry):
    """The mechanism gate lives on the mount itself, so a second caller cannot
    open the route by skipping the CLI's half of it."""
    server = ValidatorServer()
    arguments = {
        "store": None,
        "tokenizer": _DigitTokenizer(),
        "renderer": renderer_for("reliquary-jsonl-tools-v1", lambda text: []),
        "verify_signature": lambda request: True,
    }

    assert server.mount_corpus_router(_rl_entry("default", 1.0), **arguments) is False
    assert server.mount_corpus_router(None, **arguments) is False
    assert [
        path
        for path in (getattr(route, "path", "") for route in server.app.routes)
        if "corpus" in path
    ] == []


def _corpus_entry_this_binary_can_resolve(job_id):
    """A registry entry `resolve_task_config` accepts: this process's own task
    id, profile and contract digest, declaring a corpus job."""
    from dataclasses import asdict

    from reliquary.constants import (
        PROTOCOL_GENERATION_CONTRACT,
        PROTOCOL_PROFILE_ID,
        TASK_ID as PROCESS_TASK_ID,
    )
    from reliquary.environment.abi import canonical_sha256
    from reliquary.shared.task_registry import (
        MECHANISM_CORPUS_GENERATION,
        TaskEntry,
    )
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    params = asdict(PRODUCTION_PRICE_PARAMS)
    params["cap"] = params["floor"] = 0.3
    return TaskEntry(
        task_id=PROCESS_TASK_ID,
        profile_id=PROTOCOL_PROFILE_ID,
        profile_sha256=canonical_sha256(PROTOCOL_GENERATION_CONTRACT),
        mechanism=MECHANISM_CORPUS_GENERATION,
        params=params,
        status="active",
        retired_at=None,
        job_id=job_id,
    )


def _seed_manifest(job_id, *, renderer_id=None):
    """The job the entry above names, written through the real store.

    `renderer_id` defaults to the source's own declared one; a caller may
    pass another name -- `build_job_manifest` does not check that an episode
    job's `renderer_id` names a real renderer (only `renderer_for_job` does,
    at startup), so this can seed a manifest that parses and declares
    cleanly but cannot actually be rendered.
    """
    from reliquary.cli.main import build_job_manifest

    source = _prompt_source(_template())
    manifest = build_job_manifest(
        job_id=job_id,
        checkpoint_repo="org/Frozen",
        checkpoint_revision="abc123",
        checkpoint_sha256=CHECKPOINT,
        prompt_source=source,
        prompt_count=64,
        renderer_id=ENVIRONMENT_SPECS[source].renderer_id if renderer_id is None else renderer_id,
        eos_token_id=EOS,
        slots_per_prompt=SLOTS_PER_PROMPT,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_new_tokens=2,
        max_new_tokens=4096,
        n=1,
        grader_id=None,
        threshold=None,
        prompt_order="free",
        deadline_round=None,
    )
    asyncio.run(job_store.write_job(manifest, None))


def test_a_server_that_declines_the_mount_is_not_walked_past(bucket, registry):
    """Server and helper apply the same rule to the same entry, so a refusal
    means they disagree -- which must stop the process, not leave a corpus
    task running with no route."""
    from reliquary.validator.task_config import TaskConfigError

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]

    class _DecliningServer:
        def mount_corpus_router(self, entry, **kwargs):
            return False

    with pytest.raises(TaskConfigError) as caught:
        _mount_on(_DecliningServer(), entry)

    assert entry.job_id in str(caught.value)


def _boot_validate_dispatching_to_corpus_validator(monkeypatch, *, side_effect=None):
    """Run `validate` on a declared corpus task, with `run_corpus_validator`
    itself replaced.

    Task 8 moved a corpus task off the RL boot entirely (spec G2): `validate`
    now branches to `run_corpus_validator` before any RL machinery -- model,
    chain, proof plane -- is even imported, so there is no longer a
    `ValidatorServer` this path builds for a test to inspect.
    `run_corpus_validator` itself needs a GPU and R2 (covered in
    `test_corpus_validator.py`, and end to end on real hardware in Task 11);
    what only a full CLI boot can prove is the dispatch itself -- the
    resolved entry and cap reach it, and RL loading is never touched.
    """
    from types import SimpleNamespace

    import bittensor
    import reliquary.cli.main as cli_module
    import reliquary.infrastructure.chain as chain
    import reliquary.shared.modeling as modeling
    import reliquary.validator.corpus_validator as corpus_validator

    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    # The one direct evidence the RL boot was skipped: it must never load.
    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **kw: pytest.fail("loaded")
    )

    calls = []

    async def fake_run_corpus_validator(**kwargs):
        calls.append(kwargs)
        if side_effect is not None:
            await side_effect(**kwargs)

    monkeypatch.setattr(corpus_validator, "run_corpus_validator", fake_run_corpus_validator)

    result = CliRunner().invoke(cli_module.app, ["validate"])
    return result, calls


def test_the_validator_startup_path_dispatches_to_run_corpus_validator(
    monkeypatch, registry
):
    """`validate` on a corpus task has to reach `run_corpus_validator` with
    the REGISTRY's resolved entry and cap, not fall through to the RL
    service Task 8 supersedes for this mechanism -- a mistake there is
    silent, its only evidence a validator that boots holding this task's
    share and serving nothing.
    """
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    registry["entries"] = {entry.task_id: entry}

    result, calls = _boot_validate_dispatching_to_corpus_validator(monkeypatch)

    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1
    assert calls[0]["entry"].job_id == "swe-v1"
    assert calls[0]["cap"] == pytest.approx(0.3)
    assert calls[0]["set_weights"] is False


def test_a_corpus_startup_refusal_exits_four_rather_than_a_traceback(
    monkeypatch, registry
):
    """`run_corpus_validator` raises `RuntimeError` for every named startup
    refusal (an unenforced contract, a checkpoint mismatch, a missing
    manifest). The CLI's own `except RuntimeError` is what turns that into
    the CRITICAL line and exit code 4 every other startup refusal uses,
    not a bare traceback."""
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    registry["entries"] = {entry.task_id: entry}

    async def _refuse(**kwargs):
        raise RuntimeError("the task contract's toploc proof is not enforce")

    result, calls = _boot_validate_dispatching_to_corpus_validator(
        monkeypatch, side_effect=_refuse
    )

    assert result.exit_code == 4, (result.output, result.exception)
    assert len(calls) == 1


def test_a_corpus_renderer_that_cannot_build_exits_four_before_any_download(
    monkeypatch, bucket, registry
):
    """The test above replaces `run_corpus_validator` outright, which proves
    the CLI's `except RuntimeError` but not that `run_corpus_validator`
    itself converts a renderer failure into one -- that gap is what let
    `CorpusPromptSourceError` escape as a bare traceback (exit 1) instead of
    exit 4. This drives the REAL `run_corpus_validator`, with only its
    download and model-load calls stubbed to fail the test if reached: the
    renderer is resolved before any of them (a controller ruling -- this
    refusal must cost seconds, not a checkpoint download and a GPU load), so
    none should fire."""
    from types import SimpleNamespace

    import bittensor
    import huggingface_hub
    import reliquary.cli.main as cli_module
    import reliquary.corpus.encoding as encoding
    import reliquary.infrastructure.chain as chain
    import reliquary.shared.modeling as modeling
    from reliquary.validator import corpus_service

    _seed_manifest("swe-v1")
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    registry["entries"] = {entry.task_id: entry}

    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)

    def _unrenderable(job, encode, **kwargs):
        raise corpus_service.CorpusPromptSourceError(
            f"prompt source {job.prompt_source!r} renders through another template"
        )

    monkeypatch.setattr(corpus_service, "renderer_for_job", _unrenderable)

    # None of these may be reached: the renderer refusal must come first.
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("downloaded")
    )
    monkeypatch.setattr(
        encoding, "checkpoint_fingerprint", lambda *a, **kw: pytest.fail("fingerprinted")
    )
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *a, **kw: pytest.fail("tokenized"))
    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **kw: pytest.fail("loaded")
    )
    monkeypatch.setattr(
        modeling, "load_text_only_model", lambda *a, **kw: pytest.fail("loaded")
    )

    result = CliRunner().invoke(cli_module.app, ["validate"])

    assert result.exit_code == 4, (result.output, result.exception)


def test_a_corpus_job_naming_an_unknown_renderer_exits_four_before_any_download(
    monkeypatch, bucket, registry
):
    """The test above stubs `renderer_for_job` itself to raise
    `CorpusPromptSourceError`; it cannot prove the OTHER way a job's declared
    renderer is unbuildable. `jobs create` never checks that an episode job's
    `renderer_id` names a real renderer -- `build_job_manifest` only resolves
    the prompt SOURCE, and an episode source's own renderer is unchecked
    there (`resolve_prompt_source`'s docstring: "there is no second
    authority to disagree with"). Only `renderer_for` checks the name, at
    startup, and it raises a plain `ValueError`, not `CorpusPromptSourceError`
    -- so this seeds a manifest with a name no renderer answers to and drives
    the real `renderer_for_job` -> `renderer_for` chain, the only way to prove
    that `ValueError` is caught too."""
    from types import SimpleNamespace

    import bittensor
    import huggingface_hub
    import reliquary.cli.main as cli_module
    import reliquary.corpus.encoding as encoding
    import reliquary.infrastructure.chain as chain
    import reliquary.shared.modeling as modeling

    _seed_manifest("swe-v1", renderer_id="no-such-renderer")
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    registry["entries"] = {entry.task_id: entry}

    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)

    # None of these may be reached: the renderer refusal must come first.
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **kw: pytest.fail("downloaded")
    )
    monkeypatch.setattr(
        encoding, "checkpoint_fingerprint", lambda *a, **kw: pytest.fail("fingerprinted")
    )
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *a, **kw: pytest.fail("tokenized"))
    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **kw: pytest.fail("loaded")
    )
    monkeypatch.setattr(
        modeling, "load_text_only_model", lambda *a, **kw: pytest.fail("loaded")
    )

    result = CliRunner().invoke(cli_module.app, ["validate"])

    assert result.exit_code == 4, (result.output, result.exception)


def test_the_mount_refuses_the_task_config_wrapper(bucket, registry):
    """The call site's own failure mode: `TaskConfig` has no `.mechanism`, so
    a mount gated on `getattr(entry, "mechanism", None)` would decline a
    declared corpus task in silence and the validator would boot, hold its
    share of the pool and serve nothing."""
    from reliquary.validator.task_config import TaskConfig, TaskConfigError

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]
    config = TaskConfig(
        task_id=entry.task_id,
        entry=entry,
        price_params=None,
        emission_cap=0.3,
        env_caps={},
    )

    with pytest.raises(TaskConfigError) as caught:
        _mount_on(ValidatorServer(), config)

    assert "TaskConfig" in str(caught.value)


def test_a_corpus_contract_for_another_source_boots_without_the_rl_environment_mix(
    monkeypatch, registry, tmp_path
):
    """Final review, finding 2: `--environments` defaults to openmathinstruct,
    which a corpus contract for another source does not declare. The corpus
    branch must be taken before the RL mix is resolved (or the code grader
    started), or `validate` refuses a correctly declared corpus task."""
    from dataclasses import replace
    import json as _json

    import reliquary.cli.main as cli_module
    import reliquary.constants as constants
    from reliquary.cli.main import build_corpus_task_entry
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR, resolve_protocol_profile

    entry = build_corpus_task_entry(
        task_id=constants.TASK_ID, job_id="logic-v1",
        from_profile="qwen3-4b-base-dapo-reliquary-v1", model_id="Qwen/Qwen3-4B-Base",
        model_revision="main", model_architecture="Qwen3ForCausalLM",
        prompt_source="reliquary_logic_v2", cap=0.1, overrides={},
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(_json.dumps(entry.contract))
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, str(contract_path))
    # What importing under RELIQUARY_TASK_CONTRACT gives the process.
    profile = resolve_protocol_profile()
    assert "openmathinstruct" not in profile.environments
    monkeypatch.setattr(cli_module, "ACTIVE_PROTOCOL_PROFILE", profile)
    monkeypatch.setattr(constants, "PROTOCOL_PROFILE_ID", profile.profile_id)
    monkeypatch.setattr(constants, "PROTOCOL_GENERATION_CONTRACT", profile.to_generation_contract())
    registry["entries"] = {entry.task_id: replace(entry, profile_id=profile.profile_id)}

    mix_calls = []
    real_mix = cli_module._resolve_cli_environment_mix

    def spy_mix(value):
        mix_calls.append(value)
        return real_mix(value)

    monkeypatch.setattr(cli_module, "_resolve_cli_environment_mix", spy_mix)
    monkeypatch.setattr(cli_module, "_ensure_grader_running",
                        lambda *a, **kw: pytest.fail("grader started"))

    result, calls = _boot_validate_dispatching_to_corpus_validator(monkeypatch)

    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1 and calls[0]["entry"].job_id == "logic-v1"
    assert mix_calls == []
