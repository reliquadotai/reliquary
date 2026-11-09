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

from reliquary.cli.main import app as cli
from reliquary.environment.agentic.renderers import renderer_for
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from tests.unit.test_corpus_ledger_v2 import seen_union

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
    params["settlement"] = "period-ema-v1"
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


def test_the_corpus_validator_migrates_a_v1_ledger_right_after_reading_the_job(
    monkeypatch, bucket, registry
):
    """The dedicated corpus validator migrates before anything else: driven
    through the real `run_corpus_validator` up to its renderer refusal, the
    first step after the migration, the ledger is already v2."""
    from types import SimpleNamespace

    import bittensor
    import reliquary.cli.main as cli_module
    import reliquary.infrastructure.chain as chain
    from reliquary.validator import corpus_service

    _seed_manifest("swe-v1")
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    registry["entries"] = {entry.task_id: entry}
    key = "reliquary/corpus/jobs/swe-v1/ledgers.json"
    seen = sorted(f"{i:064x}" for i in range(3))
    bucket.objects[key] = (
        json.dumps({"slots": {"0": 3}, "cursors": {}, "seen": seen}).encode(), '"v1"'
    )
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)

    def _stop_here(job, encode, **kwargs):
        raise corpus_service.CorpusPromptSourceError("stop after the migration")

    monkeypatch.setattr(corpus_service, "renderer_for_job", _stop_here)

    result = CliRunner().invoke(cli_module.app, ["validate"])

    assert result.exit_code == 4, (result.output, result.exception)
    assert json.loads(bucket.objects[key][0])["schema"] == "reliquary/corpus-ledgers/v2"
    assert seen_union(bucket.objects, "swe-v1") == set(seen)


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


