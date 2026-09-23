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

    `verify_signature` is injected because the production default refuses
    everything -- `protocol/signatures.py` carries no corpus binding yet -- and
    a test of the rest of the wiring has to get past it. One test below pins
    that default.
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
                "termination": "eos",
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


def test_the_startup_path_mounts_nothing_it_cannot_authenticate(bucket, registry):
    """`protocol/signatures.py` carries no corpus binding, so the route is
    wired with a verifier that refuses everything. The endpoint exists and
    nothing can be admitted through it -- not a stub that accepts. The reason
    says this validator cannot verify, not that the miner signed badly."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]
    job, _ = asyncio.run(job_store.read_job(entry.job_id))

    server, mounted = _mount(entry, verify_signature=None)
    assert mounted is True

    with TestClient(server.app) as client:
        body = _submit(client, job, prompt_index=0, filler=1)

    assert body["accepted"] is False
    assert body["reason"] == "signature_unverifiable"


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


def _seed_manifest(job_id):
    """The job the entry above names, written through the real store."""
    from reliquary.cli.main import build_job_manifest

    source = _prompt_source(_template())
    manifest = build_job_manifest(
        job_id=job_id,
        checkpoint_repo="org/Frozen",
        checkpoint_revision="abc123",
        checkpoint_sha256=CHECKPOINT,
        prompt_source=source,
        prompt_count=64,
        renderer_id=ENVIRONMENT_SPECS[source].renderer_id,
        eos_token_id=EOS,
        slots_per_prompt=SLOTS_PER_PROMPT,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_new_tokens=1,
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


def test_the_validator_startup_path_serves_the_route(monkeypatch, bucket):
    """`mount_corpus_service` can be perfect and the validator still serve
    nothing: the call site has to pass the resolved ENTRY and the real server,
    and a mistake there is silent -- its only evidence is a missing log line.

    So this boots `validate --train` onto a corpus task with the RL machinery
    mocked the way `test_remote_proof_controller` mocks it, and asks the
    server that startup actually built whether the route is on it.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import bittensor
    import reliquary.cli.main as cli_module
    import reliquary.constants as constants
    import reliquary.infrastructure.chain as chain
    import reliquary.infrastructure.task_registry_store as task_registry_store
    import reliquary.shared.modeling as modeling
    import reliquary.validator.remote_proof as remote
    import reliquary.validator.service as service_module
    import reliquary.validator.weight_only as weights
    from reliquary.validator.proof_worker import ProofModelProxy

    from tests.unit.test_remote_proof_controller import REV, MetadataPool

    _seed_manifest("swe-v1")
    entry = _corpus_entry_this_binary_can_resolve("swe-v1")
    monkeypatch.setattr(
        task_registry_store,
        "read_registry",
        AsyncMock(return_value=({entry.task_id: entry}, None)),
    )

    # Everything below this line is the RL boot, mocked off the box: no CUDA,
    # no model, no chain, no HF.
    monkeypatch.setenv("RELIQUARY_PROOF_EXECUTOR_MODE", "remote")
    monkeypatch.setattr(constants, "DETACHED_TRAINER", True)
    monkeypatch.setattr(constants, "KL_BASE_MODEL", "")
    monkeypatch.setattr(
        cli_module, "_resolve_cli_environment_mix", lambda _v: [("fake", 1)]
    )
    monkeypatch.setattr(cli_module, "_v3_activation_checkpoint_revision", lambda *a: REV)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(
        modeling, "load_text_generation_model", lambda *a, **kw: pytest.fail("loaded")
    )
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    pool = MetadataPool()
    pool.start = lambda: None
    pool.dispatch_devices = ("cuda:0",)
    pool.proxies = lambda: {device: ProofModelProxy(device) for device in pool.dispatch_devices}
    pool.qualify = lambda revision: {"revision": revision}
    monkeypatch.setattr(remote.RemoteProofPool, "from_environment", lambda **kw: pool)

    servers = []

    class Service:
        def __init__(self, wallet, model, tokenizer, **kwargs):
            # The real service builds one; what this test is about is what
            # startup then does with it.
            self.server = ValidatorServer()
            servers.append(self.server)

        async def run(self, subtensor):
            pass

    class WeightSetter:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            pass

    monkeypatch.setattr(service_module, "ValidationService", Service)
    monkeypatch.setattr(weights, "WeightOnlyValidator", WeightSetter)

    result = CliRunner().invoke(cli_module.app, ["validate", "--resume-from", f"sha:{REV}"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert len(servers) == 1
    paths = [getattr(route, "path", "") for route in servers[0].app.routes]
    assert "/corpus/submit" in paths


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
