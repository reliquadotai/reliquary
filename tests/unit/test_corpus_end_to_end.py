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
    nothing can be admitted through it -- not a stub that accepts."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(cli, _declared_args()).exit_code == 0
    entry = registry["entries"][TASK_ID]
    job, _ = asyncio.run(job_store.read_job(entry.job_id))

    server, mounted = _mount(entry, verify_signature=None)
    assert mounted is True

    with TestClient(server.app) as client:
        body = _submit(client, job, prompt_index=0, filler=1)

    assert body["accepted"] is False
    assert body["reason"] == "bad_signature"


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
