"""Each job's task contract, served by a validator serving one or several jobs,
and the miner fetching the one of the job it mines."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.validator.corpus_validator import build_corpus_app, build_corpus_jobs_app
from tests.unit.test_corpus_service import (  # noqa: F401
    _Tokenizer, _r2_client, fake_r2, seeded_job,
)

MATH = {"model_id": "org/Frozen", "profile_id": "corpus-math", "proofs": [{"scheme": "toploc-v1"}]}
CODE = {"model_id": "org/Frozen", "profile_id": "corpus-code", "proofs": [{"scheme": "toploc-v1"}]}


def _served(seeded_job, job_id, contract):
    return SimpleNamespace(
        entry=SimpleNamespace(task_id=f"task-{job_id}", job_id=job_id, contract=contract),
        job=seeded_job.job, renderer=seeded_job.renderer,
        auditor=SimpleNamespace(enqueue=lambda sid: None), is_banned=None,
    )


def _two_jobs(seeded_job, order=("math-v1", "code-v1")):
    contracts = {"math-v1": MATH, "code-v1": CODE}
    app = build_corpus_jobs_app(
        jobs=[_served(seeded_job, job_id, contracts[job_id]) for job_id in order],
        store=seeded_job.store, records=None, tokenizer=_Tokenizer(),
        verify_signature=lambda r: True, proof_chunk_tokens=None,
        prompt_job_for=seeded_job.prompt_job_for,
    )
    return TestClient(app)


def test_each_job_serves_its_own_tasks_contract(seeded_job):
    client = _two_jobs(seeded_job)
    assert client.get("/corpus/jobs/math-v1/contract").json() == MATH
    assert client.get("/corpus/jobs/code-v1/contract").json() == CODE


def test_an_unserved_jobs_contract_is_404(seeded_job):
    response = _two_jobs(seeded_job).get("/corpus/jobs/nope/contract")
    assert response.status_code == 404
    assert response.json()["detail"] == "corpus_job_not_served"


@pytest.mark.parametrize("order", [("math-v1", "code-v1"), ("code-v1", "math-v1")])
def test_the_legacy_contract_path_serves_the_first_listed_jobs(seeded_job, order):
    contracts = {"math-v1": MATH, "code-v1": CODE}
    response = _two_jobs(seeded_job, order).get("/corpus/contract")
    assert response.status_code == 200 and response.json() == contracts[order[0]]


def test_startup_logs_which_job_the_legacy_paths_serve(seeded_job, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_validator"):
        _two_jobs(seeded_job, ("code-v1", "math-v1"))
    assert "legacy paths serve job code-v1" in caplog.text


def test_one_job_serves_its_contract_on_both_paths(seeded_job):
    app = build_corpus_app(
        entry=SimpleNamespace(task_id="corpus-math", job_id="swe-v1"), job=seeded_job.job,
        store=seeded_job.store, records=None, tokenizer=_Tokenizer(), renderer=seeded_job.renderer,
        verify_signature=lambda r: True, auditor=SimpleNamespace(enqueue=lambda sid: None),
        proof_chunk_tokens=None, prompt_job_for=seeded_job.prompt_job_for, contract=MATH,
    )
    client = TestClient(app)
    assert client.get("/corpus/contract").json() == MATH
    assert client.get("/corpus/jobs/swe-v1/contract").json() == MATH


# --------------------------------------------------------------------------
# The miner's fetch
# --------------------------------------------------------------------------


JOBS = {
    "math-v1": {"job_id": "math-v1", "checkpoint_repo": "org/M", "checkpoint_revision": "r1"},
    "code-v1": {"job_id": "code-v1", "checkpoint_repo": "org/M", "checkpoint_revision": "r1"},
}
CONTRACTS = {
    "math-v1": {"model_id": "org/M", "model_revision": "r1", "profile_id": "corpus-math",
                "proofs": [{"scheme": "toploc-v1", "mode": "enforce"}]},
    "code-v1": {"model_id": "org/M", "model_revision": "r1", "profile_id": "corpus-code",
                "proofs": [{"scheme": "toploc-v1", "mode": "enforce"}]},
}


def _multi_job_validator():
    import httpx

    seen = []

    def handle(request):
        path = request.url.path
        seen.append(path)
        # The legacy paths answer for the first job listed.
        if path == "/corpus/job":
            return httpx.Response(200, json=JOBS["math-v1"])
        if path == "/corpus/contract":
            return httpx.Response(200, json=CONTRACTS["math-v1"])
        if path == "/corpus/jobs":
            return httpx.Response(200, json={"jobs": sorted(JOBS)})
        _, _, _, job_id, what = path.split("/")
        if job_id not in JOBS:
            return httpx.Response(404, json={"detail": "corpus_job_not_served"})
        return httpx.Response(200, json=JOBS[job_id] if what == "job" else CONTRACTS[job_id])

    return httpx.MockTransport(handle), seen


def _mine(monkeypatch, tmp_path, *args):
    import os

    import httpx
    from typer.testing import CliRunner

    from reliquary.cli import main as cli
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    transport, seen = _multi_job_validator()
    real_client = httpx.Client

    class _Restarted(Exception):
        pass

    execs = []

    def _execv(executable, argv):
        execs.append(os.environ.get(TASK_CONTRACT_ENV_VAR))
        raise _Restarted

    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, "unset-by-this-test")
    monkeypatch.delenv(TASK_CONTRACT_ENV_VAR)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport,
                                                                  base_url=kw["base_url"]))
    monkeypatch.setattr(os, "execv", _execv)
    result = CliRunner().invoke(cli.app, ["corpus", "mine", "--validator-url", "http://v", *args])
    return result, execs, seen, _Restarted


def test_a_miner_with_job_id_fetches_that_jobs_own_contract(monkeypatch, tmp_path):
    result, execs, seen, restarted = _mine(monkeypatch, tmp_path, "--job-id", "code-v1")

    assert isinstance(result.exception, restarted), result.output
    (path,) = execs
    assert json.loads(open(path).read()) == CONTRACTS["code-v1"]
    assert path.endswith("code-v1.contract.json")
    assert "/corpus/jobs/code-v1/contract" in seen and "/corpus/contract" not in seen


def test_a_miner_without_job_id_on_several_jobs_takes_the_default_jobs_contract(monkeypatch, tmp_path):
    result, execs, seen, restarted = _mine(monkeypatch, tmp_path)

    assert isinstance(result.exception, restarted), result.output
    (path,) = execs
    assert json.loads(open(path).read()) == CONTRACTS["math-v1"]
    assert seen[:2] == ["/corpus/job", "/corpus/contract"]


def test_a_miner_naming_an_unserved_job_exits_with_the_list(monkeypatch, tmp_path):
    result, execs, _, _ = _mine(monkeypatch, tmp_path, "--job-id", "nope")

    assert result.exit_code == 2, (result.output, result.exception)
    assert execs == [] and "nope" in result.output and "math-v1" in result.output


@pytest.mark.parametrize("job_id", ["math-v1", "code-v1"])
def test_the_client_reads_each_jobs_contract(job_id):
    import httpx

    from reliquary.miner.corpus_miner import HttpCorpusClient

    transport, _ = _multi_job_validator()
    client = HttpCorpusClient(httpx.Client(transport=transport, base_url="http://v"), job_id=job_id)
    assert client.contract() == CONTRACTS[job_id]
