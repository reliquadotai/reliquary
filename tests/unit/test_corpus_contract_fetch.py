"""The miner takes the task's contract from the validator when none is given,
and refuses one that does not describe the job's checkpoint."""

import json
from types import SimpleNamespace

import pytest

from reliquary.miner.corpus_miner import CorpusContractError, save_served_contract

JOB = SimpleNamespace(job_id="code-v1", checkpoint_repo="org/M", checkpoint_revision="r1")
CONTRACT = {"model_id": "org/M", "model_revision": "r1",
            "proofs": [{"scheme": "toploc-v1", "chunk_tokens": 32, "topk": 128}]}


def test_a_matching_contract_is_saved_for_the_job(tmp_path):
    path = save_served_contract(CONTRACT, JOB, tmp_path)
    assert json.loads(path.read_text()) == CONTRACT
    assert path.parent == tmp_path and "code-v1" in path.name


@pytest.mark.parametrize("change", [
    {"model_id": "org/Other"},
    {"model_revision": "r2"},
    {"proofs": [{"scheme": "grail"}]},
    {"proofs": []},
])
def test_a_contract_for_another_checkpoint_or_without_toploc_is_refused(tmp_path, change):
    with pytest.raises(CorpusContractError):
        save_served_contract({**CONTRACT, **change}, JOB, tmp_path)
    assert not list(tmp_path.iterdir())


def test_corpus_mine_without_a_contract_fetches_it_and_restarts(monkeypatch, tmp_path):
    """The active profile is fixed when the process imports it, so the miner
    saves the served contract and restarts itself with it set."""
    import os
    from typer.testing import CliRunner

    from reliquary.cli import main as cli
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    job = {"job_id": "code-v1", "checkpoint_repo": "org/M", "checkpoint_revision": "r1"}
    served = {"/corpus/job": job, "/corpus/contract": CONTRACT}

    class _Response:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get(self, path):
            return _Response(served[path])

    class _Restarted(Exception):
        pass

    execs = []

    def _execv(executable, argv):
        execs.append((executable, argv, os.environ.get(TASK_CONTRACT_ENV_VAR)))
        raise _Restarted

    monkeypatch.delenv(TASK_CONTRACT_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("httpx.Client", _Client)
    monkeypatch.setattr(os, "execv", _execv)
    result = CliRunner().invoke(cli.app, ["corpus", "mine", "--validator-url", "http://v"])
    assert isinstance(result.exception, _Restarted), result.output
    (_, _, contract_path), = execs
    assert json.loads(open(contract_path).read()) == CONTRACT
