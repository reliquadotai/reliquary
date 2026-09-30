"""The real `run_corpus_validator` startup for two jobs whose entries carry
contracts from two templates, driven through the rehearsal's own validator
child (`scripts/corpus_e2e.py run_validator`), stubbed only at the GPU."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_corpus_service import CHECKPOINT, _manifest, _r2_client, fake_r2  # noqa: F401
from tests.unit.test_corpus_validator import fixed_drand_chain, wired_records  # noqa: F401

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "f" * 40
MATH_TEMPLATE = "qwen3-4b-base-dapo-reliquary-v1"
CODE_TEMPLATE = "teutonic-9b-reliquary-suite-v9-dev1"


class _Stop(Exception):
    pass


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=200_000)


def _declare(tmp_path, fake_r2):  # noqa: F811
    from scripts.corpus_e2e import declare_task

    args = SimpleNamespace(base_profile=MATH_TEMPLATE, honest_model=MODEL,
                           model_architecture="Qwen3_5ForConditionalGeneration",
                           prompt_source="openmathinstruct", cap=0.1)
    contracts = {}
    for key, task_id, job_id, source, template in (
        ("a", "corpus-math", "math-v1", "openmathinstruct", None),
        ("b", "corpus-code", "code-v1", "reliquary_code_v1", CODE_TEMPLATE),
    ):
        contracts[key] = declare_task(tmp_path, args, task_id=task_id, job_id=job_id,
                                      revision=REVISION, prompt_source=source,
                                      suffix=f"-{key}", base_profile=template)
        asyncio.run(job_store.write_job(
            {**_manifest(), "job_id": job_id, "prompt_source": source,
             "renderer_id": "chat-template-v1", "checkpoint_repo": MODEL,
             "checkpoint_revision": REVISION}, None, **fake_r2))
    return contracts


@pytest.fixture
def started(tmp_path, fake_r2, wired_records, fixed_drand_chain, monkeypatch):  # noqa: F811
    import huggingface_hub
    import uvicorn

    import reliquary.corpus.encoding as encoding
    import reliquary.protocol.profiles as profiles
    import reliquary.shared.modeling as modeling
    from reliquary.protocol.profiles import profile_from_contract
    from reliquary.validator import corpus_auditor
    from reliquary.validator.task_config import merge_corpus_contracts
    from scripts import corpus_e2e

    contracts = _declare(tmp_path, fake_r2)
    merged = merge_corpus_contracts({"corpus-math": contracts["a"], "corpus-code": contracts["b"]})
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE", profile_from_contract(merged))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None: "/nonexistent")
    monkeypatch.setattr(encoding, "checkpoint_fingerprint", lambda d: CHECKPOINT)
    monkeypatch.setattr(modeling, "load_tokenizer", lambda path: SimpleNamespace())
    monkeypatch.setattr(modeling, "load_text_only_model", lambda path, **kw: _Model())
    built = {}

    async def idle(self):
        await asyncio.sleep(3600)

    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)

    class _Server:
        def __init__(self, config):
            built["app"] = config.app

        async def serve(self):
            await asyncio.sleep(0.05)
            raise _Stop()

    monkeypatch.setattr(uvicorn, "Server", _Server)
    args = SimpleNamespace(
        task_id="corpus-math,corpus-code", job_id="math-v1,code-v1", cap="0.1,0.2", port=0,
        entry=f"{tmp_path / 'entry-a.json'},{tmp_path / 'entry-b.json'}",
    )
    with pytest.raises(_Stop):
        corpus_e2e.run_validator(args)
    return SimpleNamespace(app=built["app"], contracts=contracts)


def test_two_jobs_from_two_templates_start_and_serve(started):
    client = TestClient(started.app)
    assert client.get("/corpus/jobs").json() == {"jobs": ["code-v1", "math-v1"]}
    assert client.get("/corpus/jobs/math-v1/contract").json() == started.contracts["a"]
    assert client.get("/corpus/jobs/code-v1/contract").json() == started.contracts["b"]


def test_the_rehearsal_loads_each_entrys_contract(tmp_path, fake_r2):  # noqa: F811
    from scripts.corpus_e2e import load_entry

    contracts = _declare(tmp_path, fake_r2)
    entry = load_entry(tmp_path / "entry-b.json", task_id="corpus-code", job_id="code-v1")
    assert entry.contract == contracts["b"]
    assert json.loads((tmp_path / "entry-b.json").read_text())["params"] == entry.params
