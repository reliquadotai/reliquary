"""A corpus task's validator: refuses what it cannot run, serves what it can."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC
from reliquary.validator.corpus_validator import build_corpus_app, startup_refusal
from tests.unit.test_corpus_service import (  # noqa: F401
    _Tokenizer, _r2_client, fake_r2, seeded_job,
)


def _entry(**kw):
    return SimpleNamespace(task_id="corpus-math", job_id="swe-v1", mechanism="corpus-generation", **kw)


def _profile(toploc=True, model_id="org/Frozen", model_revision="abc123"):
    proof = SimpleNamespace(scheme=PROOF_SCHEME_TOPLOC, mode="enforce", chunk_tokens=32, topk=128)
    return SimpleNamespace(
        model_id=model_id, model_revision=model_revision, proofs=(proof,) if toploc else ()
    )


def test_a_contract_without_toploc_refuses(seeded_job):
    assert "toploc" in startup_refusal(_entry(), seeded_job.job, _profile(toploc=False), seeded_job.job.checkpoint_sha256)


def test_a_shadow_toploc_refuses(seeded_job):
    profile = _profile()
    profile.proofs[0].mode = "shadow"
    assert "enforce" in startup_refusal(_entry(), seeded_job.job, profile, seeded_job.job.checkpoint_sha256)


def test_a_contract_on_another_model_refuses(seeded_job):
    assert "model" in startup_refusal(
        _entry(), seeded_job.job, _profile(model_id="org/Other"), seeded_job.job.checkpoint_sha256
    )


def test_a_contract_on_another_revision_refuses(seeded_job):
    # `ProtocolProfile` pins both the repo and the revision it was measured
    # on; a job declared at another revision is not the checkpoint the
    # contract's proof thresholds were set against.
    assert "model" in startup_refusal(
        _entry(), seeded_job.job, _profile(model_revision="deadbeef"), seeded_job.job.checkpoint_sha256
    )


def test_a_checkpoint_that_does_not_match_the_job_refuses(seeded_job):
    assert "fingerprint" in startup_refusal(_entry(), seeded_job.job, _profile(), "0" * 64)


def test_a_matching_setup_starts(seeded_job):
    assert startup_refusal(_entry(), seeded_job.job, _profile(), seeded_job.job.checkpoint_sha256) is None


def test_the_app_serves_the_corpus_routes_and_nothing_of_rl(seeded_job):
    enqueued = []
    auditor = SimpleNamespace(enqueue=enqueued.append)
    app = build_corpus_app(entry=_entry(), job=seeded_job.job, store=seeded_job.store, records=None,
                           tokenizer=_Tokenizer(), renderer=seeded_job.renderer,
                           verify_signature=lambda r: True, auditor=auditor, proof_chunk_tokens=None,
                           prompt_job_for=seeded_job.prompt_job_for)
    client = TestClient(app)
    assert client.get("/corpus/job").status_code == 200
    assert client.get("/state").status_code == 404
