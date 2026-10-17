"""Signed-sandbox SWE jobs through the row map: an eval job's row ``i`` plays the
SWE-bench Verified instance its set's ``i``-th reference names, at the set's split,
on every side (session issuance, the validator's intake, the miner, the export). The
engagement keeps ``prompt_index`` (row space); only the ``(split, index)`` the
gateway and the session token name are mapped. A training job's rows stay SWE-smith
``train`` rows, by index.

The bridged env is faked: ``(split, index)`` has prompt ``"{split} prompt {index}"``
and instance ``"{split}__{index}"``. Tests that need ``reliquary_sandbox`` skip
without it."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.corpus.job import parse_job
from reliquary.corpus.signed_reasons import corpus_engagement
from reliquary.environment import agentic_swe, bridged_swe
from reliquary.environment.agentic_swe import SignedSweSource, job_signed_source
from reliquary.eval import prompt_source as ps
from reliquary.eval import rows as eval_rows
from reliquary.eval.prompt_source import eval_source_for
from reliquary.eval.rows import TaskRef, job_row_map
from tests.unit.test_corpus_job_episode import _manifest, _signed

SPLIT = "eval"
INDICES = (17, 3, 9)


def _prompt_of(split, index):
    return f"{split} prompt {index}"


def _instance_of(split, index):
    return f"{split}__{index}"


def _row_of(split, index):
    return None, SimpleNamespace(instance_id=_instance_of(split, index))


@pytest.fixture
def bridge(monkeypatch):
    """The fake bridged env, as `bridged_swe`'s readers; records what is asked."""
    asked = []

    def prompt_of(split, index):
        asked.append(("prompt", split, index))
        return _prompt_of(split, index)

    def instance_id_of(split, index):
        asked.append(("instance", split, index))
        return _instance_of(split, index)

    def row_of(split, index):
        asked.append(("row", split, index))
        return _row_of(split, index)

    monkeypatch.setattr(bridged_swe, "prompt_of", prompt_of)
    monkeypatch.setattr(bridged_swe, "instance_id_of", instance_id_of)
    monkeypatch.setattr(bridged_swe, "row_of", row_of)
    monkeypatch.setattr(ps, "_loaded", {})
    eval_rows._row_map.cache_clear()
    yield asked
    eval_rows._row_map.cache_clear()


def _set_rows(instances=None):
    return [{"problem_id": f"p{i}", "set_id": "swe-verified", "env": "reliquary_agentic_swe_v1",
             "interaction": "swe", "split": SPLIT, "source_index": index,
             "instance_id": (instances or {}).get(index, _instance_of(SPLIT, index))}
            for i, index in enumerate(INDICES)]


def _eval_job(monkeypatch, instances=None):
    body = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n"
                    for row in _set_rows(instances))
    monkeypatch.setattr(ps, "FETCHERS", [lambda source: body])
    source = eval_source_for("swe-verified", body, len(INDICES)).name
    return parse_job(_manifest(prompt_source=source, prompt_count=len(INDICES),
                               episode=_signed(env={"split": SPLIT, "num_images": 500})))


TRAIN_JOB = parse_job(_manifest(prompt_count=50, episode=_signed()))


# -- the bridge and the source --------------------------------------------------------

def test_signed_eval_source_maps_rows(bridge, monkeypatch):
    source = job_signed_source(_eval_job(monkeypatch))
    assert source.split == SPLIT
    assert source.prompt(0) == "eval prompt 17" and source.instance_id(2) == "eval__9"
    task = source.task_for(1)
    assert task.id == "eval__3" and task.prompt == "eval prompt 3"
    with pytest.raises(IndexError):
        source.prompt(3)                                      # outside the set's rows


def test_signed_row_map_refuses_stale_instance_ids(bridge, monkeypatch):
    job = _eval_job(monkeypatch, instances={3: "django__django-0"})
    with pytest.raises(ValueError, match="django__django-0"):
        job_signed_source(job)                                # checked when the job is built
    lazy = SignedSweSource(job_row_map(job), prompt_of=_prompt_of, row_of=_row_of)
    with pytest.raises(ValueError, match="django__django-0"):
        lazy.prompt(0)                                        # and on a source's first use


def test_non_eval_signed_job_unchanged(bridge):
    assert job_row_map(TRAIN_JOB).ref(5) == TaskRef("train", 5, None)
    source = job_signed_source(TRAIN_JOB)
    assert source.split == "train"
    assert source.prompt(5) == "train prompt 5" and source.instance_id(5) == "train__5"
    assert source.prompt(10_000) == "train prompt 10000"     # any train row, as before
    assert [a for a in bridge if a[0] == "instance"] == []    # no instance check for train
    old = SignedSweSource("train", prompt_of=_prompt_of, row_of=_row_of)
    assert old.prompt(7) == source.prompt(7) and old.instance_id(7) == source.instance_id(7)


def test_the_twenty_image_rule_is_trains_only(monkeypatch):
    from reliquary import sandbox as sandbox_support

    monkeypatch.setattr(sandbox_support, "require_sandbox", lambda: None)
    monkeypatch.setattr(sandbox_support, "sandbox_commit_refusal", lambda commit: "stop here")
    train = SimpleNamespace(sandbox=SimpleNamespace(sandbox_commit="x"),
                            env=SimpleNamespace(split="train", num_images=10))
    assert "20 images" in agentic_swe.sandbox_support_refusal(train)
    evaluated = SimpleNamespace(sandbox=SimpleNamespace(sandbox_commit="x"),
                                env=SimpleNamespace(split=SPLIT, num_images=500))
    assert agentic_swe.sandbox_support_refusal(evaluated) == "stop here"


def test_corpus_service_builds_the_eval_source(bridge, monkeypatch):
    from reliquary.validator import corpus_service

    job = _eval_job(monkeypatch)
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda episode, **kw: None)
    spec = corpus_service._resolve_agentic(job.prompt_source, None, job.episode)
    source = spec.create()
    assert isinstance(source, SignedSweSource) and source.prompt(0) == "eval prompt 17"
    stale = _eval_job(monkeypatch, instances={9: "gone"})
    eval_rows._row_map.cache_clear()
    with pytest.raises(corpus_service.CorpusPromptSourceError, match="gone"):
        corpus_service._resolve_agentic(stale.prompt_source, None, stale.episode).create()


# -- session issuance ------------------------------------------------------------------

class _Resolver:
    def __init__(self, split, asked):
        self.split, self.asked = split, asked

    async def resolve(self, index):
        from reliquary.sandbox.tasks import ResolvedTask

        self.asked.append((self.split, index))
        return ResolvedTask("img@sha256:" + "a" * 64, {}, None, None, None, None)


def _session_terms(tmp_path, job, prompt_index):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.validator.sandbox_wiring import wire_signed_job
    from tests.unit.test_signed_grading_and_wiring import _services

    _, services = _services(tmp_path)
    asked = []

    class Router:
        async def slots_remaining(self, index):
            return 2

    w = SimpleNamespace(job=job, is_banned=None)
    wire_signed_job(w, services=services, routes=lambda: SimpleNamespace(
                        routers={job.job_id: Router()}, retired=set()),
                    checkpoint_dir="/ck", tokenizer=None, vocab_size=None, chunk_tokens=32,
                    intake_factory=lambda job, **kw: SimpleNamespace(source=None),
                    resolver_factory=lambda split: _Resolver(split, asked))
    engagement = {"kind": "corpus", "job_id": job.job_id, "prompt_index": prompt_index}
    terms = asyncio.run(services.issuer._engagements["corpus"].terms("5Hot", engagement))
    return terms, asked


def test_signed_eval_session_names_mapped_task(bridge, monkeypatch, tmp_path):
    job = _eval_job(monkeypatch)
    terms, asked = _session_terms(tmp_path, job, 0)
    assert (terms.split, terms.index) == (SPLIT, 17)
    assert terms.prompt_index == 0 and terms.engagement == corpus_engagement(job.job_id, 0)
    assert asked == [(SPLIT, 17)]                             # the image of the Verified task


def test_non_eval_signed_session_unchanged(bridge, tmp_path):
    terms, asked = _session_terms(tmp_path, TRAIN_JOB, 5)
    assert (terms.split, terms.index, terms.prompt_index) == ("train", 5, 5)
    assert asked == [("train", 5)]


# -- the validator's intake ------------------------------------------------------------

def test_signed_intake_mapped_task(bridge, monkeypatch):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.validator import signed_intake

    job = _eval_job(monkeypatch)
    seen = []

    def verify(transcript, directory, verifier, expected, **kw):
        seen.append(expected)
        return SimpleNamespace(ok=False, reasons=[])

    monkeypatch.setattr(signed_intake, "verify_transcript", verify)
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda episode, **kw: None)
    monkeypatch.setattr(agentic_swe, "sandbox_support_refusal", lambda episode, **kw: None)
    monkeypatch.setattr(agentic_swe, "load_turn_renderer", lambda *a, **kw: None)
    intake = signed_intake.build_signed_episode_intake(
        job, checkpoint_dir="/ck", tokenizer=None, vocab_size=None, chunk_tokens=32,
        directory=lambda now: object(), token_verifier=None, sessions=None)
    assert intake.source.split == SPLIT and intake.source.prompt(1) == "eval prompt 3"
    request = SimpleNamespace(trajectory=SimpleNamespace(transcript={}), completions=[],
                              prompt_index=1, miner_hotkey="5Hot")
    intake.check(request, received=0.0)
    (expected,) = seen
    assert (expected.split, expected.index) == (SPLIT, 3)
    assert expected.engagement == corpus_engagement(job.job_id, 1)


# -- the miner -------------------------------------------------------------------------

def test_miner_open_request_mapped_task(bridge, monkeypatch):
    attest = pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.miner import signed_episode
    from tests.unit.test_signed_miner import HOT, FakeSessions

    job = _eval_job(monkeypatch)
    miner = signed_episode.SignedSweEpisodeRunner(
        job=job, hotkey=HOT, sign_binding=lambda binding: "sig", sessions=FakeSessions(),
        model_name="m", renderer_model_dir="/ck", generate_url="http://127.0.0.1:1",
        sampling=job.sampling)
    body = miner._open_body(2, "a" * 32)
    assert body["engagement"] == {"kind": "corpus", "job_id": job.job_id, "prompt_index": 2}
    assert miner._source.prompt(2) == "eval prompt 9"
    seen = []

    def verify(transcript, keys, claims_of, expected, **kw):
        seen.append(expected)
        return SimpleNamespace(reasons=[attest.Reason.UNKNOWN_KEY],
                               claims=SimpleNamespace(expires_at=-10**9))

    monkeypatch.setattr(attest, "verify_transcript", verify)
    refusal = signed_episode.transcript_refusal({}, job=job, hotkey=HOT, index=2, now=0.0)
    assert refusal[0] == "sandbox_session_expired"
    (expected,) = seen
    assert (expected.split, expected.index) == (SPLIT, 9)
    assert expected.engagement == corpus_engagement(job.job_id, 2)
