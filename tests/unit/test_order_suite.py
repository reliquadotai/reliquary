"""Suite episode orders: a forced-draw eval order on a suite set (a replayed tool world, telecom here) carries the
suite episode that plays its references, rendered by the ChatML tools renderer on a Qwen3.5-family model; the
order control wires it with no sandbox: its replaying intake and replay grader."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from reliquary.eval import qualification as qual
from reliquary.eval.storage import SubnetEvalStore
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_agentic import QWEN35, qwen35  # noqa: F401
from tests.unit.test_order_pinned import SAMPLING, pins  # noqa: F401

ENV = "reliquary_telecom_solo_v2"
SET = "telecom-bench-r0-n4"
RENDERER = "reliquary-chatml-tools-v1"


def _telecom_or_skip():
    from reliquary.environment.agentic.external import verify_external_artifact
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    try:
        verify_external_artifact(ENVIRONMENT_SPECS[ENV])
    except (ValueError, TypeError) as exc:
        pytest.skip(f"reliquary-telecom-solo 0.2.0a1 not usable: {exc}")


def _suite_episode(**over):
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    episode = {"env": {"package": "reliquary-suite", "name": ENV,
                       "version": ENVIRONMENT_SPECS[ENV].environment_manifest_sha256, "split": "benchmark"},
               "harness": "suite", "renderer": RENDERER, "max_turns": 40, "max_tokens_per_turn": 1024,
               "max_total_tokens": 32768}
    episode.update(over)
    return episode


@pytest.fixture
def suite_set(admin):  # noqa: F811
    """A telecom benchmark set of 4 references, built and published."""
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval import rows as eval_rows
    from reliquary.eval.sets import build_source_set
    from reliquary.eval.storage import publish_set

    _telecom_or_skip()
    eval_rows._row_map.cache_clear()
    card = build_source_set(ENV, split="benchmark", count=4, set_id=SET, out=admin.root / "suite",
                            clock=lambda: 1.0)
    asyncio.run(publish_set(admin.root / "suite", platform=LocalDirectorySink(admin.root / "p"),
                            subnet=SubnetEvalStore()))
    yield card
    eval_rows._row_map.cache_clear()


def _order(**kw):
    from tests.unit.test_admin_eval_jobs import MODEL, REVISION

    base = {"job_id": "order-eval-suite1", "model": MODEL, "revision": REVISION, "env": ENV,
            "prompt_count": 3, "samples_per_prompt": 2, "max_new_tokens": 1024, "sampling": SAMPLING,
            "eval_set_id": SET, "draw": "gumbel_max", "episode": _suite_episode()}
    return {**base, **kw}


def _served(job_id="order-eval-suite1"):
    job, _ = asyncio.run(job_store.read_job(job_id))
    return job, asyncio.run(qual.EvalJobStore().read(job_id))


# -- the renderer rule --------------------------------------------------------------------------


def test_a_suite_episode_renders_through_the_chatml_tools_renderer_on_qwen35_only():
    from reliquary.eval.episode_jobs import episode_renderer_refusal

    assert episode_renderer_refusal(QWEN35, RENDERER, suite=True) is None
    assert "mismatch" in episode_renderer_refusal(QWEN35, "renderers:qwen38@0.1", suite=True)
    # A signed episode still needs its own family renderer, never the suite one.
    assert "mismatch" in episode_renderer_refusal(QWEN35, RENDERER)
    assert "unavailable" in episode_renderer_refusal("Qwen3ForCausalLM", RENDERER, suite=True)


# -- the admin --------------------------------------------------------------------------------


def test_an_eval_order_on_a_suite_set_declares_its_drawn_suite_job(admin, suite_set, qwen35):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 201, response.text
    job, declared = _served()
    assert job.episode.is_suite and job.renderer_id == RENDERER and job.sampling.draw == "gumbel_max"
    assert job.episode.env.split == "benchmark" and job.prompt_count == 3 and job.seed is not None
    assert declared["episode"] == job.episode.to_contract() and declared["thinking"] is None
    entry = admin.registry["entries"]["order-eval-suite1"]
    assert list(entry.contract["environments"]) == [ENV]


@pytest.mark.parametrize("change,word", [
    ({"episode": None}, "pass the episode"),
    ({"thinking": True}, "thinking"),
    ({"draw": None}, "forced-draw"),
])
def test_a_suite_order_unlike_its_set_is_refused(admin, suite_set, qwen35, change, word):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order(**change))
    assert response.status_code == 422 and word in response.text, response.text


def test_a_suite_order_on_another_split_or_renderer_is_refused(admin, suite_set, qwen35):  # noqa: F811
    episode = _suite_episode()
    other_split = {**episode, "env": {**episode["env"], "split": "eval"}}
    response = admin("POST", "/admin/v1/jobs", _order(episode=other_split))
    assert response.status_code == 422 and "split" in response.text, response.text
    response = admin("POST", "/admin/v1/jobs", _order(episode={**episode, "renderer": "renderers:qwen38@0.1"}))
    assert response.status_code == 422 and "renderer" in response.text, response.text


def test_a_suite_order_on_a_model_without_the_family_is_refused(admin, suite_set, pins):  # noqa: F811
    response = admin("POST", "/admin/v1/jobs", _order())
    assert response.status_code == 422 and "episode_renderer_unavailable" in response.text, response.text


# -- the order control --------------------------------------------------------------------------


def test_the_order_control_accepts_a_suite_episode(admin, suite_set, qwen35):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal, qualification_refusal

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    job, declared = _served()
    entry = admin.registry["entries"]["order-eval-suite1"]
    assert order_job_refusal(entry, job) is None
    assert qualification_refusal(entry, job, declared, None) is None
    other = replace(entry, contract={**entry.contract, "model_architecture": "Qwen3ForCausalLM"})
    assert "episode_renderer_unavailable" in order_job_refusal(other, job)


def test_a_suite_order_is_wired_without_a_sandbox(admin, suite_set, qwen35, tmp_path, monkeypatch):  # noqa: F811
    from reliquary.validator.suite_grading import LocalEpisodeGrader
    from reliquary.validator.suite_intake import InlineSuiteReplayer, SuiteEpisodeIntake

    assert admin("POST", "/admin/v1/jobs", _order()).status_code == 201
    built = []

    def replayer_for(tokenizer):
        built.append(tokenizer)
        return InlineSuiteReplayer(encode=lambda t: [], decode=lambda ids: "")

    app, job_set = _control_with(admin, tmp_path, monkeypatch, replayer_for)
    job_id = "order-eval-suite1"

    async def go():
        await job_set.refresh()
        reasons = app.state.dispatcher.attention.get(job_id)
        assert job_id in job_set.served, reasons
        w = app.state.eval_served[job_id]
        assert isinstance(w.episode_intake, SuiteEpisodeIntake) and isinstance(w.grader, LocalEpisodeGrader)
        assert w.grader._suite_source is w.episode_intake.source
        # No gateway hold: held only for its audit.
        assert job_set._routes.held_reason(job_id) == "awaiting_audit"
        status = await app.state.control_status()
        assert "sandbox" not in status["jobs"].get(job_id, {})
        for tasks in job_set._tasks.values():
            for task in tasks:
                task.cancel()

    asyncio.run(go())
    assert len(built) == 1


def _control_with(admin, tmp_path, monkeypatch, replayer_for):  # noqa: F811
    """``test_order_agentic._control`` with no sandbox and the suite replayer given."""
    from reliquary.infrastructure import corpus_executor_store as executors
    from reliquary.infrastructure import corpus_record_store as record_store
    from reliquary.infrastructure import drand
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_auditor, corpus_period_settlement
    from reliquary.validator import suite_grading
    from reliquary.validator.eval_control import EvalExecutorDirectory, PairedAuditDispatcher, build_eval_control
    from tests.unit.test_suite_episode_miner import ByteTokenizer

    async def idle(self):
        await asyncio.sleep(3600)

    for module in (record_store, executors):
        monkeypatch.setattr(module, "get_s3_client", lambda **kw: admin.bucket)
    monkeypatch.setattr(drand, "get_current_chain", lambda: {})
    monkeypatch.setattr(BucketRecordStore, "read_submission_meta",
                        lambda self, job_id, sid: self.read_submission(job_id, sid))
    monkeypatch.setattr(corpus_auditor.CorpusAuditor, "run", idle)
    monkeypatch.setattr(suite_grading.LocalEpisodeGrader, "run", idle)
    monkeypatch.setattr(corpus_period_settlement.CorpusPeriodSettler, "settle_once", idle)

    async def read_entries():
        return dict(admin.registry["entries"])

    directory = EvalExecutorDirectory()
    return build_eval_control(
        store=BucketJobStore(), records=BucketRecordStore(), dispatcher=PairedAuditDispatcher(directory=directory),
        directory=directory, verify_signature=lambda request: True,
        tokenizer_for=lambda repo, revision: (ByteTokenizer(), 300), read_entries=read_entries,
        sandbox=None, sandbox_down="no sandbox here", model_dir_for=lambda repo, revision: str(tmp_path),
        suite_replayer_for=replayer_for)
