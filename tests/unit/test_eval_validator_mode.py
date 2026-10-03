"""Evaluations served by our own corpus validator: the model on its GPU, TOPLOC
local, the environment graded on CPU. No order control, no qualification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.eval.prompt_source import declared_environment
from tests.unit.test_corpus_hot_jobs import ENV, OTHER_MODEL, REFUSED, _hot_entry, _hot_job, _refusal
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401
from tests.unit.test_corpus_validator import seeded_job  # noqa: F401

EVAL = "eval-set:aime26-set:30:" + "a" * 64


def test_a_catalog_job_declares_its_prompt_source():
    assert declared_environment({"environments": {"src": ENV}}, "src") == "src"


def test_an_eval_job_declares_the_one_environment_of_its_contract():
    contract = {"environments": {"reliquary_external_eval_v1": ENV}}
    assert declared_environment(contract, EVAL) == "reliquary_external_eval_v1"
    assert declared_environment({"environments": {"a": ENV, "b": ENV}}, EVAL) is None
    assert declared_environment({"environments": {}}, EVAL) is None


def _eval_entry(env_name="reliquary_external_eval_v1", **kw):
    entry = _hot_entry(**kw)
    entry.contract["environments"] = {env_name: ENV}
    return entry


def test_our_validator_serves_an_eval_job_on_its_model():
    process = {"environments": {"reliquary_external_eval_v1": ENV}, "protocol_version": 5}
    assert _refusal(_eval_entry(), _hot_job(prompt_source=EVAL), process_contract=process) is None


def test_an_eval_job_the_process_was_not_started_for_is_refused():
    process = {"environments": {"src": ENV}, "protocol_version": 5}
    kind, why = _refusal(_eval_entry(), _hot_job(prompt_source=EVAL), process_contract=process)
    assert kind == REFUSED and "reliquary_external_eval_v1" in why


def test_an_eval_job_on_another_model_waits_for_its_own_validator():
    kind, _ = _refusal(_eval_entry(model="org/Other"), _hot_job(prompt_source=EVAL))
    assert kind == OTHER_MODEL


def test_several_eval_jobs_share_one_process():
    from reliquary.validator.corpus_validator import multi_job_refusal

    process = {"environments": {"reliquary_external_eval_v1": ENV}}
    pairs = [(_eval_entry(task_id=f"t{i}", job_id=f"j{i}"),
              _hot_job(job_id=f"j{i}", prompt_source=EVAL)) for i in range(2)]
    assert multi_job_refusal(pairs, proof_of=lambda e: "p", process_contract=process) is None
    process = {"environments": {"other": ENV}}
    assert "differently" in multi_job_refusal(pairs, proof_of=lambda e: "p",
                                              process_contract=process)


def test_the_corpus_control_serves_an_eval_jobs_prompts(seeded_job, monkeypatch):
    import dataclasses
    import json

    from fastapi.testclient import TestClient

    from reliquary.eval import prompt_source as ps
    from reliquary.validator.corpus_validator import build_corpus_app
    from tests.unit.test_corpus_validator import _Tokenizer, _entry

    monkeypatch.setattr(ps, "_loaded", {})
    body = b"".join(json.dumps({"problem_id": f"s-{i}", "env": "x", "set_id": "s",
                                "messages": [{"role": "user", "content": f"q{i}"}]},
                               sort_keys=True).encode() + b"\n" for i in range(3))
    source = ps.eval_source_for("s", body, 2)
    ps.register_eval_prompts(source, body)

    def app_for(job):
        return TestClient(build_corpus_app(
            entry=_entry(), job=job, store=seeded_job.store, records=None,
            tokenizer=_Tokenizer(), renderer=seeded_job.renderer, verify_signature=lambda r: True,
            auditor=SimpleNamespace(enqueue=lambda r: None), proof_chunk_tokens=None,
            prompt_job_for=seeded_job.prompt_job_for))

    eval_job = dataclasses.replace(seeded_job.job, prompt_source=source.name)
    client = app_for(eval_job)
    served = client.get(f"/corpus/jobs/{eval_job.job_id}/eval-prompts")
    assert served.status_code == 200 and served.content == b"".join(body.splitlines(True)[:2])
    assert client.get("/corpus/jobs/nope/eval-prompts").status_code == 404
    catalog = app_for(seeded_job.job)
    refused = catalog.get(f"/corpus/jobs/{seeded_job.job.job_id}/eval-prompts")
    assert refused.status_code == 404 and refused.json()["detail"] == "not_an_eval_job"


# --------------------------------------------------------------------------
# jobs create --eval-set
# --------------------------------------------------------------------------

from tests.unit.test_jobs_cli import ACK, _rl_entry, bucket, registry  # noqa: E402,F401

REVISION = "c" * 40


def _eval_args(**overrides):
    options = {"--job-id": "eval-teutonic-aime26", "--model": "org/Teutonic",
               "--model-revision": REVISION, "--model-architecture": "Qwen3ForCausalLM",
               "--checkpoint-sha256": "a" * 64, "--eval-set": "verifiers-fake-r0-n5",
               "--renderer-id": "chat-template-thinking-v1", "--eos-token-id": "151645",
               "--max-new-tokens": "1024", "--slots-per-prompt": "4", "--cap": "0.02"}
    options.update(overrides)
    argv = ["jobs", "create", ACK]
    for flag, value in options.items():
        if value is not None:
            argv += [flag, value]
    return argv


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A Verifiers set in the sets directory, the way validators and the CLI read it."""
    from reliquary.eval import prompt_source as ps
    from reliquary.eval.sets import build_source_set
    from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle, opener

    card = build_source_set("verifiers:fake", out=tmp_path / "verifiers-fake-r0-n5",
                            open_taskset=opener(fake_handle([FakeTask(i) for i in range(5)])))
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [ps._from_directory])
    monkeypatch.setenv(ps.SETS_DIR_ENV, str(tmp_path))
    return card


def test_jobs_create_declares_an_eval_job_on_any_model(bucket, registry, published):
    import asyncio

    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.eval import prompt_source as ps
    from reliquary.infrastructure import corpus_job_store as job_store

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _eval_args(**{"--prompt-count": "3"}))
    assert result.exit_code == 0, result.output
    job, _ = asyncio.run(job_store.read_job("eval-teutonic-aime26"))
    source = ps.parse_eval_source(job.prompt_source)
    assert (source.set_id, source.count) == ("verifiers-fake-r0-n5", 3)
    assert job.seed is not None and job.renderer_id == "chat-template-thinking-v1"
    entry = registry["entries"]["eval-teutonic-aime26"]
    assert list(entry.contract["environments"]) == ["reliquary_external_eval_v1"]
    assert entry.params["audit_q"] == 1.0
    assert entry.contract["model_id"] == "org/Teutonic"


def test_the_whole_set_by_default(bucket, registry, published):
    import asyncio

    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.eval import prompt_source as ps
    from reliquary.infrastructure import corpus_job_store as job_store

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _eval_args()).exit_code == 0
    job, _ = asyncio.run(job_store.read_job("eval-teutonic-aime26"))
    assert ps.parse_eval_source(job.prompt_source).count == 5


@pytest.mark.parametrize("change,message", [
    ({"--prompt-source": "reliquary_dapo_math_v1"}, "exactly one of --prompt-source and --eval-set"),
    ({"--renderer-id": "reliquary-external-prompt-v1"}, "model's chat template"),
    ({"--audit-q": "0.5"}, "audits every submission"),
    ({"--prompt-start": "2"}, "starts at the set's first problem"),
    ({"--grader-id": "x", "--threshold": "1"}, "no filter"),
    ({"--prompt-count": "9"}, "holds 5 problems"),
    ({"--from-profile": "qwen3-4b-base-dapo-reliquary-v1"}, "composed"),
])
def test_an_eval_job_is_refused_what_would_make_it_wrong(bucket, registry, published, change,
                                                       message):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _eval_args(**change))
    assert result.exit_code != 0 and message in result.output, result.output
    assert "eval-teutonic-aime26" not in registry["entries"]


def test_an_unpublished_set_is_named(bucket, registry, published):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    result = CliRunner().invoke(app, _eval_args(**{"--eval-set": "nope"}))
    assert result.exit_code != 0 and "nope" in result.output


# --------------------------------------------------------------------------
# grading a job our validator served
# --------------------------------------------------------------------------


def test_a_served_eval_job_is_graded_on_cpu_from_its_audited_records(
        bucket, registry, published, tmp_path):
    import asyncio
    import json

    import pyarrow.parquet as pq
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.corpus.delivery import LocalDirectorySink
    from reliquary.eval.job_grading import JobNotGradable, grade_served_job
    from reliquary.eval.storage import publish_set
    from tests.unit.test_admin_eval_jobs import _JobRecords
    from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _eval_args(**{"--prompt-count": "3"})).exit_code == 0
    subnet = LocalDirectorySink(tmp_path / "subnet")
    asyncio.run(publish_set(tmp_path / "verifiers-fake-r0-n5", platform=None, subnet=subnet))
    records = _JobRecords()

    def submit(sid, prompt, texts):
        records.subs[sid] = {"prompt_index": prompt, "hotkey": f"hk{prompt}",
                             "completions": [{"text": t, "tokens": [1, 151645]} for t in texts]}
        records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": f"hk{prompt}"}
        records.settled.append(sid)

    handle = fake_handle([FakeTask(i) for i in range(5)])

    async def entries():
        return registry["entries"].values()

    def grade(**kw):
        return asyncio.run(grade_served_job(
            "eval-teutonic-aime26", out=tmp_path / "out", records=records, subnet=subnet,
            entries=entries, open_taskset=lambda name, args: handle, **kw))

    # Nothing audited yet: an empty job is not complete.
    with pytest.raises(JobNotGradable, match="misses samples"):
        grade()
    for prompt in range(3):
        for k in range(4):
            right = f"<think>hmm</think>{prompt}"
            submit(f"{prompt:032x}{k:032x}", prompt, [right if k < 2 + (prompt == 0) else "x"])
    records.subs["f" * 64] = {"prompt_index": 0, "hotkey": "late", "completions": []}
    with pytest.raises(JobNotGradable, match="not drained"):
        grade()
    del records.subs["f" * 64]
    manifest = grade()
    assert manifest["complete"] is True
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    env = report["envs"]["verifiers:fake"]
    # p0: 3/4 right, p1 and p2: 2/4 right; thinking on, "x" has no closing tag.
    assert env["pass@1"]["value"] == pytest.approx((0.75 + 0.5 + 0.5) / 3)
    provenance = report["provenance"]
    assert provenance["model"] == "org/Teutonic" and provenance["thinking"] is True
    assert provenance["verification"]["source"] == "task contract"
    assert provenance["verification"]["thresholds"]["exp_mismatch_threshold"] is not None
    assert provenance["generation"] == "sn81-miners" and provenance["miner_hotkeys"] == 3
    assert pq.read_table(tmp_path / "out" / "graded.parquet").num_rows == 12


def test_an_order_job_or_a_catalog_job_is_not_graded_here(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from reliquary.eval.job_grading import JobNotGradable, grade_served_job

    async def read_job(job_id):
        return SimpleNamespace(prompt_source="reliquary_dapo_math_v1"), None

    async def read_ledgers(job_id):
        return {}, None

    with pytest.raises(JobNotGradable, match="admin service"):
        asyncio.run(grade_served_job("order-eval-1", out=tmp_path, records=object(),
                                     subnet=object(), read_job=read_job,
                                     read_ledgers=read_ledgers))
    with pytest.raises(JobNotGradable, match="not an eval set"):
        asyncio.run(grade_served_job("corpus-math", out=tmp_path, records=object(),
                                     subnet=object(), read_job=read_job,
                                     read_ledgers=read_ledgers))


def test_eval_grade_grades_a_served_job_here(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.eval import job_grading

    seen = {}

    async def grade(job_id, *, out, allow_incomplete):
        seen.update(job_id=job_id, out=out, allow_incomplete=allow_incomplete)
        return {"complete": True}

    monkeypatch.setattr(job_grading, "grade_served_job", grade)
    monkeypatch.delenv("RELIQUARY_ADMIN_SECRET", raising=False)
    result = CliRunner().invoke(app, ["eval", "grade", "--job", "eval-teutonic-aime26",
                                      "--out", str(tmp_path / "o"), "--allow-incomplete"])
    assert result.exit_code == 0, result.output
    assert seen == {"job_id": "eval-teutonic-aime26", "out": str(tmp_path / "o"),
                    "allow_incomplete": True}
    missing = CliRunner().invoke(app, ["eval", "grade", "--job", "eval-x"])
    assert missing.exit_code == 1 and "--out is required" in missing.output
