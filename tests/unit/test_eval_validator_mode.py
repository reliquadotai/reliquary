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
    argv = ["jobs", "create", ACK, "--fleet-knows-period-settlement"]
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

    def grade(out="out", **kw):
        return asyncio.run(grade_served_job(
            "eval-teutonic-aime26", out=tmp_path / out, records=records, subnet=subnet,
            entries=entries, open_taskset=lambda name, args: handle, clock=lambda: 1.0, **kw))

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
    assert not any(key.startswith("rl_") for key in provenance)       # no RL identity: nothing added
    grade(out="again")
    assert (tmp_path / "again" / "report.json").read_bytes() == (tmp_path / "out" / "report.json").read_bytes()
    # The RL identity, when given, is the only change.
    grade(out="rl", rl={"run_id": "5F3sa2TJ-r7-trainer", "step": 40})
    tagged = json.loads((tmp_path / "rl" / "report.json").read_text())
    added = {"rl_run_id": "5F3sa2TJ-r7-trainer", "rl_step": 40}
    assert tagged["provenance"] == {**provenance, **added}
    assert {k: v for k, v in tagged.items() if k != "provenance"} == {
        k: v for k, v in report.items() if k != "provenance"}
    with pytest.raises(JobNotGradable, match="rl run id"):
        grade(out="bad", rl={"run_id": "no spaces"})
    # The two gradings make a run monitor feed: the untagged one as the baseline.
    from reliquary.eval.run_feed import build_feed

    feed = build_feed("5F3sa2TJ-r7-trainer", [("rl", tagged)], [("base", report)], observed_at_ms=10_000)
    (series,) = feed["benchmarks"]
    assert series["id"] == "verifiers-fake-r0-n5" and series["protocol"]["attempts"] == 4
    assert series["checkpoints"][0]["step"] == 40 and series["checkpoints"][0]["samples"] == 3
    assert series["checkpoints"][0]["score"] == pytest.approx(env["pass@1"]["value"])
    assert series["baseline"]["step"] is None and series["checkpoints"][0]["at"] == 1000


def test_the_rl_identity_is_read_from_a_backup_name_or_given():
    from reliquary.eval.job_grading import rl_provenance

    assert rl_provenance("org/Teutonic") == {}
    assert rl_provenance("ReliquaryForge/teutonic-rl-step-00000050") == {
        "rl_step": 50, "rl_checkpoint_version": 50}
    assert rl_provenance("ReliquaryForge/teutonic-rl-step-00000050", run_id="abc-r1-trainer",
                         step=51, checkpoint_version=49) == {
        "rl_run_id": "abc-r1-trainer", "rl_step": 51, "rl_checkpoint_version": 49}
    assert rl_provenance("org/base", run_id="abc-r1-trainer", step=0) == {
        "rl_run_id": "abc-r1-trainer", "rl_step": 0}
    for bad in ({"step": -1}, {"step": True}, {"checkpoint_version": "3"}, {"run_id": ""},
                {"run_id": "a" * 129}, {"run_id": "a/b"}):
        with pytest.raises(ValueError):
            rl_provenance("org/base", **bad)


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

    async def grade_rl(job_id, *, out, allow_incomplete, rl):
        seen.update(rl=rl)
        return {"complete": True}

    monkeypatch.setattr(job_grading, "grade_served_job", grade_rl)
    tagged = CliRunner().invoke(app, ["eval", "grade", "--job", "eval-x", "--out", str(tmp_path / "o"),
                                      "--rl-run-id", "abc-r1-trainer", "--rl-step", "40"])
    assert tagged.exit_code == 0, tagged.output
    assert seen["rl"] == {"run_id": "abc-r1-trainer", "step": 40}
    order = CliRunner().invoke(app, ["eval", "grade", "--job", "order-eval-1", "--rl-step", "4"])
    assert order.exit_code == 1 and "order job" in order.output


# --------------------------------------------------------------------------
# the validator boots on an eval task
# --------------------------------------------------------------------------


def test_a_validator_boots_on_an_eval_task_with_the_external_environment(
        bucket, registry, published):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.validator.task_config import resolve_task_config

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _eval_args()).exit_code == 0
    entry = registry["entries"]["eval-teutonic-aime26"]
    config = resolve_task_config(dict(registry["entries"]), "eval-teutonic-aime26",
                                 profile_id=entry.profile_id, generation_contract=entry.contract)
    assert config.task_id == "eval-teutonic-aime26"


def test_an_rl_task_naming_the_external_environment_never_boots():
    from tests.unit.test_contract_startup_refusals import _contract, _entry, _profile

    from reliquary.protocol.external_eval import EXTERNAL_EVAL_ENVIRONMENT
    from reliquary.validator.task_config import TaskConfigError, resolve_task_config

    profile = _profile()
    contract = _contract(profile)
    contract["environments"] = {**contract["environments"],
                                EXTERNAL_EVAL_ENVIRONMENT: {"max_new_tokens": 128,
                                                            "answer_format": "text", "bft": None}}
    with pytest.raises(TaskConfigError, match=EXTERNAL_EVAL_ENVIRONMENT):
        resolve_task_config({"glm-run": _entry(profile, contract=contract)}, "glm-run",
                            profile_id=profile.profile_id, generation_contract=contract)


# --------------------------------------------------------------------------
# jobs create --eval-set over episode and SWE sets
# --------------------------------------------------------------------------

TELECOM = "reliquary_telecom_solo_v2"
CHATML_TOOLS = "reliquary-chatml-tools-v1"
SWE_SET_ID = "swe-bench-r0-n4"


def _write_reference_set(directory, set_id, rows, card_fields):
    import hashlib
    import json

    directory.mkdir()
    body = b"".join(json.dumps({**row, "set_id": set_id}, sort_keys=True).encode() + b"\n"
                    for row in rows)
    (directory / "prompts.jsonl").write_bytes(body)
    card = {"set_id": set_id, "count": len(rows), "prompts_sha256": hashlib.sha256(body).hexdigest(),
            **card_fields}
    (directory / "set.json").write_text(json.dumps(card))


@pytest.fixture
def reference_sets(tmp_path, monkeypatch):
    """A telecom episode set and a SWE set in the sets directory."""
    from reliquary.environment import registry
    from reliquary.eval import prompt_source as ps

    real = registry.ENVIRONMENT_SPECS[TELECOM]
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [ps._from_directory])
    monkeypatch.setenv(ps.SETS_DIR_ENV, str(tmp_path))
    rows = [{"problem_id": f"p{i}", "env": TELECOM, "interaction": "episode",
             "split": "benchmark", "source_index": i, "instance_id": f"t-{i}"} for i in range(4)]
    _write_reference_set(tmp_path / "telecom-bench-r0-n4", "telecom-bench-r0-n4", rows, {
        "interaction": "episode", "source": TELECOM, "env": TELECOM, "split": "benchmark",
        "renderer_id": real.renderer_id})
    rows = [{"problem_id": f"p{i}", "env": "reliquary-swe", "interaction": "swe",
             "split": "benchmark", "source_index": i, "instance_id": f"repo__x-{i}"}
            for i in range(4)]
    _write_reference_set(tmp_path / SWE_SET_ID, SWE_SET_ID, rows, {
        "interaction": "swe", "source": "reliquary-swe", "env": "reliquary-swe",
        "split": "benchmark", "renderer_id": None})
    return real


def _suite_episode(real, **over):
    episode = {"env": {"package": "reliquary-suite", "name": TELECOM,
                       "version": real.environment_manifest_sha256, "split": "benchmark"},
               "harness": "suite", "renderer": real.renderer_id, "max_turns": 8,
               "max_tokens_per_turn": 1024, "max_total_tokens": 60000}
    episode.update(over)
    return episode


def _signed_swe_episode(**over):
    from tests.unit.test_corpus_job_signed_sandbox import signed_episode

    episode = signed_episode()
    episode["env"] = {**episode["env"], "split": "benchmark"}
    episode["max_tokens_per_turn"] = 1024
    episode.update(over)
    return episode


def _episode_args(tmp_path, episode, **overrides):
    import json

    path = tmp_path / "episode.json"
    path.write_text(json.dumps(episode))
    options = {"--eval-set": "telecom-bench-r0-n4", "--episode-file": str(path),
               "--renderer-id": CHATML_TOOLS, "--eos-token-id": "151645"}
    options.update(overrides)
    return _eval_args(**options)


@pytest.fixture
def telecom_registry(reference_sets, monkeypatch):
    """The telecom spec as this binary's registry has it, minus the wheel that is not installed here."""
    from types import SimpleNamespace

    from reliquary.corpus import suite_episode

    class _Env:
        name, split, max_turns = TELECOM, "benchmark", 8

        def __len__(self):
            return 100

        def get_task(self, index):
            return SimpleNamespace(id=f"t-{index}")

    real = reference_sets
    spec = SimpleNamespace(
        name=TELECOM, interaction_mode="episode", renderer_id=real.renderer_id,
        environment_manifest_sha256=real.environment_manifest_sha256,
        create=lambda split="train": _Env())
    monkeypatch.setattr(suite_episode, "_registry", lambda given: given or {TELECOM: spec})


def test_jobs_create_suite_eval_job(bucket, registry, reference_sets, telecom_registry, tmp_path):
    import asyncio

    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.eval import prompt_source as ps
    from reliquary.infrastructure import corpus_job_store as job_store

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _episode_args(tmp_path, _suite_episode(reference_sets)))
    assert result.exit_code == 0, result.output
    job, _ = asyncio.run(job_store.read_job("eval-teutonic-aime26"))
    source = ps.parse_eval_source(job.prompt_source)
    assert (source.set_id, source.count) == ("telecom-bench-r0-n4", 4)
    assert job.prompt_count == 4 and job.episode.env.split == "benchmark"
    assert job.episode.is_suite and job.seed is not None
    entry = registry["entries"]["eval-teutonic-aime26"]
    assert TELECOM in entry.contract["environments"]
    assert entry.params["audit_q"] == 1.0


def test_jobs_create_reference_set_needs_episode_file(bucket, registry, reference_sets):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _eval_args(**{"--eval-set": "telecom-bench-r0-n4",
                                                   "--renderer-id": CHATML_TOOLS}))
    assert result.exit_code != 0 and "--episode-file" in result.output, result.output
    assert "eval-teutonic-aime26" not in registry["entries"]


def test_jobs_create_refuses_split_mismatch(bucket, registry, reference_sets, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    episode = _suite_episode(reference_sets)
    episode["env"]["split"] = "train"
    result = CliRunner().invoke(app, _episode_args(tmp_path, episode))
    assert result.exit_code != 0, result.output
    assert "set's split is 'benchmark', the episode names 'train'" in result.output
    assert "eval-teutonic-aime26" not in registry["entries"]


def test_jobs_create_refuses_renderer_mismatch(bucket, registry, reference_sets, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _episode_args(
        tmp_path, _suite_episode(reference_sets, renderer="other-renderer-v1")))
    assert result.exit_code != 0 and "renderer disagrees" in result.output, result.output
    result = CliRunner().invoke(app, _episode_args(
        tmp_path, _suite_episode(reference_sets), **{"--renderer-id": "chat-template-thinking-v1"}))
    assert result.exit_code != 0 and "renderer disagrees" in result.output, result.output
    assert "eval-teutonic-aime26" not in registry["entries"]


def test_jobs_create_refuses_an_episode_for_another_environment(
        bucket, registry, reference_sets, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    episode = _suite_episode(reference_sets)
    episode["env"]["name"] = "reliquary_other_v1"
    result = CliRunner().invoke(app, _episode_args(tmp_path, episode))
    assert result.exit_code != 0 and "the set's environment is" in result.output, result.output


@pytest.mark.parametrize("harness", [None, "mini-swe@2"])
def test_jobs_create_signed_swe_eval_job(bucket, registry, reference_sets, tmp_path, monkeypatch, harness):
    from reliquary.environment import agentic_swe

    # The renderers wheel and the SWE dataset are not installed where the tests run.
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda *a, **k: None)
    monkeypatch.setattr(agentic_swe, "load_swe_source", lambda *a, **k: [None] * 100)
    import asyncio
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.environment import bridged_swe
    from reliquary.eval import rows as eval_rows

    # The set's rows are checked against the bridged env (not installed here).
    monkeypatch.setattr(bridged_swe, "row_of", lambda split, index: (
        None, SimpleNamespace(instance_id=f"repo__x-{index}")))
    eval_rows._row_map.cache_clear()
    from reliquary.eval import prompt_source as ps
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    result = CliRunner().invoke(app, _episode_args(
        tmp_path, _signed_swe_episode(), **{"--eval-set": SWE_SET_ID, "--prompt-count": "3",
                                            "--renderer-id": _signed_swe_episode()["renderer"],
                                            **({"--harness": harness} if harness else {})}))
    assert result.exit_code == 0, result.output
    job, _ = asyncio.run(job_store.read_job("eval-teutonic-aime26"))
    assert is_signed_sandbox(job) and job.prompt_count == 3
    # The eval plays the harness it was created with (none: the compatibility harness).
    assert agentic_swe.job_harness(job).key == (harness or "bash-vf-b2e4e81@1")
    assert ps.parse_eval_source(job.prompt_source).count == 3
    entry = registry["entries"]["eval-teutonic-aime26"]
    assert list(entry.contract["environments"]) == [AGENTIC_SWE_ENVIRONMENT]


def test_jobs_create_swe_set_needs_a_signed_sandbox_episode(
        bucket, registry, reference_sets, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    episode = _signed_swe_episode()
    del episode["execution"], episode["sandbox"]
    result = CliRunner().invoke(app, _episode_args(tmp_path, episode, **{"--eval-set": SWE_SET_ID}))
    assert result.exit_code != 0 and "signed sandbox" in result.output, result.output
