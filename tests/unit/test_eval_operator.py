"""The operator's evaluations: qualify then create, grade home, compare two checkpoints."""

from __future__ import annotations

import asyncio
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reliquary.eval import operator as op
from reliquary.eval import qualification as qual
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_files import _complete_job
from tests.unit.test_admin_eval_jobs import (  # noqa: F401
    MODEL, REVISION, SAMPLING, SECRET, THRESHOLDS, admin,
)
from tests.unit.test_jobs_cli import registry  # noqa: F401


def client_for(admin):  # noqa: F811
    # The fixture's client keeps its event loop open, as a server does: a
    # grading runs beside the request that started it.
    return op.AdminClient("http://testserver", SECRET, http=admin.client)


def finish(qualification_id, status=qual.QUALIFIED):
    async def run():
        store = qual.QualificationStore()
        record, etag = await store.read(qualification_id)
        result = {"thresholds": THRESHOLDS, "architecture": "Qwen3ForCausalLM",
                  "checkpoint_sha256": "d" * 64, "eos_token_id": 151645,
                  "band": {"exp_mismatch": 50, "mant_mean": 27.6, "mant_median": 20.0,
                           "chunks": 90}, "clamped": [], "measurements": {}}
        if status != qual.QUALIFIED:
            result = {"refused_reason": "band_over_ceiling"}
        record.update(status=status, result=result)
        await store.write(record, etag)

    asyncio.run(run())


def card_of(admin):  # noqa: F811
    return json.loads((admin.root / "set" / "set.json").read_text())


def test_a_model_names_a_full_revision():
    assert op.split_model("org/Teutonic@" + "a" * 40) == ("org/Teutonic", "a" * 40)
    for bad in ("org/Teutonic", "org/Teutonic@main", "org/Teutonic@abc", "@" + "a" * 40):
        with pytest.raises(ValueError, match="repo@<40-hex commit>"):
            op.split_model(bad)


def test_ids_are_deterministic_and_bounded():
    conditions = {"model": "m", "revision": "a" * 40, "set_id": "s", "problems": 3}
    first = op.qualification_id_for("order-", conditions)
    assert first == op.qualification_id_for("order-", dict(reversed(conditions.items())))
    assert first.startswith("order-q-") and len(first) <= 63
    assert first != op.qualification_id_for("order-", {**conditions, "problems": 4})
    order = {"model": "org/m", "sampling": {"temperature": 0.6}, "max_new_tokens": 512,
             "thinking": True}
    short = op.default_job_id("order-", "a" * 40, "aime26", 30, 8, order)
    assert short.startswith("order-eval-aaaaaaaa-aime26-n30x8-") and len(short) == 39
    # Another sampling, budget or thinking mode is another job.
    for change in ({"sampling": {"temperature": 1.0}}, {"thinking": False},
                   {"max_new_tokens": 1024}, {"model": "org/other"}):
        assert op.default_job_id("order-", "a" * 40, "aime26", 30, 8,
                                 {**order, **change}) != short
    long = op.default_job_id("order-", "a" * 40, "verifiers-livecodebench-a1b2c3d4-r0-n1055",
                             1055, 1, order)
    assert len(long) <= 63 and long.startswith("order-eval-aaaaaaaa-verifiers")
    # A catalog set id holds underscores; a job id may not.
    catalog = op.default_job_id("order-", "a" * 40, "reliquary_dapo_math_v1-eval-r0-n100",
                                100, 4, order)
    assert op.checked_job_id(catalog) == catalog and "_" not in catalog
    with pytest.raises(ValueError, match=r"\[a-z0-9-\]"):
        op.checked_job_id("order-eval-a_b")


def test_create_qualifies_waits_then_creates_and_a_second_run_reuses_both(admin):  # noqa: F811
    client, logs, slept = client_for(admin), [], []
    card = card_of(admin)

    def sleep(seconds):
        slept.append(seconds)
        qid = [line.split()[1] for line in logs if line.endswith("requested for " + card["set_id"])][0]
        finish(qid)

    kwargs = dict(cards=[card], model=MODEL, revision=REVISION, samples=4, count=5,
                  max_new_tokens=512, thinking=False, sampling=SAMPLING, sleep=sleep,
                  log=logs.append, poll_seconds=7)
    created = op.create_evaluations(client, **kwargs)
    assert slept == [7]
    assert created[0]["job_id"].startswith(f"order-eval-{REVISION[:8]}-{card['set_id']}-n5x4-")
    job, _ = asyncio.run(job_store.read_job(created[0]["job_id"]))
    assert job.slots_per_prompt == 4 and job.checkpoint_revision == REVISION
    assert any("qualification" in line and "qualified" in line for line in logs)
    # The same command again finds its qualification done and the job declared.
    again = op.create_evaluations(client, **{**kwargs, "sleep": lambda s: pytest.fail("waited")})
    assert again[0]["job_id"] == created[0]["job_id"]
    assert again[0]["qualification_id"] == created[0]["qualification_id"]


def test_a_refused_model_gets_no_job(admin):  # noqa: F811
    client, logs = client_for(admin), []
    card = card_of(admin)

    def sleep(seconds):
        qid = [line.split()[1] for line in logs if line.endswith("requested for " + card["set_id"])][0]
        finish(qid, status=qual.REFUSED)

    with pytest.raises(RuntimeError, match="qualification refused: .*band_over_ceiling"):
        op.create_evaluations(client, cards=[card], model=MODEL, revision=REVISION, samples=2,
                              max_new_tokens=512, thinking=False, sampling=SAMPLING,
                              sleep=sleep, log=logs.append)
    assert not [k for k in admin.registry["entries"] if k.startswith("order-eval")]


def test_create_refuses_more_problems_than_the_set_holds(admin):  # noqa: F811
    with pytest.raises(ValueError, match="holds 8 problems, not 9"):
        op.create_evaluations(client_for(admin), cards=[card_of(admin)], model=MODEL,
                              revision=REVISION, samples=2, count=9, max_new_tokens=512,
                              thinking=False, sampling=SAMPLING)


def test_grade_brings_the_three_files_home(admin, tmp_path):  # noqa: F811
    _complete_job(admin)
    answer = op.grade_job(client_for(admin), "order-eval-7", out=tmp_path / "home",
                          sleep=lambda s: None)
    assert answer["complete"] is True
    report = json.loads((tmp_path / "home" / "report.json").read_text())
    assert report["eval_id"] == "order-eval-7"
    assert pq.read_table(tmp_path / "home" / "graded.parquet").num_rows == 20
    assert json.loads((tmp_path / "home" / "manifest.json").read_text())["complete"] is True


def write_grading(directory, *, correct, n_problems, samples=2, sampling=None, set_id="s1",
                  revision="a" * 40):
    """correct: {problem: [bool, ...]}; problems absent from it have no row."""
    directory.mkdir(parents=True)
    rows = [{"env": "math", "problem_id": p, "correct": c} for p, cs in correct.items() for c in cs]
    pq.write_table(pa.Table.from_pylist(rows), directory / "graded.parquet")
    rate = sum(sum(cs) for cs in correct.values()) / (n_problems * samples)
    (directory / "report.json").write_text(json.dumps({
        "eval_id": directory.name,
        "envs": {"math": {"pass@1": {"value": rate}, "n_problems": n_problems}},
        "provenance": {"model": "org/m", "revision": revision,
                       "sampling": sampling or {"temperature": 0.6}, "max_new_tokens": 512,
                       "thinking": True,
                       "sets": [{"set_id": set_id, "env": "math", "problems": n_problems,
                                 "samples": samples}]}}))


def test_compare_two_checkpoints_on_the_same_problems(tmp_path):
    write_grading(tmp_path / "a", n_problems=4,
                  correct={"p0": [True, False], "p1": [False, False], "p2": [True, True]})
    write_grading(tmp_path / "b", n_problems=4, revision="b" * 40,
                  correct={"p0": [True, True], "p1": [True, False], "p2": [True, True]})
    result = op.compare_reports(tmp_path / "a", tmp_path / "b")
    math = result["envs"]["math"]
    assert math["a"] == pytest.approx(3 / 8) and math["b"] == pytest.approx(5 / 8)
    # p0 +0.5, p1 +0.5, p2 0, p3 (no row on either side) 0: over 4 problems.
    assert math["diff"] == pytest.approx(0.25)
    low, high = math["ci95"]
    assert low <= 0.25 <= high and low >= 0.0
    assert result["b"]["revision"] == "b" * 40


def test_compare_refuses_different_conditions(tmp_path):
    write_grading(tmp_path / "a", n_problems=2, correct={"p0": [True, True]})
    write_grading(tmp_path / "b", n_problems=2, correct={"p0": [True, True]},
                  sampling={"temperature": 1.0})
    with pytest.raises(ValueError, match=r"differ in \['sampling'\]"):
        op.compare_reports(tmp_path / "a", tmp_path / "b")
    write_grading(tmp_path / "c", n_problems=2, correct={"p0": [True, True]}, set_id="s2")
    with pytest.raises(ValueError, match=r"\['sets'\]"):
        op.compare_reports(tmp_path / "a", tmp_path / "c")


def test_a_failed_qualification_is_asked_again_only_with_another_attempt(admin):  # noqa: F811
    client, logs = client_for(admin), []
    card = card_of(admin)

    def qid_of():
        return [line.split()[1] for line in logs
                if line.endswith("requested for " + card["set_id"])][-1]

    kwargs = dict(cards=[card], model=MODEL, revision=REVISION, samples=2,
                  max_new_tokens=512, thinking=False, sampling=SAMPLING, log=logs.append)
    with pytest.raises(RuntimeError, match="rerun with --attempt 1"):
        op.create_evaluations(client, **kwargs,
                              sleep=lambda s: finish(qid_of(), status=qual.FAILED))
    failed = qid_of()
    created = op.create_evaluations(client, **kwargs, attempt=1,
                                    sleep=lambda s: finish(qid_of()))
    assert created[0]["qualification_id"] != failed


def test_a_bad_job_id_costs_no_qualification(admin):  # noqa: F811
    with pytest.raises(ValueError, match=r"\[a-z0-9-\]"):
        op.create_evaluations(client_for(admin), cards=[card_of(admin)], model=MODEL,
                              revision=REVISION, samples=2, max_new_tokens=512, thinking=False,
                              sampling=SAMPLING, job_id="order-eval-Bad_Id")
    card = card_of(admin)
    conditions = {"model": MODEL, "revision": REVISION, "set_id": card["set_id"],
                  "problems": card["count"], "sampling": SAMPLING, "max_new_tokens": 512,
                  "thinking": False}
    qid = op.qualification_id_for("order-", {**conditions, "completions": 32, "attempt": 0})
    record, _ = asyncio.run(qual.QualificationStore().read(qid))
    assert record is None
