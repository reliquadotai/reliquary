"""Grading an episode or SWE eval job from its grade documents.

Every passing submission of a suite or signed-sandbox job already has a grade
document written by the job's grader (a regrade supersedes it): ``eval grade``
reads those, never the heavy submission records, and feeds them pre-graded
into the report path a single-turn job uses. The headline counts missing and
ungraded samples as failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pyarrow.parquet as pq
import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.corpus.job import parse_job
from reliquary.eval import prompt_source as ps
from reliquary.eval.prompt_source import eval_source_for
from reliquary.eval.storage import subnet_key
from tests.unit.test_corpus_job_episode import _manifest as _swe_manifest
from tests.unit.test_corpus_job_episode import _signed
from tests.unit.test_suite_eval_rows import ENV
from tests.unit.test_suite_eval_rows import _manifest as _suite_manifest

SPLIT = "benchmark"


def _rows(interaction, set_id, count=2):
    env = ENV if interaction == "episode" else "reliquary-swe"
    return [{"problem_id": f"{set_id}-{i:06d}", "env": env, "set_id": set_id,
             "interaction": interaction, "split": SPLIT, "source_index": 40 + i,
             "instance_id": f"t-{40 + i}"} for i in range(count)]


def _body(rows):
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _publish(tmp_path, monkeypatch, interaction="episode", count=2):
    """The set's prompts (as validators fetch them) and its card and grading
    rows in the subnet bucket; the job over its ``count`` rows, two samples a row."""
    set_id = f"{'telecom' if interaction == 'episode' else 'swe'}-bench-r0-n{count}"
    rows = _rows(interaction, set_id, count)
    body = _body(rows)
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [lambda source: body])
    grading = _body([{k: row[k] for k in ("problem_id", "interaction", "split", "source_index",
                                          "instance_id")} for row in rows])
    card = {"schema": "reliquary/eval-set/v1", "set_id": set_id, "env": rows[0]["env"],
            "source": rows[0]["env"], "source_kind": "catalog", "interaction": interaction,
            "split": SPLIT, "count": count, "prompts_sha256": hashlib.sha256(body).hexdigest(),
            "grading_sha256": hashlib.sha256(grading).hexdigest(), "disjointness": {}}
    subnet = LocalDirectorySink(tmp_path / "subnet")
    asyncio.run(subnet.put_bytes(subnet_key(set_id, "grading.jsonl"), grading))
    asyncio.run(subnet.put_bytes(subnet_key(set_id, "set.json"), json.dumps(card).encode()))
    source = eval_source_for(set_id, body, count).name
    if interaction == "episode":
        job = parse_job(_suite_manifest(prompt_source=source, split=SPLIT, prompt_count=count))
    else:
        job = parse_job(_swe_manifest(prompt_source=source, prompt_count=count,
                                      episode=_signed(env={"split": SPLIT, "num_images": 500})))
    return job, rows, subnet


class _GradeRecords:
    """An episode job's documents: verdicts, grades, regrades, voided ids, the
    settlement, and submission records that only the tail reader may touch."""

    def __init__(self, job, *, submissions_readable=True):
        self.job = job
        self.subs, self.verdicts, self.grades, self.regrades = {}, {}, {}, {}
        self.voided, self.settled = set(), []
        self.submissions_readable = submissions_readable
        self.full_reads, self.meta_reads = [], []

    async def read_job(self, job_id):
        return self.job, None

    async def read_ledgers(self, job_id):
        return {}, None

    async def list_submission_ids(self, job_id):
        return sorted(self.subs)

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_grade(self, job_id, sid):
        return self.grades.get(sid)

    async def read_regrade(self, job_id, sid):
        return self.regrades.get(sid)

    async def read_submission(self, job_id, sid):
        self.full_reads.append(sid)
        if not self.submissions_readable:
            raise AssertionError(f"read the submission record {sid}")
        return self.subs.get(sid)

    async def read_submission_meta(self, job_id, sid):
        self.meta_reads.append(sid)
        record = self.subs.get(sid)
        if record is None:
            return None
        return {"hotkey": record["hotkey"], "received_at": 1.0, "token_count": 7,
                "prompt_index": record["prompt_index"]}

    async def read_settlement(self, job_id):
        return {"settled": list(self.settled), "pending": None}, None

    def submit(self, sid, prompt, *, reward=None, status="ok", certified=True, grade=True,
               passed=True, tokens=100):
        hotkey = f"hk{prompt}"
        self.subs[sid] = {"prompt_index": prompt, "hotkey": hotkey, "completions": [{}]}
        self.verdicts[sid] = {"passed": passed, "audited": True, "hotkey": hotkey,
                              "token_count": tokens}
        self.settled.append(sid)
        if grade:
            self.grades[sid] = _grade(sid, prompt, reward=reward, status=status,
                                      certified=certified)


def _grade(sid, prompt, *, reward, status="ok", certified=True):
    document = {"submission_id": sid, "hotkey": f"hk{prompt}", "prompt_index": prompt,
                "status": status, "graded_success": reward == 1.0,
                "replay_certified": certified, "instance_id": f"t-{40 + prompt}"}
    if status == "ok":
        document["grade"] = {"reward": reward, "success": reward == 1.0}
        document["replay"] = {"status": "ok", "certified": certified,
                              "termination_reason": "agent_stop"}
    else:
        document["replay"] = None
    return document


def _sid(prompt, k):
    return f"{prompt:032x}{k:032x}"


def _grade_job(tmp_path, job, records, subnet, **kw):
    from reliquary.eval.job_grading import grade_served_job

    async def entries():
        return ()

    return asyncio.run(grade_served_job(job.job_id, out=tmp_path / "out", records=records,
                                        subnet=subnet, entries=entries,
                                        read_job=records.read_job,
                                        read_ledgers=records.read_ledgers, **kw))


def _report(tmp_path):
    return json.loads((tmp_path / "out" / "report.json").read_text())


@pytest.mark.parametrize("interaction,scheme", [("episode", "suite-replay"),
                                                ("swe", "signed-sandbox")])
def test_episode_grade_scores_rewards(tmp_path, monkeypatch, interaction, scheme):
    job, rows, subnet = _publish(tmp_path, monkeypatch, interaction)
    records = _GradeRecords(job, submissions_readable=False)
    for (prompt, k), reward in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [1.0, 0.0, 1.0, 1.0]):
        records.submit(_sid(prompt, k), prompt, reward=reward)
    manifest = _grade_job(tmp_path, job, records, subnet)
    assert manifest["complete"] is True and manifest["rows"] == 4
    report = _report(tmp_path)
    env = report["envs"][rows[0]["env"]]
    assert env["pass@1"]["value"] == pytest.approx(0.75)
    assert (env["expected_rows"], env["graded_rows"], env["ungraded_rows"],
            env["missing_rows"]) == (4, 4, 0, 0)
    assert env["mean_completion_tokens"] == 100 and env["format_failure_rate"] == 0
    provenance = report["provenance"]
    assert provenance["verification"]["scheme"] == scheme
    assert provenance["verification"]["source"] == "grade documents"
    assert provenance["job_complete"] is True and provenance["miner_hotkeys"] == 2
    table = pq.read_table(tmp_path / "out" / "graded.parquet").to_pylist()
    assert [(r["problem_id"], r["sample_index"], r["score"], r["correct"]) for r in table] == [
        (rows[0]["problem_id"], 0, 1.0, True), (rows[0]["problem_id"], 1, 0.0, False),
        (rows[1]["problem_id"], 0, 1.0, True), (rows[1]["problem_id"], 1, 1.0, True)]
    detail = json.loads(table[0]["grader_detail"])
    assert detail == {"status": "ok", "graded_success": True, "instance_id": "t-40"}
    assert all(r["completion"] == "" for r in table)
    assert records.full_reads == []


def test_episode_grade_ungraded_counts_as_failure_in_headline(tmp_path, monkeypatch):
    job, rows, subnet = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=1.0)
    records.submit(_sid(0, 1), 0, status="unreplayable", certified=False)
    records.submit(_sid(1, 0), 1, reward=1.0)
    records.submit(_sid(1, 1), 1, reward=0.0)
    manifest = _grade_job(tmp_path, job, records, subnet)
    assert manifest["complete"] is False
    env = _report(tmp_path)["envs"][ENV]
    # Headline: the unreplayable sample is a failure, 1/2 and 1/2.
    assert env["pass@1"]["value"] == pytest.approx(0.5)
    assert env["graded_rows"] == 3 and env["ungraded_rows"] == 1
    # Beside it, graded samples only: 1/1 and 1/2 problem by problem, 2/3 of the rows.
    excluded = env["excluding_ungraded_and_missing"]["pass@1"]["value"]
    assert excluded == pytest.approx((1 + 0.5) / 2)
    table = pq.read_table(tmp_path / "out" / "graded.parquet").to_pylist()
    graded = [r for r in table if r["score"] is not None]
    assert sum(r["correct"] for r in graded) / len(graded) == pytest.approx(2 / 3)
    ungraded = [r for r in table if r["score"] is None]
    assert len(ungraded) == 1 and ungraded[0]["correct"] is None
    assert json.loads(ungraded[0]["grader_detail"])["status"] == "unreplayable"


def test_episode_grade_refuses_ungraded_without_flag(tmp_path, monkeypatch):
    from reliquary.eval.job_grading import JobNotGradable

    job, rows, subnet = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    for prompt in range(2):
        for k in range(2):
            records.submit(_sid(prompt, k), prompt, reward=1.0, grade=(prompt, k) != (1, 1))
    with pytest.raises(JobNotGradable, match="ungraded"):
        _grade_job(tmp_path, job, records, subnet)
    # With the flag, it is an ungraded sample of its problem, placed through
    # the record's tail, never its body.
    manifest = _grade_job(tmp_path, job, records, subnet, allow_incomplete=True)
    assert manifest["complete"] is False
    env = _report(tmp_path)["envs"][ENV]
    assert (env["graded_rows"], env["ungraded_rows"], env["missing_rows"]) == (3, 1, 0)
    assert env["pass@1"]["value"] == pytest.approx((1 + 0.5) / 2)
    assert records.meta_reads == [_sid(1, 1)] and records.full_reads == []


def test_episode_rows_prefer_regrade(tmp_path, monkeypatch):
    from reliquary.eval.grading import EpisodeJobRows

    job, rows, _ = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=0.0)
    records.regrades[_sid(0, 0)] = _grade(_sid(0, 0), 0, reward=1.0)
    records.submit(_sid(0, 1), 0, reward=1.0)
    records.voided.add(_sid(0, 1))                       # quarantined: left out
    records.submit(_sid(1, 0), 1, reward=1.0, passed=False)   # audit failed: missing
    records.submit(_sid(1, 1), 1, reward=1.0, certified=False)  # uncertified: ungraded
    collected = asyncio.run(EpisodeJobRows.collect(job, records, allow_incomplete=False))

    async def drain():
        return [row async for row in collected]

    out = asyncio.run(drain())
    assert [(r["problem_id"], r["sample_index"], r["pregraded"][0]) for r in out] == [
        (rows[0]["problem_id"], 0, 1.0), (rows[1]["problem_id"], 0, None)]
    assert out[0]["finish_reason"] == "agent_stop" and out[0]["completion"] == ""
    assert collected.samples_by_prompt == {0: 1, 1: 1}
    assert collected.verdicts == 4 and collected.audited == 4 and collected.ungraded == 0
    assert collected.hotkeys == {"hk0", "hk1"}


def test_episode_rows_never_read_submissions(tmp_path, monkeypatch):
    from reliquary.eval.grading import EpisodeJobRows

    job, rows, _ = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=1.0)
    records.submit(_sid(0, 1), 0, grade=False)
    records.submit(_sid(1, 0), 1, reward=0.5)
    collected = asyncio.run(EpisodeJobRows.collect(job, records, allow_incomplete=True))

    async def drain():
        return [row async for row in collected]

    out = asyncio.run(drain())
    assert len(out) == 3 and collected.ungraded == 1
    assert collected.samples_by_prompt == {0: 2, 1: 1}
    assert records.full_reads == [] and records.meta_reads == [_sid(0, 1)]


def test_episode_rows_a_grade_on_another_task_scores_nothing(tmp_path, monkeypatch):
    from reliquary.eval.grading import EpisodeJobRows

    job, rows, _ = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=1.0)
    records.grades[_sid(0, 0)]["instance_id"] = "t-999"
    collected = asyncio.run(EpisodeJobRows.collect(job, records, allow_incomplete=False))

    async def drain():
        return [row async for row in collected]

    (row,) = asyncio.run(drain())
    assert row["pregraded"] == (None, {"status": "task_mismatch", "graded_success": True,
                                       "instance_id": "t-999"})


def test_single_turn_grade_unchanged(tmp_path, monkeypatch):
    """A single-turn eval job is graded from its records' completions, through
    ``JobRows``, exactly as before: the report is what that path writes."""
    from reliquary.eval import grading
    from reliquary.eval.sets import build_source_set
    from reliquary.eval.storage import publish_set
    from tests.unit.test_admin_eval_jobs import _JobRecords
    from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle, opener

    handle = fake_handle([FakeTask(i) for i in range(5)])
    card = build_source_set("verifiers:fake", out=tmp_path / "verifiers-fake-r0-n5",
                            open_taskset=opener(handle))
    subnet = LocalDirectorySink(tmp_path / "subnet")
    asyncio.run(publish_set(tmp_path / "verifiers-fake-r0-n5", platform=None, subnet=subnet))
    body = (tmp_path / "verifiers-fake-r0-n5" / "prompts.jsonl").read_bytes()
    monkeypatch.setattr(ps, "_loaded", {})
    monkeypatch.setattr(ps, "FETCHERS", [lambda source: body])
    job = parse_job(_swe_manifest(with_episode=False, job_id="eval-single",
                                  prompt_source=eval_source_for(card["set_id"], body, 2).name,
                                  prompt_count=2, renderer_id="chat-template-thinking-v1",
                                  eos_token_id=151645))
    records = _JobRecords()
    records.read_job = lambda job_id: _aresult((job, None))
    records.read_ledgers = lambda job_id: _aresult(({}, None))
    for prompt in range(2):
        for k in range(2):
            sid = _sid(prompt, k)
            text = f"<think>x</think>{prompt}" if k == 0 else "wrong"
            records.subs[sid] = {"prompt_index": prompt, "hotkey": f"hk{prompt}",
                                 "completions": [{"text": text, "tokens": [1, 151645]}]}
            records.verdicts[sid] = {"passed": True, "audited": True, "hotkey": f"hk{prompt}"}
            records.settled.append(sid)
    _grade_job(tmp_path, job, records, subnet, open_taskset=lambda name, args: handle)
    report = _report(tmp_path)
    env = report["envs"][card["env"]]
    assert env["pass@1"]["value"] == pytest.approx(0.5)
    assert report["provenance"]["verification"]["scheme"] == "toploc-v1"
    assert report["provenance"]["verification"]["source"] == "task contract"
    table = pq.read_table(tmp_path / "out" / "graded.parquet").to_pylist()
    assert {r["completion"] for r in table} == {"<think>x</think>0", "<think>x</think>1",
                                                 "wrong"}
    # The same report as grading the records through JobRows directly.
    collected = asyncio.run(grading.collect_job_records(job, records))
    direct = LocalDirectorySink(tmp_path / "direct")
    asyncio.run(grading.grade_evaluation(
        eval_id=job.job_id, set_ids=[card["set_id"]], completion_keys=[],
        problems_per_set={card["set_id"]: 2}, samples_per_set={card["set_id"]: 2},
        provenance=report["provenance"], job_rows=grading.JobRows(job=job, collected=collected),
        platform=direct, subnet=subnet, open_taskset=lambda name, args: handle,
        work_dir=tmp_path / "work"))
    again = json.loads((tmp_path / "direct" / grading.evaluation_prefix(job.job_id) /
                        "report.json").read_text())
    for document in (report, again):
        document.pop("created_at")
    assert again["envs"] == report["envs"]
    assert pq.read_table(tmp_path / "direct" / grading.evaluation_prefix(job.job_id) /
                         "graded.parquet").to_pylist() == table


async def _aresult(value):
    return value


def test_episode_grade_without_the_card_in_the_bucket_takes_the_episode_path(tmp_path,
                                                                           monkeypatch):
    """The path is decided from the set's rows (already in hand), never from the
    card: a card the bucket does not hold is the grading's SetUnknown, not a
    silent single-turn grading that reads the submission records."""
    from reliquary.eval.grading import SetUnknown

    job, rows, subnet = _publish(tmp_path, monkeypatch)
    (tmp_path / "subnet" / subnet_key(rows[0]["set_id"], "set.json")).unlink()
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=1.0)
    with pytest.raises(SetUnknown):
        _grade_job(tmp_path, job, records, subnet, allow_incomplete=True)
    assert records.full_reads == []


def test_an_upload_against_an_episode_set_is_refused(tmp_path, monkeypatch):
    """Uploads (orders) grade completions on this host; an episode or SWE set's
    samples are graded by its job's grader, so its uploads are refused, never
    graded as grader errors."""
    from reliquary.eval.grading import GradeRequestError, grade_evaluation

    for interaction in ("episode", "swe"):
        _, rows, subnet = _publish(tmp_path / interaction, monkeypatch, interaction)
        set_id = rows[0]["set_id"]
        platform = LocalDirectorySink(tmp_path / interaction / "platform")
        with pytest.raises(GradeRequestError, match="grade it from its job"):
            asyncio.run(grade_evaluation(
                eval_id="order-e1", set_ids=[set_id],
                completion_keys=["evaluations/order-e1/completions-00000.jsonl"],
                problems_per_set={set_id: 2}, samples_per_set={set_id: 1},
                provenance={}, platform=platform, subnet=subnet,
                work_dir=tmp_path / "work"))


def test_episode_correct_is_the_grade_documents_success(tmp_path, monkeypatch):
    job, rows, subnet = _publish(tmp_path, monkeypatch)
    records = _GradeRecords(job, submissions_readable=False)
    records.submit(_sid(0, 0), 0, reward=1.0)
    records.grades[_sid(0, 0)]["graded_success"] = False     # the grader's verdict wins
    records.submit(_sid(1, 0), 1, reward=0.25)
    records.grades[_sid(1, 0)]["graded_success"] = True
    records.submit(_sid(1, 1), 1, reward=float("nan"))       # not finite: ungraded
    _grade_job(tmp_path, job, records, subnet, allow_incomplete=True)
    table = pq.read_table(tmp_path / "out" / "graded.parquet").to_pylist()
    assert [(r["sample_index"], r["score"], r["correct"]) for r in table] == [
        (0, 1.0, False), (0, 0.25, True), (1, None, None)]
    note = _report(tmp_path)["provenance"]["note"]
    assert "suite-replay" in note and "TOPLOC proves the completions" not in note
