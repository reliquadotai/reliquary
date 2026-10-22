"""The run monitor's held-out feed (reliquary.run-benchmarks.v1) built from graded served-job reports."""

from __future__ import annotations

import asyncio
import copy
import functools
import json

import pytest

from reliquary.eval import run_feed
from reliquary.eval.run_feed import FeedRejected, build_feed, parse_feed

RUN = "5F3sa2TJ-r7-trainer"
NOW = 1_800_000_000_000


def report(set_id="verifiers-aime25-r0-n30", *, step=None, score=0.5, at=1_799_000_000.0, run=RUN,
           model=None, problems=30, samples=8, temperature=0.6, env="verifiers:aime25"):
    """The fields of an eval-report/v2 a served job's grading writes."""
    provenance = {
        "model": model or (f"ReliquaryForge/teutonic-rl-step-{step:08d}" if step is not None
                           else "ReliquaryForge/Teutonic-I-Reliquary-SFT"),
        "revision": "18dfb97df4d76de4b694df090b9162325327dece",
        "sampling": {"temperature": temperature, "top_p": 0.95, "top_k": 0}, "thinking": True,
        "max_new_tokens": 32768, "verification": {"scheme": "toploc-v1"},
        "sets": [{"set_id": set_id, "env": env, "source": env, "split": None, "problems": problems,
                  "samples": samples, "prompts_sha256": "c" * 64, "grading_sha256": "7" * 64}],
    }
    if step is not None:
        provenance.update(rl_step=step, rl_checkpoint_version=step)
        if run is not None:
            provenance["rl_run_id"] = run
    headline = None if score is None else {"value": score, "ci95": [0.0, 1.0]}
    return {"schema": "reliquary/eval-report/v2", "created_at": at, "provenance": provenance,
            "envs": {env: {"pass@1": headline, "n_problems": problems}}}


def test_one_benchmark_per_set_with_its_baseline_and_steps_in_order():
    gpqa = dict(set_id="verifiers-gpqa-diamond", env="verifiers:gpqa", problems=198, samples=4)
    checkpoints = [("eval-r7-s00040-aime", report(step=40, score=0.6, at=1_799_000_300.0)),
                   ("eval-r7-s00020-aime", report(step=20, score=0.55, at=1_799_000_200.0)),
                   # graded after step 40 (a backlog), its `at` is held up to keep the series ordered
                   ("eval-r7-s00030-aime", report(step=30, score=None, at=1_799_000_100.0)),
                   ("eval-r7-s00020-gpqa", report(step=20, score=0.4, **gpqa))]
    baselines = [("eval-r7-base-aime", report(score=0.5)), ("eval-r7-base-gpqa", report(score=0.38, **gpqa))]
    feed = build_feed(RUN, checkpoints, baselines, observed_at_ms=NOW,
                      labels={"verifiers-aime25-r0-n30": "AIME 2025"})
    assert parse_feed(json.loads(run_feed.encode_feed(feed)), RUN, NOW)
    assert feed["schema"] == "reliquary.run-benchmarks.v1" and feed["scope"] == "held_out"
    assert feed["run_id"] == RUN and feed["observed_at"] == NOW
    aime, gpqa_series = feed["benchmarks"]
    assert aime["id"] == "verifiers-aime25-r0-n30" and aime["label"] == "AIME 2025"
    assert gpqa_series["label"] == "verifiers:gpqa (198 problems, 4 samples)"
    assert aime["metric"] == "pass@1"
    assert aime["protocol"]["attempts"] == 8 and aime["protocol"]["max_output_tokens"] == 32768
    assert aime["protocol"]["dataset"] == "verifiers:aime25"
    assert [p["step"] for p in aime["checkpoints"]] == [20, 30, 40]
    assert [p["score"] for p in aime["checkpoints"]] == [0.55, None, 0.6]
    assert [p["at"] for p in aime["checkpoints"]] == [1_799_000_200_000, 1_799_000_200_000, 1_799_000_300_000]
    assert aime["checkpoints"][0]["checkpoint"] == "ReliquaryForge/teutonic-rl-step-00000020"
    assert all(p["samples"] == 30 and p["cost_usd"] is None and p["gpu_hours"] is None
               for p in aime["checkpoints"])
    assert aime["baseline"] == {"checkpoint": "ReliquaryForge/Teutonic-I-Reliquary-SFT@18dfb97df4d7",
                                "step": None, "at": 1_799_000_000_000, "score": 0.5, "samples": 30,
                                "cost_usd": None, "gpu_hours": None}
    assert gpqa_series["baseline"]["score"] == 0.38 and len(gpqa_series["checkpoints"]) == 1
    assert aime["protocol"]["id"] != gpqa_series["protocol"]["id"]


def test_a_set_without_a_baseline_or_points_beyond_the_cap():
    checkpoints = [(f"j{s}", report(step=s, at=1_799_000_000.0 + s)) for s in range(1, 131)]
    feed = build_feed(RUN, checkpoints, observed_at_ms=NOW)
    (series,) = feed["benchmarks"]
    assert series["baseline"] is None
    assert [p["step"] for p in series["checkpoints"]] == list(range(11, 131))         # the newest 120
    assert len(build_feed(RUN, checkpoints, observed_at_ms=NOW, max_checkpoints=5)["benchmarks"][0]
               ["checkpoints"]) == 5
    # a grading clock ahead of the feed's is held at observed_at
    ahead = build_feed(RUN, [("j", report(step=1, at=NOW / 1000 + 60))], observed_at_ms=NOW)
    assert ahead["benchmarks"][0]["checkpoints"][0]["at"] == NOW
    assert len(run_feed.encode_feed(feed)) < run_feed.MAX_FEED_BYTES


@pytest.mark.parametrize("checkpoints, baselines, match", [
    ([("j", report(step=4, run="other-r1-trainer"))], [], "a checkpoint of run other"),
    ([("j", report())], [], "no rl_step"),
    ([("a", report(step=4)), ("b", report(step=4))], [], "graded twice"),
    ([("a", report(step=4)), ("b", report(step=5, temperature=1.0))], [], "another protocol"),
    ([("a", report(step=4))], [("b", report(samples=4))], "another protocol"),
    ([("a", report(step=4))], [("b", report()), ("c", report())], "two baselines"),
    ([(f"j{i}", report(set_id=f"set-{i}", step=1)) for i in range(25)], [], "at most 24"),
])
def test_what_would_mislead_the_dashboard_is_refused(checkpoints, baselines, match):
    with pytest.raises(ValueError, match=match):
        build_feed(RUN, checkpoints, baselines, observed_at_ms=NOW)


def test_a_report_without_an_rl_run_id_is_taken_as_named():
    feed = build_feed(RUN, [("j", report(step=3, run=None))], observed_at_ms=NOW)
    assert feed["benchmarks"][0]["checkpoints"][0]["step"] == 3
    with pytest.raises(ValueError, match="run id"):
        build_feed("bad id", [("j", report(step=3))], observed_at_ms=NOW)


# --- the platform's validator, case by case (shared/run-benchmarks.ts) ------------------------------------------

def _valid():
    return build_feed(RUN, [("a", report(step=10)), ("b", report(step=20, at=1_799_000_100.0))],
                      [("base", report())], observed_at_ms=NOW)


def _bench(feed):
    return feed["benchmarks"][0]


@pytest.mark.parametrize("change", [
    lambda f: f.update(run_id="another-run"),                       # another run's feed is refused whole
    lambda f: f.update(schema="reliquary.run-benchmarks.v2"),
    lambda f: f.update(scope="training"),
    lambda f: f.update(observed_at=NOW + 5001),
    lambda f: f.update(observed_at=-1),
    lambda f: f.update(benchmarks=[{**_bench(f), "id": f"b{i}"} for i in range(25)]),
    lambda f: _bench(f).update(metric="pass@k"),
    lambda f: _bench(f).update(checkpoints=[{**_bench(f)["checkpoints"][0], "step": i, "checkpoint": f"c{i}"}
                                            for i in range(121)]),
    lambda f: _bench(f)["checkpoints"][1].update(step=10),            # strictly increasing steps
    lambda f: _bench(f)["checkpoints"][1].update(at=0),               # `at` never goes back
    lambda f: _bench(f)["checkpoints"][1].update(checkpoint=_bench(f)["checkpoints"][0]["checkpoint"]),
    lambda f: _bench(f)["checkpoints"][0].update(at=NOW + 1),         # after observed_at
    lambda f: _bench(f)["checkpoints"][0].update(score=1.01),
    lambda f: _bench(f)["checkpoints"][0].update(score=-0.1),
    lambda f: _bench(f)["checkpoints"][0].pop("score"),               # undefined is not null
    lambda f: _bench(f)["checkpoints"][0].update(samples=0),          # a score over nothing
    lambda f: _bench(f)["checkpoints"][0].update(samples=2.5),
    lambda f: _bench(f)["checkpoints"][0].update(step=1.5),
    lambda f: _bench(f)["checkpoints"][0].update(step=True),
    lambda f: _bench(f)["checkpoints"][0].update(cost_usd=-1),
    lambda f: _bench(f)["checkpoints"][0].update(checkpoint=""),
    lambda f: _bench(f)["checkpoints"][0].update(checkpoint="x" * 161),
    lambda f: _bench(f)["baseline"].update(samples=0),
    lambda f: _bench(f)["baseline"].pop("step"),                      # null, not missing
    lambda f: _bench(f)["checkpoints"][0].update(step=None),          # null only for the baseline
    lambda f: _bench(f).update(id="x" * 97),
    lambda f: _bench(f).update(label="two\nlines"),
    lambda f: _bench(f)["protocol"].update(attempts=0),
    lambda f: _bench(f)["protocol"].update(attempts=1025),
    lambda f: _bench(f)["protocol"].update(max_output_tokens=10_000_001),
    lambda f: _bench(f)["protocol"].pop("harness"),
    lambda f: f["benchmarks"].append(copy.deepcopy(_bench(f))),       # duplicate ids
])
def test_the_ported_validator_refuses_what_the_platform_refuses(change):
    feed = _valid()
    assert parse_feed(copy.deepcopy(feed), RUN, NOW)
    change(feed)
    with pytest.raises(FeedRejected):
        parse_feed(feed, RUN, NOW)


def test_the_ported_validator_accepts_gaps_and_measured_resources():
    feed = _valid()
    _bench(feed)["checkpoints"][0].update(score=None, samples=0, cost_usd=12.5, gpu_hours=3)
    _bench(feed)["baseline"] = None
    parsed = parse_feed(feed, RUN, NOW)
    assert parsed["benchmarks"][0]["checkpoints"][0]["score"] is None
    _bench(feed)["checkpoints"][0].pop("cost_usd")                   # undefined optional numbers read as null
    assert parse_feed(feed, RUN, NOW)["benchmarks"][0]["checkpoints"][0]["cost_usd"] is None


def test_the_baseline_step_is_an_explicit_null_or_an_integer():
    feed = _valid()
    assert _bench(feed)["baseline"]["step"] is None
    assert parse_feed(feed, RUN, NOW)["benchmarks"][0]["baseline"]["step"] is None
    _bench(feed)["baseline"]["step"] = 0
    assert parse_feed(feed, RUN, NOW)["benchmarks"][0]["baseline"]["step"] == 0


# --- where the reports are, publication, CLI --------------------------------------------------------------------

class FakeStore:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    async def get_bytes(self, key):
        return self.objects.get(key)

    def put_bytes(self, key, body):
        self.objects[key] = body


def _write(root, job, body):
    (root / job).mkdir(parents=True)
    (root / job / "report.json").write_text(json.dumps(body))


def test_reports_are_collected_from_a_directory_or_the_bucket(tmp_path):
    _write(tmp_path, "eval-r7-s00010-aime", report(step=10))
    _write(tmp_path, "eval-r7-base-aime", report())
    _write(tmp_path, "other-job", report(step=1))
    local = run_feed.LocalReports(tmp_path)
    checkpoints, baselines, missing = asyncio.run(run_feed.collect(
        local, prefix="eval-r7-", jobs=["eval-r7-s00020-aime"], baseline_jobs=["eval-r7-base-aime"]))
    assert [j for j, _ in checkpoints] == ["eval-r7-s00010-aime"]
    assert [j for j, _ in baselines] == ["eval-r7-base-aime"] and missing == ["eval-r7-s00020-aime"]

    store = FakeStore({"evaluations/eval-r7-s00010-aime/report.json": json.dumps(report(step=10)).encode()})

    async def list_jobs():
        return ["eval-r7-s00010-aime", "eval-r7-s00020-aime", "corpus-math"]

    bucket = run_feed.BucketReports(store, list_jobs)
    checkpoints, _, missing = asyncio.run(run_feed.collect(bucket, prefix="eval-r7-"))
    assert [j for j, _ in checkpoints] == ["eval-r7-s00010-aime"] and missing == ["eval-r7-s00020-aime"]


def test_publication_writes_the_stable_key_of_the_run():
    store = FakeStore()
    answer = run_feed.publish_feed(_valid(), store=store, base_url="https://feeds.example/")
    key = f"rl-runs/{RUN}/benchmarks.json"
    assert answer == {"key": key, "bytes": len(store.objects[key]),
                      "url": f"https://feeds.example/{key}"}
    assert parse_feed(json.loads(store.objects[key]), RUN, NOW)
    assert run_feed.publish_feed(_valid(), store=FakeStore())["url"] is None


def test_eval_feed_builds_writes_and_publishes(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    _write(tmp_path / "r", "eval-r7-s00010-aime", report(step=10))
    _write(tmp_path / "r", "eval-r7-base-aime", report())
    store = FakeStore()
    monkeypatch.setattr(run_feed, "publish_feed", functools.partial(run_feed.publish_feed, store=store))
    monkeypatch.delenv("RELIQUARY_RUN_BENCHMARKS_BASE_URL", raising=False)
    argv = ["eval", "feed", "--run-id", RUN, "--job-prefix", "eval-r7-", "--baseline-job", "eval-r7-base-aime",
            "--reports-dir", str(tmp_path / "r"), "--out", str(tmp_path / "feed.json"),
            "--label", "verifiers-aime25-r0-n30=AIME 2025"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["points"] == 1 and summary["baselines"] == 1 and "published" not in summary
    feed = json.loads((tmp_path / "feed.json").read_text())
    assert feed["benchmarks"][0]["label"] == "AIME 2025" and not store.objects
    result = CliRunner().invoke(app, argv + ["--publish"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["published"]["key"] in store.objects
    bad = CliRunner().invoke(app, ["eval", "feed", "--run-id", RUN, "--reports-dir", str(tmp_path / "r")])
    assert bad.exit_code == 1 and "--jobs" in bad.output


def test_a_grading_is_uploaded_create_only(tmp_path):
    from reliquary.eval.job_grading import GRADING_FILES, upload_grading
    from reliquary.eval.storage import SetConflict

    class Subnet:
        def __init__(self):
            self.objects = {}

        async def put_bytes(self, key, body):
            if key in self.objects and self.objects[key] != body:
                raise SetConflict(key)
            self.objects[key] = body

    for name in GRADING_FILES:
        (tmp_path / name).write_bytes(name.encode())
    subnet = Subnet()
    keys = asyncio.run(upload_grading("eval-r7-s00010-aime", tmp_path, subnet=subnet))
    assert keys == [f"evaluations/eval-r7-s00010-aime/{n}" for n in GRADING_FILES]
    asyncio.run(upload_grading("eval-r7-s00010-aime", tmp_path, subnet=subnet))       # same bytes: no-op
    (tmp_path / "report.json").write_bytes(b"regraded")
    with pytest.raises(SetConflict):
        asyncio.run(upload_grading("eval-r7-s00010-aime", tmp_path, subnet=subnet))
