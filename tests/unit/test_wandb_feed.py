from __future__ import annotations

import json

import sys
import types

import pytest

from reliquary.eval import run_feed
from reliquary.eval.wandb_feed import (SUMMARY_KEY, eval_run_ids, log_feed, mirror, new_rows, publish_summary,
                                      stored_envelope)

RUN = "5H47sFL6-teutonic-rl-async-v1-trainer"
SHORT = {"verifiers-aime25-r0-n30": "aime25", "verifiers-gpqa-r0-n198": "gpqa"}


def _point(step, score):
    return {"step": step, "score": score, "checkpoint": f"step-{step}", "at": 0, "samples": 30}


def _feed(aime=(), gpqa=()):
    return {"run_id": RUN, "benchmarks": [
        {"id": "verifiers-aime25-r0-n30", "baseline": _point(None, 0.55), "checkpoints": [_point(*p) for p in aime]},
        {"id": "verifiers-gpqa-r0-n198", "baseline": _point(None, 0.545), "checkpoints": [_point(*p) for p in gpqa]},
    ]}


class FakeRun:
    def __init__(self, sink, **kw):
        self.sink, self.kw = sink, kw

    def log(self, row):
        self.sink.append(row)

    def finish(self):
        pass


def test_the_eval_run_sits_in_the_trainer_group():
    assert eval_run_ids(RUN) == ("5H47sFL6-teutonic-rl-async-v1", "5H47sFL6-teutonic-rl-async-v1-evals")
    assert eval_run_ids("plain") == ("plain", "plain-evals")


def test_rows_are_one_per_step_with_the_delta_to_the_baseline():
    rows, marks = new_rows(_feed(aime=[(500, 0.6)], gpqa=[(500, 0.5), (1000, None)]), SHORT, set())
    assert rows[0] == {"rl_step": 0, "eval/aime25/pass@1": 0.55, "eval/aime25/delta": 0.0,
                       "eval/gpqa/pass@1": 0.545, "eval/gpqa/delta": 0.0}
    assert rows[1]["rl_step"] == 500 and abs(rows[1]["eval/aime25/delta"] - 0.05) < 1e-9
    assert len(rows) == 2 and "gpqa@1000" not in marks           # ungraded: waits for a later feed


def test_each_point_is_logged_once_and_a_failure_is_retried(tmp_path):
    state, sink = tmp_path / "logged.json", []

    def init(**kw):
        return FakeRun(sink, **kw)

    assert log_feed(_feed(aime=[(500, 0.6)]), short_of=SHORT, state_path=state, init=init, entity="team")["logged"] == 3
    assert log_feed(_feed(aime=[(500, 0.6)]), short_of=SHORT, state_path=state, init=init, entity="team")["logged"] == 0

    def down(**kw):
        raise RuntimeError("wandb down")

    out = log_feed(_feed(aime=[(500, 0.6), (1000, 0.62)]), short_of=SHORT, state_path=state, init=down, entity="team")
    assert out["logged"] == 0 and out["error"] == "RuntimeError"
    out = log_feed(_feed(aime=[(500, 0.6), (1000, 0.62)]), short_of=SHORT, state_path=state, init=init, entity="team")
    assert out == {"run": "5H47sFL6-teutonic-rl-async-v1-evals", "logged": 1}
    assert sink[-1] == {"rl_step": 1000, "eval/aime25/pass@1": 0.62, "eval/aime25/delta": 0.62 - 0.55}
    assert "aime25@1000" in next(iter(json.loads(state.read_text())["targets"].values()))


# --- the run monitor's envelope in the evaluation run's summary ------------------------------------------------

NOW = 1_791_000_000_000


def _report(set_id, score, *, step=None):
    provenance = {"model": "ReliquaryForge/Teutonic-I-Reliquary-SFT", "revision": "18dfb97df4d76de4b694df090b91",
                  "sets": [{"env": "e", "set_id": set_id, "source": "verifiers", "problems": 30, "samples": 8,
                            "prompts_sha256": "a" * 64, "grading_sha256": "b" * 64}],
                  "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": -1}, "max_new_tokens": 32768,
                  "thinking": True, "verification": {"scheme": "signed"}, "rl_run_id": None, "rl_step": step}
    return {"created_at": 1_790_000_000.0 + (step or 0), "provenance": provenance,
            "envs": {"e": {"pass@1": {"value": score}, "n_problems": 30}}}


def _envelope(observed_at=NOW, checkpoints=()):
    return run_feed.build_feed(RUN, list(checkpoints), [("base-aime", _report("verifiers-aime25-r0-n30", 0.55)),
                                                        ("base-gpqa", _report("verifiers-gpqa-r0-n198", 0.545))],
                               observed_at_ms=observed_at)


def _as_wandb_stores(summary, prefix=""):
    """The W&B server flattens nested summary objects into dotted keys; lists stay whole."""
    flat = {}
    for key, value in json.loads(json.dumps(summary)).items():
        if isinstance(value, dict):
            flat.update(_as_wandb_stores(value, f"{prefix}{key}."))
        else:
            flat[prefix + key] = value
    return flat


def test_the_envelope_is_rebuilt_from_the_flattened_summary():
    envelope = _envelope()
    flat = _as_wandb_stores({SUMMARY_KEY: envelope, "_runtime": 3})
    assert SUMMARY_KEY not in flat and f"{SUMMARY_KEY}.benchmarks" in flat
    assert stored_envelope(flat) == envelope
    assert stored_envelope({SUMMARY_KEY: envelope}) == envelope and stored_envelope({"_runtime": 3}) is None


class FakeWandb:
    """The `wandb` module: runs by id, each with its summary; `Api().run` reads it back as W&B serves it."""

    def __init__(self):
        self.runs, self.inits, self.default_entity = {}, [], "reliquary"

    def module(self):
        fake = self

        class Run:
            def __init__(self, run_id):
                self.id, self.summary, self.rows = run_id, {}, []

            def define_metric(self, *a, **kw):
                pass

            def log(self, row):
                self.rows.append(row)

            def finish(self):
                pass

        def init(**kw):
            fake.inits.append(kw)
            return fake.runs.setdefault(kw["id"], Run(kw["id"]))

        class Api:
            @property
            def default_entity(self):
                return fake.default_entity

            def run(self, path):
                run_id = path.split("/")[-1]
                if run_id not in fake.runs:
                    raise RuntimeError("Could not find run")
                return types.SimpleNamespace(summary_metrics=_as_wandb_stores(fake.runs[run_id].summary))

        return types.SimpleNamespace(init=init, Api=Api)


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake.module())
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    return fake


def test_the_envelope_goes_to_the_evaluation_runs_summary_once(fake_wandb):
    out = publish_summary(_envelope(), now_ms=NOW)
    evals = "5H47sFL6-teutonic-rl-async-v1-evals"
    assert out["written"] and out["run"] == evals and out["project"] == "reliquary-validator"
    init = fake_wandb.inits[-1]
    assert init["id"] == evals and init["resume"] == "allow" and init["job_type"] == "evaluation"
    stored = fake_wandb.runs[evals].summary[SUMMARY_KEY]
    assert stored["run_id"] == RUN                                  # the trainer's id, not the evaluation run's
    assert stored["benchmarks"][0]["baseline"]["step"] is None
    assert run_feed.parse_feed(stored, RUN, NOW)
    assert RUN not in fake_wandb.runs                               # the trainer's run is never opened
    # rebuilt later with the same results: nothing to write
    assert publish_summary(_envelope(NOW + 1000), now_ms=NOW + 1000)["written"] is False
    assert len(fake_wandb.inits) == 1
    # a new checkpoint: written again
    point = ("s5-aime", {**_report("verifiers-aime25-r0-n30", 0.6, step=5), "created_at": 1_790_000_100.0})
    point[1]["provenance"]["model"] = "ReliquaryForge/teutonic-rl-async-v1-step-00000005"
    assert publish_summary(_envelope(NOW + 2000, [point]), now_ms=NOW + 2000)["written"] is True
    assert fake_wandb.runs[evals].summary[SUMMARY_KEY]["benchmarks"][0]["checkpoints"][0]["step"] == 5


def test_the_summary_names_its_run_and_refuses_the_trainers(fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "elsewhere")
    out = publish_summary(_envelope(), run_id="teutonic-rl-async-v1-evals", entity="team", now_ms=NOW)
    assert (out["run"], out["project"], out["entity"]) == ("teutonic-rl-async-v1-evals", "elsewhere", "team")
    with pytest.raises(ValueError, match="trainer"):
        publish_summary(_envelope(), run_id=RUN, now_ms=NOW)
    bad = _envelope()
    bad["benchmarks"][0]["checkpoints"] = [{"checkpoint": "c", "step": None, "at": 0, "score": 0.5,
                                            "samples": 30, "cost_usd": None, "gpu_hours": None}]
    with pytest.raises(run_feed.FeedRejected):
        publish_summary(bad, now_ms=NOW)


def test_the_schedule_mirror_is_fail_soft(tmp_path, monkeypatch):
    def down(**kw):
        raise RuntimeError("wandb down")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=down, Api=down))
    out = mirror(_envelope(), short_of=SHORT, state_path=tmp_path / "logged.json")
    assert out["summary"] == {"written": False, "error": "RuntimeError"}
    assert out["history"]["logged"] == 0 and not (tmp_path / "logged.json").exists()


def test_eval_feed_backfills_the_baselines_into_the_evaluation_run(tmp_path, fake_wandb):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    for job, set_id, score in (("base-aime", "verifiers-aime25-r0-n30", 0.55),
                               ("base-gpqa", "verifiers-gpqa-r0-n198", 0.545)):
        (tmp_path / job).mkdir()
        (tmp_path / job / "report.json").write_text(json.dumps(_report(set_id, score)))
    argv = ["eval", "feed", "--run-id", RUN, "--baseline-job", "base-aime,base-gpqa", "--reports-dir",
            str(tmp_path), "--wandb-run-id", "teutonic-rl-async-v1-evals", "--wandb-project", "p"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["baselines"] == 2 and summary["points"] == 0 and summary["wandb"]["written"]
    stored = fake_wandb.runs["teutonic-rl-async-v1-evals"].summary[SUMMARY_KEY]
    assert [b["baseline"]["step"] for b in stored["benchmarks"]] == [None, None]
    again = CliRunner().invoke(app, argv)
    assert again.exit_code == 0 and json.loads(again.output)["wandb"]["written"] is False
    bad = CliRunner().invoke(app, argv[:-4] + ["--wandb-run-id", RUN])
    assert bad.exit_code == 1 and "trainer" in bad.output


def test_eval_schedule_writes_each_rebuilt_feed_into_the_evaluation_run(tmp_path, fake_wandb):
    from reliquary.eval import rl_schedule

    for job, set_id, score, step in (("base-aime", "verifiers-aime25-r0-n30", 0.55, 0),
                                     ("s5-aime", "verifiers-aime25-r0-n30", 0.6, 5)):
        report = _report(set_id, score, step=step)
        report["provenance"].update(rl_run_id=RUN, model=f"ReliquaryForge/m-{step}")
        (tmp_path / job).mkdir()
        (tmp_path / job / "report.json").write_text(json.dumps(report))
    settings = rl_schedule.Settings(
        run_id=RUN, run_tag="r7", suite=[rl_schedule.SuiteEntry("verifiers-aime25-r0-n30", "aime25", 8, 0.03, 32768)],
        every=5, hf_namespace="ReliquaryForge", hf_prefix="p", model_architecture="Qwen3ForCausalLM",
        eos_token_id=1, results_dir=tmp_path, state_path=tmp_path / "state.json", wandb=True)
    out = rl_schedule.live_ports(settings).feed(["s5-aime"], ["base-aime"])
    assert out["wandb"]["summary"]["written"] and out["wandb"]["history"]["logged"] == 2
    run = fake_wandb.runs["5H47sFL6-teutonic-rl-async-v1-evals"]
    bench = run.summary[SUMMARY_KEY]["benchmarks"][0]
    assert bench["baseline"]["step"] is None and [p["step"] for p in bench["checkpoints"]] == [5]
    assert [r["rl_step"] for r in run.rows] == [0, 5]


@pytest.mark.parametrize("changed", [
    {"run_id": "another-evals"}, {"project": "another-project"}, {"entity": "another-team"},
    {"trainer": "another-trainer"},
])
def test_history_marks_belong_to_the_trainer_and_effective_destination(tmp_path, fake_wandb, changed):
    state = tmp_path / "logged.json"
    feed = _feed(aime=[(500, 0.6)])
    where = {"run_id": "evals", "project": "project", "entity": "team"}
    assert log_feed(feed, short_of=SHORT, state_path=state, **where)["logged"] == 3
    if "trainer" in changed:
        feed["run_id"] = changed["trainer"]
    else:
        where.update(changed)
    assert log_feed(feed, short_of=SHORT, state_path=state, **where)["logged"] == 3
    assert log_feed(feed, short_of=SHORT, state_path=state, **where)["logged"] == 0
    saved = json.loads(state.read_text())
    identity = json.dumps({"trainer_run_id": feed["run_id"], **where}, sort_keys=True, separators=(",", ":"))
    assert len(saved["targets"]) == 2 and len(saved["targets"][identity]) == 3
    # Switching back must not replay the first destination's history either.
    assert log_feed(_feed(aime=[(500, 0.6)]), short_of=SHORT, state_path=state,
                    run_id="evals", project="project", entity="team")["logged"] == 0


def test_default_entity_is_resolved_before_reusing_history_marks(tmp_path, fake_wandb):
    state, feed = tmp_path / "logged.json", _feed()
    assert log_feed(feed, short_of=SHORT, state_path=state)["logged"] == 2
    assert fake_wandb.inits[-1]["entity"] == "reliquary"
    assert log_feed(feed, short_of=SHORT, state_path=state, entity="reliquary")["logged"] == 0
    fake_wandb.default_entity = "another-team"
    assert log_feed(feed, short_of=SHORT, state_path=state)["logged"] == 2
    assert fake_wandb.inits[-1]["entity"] == "another-team"
    assert log_feed(feed, short_of=SHORT, state_path=state)["logged"] == 0
    fake_wandb.default_entity = "reliquary"
    assert log_feed(feed, short_of=SHORT, state_path=state)["logged"] == 0


def test_unresolved_default_entity_does_not_change_saved_history(tmp_path, fake_wandb):
    state, feed = tmp_path / "logged.json", _feed()
    assert log_feed(feed, short_of=SHORT, state_path=state)["logged"] == 2
    saved = state.read_bytes()
    fake_wandb.default_entity = None
    out = log_feed(feed, short_of=SHORT, state_path=state)
    assert out["logged"] == 0 and out["error"] == "ValueError"
    assert state.read_bytes() == saved
