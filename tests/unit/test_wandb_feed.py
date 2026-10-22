from __future__ import annotations

import json

from reliquary.eval.wandb_feed import eval_run_ids, log_feed, new_rows

RUN = "5H47sFL6-teutonic-rl-async-v1-trainer"
SHORT = {"verifiers-aime25-r0-n30": "aime25", "verifiers-gpqa-r0-n198": "gpqa"}


def _point(step, score):
    return {"step": step, "score": score, "checkpoint": f"step-{step}", "at": 0, "samples": 30}


def _feed(aime=(), gpqa=()):
    return {"run_id": RUN, "benchmarks": [
        {"id": "verifiers-aime25-r0-n30", "baseline": _point(0, 0.55), "checkpoints": [_point(*p) for p in aime]},
        {"id": "verifiers-gpqa-r0-n198", "baseline": _point(0, 0.545), "checkpoints": [_point(*p) for p in gpqa]},
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

    assert log_feed(_feed(aime=[(500, 0.6)]), short_of=SHORT, state_path=state, init=init)["logged"] == 3
    assert log_feed(_feed(aime=[(500, 0.6)]), short_of=SHORT, state_path=state, init=init)["logged"] == 0

    def down(**kw):
        raise RuntimeError("wandb down")

    out = log_feed(_feed(aime=[(500, 0.6), (1000, 0.62)]), short_of=SHORT, state_path=state, init=down)
    assert out["logged"] == 0 and out["error"] == "RuntimeError"
    out = log_feed(_feed(aime=[(500, 0.6), (1000, 0.62)]), short_of=SHORT, state_path=state, init=init)
    assert out == {"run": "5H47sFL6-teutonic-rl-async-v1-evals", "logged": 1}
    assert sink[-1] == {"rl_step": 1000, "eval/aime25/pass@1": 0.62, "eval/aime25/delta": 0.62 - 0.55}
    assert "aime25@1000" in json.loads(state.read_text())
