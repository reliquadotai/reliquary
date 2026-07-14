import json

from reliquary.eval_dashboard.config import canonical_json_bytes
from reliquary.eval_dashboard.holdout import write_canonical_jsonl
from reliquary.eval_dashboard.models import MathTask
from scripts.audit_eval_holdout import main


REV = "a" * 40


def _run(tmp_path, training_prompt):
    holdout = tmp_path / "holdout.jsonl"
    write_canonical_jsonl(
        holdout,
        [MathTask(task_id="m1", prompt="Compute 1 + 1.", ground_truth="2")],
    )
    training = tmp_path / "training.jsonl"
    training.write_bytes(canonical_json_bytes({"problem": training_prompt}) + b"\n")
    output = tmp_path / "review.json"
    code = main(
        [
            "--domain",
            "math",
            "--holdout",
            str(holdout),
            "--training-source",
            "nvidia/OpenMathInstruct-2",
            REV,
            "problem",
            str(training),
            "--reviewer",
            "test-reviewer",
            "--reviewed-at",
            "2026-07-14T00:00:00Z",
            "--output",
            str(output),
        ]
    )
    return code, json.loads(output.read_bytes())


def test_audit_rejects_exact_overlap(tmp_path):
    code, review = _run(tmp_path, "  compute 1 + 1. ")
    assert code == 2
    assert review["decision"] == "rejected"
    assert review["exact_overlap_count"] == 1


def test_audit_approves_distinct_prompt(tmp_path):
    code, review = _run(tmp_path, "Find the derivative of x cubed.")
    assert code == 0
    assert review["decision"] == "approved"
    source = review["training_sources"][0]
    assert source["revision"] == REV
    assert source["prompt_field"] == "problem"
    assert source["artifacts"][0]["n_prompts"] == 1
