from reliquary.eval_dashboard.metrics import summarize_domain
from reliquary.eval_dashboard.models import SampleResult, TaskResult
from reliquary.eval_dashboard.worker import compare_replay_summaries


def _summary():
    samples = [
        SampleResult(
            sample_index=0,
            seed=1,
            reward=1.0,
            completion_length=10,
            terminated=True,
            forced=False,
            completion_sha256="a" * 64,
            duration_seconds=1.0,
        ),
        SampleResult(
            sample_index=1,
            seed=2,
            reward=0.0,
            completion_length=20,
            terminated=False,
            forced=True,
            completion_sha256="b" * 64,
            duration_seconds=1.0,
        ),
    ]
    task = TaskResult(task_id="task", prompt_sha256="c" * 64, samples=samples)
    return summarize_domain("math", [task])


def test_replay_comparison_checks_all_published_metrics():
    published = _summary()
    within = published.model_copy(
        update={
            "pass_at_1": published.pass_at_1 - 0.01,
            "pass_at_k": published.pass_at_k - 0.01,
            "pass_avg": published.pass_avg - 0.01,
            "trunc_pct": published.trunc_pct + 1.0,
            "forced_pct": published.forced_pct + 1.0,
        }
    )
    report = compare_replay_summaries(published, within, tolerance=0.02)
    assert report["passed"] is True
    assert set(report["score_deltas"]) == {"pass_at_1", "pass_at_k", "pass_avg"}
    assert set(report["percentage_point_deltas"]) == {"trunc_pct", "forced_pct"}

    outside = within.model_copy(update={"pass_at_k": published.pass_at_k - 0.03})
    assert (
        compare_replay_summaries(published, outside, tolerance=0.02)["passed"] is False
    )
