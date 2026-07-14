from datetime import datetime, timezone

from reliquary.eval_dashboard.metrics import build_dashboard, summarize_domain
from reliquary.eval_dashboard.models import CheckpointResult, SampleResult, TaskResult


REV = "c" * 40
HASH = "d" * 64


def _sample(index, reward, *, terminated=True, forced=False):
    return SampleResult(
        sample_index=index,
        seed=index,
        reward=reward,
        completion_length=100 + index,
        terminated=terminated,
        forced=forced,
        completion_sha256=f"{index + 1:064x}",
        duration_seconds=0.1,
    )


def _tasks():
    return [
        TaskResult(
            task_id="one",
            prompt_sha256="1" * 64,
            samples=[_sample(0, 1.0), _sample(1, 0.0, terminated=False, forced=True)],
        ),
        TaskResult(
            task_id="two",
            prompt_sha256="2" * 64,
            samples=[_sample(0, 0.0), _sample(1, 0.0)],
        ),
    ]


def _checkpoint(n, completed_at):
    math = summarize_domain("math", _tasks())
    code = summarize_domain("code", _tasks())
    return CheckpointResult(
        config_hash=HASH,
        config_sha256=HASH,
        lineage_id="qwen35-2b-v2",
        model_repo_id=("Qwen/Qwen3.5-2B" if n == 0 else "ReliquaryForge/model-v2"),
        model_revision=REV,
        checkpoint_n=n,
        observed_window=100 + n,
        started_at=completed_at - 5,
        completed_at=completed_at,
        completed_at_iso=datetime.fromtimestamp(completed_at, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        duration_seconds=5,
        hardware={},
        runtime={},
        domains={"math": math, "code": code},
    )


def test_domain_metrics_keep_first_best_and_average_distinct():
    result = summarize_domain("math", _tasks())
    assert result.pass_at_1 == 0.5
    assert result.pass_at_k == 0.5
    assert result.pass_avg == 0.25
    assert result.trunc_pct == 25.0
    assert result.forced_pct == 25.0
    assert result.pass1_ci95[0] < result.pass_at_1 < result.pass1_ci95[1]


def test_dashboard_matches_web_contract_and_tracks_provenance():
    dashboard = build_dashboard(
        [_checkpoint(0, 1000), _checkpoint(2, 2000)],
        generated_at=2000,
        generated_at_iso="1970-01-01T00:33:20Z",
        evidence_completed_at=2000,
        evidence_completed_at_iso="1970-01-01T00:33:20Z",
        config_hash=HASH,
        config_sha256=HASH,
        lineage_id="qwen35-2b-v2",
        checkpoint_repo_id="ReliquaryForge/model-v2",
        base_checkpoint_n=0,
        latest_checkpoint_n=2,
        latest_checkpoint_revision=REV,
        latest_model_repo_id="ReliquaryForge/model-v2",
        publish_interval_windows=10,
    )
    assert dashboard["generated_at"] == 2000
    assert dashboard["generated_at_iso"] == "1970-01-01T00:33:20Z"
    assert dashboard["evidence_completed_at"] == 2000
    assert dashboard["latest_checkpoint_revision"] == REV
    assert dashboard["latest_model_repo_id"] == "ReliquaryForge/model-v2"
    assert dashboard["n_checkpoints"] == 3
    assert dashboard["n_evaluated"] == 2
    assert dashboard["eval"][1]["step"] == 20
    assert dashboard["eval_summary"]["coverage"] == 2 / 3
    assert dashboard["training"] == []
