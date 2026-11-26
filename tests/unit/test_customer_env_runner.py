import os
import sys
from pathlib import Path

import pytest

from reliquary.customer_env.runner import (
    RewardClient,
    RewardInfrastructureError,
    ScoreResult,
    WorkerProcess,
    sealed_argv,
)

FAKE = str(Path(__file__).with_name("customer_env_fake_worker.py"))
ROOT = str(Path(__file__).resolve().parents[2])


def client(env=None, per_item=5.0):
    argv = ["env", f"PYTHONPATH={ROOT}", *(f"{k}={v}" for k, v in (env or {}).items()), sys.executable, FAKE]
    return RewardClient(WorkerProcess(argv, start_timeout_s=30), per_item_timeout_s=per_item)


def item(key, answer, prompt=None):
    from tests.unit.customer_env_fake_worker import digest

    return {"key": key, "prompt_sha256": digest(None, prompt or f"{key[1:]}+{key[1:]}="), "answer": answer}


@pytest.fixture
def rc():
    c = client()
    yield c
    c.close()


def test_freeze_takes_the_tasks_and_says_why_one_is_refused(rc):
    rows, refused = rc.freeze_all(page=2)
    assert [r["key"] for r in rows] == ["t0", "t1", "t2", "t3", "t4"]
    assert refused == [{"key": "tools", "reason": "it gives the model tools"}]
    assert rc.provenance["package"] == "fake-env"


def test_score_answers_one_result_per_item(rc):
    results = rc.score([item("t1", "2"), item("t1", "3"), item("t2", "RAISE"), item("t2", "4", prompt="other"),
                        item("t3", "PRINT")])
    assert results[:2] == [ScoreResult(1.0), ScoreResult(0.0)]
    assert results[2].score is None and results[2].error.startswith("grader_error: ZeroDivisionError")
    assert results[3] == ScoreResult(None, "source_drift")
    # The customer's print() goes to stderr: the protocol is intact.
    assert results[4] == ScoreResult(0.0)


@pytest.mark.parametrize("answer, reason", [("HANG", "worker_hung"), ("CRASH", "worker_died")])
def test_a_hung_dead_or_garbling_worker_is_infrastructure_then_restarts(answer, reason):
    rc = client(per_item=1.0)
    try:
        with pytest.raises(RewardInfrastructureError, match=reason):
            rc.score([item("t1", answer)])
        assert rc.score([item("t1", "2")]) == [ScoreResult(1.0)]
    finally:
        rc.close()


def test_the_customer_code_cannot_write_into_the_protocol(rc):
    # It writes straight to fd 1; the protocol has its own descriptor.
    assert rc.score([item("t1", "GARBLE"), item("t1", "2")]) == [ScoreResult(0.0), ScoreResult(1.0)]


def test_a_worker_answering_garbage_is_infrastructure():
    rc = RewardClient(WorkerProcess([sys.executable, "-c", "import sys; sys.stdin.readline(); print('nope')"]))
    with pytest.raises(RewardInfrastructureError, match="worker_garbled"):
        rc.score([item("t1", "2")])


def test_a_taskset_that_cannot_load_is_a_start_failure_with_its_reason():
    rc = client({"FAKE_LOAD_FAIL": "1"})
    with pytest.raises(RewardInfrastructureError, match="start_failed: load_failed: ImportError"):
        rc.score([item("t1", "2")])


def test_a_worker_that_does_not_exist_is_a_start_failure():
    rc = RewardClient(WorkerProcess(["/nonexistent/worker"]))
    with pytest.raises(RewardInfrastructureError, match="start_failed"):
        rc.score([item("t1", "2")])


def test_the_sealed_argv_isolates_the_container():
    image = "registry.reliqua.ai/customer-envs/acct/my-env@sha256:" + "a" * 64
    argv = sealed_argv(image)
    joined = " ".join(argv)
    for flag in ("--runtime runsc", "--network none", "--read-only", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--user 65534:65534", "--pull never"):
        assert flag in joined
    assert argv[-1] == image
    assert not any(a in ("-v", "--volume", "--mount", "-e", "--env", "--env-file", "--privileged") for a in argv)
    for bad in ("registry/x:latest", "registry/x", "x@sha256:short"):
        with pytest.raises(ValueError):
            sealed_argv(bad)


@pytest.mark.skipif(os.environ.get("RELIQUARY_DOCKER_TESTS") != "1", reason="needs docker")
def test_docker_argv_runs():  # pragma: no cover - exercised on a reward host
    assert sealed_argv("x@sha256:" + "a" * 64, runtime="runc")
