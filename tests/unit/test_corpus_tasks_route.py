"""`GET /corpus/tasks`: every declared task's emission share, public, cached."""

from __future__ import annotations

from types import SimpleNamespace

from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: F401



def _entry(task_id, mechanism, cap, *, status="active", job_id=None, retired_at=None):
    return SimpleNamespace(task_id=task_id, mechanism=mechanism, params={"cap": cap},
                           status=status, job_id=job_id, retired_at=retired_at)


def _registry():
    entries = {
        "default": _entry("default", "rl", 0.8),
        "corpus-math": _entry("corpus-math", "corpus_generation", 0.1, job_id="swe-v1"),
        "old": _entry("old", "corpus_generation", 0.05, status="retired",
                      job_id="old-job", retired_at=123),
    }
    return entries


def _client(seeded_job, reads):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("swe-v1",))

    async def read():
        reads.append(1)
        return _registry(), "etag"

    client.app.state.task_registry_reader = read
    return client


def test_tasks_lists_every_declared_task_with_its_share(seeded_job):  # noqa: F811
    client = _client(seeded_job, [])
    body = client.get("/corpus/tasks")
    assert body.status_code == 200, body.text
    tasks = {t["task_id"]: t for t in body.json()["tasks"]}
    assert tasks["default"] == {"task_id": "default", "mechanism": "rl", "cap": 0.8,
                                "status": "active", "job_id": None, "retired_at": None}
    assert tasks["corpus-math"]["job_id"] == "swe-v1"
    assert tasks["old"]["status"] == "retired"
    # Only active tasks count toward what the subnet may pay.
    assert body.json()["active_cap_total"] == 0.9
    assert "as_of" in body.json()


def test_tasks_reads_the_registry_at_most_once_per_minute(seeded_job):  # noqa: F811
    reads = []
    client = _client(seeded_job, reads)
    client.get("/corpus/tasks")
    client.get("/corpus/tasks")
    assert len(reads) == 1


def test_tasks_answers_503_when_the_registry_cannot_be_read(seeded_job):  # noqa: F811
    from tests.unit.test_corpus_route_skip import _app

    client = _app(seeded_job, ("swe-v1",))

    async def broken():
        raise OSError("r2 down")

    client.app.state.task_registry_reader = broken
    assert client.get("/corpus/tasks").status_code == 503
