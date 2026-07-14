import pytest

from reliquary.environment.grader_client import (
    GraderClient,
    GraderUnavailableError,
)


def test_strict_grader_returns_authoritative_score(monkeypatch):
    client = GraderClient("/unused")
    monkeypatch.setattr(
        client,
        "_round_trip",
        lambda request: {"status": "ok", "passed": 1, "total": 2},
    )
    assert client.evaluate_cases_strict("code", [{}, {}], 5.0) == 0.5


def test_strict_grader_raises_instead_of_publishing_false_zero(monkeypatch):
    client = GraderClient("/unused")

    def unavailable(request):
        raise ConnectionError("offline")

    monkeypatch.setattr(client, "_round_trip", unavailable)
    monkeypatch.setattr(
        "reliquary.environment.grader_client.time.sleep", lambda _: None
    )
    with pytest.raises(GraderUnavailableError, match="unreachable"):
        client.evaluate_cases_strict("code", [{}], 5.0)


def test_strict_grader_rejects_malformed_counts(monkeypatch):
    client = GraderClient("/unused")
    monkeypatch.setattr(
        client,
        "_round_trip",
        lambda request: {"status": "ok", "passed": 1, "total": 99},
    )
    with pytest.raises(GraderUnavailableError, match="inconsistent"):
        client.evaluate_cases_strict("code", [{}], 5.0)
