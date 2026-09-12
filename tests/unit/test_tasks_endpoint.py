"""Miners discover the other tasks without anyone restarting this one."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from reliquary.constants import PROTOCOL_PROFILE_ID, TASK_ID
from reliquary.validator.server import ValidatorServer, _load_declared_tasks


def test_a_validator_always_lists_its_own_task():
    body = TestClient(ValidatorServer().app).get("/tasks").json()

    assert body["tasks"][0]["task_id"] == TASK_ID
    assert body["tasks"][0]["profile_id"] == PROTOCOL_PROFILE_ID
    # ValidatorServer's own default: the legacy single-task pool, now sourced
    # from the registry (via ValidationService) rather than an env var.
    assert body["tasks"][0]["emission_share"] == 1.0
    assert body["tasks"][0]["url"] is None


def test_declared_peers_are_listed(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "logic-probe", "url": "http://10.0.0.9:8080"},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "logic-probe"]
    assert tasks[1]["url"] == "http://10.0.0.9:8080"


def test_a_peer_added_after_start_shows_up_without_a_restart(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text("[]")
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))
    client = TestClient(ValidatorServer().app)

    assert len(client.get("/tasks").json()["tasks"]) == 1

    directory.write_text(json.dumps([{"task_id": "logic-probe", "url": "http://10.0.0.9:8080"}]))

    assert len(client.get("/tasks").json()["tasks"]) == 2


def test_a_broken_directory_never_breaks_discovery(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text("{ not json")
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    response = TestClient(ValidatorServer().app).get("/tasks")

    assert response.status_code == 200
    assert [t["task_id"] for t in response.json()["tasks"]] == [TASK_ID]


def test_a_malformed_entry_is_skipped(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "ok", "url": "http://10.0.0.9:8080"},
        {"task_id": "missing-url"},
        {"url": "http://10.0.0.10:8080"},
        "nonsense",
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "ok"]


def test_a_nonsense_emission_share_is_listed_as_none(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "logic-probe", "url": "http://10.0.0.9:8080", "emission_share": "high"},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "logic-probe"]
    assert tasks[1]["emission_share"] is None


def test_a_string_model_is_listed_as_none(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "logic-probe", "url": "http://10.0.0.9:8080", "model": "gpt-4"},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "logic-probe"]
    assert tasks[1]["model"] is None


def test_an_out_of_range_emission_share_is_listed_as_none(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "logic-probe", "url": "http://10.0.0.9:8080", "emission_share": 1.5},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert tasks[1]["emission_share"] is None


def test_a_peer_reusing_our_own_task_id_is_not_listed(tmp_path, monkeypatch):
    """Two entries with the same id and different URLs would make a miner
    selecting by id pick arbitrarily."""
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": TASK_ID, "url": "http://10.0.0.9:8080"},
        {"task_id": "logic-probe", "url": "http://10.0.0.10:8080"},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "logic-probe"]
    assert tasks[0]["url"] is None


def test_load_declared_tasks_skips_our_own_task_id(tmp_path):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": TASK_ID, "url": "http://10.0.0.9:8080"},
    ]))

    assert _load_declared_tasks(str(directory)) == []


def test_load_declared_tasks_returns_empty_for_a_missing_path(tmp_path):
    assert _load_declared_tasks(str(tmp_path / "does-not-exist.json")) == []


def test_load_declared_tasks_returns_empty_for_invalid_json(tmp_path):
    directory = tmp_path / "tasks.json"
    directory.write_text("{ not json")

    assert _load_declared_tasks(str(directory)) == []
