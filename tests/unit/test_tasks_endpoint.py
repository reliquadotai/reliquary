"""Miners discover the other tasks without anyone restarting this one."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from reliquary.constants import PROTOCOL_PROFILE_ID, TASK_EMISSION_SHARE, TASK_ID
from reliquary.validator.server import ValidatorServer


def test_a_validator_always_lists_its_own_task():
    body = TestClient(ValidatorServer().app).get("/tasks").json()

    assert body["tasks"][0]["task_id"] == TASK_ID
    assert body["tasks"][0]["profile_id"] == PROTOCOL_PROFILE_ID
    assert body["tasks"][0]["emission_share"] == TASK_EMISSION_SHARE
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
