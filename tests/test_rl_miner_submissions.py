"""The optional public projection reads the native journal without changing it."""
import asyncio
import hashlib
import json
import sqlite3
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.validator import rl_miner_submissions as export
from reliquary.validator.rl_async_service import RlAsyncSettings, build_rl_async, mount_rl_async
from reliquary.validator.rl_submit import _SCHEMA

RUN = "journal-fixture"
# SS58/42 encodings of repeated 0x11/0x22 public bytes, not wallet credentials.
HOTKEY_A = "5CT5jwBEAhveEjgiSCQbkaKcKcUyF3VJ8qNXM9rXsuQyn3Kd"
HOTKEY_B = "5CqTdCcXCC7kv1Nwq2sCeJUNVqxbBPXjMAUXrbhRJtrUiyPP"


def add_group(db, lease, hotkey, issued, arrived, *, rewards="[0.0, 1.0]"):
    db.execute(
        "INSERT INTO rl_groups(lease_id,hotkey,environment,prompt_idx,k,nonce,issued_at,arrived_at,"
        "verdict,reason,rewards,completion_tokens,first_scan,audit,training) "
        "VALUES(?,?,?,0,17,?,?,?,'in_zone',?,?,32,1,'passed','accepted')",
        (lease, hotkey, "reliquary_logic_v2", "private-nonce", issued, arrived,
         "private-reason", rewards),
    )
    db.commit()


@pytest.fixture
def journal(tmp_path):
    path = tmp_path / "rl-async.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(_SCHEMA)
        now = time.time()
        add_group(db, "lease-a", HOTKEY_A, now - 30, now - 20)
        add_group(db, "lease-b", HOTKEY_B, now - 25, now - 15)
    return path, now


def test_two_hotkeys_have_distinct_public_rows_and_no_private_payload(journal):
    path, now = journal
    with sqlite3.connect(path) as db:
        add_group(db, "older-lease", HOTKEY_A, now - 90, now - 10)
        add_group(db, "later-arrival", HOTKEY_B, now - 5, now + 10)
    before = path.read_bytes()
    snapshot = export.read_snapshot(path, run_id=RUN, not_before=now - 60, now=now)
    expected = {
        hotkey: hashlib.sha256(("rl-submission/v1\0" + RUN + "\0" + lease).encode()).hexdigest()
        for hotkey, lease in [(HOTKEY_A, "lease-a"), (HOTKEY_B, "lease-b")]
    }
    assert {row["hotkey"]: row["id"] for row in snapshot["rows"]} == expected
    assert all(row["checkpoint"] == 17 and row["completionTokens"] == 32
               and row["rewardMean"] == 0.5 for row in snapshot["rows"])
    body = json.dumps(snapshot)
    assert not any(secret in body for secret in ["lease-a", "lease-b", "private-nonce", "private-reason"])
    assert path.read_bytes() == before


def test_oversized_value_is_refused_by_sqlite_before_python_materialization(journal):
    path, now = journal
    with sqlite3.connect(path) as db:
        add_group(db, "oversized", HOTKEY_A, now - 10, now - 5,
                  rewards=json.dumps([0.0] * 65536))
    with pytest.raises(sqlite3.DataError):
        export.read_snapshot(path, run_id=RUN, not_before=now - 60, now=now)


def test_http_requires_exact_run_selection_and_returns_both_rows(journal):
    path, now = journal
    app = FastAPI()
    app.include_router(export.build_router(path, run_id=RUN, not_before=now - 60))
    with TestClient(app) as client:
        response = client.get("/rl/miner-submissions", params={"run_id": RUN})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        assert {row["hotkey"] for row in response.json()["rows"]} == {HOTKEY_A, HOTKEY_B}
        for query in ["", "?run_id=other", f"?run_id={RUN}&run_id={RUN}", f"?run_id={RUN}&hotkey={HOTKEY_A}"]:
            assert client.get("/rl/miner-submissions" + query).status_code == 400


def test_failed_reads_are_coalesced_without_returning_old_success(journal, monkeypatch):
    path, now = journal
    calls = []

    def read(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise sqlite3.OperationalError("private database failure")
        return {"rows": []}

    monkeypatch.setattr(export, "read_snapshot", read)

    async def exercise():
        cached = export.CachedExporter(path, run_id=RUN, not_before=now - 60)
        assert await cached.get() == {"rows": []}
        cached.last_attempt -= 61
        assert await cached.get() is None
        assert await asyncio.gather(*(cached.get() for _ in range(12))) == [None] * 12
        assert len(calls) == 2

    asyncio.run(exercise())


class Server:
    def __init__(self):
        self.app = FastAPI()

        @self.app.get("/existing-route")
        def existing():
            return {"ok": True}

    def mount_rl_lease_router(self, router):
        self.app.include_router(router)


@pytest.mark.parametrize("cutoff", [None, "not-a-time", "nan", "inf", "-1", "999999999999"])
def test_invalid_optional_configuration_does_not_stop_native_mount(tmp_path, monkeypatch, cutoff):
    monkeypatch.setenv(export.MODE_ENV, "enabled")
    monkeypatch.delenv(export.START_ENV, raising=False)
    if cutoff is not None:
        monkeypatch.setenv(export.START_ENV, cutoff)
    rl = build_rl_async(
        RlAsyncSettings(task_id="fixture", cap=0.5, run=RUN,
                       targets={"reliquary_logic_v2": 1.0}, planned_tokens_per_period=1e6,
                       lease_expiry_s={"reliquary_logic_v2": 600.0}),
        path=tmp_path / "rl-async.sqlite3", grader=None,
    )
    try:
        server = Server()
        mount_rl_async(server, rl, validator_hotkey=HOTKEY_A)
        with TestClient(server.app) as client:
            assert client.get("/existing-route").status_code == 200
            assert client.get("/rl/miner-submissions", params={"run_id": RUN}).status_code == 404
            assert client.get("/rl/trainer/config").status_code == 404
            assert client.get("/rl/checkpoint").status_code == 503
    finally:
        rl.close()


def test_controller_reopen_preserves_held_training_and_open_lease(tmp_path):
    settings = RlAsyncSettings(task_id="fixture", cap=0.5, run=RUN,
                              targets={"reliquary_logic_v2": 1.0}, planned_tokens_per_period=1e6,
                              lease_expiry_s={"reliquary_logic_v2": 600.0})
    path = tmp_path / "rl-async.sqlite3"
    rl = build_rl_async(settings, path=path, grader=None)
    now = time.time()
    add_group(rl.book.db, "lease-a", HOTKEY_A, now - 30, now - 20)
    rl.book.db.execute(
        "INSERT INTO rl_leases(lease_id,hotkey,environment,prompt_idx,k,nonce,issued_at,expires_at,state) "
        "VALUES(?,?,?,1,17,?,?,?,'open')",
        ("lease-open", HOTKEY_B, "reliquary_logic_v2", "fixture", now, now + 600),
    )
    rl.book.db.execute(
        "INSERT INTO rl_buffer(lease_id,environment,k,arrived_at,hotkey,state,body,attempts) "
        "VALUES(?,?,17,?,?,'taken',?,1)",
        ("lease-a", "reliquary_logic_v2", now - 20, HOTKEY_A, b"held-fixture"),
    )
    rl.book.db.commit()
    before = {table: rl.book.db.execute(f"SELECT * FROM {table}").fetchall()
              for table in ["rl_groups", "rl_leases", "rl_buffer"]}
    rl.close()
    reopened = build_rl_async(settings, path=path, grader=None)
    try:
        assert {table: reopened.book.db.execute(f"SELECT * FROM {table}").fetchall()
                for table in before} == before
    finally:
        reopened.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_real_native_mount_is_opt_in_and_keeps_the_journal(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv(export.MODE_ENV, "enabled" if enabled else "disabled")
    monkeypatch.setenv(export.START_ENV, str(time.time() - 60))
    path = tmp_path / "rl-async.sqlite3"
    rl = build_rl_async(
        RlAsyncSettings(task_id="fixture", cap=0.5, run=RUN,
                       targets={"reliquary_logic_v2": 1.0}, planned_tokens_per_period=1e6,
                       lease_expiry_s={"reliquary_logic_v2": 600.0}), path=path, grader=None,
    )
    try:
        now = time.time()
        add_group(rl.book.db, "lease-a", HOTKEY_A, now - 30, now - 20)
        add_group(rl.book.db, "lease-b", HOTKEY_B, now - 25, now - 15)
        rows = rl.book.db.execute("SELECT * FROM rl_groups ORDER BY seq").fetchall()
        server = Server()
        mount_rl_async(server, rl, validator_hotkey=HOTKEY_A)
        with TestClient(server.app) as client:
            response = client.get("/rl/miner-submissions", params={"run_id": RUN})
            assert response.status_code == (200 if enabled else 404)
            if enabled:
                assert {row["hotkey"] for row in response.json()["rows"]} == {HOTKEY_A, HOTKEY_B}
            assert client.get("/existing-route").status_code == 200
        assert rl.book.db.execute("SELECT * FROM rl_groups ORDER BY seq").fetchall() == rows
    finally:
        rl.close()
