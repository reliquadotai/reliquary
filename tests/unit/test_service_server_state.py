"""Server side of the v2 service task: /state announcement, no eligibility mask, admin-only observations."""
import json
import re
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reliquary.protocol.submission import GrpoBatchState, MinerState, WindowState
from reliquary.services.runtime import ServiceRuntime, protocol_slot_geometry
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer, register_service_observations
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, qualification_v2
from tests.unit.test_validator_server import _batcher

BEACON = "ab" * 32
TOKEN_ENV = "RELIQUARY_SERVICE_OBSERVATIONS_TOKEN"
LEGACY_STATE = (
    '{"state":"open","window_n":500,"anchor_block":500,"cooldown_prompts":[42],"valid_submissions":0,'
    '"checkpoint_n":0,"checkpoint_repo_id":null,"checkpoint_revision":null,'
    '"randomness":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"}'
)


def runtime_for(tmp_path, window=500):
    contract = contract_v2()
    rt = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract), now=time.time(),
                        drand_round_at=lambda instant: int(instant) // 3)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision="d" * 40)
    picks, slots = protocol_slot_geometry()
    rt.open_window(window, pools={MATH: 0.25, CODE: 0.25}, picks_target=picks, batch_slots=slots, now=0.0)
    return rt


def service_server(rt, *, announce=True, window=500, cooldown=None):
    batcher = _batcher(window_start=window, cooldown_map=cooldown)
    batcher.service_runtime = rt
    if announce:
        batcher.service_policy = rt.announcement(window=window, randomness=BEACON, environment=MATH)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    return server, batcher


# ---- /state ----------------------------------------------------------------------------------

def test_state_of_an_opened_window_carries_the_v2_announcement(tmp_path):
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)
    body = TestClient(server.app).get("/state").json()
    state = GrpoBatchState(**body)
    policy = body["service_policy"]
    assert state.service_policy.pool_epoch == 500 and policy["pool_randomness"] == BEACON
    assert policy["contract"]["schema"] == "service-contract/v2"
    assert policy["checkpoint"]["revision"] == "d" * 40
    assert policy["schedule"]["order_sha256"] == rt.contract.sha256
    assert set(body).isdisjoint({"run_salt", "run_meta", "eligible", "eligibility"})


def test_state_is_not_served_for_a_window_that_was_not_announced(tmp_path):
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt, announce=False)
    client = TestClient(server.app)
    assert client.get("/state").status_code == 503
    assert client.get("/miner-state").status_code == 503


def test_state_is_not_served_with_an_announcement_of_another_window(tmp_path):
    rt = runtime_for(tmp_path)
    server, batcher = service_server(rt)
    batcher.service_policy = dict(batcher.service_policy, pool_epoch=499)
    assert TestClient(server.app).get("/state").status_code == 503


def test_state_makes_no_runtime_write(tmp_path):
    """announcement() is a write; the request path only reads what the batcher holds."""
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)

    def boom(*args, **kwargs):
        raise AssertionError("a request handler must not touch the runtime's announcement")
    rt.announcement = boom
    client = TestClient(server.app)
    assert client.get("/state").status_code == 200
    assert client.get("/miner-state").status_code == 200


def test_state_has_no_eligibility_mask_the_cooldown_is_the_batchers_alone(tmp_path):
    rt = runtime_for(tmp_path)
    cooldown = CooldownMap(cooldown_windows=50)
    cooldown.record_batched(prompt_idx=42, window=490)
    server, _ = service_server(rt, cooldown=cooldown)
    client = TestClient(server.app)
    assert client.get("/state").json()["cooldown_prompts"] == [42]
    state = MinerState.model_validate(client.get("/miner-state").json())
    (env,) = state.environments.values()
    assert env.cooldown_prompts() == {42}


def test_miner_state_closes_everything_only_when_the_order_is_exhausted(tmp_path):
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)
    server.set_service_runtime_active(False)
    state = MinerState.model_validate(TestClient(server.app).get("/miner-state").json())
    (env,) = state.environments.values()
    assert len(env.cooldown_prompts()) > 1


def test_state_key_tracks_contract_epoch_beacon_schedule_checkpoint_and_activity(tmp_path):
    rt = runtime_for(tmp_path)
    _, batcher = service_server(rt)
    server = ValidatorServer()
    key = server._service_state_key(batcher)
    assert key[0] == rt.contract.sha256 and key[1] == 500 and key[2] == BEACON and key[5] is True
    assert key[4] == "d" * 40
    batcher.service_policy = dict(batcher.service_policy, pool_randomness="cd" * 32)
    assert server._service_state_key(batcher) != key
    server.set_service_runtime_active(False)
    assert server._service_state_key(batcher)[5] is False
    assert server._service_state_key(SimpleNamespace(service_runtime=None)) is None


def test_legacy_state_bytes_are_byte_for_byte_the_golden_snapshot():
    cooldown = CooldownMap(cooldown_windows=50)
    cooldown.record_batched(prompt_idx=42, window=490)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500, cooldown_map=cooldown))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    assert client.get("/state").content.decode() == LEGACY_STATE
    assert client.get("/state").content.decode() == LEGACY_STATE  # cached bytes identical
    assert server._service_state_key(server.active_batcher) is None


# ---- /health never names the candidate window -------------------------------------------------

def test_service_health_does_not_publish_the_failed_candidate_window_or_stage():
    server = ValidatorServer()
    server.service_health_redaction = True
    server.set_window_preparation_state(last_committed_window_n=9, candidate_window_n=10, stage="prompt_manifest")
    server.record_window_preparation_failure(
        {"candidate_window_n": 10, "stage": "prompt_manifest", "error_type": "RuntimeError", "ts": 1.0})
    text = TestClient(server.app).get("/health").text
    body = json.loads(text)
    assert body["candidate_window_n"] is None and body["window_preparation_stage"] is None
    assert body["last_window_preparation_failure"] is None
    assert "prompt_manifest" not in text
    assert body["window_preparation_failures_total"] == 1


def test_legacy_health_still_reports_the_candidate_window():
    server = ValidatorServer()
    server.set_window_preparation_state(last_committed_window_n=9, candidate_window_n=10, stage="prompt_manifest")
    body = TestClient(server.app).get("/health").json()
    assert body["candidate_window_n"] == 10 and body["window_preparation_stage"] == "prompt_manifest"


# ---- /service-observations (admin only) ------------------------------------------------------

def events_runtime(count=5):
    events = [(seq, {"id": f"x{seq}", "hotkey": f"hk{seq}", "kind": "observation"}) for seq in range(1, count + 1)]
    calls = []

    def admin_events(after, limit):
        calls.append((after, limit))
        return [e for e in events if e[0] > after][:limit]
    return SimpleNamespace(admin_events=admin_events, calls=calls)


def observations_client(runtime):
    app = FastAPI()
    register_service_observations(app, lambda: SimpleNamespace(service_runtime=runtime) if runtime else None)
    return TestClient(app)


@pytest.mark.parametrize("configured, header, status", [
    (None, None, 404), (None, "Bearer secret", 404),
    ("secret", None, 401), ("secret", "Bearer wrong", 401), ("secret", "secret", 401),
    ("secret", "Bearer secret", 200)])
def test_observations_route_is_admin_only(monkeypatch, configured, header, status):
    if configured is None:
        monkeypatch.delenv(TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(TOKEN_ENV, configured)
    response = observations_client(events_runtime()).get(
        "/service-observations", headers={"Authorization": header} if header else {})
    assert response.status_code == status
    if status == 200:
        assert response.json()["events"][0]["hotkey"] == "hk1"
    else:
        assert "hk1" not in response.text


def test_auth_failure_does_not_reveal_whether_a_run_exists(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "secret")
    with_run = observations_client(events_runtime()).get("/service-observations")
    without_run = observations_client(None).get("/service-observations")
    assert (with_run.status_code, with_run.json()) == (without_run.status_code, without_run.json()) == (
        401, {"detail": "unauthorized"})
    assert observations_client(None).get(
        "/service-observations", headers={"Authorization": "Bearer secret"}).status_code == 404


def test_token_comparison_is_constant_time(monkeypatch):
    import hmac
    monkeypatch.setenv(TOKEN_ENV, "secret")
    seen = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    observations_client(events_runtime()).get("/service-observations", headers={"Authorization": "Bearer x"})
    assert seen == [(b"Bearer x", b"Bearer secret")]


def test_observations_are_paginated_by_sequence_with_a_bounded_page(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "secret")
    runtime = events_runtime(5)
    client = observations_client(runtime)
    auth = {"Authorization": "Bearer secret"}
    first = client.get("/service-observations?limit=2", headers=auth).json()
    assert first["schema"] == "service-observation-admin/v1"
    assert [e["seq"] for e in first["events"]] == [1, 2] and first["watermark"] == 2
    second = client.get(f"/service-observations?after={first['watermark']}&limit=10", headers=auth).json()
    assert [e["seq"] for e in second["events"]] == [3, 4, 5] and second["watermark"] == 5
    empty = client.get("/service-observations?after=5", headers=auth).json()
    assert empty == {"schema": "service-observation-admin/v1", "watermark": 5, "events": []}
    for query in ("after=-1", "limit=0", "limit=1001", "limit=-3", f"after={10**30}", f"after={2**62 + 1}",
                  "after=abc", "limit=abc", "after=1.5", "limit="):
        assert client.get(f"/service-observations?{query}", headers=auth).status_code == 400
    assert max(limit for _, limit in runtime.calls) <= 1000


def test_observations_without_a_service_task_are_404_after_auth(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "secret")
    response = observations_client(None).get("/service-observations", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 404


def test_real_runtime_events_carry_hotkey_only_on_the_admin_route(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "secret")
    rt = runtime_for(tmp_path)
    rt.announcement(window=500, randomness=BEACON)
    from reliquary.protocol.service_contract import ServiceContract  # noqa: F401
    pool = rt.seed_pool(environment=MATH, prompt_idx=7, window=500)
    from reliquary.constants import M_ROLLOUTS
    selection = pool.selection(list(range(M_ROLLOUTS)))
    rt.record_exploration(environment=MATH, prompt_idx=7, hotkey="5SecretHotkey", window=500,
                          rewards=[0.0] * M_ROLLOUTS, group_id=selection.sha256,
                          candidate={"pool_sha256": selection.pool_sha256, "seeds": list(selection.seeds)},
                          token_count=10, arrived_at=1.0, now=2.0)
    server, _ = service_server(rt)
    client = TestClient(server.app)
    admin = client.get("/service-observations", headers={"Authorization": "Bearer secret"})
    assert admin.status_code == 200 and "5SecretHotkey" in admin.text
    assert rt.admin_events(after=0, limit=10), "the hotkey is really in the runtime's log"
    for path, response in _call_every_public_get(server, client):
        assert "5SecretHotkey" not in response.text, path
        assert not re.search(r"run_salt|run_meta", response.text), path
    assert client.get("/service-observations").status_code == 401


# ---- accepted but unpaid groups: the response model ------------------------------------------

UNPAID = ["already_scanned", "exploration_banned", "exploration_cap_reached", "exploration_unpaid",
          "exploration_forfeited", "exploration_unaudited", "exploration_audit_queued",
          "service_unproven_published", "exploration_window_closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", UNPAID)
async def test_an_accepted_unpaid_group_ends_with_its_own_unrewarded_final_verdict(status):
    """/submit answers SUBMITTED for every admitted group; the miner then reads /verdicts."""
    from unittest.mock import MagicMock
    from reliquary.validator.service import ValidationService
    runtime = SimpleNamespace(contract=contract_v2())
    request = SimpleNamespace(merkle_root="e" * 64, service_binding={"purpose": "exploration"})
    pending = SimpleNamespace(hotkey="fixture-miner", prompt_idx=0, merkle_root=b"a", request=request,
                              reject_response=None, telemetry=None, rewards=[0.0, 0.0])
    row = {"status": status, "exploration_fraction": 0.0, "proof_status": "passed"}
    batcher = SimpleNamespace(window_start=1, difficulty_auction_enabled=True, service_runtime=runtime,
                              difficulty_auction_metadata_by_id={id(pending): row}, env=SimpleNamespace(name="math"),
                              pending_submissions=lambda: [pending], current_checkpoint_hash="d" * 40,
                              finalize_service_exploration=MagicMock())
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = runtime
    service.server = ValidatorServer()
    await service._record_auction_final_verdicts(batcher, paid_groups=[])
    body = TestClient(service.server.app).get("/verdicts/fixture-miner?details=true").json()
    (verdict,) = body["verdicts"]
    assert verdict["accepted"] is True and verdict["rewarded"] is False
    assert verdict["selected_for_batch"] is False and verdict["is_final"] is True
    assert verdict["outcome_code"] == status
    assert not verdict["explanation"].startswith("Validator outcome:")  # a real, miner-facing sentence
    assert "draw" not in json.dumps(verdict).lower()


# ---- review fixes (round 1) ------------------------------------------------------------------

def _within(seconds, fn):
    import threading
    box = []
    worker = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "the request blocked"
    return box[0]


def test_state_and_miner_state_do_not_wait_for_the_runtime_lock(tmp_path):
    """I1: the request path reads a plain attribute; no lock, no SQLite."""
    import threading
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)
    client = TestClient(server.app)
    held, release = threading.Event(), threading.Event()

    def hold():
        with rt.lock:
            held.set()
            release.wait(30)
    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        for path in ("/state", "/miner-state"):
            assert _within(5, lambda p=path: client.get(p)).status_code == 200
    finally:
        release.set()
        holder.join(5)


def test_state_request_never_calls_runtime_active(tmp_path):
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)

    def boom(*a, **k):
        raise AssertionError("runtime.active() on the request path")
    rt.active = boom
    client = TestClient(server.app)
    assert client.get("/state").status_code == 200 and client.get("/miner-state").status_code == 200


def _service(tmp_path_runtime):
    from reliquary.validator.service import ValidationService
    service = ValidationService.__new__(ValidationService)
    service._service_runtime = tmp_path_runtime
    service.server = ValidatorServer()
    return service


def test_the_service_refreshes_the_active_flag_and_keeps_the_last_value_on_failure(tmp_path):
    rt = runtime_for(tmp_path)
    service = _service(rt)
    rt.active = lambda *a, **k: False
    service._refresh_service_active()
    assert service.server.service_runtime_active is False
    rt.active = lambda *a, **k: True
    service._refresh_service_active()
    assert service.server.service_runtime_active is True

    def broken(*a, **k):
        raise RuntimeError("sqlite busy")
    rt.active = broken
    service._refresh_service_active()
    assert service.server.service_runtime_active is True
    service._service_runtime = None
    service._refresh_service_active()  # a legacy service does nothing


@pytest.mark.asyncio
async def test_the_control_heartbeat_refreshes_the_flag_without_waiting_for_it(tmp_path, monkeypatch):
    import asyncio
    rt = runtime_for(tmp_path)
    service = _service(rt)
    service._window_n = 500
    beats = []
    service._control_store = SimpleNamespace(heartbeat=lambda window: beats.append(window))
    rt.active = lambda *a, **k: False
    real_sleep = asyncio.sleep
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await real_sleep(0.05)          # let the refresh task run
        if len(sleeps) == 2:
            raise asyncio.CancelledError
    monkeypatch.setattr("reliquary.validator.service.asyncio.sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await service._control_heartbeat()
    assert service.server.service_runtime_active is False and sleeps == [5, 5] and len(beats) == 2


@pytest.mark.asyncio
async def test_a_runtime_lock_held_forever_never_stops_the_heartbeat_writes(tmp_path, monkeypatch):
    import asyncio
    import threading
    rt = runtime_for(tmp_path)
    service = _service(rt)
    service._window_n = 7
    service.SERVICE_ACTIVE_REFRESH_TIMEOUT_SECONDS = 0.2
    beats, release, entered = [], threading.Event(), threading.Event()
    service._control_store = SimpleNamespace(heartbeat=lambda window: beats.append(window))

    def stuck(*a, **k):
        entered.set()
        release.wait(10)
        return True
    rt.active = stuck
    real_sleep = asyncio.sleep
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await real_sleep(0.1)
        if len(sleeps) == 4:
            raise asyncio.CancelledError
    monkeypatch.setattr("reliquary.validator.service.asyncio.sleep", fake_sleep)
    import time
    started = time.monotonic()
    try:
        with pytest.raises(asyncio.CancelledError):
            await service._control_heartbeat()
    finally:
        release.set()
    assert time.monotonic() - started < 3, "the heartbeat waited behind the stuck refresh"
    assert entered.is_set() and len(beats) == 4        # one write per beat although the refresh never returned


@pytest.mark.asyncio
async def test_a_slow_refresh_is_given_up_on_and_keeps_the_last_value(tmp_path, caplog):
    import threading
    rt = runtime_for(tmp_path)
    service = _service(rt)
    service.SERVICE_ACTIVE_REFRESH_TIMEOUT_SECONDS = 0.1
    service.server.set_service_runtime_active(True)
    release = threading.Event()
    rt.active = lambda *a, **k: (release.wait(5), False)[1]
    try:
        with caplog.at_level("WARNING"):
            await service._refresh_service_active_bounded()
    finally:
        release.set()
    assert "slow" in caplog.text and service.server.service_runtime_active is True


def test_opening_and_the_boundary_both_refresh_the_flag():
    from reliquary.validator.service import ValidationService
    calls = []
    plan = {"window": 5, "pools": {"m": 0.5}, "picks_target": 1, "batch_slots": 1, "env_mix": [("m", 1)],
            "opened": False}
    runtime = SimpleNamespace(open_window=lambda *a, **k: calls.append("open_window"),
                              announcement=lambda **k: {"pool_randomness": k["randomness"]})
    opener = SimpleNamespace(_service_runtime=runtime, _candidate_service_window=plan,
                             _refresh_service_active=lambda: calls.append("refresh"))
    ValidationService._open_service_window(opener, 5, "beacon")
    assert calls == ["open_window", "refresh"] and plan["opened"] is True

    calls.clear()
    boundary = SimpleNamespace(_recover_leftover_service_windows=lambda window: ([], False),
                               _refresh_service_active=lambda: calls.append("refresh"),
                               _service_window_plan=lambda window: calls.append("plan") or {"window": window})
    assert ValidationService._service_window_boundary(boundary, 6)["plan"] == {"window": 6}
    assert calls == ["refresh", "plan"]


@pytest.mark.asyncio
async def test_a_legacy_service_heartbeat_does_not_touch_a_runtime(monkeypatch):
    import asyncio
    from reliquary.validator.service import ValidationService
    beats = []
    legacy = SimpleNamespace(_window_n=3, _control_store=SimpleNamespace(heartbeat=lambda window: beats.append(window)),
                             _service_runtime=None)
    legacy._refresh_service_active_bounded = lambda: (_ for _ in ()).throw(AssertionError("refresh on a legacy task"))

    async def fake_sleep(seconds):
        raise asyncio.CancelledError
    monkeypatch.setattr("reliquary.validator.service.asyncio.sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await ValidationService._control_heartbeat(legacy)
    assert beats == [3]


def test_state_503_carries_retry_after_on_both_routes(tmp_path):
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt, announce=False)
    client = TestClient(server.app)
    for path in ("/state", "/miner-state"):
        response = client.get(path)
        assert response.status_code == 503 and response.headers["Retry-After"] == "1", path


def test_health_stays_ok_after_a_preparation_failure_in_a_service_run_but_not_in_a_legacy_one():
    failure = {"candidate_window_n": 10, "stage": "prompt_manifest", "error_type": "RuntimeError", "ts": 1.0}

    def status(redaction):
        server = ValidatorServer()
        server.service_health_redaction = redaction
        server.set_window_preparation_state(last_committed_window_n=9, candidate_window_n=10, stage="prompt_manifest")
        server.record_window_preparation_failure(failure)
        return TestClient(server.app).get("/health").json()["status"]
    assert status(True) == "ok"
    assert status(False) == "degraded"  # legacy behaviour unchanged


def test_observations_validate_the_cursor_after_the_token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "secret")
    client = observations_client(events_runtime())
    for query in ("after=abc", "limit=abc", f"after={10**30}"):
        assert client.get(f"/service-observations?{query}").status_code == 401  # never a 422 before auth
        assert client.get(f"/service-observations?{query}", headers={"Authorization": "Bearer secret"}).status_code == 400
    ok = client.get(f"/service-observations?after={2**62}", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def _call_every_public_get(server, client):
    """Call every GET route of the app (typical path parameters); return (path, response)."""
    from fastapi.routing import APIRoute
    results = []
    for route in server.app.routes:
        if not isinstance(route, APIRoute) or "GET" not in route.methods:
            continue
        path = re.sub(r"\{[^}]+\}", "5Typical", route.path)
        results.append((route.path, client.get(path)))
    return results


def test_no_public_route_serves_the_runtime_events(tmp_path, monkeypatch):
    """M6: every GET route is called with events()/admin_events() armed to fail; only the admin
    route may reach admin_events, and none reaches the public events()."""
    monkeypatch.setenv(TOKEN_ENV, "secret")
    rt = runtime_for(tmp_path)
    server, _ = service_server(rt)
    reached = []

    def arm(name):
        def boom(*a, **k):
            reached.append(name)
            return []
        return boom
    rt.events = arm("events")
    rt.admin_events = arm("admin_events")
    client = TestClient(server.app, raise_server_exceptions=False)
    results = _call_every_public_get(server, client)
    paths = {path for path, _ in results}
    assert {"/state", "/miner-state", "/health", "/service-observations"} <= paths
    assert len(paths) >= 10
    assert reached == [], "an unauthenticated GET reached the runtime event log"
    assert "events" not in reached
    admin = client.get("/service-observations", headers={"Authorization": "Bearer secret"})
    assert admin.status_code == 200 and reached == ["admin_events"]
