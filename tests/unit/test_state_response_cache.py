"""/state serves cached serialized bytes instead of re-serializing per poll.

The 2026-08-01 py-spy profile of the live v3 validator showed the event loop
90% busy in per-request routing + pydantic serialization — miners poll /state
continuously, and every poll re-built and re-serialized a GrpoBatchState whose
``cooldown_prompts`` list can carry thousands of indices. That churn starves
the proof worker's Python sections of the GIL (measured as the 3-4x wrapper
overhead around GPU proof time).

The cache is keyed by everything that can change the payload — window,
window-state, submission count, checkpoint — so a hit is byte-identical to a
rebuild. No TTL: staleness is structurally zero.
"""

from fastapi.testclient import TestClient

from reliquary.protocol.submission import GrpoBatchState, WindowState
from reliquary.validator.cooldown import CooldownMap

from tests.unit.test_validator_server import (  # reuse the existing harness
    _ModelStub,
    _always_true_proof,
    _batcher,
    ValidatorServer,
)


def _count_builds(monkeypatch):
    """Spy on GrpoBatchState construction inside the server module."""
    import reliquary.validator.server as server_module

    calls = {"n": 0}
    real = server_module.GrpoBatchState

    class _Counting(real):
        def __init__(self, *args, **kwargs):
            calls["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server_module, "GrpoBatchState", _Counting)
    return calls


def _server(window_start=500):
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=window_start, cooldown_map=cd))
    server.set_current_state(WindowState.OPEN)
    return server


def test_repeat_polls_serve_identical_bytes_without_rebuilding(monkeypatch):
    calls = _count_builds(monkeypatch)
    client = TestClient(_server().app)

    first = client.get("/state")
    second = client.get("/state")

    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert calls["n"] == 1, "second poll must be a cache hit"


def test_payload_is_wire_identical_to_uncached_build():
    client = TestClient(_server().app)
    payload = client.get("/state").json()

    # v2 exclusions preserved through the cached path
    assert "protocol_version" not in payload
    assert "generation_profile_id" not in payload
    assert "generation_contract" not in payload
    # strict miner parse (extra=forbid) still accepts it
    state = GrpoBatchState(**payload)
    assert state.window_n == 500
    assert 42 in state.cooldown_prompts


def test_new_window_invalidates(monkeypatch):
    calls = _count_builds(monkeypatch)
    server = _server(window_start=500)
    client = TestClient(server.app)

    assert client.get("/state").json()["window_n"] == 500
    cd = CooldownMap(cooldown_windows=50)
    server.set_active_batcher(_batcher(window_start=501, cooldown_map=cd))

    assert client.get("/state").json()["window_n"] == 501
    assert calls["n"] == 2


def test_window_state_change_invalidates():
    server = _server()
    client = TestClient(server.app)

    assert client.get("/state").json()["state"] == WindowState.OPEN.value
    server.set_current_state(WindowState.TRAINING)
    assert client.get("/state").json()["state"] == WindowState.TRAINING.value


def test_submission_count_change_invalidates():
    server = _server()
    client = TestClient(server.app)
    batcher = server.active_batcher

    before = client.get("/state").json()["valid_submissions"]
    # bump whichever counter the handler reports for this batcher
    if getattr(batcher, "difficulty_auction_enabled", False):
        batcher.pending_count += 1
    else:
        batcher.valid_count += 1
    after = client.get("/state").json()["valid_submissions"]

    assert after == before + 1


def test_checkpoint_change_invalidates():
    from types import SimpleNamespace

    server = _server()
    client = TestClient(server.app)

    assert client.get("/state").json()["checkpoint_n"] == 0
    server._current_checkpoint = SimpleNamespace(
        checkpoint_n=54,
        repo_id="ReliquaryForge/qwen3.5-4b-reliquary-v4",
        revision="e" * 40,
    )
    payload = client.get("/state").json()
    assert payload["checkpoint_n"] == 54
    assert payload["checkpoint_revision"] == "e" * 40


def test_cache_is_per_env_no_thrash(monkeypatch):
    """Alternating per-env polls must not evict each other."""
    calls = _count_builds(monkeypatch)

    class _Env:
        def __init__(self, name):
            self.name = name

        def __len__(self):
            return 1000

        def get_problem(self, idx):
            return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}

        def compute_reward(self, p, c):
            return 0.0

    from tests.unit.test_validator_server import GrpoWindowBatcher

    def _env_batcher(env, prompt_idx):
        cd = CooldownMap(cooldown_windows=50)
        cd.record_batched(prompt_idx, 490)
        b = GrpoWindowBatcher(
            window_start=500,
            env=env,
            model=_ModelStub(),
            cooldown_map=cd,
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            completion_text_fn=lambda r: "wrong",
            drand_round_check_enabled=False,
        )
        b.randomness = "cd" * 16
        return b

    server = ValidatorServer()
    server.set_active_batchers({
        "openmathinstruct": _env_batcher(_Env("openmathinstruct"), 11),
        "opencode": _env_batcher(_Env("opencode"), 22),
    })
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    for _ in range(3):
        math = client.get("/state", params={"env": "openmathinstruct"}).json()
        code = client.get("/state", params={"env": "opencode"}).json()
        assert 11 in math["cooldown_prompts"] and 22 not in math["cooldown_prompts"]
        assert 22 in code["cooldown_prompts"] and 11 not in code["cooldown_prompts"]

    assert calls["n"] == 2, "one build per env, alternation must not thrash"


def test_cache_hits_are_served_by_the_asgi_fast_path(monkeypatch):
    """A hit must be answered before the FastAPI stack: if the fast path
    serves sentinel bytes, the response IS those bytes — the route (and its
    pydantic serialization) never ran."""
    server = _server()
    client = TestClient(server.app)
    client.get("/state")  # populate the cache

    sentinel = b'{"sentinel": true}'
    monkeypatch.setattr(
        server, "_cached_state_response", lambda env: sentinel
    )
    resp = client.get("/state")
    assert resp.status_code == 200
    assert resp.content == sentinel


def test_fast_path_records_latency_sample():
    server = _server()
    client = TestClient(server.app)
    client.get("/state")
    before = len(server._endpoint_latency_samples_ms["/state"])
    client.get("/state")  # hit, served by the middleware
    assert len(server._endpoint_latency_samples_ms["/state"]) == before + 1


def test_fastpath_kill_switch(monkeypatch):
    """RELIQUARY_STATE_FASTPATH=0 removes the middleware; the route still
    works (and still serves its own in-handler cache)."""
    monkeypatch.setenv("RELIQUARY_STATE_FASTPATH", "0")
    server = _server()
    client = TestClient(server.app)
    client.get("/state")  # populate

    sentinel = b'{"sentinel": true}'
    monkeypatch.setattr(server, "_cached_state_response", lambda env: sentinel)
    resp = client.get("/state")
    assert resp.status_code == 200
    assert resp.content != sentinel, "fast path must be absent when disabled"
    assert resp.json()["window_n"] == 500


def test_error_paths_unchanged():
    server = ValidatorServer()
    client = TestClient(server.app)
    assert client.get("/state").status_code == 503

    server = _server()
    client = TestClient(server.app)
    assert client.get("/state", params={"env": "nope"}).status_code == 404
