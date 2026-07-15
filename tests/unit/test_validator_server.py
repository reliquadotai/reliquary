"""Validator HTTP server — v2 GRPO market endpoints."""

import os
import time
from types import SimpleNamespace
import bittensor as bt
import pytest
from fastapi.testclient import TestClient

from reliquary.constants import CHALLENGE_K, M_ROLLOUTS
from reliquary.environment.virtual_parquet import PromptSourceUnavailable
from reliquary.protocol.legacy_merkle import (
    compute_legacy_rollouts_merkle_root,
)
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    GrpoBatchState,
    RolloutSubmission,
    RejectReason,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.server import ValidatorServer


# Persistent test keypair so every ``_request()`` carries a verifiable
# envelope signature. Built once per module — generating a fresh
# keypair each call is fine but slows tests.
_TEST_KEYPAIR = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


class _TestWallet:
    """Wallet-like shim around ``_TEST_KEYPAIR`` for ``sign_envelope``."""

    class hotkey:
        ss58_address = _TEST_KEYPAIR.ss58_address

        @staticmethod
        def sign(msg: bytes) -> bytes:
            return _TEST_KEYPAIR.sign(msg)


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "CORRECT" in c else 0.0


class _UnavailableEnv(FakeEnv):
    def __init__(self, fail_at):
        self.fail_at = fail_at

    def __len__(self):
        if self.fail_at == "length":
            raise PromptSourceUnavailable("source unavailable")
        return super().__len__()

    def get_problem(self, idx):
        if self.fail_at == "row":
            raise PromptSourceUnavailable("source unavailable")
        return super().get_problem(idx)

    def source_health(self):
        return {
            "status": "degraded",
            "repo": "owner/repo",
            "revision": "rev",
            "last_error_type": "PromptSourceUnavailable",
        }


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _make_commit(success: bool = False, total_reward: float = 0.0) -> dict:
    """Build a schema-compliant commit dict for server tests."""
    prompt_length = 4
    seq_len = CHALLENGE_K + prompt_length
    tokens = list(range(seq_len))
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v7",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


class _ModelStub:
    """Minimal stub for TokenValidator tests."""
    class config:
        vocab_size = 10000
        max_position_embeddings = 4096


def _batcher(window_start=500, cooldown_map=None, env=None):
    batcher = GrpoWindowBatcher(
        window_start=window_start,
        env=env or FakeEnv(),
        model=_ModelStub(),
        cooldown_map=cooldown_map,
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "CORRECT" if r.reward > 0.5 else "wrong",
        # Server tests post legacy requests without drand_round; disable the gate.
        drand_round_check_enabled=False,
    )
    batcher.current_checkpoint_hash = "sha256:test"
    # Match the per-window randomness used by ``_make_commit`` so the
    # randomness-binding check accepts the test request. Production sets
    # this in service.py's ``_set_window_randomness`` step before the
    # window opens for submissions.
    batcher.randomness = "cd" * 16
    return batcher


def _request(
    prompt_idx=42,
    window_start=500,
    k=4,
    checkpoint_hash="sha256:test",
    randomness="cd" * 16,
    valid_merkle=False,
):
    """Build a fully-signed envelope so tests pass the validator's
    ``ENFORCE_ENVELOPE_SIGNATURE`` gate. ``miner_hotkey`` is bound to
    ``_TEST_KEYPAIR.ss58_address`` because the signature covers it."""
    rollouts = []
    for i in range(M_ROLLOUTS):
        success = i < k
        reward = 1.0 if success else 0.0
        commit = _make_commit(success=success, total_reward=reward)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"],
                reward=reward,
                commit=commit,
                env_name=FakeEnv.name,
            )
        )
    merkle_root = (
        compute_legacy_rollouts_merkle_root(rollouts)
        if valid_merkle
        else "00" * 32
    )
    drand_round = 0
    protocol_version = 1
    nonce = os.urandom(8).hex()
    sig = sign_envelope(
        wallet=_TestWallet,
        miner_hotkey=_TEST_KEYPAIR.ss58_address,
        window_start=window_start,
        prompt_idx=prompt_idx,
        merkle_root=merkle_root,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        randomness=randomness,
        nonce=nonce,
    ).hex()
    return BatchSubmissionRequest(
        miner_hotkey=_TEST_KEYPAIR.ss58_address,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root=merkle_root,
        rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        protocol_version=protocol_version,
        nonce=nonce,
        envelope_signature=sig,
    )


def test_legacy_merkle_shadow_accepts_mismatch_and_exposes_telemetry():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=False).model_dump(mode="json")
    )
    health = server._health_payload()

    assert response.json()["accepted"] is True
    assert health.legacy_merkle_root_enforced is False
    assert health.legacy_merkle_checks_total == 1
    assert health.legacy_merkle_matches == 0
    assert health.legacy_merkle_mismatches == 1
    assert health.legacy_merkle_errors == 0
    assert health.legacy_merkle_distinct_hotkeys == 1
    assert health.legacy_merkle_environments == [FakeEnv.name]
    assert health.legacy_merkle_protocol_versions == {"1": 1}
    assert health.legacy_merkle_last_mismatch_ts is not None


def test_legacy_merkle_shadow_records_current_miner_match():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=True).model_dump(mode="json")
    )
    health = server._health_payload()

    assert response.json()["accepted"] is True
    assert health.legacy_merkle_matches == 1
    assert health.legacy_merkle_mismatches == 0
    assert health.legacy_merkle_last_mismatch_ts is None


def test_legacy_merkle_enforcement_rejects_before_quota(monkeypatch):
    from reliquary.protocol.submission import WindowState
    from reliquary.validator import server as server_module

    monkeypatch.setattr(server_module, "LEGACY_MERKLE_ROOT_ENFORCE", True)
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request(valid_merkle=False).model_dump(mode="json")
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.MERKLE_ROOT_MISMATCH.value,
    }
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_mixed_environment_is_rejected_before_quota():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    payload = _request(valid_merkle=True).model_dump(mode="json")
    payload["rollouts"][1]["env_name"] = "opencodeinstruct"

    response = TestClient(server.app).post("/submit", json=payload)

    assert response.json()["reason"] == RejectReason.BAD_SCHEMA.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts
    assert server._health_payload().legacy_merkle_checks_total == 0


def test_unknown_environment_never_falls_back_to_first_batcher():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    payload = _request(valid_merkle=True).model_dump(mode="json")
    for rollout in payload["rollouts"]:
        rollout["env_name"] = "opencodeinstruct"

    response = TestClient(server.app).post("/submit", json=payload)

    assert response.json()["reason"] == RejectReason.BAD_SCHEMA.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_submit_returns_queued_on_active_window():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == RejectReason.ACCEPTED.value


def test_logical_group_duplicate_is_rejected_quota_neutral_before_proof():
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    first = client.post("/submit", json=_request(k=4).model_dump(mode="json"))
    duplicate = client.post(
        "/submit", json=_request(k=5).model_dump(mode="json")
    )

    assert first.json()["accepted"] is True
    assert duplicate.json() == {
        "accepted": False,
        "reason": RejectReason.HASH_DUPLICATE.value,
    }
    assert server._per_window_counts == {_TEST_KEYPAIR.ss58_address: 1}
    assert batcher.proof_grading_attempts == 1
    assert batcher.logical_group_reservation_count == 1
    assert batcher.logical_group_duplicate_rejects == 1
    assert len(batcher.valid_submissions()) == 1
    health = client.get("/health").json()
    assert health["logical_group_reservations"] == 1
    assert health["logical_group_duplicate_rejects"] == 1
    assert health["logical_group_dedup_by_environment"] == {
        FakeEnv.name: {"reservations": 1, "duplicate_rejects": 1}
    }


def test_logical_group_digest_runs_off_event_loop(monkeypatch):
    from reliquary.protocol.submission import WindowState
    from reliquary.validator import server as server_module

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    original_to_thread = server_module.asyncio.to_thread
    offloaded = []

    async def tracking_to_thread(func, *args, **kwargs):
        if func is GrpoWindowBatcher.try_reserve_logical_group:
            offloaded.append(func)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(server_module.asyncio, "to_thread", tracking_to_thread)

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json")
    )

    assert response.json()["accepted"] is True
    assert offloaded == [GrpoWindowBatcher.try_reserve_logical_group]


@pytest.mark.parametrize("fail_at", ["length", "row"])
def test_prompt_source_outage_is_retryable_quota_neutral_and_visible(fail_at):
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.env = _UnavailableEnv(fail_at)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json")
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "prompt_source_unavailable"
    assert response.headers["retry-after"] == "30"
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts
    health = TestClient(server.app).get("/health").json()
    assert health["status"] == "degraded"
    assert health["prompt_source_unavailable_total"] == 1
    assert health["prompt_sources"][FakeEnv.name]["status"] == "degraded"


def _registration_server(refresh_callback):
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    server.configure_registration_gate(refresh_callback)
    return server


def test_registered_hotkey_passes_without_refresh():
    refresh_calls = 0

    async def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        return True

    server = _registration_server(refresh)
    server.set_registered_hotkeys({_TEST_KEYPAIR.ss58_address})
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["accepted"] is True
    assert refresh_calls == 0


def test_registration_cache_miss_refreshes_before_quota():
    refresh_calls = 0
    server = None

    async def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        server.set_registered_hotkeys({_TEST_KEYPAIR.ss58_address})
        return True

    server = _registration_server(refresh)
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["accepted"] is True
    assert refresh_calls == 1
    assert server._per_window_counts[_TEST_KEYPAIR.ss58_address] == 1


def test_unregistered_hotkey_is_rejected_without_spending_quota():
    async def refresh():
        return True

    server = _registration_server(refresh)
    server.set_registered_hotkeys({"some-other-hotkey"})
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json() == {
        "accepted": False,
        "reason": RejectReason.HOTKEY_NOT_REGISTERED.value,
    }
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_missing_registration_snapshot_fails_closed():
    async def refresh():
        return False

    server = _registration_server(refresh)
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.REGISTRATION_UNAVAILABLE.value
    assert _TEST_KEYPAIR.ss58_address not in server._per_window_counts


def test_recent_last_known_good_registration_survives_refresh_failure():
    async def refresh():
        raise ConnectionError("chain unavailable")

    server = _registration_server(refresh)
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        refreshed_at=time.time() - 400,
    )
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["accepted"] is True


def test_expired_registration_snapshot_fails_closed():
    async def refresh():
        return False

    server = _registration_server(refresh)
    server.set_registered_hotkeys(
        {_TEST_KEYPAIR.ss58_address},
        refreshed_at=time.time() - 901,
    )
    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.REGISTRATION_UNAVAILABLE.value


def test_health_reports_registration_gate_state():
    async def refresh():
        return True

    server = _registration_server(refresh)
    server.set_registered_hotkeys({_TEST_KEYPAIR.ss58_address})

    health = TestClient(server.app).get("/health").json()

    assert health["registration_gate_enforced"] is True
    assert health["registered_hotkey_count"] == 1
    assert health["registration_cache_stale"] is False
    assert health["registration_cache_age_seconds"] >= 0


def test_submit_503_when_no_active_batcher():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 503


def test_submit_409_on_window_mismatch():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request(window_start=999).model_dump(mode="json"))
    assert resp.status_code == 409


def test_state_endpoint_returns_grpo_batch_state():
    from reliquary.protocol.submission import WindowState
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    state = GrpoBatchState(**resp.json())
    assert state.window_n == 500
    assert 42 in state.cooldown_prompts


def test_state_endpoint_503_when_no_active_batcher():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "no_active_window"


def test_state_endpoint_per_env_cooldown_via_query_param():
    """Multi-env: ``/state?env=<name>`` returns THAT env's cooldown set.

    Regression for miners only ever seeing the first active batcher's
    cooldown (math) and applying it to every env, so the code env's real
    cooldown was never communicated.
    """
    from reliquary.protocol.submission import WindowState

    class _Env:
        def __init__(self, name): self.name = name
        def __len__(self): return 1000
        def get_problem(self, idx):
            return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
        def compute_reward(self, p, c): return 0.0

    def _env_batcher(env, cd):
        b = GrpoWindowBatcher(
            window_start=500, env=env, model=_ModelStub(), cooldown_map=cd,
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            completion_text_fn=lambda r: "wrong",
            drand_round_check_enabled=False,
        )
        b.randomness = "cd" * 16
        return b

    math_cd = CooldownMap(cooldown_windows=50); math_cd.record_batched(11, 490)
    code_cd = CooldownMap(cooldown_windows=50); code_cd.record_batched(22, 490)
    server = ValidatorServer()
    # Dict order = math first, so the no-arg /state reflects math (the bug).
    server.set_active_batchers({
        "openmathinstruct": _env_batcher(_Env("openmathinstruct"), math_cd),
        "opencode": _env_batcher(_Env("opencode"), code_cd),
    })
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # No-arg: unchanged — first (math) env.
    base = client.get("/state").json()
    assert 11 in base["cooldown_prompts"] and 22 not in base["cooldown_prompts"]

    # Per-env: each env's own cooldown.
    code = client.get("/state", params={"env": "opencode"}).json()
    assert 22 in code["cooldown_prompts"] and 11 not in code["cooldown_prompts"]
    math = client.get("/state", params={"env": "openmathinstruct"}).json()
    assert 11 in math["cooldown_prompts"] and 22 not in math["cooldown_prompts"]

    # Unknown env while a window is active → 404.
    unk = client.get("/state", params={"env": "nope"})
    assert unk.status_code == 404
    assert unk.json()["detail"] == "unknown_env"


def test_health_exposes_each_environment_window_independently():
    math = _batcher(window_start=500)
    code = _batcher(window_start=500)
    math.valid_count = 4
    math._valid = [SimpleNamespace(prompt_idx=i) for i in range(4)]
    math._proof_admission_count = 4
    math._proof_grading_attempts = 4
    code.valid_count = 8
    code._valid = [SimpleNamespace(prompt_idx=i) for i in range(8)]
    code._proof_admission_count = 8
    code._proof_grading_attempts = 9
    code.force_seal("batch_filled")

    server = ValidatorServer()
    server.set_active_batchers({
        "openmathinstruct": math,
        "opencodeinstruct": code,
    })
    health = TestClient(server.app).get("/health").json()

    assert health["valid_submissions_count"] == 4
    assert health["window_environments"]["openmathinstruct"] == {
        "window_n": 500,
        "sealed": False,
        "force_seal_reason": None,
        "valid_submissions_count": 4,
        "distinct_valid_prompt_count": 4,
        "last_valid_submission_ts": None,
        "seconds_since_last_valid_submission": None,
        "proof_admission_count": 4,
        "proof_grading_attempts": 4,
        "pending_proof_reservations": 0,
        "inflight_proof_reservations": 0,
        "post_trigger_proof_admission_count": 0,
    }
    code_health = health["window_environments"]["opencodeinstruct"]
    assert code_health["valid_submissions_count"] == 8
    assert code_health["sealed"] is True
    assert code_health["force_seal_reason"] == "batch_filled"


def test_state_endpoint_does_not_block_on_batcher_lock():
    """Regression for the 2026-05-12 miner-timeout outage.

    The /state handler must NOT acquire batcher._lock. The submit worker
    holds that lock for the entire GRAIL verify (~5-25s per submission).
    When /state acquired the same lock synchronously on the asyncio event
    loop, 12+ miners polling at once serialised through it and timed out
    at the 60s HTTP budget — and the lock-wait starved the event loop so
    /submit, /health, /checkpoint all backed up behind it too.
    """
    import threading
    import time

    from reliquary.protocol.submission import WindowState
    cd = CooldownMap(cooldown_windows=50)
    cd.record_batched(prompt_idx=42, window=490)
    batcher = _batcher(window_start=500, cooldown_map=cd)
    server = ValidatorServer()
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)

    # Hold batcher._lock from a thread for 2s — simulating a long GRAIL
    # verify in progress. /state must still return promptly.
    lock_held = threading.Event()
    release_lock = threading.Event()

    def _hold_lock():
        with batcher._lock:
            lock_held.set()
            release_lock.wait(timeout=5)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=1), "background thread failed to grab _lock"

    try:
        start = time.monotonic()
        resp = client.get("/state")
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert elapsed < 0.5, (
            f"/state took {elapsed:.2f}s while batcher._lock was held — "
            f"it's still on the lock hot path"
        )
        state = GrpoBatchState(**resp.json())
        assert state.window_n == 500
        assert 42 in state.cooldown_prompts
        assert state.valid_submissions == 0
    finally:
        release_lock.set()
        holder.join(timeout=1)


# --- v2.1: state-aware endpoints ---

def test_submit_rejects_when_state_not_open():
    """When state != OPEN, /submit returns a non-accepted response."""
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.TRAINING)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is False
    assert body["reason"] == "window_not_active"


def test_submit_accepted_when_state_open():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True


def test_state_endpoint_returns_window_state_enum():
    from reliquary.protocol.submission import WindowState
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    client = TestClient(server.app)
    resp = client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "open"
    assert body["window_n"] == 500
    assert body["checkpoint_n"] == 0  # no checkpoint published yet
    assert body["checkpoint_repo_id"] is None
    assert body["checkpoint_revision"] is None


def test_state_endpoint_exposes_checkpoint_when_set():
    from reliquary.protocol.submission import WindowState
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=7,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_007",
        signature="ed25519:sig",
    ))
    client = TestClient(server.app)
    resp = client.get("/state")
    body = resp.json()
    assert body["checkpoint_n"] == 7
    assert body["checkpoint_repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["checkpoint_revision"] == "rev_sha_007"


def test_checkpoint_endpoint_404_when_none_published():
    server = ValidatorServer()
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 404


def test_checkpoint_endpoint_returns_manifest_when_set():
    from reliquary.validator.checkpoint import ManifestEntry
    server = ValidatorServer()
    server.set_current_checkpoint(ManifestEntry(
        checkpoint_n=42,
        repo_id="aivolutionedge/reliquary-sn",
        revision="rev_sha_042",
        signature="ed25519:sig_42",
    ))
    client = TestClient(server.app)
    resp = client.get("/checkpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoint_n"] == 42
    assert body["repo_id"] == "aivolutionedge/reliquary-sn"
    assert body["revision"] == "rev_sha_042"
    assert body["signature"] == "ed25519:sig_42"


# --- provisional response semantics ---

def test_submit_returns_submitted_under_worker_path():
    """Under uvicorn (worker path), /submit must enqueue and return the
    ``SUBMITTED`` sentinel rather than ``ACCEPTED``. The miner needs a way
    to tell "queued, validation pending" from "fully validated and in
    _valid" — the latter is reserved for the inline sync path used by
    tests.
    """
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    batcher.current_checkpoint_hash = "sha256:test"
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    # Pretend a worker is running so the prod enqueue branch is taken.
    server._worker_task = object()

    client = TestClient(server.app)
    resp = client.post("/submit", json=_request().model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["reason"] == "submitted"


def test_full_server_queue_returns_pending_proof_reservation():
    import asyncio
    from reliquary.protocol.submission import WindowState

    server = ValidatorServer()
    batcher = _batcher(window_start=500)
    server.set_active_batcher(batcher)
    server.set_current_state(WindowState.OPEN)
    server._worker_task = object()
    server._submit_queue = asyncio.Queue(maxsize=1)
    server._submit_queue.put_nowait((object(), object()))

    response = TestClient(server.app).post(
        "/submit", json=_request().model_dump(mode="json"),
    )

    assert response.json()["reason"] == RejectReason.BATCH_FILLED.value
    assert batcher.pending_proof_reservations == 0
    assert batcher.proof_grading_attempts == 0


# --- worker drops items whose batcher is no longer active ---

@pytest.mark.asyncio
async def test_worker_drops_late_items_for_stale_batcher():
    """The worker must skip queue items whose batcher is no longer the
    server's active_batcher — those items were enqueued when the previous
    window was OPEN and would otherwise burn GRAIL compute on a sealed
    _valid that will never be archived.
    """
    import asyncio

    server = ValidatorServer()
    old_batcher = _batcher(window_start=500)
    new_batcher = _batcher(window_start=501)
    accept_calls: list[int] = []

    def _spy_accept(req):
        accept_calls.append(req.prompt_idx)
        from reliquary.protocol.submission import BatchSubmissionResponse
        return BatchSubmissionResponse(accepted=True, reason=RejectReason.ACCEPTED)

    old_batcher.accept_submission = _spy_accept
    new_batcher.accept_submission = _spy_accept

    # Active is the new batcher. The queue has a leftover stale item plus
    # one for the new batcher.
    server.set_active_batcher(new_batcher)
    old_request = _request(prompt_idx=11)
    assert old_batcher.try_reserve_proof_admission(old_request) == (True, None)
    await server._submit_queue.put((old_request, old_batcher))
    await server._submit_queue.put((_request(prompt_idx=22), new_batcher))

    # Run the worker just long enough to drain both items.
    worker_task = asyncio.create_task(server._submit_worker())
    # Yield until queue is empty.
    for _ in range(50):
        if server._submit_queue.empty():
            break
        await asyncio.sleep(0.01)
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # Only the active-batcher item should have hit accept_submission.
    assert accept_calls == [22], (
        f"expected only prompt 22 to be processed, got {accept_calls}"
    )
    assert old_batcher.pending_proof_reservations == 0
    assert old_batcher.proof_grading_attempts == 0
