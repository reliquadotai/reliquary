from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionResponse,
    RejectReason,
    WindowState,
)
from reliquary.validator.observability import (
    DrandRoundObservation,
    classify_drand_round,
)
from reliquary.validator.server import ValidatorServer


class _Env:
    def __len__(self) -> int:
        return 1000


class _ObservableBatcher:
    window_start = 500
    current_checkpoint_hash = ""
    cooldown_prompts_snapshot: list[int] = []
    valid_count = 3
    window_open_drand_round = 100
    last_valid_submission_wall_ts = 1234.5
    drand_round_backward_tolerance = 0

    def __init__(self) -> None:
        self.env = _Env()
        self._seal_trigger_round = None
        self.drand_round_check_enabled = True
        self._sealed = False

    def is_sealed(self) -> bool:
        return self._sealed

    def prompt_submission_count(self, prompt_idx: int) -> int:
        return 0

    def distinct_valid_prompt_count(self) -> int:
        return 2

    def seconds_since_last_valid_submission(self) -> float:
        return 12.5

    def observe_drand_round(self, drand_round: int, *, t_arrival=None):
        reject = None
        if drand_round > 100:
            reject = RejectReason.FUTURE_ROUND
        elif drand_round < 100:
            reject = RejectReason.STALE_ROUND
        return DrandRoundObservation(
            submitted_drand_round=drand_round,
            arrival_drand_round=100,
            drand_delta=drand_round - 100,
            drand_tolerance=0,
            drand_status=classify_drand_round(drand_round, 100, 0),
            reject_reason=reject,
        )

    def validate_drand_round(self, drand_round: int, *, t_arrival=None):
        return self.observe_drand_round(
            drand_round, t_arrival=t_arrival
        ).reject_reason

    def accept_submission(self, request, *, telemetry=None):
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED,
        )


def _submission(*, drand_round: int, hotkey: str = "hkA") -> dict:
    return {
        "miner_hotkey": hotkey,
        "prompt_idx": 42,
        "window_start": 500,
        "merkle_root": f"{drand_round:064x}",
        "rollouts": [
            {
                "tokens": [1],
                "reward": 1.0,
                "commit": {"tokens": [1]},
                "env_name": "openmathinstruct",
            }
            for _ in range(M_ROLLOUTS)
        ],
        "checkpoint_hash": "",
        "drand_round": drand_round,
    }


def _payloads(caplog):
    out = []
    for record in caplog.records:
        msg = record.getMessage()
        if not msg.startswith("validator_submit_lifecycle "):
            continue
        out.append(json.loads(msg.split(" ", 1)[1]))
    return out


def _open_server(batcher=None) -> ValidatorServer:
    server = ValidatorServer()
    server.set_current_state(WindowState.OPEN)
    server.set_active_batcher(batcher or _ObservableBatcher())
    return server


def test_drand_lifecycle_logs_current_stale_future_fields(caplog):
    server = _open_server()
    client = TestClient(server.app)

    with caplog.at_level(logging.INFO, logger="reliquary.validator.server"):
        client.post("/submit", json=_submission(drand_round=100, hotkey="hkC"))
        client.post("/submit", json=_submission(drand_round=99, hotkey="hkS"))
        client.post("/submit", json=_submission(drand_round=101, hotkey="hkF"))

    drand = [
        p for p in _payloads(caplog)
        if p.get("stage") == "drand_validated"
    ]
    by_round = {p["submitted_drand_round"]: p for p in drand}

    assert by_round[100]["arrival_drand_round"] == 100
    assert by_round[100]["drand_delta"] == 0
    assert by_round[100]["drand_status"] == "current"
    assert by_round[99]["drand_delta"] == -1
    assert by_round[99]["drand_status"] == "stale"
    assert by_round[99]["reject_reason"] == "stale_round"
    assert by_round[101]["drand_delta"] == 1
    assert by_round[101]["drand_status"] == "future"
    assert by_round[101]["reject_reason"] == "future_round"


def test_accept_reject_lifecycle_logs_required_fields(caplog):
    server = _open_server()
    client = TestClient(server.app)

    with caplog.at_level(logging.INFO, logger="reliquary.validator.server"):
        client.post("/submit", json=_submission(drand_round=100, hotkey="hkA"))
        client.post("/submit", json=_submission(drand_round=99, hotkey="hkB"))

    payloads = _payloads(caplog)
    decisions = [
        p for p in payloads
        if p.get("stage") in {"candidate_accepted", "candidate_rejected"}
    ]
    required = {
        "window_n",
        "prompt_idx",
        "hotkey",
        "t_arrival",
        "t_decision",
        "queue_wait_ms",
        "verify_ms",
        "total_ms",
        "submitted_drand_round",
        "arrival_drand_round",
        "drand_delta",
        "drand_tolerance",
        "window_open_drand_round",
        "seal_trigger_round",
        "valid_submissions_at_arrival",
        "valid_submissions_at_decision",
        "reject_stage",
        "reject_reason",
        "prompt_hash_lead",
    }

    assert any(p["stage"] == "candidate_accepted" for p in decisions)
    assert any(p["stage"] == "candidate_rejected" for p in decisions)
    for payload in decisions:
        assert required.issubset(payload.keys())


def test_batch_filled_log_explains_trigger_round(caplog):
    batcher = _ObservableBatcher()
    batcher.drand_round_check_enabled = False
    batcher._seal_trigger_round = 100
    server = _open_server(batcher)
    client = TestClient(server.app)

    with caplog.at_level(logging.INFO, logger="reliquary.validator.server"):
        resp = client.post(
            "/submit", json=_submission(drand_round=101, hotkey="hkLate")
        )

    assert resp.json()["reason"] == RejectReason.BATCH_FILLED.value
    rejected = [
        p for p in _payloads(caplog)
        if p.get("stage") == "candidate_rejected"
    ][0]
    assert rejected["batch_filled_reason"] == "submitted_round_gt_seal_trigger_round"
    assert rejected["trigger_round"] == 100
    assert rejected["seal_trigger_round"] == 100
    assert rejected["current_valid_count"] == 3


def test_health_endpoint_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret-value-123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access-key-value-123")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value_123")
    monkeypatch.setenv("R2_BUCKET_ID", "public-bucket-name")

    server = _open_server()
    server.set_training_accumulator_state({
        "checkpoint_revision": "rev-a",
        "targets": {"math": 8, "code": 8},
        "counts": {"math": 3, "code": 8},
        "ready": False,
    })
    server.configure_archive_queue_telemetry(lambda: {
        "depth": 2,
        "oldest_window": 490,
        "oldest_age_seconds": 30.5,
        "uploads_succeeded_total": 7,
        "upload_failures_total": 1,
        "last_upload_success_ts": 1200.0,
        "last_upload_failure_ts": 1100.0,
        "last_uploaded_window": 489,
        "last_failed_window": 490,
    })
    monkeypatch.setattr(server, "_current_drand_round_best_effort", lambda: 123)
    client = TestClient(server.app)
    body = client.get("/health").json()
    text = json.dumps(body)

    assert body["status"] == "ok"
    assert body["batch_size"] > 0
    assert body["current_quicknet_drand_round"] == 123
    assert body["distinct_valid_prompt_count"] == 2
    assert body["last_valid_submission_ts"] == 1234.5
    assert body["seconds_since_last_valid_submission"] == 12.5
    assert body["sparse_valid_idle_seal_seconds"] == 300.0
    assert body["forced_seed_enforced"] is True
    assert body["forced_seed_consistency_floor"] == 0.8
    assert body["forced_seed_rollout_floor"] == 0.75
    assert body["forced_seed_cdf_enforced"] is False
    assert body["forced_seed_cdf_boundary_epsilon"] == 0.002
    assert body["archive_queue_depth"] == 2
    assert body["archive_queue_oldest_window"] == 490
    assert body["archive_queue_oldest_age_seconds"] == 30.5
    assert body["archive_uploads_succeeded_total"] == 7
    assert body["archive_upload_failures_total"] == 1
    assert body["archive_last_uploaded_window"] == 489
    assert body["archive_last_failed_window"] == 490
    assert body["sparse_valid_idle_min_distinct_prompts"] == 4
    assert body["sparse_valid_max_window_seconds"] == 900.0
    assert body["training_accumulator_checkpoint_revision"] == "rev-a"
    assert body["training_accumulator_targets"] == {"math": 8, "code": 8}
    assert body["training_accumulator_counts"] == {"math": 3, "code": 8}
    assert body["training_accumulator_ready"] is False
    assert "secret-value-123" not in text
    assert "access-key-value-123" not in text
    assert "hf_secret_value_123" not in text
