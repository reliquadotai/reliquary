from __future__ import annotations

import itertools
import threading
from pathlib import Path

from reliquary.validator.prompt_mismatch_circuit import (
    PromptMismatchCircuitBreaker,
)

ENVIRONMENT = "openmathinstruct"
NAMESPACE = "run-netuid-contract-and-source"
HOTKEY = "hotkey-a"
OPERATOR = "operator-a"
_ARRIVALS = itertools.count(1)


def _identities(
    hotkey: str = HOTKEY,
    operator: str = OPERATOR,
) -> dict[str, str]:
    return {"hotkey": hotkey, "operator": operator}


def _circuit(path: Path | None = None, *, namespace: str = NAMESPACE):
    return PromptMismatchCircuitBreaker(
        path,
        namespace=namespace,
        enabled=True,
        failure_threshold=3,
        failure_window_windows=10,
        cooldown_windows=(10, 50, 250),
    )


def _arrival(window: int) -> float:
    return float(window * 1_000 + next(_ARRIVALS))


def _mismatch(
    circuit: PromptMismatchCircuitBreaker,
    signature: str,
    *,
    window: int = 100,
    arrival_ts: float | None = None,
    identities: dict[str, str] | None = None,
):
    return circuit.record_mismatch(
        environment=ENVIRONMENT,
        identities=identities or _identities(),
        window=window,
        precommit_signature=signature,
        precommit_arrival_ts=(_arrival(window) if arrival_ts is None else arrival_ts),
    )


def _success(
    circuit: PromptMismatchCircuitBreaker,
    signature: str,
    *,
    window: int,
    arrival_ts: float | None = None,
    identities: dict[str, str] | None = None,
):
    return circuit.record_binding_success(
        environment=ENVIRONMENT,
        identities=identities or _identities(),
        window=window,
        precommit_signature=signature,
        precommit_arrival_ts=(_arrival(window) if arrival_ts is None else arrival_ts),
    )


def _admit(
    circuit: PromptMismatchCircuitBreaker,
    signature: str,
    *,
    window: int,
    identities: dict[str, str] | None = None,
):
    return circuit.admit_precommit(
        environment=ENVIRONMENT,
        identities=identities or _identities(),
        window=window,
        precommit_signature=signature,
    )


def test_three_mismatches_arm_without_inflight_escalation():
    circuit = _circuit()

    assert _mismatch(circuit, "receipt-1").activated_scopes == ()
    assert _mismatch(circuit, "receipt-2").activated_scopes == ()
    armed = _mismatch(circuit, "receipt-3")
    assert armed.activated_scopes == ("hotkey", "operator")
    assert armed.cooldown_until_window == 110

    blocked = _admit(circuit, "new", window=109)
    assert blocked.allowed is False
    assert blocked.status == "cooldown"
    assert blocked.blocked_scopes == ("hotkey", "operator")
    assert blocked.retry_after_window == 110

    # This receipt was admitted before the breaker armed. Its late terminal
    # mismatch cannot consume the future canary or escalate to 50.
    assert _mismatch(circuit, "already-inflight", window=101).escalated_scopes == ()
    health = circuit.health_snapshot(current_window=109)
    assert health["cooldowns_started_total"] == 1
    assert health["cooldowns_escalated_total"] == 0


def test_mismatches_count_by_arrival_identity_not_completion_order():
    circuit = _circuit()

    # Simulate three parallel workers completing newest-first.
    _mismatch(circuit, "third", window=105, arrival_ts=105_003.0)
    _mismatch(circuit, "first", window=100, arrival_ts=100_001.0)
    armed = _mismatch(circuit, "second", window=101, arrival_ts=101_002.0)

    assert armed.activated_scopes == ("hotkey", "operator")
    assert armed.cooldown_until_window == 115
    assert circuit.health_snapshot(current_window=105)["armed_entries"] == 2


def test_ordinary_binding_success_cannot_reset_rolling_mismatch_debt():
    circuit = _circuit()

    _mismatch(circuit, "mismatch-1")
    _mismatch(circuit, "mismatch-2")
    ignored = _success(circuit, "ordinary-valid", window=100)
    assert ignored.cleared_scopes == ()

    armed = _mismatch(circuit, "mismatch-3", window=101)
    assert armed.activated_scopes == ("hotkey", "operator")
    health = circuit.health_snapshot(current_window=101)
    assert health["noncanary_successes_ignored_total"] == 2


def test_partial_debt_expires_outside_rolling_window():
    circuit = _circuit()
    _mismatch(circuit, "old-1", window=100)
    _mismatch(circuit, "old-2", window=101)

    assert _admit(circuit, "trim", window=111).allowed is True
    _mismatch(circuit, "fresh", window=111)
    health = circuit.health_snapshot(current_window=111)
    assert health["armed_entries"] == 0
    assert health["pending_mismatch_debt"] == 2  # hotkey + operator


def test_only_exact_canary_recovers_or_escalates_cooldown():
    circuit = _circuit()
    for index in range(3):
        _mismatch(circuit, f"initial-{index}")

    canary = _admit(circuit, "canary-1", window=110)
    assert canary.allowed is True
    assert canary.canary_scopes == ("hotkey", "operator")

    stale_success = _success(
        circuit,
        "older-valid-receipt",
        window=99,
        arrival_ts=99_999.0,
    )
    assert stale_success.cleared_scopes == ()
    assert circuit.health_snapshot(current_window=110)["armed_entries"] == 2

    escalated = _mismatch(circuit, "canary-1", window=110)
    assert escalated.escalated_scopes == ("hotkey", "operator")
    assert escalated.cooldown_until_window == 160

    second_canary = _admit(circuit, "canary-fixed", window=160)
    assert second_canary.canary is True
    recovered = _success(circuit, "canary-fixed", window=160)
    assert recovered.cleared_scopes == ("hotkey", "operator")
    health = circuit.health_snapshot(current_window=160)
    assert health["armed_entries"] == 0
    assert health["recovery_tombstones"] == 2


def test_recovery_watermark_ignores_older_outcomes_finishing_late():
    circuit = _circuit()
    for index in range(3):
        _mismatch(circuit, f"initial-{index}", arrival_ts=100_010.0 + index)
    _admit(circuit, "canary", window=110)
    _success(circuit, "canary", window=110, arrival_ts=110_100.0)

    for index in range(3):
        _mismatch(
            circuit,
            f"old-inflight-{index}",
            window=100,
            arrival_ts=100_100.0 + index,
        )

    health = circuit.health_snapshot(current_window=110)
    assert health["armed_entries"] == 0
    assert health["pending_mismatch_debt"] == 0
    assert health["stale_outcomes_ignored_total"] >= 6


def test_operator_canary_from_sibling_clears_only_operator_scope():
    circuit = _circuit()
    for index in range(3):
        _mismatch(circuit, f"bad-{index}")

    sibling_identities = _identities(hotkey="hotkey-b")
    sibling = _admit(
        circuit,
        "sibling-canary",
        window=110,
        identities=sibling_identities,
    )
    assert sibling.canary_scopes == ("operator",)
    recovered = _success(
        circuit,
        "sibling-canary",
        window=110,
        identities=sibling_identities,
    )
    assert recovered.cleared_scopes == ("operator",)

    assert _admit(
        circuit,
        "sibling-normal",
        window=110,
        identities=sibling_identities,
    ).allowed
    offender = _admit(circuit, "offender", window=110)
    assert offender.canary_scopes == ("hotkey",)
    blocked_retry = _admit(circuit, "offender-2", window=110)
    assert blocked_retry.allowed is False
    assert blocked_retry.blocked_scopes == ("hotkey",)


def test_state_survives_restart_and_namespace_change_resets_it(tmp_path):
    path = tmp_path / "prompt-mismatch-circuit.json"
    circuit = _circuit(path)
    for index in range(3):
        _mismatch(circuit, f"persist-{index}")
    assert circuit.flush()
    assert circuit.close()

    restored = _circuit(path)
    blocked = _admit(restored, "after-restart", window=109)
    assert blocked.allowed is False
    assert blocked.retry_after_window == 110
    assert restored.close()

    new_contract = _circuit(path, namespace="different-source-revision")
    health = new_contract.health_snapshot(current_window=109)
    assert health["entries"] == 0
    assert health["state_reset_reason"] == "namespace_changed"
    assert _admit(new_contract, "fresh-namespace", window=109).allowed is True
    assert new_contract.close()


def test_failed_registration_releases_persisted_canary(tmp_path):
    path = tmp_path / "prompt-mismatch-circuit.json"
    circuit = _circuit(path)
    for index in range(3):
        _mismatch(circuit, f"persist-{index}")
    canary = _admit(circuit, "registration-failed", window=110)
    assert canary.canary is True
    circuit.cancel_canary(
        environment=ENVIRONMENT,
        identities=_identities(),
        window=110,
        precommit_signature="registration-failed",
    )
    replacement = _admit(circuit, "replacement", window=110)
    assert replacement.canary is True
    assert circuit.flush()
    assert circuit.close()

    restored = _circuit(path)
    pending = _admit(restored, "other", window=110)
    assert pending.allowed is False
    assert pending.status == "canary_pending"
    assert restored.close()


def test_persistence_io_runs_outside_state_transition(monkeypatch, tmp_path):
    circuit = _circuit(tmp_path / "state.json")
    writer_started = threading.Event()
    release_writer = threading.Event()

    def blocked_writer(_payload):
        writer_started.set()
        release_writer.wait(timeout=2.0)

    monkeypatch.setattr(circuit, "_write_payload", blocked_writer)
    _mismatch(circuit, "nonblocking")

    assert writer_started.wait(timeout=1.0)
    assert circuit.health_snapshot(current_window=100)["persistence_pending"] is True
    release_writer.set()
    assert circuit.flush()
    assert circuit.close()


def test_persistence_failure_is_visible_as_degraded(monkeypatch, tmp_path):
    circuit = _circuit(tmp_path / "state.json")
    monkeypatch.setattr(
        circuit,
        "_write_payload",
        lambda _payload: "OSError: disk unavailable",
    )
    _mismatch(circuit, "persistence-error")
    assert circuit.flush()

    health = circuit.health_snapshot(current_window=100)
    assert health["status"] == "degraded"
    assert health["last_persistence_error"] == "OSError: disk unavailable"
    assert circuit.close()


def test_unreadable_state_path_fails_open_and_degrades_health(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "unreadable" / "prompt-mismatch-circuit.json"

    def deny_access(_path):
        raise PermissionError("state directory is not readable")

    monkeypatch.setattr(Path, "exists", deny_access)
    circuit = _circuit(path)

    health = circuit.health_snapshot(current_window=100)
    assert health["status"] == "degraded"
    assert health["entries"] == 0
    assert health["last_load_error"].startswith("PermissionError:")
    assert _admit(circuit, "allowed-fail-open", window=100).allowed is True
    assert circuit.close()
