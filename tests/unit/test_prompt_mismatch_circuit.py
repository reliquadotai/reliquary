from __future__ import annotations

from reliquary.validator.prompt_mismatch_circuit import (
    PromptMismatchCircuitBreaker,
)

ENVIRONMENT = "openmathinstruct"
PROFILE = "v5-test-profile"
HOTKEY = "hotkey-a"
OPERATOR = "operator-a"


def _identities(
    hotkey: str = HOTKEY,
    operator: str = OPERATOR,
) -> dict[str, str]:
    return {"hotkey": hotkey, "operator": operator}


def _mismatch(
    circuit: PromptMismatchCircuitBreaker,
    signature: str,
    *,
    window: int = 100,
    identities: dict[str, str] | None = None,
):
    return circuit.record_mismatch(
        environment=ENVIRONMENT,
        generation_profile_id=PROFILE,
        identities=identities or _identities(),
        window=window,
        precommit_signature=signature,
    )


def _admit(
    circuit: PromptMismatchCircuitBreaker,
    signature: str,
    *,
    window: int,
    identities: dict[str, str] | None = None,
    profile: str = PROFILE,
):
    return circuit.admit_precommit(
        environment=ENVIRONMENT,
        generation_profile_id=profile,
        identities=identities or _identities(),
        window=window,
        precommit_signature=signature,
    )


def test_three_consecutive_mismatches_arm_without_inflight_escalation():
    circuit = PromptMismatchCircuitBreaker(
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )

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

    # This receipt was admitted before the breaker armed.  Its late terminal
    # mismatch must not consume the post-cooldown canary or escalate to 50.
    assert (
        _mismatch(
            circuit,
            "already-inflight",
            window=101,
        ).escalated_scopes
        == ()
    )
    assert _admit(circuit, "still-blocked", window=109).retry_after_window == 110
    assert circuit.health_snapshot(current_window=109)["cooldowns_escalated_total"] == 0


def test_verified_prompt_binding_resets_a_partial_streak():
    circuit = PromptMismatchCircuitBreaker(
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    _mismatch(circuit, "before-success-1")
    _mismatch(circuit, "before-success-2")
    cleared = circuit.record_binding_success(
        environment=ENVIRONMENT,
        generation_profile_id=PROFILE,
        identities=_identities(),
        window=100,
    )
    assert cleared.cleared_scopes == ("hotkey", "operator")

    _mismatch(circuit, "after-success-1", window=101)
    _mismatch(circuit, "after-success-2", window=101)
    assert _admit(circuit, "still-compatible", window=101).allowed is True
    assert circuit.health_snapshot(current_window=101)["armed_entries"] == 0


def test_one_canary_recovers_or_escalates_cooldown():
    circuit = PromptMismatchCircuitBreaker(
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    for index in range(3):
        _mismatch(circuit, f"initial-{index}")

    canary = _admit(circuit, "canary-1", window=110)
    assert canary.allowed is True
    assert canary.canary is True
    assert canary.canary_scopes == ("hotkey", "operator")

    second = _admit(circuit, "canary-2", window=110)
    assert second.allowed is False
    assert second.status == "canary_pending"
    assert second.retry_after_window == 111

    escalated = _mismatch(circuit, "canary-1", window=110)
    assert escalated.escalated_scopes == ("hotkey", "operator")
    assert escalated.cooldown_until_window == 160
    assert _admit(circuit, "too-early", window=159).retry_after_window == 160

    second_canary = _admit(circuit, "canary-50", window=160)
    assert second_canary.allowed is True
    assert second_canary.canary is True
    escalated_again = _mismatch(circuit, "canary-50", window=160)
    assert escalated_again.escalated_scopes == ("hotkey", "operator")
    assert escalated_again.cooldown_until_window == 410

    recovered_canary = _admit(circuit, "canary-fixed", window=410)
    assert recovered_canary.allowed is True
    assert recovered_canary.canary is True
    recovered = circuit.record_binding_success(
        environment=ENVIRONMENT,
        generation_profile_id=PROFILE,
        identities=_identities(),
        window=410,
    )
    assert recovered.cleared_scopes == ("hotkey", "operator")
    assert _admit(circuit, "normal-again", window=410).canary is False
    assert circuit.health_snapshot(current_window=410)["entries"] == 0


def test_operator_scope_blocks_sibling_but_profile_change_does_not():
    circuit = PromptMismatchCircuitBreaker(
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    for index in range(3):
        _mismatch(circuit, f"bad-{index}")

    sibling = _admit(
        circuit,
        "sibling",
        window=101,
        identities=_identities(hotkey="hotkey-b"),
    )
    assert sibling.allowed is False
    assert sibling.blocked_scopes == ("operator",)

    unrelated_operator = _admit(
        circuit,
        "unrelated",
        window=101,
        identities=_identities(hotkey="hotkey-c", operator="operator-c"),
    )
    assert unrelated_operator.allowed is True

    upgraded_profile = _admit(
        circuit,
        "upgraded",
        window=101,
        profile="v6-new-renderer",
    )
    assert upgraded_profile.allowed is True


def test_state_survives_restart_and_failed_registration_releases_canary(
    tmp_path,
):
    path = tmp_path / "prompt-mismatch-circuit.json"
    circuit = PromptMismatchCircuitBreaker(
        path,
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    for index in range(3):
        _mismatch(circuit, f"persist-{index}")

    restored = PromptMismatchCircuitBreaker(
        path,
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    blocked = _admit(restored, "after-restart", window=109)
    assert blocked.allowed is False
    assert blocked.retry_after_window == 110

    canary = _admit(restored, "registration-failed", window=110)
    assert canary.allowed is True
    assert canary.canary is True
    restored.cancel_canary(
        environment=ENVIRONMENT,
        generation_profile_id=PROFILE,
        identities=_identities(),
        window=110,
        precommit_signature="registration-failed",
    )
    replacement = _admit(restored, "replacement", window=110)
    assert replacement.allowed is True
    assert replacement.canary is True

    restored_again = PromptMismatchCircuitBreaker(
        path,
        enabled=True,
        failure_threshold=3,
        cooldown_windows=(10, 50, 250),
    )
    pending = _admit(restored_again, "other", window=110)
    assert pending.allowed is False
    assert pending.status == "canary_pending"
