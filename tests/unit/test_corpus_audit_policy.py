"""The per-hotkey audit state machine, the draw, and its parameters — no I/O."""

import pytest

from reliquary.corpus.audit_policy import (
    AuditParams, MinerState, after_confirmed_failure, after_pass, decision,
    drawn, effective_state, validate_audit_params,
)

P = AuditParams(q=0.1, probation_submissions=3, hold_seconds=100, suspect_seconds=50,
                ban_after_failures=2, ban_window_seconds=1000, ban_seconds=500)
R = "ab" * 32
SID = "cd" * 32  # has hex letters, so SID.upper() below is a genuinely different (invalid) string


def test_defaults_are_v0():
    assert AuditParams.from_params({}).q == 1.0


@pytest.mark.parametrize("bad", [
    {"audit_q": 0.0}, {"audit_q": 1.5}, {"audit_probation_submissions": 0},
    {"audit_hold_seconds": -1}, {"audit_ban_after_failures": 0}, {"audit_q": "0.1"},
    {"audit_hold_seconds": float("nan")}, {"audit_suspect_seconds": float("nan")},
    {"audit_ban_seconds": float("inf")}, {"audit_ban_window_seconds": float("inf")},
    {"audit_q": 0.5, "audit_hold_seconds": 0},
])
def test_nonsense_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        validate_audit_params(bad)


@pytest.mark.parametrize("key", [
    "audit_suspect_seconds", "audit_ban_seconds", "audit_ban_window_seconds",
])
def test_a_zero_suspect_ban_or_window_length_is_refused(key):
    # Zero turns suspect or a ban into a no-op, or never counts a failure towards a ban.
    with pytest.raises(ValueError, match=key):
        validate_audit_params({key: 0})


def test_a_new_hotkey_is_audited_until_its_probation_passes():
    m = MinerState()
    for _ in range(3):
        assert decision(m, params=P, now=0, received_at=0, recent_submissions=50,
                        randomness_hex=R, submission_id=SID) == "audit"
        m = after_pass(m, P, 1.0, f"{len(m.pass_ids):064x}")
    assert effective_state(m, 0, P) == "sampled"


def test_the_draw_is_deterministic_and_close_to_q():
    ids = [f"{i:064x}" for i in range(4000)]
    hits = sum(drawn(R, i, 0.1) for i in ids)
    assert 300 < hits < 500
    assert [drawn(R, i, 0.1) for i in ids[:50]] == [drawn(R, i, 0.1) for i in ids[:50]]


def _sampled():
    return MinerState(audited_passed=3)


def test_a_sampled_miner_waits_then_passes_unaudited_unless_drawn():
    m = _sampled()
    sid_not = next(f"{i:064x}" for i in range(1000) if not drawn(R, f"{i:064x}", 0.1))
    sid_yes = next(f"{i:064x}" for i in range(1000) if drawn(R, f"{i:064x}", 0.1))
    kw = dict(params=P, received_at=0, recent_submissions=50, randomness_hex=R)
    assert decision(m, now=10, submission_id=sid_yes, **kw) == "audit"
    assert decision(m, now=10, submission_id=sid_not, **kw) == "wait"
    assert decision(m, now=100, submission_id=sid_not, **kw) == "pass_unaudited"


def test_no_beacon_means_audit():
    assert decision(_sampled(), params=P, now=10, received_at=0, recent_submissions=50,
                    randomness_hex=None, submission_id="c" * 64) == "audit"


def test_a_slow_hotkey_is_audited_in_full():
    assert decision(_sampled(), params=P, now=10, received_at=0, recent_submissions=5,
                    randomness_hex=R, submission_id="c" * 64) == "audit"


def test_a_confirmed_failure_makes_it_suspect_then_a_second_bans():
    m = after_confirmed_failure(_sampled(), P, now=0, submission_id="1" * 64)
    assert effective_state(m, 10, P) == "suspect"
    assert decision(m, params=P, now=10, received_at=0, recent_submissions=50,
                    randomness_hex=R, submission_id="c" * 64) == "audit"
    m = after_confirmed_failure(m, P, now=20, submission_id="2" * 64)
    assert effective_state(m, 30, P) == "banned"
    assert decision(m, params=P, now=30, received_at=25, recent_submissions=50,
                    randomness_hex=R, submission_id="c" * 64) == "void_banned"


def test_a_ban_ends_into_a_fresh_probation():
    m = after_confirmed_failure(after_confirmed_failure(_sampled(), P, now=0, submission_id="3" * 64),
                                P, now=1, submission_id="8" * 64)
    assert effective_state(m, 1 + 500 + 1, P) == "probation"


def test_state_round_trips():
    m = after_pass(_sampled(), P, 2.5, SID)
    assert MinerState.from_dict(m.to_dict()) == m


# --- Fix round 1: reviewer findings, ruled by the controller, spec §5/§8 amended ---


def test_the_default_escalation_reaches_a_ban_across_suspect_cycles():
    """Reviewer's reproduction: with the *defaults* (ban_window_seconds now
    604800, strictly longer than suspect_seconds), three failures spaced one
    suspect period apart must still fall inside the same ban window and ban,
    not loop as suspect forever."""
    D = AuditParams()
    m = after_confirmed_failure(MinerState(), D, now=0, submission_id="4" * 64)
    assert effective_state(m, 86401, D) == "probation"
    m = after_confirmed_failure(m, D, now=86401, submission_id="5" * 64)
    m = after_confirmed_failure(m, D, now=172802, submission_id="6" * 64)
    assert effective_state(m, 172802, D) == "banned"


def test_a_confirmed_failure_resets_probation_progress_not_just_on_ban():
    """§5: probation is left only with no confirmed failure, so a hotkey deep
    into probation that fails and later leaves `suspect` must land back in
    `probation` (fresh audits), not jump straight to `sampled` on its old
    audited_passed count."""
    D = AuditParams()
    m = after_confirmed_failure(MinerState(audited_passed=100), D, now=0, submission_id="7" * 64)
    assert effective_state(m, 100, D) == "suspect"
    assert effective_state(m, D.suspect_seconds, D) == "probation"


def test_miner_state_has_no_persisted_state_field():
    assert "state" not in MinerState.__dataclass_fields__


def test_from_dict_ignores_a_stale_state_key():
    d = MinerState(audited_passed=5).to_dict()
    d["state"] = "sampled"  # an old persisted dict; must not resurrect the field
    assert MinerState.from_dict(d) == MinerState(audited_passed=5)


@pytest.mark.parametrize("bad", ["", "x" * 64, R.upper(), R[:-1], R + "00", " " * 64])
def test_malformed_randomness_hex_is_refused(bad):
    with pytest.raises(ValueError):
        drawn(bad, SID, 0.5)


@pytest.mark.parametrize("bad", ["", "x" * 64, SID.upper(), SID[:-1], SID + "00", " " * 64])
def test_malformed_submission_id_is_refused(bad):
    with pytest.raises(ValueError):
        drawn(R, bad, 0.5)


def test_decision_requires_submission_id():
    with pytest.raises(TypeError):
        decision(_sampled(), params=P, now=10, received_at=0, recent_submissions=50,
                 randomness_hex=R)


def test_effective_state_requires_params():
    with pytest.raises(TypeError):
        effective_state(MinerState(), 0)


# --- Task 5 fix round 1: the escalation is written before the verdict, idempotently ---


def test_a_confirmed_failure_counts_once_per_submission():
    sid = "d" * 64
    once = after_confirmed_failure(_sampled(), P, now=0, submission_id=sid)
    assert once.failure_ids == [sid]
    assert after_confirmed_failure(once, P, now=5, submission_id=sid) == once


def test_failure_ids_are_bounded():
    from reliquary.corpus.audit_policy import FAILURE_IDS

    m = MinerState(failure_ids=[f"{i:064x}" for i in range(FAILURE_IDS)])
    m = after_confirmed_failure(m, AuditParams(ban_after_failures=10**6), now=0,
                                submission_id="e" * 64)
    assert len(m.failure_ids) == FAILURE_IDS == 256 and m.failure_ids[-1] == "e" * 64


def test_a_pass_is_counted_once_per_submission_and_the_ids_stay_bounded():
    from reliquary.corpus.audit_policy import PASS_IDS

    m = after_pass(_sampled(), P, 2.5, SID)
    assert after_pass(m, P, 9.0, SID) == m
    for i in range(PASS_IDS + 5):
        m = after_pass(m, P, 1.0, f"{i:064x}")
    assert len(m.pass_ids) == PASS_IDS and m.audited_passed == 3 + 1 + PASS_IDS + 5
