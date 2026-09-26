"""The pure checks of the end-to-end rehearsal (scripts/corpus_e2e.py) for a
partially audited task: they must fail when the mechanism leaks."""

from types import SimpleNamespace

from scripts.corpus_e2e import (
    _audit_throughput,
    _caught_by,
    _route_ok,
    audit_params_from_args,
    partial_audit_checks,
    submission_rows,
)

Q, HOLD = 0.2, 300.0
HONEST, LATE, CHEAT = "5honest", "5late", "5cheat"
ROLE_OF = {HONEST: "honest", LATE: "late", CHEAT: "dishonest"}


def _sid(i: int) -> str:
    return f"{i:064x}"


def _sub(i, hotkey, received_at, tokens=100):
    return {"hotkey": hotkey, "received_at": received_at, "token_count": tokens}


def _verdict(i, hotkey, audited_at, *, passed, audited=True, draw=None, reason=None):
    v = {"submission_id": _sid(i), "hotkey": hotkey, "audited_at": audited_at,
         "passed": passed, "audited": audited, "reason": reason, "worst_exp": 0}
    if draw is not None:
        v["draw"] = draw
    return v


def _run(late_tail=None, *, settle_all_passed=True):
    """Honest: 5 audited + 2 unaudited passes. Late: 5 honest audited, then
    switched records 11..14: 11 waits, 12 is drawn and fails, 11/13 are then
    audited backwards, 14 arrives after detection and is audited."""
    subs, verdicts = [], []

    def add(i, hotkey, received, audited_at, **kw):
        subs.append(_sub(i, hotkey, received))
        verdicts.append(_verdict(i, hotkey, audited_at, **kw))

    for i in range(5):
        add(i, HONEST, 10 + i, 11 + i, passed=True)
    add(5, HONEST, 20, 330, passed=True, audited=False, draw={"round": 1, "q": Q, "drawn": False})
    add(6, HONEST, 21, 331, passed=True, audited=False, draw={"round": 2, "q": Q, "drawn": False})
    for i in range(7, 11):
        add(i, LATE, 40 + i, 41 + i, passed=True)
    add(20, LATE, 44, 45, passed=True)  # fifth pre-switch
    tail = late_tail or [
        (11, 100, 161, dict(passed=False, reason="exp_mismatch")),
        (12, 104, 160, dict(passed=False, reason="exp_mismatch",
                            draw={"round": 9, "q": Q, "drawn": True})),
        (13, 108, 162, dict(passed=False, reason="exp_mismatch")),
        (14, 170, 171, dict(passed=False, reason="exp_mismatch")),
    ]
    for i, received, audited_at, kw in tail:
        add(i, LATE, received, audited_at, **kw)
    settled = {v["submission_id"] for v in verdicts} if settle_all_passed else set()
    rows = submission_rows(subs, verdicts, role_of=ROLE_OF, switch_at=90.0, settled=settled)
    return subs, verdicts, rows


LATE_STATE = {"late": {"effective_state": "suspect", "confirmed_failures": [159.5]}}


def test_a_clean_run_passes_every_check():
    _, _, rows = _run()
    report, checks = partial_audit_checks(rows, LATE_STATE, q=Q, hold=HOLD)
    assert all(checks.values()), checks
    assert report["detection"]["caught_by"] == "draw"
    assert report["detection"]["held_at_detection"] == 2
    assert report["by_role"]["late/pre_switch"]["paid"] == 5
    assert report["by_role"]["honest"]["passed_unaudited"] == 2


def test_rows_split_the_late_hotkey_at_the_switch_and_mark_paid_only_passed_settled():
    _, _, rows = _run()
    phases = {r["submission_id"]: r["phase"] for r in rows if r["role"] == "late"}
    assert phases[_sid(20)] == "pre_switch" and phases[_sid(11)] == "switched"
    assert not any(r["paid"] for r in rows if not r["passed"])
    assert [r["received_at"] for r in rows] == sorted(r["received_at"] for r in rows)


def test_a_held_record_passed_unaudited_after_detection_fails_the_run():
    tail = [
        (11, 100, 161, dict(passed=True, audited=False, draw={"round": 8, "q": Q, "drawn": False})),
        (12, 104, 160, dict(passed=False, reason="exp_mismatch",
                            draw={"round": 9, "q": Q, "drawn": True})),
    ]
    _, _, rows = _run(tail)
    _, checks = partial_audit_checks(rows, LATE_STATE, q=Q, hold=HOLD)
    assert not checks["late_held_records_all_audited"]
    assert not checks["late_no_pass_after_detection"]
    assert not checks["late_post_switch_none_paid"]


def test_a_switched_record_paid_before_any_detection_fails_the_run():
    tail = [
        (11, 100, 400, dict(passed=True, audited=False, draw={"round": 8, "q": Q, "drawn": False})),
        (12, 104, 500, dict(passed=False, reason="exp_mismatch",
                            draw={"round": 9, "q": Q, "drawn": True})),
    ]
    _, _, rows = _run(tail)
    _, checks = partial_audit_checks(rows, LATE_STATE, q=Q, hold=HOLD)
    assert not checks["late_post_switch_none_paid"]


def test_a_cheater_never_caught_fails_the_run():
    tail = [(11, 100, 400, dict(passed=True, audited=False,
                                draw={"round": 8, "q": Q, "drawn": False}))]
    _, _, rows = _run(tail)
    report, checks = partial_audit_checks(rows, {}, q=Q, hold=HOLD)
    assert not checks["late_caught"] and report["detection"] is None


def test_a_state_not_suspect_after_the_failure_fails_the_run():
    _, _, rows = _run()
    state = {"late": {"effective_state": "sampled", "confirmed_failures": [159.5]}}
    _, checks = partial_audit_checks(rows, state, q=Q, hold=HOLD)
    assert not checks["late_suspect_at_first_failure"]
    banned = {"late": {"effective_state": "banned", "confirmed_failures": [159.5, 161, 162]}}
    assert partial_audit_checks(rows, banned, q=Q, hold=HOLD)[1]["late_suspect_at_first_failure"]


def test_an_unpaid_honest_record_fails_the_run():
    _, _, rows = _run(settle_all_passed=False)
    _, checks = partial_audit_checks(rows, LATE_STATE, q=Q, hold=HOLD)
    assert not checks["honest_all_passed_and_paid"]
    assert not checks["late_pre_switch_all_paid"]


def test_caught_by_names_the_slow_hotkey_rule_when_the_record_was_not_drawn():
    first = {"phase": "switched", "draw": None, "audited_at": 100.0}
    rows = [{"role": "late", "received_at": t, "audited_at": t + 1, "audited": False, "passed": False}
            for t in (98.0, 99.0)]
    assert _caught_by(first, rows, q=Q, hold=HOLD) == "slow_hotkey"
    rows = [{"role": "late", "received_at": 90.0 + i, "audited_at": 91.0 + i, "audited": False,
             "passed": False} for i in range(6)]
    assert _caught_by(first, rows, q=Q, hold=HOLD) == "no_beacon"
    drawn = {**first, "draw": {"drawn": True}}
    assert _caught_by(drawn, rows, q=Q, hold=HOLD) == "draw"


def test_caught_by_names_probation_when_the_switch_came_before_it_ended():
    first = {"phase": "switched", "draw": None, "audited_at": 100.0}
    rows = [{"role": "late", "received_at": 10.0 + i, "audited_at": 11.0 + i,
             "audited": True, "passed": True} for i in range(4)]
    assert _caught_by(first, rows, q=Q, hold=HOLD, probation=5) == "probation"
    rows.append({"role": "late", "received_at": 20.0, "audited_at": 21.0,
                 "audited": True, "passed": True})
    assert _caught_by(first, rows, q=Q, hold=HOLD, probation=5) == "no_beacon"


def test_throughput_counts_only_records_audited_on_arrival():
    subs = [_sub(0, HONEST, 0.0, 1000), _sub(1, HONEST, 0.0, 1000), _sub(2, HONEST, 5.0, 500)]
    verdicts = [
        _verdict(0, HONEST, 2.0, passed=True),
        _verdict(1, HONEST, 90.0, passed=True, draw={"drawn": True}),
        _verdict(2, HONEST, 91.0, passed=True, audited=False),
    ]
    out = _audit_throughput(subs, verdicts)
    assert out["audited_submissions"] == 1 and out["audited_later_not_counted"] == 1
    assert out["busy_seconds"] == 2.0 and out["completion_tokens_per_second"] == 500.0


def test_throughput_ignores_a_drawn_records_wait_for_its_beacon():
    subs = [_sub(0, HONEST, 0.0, 1000), _sub(1, HONEST, 1.0, 1000)]
    verdicts = [_verdict(0, HONEST, 2.0, passed=True),
                _verdict(1, HONEST, 20.0, passed=True, draw={"drawn": True})]
    out = _audit_throughput(subs, verdicts)
    assert out["busy_seconds"] == 2.0 and out["audited_later_not_counted"] == 1


def test_throughput_ignores_the_idle_wait_before_a_rescan():
    # Held records audited by a rescan a minute after the auditor went idle.
    subs = [_sub(0, HONEST, 0.0, 1000), _sub(1, HONEST, 1.0, 1000)]
    verdicts = [_verdict(0, HONEST, 2.0, passed=True), _verdict(1, HONEST, 400.0, passed=True)]
    out = _audit_throughput(subs, verdicts)
    assert out["busy_seconds"] == 2.0 and out["audited_submissions"] == 1


def test_route_ok_allows_a_ban_only_where_expected():
    assert _route_ok({"accepted": 3, "miner_banned": 2}, 5, may_be_banned=True)
    assert not _route_ok({"accepted": 3, "miner_banned": 2}, 5, may_be_banned=False)
    assert not _route_ok({"accepted": 4}, 5, may_be_banned=True)


def test_audit_params_from_args_declares_only_what_was_given():
    args = SimpleNamespace(audit_q=0.2, audit_probation=5, audit_hold_seconds=300.0,
                           audit_suspect_seconds=None, audit_ban_after_failures=None)
    assert audit_params_from_args(args) == {
        "audit_q": 0.2, "audit_probation_submissions": 5, "audit_hold_seconds": 300.0}
    assert audit_params_from_args(SimpleNamespace()) == {}
