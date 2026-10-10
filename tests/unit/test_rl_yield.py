"""The RL session yield ledger: useful = a group of the precommit was trained
or paid as first-scan exploration; faults (the machine's or ours) are neither; stored per hotkey,
aggregated through the keying point; persisted."""
import threading
import time
from types import SimpleNamespace

import pytest

from reliquary.protocol.service_episode import rl_engagement
from reliquary.sandbox.rl_capacity import YieldStats
from reliquary.sandbox.rl_yield import (
    GRADING_POOL,
    NO_VERDICT_STAGES,
    OPEN_REFUSED,
    RUNTIME_EXEC_FAILED,
    VALIDATOR_RESTARTED,
    VALIDATOR_STAGES,
    SessionYieldLedger,
)

A, B = "a" * 64, "b" * 64
HOT = "5Hot"
NOW = 10_000.0


def record(session_id, *, hotkey=HOT, sha=A, window=1, seed=0, issued_at=1_000, state="closed",
           status="withdrawn", fault=None, index=3):
    return SimpleNamespace(session_id=session_id, hotkey=hotkey, engagement=rl_engagement(window, sha, seed),
                           env="reliquary-swe", split="train:rl", index=index, issued_at=issued_at, state=state,
                           closed_status=status, fault=fault)


def ledger(tmp_path, **kwargs):
    return SessionYieldLedger(tmp_path / "rl-yield.sqlite3", clock=lambda: NOW, **kwargs)


def opened(led, n, *, sha=A, hotkey=HOT, window=1, start=0, issued_at=1_000):
    records = [record(f"{sha[:2]}-{hotkey}-{start + i}", hotkey=hotkey, sha=sha, window=window,
                      seed=(start + i) % 128,                  # the engagement seed range is 0..127
                      issued_at=issued_at + start + i) for i in range(n)]
    for each in records:
        led.note_opened(each)
    return records


def settle(led, records, *, sha=A, used=False, aborted=False, resumed=False, now=NOW, group_size=None):
    window = led._windows[sha]
    assert sha in led.precommits_of(window)
    led.decide(sha, [(r.session_id, r.state, r.closed_status, r.fault) for r in records], used=used,
               aborted=aborted, resumed=resumed, now=now, group_size=group_size)
    led.flush(now)


def test_a_trained_or_paid_group_makes_every_session_of_its_precommit_useful(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=True)
    settle(led, records, used=True)
    assert led.stats(HOT) == YieldStats(useful=32, decided=32, useful_24h=32)


def test_a_group_accepted_at_intake_but_neither_trained_nor_paid_is_waste(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=True)
    settle(led, records, used=False)                          # e.g. a forfeited exploration
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (0, 32)


def test_without_a_used_group_every_session_is_waste(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage="episode_transcript")
    settle(led, records)
    stats = led.stats(HOT)
    assert (stats.useful, stats.decided) == (0, 32)


@pytest.mark.parametrize("stage", sorted(VALIDATOR_STAGES))
def test_a_refusal_of_ours_after_the_check_is_neither(tmp_path, stage):
    assert stage in {"episode_persist_failed", "admission_timeout", "admission_worker", "prompt_source"}
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage=stage)
    settle(led, records)
    assert led.stats(HOT) == YieldStats()


def test_the_retryable_refusals_and_a_drain_give_no_verdict():
    from reliquary.protocol.episode_retry import RETRYABLE_STAGES

    assert (RETRYABLE_STAGES - VALIDATOR_STAGES) <= NO_VERDICT_STAGES and "admission_drain" in NO_VERDICT_STAGES
    assert not NO_VERDICT_STAGES & VALIDATOR_STAGES


@pytest.mark.parametrize("stage", sorted(NO_VERDICT_STAGES))
def test_a_refusal_before_the_check_gives_no_verdict(tmp_path, stage):      # a miner provokes these at will
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage=stage)
    assert A not in led._verdicts
    settle(led, records)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (0, 32)


@pytest.mark.parametrize("first, second", [("episode_transcript", "episode_persist_failed"),
                                           ("episode_persist_failed", "episode_transcript")])
def test_a_real_refusal_is_never_cleared_by_one_of_ours(tmp_path, first, second):
    led = ledger(tmp_path)
    records = opened(led, 32)
    for stage in (first, second):
        led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage=stage)
    settle(led, records)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (0, 32)


def test_a_refusal_of_ours_then_a_used_group_is_useful(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage="episode_persist_failed")
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=True)
    settle(led, records, used=True)
    assert led.stats(HOT).useful == 32


@pytest.mark.parametrize("then, decided", [(None, 2), ("refused", 5), ("used", 5)])
def test_a_capacity_refusal_neutralises_only_the_sessions_that_ran_in_a_group_that_could_not_form(
        tmp_path, then, decided):
    led = ledger(tmp_path)
    records = opened(led, 5)
    for r in records[:3]:
        r.state, r.closed_status = "closed_graded", "graded"            # ran: a machine-signed final
    for r in records[3:]:
        r.state, r.closed_status = "lapsed", None                        # issued, never presented: squatted
    led.capacity_refused(hotkey=HOT, precommit_sha256=A)                 # the rest of its group was refused
    led.capacity_refused(hotkey="5Other", precommit_sha256=A)            # another hotkey's: ignored
    if then == "refused":
        led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False,
                          stage="episode_transcript")
    settle(led, records, used=then == "used", group_size=16)
    assert led.stats(HOT).decided == decided                             # (None): 3 neutral, 2 squatted wasted


@pytest.mark.parametrize("status", [None, "open_failed", "withdrawn"])
def test_a_capacity_refusal_never_neutralises_a_session_that_did_not_run(tmp_path, status):
    led = ledger(tmp_path)
    records = opened(led, 4)
    for r in records:
        r.state, r.closed_status = ("lapsed" if status is None else "closed"), status
    led.capacity_refused(hotkey=HOT, precommit_sha256=A)
    settle(led, records, group_size=16)
    assert led.stats(HOT).decided == 4


def test_a_capacity_refusal_after_a_formable_group_is_waste(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 20)
    for r in records[:16]:
        r.state, r.closed_status = "closed_graded", "graded"            # M graded: the miner could have submitted
    led.capacity_refused(hotkey=HOT, precommit_sha256=A)
    settle(led, records, group_size=16)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (0, 20)


def test_capacity_verdicts_are_capped_per_hotkey_and_window(tmp_path):
    led = ledger(tmp_path)
    led.allow_capacity_verdicts(lambda hotkey: 1)
    opened(led, 2, sha=A)
    opened(led, 2, sha=B)
    led.capacity_refused(hotkey=HOT, precommit_sha256=A)
    led.capacity_refused(hotkey=HOT, precommit_sha256=A)                 # the same precommit again: no new one
    led.capacity_refused(hotkey=HOT, precommit_sha256=B)
    assert led._verdicts[A][0] == "capacity" and B not in led._verdicts


@pytest.mark.parametrize("refused, decided", [(False, 0), (True, 32)])
def test_the_window_a_restart_interrupted_is_neither_unless_really_refused(tmp_path, refused, decided):
    led = ledger(tmp_path)
    records = opened(led, 32)
    if refused:
        led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False,
                          stage="episode_transcript")
    settle(led, records, resumed=True)
    assert led.stats(HOT).decided == decided


@pytest.mark.parametrize("used, useful", [(True, 32), (False, 0)])
def test_box_failed_counts_like_any_other_close(tmp_path, used, useful):
    led = ledger(tmp_path)
    records = opened(led, 32)
    for r in records:
        r.closed_status = "box_failed"                      # the agent's box, the agent's doing
    settle(led, records, used=used)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (useful, 32)


def test_faults_are_neither_useful_nor_wasted(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 40)
    for r in records[0:4]:
        r.state, r.closed_status = "aborted", "aborted"          # machine-signed abort
    for r in records[4:8]:
        r.state, r.closed_status = "voided", "machine_drained"   # a drain
    for r in records[8:12]:
        r.state, r.closed_status, r.fault = "closed_graded", "graded", RUNTIME_EXEC_FAILED
    for r in records[12:16]:
        r.closed_status, r.fault = "open_failed", f"{OPEN_REFUSED}:open_failed"   # gateway-signed receipt
    for r in records[16:20]:
        r.closed_status, r.fault = "open_failed", VALIDATOR_RESTARTED             # the token we lost
    for r in records[20:24]:
        r.closed_status = "open_failed"                                          # unsigned: the miner's word
    for r in records[24:28]:
        r.state, r.closed_status, r.fault = "aborted", "aborted", GRADING_POOL   # others held the grading pool
    settle(led, records, used=False)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (0, 16)


def test_an_aborted_window_or_a_hand_back_makes_the_whole_precommit_neither(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    settle(led, records, aborted=True)
    other = opened(led, 32, sha=B)
    led.group_settled(hotkey=HOT, precommit_sha256=B, session_ids=(), accepted=True)
    led.group_handed_back(hotkey=HOT, precommit_sha256=B, session_ids=(), stage="validator_lost")
    settle(led, other, sha=B)
    assert led.stats(HOT) == YieldStats()


def test_pending_windows_precommits_and_their_task(tmp_path):
    led = ledger(tmp_path)
    opened(led, 2, window=5)
    opened(led, 2, window=6, sha=B)
    assert led.pending_windows() == (5, 6)
    assert led.precommits_of(5) == (A,) and led.precommits_of(7) == ()
    assert led.task_of(A) == (HOT, "reliquary-swe", "train:rl", 3) and led.task_of("c" * 64) is None


def test_a_verdict_naming_another_hotkeys_precommit_is_ignored(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    led.group_handed_back(hotkey="5Other", precommit_sha256=A, session_ids=(), stage="validator_lost")
    led.group_settled(hotkey="5Other", precommit_sha256=A, session_ids=(), accepted=False,
                      stage="episode_persist_failed")
    settle(led, records)
    assert led.stats(HOT).decided == 32                      # neither spoiled nor laundered


def test_the_yield_counts_the_last_200_decided_sessions(tmp_path):
    led = ledger(tmp_path)
    old = opened(led, 200, sha=B, issued_at=1_000)
    settle(led, old, sha=B)
    new = opened(led, 200, sha=A, issued_at=5_000)
    settle(led, new, used=True)
    assert (led.stats(HOT).useful, led.stats(HOT).decided) == (200, 200)


def test_the_useful_volume_is_counted_in_held_seconds(tmp_path):
    from reliquary.sandbox.rl_yield import held_seconds

    led = ledger(tmp_path)
    records = opened(led, 4)
    rows = [(r.session_id, r.state, r.closed_status, r.fault, 1_800.0) for r in records]
    led.decide(A, rows, used=True, aborted=False, now=NOW)
    led.flush(NOW)
    assert led.stats(HOT).useful_s_24h == 4 * 1_800.0 and led.stats(HOT).useful_24h == 4
    assert held_seconds(SimpleNamespace(issued_at=100, closed_at=400), 1_000) == 300.0
    assert held_seconds(SimpleNamespace(issued_at=100, closed_at=None), 1_000) == 900.0
    late = SimpleNamespace(issued_at=100, ended_at=400, closed_at=5_000, expires_at=10_000)
    assert held_seconds(late, 9_000) == 300.0                         # the submission's delay counts nothing (m2)
    assert held_seconds(SimpleNamespace(issued_at=100, closed_at=None, expires_at=700), 9_000) == 600.0


def test_useful_24h_forgets_what_was_decided_a_day_ago(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)
    settle(led, records, used=True, now=NOW)
    led.flush(NOW + 86_401)
    assert led.stats(HOT) == YieldStats(useful=32, decided=32, useful_24h=0)


def test_every_quantity_is_the_hotkeys_and_a_flush_publishes_a_snapshot(tmp_path):
    led = ledger(tmp_path)
    before = led.version
    settle(led, opened(led, 16, hotkey="5H1", sha=A), sha=A)
    settle(led, opened(led, 16, hotkey="5H2", sha=B), sha=B, used=True)
    assert (led.stats("5H1").decided, led.stats("5H1").useful) == (16, 0)       # never another hotkey's rows
    assert led.stats("5Lone") == YieldStats()
    snapshot = led.snapshot()
    assert set(snapshot) == {"5H1", "5H2"} and snapshot["5H2"] == YieldStats(useful=16, decided=16, useful_24h=16)
    assert led.version > before
    led.flush(NOW)
    assert led.snapshot() is not snapshot and led.snapshot() == snapshot        # replaced whole, never mutated


def test_the_window_accepting_groups_at_the_last_flush_survives_a_restart(tmp_path):
    led = ledger(tmp_path)
    assert led.accepting_at_last_flush() is None
    led.note_accepting(7, NOW)
    led.flush(NOW)
    led.close()
    assert ledger(tmp_path).accepting_at_last_flush() == 7
    again = ledger(tmp_path)
    again.note_accepting(None, NOW + 10)                                         # between windows
    again.flush(NOW + 10)
    again.close()
    assert ledger(tmp_path).accepting_at_last_flush() is None


def test_the_ledger_survives_a_restart(tmp_path):
    led = ledger(tmp_path)
    settle(led, opened(led, 32, sha=B), sha=B)
    pending = opened(led, 4, sha=A, window=3)
    led.group_settled(hotkey=HOT, precommit_sha256=A, session_ids=(), accepted=False, stage="episode_persist_failed")
    led.flush(NOW)
    led.close()
    again = ledger(tmp_path)
    assert again.stats(HOT).decided == 32
    assert again.pending_windows() == (3,) and again.task_of(A)[0] == HOT
    settle(again, pending)
    assert again.stats(HOT).decided == 32                    # the refusal of ours was reloaded: neither


def test_a_restart_before_the_flush_keeps_a_trained_precommit_useful(tmp_path):
    led = ledger(tmp_path)
    records = opened(led, 32)                                # never flushed: the crash loses the journal
    again = ledger(tmp_path)
    assert again.pending_windows() == ()
    again.reconcile(records)                                 # the book's restored records
    settle(again, records, used=True)                        # the runtime's settlement says: trained
    assert again.stats(HOT).useful == 32


def test_a_failed_flush_keeps_its_journal(tmp_path, monkeypatch):
    led = ledger(tmp_path)
    opened(led, 2)
    real = led._apply
    calls = {"n": 0}

    def flaky(entry):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk")
        return real(entry)

    monkeypatch.setattr(led, "_apply", flaky)
    with pytest.raises(OSError):
        led.flush(NOW)
    assert led.flush(NOW) == 2


def test_two_flushes_never_run_at_once_and_close_waits_for_one(tmp_path, monkeypatch):
    led = ledger(tmp_path)
    opened(led, 4)
    real, inside, overlaps, started = led._apply, [0], [], threading.Event()

    def slow(entry):
        inside[0] += 1
        overlaps.append(inside[0])
        started.set()
        time.sleep(0.01)
        real(entry)
        inside[0] -= 1

    monkeypatch.setattr(led, "_apply", slow)
    threads = [threading.Thread(target=led.flush, args=(NOW,)) for _ in range(2)]
    for t in threads:
        t.start()
    assert started.wait(5.0)                                 # a flush is in progress
    led.close()                                              # waits for it, then closes
    for t in threads:
        t.join()
    assert overlaps and max(overlaps) == 1
