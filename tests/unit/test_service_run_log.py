# tests/unit/test_service_run_log.py
import sqlite3
from dataclasses import replace

import pytest

from reliquary.constants import M_ROLLOUTS

from reliquary.services.run_log import NotAnObservation, Observation, RunObservationLog

ORDER = "a" * 64
SEEDS = list(range(1, 2 * M_ROLLOUTS, 2))        # any M distinct seeds of the 2M pool, ascending


def obs(*, env="reliquary_dapo_math_v1", prompt=7, group="g1", window=1, checkpoint_n=1,
        rewards=(10000,) * (M_ROLLOUTS // 2) + (0,) * (M_ROLLOUTS - M_ROLLOUTS // 2), lane="training", hotkey="hk-1",
        seeds=SEEDS):
    return Observation(environment=env, dataset_id=f"{env}-train", prompt_idx=prompt, group_id=group,
                       window=window, checkpoint_n=checkpoint_n, checkpoint_revision="c" * 40,
                       observed_at=100.0 + window, rewards_bps=tuple(rewards), lane=lane,
                       candidate={"pool_sha256": "ab" * 32, "seeds": seeds}, hotkey=hotkey, token_count=1234)


@pytest.fixture
def log(tmp_path):
    db = sqlite3.connect(tmp_path / "log.sqlite3")
    return RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)


def test_first_scan_survives_checkpoint_changes(log):
    with log.db:
        first = log.record(obs(checkpoint_n=1), status="proven", proof="proven")
        later = log.record(obs(group="g2", window=9, checkpoint_n=5), status="proven", proof="proven")
    assert first.first_scan and first.inserted and first.category == "in-zone"
    assert not later.first_scan and later.inserted
    assert log.is_scanned("reliquary_dapo_math_v1", 7)


def test_same_identity_is_idempotent_and_conflict_raises(log):
    with log.db:
        a = log.record(obs(), status="proven", proof="proven")
        b = log.record(obs(), status="proven", proof="proven")
    assert b.observation_id == a.observation_id and not b.inserted
    with pytest.raises(ValueError, match="different evidence"):
        with log.db:
            log.record(obs(rewards=(0,) * M_ROLLOUTS), status="proven", proof="proven")


def test_incomplete_or_ungraded_groups_are_not_observations(log):
    with pytest.raises(NotAnObservation):
        with log.db:
            log.record(obs(rewards=(0,) * (M_ROLLOUTS - 1)), status="proven", proof="proven")


def test_public_events_carry_no_hotkey_or_tokens(log):
    with log.db:
        result = log.record(obs(), status="proven", proof="proven")
        log.settle(result.observation_id, status="trained", proof="proven", at=200.0)
    events = [event for _, event in log.events()]
    assert [e["type"] for e in events] == ["observation", "settle"]
    for event in events:
        assert "hotkey" not in event and "token_count" not in event and "tokens" not in event
    first = events[0]
    assert first["env"] == "reliquary_dapo_math_v1" and first["prompt_idx"] == 7
    assert first["checkpoint_n"] == 1 and first["window"] == 1 and len(first["rewards_bps"]) == M_ROLLOUTS
    assert first["verdict"] == "in-zone" and first["candidate"] == {"pool_sha256": "ab" * 32, "seeds": SEEDS}
    assert log.admin_events()[0][1]["hotkey"] == "hk-1"


def test_released_first_scan_reopens_the_prompt(log):
    with log.db:
        result = log.record(obs(rewards=(0,) * M_ROLLOUTS, lane="exploration"), status="exploration_pending", proof="pending")
        assert log.release_first_scan(result.observation_id)
    assert not log.is_scanned("reliquary_dapo_math_v1", 7)
    with log.db:
        again = log.record(obs(group="g9", rewards=(0,) * M_ROLLOUTS, lane="exploration", hotkey="hk-2"),
                           status="exploration_pending", proof="pending")
    assert again.first_scan


def test_first_scan_stats_count_only_first_scans(log):
    with log.db:
        log.record(obs(prompt=1), status="proven", proof="proven")
        log.record(obs(prompt=1, group="x", rewards=(0,) * M_ROLLOUTS), status="proven", proof="proven")
        log.record(obs(prompt=2, rewards=(0,) * M_ROLLOUTS, lane="exploration"), status="exploration_pending", proof="pending")
    assert log.first_scan_stats("reliquary_dapo_math_v1") == (2, 1)


def test_first_scan_is_unique_across_connections_racing_on_one_database(tmp_path):
    """Twelve writers, each with its OWN connection to one file, all released at once on the same
    never-scanned prompt: SQLite serialises them and the primary key of ``run_scans`` lets one win."""
    import threading
    path = tmp_path / "c.sqlite3"
    RunObservationLog(sqlite3.connect(path), order_sha256=ORDER, sigma_min_bps=2400).db.close()  # schema + salt
    writers = 12
    barrier = threading.Barrier(writers)
    results, errors = [], []

    def work(i):
        try:
            db = sqlite3.connect(path, timeout=60)
            mine = RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
            barrier.wait()
            with db:
                results.append(mine.record(obs(group=f"g{i}", hotkey=f"hk-{i}"), status="proven", proof="proven"))
            db.close()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(writers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert len({r.observation_id for r in results}) == writers and all(r.inserted for r in results)
    assert sum(r.first_scan for r in results) == 1
    check = sqlite3.connect(path)
    winner = next(r.observation_id for r in results if r.first_scan)
    assert check.execute("SELECT first_id FROM run_scans").fetchall() == [(winner,)]
    assert check.execute("SELECT id FROM run_observations WHERE first_scan=1").fetchall() == [(winner,)]
    assert check.execute("SELECT COUNT(*) FROM run_events").fetchone()[0] == writers


def test_log_and_event_sequence_survive_reopen(tmp_path):
    path = tmp_path / "r.sqlite3"
    db = sqlite3.connect(path)
    first = RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
    with db:
        r = first.record(obs(), status="proven", proof="proven")
        first.settle(r.observation_id, status="trained", proof="proven", at=5.0)
    before = first.events()
    db.close()
    reopened = RunObservationLog(sqlite3.connect(path), order_sha256=ORDER, sigma_min_bps=2400)
    assert reopened.events() == before and reopened.is_scanned("reliquary_dapo_math_v1", 7)
    with reopened.db:
        again = reopened.record(obs(group="g2"), status="proven", proof="proven")
    assert not again.first_scan
    assert reopened.events()[:2] == before and [s for s, _ in reopened.events()] == [1, 2, 3]
    assert reopened.events(after=2)[0][0] == 3


def test_two_miners_on_the_same_never_scanned_prompt_get_two_observations(log):
    with log.db:
        a = log.record(obs(hotkey="hk-a"), status="proven", proof="proven")
        b = log.record(obs(hotkey="hk-b"), status="proven", proof="proven")
    assert a.observation_id != b.observation_id and a.inserted and b.inserted
    assert a.first_scan and not b.first_scan


def test_same_miner_retry_changes_nothing_and_publishes_nothing(log):
    """A retry of the same submission (same identity) is answered from the stored row: no second
    row (the unique id would raise IntegrityError on a plain INSERT), no second event, and the
    retry's status/proof/ts are not written anywhere."""
    with log.db:
        a = log.record(obs(hotkey="hk-a"), status="proven", proof="proven")
    before = log.db.execute("SELECT * FROM run_observations").fetchall(), log.events()
    with log.db:
        again = log.record(obs(hotkey="hk-a"), status="exploration_unpaid", proof="unproven", reason="cap")
    assert not again.inserted and again.first_scan and again.observation_id == a.observation_id
    assert (log.db.execute("SELECT * FROM run_observations").fetchall(), log.events()) == before
    assert len(log.events()) == 1 and log.events()[0][1]["status"] == "proven"
    assert log.db.execute("SELECT COUNT(*) FROM run_scans").fetchone()[0] == 1


def test_run_salt_is_persisted_and_ids_are_stable_across_reopen(tmp_path):
    path = tmp_path / "s.sqlite3"
    first = RunObservationLog(sqlite3.connect(path), order_sha256=ORDER, sigma_min_bps=2400)
    with first.db:
        a = first.record(obs(), status="proven", proof="proven")
    salt = first.db.execute("SELECT value FROM run_meta WHERE key='run_salt'").fetchone()[0]
    first.db.close()
    again = RunObservationLog(sqlite3.connect(path), order_sha256=ORDER, sigma_min_bps=2400)
    assert len(salt) == 32
    with again.db:
        b = again.record(obs(), status="proven", proof="proven")
    assert not b.inserted and b.observation_id == a.observation_id
    other = RunObservationLog(sqlite3.connect(tmp_path / "o.sqlite3"), order_sha256=ORDER, sigma_min_bps=2400)
    with other.db:
        c = other.record(obs(), status="proven", proof="proven")
    assert c.observation_id != a.observation_id  # a different run has a different salt


def test_public_event_cannot_carry_hotkey_or_tokens_even_via_candidate(log):
    o = obs(hotkey="SECRET-HOTKEY")
    o = Observation(**{**{f: getattr(o, f) for f in o.__slots__},
                       "candidate": {"pool_sha256": "ab" * 32, "seeds": tuple(SEEDS), "candidate_id": 1,
                                     "hotkey": "SECRET-HOTKEY",
                                     "tokens": [1, 2, 3], "token_ids": [4]}})
    with log.db:
        log.record(o, status="proven", proof="proven")
    raw = log.db.execute("SELECT group_concat(payload) FROM run_events").fetchone()[0]
    assert "SECRET-HOTKEY" not in raw and "tokens" not in raw and "token_ids" not in raw
    assert log.events()[0][1]["candidate"] == {"pool_sha256": "ab" * 32, "seeds": SEEDS}


def test_public_event_pairs_each_reward_with_its_seed(log):
    rewards = tuple(10000 if i % 3 == 0 else 0 for i in range(M_ROLLOUTS))
    with log.db:
        log.record(obs(rewards=rewards, hotkey="SECRET-HOTKEY"), status="proven", proof="proven")
    event = log.events()[0][1]
    assert event["candidate"]["seeds"] == SEEDS and event["rewards_bps"] == list(rewards)
    assert dict(zip(event["candidate"]["seeds"], event["rewards_bps"])) == dict(zip(SEEDS, rewards))
    assert "hotkey" not in event and "SECRET-HOTKEY" not in str(event)
    assert set(event["candidate"]) == {"pool_sha256", "seeds"}


@pytest.mark.parametrize("seeds", [
    SEEDS[:-1], SEEDS + [2 * M_ROLLOUTS - 2], SEEDS[::-1], [SEEDS[0]] * M_ROLLOUTS,
    SEEDS[:-1] + [2 * M_ROLLOUTS], [-1] + SEEDS[1:], [True] + SEEDS[1:], [1.0] + SEEDS[1:],
    ["1"] + SEEDS[1:], "seeds", {"0": 1}, None,
])
def test_seeds_that_do_not_pair_with_the_rewards_are_refused(log, seeds):
    with pytest.raises(ValueError):
        with log.db:
            log.record(obs(seeds=seeds), status="proven", proof="proven")
    assert log.events() == []


def test_two_subsets_of_one_prompt_are_two_observations_one_first_scan(log):
    other = list(range(M_ROLLOUTS))
    with log.db:
        a = log.record(obs(group="subset-a"), status="proven", proof="proven")
        b = log.record(obs(group="subset-b", seeds=other), status="proven", proof="proven")
    assert a.observation_id != b.observation_id and a.first_scan and not b.first_scan
    assert [e["candidate"]["seeds"] for _, e in log.events()] == [SEEDS, other]


def test_group_without_a_pool_has_no_candidate(log):
    o = obs()
    o = Observation(**{**{f: getattr(o, f) for f in o.__slots__}, "candidate": None})
    with log.db:
        log.record(o, status="proven", proof="proven")
    assert log.events()[0][1]["candidate"] is None


def test_settle_is_idempotent_and_refuses_another_order(log, tmp_path):
    with log.db:
        r = log.record(obs(), status="proven", proof="proven")
        log.settle(r.observation_id, status="trained", proof="proven", at=1.0)
        log.settle(r.observation_id, status="trained", proof="proven", at=2.0)
        log.settle(r.observation_id, status="exploration_unpaid", proof="proven", at=3.0)
    assert [e["type"] for _, e in log.events()] == ["observation", "settle", "settle"]
    assert [e["ts"] for _, e in log.events()][1:] == [1.0, 3.0]
    foreign = RunObservationLog(log.db, order_sha256="b" * 64, sigma_min_bps=2400)
    with pytest.raises(ValueError, match="another order"):
        foreign.settle(r.observation_id, status="trained", proof="proven", at=1.0)
    with pytest.raises(ValueError, match="unknown"):
        log.settle("f" * 64, status="trained", proof="proven", at=1.0)


def test_settle_dedup_looks_at_the_latest_settlement_only(log):
    with log.db:
        r = log.record(obs(), status="proven", proof="proven")
        other = log.record(obs(prompt=8), status="proven", proof="proven")
        for at, status in ((1.0, "a"), (2.0, "b"), (3.0, "a"), (4.0, "a")):
            log.settle(r.observation_id, status=status, proof="proven", at=at)
        log.settle(other.observation_id, status="a", proof="proven", at=5.0)   # another observation: its own history
        log.settle(r.observation_id, status="a", proof="audited", at=6.0)      # same status, new proof: an event
    mine = [(e["status"], e["proof"], e["ts"]) for _, e in log.events()
            if e["type"] == "settle" and e["id"] == r.observation_id]
    assert mine == [("a", "proven", 1.0), ("b", "proven", 2.0), ("a", "proven", 3.0), ("a", "audited", 6.0)]
    assert sum(e["type"] == "settle" and e["id"] == other.observation_id for _, e in log.events()) == 1


def test_constructor_refuses_an_open_transaction(tmp_path):
    db = sqlite3.connect(tmp_path / "t.sqlite3")
    db.execute("CREATE TABLE t(x)")
    db.execute("INSERT INTO t VALUES(1)")
    assert db.in_transaction
    with pytest.raises(ValueError, match="transaction"):
        RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
    assert db.in_transaction                         # nothing was committed behind the caller's back
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)


def test_reason_is_a_published_field_only_when_given(log):
    with log.db:
        log.record(obs(), status="proven", proof="proven")
        log.record(obs(prompt=8, lane="exploration"), status="exploration_unpaid", proof="unproven", reason="cap")
    plain, refused = [e for _, e in log.events()]
    assert "reason" not in plain and refused["reason"] == "cap" and refused["status"] == "exploration_unpaid"
    for bad in ("", "Cap", "has space", "x" * 41, 5, "hk:5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX"):
        with pytest.raises(ValueError, match="reason"):
            with log.db:
                log.record(obs(prompt=9), status="exploration_unpaid", proof="unproven", reason=bad)
    assert len(log.events()) == 2


def test_too_many_rewards_is_not_an_observation(log):
    with pytest.raises(NotAnObservation):
        with log.db:
            log.record(obs(rewards=(10000, 0) * M_ROLLOUTS), status="proven", proof="proven")


@pytest.mark.parametrize("pool", ["p" * 64, "AB" * 32, "ab" * 31, "ab" * 33, "", None, 5, b"ab" * 32, "ab" * 31 + "a\n"])
def test_a_public_pool_hash_that_is_not_64_lowercase_hex_is_refused_not_published(log, pool):
    candidate = {"pool_sha256": pool, "seeds": SEEDS}
    with pytest.raises(ValueError, match="pool_sha256"):
        with log.db:
            log.record(replace(obs(), candidate=candidate), status="proven", proof="proven")
    assert log.events() == []


def test_a_candidate_without_a_pool_hash_is_refused(log):
    with pytest.raises(ValueError, match="pool_sha256"):
        with log.db:
            log.record(replace(obs(), candidate={"seeds": SEEDS}), status="proven", proof="proven")


# ---- Task 7 review: I1 (a trained prompt is scanned), settle reason ----

ENV = "reliquary_dapo_math_v1"
ZERO = (0,) * M_ROLLOUTS


def holder(log, prompt=7):
    row = log.db.execute("SELECT first_id FROM run_scans WHERE environment=? AND prompt_idx=?", (ENV, prompt)).fetchone()
    return None if row is None else row[0]


def test_release_reseats_the_scan_on_the_earliest_counting_training_observation(log):
    with log.db:
        probe = log.record(obs(rewards=ZERO, lane="exploration", hotkey="a"), status="exploration_pending", proof="pending")
        early = log.record(obs(group="t1", window=2, hotkey="b"), status="proven", proof="proven")
        late = log.record(obs(group="t2", window=3, hotkey="c"), status="proven", proof="proven")
    assert probe.first_scan and not early.first_scan and not late.first_scan
    assert log.first_scan_stats(ENV) == (1, 0)                    # the trained prompt is counted (first scan: out-of-zone)
    with log.db:
        assert log.release_first_scan(probe.observation_id) is True
    assert log.is_scanned(ENV, 7) and holder(log) == early.observation_id
    assert log.first_scan_stats(ENV) == (1, 1)
    flags = {r["id"]: r["first_scan"] for r in log.window_observations(1) + log.window_observations(2) + log.window_observations(3)}
    assert flags == {probe.observation_id: False, early.observation_id: True, late.observation_id: False}
    with log.db:                                                  # idempotent; a counting training scan is never released
        assert log.release_first_scan(probe.observation_id) is False
        assert log.release_first_scan(early.observation_id) is False
        assert log.release_first_scan("f" * 64) is False
    assert holder(log) == early.observation_id
    with log.db:                                                  # an exploration observation on it is not a first scan
        again = log.record(obs(group="g9", rewards=ZERO, lane="exploration", hotkey="d", window=4),
                           status="exploration_unpaid", proof="unproven", reason="already_scanned")
    assert not again.first_scan


def test_only_the_training_observations_of_an_aborted_window_stop_counting_as_scans(log):
    with log.db:
        first = log.record(obs(group="t1", window=1, hotkey="b"), status="proven", proof="proven")
        second = log.record(obs(group="t2", window=2, hotkey="c"), status="proven", proof="proven")
        other = log.record(obs(prompt=8, group="t3", window=1, hotkey="b"), status="proven", proof="proven")
    assert log.trained_prompts(1, ENV) == {7, 8} and log.trained_prompts(2, ENV) == {7} and log.trained_prompts(3, ENV) == set()
    with log.db:                                                  # proven but unpaid: still a scan (public rewards)
        log.settle(first.observation_id, status="proven_unpaid", proof="proven", at=5.0)
    assert holder(log) == first.observation_id and log.trained_prompts(1, ENV) == {7, 8}
    with log.db:
        log.set_window_aborted(1, True)
    assert holder(log) == second.observation_id                   # 7: moved to the next counting one
    assert holder(log, 8) is None and not log.is_scanned(ENV, 8)  # 8: nothing else trained it, free again
    assert log.trained_prompts(1, ENV) == set() and log.trained_prompts(2, ENV) == {7}
    with log.db:
        log.set_window_aborted(1, True)                           # idempotent
        log.set_window_aborted(2, True)
    assert not log.is_scanned(ENV, 7) and holder(log) is None
    with log.db:
        log.set_window_aborted(2, False)                          # not aborted after all: it counts again
    assert holder(log) == second.observation_id and holder(log, 8) is None
    with log.db:                                                  # exploration observations are not concerned
        probe = log.record(obs(prompt=9, rewards=ZERO, lane="exploration", window=3), status="exploration_pending", proof="pending")
        log.set_window_aborted(3, True)
    assert holder(log, 9) == probe.observation_id


def test_settle_publishes_its_reason_and_is_idempotent_on_it(log):
    with log.db:
        r = log.record(obs(rewards=ZERO, lane="exploration"), status="exploration_pending", proof="pending")
        log.settle(r.observation_id, status="exploration_unpaid", proof="unproven", at=1.0)
        log.settle(r.observation_id, status="exploration_unpaid", proof="unproven", at=2.0, reason="trained")
        log.settle(r.observation_id, status="exploration_unpaid", proof="unproven", at=3.0, reason="trained")
    settles = [e for _, e in log.events() if e["type"] == "settle"]
    assert [e.get("reason") for e in settles] == [None, "trained"] and "reason" not in settles[0]
    assert set(settles[1]) == {"type", "id", "window", "status", "proof", "ts", "reason"}
    for bad in ("Trained", "", "x" * 41, 5):
        with pytest.raises(ValueError, match="reason"):
            log.settle(r.observation_id, status="exploration_unpaid", proof="unproven", at=4.0, reason=bad)


def test_a_log_file_written_before_the_untrained_column_is_upgraded(tmp_path):
    db = sqlite3.connect(tmp_path / "old.sqlite3")
    db.executescript("""CREATE TABLE run_observations(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, order_id TEXT NOT NULL,
        environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL, window INTEGER NOT NULL,
        lane TEXT NOT NULL, category TEXT NOT NULL, hotkey TEXT NOT NULL, token_count INTEGER NOT NULL,
        first_scan INTEGER NOT NULL, public TEXT NOT NULL);""")
    old = RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
    with old.db:
        r = old.record(obs(), status="proven", proof="proven")
    assert old.trained_prompts(1, ENV) == {7} and r.first_scan
