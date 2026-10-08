# tests/unit/test_service_run_log.py
import sqlite3

import pytest

from reliquary.constants import M_ROLLOUTS

from reliquary.services.run_log import NotAnObservation, Observation, RunObservationLog

ORDER = "a" * 64


def obs(*, env="reliquary_dapo_math_v1", prompt=7, group="g1", window=1, checkpoint_n=1,
        rewards=(10000,) * (M_ROLLOUTS // 2) + (0,) * (M_ROLLOUTS - M_ROLLOUTS // 2), lane="training", hotkey="hk-1"):
    return Observation(environment=env, dataset_id=f"{env}-train", prompt_idx=prompt, group_id=group,
                       window=window, checkpoint_n=checkpoint_n, checkpoint_revision="c" * 40,
                       observed_at=100.0 + window, rewards_bps=tuple(rewards), lane=lane,
                       candidate={"pool_sha256": "p" * 64, "candidate_id": 1}, hotkey=hotkey, token_count=1234)


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
    assert first["verdict"] == "in-zone" and first["candidate"]["candidate_id"] == 1
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


def test_concurrent_first_observations_yield_exactly_one_first(tmp_path):
    import threading
    db = sqlite3.connect(tmp_path / "c.sqlite3", check_same_thread=False)
    shared = RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
    lock = threading.Lock()
    results = []

    def work(i):
        with lock:
            with db:
                results.append(shared.record(obs(group=f"g{i}", hotkey=f"hk-{i}"), status="proven", proof="proven"))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r.first_scan for r in results) == 1 and all(r.inserted for r in results)


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


def test_same_miner_retry_is_idempotent_with_no_integrity_error(log):
    with log.db:
        a = log.record(obs(hotkey="hk-a"), status="proven", proof="proven")
        again = log.record(obs(hotkey="hk-a"), status="proven", proof="proven")
    assert not again.inserted and again.first_scan and again.observation_id == a.observation_id
    assert len(log.events()) == 1


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
                       "candidate": {"pool_sha256": "p" * 64, "candidate_id": 1, "hotkey": "SECRET-HOTKEY",
                                     "tokens": [1, 2, 3], "token_ids": [4]}})
    with log.db:
        log.record(o, status="proven", proof="proven")
    raw = log.db.execute("SELECT group_concat(payload) FROM run_events").fetchone()[0]
    assert "SECRET-HOTKEY" not in raw and "tokens" not in raw and "token_ids" not in raw
    assert log.events()[0][1]["candidate"] == {"pool_sha256": "p" * 64, "candidate_id": 1}


def test_settle_is_idempotent_and_refuses_another_order(log, tmp_path):
    with log.db:
        r = log.record(obs(), status="proven", proof="proven")
        log.settle(r.observation_id, status="trained", proof="proven", at=1.0)
        log.settle(r.observation_id, status="trained", proof="proven", at=2.0)
        log.settle(r.observation_id, status="exploration_unpaid", proof="proven", at=3.0)
    assert [e["type"] for _, e in log.events()] == ["observation", "settle", "settle"]
    foreign = RunObservationLog(log.db, order_sha256="b" * 64, sigma_min_bps=2400)
    with pytest.raises(ValueError, match="another order"):
        foreign.settle(r.observation_id, status="trained", proof="proven", at=1.0)


def test_too_many_rewards_is_not_an_observation(log):
    with pytest.raises(NotAnObservation):
        with log.db:
            log.record(obs(rewards=(10000, 0) * M_ROLLOUTS), status="proven", proof="proven")
