"""Trainer worker loop: cadence, adaptive publish, single-writer guard."""

import pytest

from reliquary.trainer.worker import TrainerLockLost, TrainerWorker
from reliquary.validator.training import TrainingStepSkipped


class _Env:
    def __init__(self, entries):
        # entries: {window_n: ("payload", decoded) | ("tombstone", dict)}
        self.entries = entries
        self.trained = []
        self.published = []
        self.head = "rev-0"

    def journal_next(self, cursor, *, stride):
        return self.entries.get(cursor + stride)

    def train(self, decoded):
        self.trained.append(decoded)
        return True

    def publish(self, reason):
        rev = f"rev-{len(self.published) + 1}"
        self.published.append(reason)
        self.head = rev
        return rev


class _Decoded:
    def __init__(self, n, quarantined=False):
        self.window_start = n
        self.window_quarantine = {"quarantined": quarantined}


def _worker(env, **kw):
    journal = type("J", (), {"next_entry": staticmethod(env.journal_next)})()
    return TrainerWorker(
        journal=journal,
        train_fn=env.train,
        publish_fn=env.publish,
        head_revision_fn=lambda: env.head,
        cursor=100,
        stride=1,
        publish_every=kw.pop("publish_every", 2),
        last_published_revision="rev-0",
        **kw,
    )


def test_waits_without_advancing():
    env = _Env({})
    w = _worker(env)
    assert w.run_once() == "waited"
    assert w.cursor == 100


def test_trains_in_order_and_publishes_on_cadence():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=2)
    assert w.run_once() == "trained" and w.cursor == 101
    assert w.run_once() == "trained" and w.cursor == 102
    assert w.run_once() == "published"
    assert env.published == ["cadence"]
    assert w.trained_since_publish == 0
    assert w.last_published_revision == "rev-1"


def test_tombstone_advances_and_counts():
    env = _Env({101: ("tombstone", {"failure_stage": "s"})})
    w = _worker(env)
    assert w.run_once() == "tombstone"
    assert w.cursor == 101 and env.trained == []
    assert w.tombstones_seen == 1


def test_quarantined_window_advances_without_training():
    env = _Env({101: ("payload", _Decoded(101, quarantined=True))})
    w = _worker(env)
    assert w.run_once() == "quarantined"
    assert w.cursor == 101 and env.trained == []


def test_untrained_step_does_not_count_toward_cadence():
    env = _Env({101: ("payload", _Decoded(101))})
    w = _worker(env, publish_every=1)

    def no_train(decoded):
        return False  # accumulator not ready yet

    w._train_fn = no_train
    assert w.run_once() == "trained"
    assert w.trained_since_publish == 0
    assert w.run_once() == "waited"  # no publish due


def test_policy_ratio_drift_triggers_adaptive_publish():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=10)
    assert w.run_once() == "trained"
    assert w.trained_since_publish == 1

    def drift(decoded):
        raise TrainingStepSkipped("policy_ratio_drift", 0.0)

    w._train_fn = drift
    w.run_once()  # window consumed, drift flagged
    assert w.cursor == 102
    assert w.adaptive_publication_pending
    assert w.run_once() == "published"
    assert env.published == ["adaptive_policy_ratio_drift"]


def test_other_skip_reasons_do_not_trigger_adaptive():
    env = _Env({101: ("payload", _Decoded(101))})
    w = _worker(env, publish_every=10)

    def gate(decoded):
        raise TrainingStepSkipped("grad_norm", 123.0)

    w._train_fn = gate
    w.run_once()
    assert not w.adaptive_publication_pending


def test_lock_lost_on_foreign_head():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=2)
    w.run_once()
    w.run_once()
    env.head = "someone-else"
    with pytest.raises(TrainerLockLost):
        w.run_once()


def test_shadow_mode_never_publishes():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=2, shadow=True)
    w.run_once()
    w.run_once()
    assert w.run_once() == "published"  # counter reset, but...
    assert env.published == []          # ...nothing actually published


# ---------------- v6.1: trainer-paced picks step cursor ----------------


def test_writes_cursor_after_trained_payload():
    env = _Env({101: ("payload", _Decoded(101))})
    written = []
    w = _worker(env, cursor_writer=written.append)
    assert w.run_once() == "trained"
    assert written == [101]


def test_writes_cursor_after_tombstone():
    """Tombstones advance the walk but are not training steps. They MUST
    still bump the cursor: the validator paces on consumption of journal
    keys, and a tombstoned key will never be trained -- not bumping here
    would stall the pacer forever on every skipped window."""
    env = _Env({101: ("tombstone", {"failure_stage": "s"})})
    written = []
    w = _worker(env, cursor_writer=written.append)
    assert w.run_once() == "tombstone"
    assert written == [101]


def test_writes_cursor_after_quarantined_window():
    env = _Env({101: ("payload", _Decoded(101, quarantined=True))})
    written = []
    w = _worker(env, cursor_writer=written.append)
    assert w.run_once() == "quarantined"
    assert written == [101]


def test_cursor_write_failure_does_not_fail_training_step():
    env = _Env({101: ("payload", _Decoded(101))})

    def boom(journal_key):
        raise RuntimeError("R2 down")

    w = _worker(env, cursor_writer=boom)
    assert w.run_once() == "trained"
    assert w.cursor == 101
    assert env.trained  # training happened despite the telemetry failure


def test_no_cursor_writer_configured_is_a_noop():
    env = _Env({101: ("payload", _Decoded(101))})
    w = _worker(env)  # default cursor_writer=None
    assert w.run_once() == "trained"
    assert w.cursor == 101


# ------------------------------------------------------- catch-up (R42)

def _worker_over_journal(env, store, cursor=100, **kw):
    from reliquary.trainer.journal import WindowJournal
    from reliquary.trainer.worker import TrainerWorker
    return TrainerWorker(
        journal=WindowJournal(store.get),
        train_fn=env.train,
        publish_fn=env.publish,
        head_revision_fn=lambda: env.head,
        cursor=cursor,
        stride=1,
        publish_every=kw.pop("publish_every", 16),
        last_published_revision="rev-0",
        **kw,
    )


def _payload_store(first, last):
    from reliquary.infrastructure.training_payload_queue import payload_key
    from reliquary.shared.training_payload import encode_training_payload
    store = {}
    for n in range(first, last + 1):
        store[payload_key(n)] = encode_training_payload(
            {},
            window_start=n,
            checkpoint_revision="rev-0",
            env_order=[],
            window_quarantine={"quarantined": False, "reasons": []},
        )
    return store


def test_a_deep_backlog_is_skipped_down_to_the_catchup_cap():
    """The restart incident: the trainer was down for 40 keys' worth of
    emissions; it must NOT grind an optimizer step through stale data --
    it jumps to the 16 most recent, counted and loud."""
    env = _Env({})
    store = _payload_store(101, 140)
    w = _worker_over_journal(env, store, cursor=100, max_catchup=16)
    assert w.skip_stale_backlog() == 24
    assert w.cursor == 124
    assert w.snapshot()["stale_skipped"] == 24
    assert w.trained_since_publish == 0


def test_a_shallow_backlog_is_not_touched():
    env = _Env({})
    store = _payload_store(101, 110)
    w = _worker_over_journal(env, store, cursor=100, max_catchup=16)
    assert w.skip_stale_backlog() == 0
    assert w.cursor == 100


def test_a_zero_cap_disables_the_skip_entirely():
    env = _Env({})
    store = _payload_store(101, 140)
    w = _worker_over_journal(env, store, cursor=100, max_catchup=0)
    assert w.skip_stale_backlog() == 0
    assert w.cursor == 100


def test_the_jump_writes_the_cursor_once_so_the_validator_unsticks():
    """The one cursor write after the jump is what releases the
    validator's pacing and rotation gates without waiting for their
    backstops -- the skip must publish where training will resume."""
    env = _Env({})
    store = _payload_store(101, 140)
    written = []
    w = _worker_over_journal(
        env, store, cursor=100, max_catchup=16,
        cursor_writer=written.append,
    )
    w.skip_stale_backlog()
    assert written == [124]


def test_training_resumes_normally_after_the_skip():
    env = _Env({})
    store = _payload_store(101, 120)
    w = _worker_over_journal(env, store, cursor=100, max_catchup=16)
    assert w.skip_stale_backlog() == 4
    assert w.run_once() == "trained"
    assert w.cursor == 105
