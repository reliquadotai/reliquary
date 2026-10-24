"""Group shares at the target mix and the per-env limits drawn from them (rl_env_shares)."""
import contextlib
import sqlite3
import threading

import pytest

from reliquary.validator.rl_buffer import TrainingBuffer
from reliquary.validator.rl_env_shares import EnvGroupCaps, group_shares


class Book:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("CREATE TABLE rl_groups(seq INTEGER PRIMARY KEY, environment TEXT, verdict TEXT, "
                        "completion_tokens INTEGER)")

    @contextlib.contextmanager
    def transaction(self):
        with self.lock:
            yield
            self.db.commit()


def groups(book, env, tokens, n, verdict="in_zone"):
    book.db.executemany("INSERT INTO rl_groups(environment, verdict, completion_tokens) VALUES(?,?,?)",
                        [(env, verdict, tokens)] * n)


def test_a_cheap_env_is_many_groups_for_few_tokens():
    book = Book()
    groups(book, "if", 6_000, 10)
    groups(book, "math", 500_000, 10)
    groups(book, "math", 1, 5, verdict="too_hard")                       # out of zone: not counted
    shares, tokens = group_shares(book.db, {"if": 0.01, "math": 0.99})
    assert tokens == {"if": 6_000, "math": 500_000}
    assert shares["if"] == pytest.approx((0.01 / 6e3) / (0.01 / 6e3 + 0.99 / 5e5))   # ~46 % of the groups
    assert sum(shares.values()) == pytest.approx(1.0)


def test_lane_max_and_batch_caps_follow_the_shares():
    book = Book()
    groups(book, "if", 6_000, 10)
    groups(book, "math", 500_000, 10)
    caps = EnvGroupCaps(book, {"if": 0.01, "math": 0.99}, batch_groups=512, lane_batches=2.0, batch_factor=1.5)
    share = caps.shares()["if"]
    assert caps.lane_max()["if"] == pytest.approx(2 * 512 * share, abs=1)
    assert caps.batch_caps(512)["if"] == pytest.approx(1.5 * 512 * share, abs=1)
    assert EnvGroupCaps(book, {"if": 1.0}, batch_groups=4).lane_max() == {}


def test_a_take_hands_out_at_most_an_envs_cap_and_keeps_the_rest_queued():
    from tests.unit.test_rl_async_trainer import Book as TrainerBook, submitted
    from tests.unit.test_substitution_loss import _Model

    model, book = _Model(), TrainerBook()
    buffer = TrainingBuffer(book, batch_caps=lambda limit: {"if": 2})
    for i in range(5):
        buffer.push(submitted(model, f"if{i}", "if", arrived=float(i), n=2))
    buffer.push(submitted(model, "m0", "math", arrived=10.0, n=2))
    taken = buffer.take(4)
    assert [g.lease_id for g in taken] == ["if0", "if1", "m0"]           # short: the trainer waits for more math
    assert buffer.depth() == {"if": 3}
    assert [g.lease_id for g in buffer.take(4)] == ["if2", "if3"]
