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


def test_a_take_spills_a_capped_envs_extra_groups_rather_than_come_back_short():
    from tests.unit.test_rl_async_trainer import Book as TrainerBook, submitted
    from tests.unit.test_substitution_loss import _Model

    model, book = _Model(), TrainerBook()
    buffer = TrainingBuffer(book, batch_caps=lambda limit: {"if": 2})
    for i in range(5):
        buffer.push(submitted(model, f"if{i}", "if", arrived=float(i), n=2))
    buffer.push(submitted(model, "m0", "math", arrived=10.0, n=2))
    taken = buffer.take(4)
    assert [g.lease_id for g in taken] == ["if0", "if1", "m0", "if2"]     # the cap, then the spill: never short
    assert buffer.depth() == {"if": 2}


def test_a_rich_queue_gives_each_env_its_cap():
    from tests.unit.test_rl_async_trainer import Book as TrainerBook, submitted
    from tests.unit.test_substitution_loss import _Model

    model, book = _Model(), TrainerBook()
    buffer = TrainingBuffer(book, batch_caps=lambda limit: {"code": 1, "math": 3})
    for i in range(6):                                   # a wave of code arrives first
        buffer.push(submitted(model, f"c{i}", "code", arrived=float(i), n=2))
    for i in range(4):
        buffer.push(submitted(model, f"m{i}", "math", arrived=10.0 + i, n=2))
    assert [g.lease_id for g in buffer.take(4)] == ["c0", "m0", "m1", "m2"]
    assert buffer.depth() == {"code": 5, "math": 1}


def test_a_group_about_to_pass_the_max_lag_goes_first_above_its_cap():
    from types import SimpleNamespace

    from tests.unit.test_rl_async_trainer import Book as TrainerBook, submitted
    from tests.unit.test_substitution_loss import _Model

    model, book = _Model(), TrainerBook(latest=20)
    book.config = SimpleNamespace(max_lag=16)            # dropped below 4 now, below 5 by the prefetched step
    buffer = TrainingBuffer(book, batch_caps=lambda limit: {"code": 1, "math": 2})
    buffer.push(submitted(model, "c-new", "code", k=19, arrived=0.0, n=2))
    buffer.push(submitted(model, "m0", "math", k=19, arrived=1.0, n=2))
    buffer.push(submitted(model, "m1", "math", k=19, arrived=2.0, n=2))
    buffer.push(submitted(model, "c-old", "code", k=5, arrived=3.0, n=2))
    buffer.push(submitted(model, "c-older", "code", k=4, arrived=4.0, n=2))
    assert [g.lease_id for g in buffer.take(3)] == ["c-older", "c-old", "m0"]


def test_mixed_take_never_returns_short_when_enough_is_queued():
    import random

    from reliquary.validator.rl_buffer import mixed_take

    rng = random.Random(3)
    for _ in range(200):
        rows = [(f"g{i}", rng.choice("abc"), rng.randint(0, 9)) for i in range(rng.randint(0, 40))]
        caps = {env: rng.randint(1, 5) for env in "abc" if rng.random() < 0.8}
        limit = rng.randint(1, 30)
        chosen = mixed_take(rows, caps, limit, urgent_below=rng.choice([None, 2]))
        assert len(chosen) == min(limit, len(rows)) and len(set(chosen)) == len(chosen)
        assert set(chosen) <= {r[0] for r in rows}


def test_suspects_still_come_one_per_take_after_the_fresh_groups():
    from tests.unit.test_rl_async_trainer import Book as TrainerBook, submitted
    from tests.unit.test_substitution_loss import _Model
    from reliquary.validator.rl_buffer import SUSPECT_ATTEMPTS

    model, book = _Model(), TrainerBook()
    buffer = TrainingBuffer(book, batch_caps=lambda limit: {"if": 1})
    for i in range(3):
        buffer.push(submitted(model, f"s{i}", "if", arrived=float(i), n=2))
    book.db.execute("UPDATE rl_buffer SET attempts=?", (SUSPECT_ATTEMPTS,))
    buffer.push(submitted(model, "f0", "if", arrived=9.0, n=2))
    assert [g.lease_id for g in buffer.take(4)] == ["f0", "s0"]
