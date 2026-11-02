"""Running cooldowns follow the env's current cooldown (LeaseBook.revise_cooldowns): shortened, never lengthened,
published to the miners' prompt table without a new scan."""
from __future__ import annotations

import random
import sqlite3

from reliquary.miner.rl_async import PromptPlanner
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import (
    BLACKLISTED, COOLING, IN_ZONE, TOO_EASY, LeaseBook, LeaseBookConfig, LeaseRequest,
)

ENV, OTHER_ENV = "math", "code"


def book_at(tmp_path, cooldown_of, clock=lambda: 1.8e9, name="leases.sqlite3"):
    cfg = LeaseBookConfig(expiry_s={ENV: 600.0, OTHER_ENV: 600.0}, cooldown_of=cooldown_of)
    return LeaseBook(tmp_path / name, cfg, quota=QuotaController(QuotaConfig(initial=64, ceiling=4096)),
                     clock=clock, rng=random.Random(7))


def ask(prompts, k, env=ENV):
    return LeaseRequest(hotkey="A", environment=env, prompt_ids=tuple(prompts), checkpoint=k)


def until(book, prompt, env=ENV):
    return book.db.execute("SELECT until_k FROM rl_prompt_state WHERE environment=? AND prompt_idx=?",
                           (env, prompt)).fetchone()[0]


IN = [1.0] * 8 + [0.0] * 8


def test_a_shorter_env_cooldown_shortens_the_running_ones_at_the_next_checkpoint(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, IN, IN_ZONE)
    assert until(book, 1) == 30
    live[ENV] = 3
    book.adopt_checkpoint(21)
    assert until(book, 1) == 23                          # counted from where it started cooling, 20
    assert book.grant([ask([1], 21)])[0].refused == {1: COOLING}
    book.adopt_checkpoint(23)
    assert book.grant([ask([1], 23)])[0].leases


def test_a_longer_env_cooldown_never_lengthens_a_running_one(tmp_path):
    live = {ENV: 3, OTHER_ENV: 3}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, IN, IN_ZONE)
    live[ENV] = 50
    book.adopt_checkpoint(21)
    assert until(book, 1) == 23


def test_a_lagged_outcome_still_counts_from_the_latest_checkpoint(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 12, IN, IN_ZONE)         # leased at 12, graded at 20: it starts at 20
    live[ENV] = 2
    book.adopt_checkpoint(21)
    assert until(book, 1) == 22


def test_blacklisted_prompts_and_other_envs_are_left_alone(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, [1.0] * 16, TOO_EASY)
    book.record_outcome(ENV, 1, 0, [1.0] * 16, TOO_EASY)
    book.record_outcome(OTHER_ENV, 2, 0, IN, IN_ZONE)
    live[ENV] = 1
    book.adopt_checkpoint(1)
    assert book.grant([ask([1], 1)])[0].refused == {1: BLACKLISTED}
    assert until(book, 2, OTHER_ENV) == 10


def test_an_env_whose_cooldown_cannot_be_read_does_not_stop_the_checkpoint(tmp_path):
    live = {ENV: 10}                                     # OTHER_ENV has no value: cooldown_of raises KeyError
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, IN, IN_ZONE)
    live[ENV] = 4
    book.adopt_checkpoint(1)
    assert book.latest_checkpoint == 1 and until(book, 1) == 4


def test_a_miner_sees_the_shorter_cooldown_without_a_new_scan(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, IN, IN_ZONE)
    planner = PromptPlanner(ENV, (0, 100))
    planner.apply(book.prompt_state(environment=ENV))
    assert not planner.leasable(1)
    live[ENV] = 3
    book.adopt_checkpoint(23)                            # the env is dry: no scan follows
    page = book.prompt_state(planner.cursor, environment=ENV)
    assert page["scans"] == [] and page["prompts"] == [{"environment": ENV, "prompt_idx": 1, "state": "cooldown",
                                                         "until_k": 23}]
    planner.apply(page)
    assert planner.leasable(1)
    assert book.prompt_state(planner.cursor, environment=ENV)["prompts"] == []   # delivered once
    later = book.record_outcome(ENV, 2, 23, IN, IN_ZONE)
    assert later.seq > planner.cursor                    # a later scan is never hidden behind the cursor
    assert [s["prompt_idx"] for s in book.prompt_state(planner.cursor, environment=ENV)["scans"]] == [2]


def test_a_miner_that_missed_a_revision_gets_it_at_the_next_checkpoint(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, IN, IN_ZONE)
    live[ENV] = 5
    book.adopt_checkpoint(21)                            # 20 + 5 = 25
    late = PromptPlanner(ENV, (0, 100))                  # bootstrapped past the revision (from the scan log)
    late.apply({"latest": 21, "cursor": book.prompt_state(environment=ENV)["cursor"], "scans": [],
                "prompts": [{"environment": ENV, "prompt_idx": 1, "state": "cooldown", "until_k": 30}]})
    book.adopt_checkpoint(22)                            # still running: published again
    late.apply(book.prompt_state(late.cursor, environment=ENV))
    assert late.records[1].until_k == 25


def test_a_new_scan_ends_the_revision(tmp_path):
    live = {ENV: 10, OTHER_ENV: 10}
    book = book_at(tmp_path, live.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, IN, IN_ZONE)
    live[ENV] = 5
    book.adopt_checkpoint(25)
    book.record_outcome(ENV, 1, 25, IN, IN_ZONE)         # leased again once free: a new cooldown from 25
    assert until(book, 1) == 30
    cursor = book.prompt_state(environment=ENV)["cursor"]
    book.adopt_checkpoint(26)
    assert book.prompt_state(cursor, environment=ENV)["prompts"] == []


def test_a_book_from_before_the_revision_learns_where_each_cooldown_started(tmp_path):
    live = {ENV: 30, OTHER_ENV: 30}
    now = [1.8e9]
    book = book_at(tmp_path, live.__getitem__, clock=lambda: now[0])
    book.adopt_checkpoint(5)
    now[0] += 60
    book.record_outcome(ENV, 1, 5, IN, IN_ZONE)          # starts at 5: 35
    now[0] += 60
    book.adopt_checkpoint(9)
    now[0] += 60
    book.record_outcome(ENV, 2, 3, IN, IN_ZONE)          # leased at 3, graded while 9 was the latest: 39
    book.close()
    db = sqlite3.connect(tmp_path / "leases.sqlite3")     # the live run's file: no start_k, no revised flag
    db.execute("ALTER TABLE rl_prompt_state DROP COLUMN start_k")
    db.execute("ALTER TABLE rl_prompt_state DROP COLUMN revised")
    db.commit()
    db.close()
    live[ENV] = 4
    book = book_at(tmp_path, live.__getitem__, clock=lambda: now[0])
    book.revise_cooldowns()
    assert (until(book, 1), until(book, 2)) == (9, 13)
