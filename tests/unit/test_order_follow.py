"""Mining orders one after another: pick an open order (the loaded model first), fetch a starting order's
model ahead, move on when an order ends, and skip one that fails at once."""

from __future__ import annotations

import sys

from reliquary.miner import order_follow as follow
from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR


def _order(job_id, state, model="org/A", revision="a" * 40):
    return {"job_id": job_id, "state": state, "model": model, "revision": revision}


A, B = f"org/A@{'a' * 40}", f"org/B@{'b' * 40}"


def test_an_open_order_on_the_loaded_model_comes_first():
    orders = [_order("x", "open", "org/A"), _order("y", "open", "org/B", "b" * 40)]
    assert follow.choose_order(orders) == ("x", None)
    assert follow.choose_order(orders, current=B) == ("y", None)
    assert follow.choose_order(orders, skip=frozenset({"x"})) == ("y", None)


def test_with_nothing_open_a_starting_orders_model_is_fetched_ahead():
    orders = [_order("x", "draining"), _order("y", "starting", "org/B", "b" * 40), _order("z", "paused")]
    assert follow.choose_order(orders) == (None, B)
    assert follow.choose_order([_order("x", "draining")]) == (None, None)
    assert follow.choose_order([]) == (None, None)


def test_the_child_mines_one_order_with_the_parents_options_and_its_own_contract():
    command = follow.child_command("order-gen-1", ["--validator-url", "https://v", "--hotkey", "h"])
    assert command[:5] == [sys.executable, "-m", "reliquary.cli.main", "corpus", "mine"]
    assert command[-2:] == ["--job-id", "order-gen-1"] and "--hotkey" in command
    env = follow.child_env({TASK_CONTRACT_ENV_VAR: "/tmp/old.json", "PATH": "/bin"})
    assert env == {"PATH": "/bin"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_orders_are_mined_one_after_another_and_a_starting_model_is_fetched_once():
    lists = iter([
        [_order("s", "starting")],
        [_order("s", "starting")],
        [_order("s", "open")],
        [_order("s", "draining"), _order("t", "open", "org/B", "b" * 40)],
        [],
    ])
    clock, mined, fetched, slept = _Clock(), [], [], []

    def mine(job_id):
        mined.append(job_id)
        clock.now += 3600
        return 0

    follow.follow_orders(list_orders=lambda: next(lists), mine=mine, fetch=fetched.append,
                         sleep=slept.append, clock=clock, rounds=5)
    assert fetched == [A]
    assert mined == ["s", "t"]
    assert slept == [follow.IDLE_SECONDS] * 3


def test_an_order_that_fails_at_once_is_skipped_while_another_is_open():
    lists = iter([[_order("bad", "open"), _order("good", "open")]] * 2 + [[_order("bad", "open")]] * 2)
    clock, mined = _Clock(), []

    def mine(job_id):
        mined.append(job_id)
        clock.now += 5
        return 1 if job_id == "bad" else 0

    follow.follow_orders(list_orders=lambda: next(lists), mine=mine, fetch=lambda m: None,
                         sleep=lambda s: None, clock=clock, rounds=4)
    # bad fails at once, good is mined; with only bad open it is skipped once, then tried again.
    assert mined == ["bad", "good", "bad"]


def test_a_validator_without_an_order_list_stops_the_loop():
    calls = []
    follow.follow_orders(list_orders=lambda: None, mine=calls.append, fetch=calls.append,
                         sleep=calls.append, rounds=3)
    assert calls == []


def test_corpus_mine_without_a_job_follows_the_listed_orders(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli import main

    seen = {}
    monkeypatch.setattr(main, "_served_orders", lambda url, strict=False: [])

    def fake_follow(*, list_orders, mine, fetch):
        seen["orders"] = list_orders()
        mine("order-gen-1")

    monkeypatch.setattr(follow, "follow_orders", fake_follow)
    monkeypatch.setattr(follow, "run_child", lambda job_id, argv: seen.setdefault("child", (job_id, argv)) and 0)
    result = CliRunner().invoke(main.app, [
        "corpus", "mine", "--validator-url", "https://api.example", "--hotkey", "hk",
        "--ask-alpha-per-mtoken", "0.3", "--ask-capacity", "8"])
    assert result.exit_code == 0, result.output
    job_id, argv = seen["child"]
    assert job_id == "order-gen-1" and seen["orders"] == []
    assert argv[argv.index("--validator-url") + 1] == "https://api.example"
    assert argv[argv.index("--hotkey") + 1] == "hk"
    assert argv[argv.index("--ask-alpha-per-mtoken") + 1] == "0.3"
    assert argv[argv.index("--ask-capacity") + 1] == "8"
    assert "--gpu-memory-utilization" not in argv and "--job-id" not in argv


def test_an_unreadable_list_is_retried():
    answers = iter([OSError("down"), [_order("x", "open")]])
    mined, slept = [], []

    def list_orders():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    follow.follow_orders(list_orders=list_orders, mine=lambda j: mined.append(j) or 0, fetch=lambda m: None,
                         sleep=slept.append, rounds=2)
    assert slept == [follow.IDLE_SECONDS] and mined == ["x"]
