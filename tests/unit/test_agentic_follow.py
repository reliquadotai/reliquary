"""Following agentic orders over fakes: discovery from the order list and each manifest, the orders this install
serves, ranking by pay, several at once, and switching when one ends or leaves the list."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from reliquary.miner import agentic_follow as follow
from reliquary.miner import order_follow

SIGNED, SUITE, REPLAY = "signed", "suite", "replay"


def _job(job_id, kind=SIGNED, repo="org/A", revision="a" * 40, max_total_tokens=60000):
    episode = None if kind is None else SimpleNamespace(kind=kind, max_total_tokens=max_total_tokens)
    return SimpleNamespace(job_id=job_id, checkpoint_repo=repo, checkpoint_revision=revision, episode=episode)


def _order(job_id, state="open", price=None, ceiling=None, budget=None, leased=None, repo="org/A",
           revision="a" * 40):
    order = {"job_id": job_id, "state": state, "model": repo, "revision": revision}
    if price is not None or ceiling is not None:
        order["pricing"] = "auction"
        order["auction"] = {k: v for k, v in (("price_alpha_per_mtoken", price), ("ceiling_alpha_per_mtoken", ceiling),
                                              ("budget", budget), ("leased", leased)) if v is not None}
    return order


class FakeChild:
    def __init__(self, job_id):
        self.job_id, self.code, self.terminated = job_id, None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def _signed_is_the_episode_kind(monkeypatch):
    from reliquary.corpus import job as job_module

    monkeypatch.setattr(job_module, "is_signed_sandbox",
                        lambda job: job.episode is not None and job.episode.kind == SIGNED)


def _follower(documents, jobs, *, refuse=None, **options):
    """A follower over ``documents`` (one per read; the last repeats) and the manifests in ``jobs``."""
    documents = list(documents)
    launched, children, fetched = [], {}, []

    def list_jobs():
        return documents.pop(0) if len(documents) > 1 else documents[0]

    def launch(job_id, slot, hotkey):
        launched.append((job_id, slot, hotkey))
        children[job_id] = FakeChild(job_id)
        return children[job_id]

    follower = follow.AgenticFollower(
        list_jobs=list_jobs, read_manifest=lambda job_id: jobs[job_id], launch=launch, fetch=fetched.append,
        parse=lambda raw: raw, serve_refusal=refuse or (lambda job: None), sleep=lambda s: None,
        clock=options.pop("clock", Clock()), **options)
    return follower, launched, children, fetched


def test_only_open_episode_orders_this_install_serves_are_taken():
    jobs = {"st": _job("st", kind=None), "swe": _job("swe"), "tb": _job("tb", kind=SUITE),
            "pro": _job("pro"), "big": _job("big", max_total_tokens=200_000),
            "other": _job("other", repo="org/B")}
    document = {"sandbox_hotkey": "5Val", "orders": [
        _order("st"), _order("swe"), _order("tb"), _order("pro"), _order("big"), _order("other", repo="org/B"),
        _order("draining", state="draining")]}

    def refuse(job):
        return "reliquary-swebench-pro is not installed" if job.job_id == "pro" else None

    follower, launched, _, _ = _follower([document], jobs, refuse=refuse, max_jobs=8,
                                         limits=follow.Limits(models=frozenset({"org/A"}), max_context=131072))
    follower.step()
    assert sorted(job for job, _, _ in launched) == ["swe", "tb"]
    assert follower.refused == {"st": "single-turn", "pro": "reliquary-swebench-pro is not installed",
                                "big": "its episodes take up to 200000 tokens, over --max-context 131072",
                                "other": f"its model org/B@{'a' * 40} is not one of --model"}


def test_the_validator_hotkey_comes_from_the_listing_and_only_signed_orders_get_it():
    jobs = {"swe": _job("swe"), "tb": _job("tb", kind=SUITE)}
    follower, launched, _, _ = _follower([{"sandbox_hotkey": "5Val", "orders": [_order("swe"), _order("tb")]}],
                                         jobs, max_jobs=2, validator_hotkey="5Flag")
    follower.step()
    assert sorted(launched) == [("swe", 0, "5Val"), ("tb", 1, None)]


def test_without_any_validator_hotkey_signed_orders_wait_and_suite_ones_run():
    jobs = {"swe": _job("swe"), "tb": _job("tb", kind=SUITE)}
    follower, launched, _, _ = _follower([{"orders": [_order("swe"), _order("tb")]}], jobs, max_jobs=2)
    follower.step()
    assert launched == [("tb", 0, None)] and "swe" not in follower.refused
    follower, launched, _, _ = _follower([{"orders": [_order("swe")]}], jobs, validator_hotkey="5Flag")
    follower.step()
    assert launched == [("swe", 0, "5Flag")]


def test_orders_rank_by_leases_left_then_pay_then_a_model_already_loaded():
    first = follow.Candidate("cheap", "org/A", 0.1, True, True)
    rich = follow.Candidate("rich", "org/B", 0.5, True, True)
    full = follow.Candidate("full", "org/A", 0.9, False, True)
    same = follow.Candidate("same", "org/C", 0.1, True, True)
    assert [c.job_id for c in follow.rank([first, rich, full, same])] == ["rich", "cheap", "same", "full"]
    assert [c.job_id for c in follow.rank([first, same], loaded=frozenset({"org/C"}))] == ["same", "cheap"]
    assert follow.order_pay(_order("x", price=0.3, ceiling=0.6)) == 0.3
    assert follow.order_pay(_order("x", ceiling=0.6)) == 0.6       # nobody working yet: its ceiling
    assert follow.order_pay(_order("x")) == 0.0
    assert not follow.leases_free(_order("x", price=0.3, budget=4, leased=4))
    assert follow.leases_free(_order("x"))


def test_the_best_paying_orders_fill_the_slots():
    jobs = {j: _job(j) for j in ("a", "b", "c")}
    document = {"sandbox_hotkey": "5Val", "orders": [
        _order("a", price=0.1), _order("b", price=0.4), _order("c", ceiling=0.2)]}
    follower, launched, _, _ = _follower([document], jobs, max_jobs=2)
    follower.step()
    assert [(job, slot) for job, slot, _ in launched] == [("b", 0), ("c", 1)]


def test_a_finished_order_frees_its_slot_for_the_next_and_a_retired_one_is_stopped():
    jobs = {j: _job(j) for j in ("a", "b", "c")}
    clock = Clock()
    documents = [
        {"sandbox_hotkey": "5Val", "orders": [_order("a", price=0.5), _order("b", price=0.1)]},
        {"sandbox_hotkey": "5Val", "orders": [_order("a", price=0.5), _order("b", price=0.1)]},
        # a retires: one read may race its state, the second stops it.
        {"sandbox_hotkey": "5Val", "orders": [_order("a", "draining"), _order("b", price=0.1),
                                              _order("c", price=0.2)]},
    ]
    follower, launched, children, _ = _follower(documents, jobs, clock=clock)
    follower.step()
    assert [job for job, _, _ in launched] == ["a"]
    clock.now += 3600
    follower.step()                                    # a still runs: nothing new
    assert [job for job, _, _ in launched] == ["a"]
    follower.step()
    assert not children["a"].terminated and [job for job, _, _ in launched] == ["a"]
    follower.step()
    assert children["a"].terminated
    assert [job for job, _, _ in launched] == ["a", "c"]     # c pays more than b
    children["c"].code = 0                             # c filled: its child ends by itself
    clock.now += 3600
    follower.step()
    assert [job for job, _, _ in launched] == ["a", "c", "b"]


def test_an_order_that_fails_at_once_is_skipped_while_another_is_open():
    jobs = {j: _job(j) for j in ("bad", "good")}
    clock = Clock()
    document = {"sandbox_hotkey": "5Val", "orders": [_order("bad", price=0.9), _order("good", price=0.1)]}
    follower, launched, children, _ = _follower([document], jobs, clock=clock)
    follower.step()
    children["bad"].code = 4
    clock.now += 5
    follower.step()
    assert [job for job, _, _ in launched] == ["bad", "good"] and follower.failed == {"bad"}


def test_with_nothing_open_a_starting_agentic_orders_model_is_fetched_once():
    jobs = {"st": _job("st", kind=None, repo="org/S"), "swe": _job("swe", repo="org/B", revision="b" * 40)}
    document = {"orders": [_order("st", "starting", repo="org/S"),
                           _order("swe", "starting", repo="org/B", revision="b" * 40)]}
    follower, launched, _, fetched = _follower([document], jobs)
    follower.step()
    follower.step()
    assert launched == [] and fetched == [f"org/B@{'b' * 40}"]


def test_a_validator_without_an_order_list_stops_and_children_are_stopped_on_exit():
    jobs = {"a": _job("a")}
    documents = [{"sandbox_hotkey": "5Val", "orders": [_order("a")]}, {"jobs": []}]
    follower, launched, children, _ = _follower(documents, jobs)
    follower.run(rounds=5)
    assert launched == [("a", 0, "5Val")] and children["a"].terminated and follower.running == {}


def test_slots_get_their_own_port_card_and_share():
    assert follow.slot_settings(0, max_jobs=1, port=8011, concurrency=8, gpus=[],
                                gpu_memory_utilization=None) == (8011, 8, None, None)
    assert follow.slot_settings(1, max_jobs=2, port=8011, concurrency=16, gpus=["0", "1"],
                                gpu_memory_utilization=None) == (8012, 8, None, "1")
    assert follow.slot_settings(1, max_jobs=2, port=8011, concurrency=8, gpus=[],
                                gpu_memory_utilization=None) == (8012, 4, 0.45, None)


def test_the_child_mines_one_order_with_the_parents_options():
    command = follow.child_command("order-eval-1", ["--validator-url", "https://v", "--hotkey", "h"],
                                   validator_hotkey="5Val", port=8012, concurrency=4, gpu_memory_utilization=0.45)
    assert command[:5] == [sys.executable, "-m", "reliquary.cli.main", "corpus", "mine-agentic"]
    assert command[command.index("--job-id") + 1] == "order-eval-1"
    assert command[command.index("--validator-hotkey") + 1] == "5Val"
    assert command[command.index("--port") + 1] == "8012"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.45"
    assert "--validator-hotkey" not in follow.child_command("x", [], validator_hotkey=None, port=1,
                                                            concurrency=1, gpu_memory_utilization=None)


def test_the_single_turn_follower_leaves_agentic_orders_to_mine_agentic():
    manifests = {"st": {"job_id": "st"}, "swe": {"job_id": "swe", "episode": {"harness": "x"}}}
    reads = []

    def read(job_id):
        reads.append(job_id)
        return manifests[job_id]

    takes = order_follow.single_turn_filter(read)
    lists = iter([[_order("swe"), _order("st")]] * 2)
    mined = []
    order_follow.follow_orders(list_orders=lambda: next(lists), mine=lambda j: mined.append(j) or 0,
                               fetch=lambda m: None, sleep=lambda s: None, rounds=2, takes=takes)
    assert mined == ["st", "st"] and sorted(reads) == ["st", "swe"]     # each manifest read once


def test_mine_agentic_without_a_job_follows_the_orders(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli import main

    seen = {}

    class FakeFollower:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(follow, "AgenticFollower", FakeFollower)
    monkeypatch.setattr(follow, "http_listing", lambda url: (lambda: {}, lambda job_id: {}))
    result = CliRunner().invoke(main.app, [
        "corpus", "mine-agentic", "--validator-url", "https://api.example", "--hotkey", "hk", "--no-sync-envs",
        "--gpus", "0,1", "--model", "org/A", "--max-context", "131072", "--ask-alpha-per-mtoken", "0.4"])
    assert result.exit_code == 0, result.output
    assert seen["ran"] and seen["max_jobs"] == 2 and seen["validator_hotkey"] is None
    assert seen["limits"] == follow.Limits(models=frozenset({"org/A"}), max_context=131072)


def test_the_real_serve_check_reads_a_parsed_manifest_without_raising():
    from reliquary.corpus.job import parse_job
    from tests.unit.test_corpus_job_episode import _manifest

    single = parse_job(_manifest(with_episode=False, prompt_source="openmathinstruct", renderer_id="x-v1"))
    assert follow.serve_refusal(single).startswith("not an episode job")
    refusal = follow.serve_refusal(parse_job(_manifest()))
    assert refusal is None or isinstance(refusal, str)
