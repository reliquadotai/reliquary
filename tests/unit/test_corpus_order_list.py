"""The job list a miner reads (`/corpus/jobs`): every served job with its state, model and pricing."""

from __future__ import annotations

from types import SimpleNamespace

from reliquary.validator.corpus_service import CorpusJobRoutes as JobRoutes


def _spec(job_id):
    return SimpleNamespace(job_id=job_id, checkpoint_repo="org/M", checkpoint_revision="r" * 40,
                           prompt_count=100, slots_per_prompt=2, max_prompt_tokens=None,
                           sampling=SimpleNamespace(max_new_tokens=4096))


def test_each_served_job_is_listed_with_its_state_and_what_it_runs():
    view = {"pricing": "auction", "price_alpha_per_mtoken": 0.3, "ceiling_alpha_per_mtoken": 0.5}
    routes = JobRoutes()
    routes.add("a", SimpleNamespace(auction_view=lambda: view), spec=_spec("a"))
    routes.add("b", SimpleNamespace(auction_view=lambda: None), spec=_spec("b"))
    routes.add("c", SimpleNamespace(), spec=_spec("c"))
    routes.add("d", SimpleNamespace())
    ready = {"b": False}
    routes.hold("b", lambda: ready["b"])
    routes.paused.add("c")
    routes.retire("d")

    orders = {o["job_id"]: o for o in routes.orders()}
    assert {j: o["state"] for j, o in orders.items()} == {
        "a": "open", "b": "starting", "c": "paused", "d": "draining"}
    assert orders["a"]["pricing"] == "auction" and orders["a"]["auction"] == view
    assert orders["b"]["pricing"] == "first_come" and "auction" not in orders["b"]
    assert orders["a"]["model"] == "org/M" and orders["a"]["revision"] == "r" * 40
    assert (orders["a"]["prompts"], orders["a"]["samples_per_prompt"], orders["a"]["max_new_tokens"]) == (
        100, 2, 4096)
    assert "max_prompt_tokens" not in orders["a"]
    assert "model" not in orders["d"]                 # no spec given: its id and state alone
    assert routes.open_jobs() == ["a"]
    ready["b"] = True
    assert {o["job_id"]: o["state"] for o in routes.orders()}["b"] == "open"


def test_a_failing_auction_view_lists_the_job_without_it():
    def broken():
        raise RuntimeError("down")

    routes = JobRoutes()
    routes.add("a", SimpleNamespace(auction_view=broken), spec=_spec("a"))
    listed = routes.orders()[0]
    assert listed["state"] == "open" and "pricing" not in listed
