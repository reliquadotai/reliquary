"""Training orders on the admin API (reliquary/admin/training.py): an async RL task priced by the RL auction."""
from __future__ import annotations

import pytest

from reliquary.admin import training
from reliquary.validator import corpus_periods as cp
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_admin_token_ceilings import rates  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401

PROFILE = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"


def _order(**over):
    body = {"training_id": "t1", "profile_id": PROFILE,
            "environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 6000},
                             {"name": "envscaler_tools_v2", "share_bps": 4000}],
            "limits": {"max_steps": 200, "deadline_seconds": 86400}, "max_price_usd_per_mtoken": 2.0,
            "batch_groups": 128}
    return {**body, **over}


@pytest.fixture
def on(monkeypatch, request, rates):  # noqa: F811
    monkeypatch.setenv(training.TRAINING_ORDERS_ENV, "on")
    return request.getfixturevalue("admin")


def test_off_by_default(monkeypatch, request, rates):  # noqa: F811
    monkeypatch.delenv(training.TRAINING_ORDERS_ENV, raising=False)
    client = request.getfixturevalue("admin")
    assert client("POST", "/admin/v1/training", _order()).status_code == 404


def test_a_bad_flag_is_a_configuration_error():
    with pytest.raises(ValueError):
        training.training_enabled({training.TRAINING_ORDERS_ENV: "yes"})


def test_an_order_declares_an_async_rl_task_priced_by_auction(on):
    created = on("POST", "/admin/v1/training", _order())
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["task_id"] == "order-train-t1" and body["created"] is True and body["status"] == "active"
    assert body["environments"] == {"reliquary_dapo_math_v1": 0.6, "envscaler_tools_v2": 0.4}
    entry = on.registry["entries"]["order-train-t1"]
    params = entry.params
    assert entry.mechanism == "rl-discovered-price" and entry.profile_id == PROFILE
    assert params["settlement"] == "period-ema-v1" and params["pricing"] == "auction"
    assert params["rl_auction_lag_target"] == 2 and params["rl_batch_groups"] == 128
    assert params[cp.MAX_PRICE_ALPHA_PARAM] == pytest.approx(0.5)          # 2 USD / 4 USD an alpha
    assert cp.max_input_price(params) is None
    assert params["training_order"]["max_steps"] == 200 and params["cap"] == training.DEFAULT_TRAINING_CAP
    from reliquary.validator.rl_async_service import rl_async_settings

    settings = rl_async_settings(entry, params["cap"], environ={})
    assert settings.auction is not None and settings.batch_groups == 128


def test_a_replay_answers_the_task_and_another_order_under_its_id_is_refused(on):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    replay = on("POST", "/admin/v1/training", _order())
    assert replay.status_code == 200 and replay.json()["created"] is False


@pytest.mark.parametrize("over", [
    {"max_price_usd_per_mtoken": 3.0},
    {"environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 10000}]},
    {"limits": {"max_steps": 300, "deadline_seconds": 86400}},
    {"profile_id": "nope"},
])
def test_another_order_under_a_used_id_is_refused(on, over):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    replay = on("POST", "/admin/v1/training", _order(**over))
    assert replay.status_code == 409 and "training_id_reused" in replay.text
    assert on.registry["entries"]["order-train-t1"].params[cp.MAX_PRICE_ALPHA_PARAM] == pytest.approx(0.5)


@pytest.mark.parametrize("over, why", [
    ({"profile_id": "qwen35-2b-auction-v2"}, "Gumbel"),
    ({"profile_id": "nope"}, "unknown profile"),
    ({"environments": [{"name": "reliquary_logic_v2", "share_bps": 10000}]}, "not in profile"),
])
def test_an_order_that_cannot_train_is_refused(on, over, why):
    refused = on("POST", "/admin/v1/training", _order(**over))
    assert refused.status_code == 422 and why in refused.text
    assert "order-train-t1" not in on.registry["entries"]


@pytest.mark.parametrize("over", [
    {"environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 5000}]},
    {"environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 5000}] * 2},
    {"limits": {"max_steps": 0, "deadline_seconds": 86400}},
    {"max_price_usd_per_mtoken": 0},
    {"training_id": "Bad_Id"},
    {"extra": 1},
])
def test_a_malformed_order_is_refused(on, over):
    assert on("POST", "/admin/v1/training", _order(**over)).status_code == 422


def test_an_env_without_rl_rows_is_refused(monkeypatch):
    from reliquary import env_catalog

    monkeypatch.setattr(env_catalog, "entry", lambda name: None)
    body = training.CreateTraining(**_order())
    assert "no live RL rows" in training.training_refusal(body)


def test_status_and_retire(on):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    status = on("GET", "/admin/v1/training/order-train-t1")
    assert status.status_code == 200 and status.json()["order"]["training_id"] == "t1"
    retired = on("POST", "/admin/v1/training/order-train-t1/retire", {})
    assert retired.status_code == 200 and retired.json()["status"] == "retired"
    assert on("POST", "/admin/v1/training/order-train-t1/retire", {}).json()["status"] == "retired"
    assert on("GET", "/admin/v1/training/order-train-nope").status_code == 404
    assert on("GET", "/admin/v1/training/default").status_code == 409


def test_the_order_freezes_its_alpha_rate(on):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    order = on.registry["entries"]["order-train-t1"].params["training_order"]
    assert order["usd_per_alpha"] == pytest.approx(4.0)                    # the rate the ceiling was converted at


def _summary(*settled, leases_out=0):
    return {"task_id": "order-train-t1", "run": "order-train-t1", "leases_out": leases_out, "starved": [],
            "periods": [{"period": i, "settled": s, "accrued_alpha": 1.0, "paid_alpha": 1.0 if s else None,
                         "buckets": {}} for i, s in enumerate(settled)], "stopped": None, "latest_checkpoint": 3}


@pytest.fixture
def spend(monkeypatch):
    from reliquary.validator import rl_spend

    held = {}
    monkeypatch.setattr(rl_spend, "read_spend", lambda run: held.get(run))
    return held


def test_the_spend_route_adds_the_frozen_rate(on, spend):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    assert on("GET", "/admin/v1/training/order-train-t1/spend").status_code == 404     # not written yet
    spend["order-train-t1"] = _summary(True, False)
    answer = on("GET", "/admin/v1/training/order-train-t1/spend")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["usd_per_alpha"] == pytest.approx(4.0) and body["final"] is False
    assert [p["settled"] for p in body["periods"]] == [True, False] and body["latest_checkpoint"] == 3
    spend["order-train-t1"] = {**_summary(True), "task_id": "order-train-other"}
    assert on("GET", "/admin/v1/training/order-train-t1/spend").status_code == 503
    assert on("GET", "/admin/v1/training/order-train-nope/spend").status_code == 404


def test_spend_is_final_only_when_retired_and_settled(on, spend):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201

    def final(summary):
        spend["order-train-t1"] = summary
        return on("GET", "/admin/v1/training/order-train-t1/spend").json()["final"]

    assert final(_summary(True, True)) is False                                     # still active
    assert on("POST", "/admin/v1/training/order-train-t1/retire", {}).status_code == 200
    assert final(_summary(True, False)) is False                                    # a period still open
    assert final(_summary(True, True, leases_out=2)) is False                       # leases may still deliver
    assert final(_summary(True, True)) is True
    assert final(_summary()) is True                                                # nothing was ever paid
