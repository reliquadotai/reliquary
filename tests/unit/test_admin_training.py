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
