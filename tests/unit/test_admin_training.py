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
    assert client("GET", "/admin/v1/training/capabilities").status_code == 404


def test_a_bad_flag_is_a_configuration_error():
    with pytest.raises(ValueError):
        training.training_enabled({training.TRAINING_ORDERS_ENV: "yes"})


def test_legacy_order_digest_is_preserved_when_quote_binding_is_absent():
    assert training.order_digest(training.CreateTraining(**_order())) == (
        "33a31835ba45c4f5eaa0904cd701b016eabd6377351a608f00edd735ce688bb0")


def test_quote_binding_is_opaque_immutable_and_echoed_by_the_admin(on):
    bound = dict(quote_id="quote-1", quote_sha256="c" * 64, billing_basis=training.BILLING_BASIS)
    created = on("POST", "/admin/v1/training", _order(**bound))
    assert created.status_code == 201
    assert all(created.json()["order"][key] == value for key, value in bound.items())
    assert on("POST", "/admin/v1/training", _order(**bound)).status_code == 200
    assert on("POST", "/admin/v1/training", _order(**{**bound, "quote_sha256": "d" * 64})).status_code == 409


@pytest.mark.parametrize("binding", [
    {"quote_id": "quote-1"}, {"quote_sha256": "c" * 64},
    {"quote_id": "quote-1", "quote_sha256": "c" * 64},
    {"quote_id": "quote-1", "quote_sha256": "not-a-digest", "billing_basis": training.BILLING_BASIS},
])
def test_partial_or_malformed_quote_binding_is_refused(on, binding):
    assert on("POST", "/admin/v1/training", _order(**binding)).status_code == 422


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
    assert status.json()["runtime"] == {"state": "unknown", "managed": False}
    retired = on("POST", "/admin/v1/training/order-train-t1/retire", {})
    assert retired.status_code == 200 and retired.json()["status"] == "retired"
    assert on("POST", "/admin/v1/training/order-train-t1/retire", {}).json()["status"] == "retired"
    assert on("GET", "/admin/v1/training/order-train-nope").status_code == 404
    assert on("GET", "/admin/v1/training/default").status_code == 409


def test_capabilities_do_not_claim_a_managed_paid_runtime(on):
    assert on.client.get("/admin/v1/training/capabilities").status_code == 401
    response = on("GET", "/admin/v1/training/capabilities")
    assert response.status_code == 200
    assert response.json() == {"schema": "reliquary/training-capabilities/v1", "task_declaration": True,
                               "runtime_orchestration": False, "final_usage": False,
                               "checkpoint_delivery": False, "paid_admission": False}


def test_admin_retirement_fences_a_live_training_book_without_cancelling_issued_work(on, tmp_path):
    import asyncio

    from reliquary.validator.rl_async_service import build_rl_async, rl_async_settings
    from reliquary.validator.rl_leases import LeaseRequest
    from reliquary.validator.rl_training_orders import TrainingAdmission, load_training_entry

    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    entry = on.registry["entries"]["order-train-t1"]
    settings = rl_async_settings(entry, entry.params["cap"], max_tokens=lambda env: 4000)
    rl = build_rl_async(settings, path=tmp_path / "live.sqlite3", grader=None)
    try:
        gate = TrainingAdmission(entry, rl.book, load=load_training_entry)
        rl.book.auctions.admission = gate.reason
        assert asyncio.run(gate.refresh())
        rl.book.adopt_checkpoint(0)
        env = "reliquary_dapo_math_v1"
        rl.book.auctions.post_ask(env, "A", .2, 8)
        lease = rl.book.grant([LeaseRequest("A", env, (1,), 0)])[0].leases[0]
        assert on("POST", "/admin/v1/training/order-train-t1/retire", {}).status_code == 200
        assert not asyncio.run(gate.refresh()) and gate.reason() == "registry_retired"
        assert not rl.book.grant([LeaseRequest("A", env, (2,), 0)])[0].leases
        assert rl.book.submit(lease.lease_id, "A", in_zone=True).state == "submitted"
    finally:
        rl.close()


def test_the_order_freezes_its_alpha_rate(on):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    order = on.registry["entries"]["order-train-t1"].params["training_order"]
    assert order["usd_per_alpha"] == pytest.approx(4.0)                    # the rate the ceiling was converted at


RETIRED_AT = cp.period_end(cp.period_of(1_800_000_000.0)) - 600.0        # the retire's time (round 1)
SETTLED_AFTER = RETIRED_AT + 600.0 + 120.0                                # its period's end + the settle margin


def _summary(*settled, open_leases=0, stopped_at=RETIRED_AT, written_at=SETTLED_AFTER + 1, stopped="retired",
             settled_through=cp.period_of(RETIRED_AT)):
    return {"task_id": "order-train-t1", "run": "order-train-t1", "starved": [], "dead": [[10.0, 20.0]], "stopped_at": stopped_at,
            "pending": {"open_leases": open_leases, "buffered": 0, "audits": 0, "unledgered": 0},
            "settle_margin_s": 120.0, "written_at": written_at, "settled_through": settled_through,
            "periods": [{"period": i, "settled": s, "accrued_alpha": 1.0, "paid_alpha": 1.0 if s else None,
                         "buckets": {}} for i, s in enumerate(settled)], "stopped": stopped,
            "latest_checkpoint": 3}


@pytest.fixture
def spend(monkeypatch):
    from reliquary.validator import rl_spend

    from reliquary.admin import service

    held = {}
    monkeypatch.setattr(rl_spend, "read_spend", lambda run: held.get(run))
    monkeypatch.setattr(service, "_round_time", lambda r: {1: RETIRED_AT}[r])
    return held


def test_the_spend_route_adds_the_frozen_rate(on, spend):
    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    assert on.client.get("/admin/v1/training/order-train-t1/spend").status_code == 401
    assert on("GET", "/admin/v1/training/order-train-t1/spend").status_code == 404     # not written yet
    spend["order-train-t1"] = _summary(True, False)
    answer = on("GET", "/admin/v1/training/order-train-t1/spend")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["usd_per_alpha"] == pytest.approx(4.0) and body["final"] is False
    assert [p["settled"] for p in body["periods"]] == [True, False] and body["latest_checkpoint"] == 3
    assert body["dead"] == [[10.0, 20.0]] and body["starved"] == []                # passed through unchanged
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
    assert final(_summary(True, True, open_leases=2)) is False                      # leases may still deliver
    assert final(_summary(True, True, written_at=SETTLED_AFTER - 1)) is False       # before its period settled
    assert final(_summary(True, True, stopped_at=RETIRED_AT + 3600)) is False       # stopped in a later period
    assert final(_summary(True, True)) is True
    assert final(_summary()) is True                                                # nothing was ever paid
    assert final(_summary(True, True, stopped=None)) is False                       # the controller has not stopped
    assert final(_summary(True, True, stopped_at=None)) is False
    stop_period = cp.period_of(RETIRED_AT)
    assert final(_summary(True, True, settled_through=stop_period - 1)) is False    # its settle had not closed it
    assert final(_summary(True, True, settled_through=None)) is False
    assert final(_summary(True, True, settled_through=stop_period + 1)) is True


def test_retire_revokes_the_tasks_trainer_pods(on):
    import hashlib

    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    assert on("POST", "/admin/v1/training", _order(training_id="t2")).status_code == 201

    def pod(executor_id, task_id):
        body = {"executor_id": executor_id, "token_sha256": hashlib.sha256(executor_id.encode()).hexdigest(),
                "model_id": "m", "model_revision": "r", "expires_at": 4e12, "scope": "trainer", "task_id": task_id}
        assert on("POST", "/admin/v1/executors", body).status_code == 201

    pod("pod-1", "order-train-t1")
    pod("pod-2", "order-train-t2")
    retired = on("POST", "/admin/v1/training/order-train-t1/retire", {})
    assert retired.status_code == 200 and retired.json()["final_usage"] is None
    assert on("GET", "/admin/v1/executors/pod-1").json()["status"] == "revoked"
    assert on("GET", "/admin/v1/executors/pod-2").json()["status"] == "active"


def test_quote_bound_order_refuses_external_trainer_registration(on):
    order = _order(quote_id="quote-1", quote_sha256="c" * 64, billing_basis=training.BILLING_BASIS)
    created = on("POST", "/admin/v1/training", order)
    assert created.status_code == 201
    pod = {"executor_id": "external-pod", "token_sha256": "d" * 64, "model_id": "m", "model_revision": "r",
           "expires_at": 4e12, "scope": "trainer", "task_id": created.json()["task_id"]}
    refused = on("POST", "/admin/v1/executors", pod)
    assert refused.status_code == 409 and refused.json()["detail"] == "training_requires_owned_runtime"
    assert on("GET", "/admin/v1/executors/external-pod").status_code == 404


def _order_entry(on, **params):
    from dataclasses import replace

    assert on("POST", "/admin/v1/training", _order()).status_code == 201
    entry = on.registry["entries"]["order-train-t1"]
    return replace(entry, params={**entry.params, **params})


def test_a_training_order_never_carries_the_per_env_pay_guard(on):
    from reliquary.validator.rl_async_service import ENV_PAY_K_PARAM, rl_async_settings

    entry = _order_entry(on, **{ENV_PAY_K_PARAM: 2.0})
    with pytest.raises(ValueError, match="auction"):
        rl_async_settings(entry, entry.params["cap"], environ={})


@pytest.mark.parametrize("where", ["param", "env"])
def test_a_training_order_refuses_a_pay_carry(on, where):
    from reliquary.validator.rl_async_service import CARRY_ENV, CARRY_PARAM, rl_async_settings

    entry = _order_entry(on, **({CARRY_PARAM: 2} if where == "param" else {}))
    environ = {CARRY_ENV: "2"} if where == "env" else {}
    with pytest.raises(ValueError, match="carry"):
        rl_async_settings(entry, entry.params["cap"], environ=environ)
    plain = on.registry["entries"]["order-train-t1"]
    assert rl_async_settings(plain, plain.params["cap"], environ={}).pay_carry_periods == 0
