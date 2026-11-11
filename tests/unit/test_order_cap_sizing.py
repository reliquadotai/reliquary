"""Order caps sized to what an order's audit lets it pay (docs/design/2026-11-11-order-cap-sizing-design.md)."""

from __future__ import annotations

import asyncio
import time

import pytest

from reliquary.admin import cap_sizing as sizing
from reliquary.validator import corpus_periods as cp
from reliquary.validator import eval_control
from reliquary.validator.corpus_period_settlement import RECENT_PAID_PERIODS, CorpusPeriodSettler
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_admin_token_ceilings import CEILING, rates  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import MODEL, REVISION, _order, pins  # noqa: F401

PRICE = {cp.MAX_PRICE_PARAM: 4e-9}


def test_a_card_lets_through_its_audit_rate_divided_by_the_audited_share():
    evals = sizing.spend_estimate({**PRICE, "audit_q": 1.0}, completion_tps=1000, prompt_tps=0, cards=1)
    assert evals == pytest.approx(1000 * 4e-9 * cp.PERIOD_SECONDS)
    batch = sizing.spend_estimate({**PRICE, "audit_q": 0.2}, completion_tps=1000, prompt_tps=0, cards=1)
    assert batch == pytest.approx(5 * evals)        # four records in five are paid without the card
    assert sizing.spend_estimate({**PRICE, "audit_q": 1.0}, completion_tps=1000, prompt_tps=0,
                                 cards=3) == pytest.approx(3 * evals)
    # Prompt tokens count at the input price only.
    priced = {**PRICE, "audit_q": 1.0, cp.MAX_PRICE_INPUT_PARAM: 1e-9}
    assert sizing.spend_estimate(priced, completion_tps=1000, prompt_tps=500, cards=1) == pytest.approx(
        (1000 * 4e-9 + 500 * 1e-9) * cp.PERIOD_SECONDS)
    assert sizing.spend_estimate({**PRICE}, completion_tps=1000, prompt_tps=500, cards=1) == pytest.approx(evals)
    assert sizing.spend_estimate({"audit_q": 1.0}, completion_tps=1000, prompt_tps=0, cards=1) is None


def test_the_cap_has_headroom_over_what_was_paid_or_accrued_within_the_asked_cap():
    assert sizing.sized_cap(0.25, estimate=0.01) == pytest.approx(0.015)
    assert sizing.sized_cap(0.25, estimate=0.01, recent_paid=[0.004, 0.03]) == pytest.approx(0.045)
    assert sizing.sized_cap(0.25, estimate=0.01, open_value=0.02) == pytest.approx(0.03)
    assert sizing.sized_cap(0.25, estimate=1.0) == 0.25
    assert sizing.sized_cap(0.25, estimate=0.0) == sizing.CAP_FLOOR


def test_an_unmeasured_model_is_sized_on_the_high_defaults():
    status = {"models": {"m@r": {"audit_tokens_per_second": 120.0, "audit_prompt_tokens_per_second": 30.0},
                         "n@r": {"audit_tokens_per_second": None, "audit_prompt_tokens_per_second": None}}}
    assert sizing.model_rates(status, "m@r") == (120.0, 30.0)
    defaults = (sizing.DEFAULT_COMPLETION_TOKENS_PER_SECOND, sizing.DEFAULT_PROMPT_TOKENS_PER_SECOND)
    assert sizing.model_rates(status, "n@r") == defaults
    assert sizing.model_rates(None, "m@r") == defaults


def test_a_resize_under_the_minimum_change_is_not_written():
    params = {**PRICE, "audit_q": 1.0, "cap": 0.015, sizing.CAP_MAX_PARAM: 0.25}
    status = {"models": {"m@r": {"audit_tokens_per_second": 10.0, "audit_prompt_tokens_per_second": 0.0}}}
    facts = {"model": "m@r", "audit_executors": 1, "recent_paid": [0.0101], "open_value": 0.0}
    assert sizing.resized(params, facts, status) is None           # 0.01515: within 10 %
    assert sizing.resized(params, {**facts, "recent_paid": [0.03]}, status) == pytest.approx(0.045)
    assert sizing.resized({k: v for k, v in params.items() if k != sizing.CAP_MAX_PARAM}, facts, status) is None


def test_the_switch_is_on_or_unset():
    assert sizing.sizing_enabled({}) is False
    assert sizing.sizing_enabled({sizing.SIZING_ENV: "on"}) is True
    with pytest.raises(ValueError):
        sizing.sizing_enabled({sizing.SIZING_ENV: "yes"})


def _settler(max_price=None, max_input_price=None):
    return CorpusPeriodSettler(task_id="t", job_id="j", cap=0.25, records=None, archives=None,
                               oldest_pending=lambda: None, genesis=lambda: 0.0,
                               max_price=max_price, max_input_price=max_input_price)


def test_the_settler_keeps_the_recent_periods_paid_and_the_open_value():
    s = _settler(max_price=2e-9, max_input_price=1e-9)
    paid = {str(p): p / 1000 for p in range(1, 6)}
    s._report({"settled": [], "period_paid": paid}, [])
    assert s.recent_paid == [0.003, 0.004, 0.005]
    assert len(s.recent_paid) == RECENT_PAID_PERIODS
    assert s.open_value() == 0.0
    at = cp.PERIOD_SECONDS
    verdicts = {"a": {"passed": True, "token_count": 1000, "prompt_token_count": 400, "received_at": at + 1},
                "b": {"passed": True, "token_count": 3000, "prompt_token_count": 0, "received_at": 2 * at + 1},
                "c": {"passed": False, "token_count": 9000, "prompt_token_count": 9000, "received_at": at + 1}}
    s._count_open(list(verdicts), verdicts, set(), 0.0)
    assert set(s.open_periods["1"]) == {"verdicts", "passed", "verified_tokens"}   # the public view is unchanged
    assert s.open_value() == pytest.approx(max(1000 * 2e-9 + 400 * 1e-9, 3000 * 2e-9))
    assert _settler().open_value() is None


@pytest.fixture
def control(monkeypatch):
    """The control status the admin reads."""
    state = {"document": None}

    async def read(**kwargs):
        return state["document"]

    monkeypatch.setattr(eval_control, "read_control_status", read)
    return state


def _rates_status(completion=100.0, prompt=50.0, jobs=None):
    return {"at": time.time(), "jobs": jobs or {},
            "models": {f"{MODEL}@{REVISION}": {"audit_tokens_per_second": completion,
                                               "audit_prompt_tokens_per_second": prompt}}}


def test_an_order_is_declared_at_its_sized_cap_with_the_asked_cap_kept(
        admin, pins, rates, control, monkeypatch):  # noqa: F811
    monkeypatch.setenv(sizing.SIZING_ENV, "on")
    control["document"] = _rates_status()
    created = admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING))
    assert created.status_code == 201, created.text
    params = admin.registry["entries"]["order-eval-p1"].params
    estimate = (100 * cp.max_price(params) + 50 * cp.max_input_price(params)) * cp.PERIOD_SECONDS
    assert params["cap"] == pytest.approx(max(sizing.CAP_FLOOR, estimate * sizing.HEADROOM))
    assert params["cap"] < 0.25 and params["floor"] == params["cap"]
    assert params[sizing.CAP_MAX_PARAM] == 0.25
    # A replay names the asked cap: the same order, not another contract.
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 200


def test_without_the_switch_an_order_keeps_its_asked_cap(admin, pins, rates, control):  # noqa: F811
    control["document"] = _rates_status()
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 201
    params = admin.registry["entries"]["order-eval-p1"].params
    assert params["cap"] == 0.25 and sizing.CAP_MAX_PARAM not in params


def test_an_unreadable_status_sizes_on_the_defaults(admin, pins, rates, monkeypatch):  # noqa: F811
    monkeypatch.setenv(sizing.SIZING_ENV, "on")

    async def down(**kwargs):
        raise OSError("bucket down")

    monkeypatch.setattr(eval_control, "read_control_status", down)
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 201
    params = admin.registry["entries"]["order-eval-p1"].params
    assert params[sizing.CAP_MAX_PARAM] == 0.25 and params["cap"] <= 0.25


def test_the_resize_follows_what_the_order_pays(admin, pins, rates, control, monkeypatch):  # noqa: F811
    monkeypatch.setenv(sizing.SIZING_ENV, "on")
    control["document"] = _rates_status()
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 201
    first = admin.registry["entries"]["order-eval-p1"].params["cap"]
    resize = admin.app.state.resize_order_caps

    facts = {"model": f"{MODEL}@{REVISION}", "audit_executors": 1, "recent_paid": [0.02], "open_value": 0.0}
    control["document"] = _rates_status(jobs={"order-eval-p1": {"sizing": facts}})
    assert asyncio.run(resize()) == 1
    params = admin.registry["entries"]["order-eval-p1"].params
    assert params["cap"] == pytest.approx(0.03) and params["floor"] == params["cap"]
    assert params[sizing.CAP_MAX_PARAM] == 0.25

    # Paying less again: down, never under what the open period accrued.
    control["document"] = _rates_status(jobs={"order-eval-p1": {"sizing": {**facts, "recent_paid": [0.0],
                                                                           "open_value": 0.004}}})
    assert asyncio.run(resize()) == 1
    assert admin.registry["entries"]["order-eval-p1"].params["cap"] == pytest.approx(max(first, 0.006))

    # A stale status resizes nothing.
    control["document"] = {**_rates_status(jobs={"order-eval-p1": {"sizing": facts}}), "at": time.time() - 3600}
    assert asyncio.run(resize()) == 0


def test_a_resize_the_pool_cannot_take_keeps_the_cap(admin, pins, rates, control, monkeypatch):  # noqa: F811
    from reliquary.infrastructure import task_registry_store as store
    from reliquary.shared.task_registry import RegistryError

    monkeypatch.setenv(sizing.SIZING_ENV, "on")
    control["document"] = _rates_status()
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 201
    before = admin.registry["entries"]["order-eval-p1"].params["cap"]
    facts = {"model": f"{MODEL}@{REVISION}", "audit_executors": 1, "recent_paid": [0.5], "open_value": 0.0}
    control["document"] = _rates_status(jobs={"order-eval-p1": {"sizing": facts}})

    async def full(*args, **kwargs):
        raise RegistryError("corpus caps would total 0.9, above the admin pool of 0.3")

    monkeypatch.setattr(store, "set_task_cap", full)
    assert asyncio.run(admin.app.state.resize_order_caps()) == 0
    assert admin.registry["entries"]["order-eval-p1"].params["cap"] == before
    monkeypatch.undo()


def test_a_resize_never_passes_the_asked_cap(admin, pins, rates, control, monkeypatch):  # noqa: F811
    monkeypatch.setenv(sizing.SIZING_ENV, "on")
    control["document"] = _rates_status()
    assert admin("POST", "/admin/v1/jobs", _order(cap=0.25, max_price_usd_per_mtoken=CEILING)).status_code == 201
    facts = {"model": f"{MODEL}@{REVISION}", "audit_executors": 1, "recent_paid": [0.5], "open_value": 0.0}
    control["document"] = _rates_status(jobs={"order-eval-p1": {"sizing": facts}})
    assert asyncio.run(admin.app.state.resize_order_caps()) == 1
    assert admin.registry["entries"]["order-eval-p1"].params["cap"] == 0.25
