"""What an order's miners were paid, per settled period: the platform bills the order on it."""

from __future__ import annotations

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_job_status import miner_pay
from tests.unit.test_admin_eval_jobs import admin  # noqa: F401
from tests.unit.test_admin_token_ceilings import CEILING, PER_BLOCK, rates  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import _order, pins  # noqa: F401


def test_each_settled_period_gives_its_tokens_and_their_pay_in_alpha():
    state = {"period_paid": {"12": 0.002, "11": 0.001}, "period_tokens": {"11": 1_000_000, "12": 2_000_000}}
    pay = miner_pay(state, 150.0)
    assert [p["work_period"] for p in pay["periods"]] == [11, 12]
    first = pay["periods"][0]
    assert first["verified_tokens"] == 1_000_000 and first["paid_pool_share"] == 0.001
    assert first["paid_alpha"] == pytest.approx(0.15)
    assert first["alpha_per_mtoken"] == pytest.approx(0.15)
    assert first["ended_at"] == cp.period_end(11)
    assert pay["verified_tokens"] == 3_000_000 and pay["paid_alpha"] == pytest.approx(0.45)


def test_without_an_emission_the_pay_stays_in_pool_share():
    pay = miner_pay({"period_paid": {"3": 0.01}, "period_tokens": {}}, None)
    assert pay["paid_alpha"] is None
    assert pay["periods"][0]["paid_alpha"] is None and pay["periods"][0]["alpha_per_mtoken"] is None
    assert miner_pay(None, 1.0) == {"periods": [], "verified_tokens": 0, "paid_alpha": 0.0}


def test_the_admin_status_carries_the_miners_pay(admin, pins, rates):  # noqa: F811
    assert admin("POST", "/admin/v1/jobs", _order(max_price_usd_per_mtoken=CEILING)).status_code == 201
    state = {"settled": [], "pending": None, "period_paid": {"5": 0.004}, "period_tokens": {"5": 400_000}}

    async def settlement(job_id):
        return state, None

    admin.records.read_settlement = settlement
    status = admin("GET", "/admin/v1/jobs/order-eval-p1/status")
    assert status.status_code == 200, status.text
    pay = status.json()["miner_pay"]
    emission = PER_BLOCK * cp.PERIOD_SECONDS / 12
    assert pay["verified_tokens"] == 400_000
    assert pay["paid_alpha"] == pytest.approx(0.004 * emission)
    assert pay["periods"][0]["alpha_per_mtoken"] == pytest.approx(0.004 * emission / 0.4)
