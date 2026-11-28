import asyncio
import json
from dataclasses import replace
from fractions import Fraction

import pytest

from reliquary.admin import service
from reliquary.corpus import contract_spec
from reliquary.corpus.job import JobError, parse_job
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.shared import task_registry as reg
from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_settlement import rewards_for
from tests.unit.test_admin_eval_jobs import MODEL, admin  # noqa: F401
from tests.unit.test_admin_token_ceilings import CEILING, rates  # noqa: F401
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import _order, pins  # noqa: F401


def configure(monkeypatch, mode):
    monkeypatch.setenv(service.ORDER_PRICING_ENV, "auction")
    monkeypatch.setenv(
        contract_spec.MARKETS_ENV,
        json.dumps({MODEL + "|turn": {"mode": mode, "target_rate": 50, "duration_s": 1200, "new_rate_cap": 25}}),
    )


def test_admin_freezes_new_observation_keeps_quote_and_replays_old_task(admin, pins, rates, monkeypatch):  # noqa: F811
    configure(monkeypatch, "observation")
    body = _order(max_price_usd_per_mtoken=CEILING)
    created = admin("POST", "/admin/v1/jobs", body)
    assert created.status_code == 201, created.text
    entry = admin.registry["entries"][body["job_id"]]
    marker = entry.params[contract_spec.PARAM]
    assert marker["market"] == MODEL + "|turn" and marker["mode"] == "observation"
    assert marker["opening_valuation"] == {key: entry.params[key] for key in (cp.MAX_PRICE_PARAM, cp.MAX_PRICE_EMISSION_PARAM)}
    assert marker["evaluation"] is True and marker["max_tokens"] == 5 * 4 * 512
    assert (marker["budget_rao"], marker["ceiling_micro"]) == contract_spec.period_envelope(entry.params)
    assert entry.params[cp.MAX_PRICE_USD_OUTPUT_PARAM] == CEILING["output"]
    configure(monkeypatch, "pilot")
    rates["down"] = True
    assert admin("POST", "/admin/v1/jobs", body).status_code == 200
    assert admin.registry["entries"][body["job_id"]].params[contract_spec.PARAM] == marker
    # Simulation has no effect on legacy repricing/cap controls.
    updated = reg.set_cap({entry.task_id: entry}, entry.task_id, 0.01)
    assert updated[entry.task_id].params["cap"] == 0.01
    bad = replace(
        entry, params={**entry.params, contract_spec.PARAM: {**marker, "budget_rao": marker["budget_rao"] + 1}}
    )
    with pytest.raises(reg.RegistryError, match="exceeds"):
        reg.add_task({}, bad)


def test_old_orders_never_gain_a_marker_and_pilot_cannot_be_served_as_legacy(admin, pins, rates, monkeypatch):  # noqa: F811
    monkeypatch.setenv(service.ORDER_PRICING_ENV, "auction")
    body = _order(max_price_usd_per_mtoken=CEILING)
    assert admin("POST", "/admin/v1/jobs", body).status_code == 201
    configure(monkeypatch, "observation")
    assert admin("POST", "/admin/v1/jobs", body).status_code == 200
    assert contract_spec.PARAM not in admin.registry["entries"][body["job_id"]].params
    manifest, _ = asyncio.run(job_store.read_job(body["job_id"]))
    raw = manifest.to_contract()
    assert "order_contract_version" not in raw and parse_job(raw).to_contract() == raw
    marked = {**raw, "order_contract_version": 1}
    with pytest.raises(JobError, match="contract-aware"):
        parse_job(marked)
    assert parse_job(marked, allow_order_contract=True).to_contract() == marked
    configure(monkeypatch, "pilot")
    refused = admin("POST", "/admin/v1/jobs", {**body, "job_id": "order-eval-new"})
    assert refused.status_code == 503 and contract_spec.PILOT_UNAVAILABLE in refused.text
    assert "order-eval-new" not in admin.registry["entries"]
    assert asyncio.run(job_store.read_job("order-eval-new"))[0] is None


@pytest.mark.parametrize("split", [(100,), (75, 25), (1, 1, 98)])
def test_whole_order_envelope_bounds_every_period_and_replacement(split):
    # 3 rao available in each period; the total paid across every period must also fit.
    # Both output prices come from one offer snapshot, including an expensive replacement.
    params = {**cp.price_params(0.000_040, 1.0, 0.0), "cap": 3.5e-9}
    budget, ceiling = contract_spec.period_envelope(params)
    assert budget == 3
    prices = ([10] * 60) + ([min(40, ceiling)] * 40)
    consumed = 0
    for size in split:
        these = prices[consumed : consumed + size]
        consumed += size
        exact = sum(Fraction(p, 1000) for p in these)
        pool = cp.period_budget(params["cap"], cp.max_price(params), size)
        assert exact <= Fraction(pool) * 10**9
        verdicts = [{"passed": True, "hotkey": "earner", "lease_price": p / 10**12, "token_count": 1} for p in these]
        assert rewards_for(verdicts, pool)["earner"] == pytest.approx(float(exact) / 10**9, abs=1e-22)
    assert sum(Fraction(p, 1000) for p in prices) <= budget
