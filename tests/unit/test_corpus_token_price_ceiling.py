"""A period task's price ceiling: a period pays min(cap, price x its verified
tokens), split by tokens; the rest is never paid."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
from tests.unit.test_corpus_period_settlement import GENESIS, Archives, Records, at, verdict

PRICE = 1e-6  # pool share per verified token


def settler(records, archives, now, cap=0.1, price=PRICE):
    return CorpusPeriodSettler(task_id="corpus-x", job_id="job-x", cap=cap, records=records,
                               archives=archives, oldest_pending=lambda: None,
                               genesis=lambda: GENESIS, clock=lambda: now, max_price=price)


def paid(archives, work):
    return sum(sum(doc["rewards_by_hotkey"].values())
               for (w, _), doc in archives.docs.items() if w == work)


# -- the settler ---------------------------------------------------------------


def test_low_supply_pays_the_price_per_token_and_leaves_the_rest_unpaid():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 300, at(3)), "b": verdict("b", "b", 100, at(3))}
    s = settler(records, archives, at(5, 1000))
    assert asyncio.run(s.settle_once()) == 3
    doc = archives.docs[(3, 6)]
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(300 * PRICE),
                                        "b": pytest.approx(100 * PRICE)}
    assert doc["cap"] == 0.1 and doc["budget"] == pytest.approx(400 * PRICE)
    assert doc["max_price_per_token"] == PRICE
    assert s.last_period == {"work_period": 3, "verified_tokens": 400,
                             "paid": pytest.approx(400 * PRICE)}


def test_high_supply_pays_the_whole_cap_as_before():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 300_000, at(3)),
                        "b": verdict("b", "b", 100_000, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=0.1, price=1e-6).settle_once())
    assert archives.docs[(3, 6)]["rewards_by_hotkey"] == {"a": pytest.approx(0.075),
                                                          "b": pytest.approx(0.025)}


@pytest.mark.parametrize("cap, price, first, late, expected_late", [
    (0.1, 1e-6, 100, 200, 200e-6),   # price binds: the late tokens earn the price
    (0.1, 1e-3, 200, 100, 0.0),      # the cap was already paid: nothing more
    (0.25, 1e-3, 200, 100, 0.05),    # the cap binds part way
])
def test_a_period_settled_in_several_passes_never_pays_past_its_budget(
        cap, price, first, late, expected_late):
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", first, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=cap, price=price).settle_once())
    # A late verdict for the same period (its close missed it).
    records.verdicts["b"] = verdict("b", "b", late, at(3))
    asyncio.run(settler(records, archives, at(6, 1000), cap=cap, price=price).settle_once())
    total = paid(archives, 3)
    assert total <= cp.period_budget(cap, price, first + late) * (1 + 1e-12)
    assert total == pytest.approx(min(cap, price * first) + expected_late)
    assert sorted(records.state["settled"]) == ["a", "b"]
    assert records.state["period_paid"]["3"] == pytest.approx(total)


def test_an_earlier_pass_by_an_older_binary_counts_as_its_full_cap():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 100, at(3))}
    # Settled without the ceiling by a binary that kept no pay per period.
    asyncio.run(settler(records, archives, at(5, 1000), price=None).settle_once())
    records.state.pop("period_paid")
    records.verdicts["b"] = verdict("b", "b", 100, at(3))
    asyncio.run(settler(records, archives, at(6, 1000), cap=0.1, price=1.0).settle_once())
    assert paid(archives, 3) == pytest.approx(0.1)


def test_a_price_change_applies_to_the_periods_settled_after_it():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 100, at(3))}
    s = settler(records, archives, at(5, 1000))
    asyncio.run(s.settle_once())
    s.set_max_price(2 * PRICE)
    s._clock = lambda: at(6, 1000)
    records.verdicts["b"] = verdict("b", "b", 100, at(4))
    asyncio.run(s.settle_once())
    s.set_max_price(None)
    s._clock = lambda: at(7, 1000)
    records.verdicts["c"] = verdict("c", "c", 100, at(5))
    asyncio.run(s.settle_once())
    assert paid(archives, 3) == pytest.approx(100 * PRICE)
    assert paid(archives, 4) == pytest.approx(200 * PRICE)
    assert paid(archives, 5) == pytest.approx(0.1)


def test_a_task_without_a_price_writes_the_same_archive_as_before():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 300, at(3)), "b": verdict("b", "b", 100, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), price=None).settle_once())
    doc = archives.docs[(3, 6)]
    assert set(doc) == {"schema", "task_id", "job_id", "mechanism", "work_period",
                        "entry_period", "rewards_by_hotkey", "tokens", "verdicts", "cap"}
    assert sum(doc["rewards_by_hotkey"].values()) == pytest.approx(0.1)


def test_the_last_period_paid_is_read_back_after_a_restart():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 100, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000)).settle_once())
    fresh = settler(records, archives, at(5, 1100))
    asyncio.run(fresh.settle_once())
    assert fresh.last_period["paid"] == pytest.approx(100 * PRICE)


# -- the weight setter -----------------------------------------------------------


def test_the_weight_setter_pays_a_priced_archive_as_written():
    """Nothing changes for the weight setters: the archive already holds the
    reduced amounts, which is less than the cap they bound it by."""
    from reliquary.validator.weight_only import WeightOnlyValidator

    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 300, at(3)), "b": verdict("b", "b", 100, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000)).settle_once())

    class Stored:
        async def list(self, task_id):
            return sorted(archives.docs)

        async def read(self, task_id, work, entry):
            return archives.docs.get((work, entry))

    params = {"cap": 0.1, "settlement": cp.SETTLEMENT_PERIOD_EMA,
              **cp.price_params(0.9, 450.0, 1.0)}
    declared = {"corpus-x": SimpleNamespace(params=params)}
    weights = asyncio.run(WeightOnlyValidator._period_weights(
        declared, archives=Stored(), now=at(6, 5), genesis=GENESIS))
    assert weights["corpus-x"] == {"a": pytest.approx(cp.PERIOD_ALPHA * 300 * PRICE),
                                   "b": pytest.approx(cp.PERIOD_ALPHA * 100 * PRICE)}
    assert WeightOnlyValidator._period_caps_by_task(declared) == {"corpus-x": 0.1}


# -- the registry ----------------------------------------------------------------


def _entry(cap=0.1, **params):
    from reliquary.shared.task_registry import TaskEntry

    base = {"start": cap, "decay": 0.98, "rounds_per_step": 1, "deadband": 0.0,
            "snap": 0.0, "floor": cap, "cap": cap, "median_rounds": 1,
            "last_good_fills": 1, "settlement": cp.SETTLEMENT_PERIOD_EMA, **params}
    return TaskEntry(task_id="corpus-x", profile_id="p", profile_sha256="0" * 64,
                     mechanism="corpus-generation", params=base, status="active",
                     retired_at=None, job_id="job-x")


def test_price_params_convert_alpha_per_million_tokens_to_pool_share():
    params = cp.price_params(0.9, 450.0, 1234.0)
    assert params[cp.MAX_PRICE_PARAM] == pytest.approx(0.9 / 1e6 / 450.0)
    assert params[cp.MAX_PRICE_ALPHA_PARAM] == 0.9
    assert params[cp.MAX_PRICE_EMISSION_PARAM] == 450.0
    assert params[cp.MAX_PRICE_SET_AT_PARAM] == 1234.0
    with pytest.raises(ValueError):
        cp.price_params(0.0, 450.0, 0.0)
    with pytest.raises(ValueError):
        cp.price_params(0.9, 0.0, 0.0)


def test_set_cap_sets_changes_and_clears_the_price_keeping_the_cap():
    from reliquary.shared.task_registry import set_cap

    entries = {"corpus-x": _entry(cap=0.15)}
    priced = set_cap(entries, "corpus-x", None, token_price=cp.price_params(0.9, 450.0, 1.0))
    assert priced["corpus-x"].params["cap"] == 0.15
    assert cp.max_price(priced["corpus-x"].params) == pytest.approx(0.9 / 1e6 / 450.0)
    cleared = set_cap(priced, "corpus-x", None, token_price=None)
    assert cp.max_price(cleared["corpus-x"].params) is None
    assert not set(cp.MAX_PRICE_PARAMS) & set(cleared["corpus-x"].params)
    # Omitted, the price is kept through a cap change.
    recapped = set_cap(priced, "corpus-x", 0.2)
    assert recapped["corpus-x"].params[cp.MAX_PRICE_ALPHA_PARAM] == 0.9


def test_a_price_on_a_task_not_settled_by_period_is_refused():
    from dataclasses import replace

    from reliquary.shared.task_registry import RegistryError, set_cap

    entry = _entry()
    unperiodic = replace(entry, params={k: v for k, v in entry.params.items()
                                        if k != "settlement"})
    with pytest.raises(RegistryError, match="period"):
        set_cap({"corpus-x": unperiodic}, "corpus-x", None,
                token_price=cp.price_params(0.9, 450.0, 1.0))


@pytest.mark.parametrize("broken", [
    {cp.MAX_PRICE_PARAM: 1e-9},                                   # half declared
    {**cp.price_params(0.9, 450.0, 1.0), cp.MAX_PRICE_PARAM: 1.0},  # inconsistent
    {**cp.price_params(0.9, 450.0, 1.0), cp.MAX_PRICE_ALPHA_PARAM: "0.9"},
])
def test_a_malformed_price_is_refused(broken):
    from reliquary.shared.task_registry import RegistryError, validate_entry

    with pytest.raises(RegistryError):
        validate_entry(_entry(**broken))


def test_a_priced_registry_round_trips_and_an_unpriced_one_is_unchanged():
    from reliquary.shared.task_registry import parse_registry, render_registry

    plain = {"corpus-x": _entry()}
    assert render_registry(parse_registry(render_registry(plain))) == render_registry(plain)
    priced = {"corpus-x": _entry(**cp.price_params(0.9, 450.0, 1.0))}
    raw = render_registry(priced)
    assert parse_registry(raw)["corpus-x"].params == priced["corpus-x"].params
    # The price is plain params: a binary that does not know it reads the
    # entry's controller fields exactly as before.
    body = json.loads(raw)["tasks"]["corpus-x"]
    assert body["incentive"]["params"][cp.MAX_PRICE_ALPHA_PARAM] == 0.9



# -- the operator CLI --------------------------------------------------------------


def _cli(monkeypatch, *args):
    from typer.testing import CliRunner

    from reliquary.cli.main import app
    from reliquary.infrastructure import task_registry_store as store

    calls = []

    async def _set(task_id, cap, **kwargs):
        calls.append((task_id, cap, kwargs))

    monkeypatch.setattr(store, "set_task_cap", _set)
    return CliRunner().invoke(app, ["tasks", "set-cap", "--task-id", "corpus-x", *args]), calls


def test_the_cli_sets_a_price_from_a_given_emission_keeping_the_cap(monkeypatch):
    result, calls = _cli(monkeypatch, "--max-price-alpha-per-mtoken", "0.9",
                         "--miner-emission-alpha-per-day", "2952")
    assert result.exit_code == 0, result.output
    (task_id, cap, kwargs), = calls
    assert (task_id, cap) == ("corpus-x", None)
    price = kwargs["token_price"]
    per_period = 2952 * cp.PERIOD_SECONDS / 86400
    assert price[cp.MAX_PRICE_EMISSION_PARAM] == pytest.approx(per_period)
    assert price[cp.MAX_PRICE_PARAM] == pytest.approx(0.9e-6 / per_period)
    assert "0.9 alpha per million" in result.output


def test_the_cli_reads_the_miner_emission_on_chain(monkeypatch):
    import reliquary.cli.main as cli

    async def chain():
        return {"alpha_out_per_block": 1.0, "owner_cut": 0.18,
                "miner_alpha_per_block": 0.41, "netuid": 81}

    monkeypatch.setattr(cli, "_read_miner_emission", chain)
    result, calls = _cli(monkeypatch, "--max-price-alpha-per-mtoken", "0.9")
    assert result.exit_code == 0, result.output
    price = calls[0][2]["token_price"]
    assert price[cp.MAX_PRICE_EMISSION_PARAM] == pytest.approx(0.41 * cp.PERIOD_SECONDS / 12)


def test_the_cli_clears_a_price_and_refuses_contradictions(monkeypatch):
    result, calls = _cli(monkeypatch, "--clear-max-price")
    assert result.exit_code == 0, result.output
    assert calls == [("corpus-x", None, {**calls[0][2], "token_price": None})]
    result, calls = _cli(monkeypatch, "--clear-max-price", "--max-price-alpha-per-mtoken", "1")
    assert result.exit_code == 1 and not calls
    result, calls = _cli(monkeypatch)
    assert result.exit_code == 1 and not calls


def test_a_cap_alone_sends_no_price(monkeypatch):
    result, calls = _cli(monkeypatch, "--cap", "0.2")
    assert result.exit_code == 0, result.output
    assert calls[0][1] == 0.2 and "token_price" not in calls[0][2]
