"""A period task's input price: beside the output ceiling, prompt tokens are
priced too, and a period pays min(cap, its value), split by value."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler, apply_max_price
from reliquary.validator.corpus_settlement import rewards_for
from tests.unit.test_corpus_period_settlement import GENESIS, Archives, Records, at, verdict
from tests.unit.test_corpus_token_price_ceiling import _entry, paid

OUT = 4e-6   # pool share per output token
IN = 1e-6    # pool share per prompt token


def priced(sid, hotkey, tokens, prompt, received, passed=True):
    return {**verdict(sid, hotkey, tokens, received, passed), "prompt_token_count": prompt}


def settler(records, archives, now, cap=0.1, price=OUT, input_price=IN):
    return CorpusPeriodSettler(task_id="corpus-x", job_id="job-x", cap=cap, records=records,
                               archives=archives, oldest_pending=lambda: None,
                               genesis=lambda: GENESIS, clock=lambda: now, max_price=price,
                               max_input_price=input_price)


def test_low_supply_pays_each_hotkey_its_output_and_input_value():
    records, archives = Records(), Archives()
    records.verdicts = {"a": priced("a", "a", 300, 1000, at(3)),
                        "b": priced("b", "b", 100, 0, at(3)),
                        # An older record with no prompt count: its prompt is unpaid.
                        "c": verdict("c", "c", 100, at(3)),
                        "d": priced("d", "d", 100, 5000, at(3), passed=False)}
    assert asyncio.run(settler(records, archives, at(5, 1000)).settle_once()) == 3
    doc = archives.docs[(3, 6)]
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(300 * OUT + 1000 * IN),
                                        "b": pytest.approx(100 * OUT),
                                        "c": pytest.approx(100 * OUT)}
    assert doc["budget"] == pytest.approx(500 * OUT + 1000 * IN)
    assert doc["max_price_input_per_token"] == IN and doc["input_tokens"] == 1000
    assert doc["tokens"] == 500
    assert records.state["period_value"]["3"] == pytest.approx(500 * OUT + 1000 * IN)


def test_high_supply_pays_the_cap_split_by_value():
    records, archives = Records(), Archives()
    records.verdicts = {"a": priced("a", "a", 10_000, 60_000, at(3)),
                        "b": priced("b", "b", 20_000, 0, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=0.05).settle_once())
    value_a, value_b = 10_000 * OUT + 60_000 * IN, 20_000 * OUT
    assert value_a + value_b > 0.05
    assert archives.docs[(3, 6)]["rewards_by_hotkey"] == {
        "a": pytest.approx(0.05 * value_a / (value_a + value_b)),
        "b": pytest.approx(0.05 * value_b / (value_a + value_b))}


@pytest.mark.parametrize("cap, first, late", [
    (0.1, (100, 400), (200, 800)),          # the price binds both passes
    (0.0012, (100, 400), (200, 800)),       # the cap binds part way
    (0.0004, (100, 400), (200, 800)),       # the cap was already paid
])
def test_a_late_verdict_never_takes_a_period_past_its_value_budget(cap, first, late):
    records, archives = Records(), Archives()
    records.verdicts = {"a": priced("a", "a", *first, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), cap=cap).settle_once())
    records.verdicts["b"] = priced("b", "b", *late, at(3))
    asyncio.run(settler(records, archives, at(6, 1000), cap=cap).settle_once())
    value = (first[0] + late[0]) * OUT + (first[1] + late[1]) * IN
    assert paid(archives, 3) == pytest.approx(min(cap, value))
    if cap > first[0] * OUT + first[1] * IN:
        # Paid something late: the period's value is its whole work (as its tokens).
        assert records.state["period_value"]["3"] == pytest.approx(value)


def test_without_an_input_price_the_archive_is_as_before():
    records, archives = Records(), Archives()
    records.verdicts = {"a": priced("a", "a", 300, 10_000, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), input_price=None).settle_once())
    doc = archives.docs[(3, 6)]
    assert doc["rewards_by_hotkey"] == {"a": pytest.approx(300 * OUT)}
    assert "input_tokens" not in doc and "period_value" not in records.state


def test_an_input_price_without_an_output_price_is_ignored():
    records, archives = Records(), Archives()
    records.verdicts = {"a": priced("a", "a", 300, 10_000, at(3))}
    asyncio.run(settler(records, archives, at(5, 1000), price=None).settle_once())
    assert paid(archives, 3) == pytest.approx(0.1)


def test_an_auction_batch_does_not_price_prompts():
    batch = [{**priced("a", "a", 100, 10_000, 0.0), "lease_price": OUT, "lease_debit": 0.0}]
    assert rewards_for(batch, 1.0, price=OUT, input_price=IN) == {"a": pytest.approx(100 * OUT)}


def test_the_registry_change_reaches_the_input_price_setter():
    from types import SimpleNamespace

    out, inp = [], []
    w = SimpleNamespace(entry=SimpleNamespace(params={"cap": 0.1}), cap=0.1,
                        settler=SimpleNamespace(set_max_price=out.append,
                                                set_max_input_price=inp.append))
    params = {"cap": 0.1, **cp.price_params(0.9, 150.0, 1.0, input_alpha_per_mtoken=0.3)}
    apply_max_price(w, SimpleNamespace(task_id="corpus-x", params=params))
    assert out == [pytest.approx(0.9e-6 / 150)] and inp == [pytest.approx(0.3e-6 / 150)]


# -- params and the registry ---------------------------------------------------------


def test_usd_params_convert_at_the_alpha_rate_and_keep_the_audit_trail():
    params = cp.usd_price_params(usd_per_mtoken_input=0.5, usd_per_mtoken_output=2.0,
                                 alpha_usd=4.0, tao_usd=400.0, miner_alpha_per_period=150.0,
                                 set_at=7.0)
    assert params[cp.MAX_PRICE_ALPHA_PARAM] == pytest.approx(0.5)
    assert params[cp.MAX_PRICE_INPUT_ALPHA_PARAM] == pytest.approx(0.125)
    assert cp.max_input_price(params) == pytest.approx(0.125e-6 / 150)
    view = cp.price_view(params)
    assert view["max_price_input_alpha_per_mtoken"] == pytest.approx(0.125)
    assert view["usd"] == {"input_per_mtoken": 0.5, "output_per_mtoken": 2.0,
                           "alpha_usd": 4.0, "tao_usd": 400.0}
    free_input = cp.usd_price_params(usd_per_mtoken_input=0.0, usd_per_mtoken_output=2.0,
                                     alpha_usd=4.0, tao_usd=400.0,
                                     miner_alpha_per_period=150.0, set_at=7.0)
    assert cp.max_input_price(free_input) is None
    with pytest.raises(ValueError):
        cp.usd_price_params(usd_per_mtoken_input=0.5, usd_per_mtoken_output=0.0, alpha_usd=4.0,
                            tao_usd=400.0, miner_alpha_per_period=150.0, set_at=7.0)


def test_set_cap_sets_and_clears_the_input_and_usd_params_together():
    from reliquary.shared.task_registry import set_cap

    usd = cp.usd_price_params(usd_per_mtoken_input=0.5, usd_per_mtoken_output=2.0,
                              alpha_usd=4.0, tao_usd=400.0, miner_alpha_per_period=150.0,
                              set_at=7.0)
    entries = set_cap({"corpus-x": _entry()}, "corpus-x", None, token_price=usd)
    assert {k: entries["corpus-x"].params[k] for k in usd} == usd
    output_only = set_cap(entries, "corpus-x", None, token_price=cp.price_params(0.9, 150.0, 1.0))
    assert not set(cp.MAX_PRICE_INPUT_PARAMS + cp.MAX_PRICE_USD_PARAMS) & set(
        output_only["corpus-x"].params)
    cleared = set_cap(entries, "corpus-x", None, token_price=None)
    assert not set(cp.ALL_PRICE_PARAMS) & set(cleared["corpus-x"].params)


_USD = cp.usd_price_params(usd_per_mtoken_input=0.5, usd_per_mtoken_output=2.0, alpha_usd=4.0,
                           tao_usd=400.0, miner_alpha_per_period=150.0, set_at=7.0)


@pytest.mark.parametrize("broken", [
    {cp.MAX_PRICE_INPUT_PARAM: 1e-9, cp.MAX_PRICE_INPUT_ALPHA_PARAM: 0.15},   # no output price
    {**cp.price_params(0.9, 150.0, 1.0), cp.MAX_PRICE_INPUT_PARAM: 1e-9},       # half declared
    {**cp.price_params(0.9, 150.0, 1.0, input_alpha_per_mtoken=0.3),
     cp.MAX_PRICE_INPUT_PARAM: 1.0},                                          # inconsistent
    {**_USD, cp.MAX_PRICE_TAO_USD_PARAM: 0.0},
    {**_USD, cp.MAX_PRICE_ALPHA_USD_PARAM: 5.0},                              # not the alpha used
    {k: v for k, v in _USD.items() if k != cp.MAX_PRICE_TAO_USD_PARAM},       # half declared
    {k: v for k, v in _USD.items() if k not in cp.MAX_PRICE_INPUT_PARAMS},    # input USD unpriced
])
def test_a_malformed_input_or_usd_price_is_refused(broken):
    from reliquary.shared.task_registry import RegistryError, validate_entry

    with pytest.raises(RegistryError):
        validate_entry(_entry(**broken))


def test_a_full_usd_ceiling_is_valid():
    from reliquary.shared.task_registry import validate_entry

    validate_entry(_entry(**_USD))
