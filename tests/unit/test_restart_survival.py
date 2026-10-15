"""A validator restarted on this image reads every live task exactly as origin/main did.

The fixture was written by origin/main's code: one manifest per live corpus job kind
(text general/IF/safety/structured direct+thinking, competitive code, hard math,
science, eval set, SWE replay, suite telecom/envscaler) and a registry holding the
legacy RL task and corpus period tasks, with main's fingerprints. A task is immutable
once created: the same stored bytes must give the same contracts, hashes and pay bounds.
"""

import hashlib
import json
from pathlib import Path

import pytest

from reliquary.corpus.job import parse_job
from reliquary.protocol.release_contract import canonical_sha256
from reliquary.shared import task_registry as tr
from reliquary.validator import corpus_periods as cp

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures" / "restart_survival_main.json").read_text())
STORED, EXPECTED = FIXTURE["stored"], FIXTURE["expected"]


@pytest.mark.parametrize("name", sorted(STORED["manifests"]))
def test_each_live_job_kind_keeps_its_contract(name):
    job = parse_job(STORED["manifests"][name])
    contract = job.to_contract()
    assert {"contract_sha": canonical_sha256(contract), "reparse_equal": parse_job(contract) == job,
            "total_slots": job.total_slots} == EXPECTED[f"job:{name}"]


def test_the_stored_registry_renders_to_the_same_bytes():
    raw = STORED["registry"].encode()
    entries = tr.parse_registry(raw)
    assert hashlib.sha256(tr.render_registry(entries)).hexdigest() == EXPECTED["registry:stored_sha"]
    assert hashlib.sha256(raw).hexdigest() == EXPECTED["registry:stored_sha"]


def test_every_registry_entry_keeps_its_contract_price_and_period_settlement():
    entries = tr.parse_registry(STORED["registry"].encode())
    assert {f"entry:{t}" for t in entries} == {k for k in EXPECTED if k.startswith("entry:")}
    for task_id, entry in entries.items():
        assert {
            "mechanism": entry.mechanism, "profile_sha256": entry.profile_sha256,
            "contract_sha": canonical_sha256(entry.contract) if entry.contract else None,
            "params_sha": canonical_sha256(dict(entry.params)),
            "period_task": cp.is_period_task(entry),
            "pay_ceiling": cp.pay_ceiling(entry.params),
            "job_id": entry.job_id,
        } == EXPECTED[f"entry:{task_id}"], task_id


def _legacy_walk():
    """The legacy run's per-environment price walk over 40 synthetic windows, some short,
    most with a trainer wait past 5 % of the window (main ignored the wait)."""
    from reliquary.validator import emission_price as ep

    envs = ["openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"]
    archives, prices = [], []
    states = {e: ep.PriceState(price=0.5, last_good=0.5, recent_fill_prices=(0.5,)) for e in envs}
    for w in range(40):
        archives.append({
            "window_status": "completed", "window_open_round": 0, "window_close_round": 150,
            "collect_ready_round": 40 + (w % 7) * 15, "training_rounds": 110,
            "collect_ready_round_by_environment": {
                e: (30 + ((w + i) % 9) * 14 if (w + i) % 5 else None) for i, e in enumerate(envs)},
            "trainer_wait_seconds_by_environment": {
                e: float((w * 37 + i * 11) % 90) for i, e in enumerate(envs)}})
        history = {e: [o for a in archives
                       for o in [(ep.outcomes_by_environment_from_archive(a) or {}).get(e)] if o]
                   for e in envs}
        decisions = ep.advance_by_environment(states, history, ep.PRODUCTION_PRICE_PARAMS)
        states = {e: d.state for e, d in decisions.items()}
        prices.append({e: [round(d.state.price, 12), d.regime] for e, d in sorted(decisions.items())})
    return prices


def test_the_legacy_price_params_and_walk_are_mains(monkeypatch):
    import dataclasses

    from reliquary.validator import emission_price as ep

    if ep.PRICE_RISE != 1.0:
        pytest.skip("RELIQUARY_EMISSION_PRICE_RISE is set in this environment")
    params = dataclasses.asdict(ep.PRODUCTION_PRICE_PARAMS)
    params.pop("rise"), params.pop("rise_wait_fraction")
    expected = FIXTURE["expected_legacy_price"]
    assert params == expected["price_params"]
    assert json.loads(json.dumps(_legacy_walk())) == expected["walk"]
