"""The collector's surplus burn: burn_alpha of its stake beyond what it owes
collector-rail work, what is in flight and its reserve, at most a cap; once,
on the owner's confirmed amount; a dry run by default. And the read-only
collector status a tier release is checked against."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import pytest

from reliquary.ledger import payout as po
from reliquary.ledger import surplus as sp
from reliquary.validator import corpus_periods as cp
from tests.unit.test_ledger_payout import (
    COLLECTOR,
    COLLECTOR_HOT,
    NOW,
    OWNERS,
    PAID,
    PASS_DAY,
    FakeChain,
    FakeSigner,
    _config,
    _ledger,
    _pass,
)

ALPHA = 10**9


@pytest.fixture()
def journal(tmp_path):
    handle = po.Journal(tmp_path / "payout.sqlite")
    yield handle
    handle.close()


def _burn(journal, chain, *, obligations=(), cap=5 * ALPHA, execute=True, confirm="dry run", reserve=0,
          collector=(PAID,), now=NOW, wait_seconds=600.0):
    clock = SimpleNamespace(t=0.0)

    async def sleep(seconds):
        clock.t += seconds
        chain.head += 5

    def run(execute, confirm_rao):
        return asyncio.run(sp.run_burn(
            obligations, journal=journal, chain=chain, config=_config(reserve_rao=reserve),
            collector_tasks=collector, pass_day=PASS_DAY, now=now, cap_rao=cap, execute=execute,
            signer=FakeSigner() if execute else None, confirm_rao=confirm_rao, wait_seconds=wait_seconds,
            poll_seconds=12.0, clock=lambda: clock.t, sleep=sleep))

    if execute and confirm == "dry run":
        confirm = run(False, None)["planned_rao"]
    return run(execute, confirm if execute else None)


def test_a_dry_run_shows_the_surplus_and_burns_nothing(tmp_path):
    path = tmp_path / "payout.sqlite"
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    journal = po.Journal(path, read_only=True)
    receipt = _burn(journal, chain, execute=False, reserve=3 * ALPHA)
    journal.close()
    assert receipt["mode"] == po.MODE_DRY_RUN and not path.exists() and chain.submitted == []
    assert receipt["surplus_rao"] == 17 * ALPHA and receipt["planned_rao"] == 5 * ALPHA   # held to the cap
    assert receipt["burned"]["rao"] == 0


def test_a_burn_happens_once_on_the_confirmed_amount(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    for wrong in (None, 5 * ALPHA - 1):
        stopped = _burn(journal, chain, confirm=wrong)
        assert stopped["stopped"] and f"--confirm-rao {5 * ALPHA}" in stopped["stopped"] and not chain.submitted
    done = _burn(journal, chain, confirm=5 * ALPHA)
    assert done["burned"]["rao"] == 5 * ALPHA and chain.stake == 15 * ALPHA
    (batch,) = journal.batches()
    assert batch["kind"] == po.KIND_BURN and batch["state"] == po.BATCH_FINALIZED and "AlphaBurned" in batch["detail"]
    # The cap is per UTC day: run again the same day, nothing is left of it.
    again = _burn(journal, chain, cap=5 * ALPHA)
    assert again["burned_today_rao"] == 5 * ALPHA and again["planned_rao"] == 0
    assert len(chain.submitted) == 1 and chain.stake == 15 * ALPHA
    # A higher cap the same day burns only what it adds.
    more = _burn(journal, chain, cap=7 * ALPHA)
    assert more["burned"]["rao"] == 2 * ALPHA and chain.stake == 13 * ALPHA


def test_what_collector_rail_work_is_owed_is_never_burned(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    owed = sum(o.amount_rao for o in obligations)
    chain = FakeChain(stake_rao=owed + 2 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, obligations=obligations, cap=100 * ALPHA, execute=False)
    assert receipt["owed_rao"] == owed and receipt["planned_rao"] == 2 * ALPHA
    # Once paid, it is owed no more.
    _pass(obligations, journal, chain)
    after = _burn(journal, chain, obligations=obligations, cap=100 * ALPHA, execute=False)
    assert after["owed_rao"] == 0 and after["planned_rao"] == 2 * ALPHA


def test_native_rail_work_is_not_owed_by_the_collector(journal):
    obligations = _ledger({"order-gen-native": {"5A": 0.05}})
    chain = FakeChain(stake_rao=3 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, obligations=obligations, cap=100 * ALPHA, execute=False)
    assert receipt["owed_rao"] == 0 and receipt["planned_rao"] == 3 * ALPHA


def test_a_payout_in_flight_holds_the_burn_and_a_burn_in_flight_holds_the_payer(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=50 * ALPHA, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    held = _burn(journal, chain)
    assert held["stopped"] and "may still land" in held["stopped"] and held["in_flight_rao"] == obligations[0].amount_rao
    assert len(chain.submitted) == 1
    chain.head += 200
    chain.on_submit = "drop"
    burned_later = _burn(journal, chain, wait_seconds=1.0)      # the payout expires; the burn itself is dropped
    assert burned_later["recovered"][0]["state"] == po.BATCH_EXPIRED and burned_later["stopped"]
    blocked = _pass(obligations, journal, chain)
    assert blocked["stopped"] and "may still land" in blocked["stopped"]
    assert [b["kind"] for b in blocked["recovered"]] == [po.KIND_BURN]


def test_a_dropped_burn_expires_and_the_next_pass_burns_again_once(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS, on_submit="drop")
    _burn(journal, chain, wait_seconds=1.0)
    chain.head += 200
    chain.on_submit = "include"
    preview = _burn(journal, chain, execute=False)
    assert preview["burned_today_rao"] == 0 and preview["planned_rao"] == 5 * ALPHA   # the expired one frees the cap
    again = _burn(journal, chain)
    assert [b["state"] for b in again["recovered"]] == [po.BATCH_EXPIRED]
    assert again["burned"]["rao"] == 5 * ALPHA and chain.stake == 15 * ALPHA


def test_no_cap_burns_nothing(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, cap=0)
    assert receipt["planned_rao"] == 0 and chain.submitted == [] and receipt["stopped"] is None


def test_burning_needs_the_collectors_signer_and_a_whole_cap(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    with pytest.raises(po.PayoutError):
        asyncio.run(sp.run_burn((), journal=journal, chain=chain, config=_config(), collector_tasks=(),
                                pass_day=PASS_DAY, now=NOW, cap_rao=ALPHA, execute=True, signer=None))
    with pytest.raises(po.PayoutError):
        asyncio.run(sp.run_burn((), journal=journal, chain=chain, config=_config(), collector_tasks=(),
                                pass_day=PASS_DAY, now=NOW, cap_rao=1.5, execute=False))


def test_a_burn_event_must_be_the_collectors_and_no_more_than_asked():
    batch = {"burn_rao": 10}
    def inclusion(*attributes):
        return po.Inclusion(block_number=1, block_hash="0x1", success=True, events=(
            {"module": "SubtensorModule", "event": "AlphaBurned", "attributes": list(attributes)},))
    ok, _ = po.reconcile_burn(batch, inclusion(COLLECTOR, COLLECTOR_HOT, 10, 81), collector_coldkey=COLLECTOR,
                              collector_hotkey=COLLECTOR_HOT)
    assert ok
    for attributes in ((COLLECTOR, COLLECTOR_HOT, 11, 81), ("5Other", COLLECTOR_HOT, 10, 81)):
        bad, _ = po.reconcile_burn(batch, inclusion(*attributes), collector_coldkey=COLLECTOR,
                                   collector_hotkey=COLLECTOR_HOT)
        assert not bad


def test_burn_receipts_are_published_beside_payouts(tmp_path, monkeypatch, journal):
    from reliquary.ledger import receipts as rc
    from tests.unit.test_ledger_payout import _published

    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    path = po.write_receipt(_burn(journal, chain), tmp_path / "receipts", prefix="burn")
    bucket = _published(monkeypatch)
    done = asyncio.run(rc.publish([path], execute=True))
    assert done[0]["key"] == f"reliquary/public/payouts/burns/2026-11-28/{path.name}"
    index = json.loads(bucket.objects[rc.INDEX_KEY][0])
    assert index["receipts"][0]["kind"] == "burns" and index["receipts"][0]["rao"] == 5 * ALPHA


def test_a_journal_from_before_burns_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript(po._SCHEMA.replace(",\n    kind TEXT NOT NULL DEFAULT 'transfers', burn_rao INTEGER", ""))
    db.close()
    journal = po.Journal(path)
    journal.reserve_burn("b", PASS_DAY, 7)
    assert journal.batch("b")["kind"] == po.KIND_BURN and journal.in_flight_rao() == 7
    journal.close()


# -- the collector's status ------------------------------------------------------------------------------------------


class _Chain:
    async def get_current_block(self):
        return 9_600_000

    async def query_subtensor(self, name, params=None, block=None):
        return SimpleNamespace(value={"bits": int(0.866 * 2**32)} if name == "MinerBurned" else True)

    async def get_metagraph_info(self, netuid, selected_indices=None, block=None):
        return SimpleNamespace(hotkeys=["5Owner", COLLECTOR_HOT], coldkeys=["ck-owner", COLLECTOR],
                               incentives=[0.9, 0.05], emission=[130.0, 7.38], owner_hotkey="5Owner",
                               owner_coldkey="ck-owner", last_step=9_599_990)

    async def get_stake(self, coldkey, hotkey, netuid, block=None):
        return SimpleNamespace(rao=12 * ALPHA)


def test_the_collector_status_reads_miner_burned_and_the_collector():
    from reliquary.ledger.sources import read_collector_status, tier_check

    status = asyncio.run(read_collector_status(_Chain(), 81, collector_hotkey=COLLECTOR_HOT,
                                               collector_coldkey=COLLECTOR))
    assert status["miner_burned"] == pytest.approx(0.866, abs=1e-9) and status["transfer_toggle"] is True
    assert (status["uid"], status["coldkey"], status["held_by_owner"]) == (1, COLLECTOR, False)
    assert status["stake_rao"] == 12 * ALPHA and status["emission_alpha"] == 7.38
    missing = asyncio.run(read_collector_status(_Chain(), 81, collector_hotkey="5Nobody"))
    assert missing["uid"] is None and missing["stake_rao"] is None
    check = tier_check(0.866, 0.01, 0.8573)
    assert check["expected"] == pytest.approx(0.85734) and check["within"]
    assert not tier_check(0.866, 0.10, 0.866)["within"]


# -- the CLI ----------------------------------------------------------------------------------------------------------


def _quiet_chain(monkeypatch, chain):
    from reliquary.ledger import cli

    class _Subtensor(_Chain):
        async def close(self):
            return None

    async def fake_subtensor(network):
        return _Subtensor()

    monkeypatch.setattr(cli, "_subtensor", fake_subtensor)
    monkeypatch.setattr(cli, "_payout_chain", lambda subtensor, **kwargs: chain)
    return cli


def test_burn_surplus_is_a_dry_run_by_default(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    cli = _quiet_chain(monkeypatch, chain)
    result = CliRunner().invoke(cli.ledger_app, ["burn-surplus", "--collector-coldkey", COLLECTOR,
                                                 "--collector-hotkey", COLLECTOR_HOT, "--cap", "4", "--prices", "none",
                                                 "--as-of", "2026-11-28T13:00:00"])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN surplus burn" in result.output and "this pass burns 4.000000000 alpha" in result.output
    assert chain.submitted == [] and not (tmp_path / "home/.reliquary/payout/journal.sqlite").exists()
    assert list((tmp_path / "home/.reliquary/payout/receipts").glob("burn-2026-11-28-dry-run-*.json"))
    for args in (["--execute", "--wallet-name", "w"], ["--execute", "--cap", "1", "--prices", "none",
                                                       "--wallet-name", "w"], ["--confirm-rao", "5", "--cap", "1"]):
        refused = CliRunner().invoke(cli.ledger_app, ["burn-surplus", "--collector-coldkey", COLLECTOR,
                                                      "--collector-hotkey", COLLECTOR_HOT, *args])
        assert refused.exit_code != 0, args


def test_collector_status_says_whether_a_tier_landed(monkeypatch):
    from typer.testing import CliRunner

    cli = _quiet_chain(monkeypatch, None)
    result = CliRunner().invoke(cli.ledger_app, ["collector-status", "--collector-hotkey", COLLECTOR_HOT,
                                                 "--collector-coldkey", COLLECTOR, "--before", "0.8747",
                                                 "--fraction", "0.01"])
    assert result.exit_code == 0, result.output
    assert "MinerBurned 86.6000%" in result.output and "UID 1" in result.output and "AS EXPECTED" in result.output
    lone = CliRunner().invoke(cli.ledger_app, ["collector-status", "--collector-hotkey", COLLECTOR_HOT,
                                               "--before", "0.9"])
    assert lone.exit_code != 0
