"""The collector's surplus burn: burn_alpha of its stake beyond what it owes
collector-rail work (whatever its age), what that work awaiting audit may
owe, what is in flight and its reserve, at most a cap per real UTC day; once,
on the owner's confirmed amount; a dry run by default; nothing at all when
what is owed cannot be known. And the read-only collector status a tier
release is checked against."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from reliquary.ledger import obligations as ob
from reliquary.ledger import payout as po
from reliquary.ledger import surplus as sp
from reliquary.validator import corpus_periods as cp
from tests.unit.test_ledger_payout import (
    COLLECTOR,
    COLLECTOR_HOT,
    NOW,
    OWNERS,
    PAID,
    WORK,
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
    handle.initialise(collector_coldkey=COLLECTOR, netuid=81, nonce=0, reason="a first journal")
    yield handle
    handle.close()


def _burn(journal, chain, *, obligations=(), cap=5 * ALPHA, execute=True, confirm="dry run", reserve=0,
          collector=(PAID,), now=NOW, wait_seconds=600.0, unsettled=0, cannot_know=(), signer=None):
    """One burn at ``now``, the clock reading ``now`` too; executing confirms
    the amount a dry run just printed and marks what its receipt reports."""
    clock = SimpleNamespace(t=float(now))

    async def sleep(seconds):
        clock.t += seconds
        chain.head += 5

    def run(execute, confirm_rao):
        return asyncio.run(sp.run_burn(
            obligations, journal=journal, chain=chain, config=_config(reserve_rao=reserve),
            collector_tasks=collector, now=now, cap_rao=cap, unsettled=unsettled, cannot_know=cannot_know,
            execute=execute, signer=(signer or FakeSigner()) if execute else None, confirm_rao=confirm_rao,
            wait_seconds=wait_seconds, poll_seconds=12.0, clock=lambda: clock.t, sleep=sleep))

    if execute and confirm == "dry run":
        confirm = run(False, None)["planned_rao"]
    receipt = run(execute, confirm if execute else None)
    if execute and not journal.read_only:
        journal.mark_receipted(receipt["reported_batch_ids"])
    return receipt


def test_a_dry_run_shows_the_surplus_and_burns_nothing(tmp_path):
    path = tmp_path / "payout.sqlite"
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    journal = po.Journal(path, read_only=True)
    receipt = _burn(journal, chain, execute=False, reserve=3 * ALPHA)
    journal.close()
    assert receipt["mode"] == po.MODE_DRY_RUN and not path.exists() and chain.submitted == []
    assert receipt["surplus_rao"] == 17 * ALPHA and receipt["planned_rao"] == 5 * ALPHA   # held to the cap
    assert receipt["burned"]["rao"] == 0 and "not initialised" in receipt["stopped"]


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


def test_the_cap_is_the_clocks_day_never_a_day_given(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    _burn(journal, chain, cap=5 * ALPHA)
    # An execution as of yesterday is refused: it would open yesterday's cap again.
    with pytest.raises(po.PayoutError):
        asyncio.run(sp.run_burn((), journal=journal, chain=chain, config=_config(), collector_tasks=(PAID,),
                                now=NOW - 86_400, cap_rao=5 * ALPHA, execute=True, signer=FakeSigner(),
                                confirm_rao=5 * ALPHA, clock=lambda: NOW))
    assert len(chain.submitted) == 1
    # A dry run may preview yesterday: its cap is yesterday's, nothing of today's is spent there.
    preview = _burn(journal, chain, cap=5 * ALPHA, execute=False, now=NOW - 86_400)
    assert preview["burned_today_rao"] == 0 and preview["pass_day"] == "2026-11-27"


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


def test_owed_counts_due_journal_rows_the_ledger_no_longer_lists_and_the_larger_amount(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=50 * ALPHA, owners=OWNERS, on_submit="fail")
    _pass(obligations, journal, chain)                    # refused once: due again, in the journal
    amount = obligations[0].amount_rao
    assert sp.owed_rao([], journal, (PAID,), NOW) == amount
    assert sp.owed_rao([replace(obligations[0], amount_rao=amount // 2)], journal, (PAID,), NOW) == amount
    assert sp.owed_rao([replace(obligations[0], amount_rao=amount * 2)], journal, (PAID,), NOW) == amount * 2
    # Work before the rail's first period is the weights' to pay: not owed.
    assert sp.owed_rao(_ledger({PAID: {"5B": 0.01}}, work=WORK - 1), journal, {PAID: WORK}, NOW) == amount


def test_native_rail_work_is_not_owed_by_the_collector(journal):
    obligations = _ledger({"order-gen-native": {"5A": 0.05}})
    chain = FakeChain(stake_rao=3 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, obligations=obligations, cap=100 * ALPHA, execute=False)
    assert receipt["owed_rao"] == 0 and receipt["planned_rao"] == 3 * ALPHA


def test_work_awaiting_audit_is_held_back_at_its_pay_ceiling():
    now = cp.period_end(WORK + 3) - 60                      # inside work period WORK + 3
    held, unbounded = sp.unsettled_rao({PAID: WORK, "order-b": WORK + 2}, [(PAID, WORK), (PAID, WORK + 2)],
                                       pay_ceilings={PAID: 0.05, "order-b": 0.1}, rates={PAID: 100.0}, now=now,
                                       days=1)
    # PAID: periods WORK+1 and WORK+3 have no archive; order-b has no frozen rate to bound it.
    assert held == 2 * ob.amount_rao(0.05, 100.0) and unbounded == ["order-b"]
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    snapshot = po.Journal(":memory:", read_only=True)
    receipt = _burn(snapshot, chain, execute=False, cap=100 * ALPHA, unsettled=held)
    assert receipt["unsettled_rao"] == held and receipt["surplus_rao"] == 20 * ALPHA - held


def test_nothing_is_burned_when_what_is_owed_cannot_be_known(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, cannot_know=["the task registry is not in the bucket"], confirm=5 * ALPHA)
    assert "cannot be known" in receipt["stopped"] and chain.submitted == [] and journal.batches() == []


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


def test_a_burn_whose_event_is_missing_counts_nothing_and_stops_for_a_person(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS, on_submit="drop")
    _burn(journal, chain, wait_seconds=1.0)
    (signed,) = chain.submitted
    chain.land(signed, events=[{"module": "System", "event": "ExtrinsicSuccess", "attributes": {}}])
    receipt = _burn(journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_MISMATCH and receipt["stopped"]
    assert receipt["burned"]["rao"] == 0
    assert receipt["landed_not_as_planned"] == [receipt["recovered"][0]["batch_id"]]
    assert len(chain.submitted) == 1
    assert receipt["burned_today_rao"] == 5 * ALPHA       # it may have burned: it still counts against the cap


def test_a_burn_receipt_totals_the_payouts_it_settles(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=50 * ALPHA, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    (signed,) = chain.submitted
    chain.land(signed)                                    # the payout lands; a burn run settles it
    chain.on_submit = "include"
    receipt = _burn(journal, chain, obligations=obligations)
    assert receipt["recovered"][0]["kind"] == po.KIND_TRANSFERS and receipt["paid"]["transfers"] == 1
    assert receipt["paid"]["rao"] == obligations[0].amount_rao and receipt["burned"]["rao"] == 5 * ALPHA


def test_a_failed_last_stake_read_still_leaves_a_receipt(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    reads = {"n": 0}
    original = chain.collector_stake_rao

    async def flaky(block):
        reads["n"] += 1
        if reads["n"] > 2:                                # the dry run's and the execution's first reads work
            raise po.ChainReadError("read timed out")
        return await original(block)

    chain.collector_stake_rao = flaky
    receipt = _burn(journal, chain)
    assert receipt["burned"]["rao"] == 5 * ALPHA and receipt["collector_stake_after_rao"] is None


def test_a_collector_held_by_the_subnet_owner_burns_nothing(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    chain.subnet = (COLLECTOR, "5OwnerHot")
    receipt = _burn(journal, chain, confirm=5 * ALPHA)
    assert "subnet owner's" in receipt["stopped"] and chain.submitted == []


def test_an_extrinsic_signed_outside_stops_the_burn(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    chain.foreign()
    receipt = _burn(journal, chain, confirm=5 * ALPHA)
    assert "expects 0" in receipt["stopped"] and chain.submitted == []


def test_no_cap_burns_nothing(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    receipt = _burn(journal, chain, cap=0)
    assert receipt["planned_rao"] == 0 and chain.submitted == [] and receipt["stopped"] is None


def test_burning_needs_the_collectors_signer_and_a_whole_cap(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    with pytest.raises(po.PayoutError):
        asyncio.run(sp.run_burn((), journal=journal, chain=chain, config=_config(), collector_tasks=(),
                                now=NOW, cap_rao=ALPHA, execute=True, signer=None, clock=lambda: NOW))
    with pytest.raises(po.PayoutError):
        asyncio.run(sp.run_burn((), journal=journal, chain=chain, config=_config(), collector_tasks=(),
                                now=NOW, cap_rao=1.5, execute=False))


def test_a_burn_event_must_be_the_collectors_on_its_subnet_and_no_more_than_asked():
    batch = {"burn_rao": 10}

    def inclusion(*attributes):
        return po.Inclusion(block_number=1, block_hash="0x1", success=True, events=(
            {"module": "SubtensorModule", "event": "AlphaBurned", "attributes": list(attributes)},))

    ok, _ = po.reconcile_burn(batch, inclusion(COLLECTOR, COLLECTOR_HOT, 10, 81), collector_coldkey=COLLECTOR,
                              collector_hotkey=COLLECTOR_HOT, netuid=81)
    assert ok
    for attributes in ((COLLECTOR, COLLECTOR_HOT, 11, 81), ("5Other", COLLECTOR_HOT, 10, 81),
                       (COLLECTOR, COLLECTOR_HOT, 10, 3)):
        bad, _ = po.reconcile_burn(batch, inclusion(*attributes), collector_coldkey=COLLECTOR,
                                   collector_hotkey=COLLECTOR_HOT, netuid=81)
        assert not bad


def test_burn_receipts_are_published_beside_payouts(tmp_path, monkeypatch, journal):
    from reliquary.ledger import receipts as rc
    from tests.unit.test_ledger_payout import _published

    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    path = po.write_receipt(_burn(journal, chain), tmp_path / "receipts", prefix="burn")
    bucket = _published(monkeypatch)
    done = asyncio.run(rc.publish([path], execute=True))
    assert done[0]["key"] == f"reliquary/public/payouts/burns/2026-11-28/{path.name}"
    entry = json.loads(bucket.objects[rc.INDEX_KEY][0])["receipts"][0]
    assert entry["kind"] == "burns" and entry["burned_rao"] == 5 * ALPHA and len(entry["paid_batches"]) == 1


def test_a_journal_from_before_burns_is_migrated_at_once_and_read_only_dry_runs_work(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript(po._TABLES)          # the tables as first released: no receipted, kind or burn_rao column
    db.execute("INSERT INTO payout_batches(batch_id, pass_day, state, transfers, created_at, updated_at) "
               "VALUES('old', '2026-11-28', 'finalized', '[]', 1, 1)")
    db.commit()
    db.close()
    snapshot = po.Journal(path, read_only=True)              # a dry run's view: migrated in memory only
    assert snapshot.batch("old")["kind"] == po.KIND_TRANSFERS and snapshot.burned_on(date(2026, 11, 28)) == 0
    snapshot.close()
    assert "kind" not in {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(payout_batches)")}
    journal = po.Journal(path)
    journal.reserve_burn("b", date(2026, 11, 28), 7)
    assert journal.batch("b")["kind"] == po.KIND_BURN and journal.in_flight_rao() == 7
    journal.close()


# -- the collector's status ------------------------------------------------------------------------------------------


class _Chain:
    def __init__(self, *, owner_hotkey="5Owner"):
        self.owner_hotkey = owner_hotkey

    async def get_current_block(self):
        return 9_600_000

    async def query_subtensor(self, name, params=None, block=None):
        return SimpleNamespace(value={"bits": int(0.866 * 2**32)} if name == "MinerBurned" else True)

    async def get_metagraph_info(self, netuid, selected_indices=None, block=None):
        return SimpleNamespace(hotkeys=["5Owner", COLLECTOR_HOT], coldkeys=["ck-owner", COLLECTOR],
                               incentives=[0.9, 0.0087], emission=[130.0, 1.28], owner_hotkey=self.owner_hotkey,
                               owner_coldkey="ck-owner", last_step=9_599_990, tempo=360, immunity_period=4096,
                               commit_reveal_weights_enabled=True, commit_reveal_period=1,
                               block_at_registration=[1, 9_590_000])

    async def get_stake(self, coldkey, hotkey, netuid, block=None):
        return SimpleNamespace(rao=12 * ALPHA)


def test_the_collector_status_reads_miner_burned_the_collector_and_the_owner_hotkey():
    from reliquary.ledger.sources import read_collector_status

    status = asyncio.run(read_collector_status(_Chain(), 81, collector_hotkey=COLLECTOR_HOT,
                                               collector_coldkey=COLLECTOR))
    assert status["miner_burned"] == pytest.approx(0.866, abs=1e-9) and status["transfer_toggle"] is True
    assert (status["uid"], status["coldkey"], status["held_by_owner"]) == (1, COLLECTOR, None)
    assert status["stake_rao"] == 12 * ALPHA and status["emission_alpha"] == 1.28
    assert (status["commit_reveal"], status["commit_reveal_period"], status["registered_at"]) == (True, 1, 9_590_000)
    missing = asyncio.run(read_collector_status(_Chain(), 81, collector_hotkey="5Nobody"))
    assert missing["uid"] is None and missing["stake_rao"] is None
    owners = asyncio.run(read_collector_status(_Chain(owner_hotkey=COLLECTOR_HOT), 81, collector_hotkey=COLLECTOR_HOT))
    assert owners["held_by_owner"] == ["it is the subnet owner hotkey"]


def test_the_tier_check_scales_from_the_previous_tier_and_reads_the_collectors_share():
    from reliquary.ledger.sources import tier_check

    unpaid = 0.945
    # From 0 to 1 %: before x 0.99.
    assert tier_check(unpaid, 0.01, unpaid * 0.99)["within"]
    # From 1 % to 10 %, a correct release: before x 0.90 / 0.99 (not before x 0.90).
    before = unpaid * 0.99
    assert tier_check(before, 0.10, unpaid * 0.90, from_fraction=0.01)["within"]
    assert not tier_check(before, 0.10, unpaid * 0.90)["within"]
    assert tier_check(unpaid * 0.90, 0.50, unpaid * 0.50, from_fraction=0.10)["within"]
    # The drift-free check: the collector's part of what is unpaid is the fraction, whatever demand did.
    drifted = 0.80
    check = tier_check(unpaid, 0.01, drifted * 0.99, collector_incentive=drifted * 0.01)
    assert not check["within"] and check["collector_share_within"] and check["landed"]
    assert not tier_check(unpaid, 0.01, unpaid, collector_incentive=0.0)["landed"]
    for bad in ({"before": 86.6}, {"fraction": 1.5}, {"from_fraction": 1.0}):
        with pytest.raises(ValueError):
            tier_check(**{"before": 0.9, "fraction": 0.5, "now": 0.45, **bad})


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


def test_burn_surplus_is_a_dry_run_by_default_and_execution_inputs_are_pinned(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    cli = _quiet_chain(monkeypatch, chain)
    base = ["burn-surplus", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT]
    result = CliRunner().invoke(cli.ledger_app, base + ["--cap", "4", "--prices", "none",
                                                        "--as-of", "2026-11-28T13:00:00"])
    assert result.exit_code == 3, result.output            # no journal yet: an execution would stop
    assert "DRY-RUN surplus burn" in result.output and "this pass burns 4.000000000 alpha" in result.output
    assert chain.submitted == [] and not (tmp_path / "home/.reliquary/payout/journal.sqlite").exists()
    assert list((tmp_path / "home/.reliquary/payout/receipts").glob("burn-2026-11-28-dry-run-*.json"))
    wallet = ["--execute", "--cap", "1", "--wallet-name", "w"]
    for args in (["--execute", "--wallet-name", "w"], wallet + ["--prices", "none"],
                 ["--confirm-rao", "5", "--cap", "1"], wallet + ["--as-of", "2026-11-27T13:00:00"],
                 wallet + ["--archives", str(tmp_path)], wallet + ["--assume-collector", PAID],
                 ["--rate", "chain"], ["--history-days", "30"]):
        refused = CliRunner().invoke(cli.ledger_app, base + args)
        assert refused.exit_code != 0, args


def _archives(tmp_path, docs):
    folder = tmp_path / "archives"
    for task, work, body in docs:
        (folder / task).mkdir(parents=True, exist_ok=True)
        (folder / task / f"{work:010d}-{work + 1:010d}.json").write_text(json.dumps(body))
    return folder


def _view(monkeypatch, *, present=True, rails=None, ceilings=None):
    from reliquary.ledger import sources

    async def read(**_):
        return sources.RegistryView(ceilings=ceilings or {}, pay_ceilings={task: 0.05 for task in (rails or {})},
                                    floored=[], collector_tasks=sorted(rails or {}),
                                    collector_from=dict(rails or {}), present=present)

    monkeypatch.setattr(sources, "read_registry_ceilings", read)


def _collect(tmp_path, journal_path, archives, **kwargs):
    from reliquary.ledger import cli

    return asyncio.run(cli.collect_burn(
        now=NOW, archives=str(archives), prices="registry", network="test", netuid=81, collector_coldkey=COLLECTOR,
        collector_hotkey=COLLECTOR_HOT, journal_path=journal_path, receipts=tmp_path / "r", cap_rao=5 * ALPHA,
        reserve_rao=0, era_period=64, assume_collector=[], execute=False, confirm_rao=None, wallet_name=None,
        wallet_path=None, wait_seconds=1.0, **kwargs))


def test_collect_burn_fails_closed_when_what_is_owed_cannot_be_known(tmp_path, monkeypatch, journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS)
    _quiet_chain(monkeypatch, chain)
    body = {"task_id": PAID, "job_id": PAID, "work_period": WORK, "rewards_by_hotkey": {"5A": 0.01}, "cap": 0.5,
            "budget": 0.5, "max_price_per_token": 1e-8, "settled_at": cp.period_end(WORK) + 600}
    archives = _archives(tmp_path, [(PAID, WORK, body)])
    priced = {PAID: [ob.Ceiling(miner_alpha_per_period=147.6, max_price_pool_share_per_token=1e-8)]}
    journal.close()
    # The registry object is absent: no task reads as on the rail, and nothing may be burned.
    _view(monkeypatch, present=False)
    receipt, _ = _collect(tmp_path, journal.path, archives)
    assert "registry is not in the bucket" in receipt["stopped"]
    # An archive of a collector-rail task priced under no known ceiling: what it owes is unknown.
    _view(monkeypatch, rails={PAID: 0})
    receipt, _ = _collect(tmp_path, journal.path, archives)
    assert "could not be valued" in receipt["stopped"]
    # Priced and on the rail: the burn holds back what it owes and what may still settle.
    _view(monkeypatch, rails={PAID: 0}, ceilings=priced)
    receipt, _ = _collect(tmp_path, journal.path, archives, unsettled_days=0.0)
    assert receipt["stopped"] is None and receipt["owed_rao"] == ob.amount_rao(0.01, 147.6)
    assert receipt["unsettled_rao"] > 0
    # A task the journal paid that the registry no longer puts on the rail.
    handle = po.Journal(journal.path)
    obligations = _ledger({PAID: {"5A": 0.01}})
    transfers, _, _ = po.plan_transfers([(o, "5ColdA") for o in obligations], available_rao=10**12,
                                        min_transfer_rao=0)
    handle.reserve("x", date(2026, 11, 28), transfers, {o.obligation_id: o for o in obligations},
                   {o.obligation_id: "5ColdA" for o in obligations})
    handle.close()
    _view(monkeypatch, rails={})
    receipt, _ = _collect(tmp_path, journal.path, archives)
    assert f"task {PAID} was paid by the collector" in receipt["stopped"]


def test_collector_status_says_whether_a_tier_landed(monkeypatch):
    from typer.testing import CliRunner

    cli = _quiet_chain(monkeypatch, None)
    base = ["collector-status", "--collector-hotkey", COLLECTOR_HOT, "--collector-coldkey", COLLECTOR]
    result = CliRunner().invoke(cli.ledger_app, base + ["--before", "0.8747", "--fraction", "0.01"])
    assert result.exit_code == 0, result.output
    assert "MinerBurned 0.866000 (86.6000%)" in result.output and "UID 1" in result.output
    assert "AS EXPECTED" in result.output and "commit-reveal on, period 1" in result.output
    assert "immune until about block 9594096" in result.output
    for bad in (["--before", "0.9"], ["--before", "86.6", "--fraction", "0.01"],
                ["--before", "0.9", "--fraction", "1.5"], ["--before", "0.9", "--fraction", "0.1",
                                                           "--from-fraction", "1"]):
        assert CliRunner().invoke(cli.ledger_app, base + bad).exit_code != 0, bad


def test_a_burn_the_nodes_pool_refuses_was_never_sent(journal):
    chain = FakeChain(stake_rao=20 * ALPHA, owners=OWNERS, on_submit="reject")
    receipt = _burn(journal, chain)
    assert receipt["batches"][0]["state"] == po.BATCH_EXPIRED and "nothing was burned" in receipt["stopped"]
    assert receipt["burned"]["rao"] == 0 and journal.burned_on(date(2026, 11, 28)) == 0   # the cap is not spent
    chain.on_submit = "include"
    assert _burn(journal, chain)["burned"]["rao"] == 5 * ALPHA and chain.stake == 15 * ALPHA
