"""The J+2 payer: due obligations of tasks on the collector rail are paid from
the collector's stake by transfer_stake, each once, through crashes, dropped
sends, failures and replays; a dry run signs nothing and writes no journal."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from reliquary.ledger import obligations as ob
from reliquary.ledger import payout as po
from reliquary.validator import corpus_periods as cp

E = 147.60082398718242
COLLECTOR, COLLECTOR_HOT = "5Collector", "5CollectorHot"
PAID, NATIVE = "order-gen-collector", "order-gen-native"
WORK = 23_817                       # ended 2026-11-26: due 2026-11-28 12:00 UTC
PASS_DAY = date(2026, 11, 28)
NOW = datetime(2026, 11, 28, 12, 5, tzinfo=timezone.utc).timestamp()
TERMS = ob.PayTerms()
T0 = cp.period_end(WORK) - 3 * 86_400   # block 0's time in the fake chain


def _ceilings(*tasks):
    return {task: [ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=1e-8)] for task in tasks}


def _ledger(rewards_by_task, *, work=WORK, settled_after=600.0):
    refs = []
    for i, (task, rewards) in enumerate(sorted(rewards_by_task.items())):
        doc = {"task_id": task, "job_id": task, "work_period": work, "rewards_by_hotkey": rewards, "cap": 0.5,
               "budget": 0.5, "max_price_per_token": 1e-8, "tokens": 10**6}
        refs.append(ob.ArchiveRef(task_id=task, work_period=work, entry_period=work + 1, document=doc,
                                  settled_at=cp.period_end(work) + settled_after + i))
    return ob.build_ledger(refs, ceilings=_ceilings(*rewards_by_task)).obligations


class FakeChain:
    """A chain of 12 s blocks: the collector's stake and nonce, owners, and
    sends that land, fail, are dropped, or land and then raise."""

    def __init__(self, *, stake_rao, owners, head=21_000, on_submit="include"):
        self.head, self.nonce, self.stake = head, 0, stake_rao
        self.owners = owners
        self.blocks: dict[int, list[tuple[str, po.Inclusion]]] = {}
        self.submitted: list[po.SignedBatch] = []
        self.on_submit = on_submit
        self.calls: list[str] = []

    async def finalized_block(self):
        return self.head

    async def collector_stake_rao(self, block):
        return self.stake

    async def collector_nonce(self, block):
        return self.nonce

    async def hotkey_owner(self, hotkey, block):
        owner = self.owners.get(hotkey)
        return owner(block) if callable(owner) else owner

    async def block_at(self, at):
        return int((at - T0) // 12)

    async def find_extrinsic(self, extrinsic_hash, first_block, last_block):
        self.calls.append("find_extrinsic")
        for number in range(first_block, last_block + 1):
            for digest, inclusion in self.blocks.get(number, []):
                if digest == extrinsic_hash:
                    return inclusion
        return None

    async def submit(self, signed):
        self.submitted.append(signed)
        if self.on_submit == "raise_before":
            raise ConnectionError("dropped before the node had it")
        if self.on_submit in ("include", "fail", "raise_after_include"):
            self.land(signed, success=self.on_submit != "fail")
        if self.on_submit == "raise_after_include":
            raise ConnectionError("dropped after the node had it")

    def land(self, signed, *, success=True, events=None):
        self.head += 3
        self.nonce = max(self.nonce, signed.nonce + 1)
        transfers = signed.payload
        if events is None:
            events = ([{"module": "SubtensorModule", "event": "StakeTransferred",
                        "attributes": [COLLECTOR, t.coldkey, COLLECTOR_HOT, 81, 81, t.amount_rao]} for t in transfers]
                      if success else []) + [{"module": "System",
                                              "event": "ExtrinsicSuccess" if success else "ExtrinsicFailed",
                                              "attributes": {}}]
        if success:
            self.stake -= sum(t.amount_rao for t in transfers)
        self.blocks.setdefault(self.head, []).append(
            (signed.extrinsic_hash, po.Inclusion(block_number=self.head, block_hash=f"0x{self.head:064x}",
                                                 success=success, events=tuple(events), fee_rao=125_000,
                                                 error=None if success else "{'name': 'InsufficientStake'}")))

    def foreign(self):
        """Another extrinsic signed with the collector's key takes the next nonce."""
        self.head += 1
        self.nonce += 1


class FakeSigner:
    coldkey = COLLECTOR

    def __init__(self):
        self.signed = []

    async def sign(self, transfers, *, nonce, era_block, era_period):
        body = json.dumps([[t.coldkey, t.amount_rao, list(t.obligation_ids)] for t in transfers] + [nonce, era_block])
        signed = po.SignedBatch(extrinsic_hash="0x" + hashlib.sha256(body.encode()).hexdigest(), nonce=nonce,
                                era_block=era_block, era_period=era_period, payload=list(transfers))
        self.signed.append(signed)
        return signed


def _config(**overrides):
    values = {"netuid": 81, "collector_coldkey": COLLECTOR, "collector_hotkey": COLLECTOR_HOT,
              "min_transfer_rao": 0, "batch_size": 32, "reserve_rao": 0, "era_period": 64}
    values.update(overrides)
    return po.PayoutConfig(**values)


def _pass(obligations, journal, chain, *, execute=True, signer=None, now=NOW, day=PASS_DAY, config=None,
          collector=(PAID,), wait_seconds=600.0):
    clock = SimpleNamespace(t=0.0)

    async def sleep(seconds):
        clock.t += seconds
        chain.head += 5     # time passes on the chain too

    return asyncio.run(po.run_pass(
        obligations, journal=journal, chain=chain, config=config or _config(), collector_tasks=collector,
        pass_day=day, terms=TERMS, now=now, execute=execute, signer=signer or (FakeSigner() if execute else None),
        wait_seconds=wait_seconds, poll_seconds=12.0, clock=lambda: clock.t, sleep=sleep))


OWNERS = {"5A": "5ColdA", "5B": "5ColdB", "5C": "5ColdC"}


@pytest.fixture()
def journal(tmp_path):
    handle = po.Journal(tmp_path / "payout.sqlite")
    yield handle
    handle.close()


def _states(journal, obligations):
    return {oid: row["state"] for oid, row in journal.rows(o.obligation_id for o in obligations).items()}


# -- a dry run -------------------------------------------------------------------------------------------------------


def test_a_dry_run_plans_and_signs_nothing_and_writes_no_journal(tmp_path):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}, NATIVE: {"5C": 0.05}})
    path = tmp_path / "payout.sqlite"
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    journal = po.Journal(path, read_only=True)
    receipt = _pass(obligations, journal, chain, execute=False)
    journal.close()
    assert receipt["mode"] == po.MODE_DRY_RUN and not path.exists()
    assert chain.submitted == [] and receipt["batches"] == []
    assert {t["coldkey"] for t in receipt["transfers"]} == {"5ColdA", "5ColdB"}
    assert receipt["planned"]["rao"] == ob.amount_rao(0.01, E) + ob.amount_rao(0.02, E)
    # The native task's work is the weights': listed, never paid here.
    assert receipt["native_rail"]["obligations"] == 1 and receipt["native_rail"]["rao"] == ob.amount_rao(0.05, E)


def test_executing_needs_the_collectors_signer_and_a_writable_journal(tmp_path, journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    with pytest.raises(po.PayoutError):
        asyncio.run(po.run_pass(obligations, journal=journal, chain=chain, config=_config(), collector_tasks=[PAID],
                                pass_day=PASS_DAY, terms=TERMS, now=NOW, execute=True, signer=None))
    stranger = FakeSigner()
    stranger.coldkey = "5Someone"
    with pytest.raises(po.PayoutError):
        _pass(obligations, journal, chain, signer=stranger)
    read_only = po.Journal(tmp_path / "other.sqlite", read_only=True)
    with pytest.raises(po.PayoutError):
        _pass(obligations, read_only, chain)
    read_only.close()
    assert chain.submitted == []


# -- paying once ------------------------------------------------------------------------------------------------------


def test_a_pass_pays_each_obligation_once_and_a_replay_pays_nothing(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    first = _pass(obligations, journal, chain)
    assert len(chain.submitted) == 1 and first["paid"]["transfers"] == 2
    assert first["batches"][0]["state"] == po.BATCH_FINALIZED
    assert set(_states(journal, obligations).values()) == {ob.RECONCILED}
    paid = sum(o.amount_rao for o in obligations)
    assert chain.stake == 10**12 - paid and first["paid"]["rao"] == paid
    again = _pass(obligations, journal, chain)
    assert len(chain.submitted) == 1 and again["planned"]["transfers"] == 0
    assert again["already_past_due"]["obligations"] == 2 and chain.stake == 10**12 - paid


def test_only_the_collector_rail_is_ever_paid(journal):
    obligations = _ledger({PAID: {"5A": 0.01}, NATIVE: {"5A": 0.03, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(obligations, journal, chain)
    paid_ids = {oid for b in receipt["batches"] for t in b["transfers"] for oid in t["obligation_ids"]}
    assert paid_ids == {o.obligation_id for o in obligations if o.task_id == PAID}
    assert all(o.obligation_id not in _states(journal, obligations) for o in obligations if o.task_id == NATIVE)


def test_a_send_that_raised_after_landing_is_read_from_the_chain_never_resent(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="raise_after_include")
    receipt = _pass(obligations, journal, chain)
    assert len(chain.submitted) == 1 and receipt["batches"][0]["state"] == po.BATCH_FINALIZED
    assert set(_states(journal, obligations).values()) == {ob.RECONCILED}


def test_a_crash_after_sending_is_settled_by_the_next_pass_without_resending(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    # A pass that died right after sending: reserved, hash kept, sent, never told.
    transfers, _, _ = po.plan_transfers([(o, OWNERS[o.hotkey]) for o in obligations], available_rao=10**12,
                                        min_transfer_rao=0)
    journal.reserve("crashed", PASS_DAY, transfers, {o.obligation_id: o for o in obligations},
                    {o.obligation_id: OWNERS[o.hotkey] for o in obligations})
    signed = asyncio.run(FakeSigner().sign(transfers, nonce=0, era_block=chain.head, era_period=64))
    journal.mark_submitted("crashed", signed)
    chain.land(signed)
    receipt = _pass(obligations, journal, chain)
    assert [b["state"] for b in receipt["recovered"]] == [po.BATCH_FINALIZED]
    assert chain.submitted == [] and receipt["planned"]["transfers"] == 0
    assert set(_states(journal, obligations).values()) == {ob.RECONCILED}


def test_a_prepared_batch_never_signed_is_released_and_paid_once(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    transfers, _, _ = po.plan_transfers([(o, OWNERS[o.hotkey]) for o in obligations], available_rao=10**12,
                                        min_transfer_rao=0)
    journal.reserve("unsigned", PASS_DAY, transfers, {o.obligation_id: o for o in obligations},
                    {o.obligation_id: OWNERS[o.hotkey] for o in obligations})
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_EXPIRED
    assert len(chain.submitted) == 1 and receipt["paid"]["transfers"] == 1


def test_a_dropped_send_waits_for_its_window_then_is_paid_once(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    first = _pass(obligations, journal, chain, wait_seconds=30.0)
    assert first["stopped"] and first["batches"][0]["state"] == po.BATCH_PENDING
    # Still inside its window: the next pass sends nothing new.
    second = _pass(obligations, journal, chain)
    assert second["stopped"] and len(chain.submitted) == 1
    assert set(_states(journal, obligations).values()) == {ob.SUBMITTED}
    # Its window over, it can never land: due again, paid once.
    chain.head += 200
    chain.on_submit = "include"
    third = _pass(obligations, journal, chain)
    assert [b["state"] for b in third["recovered"]] == [po.BATCH_EXPIRED]
    assert len(chain.submitted) == 2 and third["paid"]["transfers"] == 1
    assert chain.stake == 10**12 - obligations[0].amount_rao


def test_a_failed_batch_releases_its_obligations_for_the_next_pass(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="fail")
    failed = _pass(obligations, journal, chain)
    assert failed["batches"][0]["state"] == po.BATCH_FAILED and failed["paid"]["transfers"] == 0
    assert set(_states(journal, obligations).values()) == {ob.DUE}
    chain.on_submit = "include"
    paid = _pass(obligations, journal, chain)
    assert paid["paid"]["transfers"] == 1 and len(chain.submitted) == 2
    assert journal.rows([obligations[0].obligation_id])[obligations[0].obligation_id]["attempts"] == 2


def test_a_nonce_used_by_another_extrinsic_stops_the_payer_and_keeps_the_obligations_held(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    chain.foreign()
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_FOREIGN and receipt["stopped"]
    assert len(chain.submitted) == 1
    assert set(_states(journal, obligations).values()) == {ob.SUBMITTED}
    # Every later pass stays stopped until a person decides.
    later = _pass(obligations, journal, chain)
    assert later["stopped"] and "settled by hand" in later["stopped"] and len(chain.submitted) == 1
    (batch,) = journal.foreign()
    journal.settle_foreign(batch["batch_id"], paid=False, detail="read the chain: the transfer was never made")
    chain.on_submit = "include"
    resumed = _pass(obligations, journal, chain)
    assert resumed["stopped"] is None and resumed["paid"]["transfers"] == 1 and len(chain.submitted) == 2
    with pytest.raises(po.PayoutError):
        journal.settle_foreign(batch["batch_id"], paid=True, detail="twice")


# -- what is planned --------------------------------------------------------------------------------------------------


def test_funding_goes_oldest_first_and_stops_at_the_first_that_does_not_fit():
    old = _ledger({PAID: {"5A": 0.01}}, work=WORK - 20)
    new = _ledger({PAID: {"5B": 0.001, "5C": 0.02}})
    pairs = [(o, OWNERS[o.hotkey]) for o in old + new]
    room = old[0].amount_rao + ob.amount_rao(0.001, E)
    transfers, deferred, unfunded = po.plan_transfers(pairs, available_rao=room - 1, min_transfer_rao=0)
    # 5A (oldest) fits; 5B and 5C settled together after it: 5B comes first and does not fit, so neither is paid.
    assert [t.coldkey for t in transfers] == ["5ColdA"]
    assert {o.hotkey for o in unfunded} == {"5B", "5C"} and deferred == []


def test_the_reserve_is_never_paid_out(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=obligations[0].amount_rao + 10, owners=OWNERS)
    receipt = _pass(obligations, journal, chain, config=_config(reserve_rao=11))
    assert receipt["planned"]["transfers"] == 0 and receipt["unfunded"]["obligations"] == 1 and not chain.submitted


def test_a_destination_owed_under_the_minimum_waits_and_adds_up(journal):
    day_one = _ledger({PAID: {"5A": 0.01}})
    config = _config(min_transfer_rao=day_one[0].amount_rao + 1)
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    first = _pass(day_one, journal, chain, config=config)
    assert first["deferred_below_minimum"]["obligations"] == 1 and not chain.submitted
    day_two = day_one + _ledger({PAID: {"5A": 0.01}}, work=WORK + 20)
    second = _pass(day_two, journal, chain, config=config, now=NOW + 86_400, day=date(2026, 11, 29))
    assert second["paid"]["transfers"] == 1 and second["paid"]["rao"] == sum(o.amount_rao for o in day_two)
    assert len(second["batches"][0]["transfers"][0]["obligation_ids"]) == 2


def test_the_destination_is_the_owner_when_earned_and_is_kept_for_retries(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    earned_block = int((obligations[0].earned_at - T0) // 12)
    owners = {"5A": lambda block: "5ColdThen" if block <= earned_block else "5ColdNow"}
    chain = FakeChain(stake_rao=10**12, owners=owners, on_submit="fail")
    failed = _pass(obligations, journal, chain)
    assert failed["batches"][0]["transfers"][0]["coldkey"] == "5ColdThen"
    chain.owners = {"5A": "5ColdLater"}
    chain.on_submit = "include"
    paid = _pass(obligations, journal, chain)
    assert paid["batches"][0]["transfers"][0]["coldkey"] == "5ColdThen"


def test_a_hotkey_without_an_owner_is_not_paid(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5Gone": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(obligations, journal, chain)
    assert receipt["ownerless"]["obligations"] == 1 and receipt["paid"]["transfers"] == 1


def test_more_transfers_than_a_batch_holds_go_in_batches_with_the_next_nonce_each(journal):
    hotkeys = {f"5H{i}": 0.001 * (i + 1) for i in range(5)}
    obligations = _ledger({PAID: hotkeys})
    chain = FakeChain(stake_rao=10**12, owners={hk: f"5Cold{hk}" for hk in hotkeys})
    receipt = _pass(obligations, journal, chain, config=_config(batch_size=2))
    assert [len(b["transfers"]) for b in receipt["batches"]] == [2, 2, 1]
    assert [s.nonce for s in chain.submitted] == [0, 1, 2]
    assert receipt["paid"]["transfers"] == 5


def test_obligations_not_due_yet_are_not_paid(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    early = _pass(obligations, journal, chain, now=NOW - 3600)
    assert early["due"]["obligations"] == 0 and not chain.submitted


# -- the journal and the receipts -----------------------------------------------------------------------------------------


def test_one_payer_at_a_time(tmp_path, journal):
    with pytest.raises(po.PayoutError):
        po.Journal(journal.path)


def test_the_journal_refuses_to_reserve_an_obligation_twice(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    transfers, _, _ = po.plan_transfers([(o, "5ColdA") for o in obligations], available_rao=10**12,
                                        min_transfer_rao=0)
    by_id, cold = {o.obligation_id: o for o in obligations}, {o.obligation_id: "5ColdA" for o in obligations}
    journal.reserve("one", PASS_DAY, transfers, by_id, cold)
    with pytest.raises(po.PayoutError):
        journal.reserve("two", PASS_DAY, transfers, by_id, cold)
    assert [b["batch_id"] for b in journal.batches()] == ["one"]


def test_events_that_do_not_match_leave_a_paid_batch_finalized_not_reconciled(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    (signed,) = chain.submitted
    chain.land(signed, events=[{"module": "System", "event": "ExtrinsicSuccess", "attributes": {}}])
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_FINALIZED
    assert "differ" in receipt["recovered"][0]["detail"]
    assert set(_states(journal, obligations).values()) == {ob.FINALIZED}


def test_reconciling_reads_mapped_and_listed_event_fields():
    inclusion = po.Inclusion(block_number=1, block_hash="0x1", success=True, events=(
        {"module": "SubtensorModule", "event": "StakeTransferred",
         "attributes": {"origin_coldkey": COLLECTOR, "destination_coldkey": "5ColdA", "hotkey": COLLECTOR_HOT}},
        {"module": "SubtensorModule", "event": "StakeTransferred",
         "attributes": [COLLECTOR, "5ColdB", COLLECTOR_HOT, 81, 81, 5]}))
    ok, _ = po.reconcile_events([{"coldkey": "5ColdA"}, {"coldkey": "5ColdB"}], inclusion,
                                collector_coldkey=COLLECTOR, collector_hotkey=COLLECTOR_HOT)
    assert ok
    wrong, why = po.reconcile_events([{"coldkey": "5ColdA"}, {"coldkey": "5ColdB"}], inclusion,
                                     collector_coldkey="5Other", collector_hotkey=COLLECTOR_HOT)
    assert not wrong and "not the collector's" in why


def test_a_receipt_is_written_beside_earlier_ones_never_over_them(tmp_path, journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    receipt = _pass(obligations, journal, FakeChain(stake_rao=10**12, owners=OWNERS))
    first = po.write_receipt(receipt, tmp_path / "receipts")
    second = po.write_receipt(receipt, tmp_path / "receipts")
    assert first != second and json.loads(first.read_text())["schema"] == po.PAYOUT_RECEIPT_SCHEMA


@pytest.mark.parametrize("bad", [{"era_period": 63}, {"batch_size": 0}, {"min_transfer_rao": -1},
                                 {"collector_hotkey": ""}])
def test_a_payout_config_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        _config(**bad)


# -- the CLI ----------------------------------------------------------------------------------------------------------


def test_ledger_pay_is_a_dry_run_by_default(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    folder = tmp_path / "archives" / PAID
    folder.mkdir(parents=True)
    doc = {"task_id": PAID, "job_id": PAID, "work_period": WORK, "rewards_by_hotkey": {"5A": 0.01}, "cap": 0.5,
           "budget": 0.5, "max_price_per_token": 1e-8, "settled_at": cp.period_end(WORK) + 600}
    (folder / f"{WORK:010d}-{WORK + 1:010d}.json").write_text(json.dumps(doc))
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)

    class _Subtensor:
        async def close(self):
            return None

    async def fake_subtensor(network):
        return _Subtensor()

    async def fake_min(subtensor):
        return 0

    monkeypatch.setattr(cli, "_subtensor", fake_subtensor)
    monkeypatch.setattr(cli, "_payout_chain", lambda subtensor, **kwargs: chain)
    monkeypatch.setattr(cli, "_min_transfer_rao", fake_min)
    journal = tmp_path / "journal.sqlite"
    result = CliRunner().invoke(cli.ledger_app, [
        "pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT, "--journal", str(journal),
        "--receipts", str(tmp_path / "receipts"), "--archives", str(tmp_path / "archives"), "--prices", "none",
        "--rate", str(E), "--assume-collector", PAID, "--as-of", "2026-11-28T12:05:00"])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN pass of 2026-11-28" in result.output and "planned 1 transfer(s)" in result.output
    assert not journal.exists() and chain.submitted == []
    (receipt,) = (tmp_path / "receipts").glob("payout-2026-11-28-dry-run-*.json")
    assert json.loads(receipt.read_text())["sources"]["assumed_collector_tasks"] == [PAID]


def test_ledger_pay_executes_only_from_the_registry_with_a_wallet(tmp_path):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    base = ["pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT, "--journal",
            str(tmp_path / "j.sqlite"), "--receipts", str(tmp_path / "r"), "--execute"]
    for extra in (["--prices", "none", "--wallet-name", "w"], ["--assume-collector", PAID, "--wallet-name", "w"],
                  []):
        result = CliRunner().invoke(cli.ledger_app, base + extra)
        assert result.exit_code != 0, extra
    assert not (tmp_path / "j.sqlite").exists()


def test_events_are_normalized_to_ss58_accounts():
    from scalecodec.utils.ss58 import ss58_encode

    from reliquary.ledger.payout_chain import normalize_event, ss58

    key = "0x" + "11" * 32
    event = {"phase": {"ApplyExtrinsic": 1}, "extrinsic_idx": 1,
             "event": {"module_id": "SubtensorModule", "event_id": "StakeTransferred",
                       "attributes": (key, bytes.fromhex("22" * 32), "5Already", 81, 81, 10)}}
    normalized = normalize_event(event)
    assert normalized["module"] == "SubtensorModule" and normalized["event"] == "StakeTransferred"
    assert normalized["attributes"][:3] == [ss58_encode(key, ss58_format=42),
                                            ss58_encode(bytes.fromhex("22" * 32), ss58_format=42), "5Already"]
    assert ss58(81) == 81
