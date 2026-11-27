"""The J+2 payer: due obligations of tasks on the collector rail are paid from
the collector's stake by transfer_stake, each once, through crashes, dropped
sends, failures, replays, unreadable reads, and stale or second journals; a
dry run signs nothing and writes no journal."""

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
        self.owners = {COLLECTOR_HOT: COLLECTOR, **owners}
        self.subnet = ("5Owner", "5OwnerHot")
        self.bad_nonce_reads = 0      # how many nonce reads come back as an RPC error decodes them
        self.unreadable_blocks = False
        self.price_rao = 5_000_000    # 0.005 TAO per alpha
        self.max_weight = None        # the runtime's per-extrinsic limit; None: batches are not sized by weight
        self.blocks: dict[int, list[tuple[str, po.Inclusion]]] = {}
        self.submitted: list[po.SignedBatch] = []
        self.on_submit = on_submit
        self.refuse: set[str] = set()
        self.calls: list[str] = []

    async def finalized_block(self):
        return self.head

    async def collector_stake_rao(self, block):
        return self.stake

    async def collector_nonce(self, block):
        if self.bad_nonce_reads:
            self.bad_nonce_reads -= 1
            return 0
        return self.nonce

    async def subnet_owner(self, block):
        return self.subnet

    async def alpha_price_rao(self, block):
        return self.price_rao

    async def max_extrinsic_weight(self):
        return self.max_weight

    async def hotkey_owner(self, hotkey, block):
        owner = self.owners.get(hotkey)
        return owner(block) if callable(owner) else owner

    async def block_at(self, at):
        return int((at - T0) // 12)

    async def find_extrinsic(self, extrinsic_hash, first_block, last_block):
        self.calls.append("find_extrinsic")
        if self.unreadable_blocks:
            raise po.ChainReadError(f"block {first_block} could not be fetched")
        for number in range(first_block, last_block + 1):
            for digest, inclusion in self.blocks.get(number, []):
                if digest == extrinsic_hash:
                    return inclusion
        return None

    async def submit(self, signed):
        self.submitted.append(signed)
        if self.on_submit == "reject":
            raise po.PoolRejected("Invalid Transaction")    # the node's pool refused it: never imported
        if self.refuse and isinstance(signed.payload, list) and any(t.coldkey in self.refuse for t in signed.payload):
            self.land(signed, success=False)    # the chain refuses one destination: the whole batch_all reverts
            return
        if self.on_submit == "raise_before":
            raise ConnectionError("dropped before the node had it")
        if self.on_submit in ("include", "fail", "raise_after_include"):
            self.land(signed, success=self.on_submit != "fail")
        if self.on_submit == "raise_after_include":
            raise ConnectionError("dropped after the node had it")

    def land(self, signed, *, success=True, events=None, outcome="from success"):
        self.head += 3
        self.nonce = max(self.nonce, signed.nonce + 1)
        if isinstance(signed.payload, dict):        # a burn_alpha of the collector's stake
            burned = min(signed.payload["burn_rao"], self.stake) if success else 0
            events = events if events is not None else (
                ([{"module": "SubtensorModule", "event": "AlphaBurned",
                   "attributes": [COLLECTOR, COLLECTOR_HOT, burned, 81]}] if success else [])
                + [{"module": "System", "event": "ExtrinsicSuccess" if success else "ExtrinsicFailed",
                    "attributes": {}}])
            self.stake -= burned
            self.blocks.setdefault(self.head, []).append(
                (signed.extrinsic_hash, po.Inclusion(block_number=self.head, block_hash=f"0x{self.head:064x}",
                                                     success=success, events=tuple(events), fee_rao=90_000)))
            return
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
                                                 success=success if outcome == "from success" else outcome,
                                                 events=tuple(events), fee_rao=125_000,
                                                 error=None if success else "{'name': 'InsufficientStake'}")))

    def foreign(self):
        """Another extrinsic signed with the collector's key takes the next nonce."""
        self.head += 1
        self.nonce += 1


class FakeSigner:
    coldkey = COLLECTOR
    weight_per_transfer = 100

    def __init__(self):
        self.signed = []
        self.weighed = []

    async def declared_weight(self, transfers, *, nonce, era_block, era_period):
        self.weighed.append(len(transfers))
        return self.weight_per_transfer * len(transfers), None

    async def sign(self, transfers, *, nonce, era_block, era_period):
        body = json.dumps([[t.coldkey, t.amount_rao, list(t.obligation_ids)] for t in transfers] + [nonce, era_block])
        signed = po.SignedBatch(extrinsic_hash="0x" + hashlib.sha256(body.encode()).hexdigest(), nonce=nonce,
                                era_block=era_block, era_period=era_period, payload=list(transfers))
        self.signed.append(signed)
        return signed

    async def sign_burn(self, amount_rao, *, nonce, era_block, era_period):
        body = json.dumps(["burn", amount_rao, nonce, era_block])
        signed = po.SignedBatch(extrinsic_hash="0x" + hashlib.sha256(body.encode()).hexdigest(), nonce=nonce,
                                era_block=era_block, era_period=era_period, payload={"burn_rao": amount_rao})
        self.signed.append(signed)
        return signed


def _config(**overrides):
    values = {"netuid": 81, "collector_coldkey": COLLECTOR, "collector_hotkey": COLLECTOR_HOT,
              "min_transfer_rao": 0, "batch_size": 16, "reserve_rao": 0, "era_period": 64}
    values.update(overrides)
    return po.PayoutConfig(**values)


def _pass(obligations, journal, chain, *, execute=True, signer=None, now=NOW, day=PASS_DAY, config=None,
          collector=(PAID,), wait_seconds=600.0, confirm="dry run", plan="dry run", receipted=True):
    """One pass. Executing confirms, by default, the total and the plan a dry
    run just printed, and marks what its receipt reports once written: what
    an operator (the CLI) does."""
    clock = SimpleNamespace(t=0.0)

    async def sleep(seconds):
        clock.t += seconds
        chain.head += 5     # time passes on the chain too

    def run(execute, confirm_rao, confirm_plan):
        return asyncio.run(po.run_pass(
            obligations, journal=journal, chain=chain, config=config or _config(), collector_tasks=collector,
            pass_day=day, terms=TERMS, now=now, execute=execute,
            signer=signer or (FakeSigner() if execute else None), confirm_rao=confirm_rao,
            confirm_plan=confirm_plan, wait_seconds=wait_seconds, poll_seconds=12.0, clock=lambda: clock.t,
            sleep=sleep))

    if execute and "dry run" in (confirm, plan):
        preview = run(False, None, None)
        confirm = preview["planned"]["rao"] if confirm == "dry run" else confirm
        plan = preview["plan_digest"] if plan == "dry run" else plan
    receipt = run(execute, confirm if execute else None, plan if execute else None)
    if execute and receipted and not journal.read_only:
        journal.mark_receipted(receipt["reported_batch_ids"])
    return receipt


OWNERS = {"5A": "5ColdA", "5B": "5ColdB", "5C": "5ColdC"}


@pytest.fixture()
def journal(tmp_path):
    handle = po.Journal(tmp_path / "payout.sqlite")
    handle.initialise(collector_coldkey=COLLECTOR, netuid=81, nonce=0, reason="a first journal")
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
    assert {t["coldkey"] for t in receipt["planned_transfers"]} == {"5ColdA", "5ColdB"}
    # With no journal it knows nothing of what was paid: an execution would stop, and the dry run says so.
    assert "not initialised" in receipt["stopped"] and "no such file" in receipt["stopped"]
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
    # One failure on chain, one attempt counted.
    assert journal.rows([obligations[0].obligation_id])[obligations[0].obligation_id]["attempts"] == 1


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
    (batch,) = journal.needs_person()
    with pytest.raises(po.PayoutError):
        journal.settle_by_hand(batch["batch_id"], landed=False, evidence=" ")
    journal.settle_by_hand(batch["batch_id"], landed=False, evidence="read the chain: the transfer was never made")
    chain.on_submit = "include"
    resumed = _pass(obligations, journal, chain)
    assert resumed["stopped"] is None and resumed["paid"]["transfers"] == 1 and len(chain.submitted) == 2
    # The other extrinsic spent nonce 0; the payer signed the next one.
    assert [s.nonce for s in chain.submitted] == [0, 1]
    with pytest.raises(po.PayoutError):
        journal.settle_by_hand(batch["batch_id"], landed=True, evidence="twice")
    assert [d["kind"] for d in journal.decisions()] == ["initialise", "settle"]


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


def test_a_destination_that_does_not_fit_whole_is_paid_its_oldest_obligations_that_fit():
    old = _ledger({PAID: {"5A": 0.01}}, work=WORK - 20)
    new = _ledger({PAID: {"5A": 0.02, "5B": 0.001}})
    pairs = [(o, OWNERS[o.hotkey]) for o in old + new]
    transfers, deferred, unfunded = po.plan_transfers(pairs, available_rao=old[0].amount_rao + 5,
                                                      min_transfer_rao=0)
    # 5A's oldest obligation fits, its younger one does not; nothing younger (5B) overtakes it.
    assert [(t.coldkey, t.obligation_ids) for t in transfers] == [("5ColdA", (old[0].obligation_id,))]
    assert {o.obligation_id for o in unfunded} == {o.obligation_id for o in new}
    # The part that fits must reach the floor.
    transfers, deferred, unfunded = po.plan_transfers(pairs, available_rao=old[0].amount_rao + 5,
                                                      min_transfer_rao=old[0].amount_rao + 1)
    assert transfers == [] and len(unfunded) == 2 and [o.hotkey for o in deferred] == ["5B"]


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


def test_the_destination_is_the_owner_at_the_end_of_the_work_and_is_kept_for_retries(journal):
    obligations = _ledger({PAID: {"5A": 0.01}}, settled_after=6 * 3600)
    worked_block = int((obligations[0].earn_at - T0) // 12)
    # Sold between the end of the work and the settlement: the seller's coldkey did the work.
    owners = {"5A": lambda block: "5ColdThen" if block <= worked_block else "5ColdNow"}
    chain = FakeChain(stake_rao=10**12, owners=owners, on_submit="fail")
    failed = _pass(obligations, journal, chain)
    assert failed["batches"][0]["transfers"][0]["coldkey"] == "5ColdThen"
    chain.owners = {**chain.owners, "5A": "5ColdLater"}
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


def test_events_that_do_not_match_mark_a_paid_batch_and_stop_the_payer_for_a_person(journal):
    obligations = _ledger({PAID: {"5A": 0.01}, PAID + "-b": {"5B": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0, collector=(PAID,))
    (signed,) = chain.submitted
    chain.land(signed, events=[{"module": "System", "event": "ExtrinsicSuccess", "attributes": {}}])
    receipt = _pass(obligations, journal, chain, collector=(PAID, PAID + "-b"))
    assert receipt["recovered"][0]["state"] == po.BATCH_MISMATCH and "differ" in receipt["recovered"][0]["detail"]
    assert receipt["stopped"] and receipt["paid"]["transfers"] == 1 and len(chain.submitted) == 1
    paid_ids = [o.obligation_id for o in obligations if o.task_id == PAID]
    assert {journal.rows(paid_ids)[i]["state"] for i in paid_ids} == {ob.FINALIZED}
    # Paid, so it can only be settled as landed, once a person checked where the stake went.
    (batch,) = journal.needs_person()
    with pytest.raises(po.PayoutError):
        journal.settle_by_hand(batch["batch_id"], landed=False, evidence="looked")
    journal.settle_by_hand(batch["batch_id"], landed=True, evidence="block read: the stake went to 5ColdA")
    chain.on_submit = "include"
    resumed = _pass(obligations, journal, chain, collector=(PAID, PAID + "-b"))
    assert resumed["stopped"] is None and resumed["paid"]["transfers"] == 1


def test_reconciling_reads_mapped_and_listed_event_fields_and_their_subnets():
    def inclusion(netuid=81):
        return po.Inclusion(block_number=1, block_hash="0x1", success=True, events=(
            {"module": "SubtensorModule", "event": "StakeTransferred",
             "attributes": {"origin_coldkey": COLLECTOR, "destination_coldkey": "5ColdA", "hotkey": COLLECTOR_HOT,
                            "origin_netuid": 81, "destination_netuid": netuid}},
            {"module": "SubtensorModule", "event": "StakeTransferred",
             "attributes": [COLLECTOR, "5ColdB", COLLECTOR_HOT, 81, 81, 5]}))

    planned = [{"coldkey": "5ColdA"}, {"coldkey": "5ColdB"}]
    ok, _ = po.reconcile_events(planned, inclusion(), collector_coldkey=COLLECTOR, collector_hotkey=COLLECTOR_HOT,
                                netuid=81)
    assert ok
    wrong, why = po.reconcile_events(planned, inclusion(), collector_coldkey="5Other",
                                     collector_hotkey=COLLECTOR_HOT, netuid=81)
    assert not wrong and "not the collector's" in why
    elsewhere, why = po.reconcile_events(planned, inclusion(netuid=3), collector_coldkey=COLLECTOR,
                                         collector_hotkey=COLLECTOR_HOT, netuid=81)
    assert not elsewhere and "subnet 81" in why


def test_a_receipt_is_written_beside_earlier_ones_never_over_them(tmp_path, journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    receipt = _pass(obligations, journal, FakeChain(stake_rao=10**12, owners=OWNERS))
    first = po.write_receipt(receipt, tmp_path / "receipts")
    second = po.write_receipt(receipt, tmp_path / "receipts")
    assert first != second and json.loads(first.read_text())["schema"] == po.PAYOUT_RECEIPT_SCHEMA


@pytest.mark.parametrize("bad", [{"era_period": 63}, {"batch_size": 0}, {"batch_size": 29}, {"min_transfer_rao": -1},
                                 {"price_margin": -0.1}, {"price_margin": float("nan")}, {"min_transfer_rao": True},
                                 {"collector_hotkey": ""}])
def test_a_payout_config_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        _config(**bad)


# -- the CLI ----------------------------------------------------------------------------------------------------------


def _registry_view(monkeypatch, rails=None):
    from reliquary.ledger import sources

    async def read(**_):
        return sources.RegistryView(ceilings=_ceilings(PAID), pay_ceilings={}, floored=[],
                                    collector_tasks=sorted(rails or {}), collector_from=dict(rails or {}))

    monkeypatch.setattr(sources, "read_registry_ceilings", read)


def test_ledger_pay_is_a_dry_run_by_default(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    _registry_view(monkeypatch)
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

    monkeypatch.setattr(cli, "_subtensor", fake_subtensor)
    monkeypatch.setattr(cli, "_payout_chain", lambda subtensor, **kwargs: chain)
    journal = tmp_path / "journal.sqlite"
    args = ["pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT, "--journal", str(journal),
            "--receipts", str(tmp_path / "receipts"), "--archives", str(tmp_path / "archives"),
            "--assume-collector", PAID, "--as-of", "2026-11-28T12:05:00"]
    result = CliRunner().invoke(cli.ledger_app, args)
    # No journal yet: the dry run plans, and says an execution would stop until one is initialised.
    assert result.exit_code == 3, result.output
    assert "DRY-RUN pass of 2026-11-28" in result.output and "planned 1 transfer(s)" in result.output
    assert "not initialised" in result.output and "--confirm-plan" not in result.output
    assert not journal.exists() and chain.submitted == []
    (receipt,) = (tmp_path / "receipts").glob("payout-2026-11-28-dry-run-*.json")
    sources_used = json.loads(receipt.read_text())["sources"]
    assert sources_used["assumed_collector_tasks"] == [PAID] and sources_used["as_of_supplied"] is True
    handle = po.Journal(journal)
    handle.initialise(collector_coldkey=COLLECTOR, netuid=81, nonce=0, reason="a first journal")
    handle.close()
    ready = CliRunner().invoke(cli.ledger_app, args)
    assert ready.exit_code == 0, ready.output
    assert "to pay this plan: --execute --confirm-rao" in ready.output and "--confirm-plan" in ready.output


def test_ledger_pay_executes_only_from_the_registry_with_a_wallet(tmp_path):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    base = ["pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT, "--journal",
            str(tmp_path / "j.sqlite"), "--receipts", str(tmp_path / "r"), "--execute"]
    wallet = ["--wallet-name", "w"]
    # What an execution pays is pinned: no other source, no other time, no other terms.
    for extra in (["--prices", "none"] + wallet, ["--assume-collector", PAID] + wallet, [],
                  ["--archives", str(tmp_path)] + wallet, ["--as-of", "2026-11-28T12:05:00"] + wallet,
                  ["--pay-day", "2026-11-28"] + wallet, ["--lag-days", "1"] + wallet, ["--pay-at", "06:00"] + wallet,
                  ["--hold-hours", "1"] + wallet):
        result = CliRunner().invoke(cli.ledger_app, base + extra)
        assert result.exit_code != 0, extra
    assert not (tmp_path / "j.sqlite").exists()
    for gone in (["--rate", "chain"], ["--order-budget", f"{PAID}=1"]):
        assert CliRunner().invoke(cli.ledger_app, base[:-1] + gone).exit_code == 2


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


# -- the owner's go, and transfers the chain refuses -------------------------------------------------------------------


def test_executing_pays_only_the_total_and_plan_a_confirmation_names(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    total = sum(o.amount_rao for o in obligations)
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    digest = _pass(obligations, journal, chain, execute=False)["plan_digest"]
    for wrong_total, wrong_plan in ((None, digest), (total - 1, digest), (total + 1, digest), (total, None),
                                    (total, "0" * 16)):
        stopped = _pass(obligations, journal, chain, confirm=wrong_total, plan=wrong_plan)
        assert stopped["stopped"] and f"--confirm-rao {total} --confirm-plan {digest}" in stopped["stopped"]
        assert chain.submitted == [] and journal.batches() == []
    paid = _pass(obligations, journal, chain, confirm=total, plan=digest)
    assert paid["paid"]["rao"] == total and paid["confirmed_rao"] == total and paid["confirmed_plan"] == digest


def test_the_plan_digest_binds_destinations_amounts_and_obligations():
    one = [po.Transfer(coldkey="a", amount_rao=5, obligation_ids=("x", "y"))]
    assert po.plan_digest(one) == po.plan_digest([po.Transfer(coldkey="a", amount_rao=5, obligation_ids=("y", "x"))])
    for other in ([po.Transfer(coldkey="b", amount_rao=5, obligation_ids=("x", "y"))],
                  [po.Transfer(coldkey="a", amount_rao=6, obligation_ids=("x", "y"))],
                  [po.Transfer(coldkey="a", amount_rao=5, obligation_ids=("x",))]):
        assert po.plan_digest(other) != po.plan_digest(one)


def test_a_dry_run_shows_the_total_an_execution_will_ask_to_confirm(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    chain.head += 200          # the dropped batch's window is over: it will be released
    preview = _pass(obligations, journal, chain, execute=False)
    assert preview["recovered"][0]["state"] == po.BATCH_EXPIRED
    assert preview["planned"]["rao"] == obligations[0].amount_rao and preview["in_flight_rao"] == 0
    chain.on_submit = "include"
    paid = _pass(obligations, journal, chain, confirm=preview["planned"]["rao"])
    assert paid["paid"]["rao"] == obligations[0].amount_rao


def test_a_refused_destination_is_retried_alone_then_held_for_a_person(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    chain.refuse = {"5ColdB"}
    first = _pass(obligations, journal, chain)
    assert first["batches"][0]["state"] == po.BATCH_FAILED and len(first["batches"][0]["transfers"]) == 2
    second = _pass(obligations, journal, chain)
    # Each transfer that failed goes alone: 5A is paid, 5B fails again by itself.
    assert sorted((len(b["transfers"]), b["state"]) for b in second["batches"]) == [
        (1, po.BATCH_FAILED), (1, po.BATCH_FINALIZED)]
    third = _pass(obligations, journal, chain)
    assert third["batches"][0]["state"] == po.BATCH_FAILED
    held = _pass(obligations, journal, chain)
    b_ids = [o.obligation_id for o in obligations if o.hotkey == "5B"]
    assert held["held"]["coldkey_refused"]["obligation_ids"] == b_ids and held["planned"]["transfers"] == 0
    # The coldkey is held, not only its old obligations: its new work is held too, and batches no one else.
    later = obligations + _ledger({PAID: {"5B": 0.01, "5C": 0.01}}, work=WORK + 20)
    next_day = _pass(later, journal, chain, now=NOW + 86_400, day=date(2026, 11, 29))
    assert next_day["held"]["coldkey_refused"]["obligations"] == 2
    assert [b["state"] for b in next_day["batches"]] == [po.BATCH_FINALIZED]
    assert [t["coldkey"] for t in next_day["batches"][0]["transfers"]] == ["5ColdC"]
    # A person fixes it (the swap announcement is gone) and clears it: paid once.
    chain.refuse = set()
    with pytest.raises(po.PayoutError):
        journal.clear_held(evidence="", coldkey="5ColdB")
    assert journal.clear_held(evidence="swap announcement withdrawn", coldkey="5ColdB") == 1
    paid = _pass(later, journal, chain, now=NOW + 86_400, day=date(2026, 11, 29))
    assert paid["paid"]["transfers"] == 1 and paid["paid"]["rao"] == sum(o.amount_rao for o in later
                                                                         if o.hotkey == "5B")
    assert set(_states(journal, later).values()) == {ob.RECONCILED}


def test_batches_put_retried_transfers_alone():
    transfers = [po.Transfer(coldkey=c, amount_rao=1, obligation_ids=(c,)) for c in ("a", "b", "c", "d")]
    assert [[t.coldkey for t in b] for b in po.batches_of(transfers, 2, alone={"b"})] == [["a", "c"], ["d"], ["b"]]


# -- receipts, published -------------------------------------------------------------------------------------------------


class _Bucket:
    """An S3 client over a dict, honouring IfNoneMatch and IfMatch; nothing else."""

    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.puts = 0

    def __getattr__(self, name):
        raise AssertionError(f"the publisher called {name}")

    async def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, etag = self.objects[Key]

        class _Body:
            async def read(self, *_):
                return body

        return {"Body": _Body(), "ETag": etag}

    async def put_object(self, Bucket, Key, Body, ContentType=None, IfNoneMatch=None, IfMatch=None):
        from botocore.exceptions import ClientError

        held = self.objects.get(Key)
        if (IfNoneMatch == "*" and held is not None) or (IfMatch is not None and (held is None or held[1] != IfMatch)):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts += 1
        self.objects[Key] = (bytes(Body), f'"{self.puts}"')


def _published(monkeypatch):
    from contextlib import asynccontextmanager

    from reliquary.infrastructure import storage

    bucket = _Bucket()

    @asynccontextmanager
    async def client(**_):
        yield bucket

    monkeypatch.setattr(storage, "get_s3_client", client)
    return bucket


def test_executed_receipts_are_published_once_with_an_index_and_dry_runs_never(tmp_path, monkeypatch, journal):
    from reliquary.ledger import receipts as rc

    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    folder = tmp_path / "receipts"
    dry = po.write_receipt(_pass(obligations, journal, chain, execute=False), folder)
    paid = po.write_receipt(_pass(obligations, journal, chain), folder)
    bucket = _published(monkeypatch)
    plan = asyncio.run(rc.publish(folder.glob("*.json")))
    assert {r["path"]: r["action"] for r in plan} == {str(dry): "not_executed", str(paid): "would_write"}
    assert bucket.objects == {}
    done = asyncio.run(rc.publish(folder.glob("*.json"), execute=True))
    key = f"reliquary/public/payouts/transfers/2026-11-28/{paid.name}"
    assert [r["action"] for r in done if r["path"] == str(paid)] == ["written"]
    assert bucket.objects[key][0] == paid.read_bytes()
    index = json.loads(bucket.objects[rc.INDEX_KEY][0])
    assert [e["key"] for e in index["receipts"]] == [key] and index["receipts"][0]["rao"] == obligations[0].amount_rao
    again = asyncio.run(rc.publish(folder.glob("*.json"), execute=True))
    assert [r["action"] for r in again if r["path"] == str(paid)] == ["already_published"]
    assert len(json.loads(bucket.objects[rc.INDEX_KEY][0])["receipts"]) == 1
    paid.write_text(paid.read_text().replace('"pass_day"', '"pass_day" ', 1))
    with pytest.raises(rc.ReceiptError):
        asyncio.run(rc.publish([paid], execute=True))


def test_the_index_keeps_every_receipt_newest_first():
    from reliquary.ledger.receipts import merge_index

    old = {"key": "k1", "pass_day": "2026-11-28", "as_of": 1.0}
    new = {"key": "k2", "pass_day": "2026-11-29", "as_of": 2.0}
    merged = merge_index(merge_index(None, [old]), [new, old])
    assert [e["key"] for e in merged["receipts"]] == ["k2", "k1"]


def test_ledger_pay_keeps_its_journal_and_receipts_in_one_documented_place(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _registry_view(monkeypatch)
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

    monkeypatch.setattr(cli, "_subtensor", fake_subtensor)
    monkeypatch.setattr(cli, "_payout_chain", lambda subtensor, **kwargs: chain)
    result = CliRunner().invoke(cli.ledger_app, [
        "pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey", COLLECTOR_HOT,
        "--archives", str(tmp_path / "archives"), "--assume-collector", PAID, "--as-of", "2026-11-28T12:05:00"])
    assert result.exit_code == 3, result.output       # no journal yet
    assert str(tmp_path / "home/.reliquary/payout/journal.sqlite") in result.output
    assert list((tmp_path / "home/.reliquary/payout/receipts").glob("payout-2026-11-28-dry-run-*.json"))
    refused = CliRunner().invoke(cli.ledger_app, ["pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey",
                                                  COLLECTOR_HOT, "--confirm-rao", "5"])
    assert refused.exit_code != 0


# -- one journal: a fresh, restored or second journal never pays again --------------------------------------------------


def test_an_execution_on_a_journal_never_initialised_sends_nothing(tmp_path):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    handle = po.Journal(tmp_path / "fresh.sqlite")
    receipt = _pass(obligations, handle, chain)
    handle.close()
    assert "not initialised" in receipt["stopped"] and chain.submitted == []
    assert receipt["planned"]["transfers"] == 1     # what it would have paid is shown, never sent


def test_a_restored_journal_stops_on_the_collectors_nonce_and_pays_nothing_twice(tmp_path, journal):
    import shutil

    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    journal.db.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(journal.path, tmp_path / "backup.sqlite")          # yesterday's backup
    paid = _pass(obligations, journal, chain)
    assert paid["paid"]["transfers"] == 2
    journal.close()
    restored = po.Journal(tmp_path / "backup.sqlite")
    again = _pass(obligations, restored, chain)
    # The backup knows nothing of the batch sent since: it would plan the same obligations again. The
    # collector's nonce is past what it expects, so nothing is sent.
    assert again["planned"]["transfers"] == 2 and again["nonce"] == {"chain": 1, "journal": 0, "at_block": chain.head}
    assert "expects 0" in again["stopped"] and len(chain.submitted) == 1
    restored.close()


def test_an_extrinsic_signed_outside_the_payer_stops_it_until_a_person_resyncs(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    chain.foreign()                                   # the owner used the collector's key for something else
    stopped = _pass(obligations, journal, chain)
    assert stopped["stopped"] and "find-nonce" in stopped["stopped"] and chain.submitted == []
    with pytest.raises(po.PayoutError):
        journal.resync(nonce=1, reason="")
    assert journal.resync(nonce=1, reason="checked: one add_stake by the owner at nonce 0") == 0
    resumed = _pass(obligations, journal, chain)
    assert resumed["stopped"] is None and [s.nonce for s in chain.submitted] == [1]
    with pytest.raises(po.PayoutError):
        journal.resync(nonce=0, reason="going back")
    assert [d["kind"] for d in journal.decisions()] == ["initialise", "resync"]


def test_a_journal_kept_for_another_collector_or_subnet_stops_the_pass(tmp_path):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    handle = po.Journal(tmp_path / "other.sqlite")
    handle.initialise(collector_coldkey="5SomeoneElse", netuid=81, nonce=0, reason="test")
    receipt = _pass(obligations, handle, chain)
    handle.close()
    assert "belongs to collector 5SomeoneElse" in receipt["stopped"] and chain.submitted == []


def test_initialising_twice_is_refused_and_a_dry_run_never_writes_the_file(tmp_path, journal):
    with pytest.raises(po.PayoutError):
        journal.initialise(collector_coldkey=COLLECTOR, netuid=81, nonce=0, reason="again")
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    chain.head += 200
    # A dry run handed the writable journal settles the expired batch in a snapshot, not in the file.
    preview = _pass(obligations, journal, chain, execute=False)
    assert preview["recovered"][0]["state"] == po.BATCH_EXPIRED
    assert [b["state"] for b in journal.batches()] == [po.BATCH_SUBMITTED]


def test_journal_init_refuses_a_journal_that_does_not_hold_every_published_paid_batch(tmp_path, monkeypatch, journal):
    from reliquary.ledger import cli

    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(obligations, journal, chain)
    po.write_receipt(receipt, tmp_path / "receipts")
    published = {b["extrinsic_hash"] for b in receipt["batches"]}

    class _Subtensor:
        async def close(self):
            return None

    async def fake_subtensor(network):
        return _Subtensor()

    monkeypatch.setattr(cli, "_subtensor", fake_subtensor)
    monkeypatch.setattr(cli, "_payout_chain", lambda subtensor, **kwargs: chain)

    def init(path, *, resync=False, published=published, receipts=tmp_path / "none"):
        return asyncio.run(cli.collect_journal_init(
            collector_coldkey=COLLECTOR, collector_hotkey=COLLECTOR_HOT, reason="disk lost", resync=resync,
            journal_path=path, receipts=receipts, network="test", netuid=81, published=published))

    # A new journal after paying: the published index (or a local receipt) lists a batch it does not hold.
    for kwargs in ({}, {"published": set(), "receipts": tmp_path / "receipts"}):
        with pytest.raises(typer_bad_parameter()):
            init(tmp_path / "new.sqlite", **kwargs)
    # The journal that paid holds them all: it can be re-aligned, and a first journal starts where nothing was paid.
    journal.close()
    assert "re-aligned" in init(journal.path, resync=True)
    assert "initialised at the collector's nonce 1" in init(tmp_path / "first.sqlite", published=set())


def typer_bad_parameter():
    import typer

    return typer.BadParameter


# -- an outcome the chain does not tell, or a read that cannot be right, is never settled as unpaid ----------------------


def test_an_included_batch_whose_outcome_is_unreadable_is_held_for_a_person_never_repaid(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    (signed,) = chain.submitted
    # Included, but its events came back empty (a node error decodes as no events): it may have paid.
    chain.land(signed, events=[], outcome=None)
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_UNKNOWN and receipt["stopped"]
    assert set(_states(journal, obligations).values()) == {ob.SUBMITTED} and len(chain.submitted) == 1
    assert receipt["planned"]["transfers"] == 0
    (batch,) = journal.needs_person()
    journal.settle_by_hand(batch["batch_id"], landed=True, evidence="block read: StakeTransferred to 5ColdA")
    assert set(_states(journal, obligations).values()) == {ob.FINALIZED}
    assert _pass(obligations, journal, chain)["planned"]["transfers"] == 0


def test_a_failed_outcome_with_stake_moved_is_unknown_not_failed(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    (signed,) = chain.submitted
    chain.land(signed, success=False, events=[
        {"module": "SubtensorModule", "event": "StakeTransferred",
         "attributes": [COLLECTOR, "5ColdA", COLLECTOR_HOT, 81, 81, 1]},
        {"module": "System", "event": "ExtrinsicFailed", "attributes": {}}])
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_UNKNOWN and len(chain.submitted) == 1


def test_a_nonce_read_below_the_batch_settles_nothing(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    _pass(obligations[:1], journal, chain, collector=(PAID,))
    chain.on_submit = "drop"
    second = _pass(obligations, journal, chain, wait_seconds=1.0)
    assert second["batches"][0]["state"] == po.BATCH_PENDING and second["batches"][0]["nonce"] == 1
    chain.head += 200                       # past its window: a nonce read of 0 would call it expired
    chain.bad_nonce_reads = 1               # one read comes back as the empty default
    receipt = _pass(obligations, journal, chain, confirm=None, plan=None)
    assert receipt["recovered"][0]["state"] == po.BATCH_UNREADABLE and "could not be read" in receipt["stopped"]
    assert [b["state"] for b in journal.batches()][-1] == po.BATCH_SUBMITTED and len(chain.submitted) == 2


def test_an_unfetchable_block_settles_nothing(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    chain.foreign()
    chain.unreadable_blocks = True
    receipt = _pass(obligations, journal, chain)
    assert receipt["recovered"][0]["state"] == po.BATCH_UNREADABLE and receipt["stopped"]
    assert [b["state"] for b in journal.batches()] == [po.BATCH_SUBMITTED]


def test_attempts_count_only_failures_on_chain(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)

    class Aborts(FakeSigner):
        async def sign(self, transfers, **kwargs):
            raise KeyboardInterrupt("the owner cancelled the password prompt")

    for _ in range(po.MAX_ATTEMPTS + 1):
        with pytest.raises(KeyboardInterrupt):
            _pass(obligations, journal, chain, signer=Aborts())
    paid = _pass(obligations, journal, chain)
    assert paid["paid"]["transfers"] == 1 and paid["held"]["coldkey_refused"]["obligations"] == 0
    assert journal.rows([obligations[0].obligation_id])[obligations[0].obligation_id]["attempts"] == 0


# -- what is held, and what is not the collector's to pay -----------------------------------------------------------------


def test_only_frozen_rates_are_paid_and_the_amount_first_sent_is_kept(journal):
    frozen = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="fail")
    assert _pass(frozen, journal, chain)["batches"][0]["state"] == po.BATCH_FAILED
    # The next day the same obligation is valued otherwise (another rate): it is held, never sent recomputed.
    from dataclasses import replace

    revalued = [replace(o, amount_rao=o.amount_rao * 2) for o in frozen]
    chain.on_submit = "include"
    held = _pass(revalued, journal, chain)
    assert held["held"]["amount_changed"]["obligation_ids"] == [frozen[0].obligation_id]
    assert held["planned"]["transfers"] == 0 and len(chain.submitted) == 1
    chained = [replace(o, rate_source=ob.RATE_CHAIN) for o in _ledger({PAID: {"5B": 0.01}})]
    receipt = _pass(chained, journal, chain)
    assert receipt["held"]["rate_not_frozen"]["obligations"] == 1 and receipt["planned"]["transfers"] == 0


def test_work_before_the_rails_first_period_is_the_weights_to_pay(journal):
    before = _ledger({PAID: {"5A": 0.01}}, work=WORK - 1)
    after = _ledger({PAID: {"5B": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(before + after, journal, chain, collector={PAID: WORK})
    paid = {oid for b in receipt["batches"] for t in b["transfers"] for oid in t["obligation_ids"]}
    assert paid == {after[0].obligation_id} and receipt["native_rail"]["obligations"] == 1


def test_a_collector_that_is_the_owners_or_does_not_hold_its_hotkey_pays_nothing(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    for subnet, holder in (((COLLECTOR, "5OwnerHot"), COLLECTOR), (("5Owner", COLLECTOR_HOT), COLLECTOR),
                           (("5Owner", "5OwnerHot"), "5SomeoneElse")):
        chain = FakeChain(stake_rao=10**12, owners={**OWNERS, COLLECTOR_HOT: holder})
        chain.owners[COLLECTOR_HOT] = holder
        chain.subnet = subnet
        receipt = _pass(obligations, journal, chain)
        assert receipt["stopped"] and chain.submitted == [], subnet


def test_an_obligation_owed_to_the_subnet_owners_coldkey_is_held(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners={**OWNERS, "5B": "5Owner"})
    receipt = _pass(obligations, journal, chain)
    assert receipt["held"]["to_subnet_owner"]["obligations"] == 1
    assert [t["coldkey"] for b in receipt["batches"] for t in b["transfers"]] == ["5ColdA"]


def test_a_due_journal_row_whose_archive_is_gone_is_reported(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="fail")
    _pass(obligations, journal, chain)
    receipt = _pass([], journal, chain)
    assert receipt["journal_only"]["obligation_ids"] == [obligations[0].obligation_id]
    assert receipt["journal_only"]["rao"] == obligations[0].amount_rao


# -- receipts from the journal ------------------------------------------------------------------------------------------


def test_every_settled_batch_goes_in_a_receipt_once_even_after_a_crash(tmp_path, journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    first = _pass(obligations, journal, chain, receipted=False)
    # The pass "crashed" before its receipt was written: the next one lists the batch, until one is written.
    second = _pass(obligations, journal, chain, receipted=False)
    assert [b["batch_id"] for b in second["unreported"]] == [first["batches"][0]["batch_id"]]
    assert second["paid"]["transfers"] == 2 and second["reported_batch_ids"] == [first["batches"][0]["batch_id"]]
    journal.mark_receipted(second["reported_batch_ids"])
    third = _pass(obligations, journal, chain)
    assert third["unreported"] == [] and third["paid"]["transfers"] == 0
    # A pass that raises leaves a receipt marked incomplete with what the journal settled.
    partial = po.incomplete_receipt(journal, config=_config(), pass_day=PASS_DAY, now=NOW, execute=True,
                                    error=RuntimeError("read timed out"))
    assert partial["incomplete"] and partial["unreported"] == [] and "read timed out" in partial["stopped"]


def test_fees_are_reported_in_tao(journal):
    obligations = _ledger({PAID: {"5A": 0.01}})
    receipt = _pass(obligations, journal, FakeChain(stake_rao=10**12, owners=OWNERS))
    assert receipt["batches"][0]["fee_tao_rao"] == 125_000
    assert receipt["paid"]["fee_tao_rao"] == 125_000 and receipt["paid"]["fee_tao"] == 125_000 / 10**9


def test_an_execution_stopped_before_sending_is_not_published(tmp_path, monkeypatch, journal):
    from reliquary.ledger import receipts as rc

    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    stopped = po.write_receipt(_pass(obligations, journal, chain, confirm=1), tmp_path / "r")
    _published(monkeypatch)
    assert [r["action"] for r in asyncio.run(rc.publish([stopped]))] == ["no_batch"]


def test_the_index_lists_every_paid_batch_hash(tmp_path, monkeypatch, journal):
    from reliquary.ledger import receipts as rc

    obligations = _ledger({PAID: {"5A": 0.01}})
    receipt = _pass(obligations, journal, FakeChain(stake_rao=10**12, owners=OWNERS))
    path = po.write_receipt(receipt, tmp_path / "r")
    bucket = _published(monkeypatch)
    asyncio.run(rc.publish([path], execute=True))
    assert asyncio.run(rc.published_batches()) == {receipt["batches"][0]["extrinsic_hash"].lower()}
    assert rc.local_paid_hashes(tmp_path / "r") == {receipt["batches"][0]["extrinsic_hash"].lower()}
    assert json.loads(bucket.objects[rc.INDEX_KEY][0])["receipts"][0]["paid_batches"]


# -- the person's commands ------------------------------------------------------------------------------------------------


def test_settle_needs_an_explicit_landed_or_not_and_evidence(tmp_path, journal):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    obligations = _ledger({PAID: {"5A": 0.01}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="drop")
    _pass(obligations, journal, chain, wait_seconds=1.0)
    chain.foreign()
    _pass(obligations, journal, chain)
    (batch,) = journal.needs_person()
    journal.close()
    base = ["settle", batch["batch_id"], "--journal", str(journal.path)]
    assert CliRunner().invoke(cli.ledger_app, base + ["--evidence", "read"]).exit_code != 0
    assert CliRunner().invoke(cli.ledger_app, base + ["--not-landed"]).exit_code != 0
    done = CliRunner().invoke(cli.ledger_app, base + ["--not-landed", "--evidence", "block 21004: an add_stake"])
    assert done.exit_code == 0, done.output
    reopened = po.Journal(journal.path)
    assert reopened.batch(batch["batch_id"])["state"] == po.BATCH_VOID
    assert reopened.decisions()[-1]["evidence"] == "block 21004: an add_stake"
    reopened.close()
    cleared = CliRunner().invoke(cli.ledger_app, ["clear-held", "--coldkey", "5ColdA", "--evidence", "fixed",
                                                  "--journal", str(journal.path)])
    assert cleared.exit_code == 0 and "0 obligation(s) cleared" in cleared.output


# -- the floor: the chain's minimum stake at the subnet's price, never below its transfer minimum -------------------------


def _owed(alpha):
    """The share of a period that owes ``alpha`` at the frozen rate E."""
    return alpha / E


def test_the_default_floor_is_the_chain_minimum_stake_at_the_subnet_price(journal):
    obligations = _ledger({PAID: {"5A": _owed(0.4), "5B": _owed(0.6)}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(obligations, journal, chain, config=_config(min_transfer_rao=None))
    # 0.002 TAO at 0.005 TAO per alpha is 0.4 alpha; +25 % for the price to move: 0.5 alpha.
    assert receipt["min_transfer_rao"] == 500_000_000 and receipt["min_transfer_tao"] == pytest.approx(0.0025)
    assert receipt["alpha_price_rao"] == 5_000_000 and receipt["alpha_price_tao"] == 0.005
    assert "DefaultMinStake" in receipt["min_transfer_source"] and "current_alpha_price" in receipt["alpha_price_source"]
    assert receipt["chain_transfer_minimum_rao"] == 25_000_000          # 0.0001 TAO, +25 %
    assert [t["coldkey"] for b in receipt["batches"] for t in b["transfers"]] == ["5ColdB"]
    assert receipt["deferred_below_minimum"]["obligations"] == 1 and receipt["min_transfer_given_rao"] is None
    # The price doubles: the same 0.4 alpha is now worth enough.
    chain.price_rao = 10_000_000
    later = _pass(obligations, journal, chain, config=_config(min_transfer_rao=None))
    assert later["min_transfer_rao"] == 250_000_000 and later["paid"]["transfers"] == 1


def test_no_floor_given_goes_below_the_chains_transfer_minimum(journal):
    obligations = _ledger({PAID: {"5A": _owed(0.02), "5B": _owed(0.03)}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    receipt = _pass(obligations, journal, chain, config=_config(min_transfer_rao=0))
    # 0.02 alpha is worth exactly 0.0001 TAO: under the chain's minimum once the margin is counted, never sent.
    assert receipt["min_transfer_rao"] == 25_000_000 and "raised to the chain's transfer minimum" in \
        receipt["min_transfer_source"]
    assert [t["coldkey"] for b in receipt["batches"] for t in b["transfers"]] == ["5ColdB"]
    assert po.transfer_floor(10**9, 5_000_000, 0.25)["alpha_rao"] == 10**9          # a higher floor is kept


def test_the_floor_is_exact_and_a_price_that_cannot_be_read_stops_the_pass(journal):
    assert po.alpha_worth(2_000_000, 3, 0.25) == -(-2_000_000 * 5 * 10**9 // (4 * 3))       # rounded up, exact
    for bad in (0, -1, None, 1.5):
        with pytest.raises(po.ChainReadError):
            po.alpha_worth(2_000_000, bad, 0.25)
    chain = FakeChain(stake_rao=10**12, owners=OWNERS)
    chain.price_rao = 0
    with pytest.raises(po.ChainReadError):
        _pass(_ledger({PAID: {"5A": 0.01}}), journal, chain)
    assert chain.submitted == []


# -- batches: at most 16 by default, cut to the runtime's weight; a pool refusal is never sent -----------------------------


def test_batches_are_cut_to_the_runtimes_per_extrinsic_weight(journal):
    hotkeys = {f"5H{i:02d}": 0.001 * (i + 1) for i in range(20)}
    obligations = _ledger({PAID: hotkeys})
    chain = FakeChain(stake_rao=10**13, owners={hk: f"5Cold{hk}" for hk in hotkeys})
    chain.max_weight = (1_000, None)            # 100 per transfer: nine fit under 90 % of the limit
    signer = FakeSigner()
    receipt = _pass(obligations, journal, chain, signer=signer)
    assert [len(b["transfers"]) for b in receipt["batches"]] == [9, 9, 2]
    assert [b["declared_weight"] for b in receipt["batches"]] == [[900, None], [900, None], [200, None]]
    assert signer.weighed == [16, 9, 11, 9, 2] and receipt["paid"]["transfers"] == 20
    assert receipt["max_extrinsic_weight"] == [1_000, None] and [s.nonce for s in chain.submitted] == [0, 1, 2]


def test_a_batch_size_above_the_runtimes_cap_is_refused():
    assert po.DEFAULT_BATCH_SIZE == 16 and po.MAX_BATCH_SIZE == 28
    _config(batch_size=28)
    with pytest.raises(ValueError):
        _config(batch_size=29)


def test_a_batch_the_nodes_pool_refuses_was_never_sent(journal):
    obligations = _ledger({PAID: {"5A": 0.01, "5B": 0.02}})
    chain = FakeChain(stake_rao=10**12, owners=OWNERS, on_submit="reject")
    head = chain.head
    receipt = _pass(obligations, journal, chain, wait_seconds=600.0)
    (batch,) = receipt["batches"]
    # Released at once: no waiting out the window, no attempt counted, nothing more sent this pass.
    assert batch["state"] == po.BATCH_EXPIRED and "refused by the node's transaction pool" in batch["detail"]
    assert chain.head == head and "due again" in receipt["stopped"]
    rows = journal.rows(o.obligation_id for o in obligations)
    assert {r["state"] for r in rows.values()} == {ob.DUE} and {r["attempts"] for r in rows.values()} == {0}
    chain.on_submit = "include"
    paid = _pass(obligations, journal, chain)
    assert paid["paid"]["transfers"] == 2 and [s.nonce for s in chain.submitted] == [0, 0]


def test_ledger_pay_refuses_a_batch_size_above_the_cap(tmp_path):
    from typer.testing import CliRunner

    from reliquary.ledger import cli

    result = CliRunner().invoke(cli.ledger_app, ["pay", "--collector-coldkey", COLLECTOR, "--collector-hotkey",
                                                 COLLECTOR_HOT, "--batch-size", "29"])
    assert result.exit_code == 2 and "28" in result.output
