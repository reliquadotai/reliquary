"""The RL auction (docs/design/2026-11-16-rl-auction.md): one order auction per env, its budget from the trainer's
batch, each lease paid its own price, undelivered leases charged against the next pay."""
from __future__ import annotations

import random

import pytest

from reliquary.corpus.token_auction import AuctionParams
from reliquary.validator.rl_auction import RlAuctions, env_budget
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from reliquary.validator.rl_leases import (
    ASK_ABOVE_PRICE, FULL, NO_ASK, LeaseBook, LeaseBookConfig, LeaseRequest,
)
from tests.unit.test_rl_submit import ENV, NOW, RIGHT, group, service, small_cap  # noqa: F401

OTHER = "other"
CEILING = 1e-6                                  # pool share per token: 1 alpha per million at 1 alpha per period
GROUP_TOKENS = 16 * len(RIGHT.tokens)


class Clock:
    def __init__(self, t: float = NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def auctions_for(*, batch_groups=4, lag=2, targets=None, clock=None):
    import uuid

    params = AuctionParams(ceiling=CEILING, miner_alpha_per_period=1.0)
    # A task of its own: an env's market (its deliveries) lives for the process.
    return RlAuctions(params, targets or {ENV: 0.5, OTHER: 0.5}, batch_groups=batch_groups, lag=lag,
                      task_id=f"rl-test-{uuid.uuid4().hex[:8]}", clock=clock or (lambda: NOW),
                      state_dir_job=lambda job: None)


def auctioned_book(tmp_path, *, clock=None, **kw):
    clock = clock or Clock()
    auctions = auctions_for(clock=clock, **kw)
    book = LeaseBook(tmp_path / "leases.sqlite3",
                     LeaseBookConfig(expiry_s={ENV: 600.0, OTHER: 600.0}, cooldown_of=lambda env: 2),
                     quota=None, clock=clock, rng=random.Random(7), auctions=auctions)
    book.adopt_checkpoint(0)
    return book, auctions, clock


def ask(hotkey, prompts, env=ENV):
    return LeaseRequest(hotkey=hotkey, environment=env, prompt_ids=tuple(prompts), checkpoint=0)


def test_an_env_leases_what_its_next_steps_train_less_what_is_buffered():
    assert env_budget(0.5, 512, 2, 0) == 512
    assert env_budget(0.5, 512, 2, 500) == 12
    assert env_budget(0.5, 512, 2, 600) == 0
    assert env_budget(0.1, 512, 1, 0) == 52                       # rounded up: a small env is never starved


def test_the_leases_out_cover_the_groups_that_will_not_train():
    """Two checkpoints ahead with a third of the groups in zone: three leases per missing group."""
    assert env_budget(0.5, 512, 2, 0, trainable=1 / 3) == 1536
    assert env_budget(0.5, 512, 2, 500, trainable=0.25) == 48
    assert env_budget(0.5, 512, 2, 0, trainable=0.0) == 5120     # a dry env: bounded by the floor (10 %)


def test_a_book_takes_a_quota_or_an_auction_not_both(tmp_path):
    with pytest.raises(ValueError, match="quota or by an auction"):
        LeaseBook(tmp_path / "x.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}), quota=None)


def test_a_lease_needs_an_offer_on_its_env_and_carries_its_price(tmp_path):
    book, auctions, _ = auctioned_book(tmp_path)
    assert book.grant([ask("A", [1])])[0].refused == {1: NO_ASK}
    assert auctions.post_ask(ENV, "A", 0.5, 8) is None
    assert book.grant([ask("A", [1], env=OTHER)])[0].refused == {1: NO_ASK}      # an offer is per env
    lease = book.grant([ask("A", [1])])[0].leases[0]
    assert book.lease_price(lease.lease_id) == pytest.approx(0.5e-6)              # its own offer: nothing delivered
    assert auctions.post_ask("nope", "A", 0.5, 8) == "environment_not_served"


def test_an_env_hands_out_no_more_than_its_budget_cheapest_first(tmp_path):
    # ENV: 0.5 x 2 x 1 = 1 group ahead; nothing graded yet, so a prior yield of 1/2: 2 leases out.
    book, auctions, _ = auctioned_book(tmp_path, batch_groups=2, lag=1)
    auctions.post_ask(ENV, "dear", 0.9, 8)
    auctions.post_ask(ENV, "cheap", 0.2, 8)
    answers = book.grant([ask("cheap", [1, 2, 3]), ask("dear", [4])])
    # Within one interval the cheaper offer is served first, whatever the order the requests are resolved in.
    assert len(answers[0].leases) == 2 and answers[0].refused == {3: FULL}
    assert answers[1].refused == {4: ASK_ABOVE_PRICE}                          # the budget went to the cheaper
    view = auctions.view()["environments"][ENV]
    assert (view["target_groups"], view["trainable_share"], view["budget"]) == (1, 0.5, 2)


def test_an_offer_above_the_ceiling_gets_nothing(tmp_path):
    book, auctions, _ = auctioned_book(tmp_path)
    auctions.post_ask(ENV, "A", 2.0, 8)                                        # 2 alpha/M over a 1 alpha/M ceiling
    assert book.grant([ask("A", [1])])[0].refused == {1: ASK_ABOVE_PRICE}


def test_an_expired_lease_is_charged_and_halves_the_capacity(tmp_path):
    book, auctions, clock = auctioned_book(tmp_path)
    auctions.post_ask(ENV, "A", 0.5, 4)
    lease = book.grant([ask("A", [1])])[0].leases[0]
    clock.t += 601
    book.expire_due()
    auction = auctions.auction(ENV)
    assert auction.debt_alpha("A") == pytest.approx(0.5e-6 * 16384)           # price x the env's group tokens
    # In debt: one lease at most until paid.
    granted = book.grant([ask("A", [2, 3])])[0]
    assert len(granted.leases) == 1 and lease.lease_id not in {x.lease_id for x in granted.leases}


def test_a_lease_our_side_failed_charges_nothing(tmp_path):
    book, auctions, clock = auctioned_book(tmp_path)
    auctions.post_ask(ENV, "A", 0.5, 4)
    book.sandbox_failure = lambda lease: ("box_failed", "gateway lost the box")
    lease = book.grant([ask("A", [1])])[0].leases[0]
    assert book.forfeit(lease.lease_id, "A").state == "released"
    assert auctions.auction(ENV).debt_alpha("A") == 0.0


def test_a_delivered_lease_counts_at_its_holders_offer(tmp_path):
    from reliquary.corpus.token_auction import PRICE_MIN_WEIGHT

    book, auctions, clock = auctioned_book(tmp_path, batch_groups=256, lag=2)
    auctions.post_ask(ENV, "A", 0.5, 64)
    count = int(PRICE_MIN_WEIGHT) + 1
    leases = book.grant([ask("A", list(range(count)))])[0].leases
    assert len(leases) == count
    for lease in leases:
        book.submit(lease.lease_id, "A", in_zone=True)
    # Delivered at 0.5: while A's offer still gets work, the price of the moment is 0.5, even for a cheaper offer.
    auctions.post_ask(ENV, "B", 0.1, 1)
    later = book.grant([ask("B", [100]), ask("A", [101])])
    assert book.lease_price(later[0].leases[0].lease_id) == pytest.approx(0.5e-6)
    # Once B's offer covers the budget alone, the dearest offer given work caps it: B is paid its own 0.1.
    assert auctions.post_ask(ENV, "B", 0.1, 10_000) is None
    clock.t += 1
    again = book.grant([ask("B", [102])])[0].leases[0]
    assert book.lease_price(again.lease_id) == pytest.approx(0.1e-6)


# --- pay -------------------------------------------------------------------------------------------------------


@pytest.fixture
def paid_book(tmp_path):
    clock = Clock()
    auctions = auctions_for(clock=clock, targets={ENV: 1.0}, batch_groups=64)
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}, cooldown_of=lambda env: 2),
                     quota=None, clock=clock, auctions=auctions)
    book.adopt_checkpoint(0)
    yield book, auctions, clock
    book.close()


def test_a_trained_group_is_paid_its_leases_price_less_the_holders_charge(paid_book):
    book, auctions, clock = paid_book
    subs = service(book, probation_groups=0)
    ledger = RlPayLedger(book, subs, None, LedgerConfig(task_id="rl-test", cap=0.5), clock=clock)
    auctions.post_ask(ENV, "A", 0.5, 8)
    missed = book.grant([LeaseRequest("A", ENV, (1,), 0)])[0].leases[0]
    clock.t += 601
    book.expire_due()                                                          # charged 0.5e-6 x 16384
    owed = auctions.auction(ENV).debt_alpha("A")
    held = book.grant([LeaseRequest("A", ENV, (2,), 0)])[0].leases[0]
    assert held.lease_id != missed.lease_id
    subs.submit(held.lease_id, "A", group(8))
    subs.report_training(held.lease_id, rollout_match=[1.0] * 16, accepted=True)
    assert ledger.ingest() == 1
    tokens, price, amount = book.db.execute(
        "SELECT matched_tokens, token_price, amount FROM rl_accruals").fetchone()
    assert (tokens, price) == (GROUP_TOKENS, pytest.approx(0.5e-6))
    assert amount == pytest.approx(max(0.0, GROUP_TOKENS * 0.5e-6 - owed))
    assert auctions.auction(ENV).debt_alpha("A") == pytest.approx(max(0.0, owed - GROUP_TOKENS * 0.5e-6))
    assert auctions.paid_tokens() == GROUP_TOKENS          # debt does not buy extra customer work
    book.db.execute("UPDATE rl_accruals SET status='forfeited'")
    assert auctions.paid_tokens() == GROUP_TOKENS


# --- declaration ------------------------------------------------------------------------------------------------


def _rl_entry():
    from reliquary.cli.main import build_task_entry, with_rl_async_params
    from reliquary.constants import PROTOCOL_PROFILE_ID
    from reliquary.protocol.profiles import resolve_protocol_profile

    envs = sorted(resolve_protocol_profile(PROTOCOL_PROFILE_ID).environments)
    split = {env: (1.0 if i == 0 else 0.0) for i, env in enumerate(envs)}
    entry = build_task_entry(task_id="rl-auction", profile_id=PROTOCOL_PROFILE_ID, cap=0.4, overrides={},
                             env_split=split)
    return with_rl_async_params(entry, planned_tokens_per_period=2e8), envs[0]


def _auctioned(entry, **extra):
    from dataclasses import replace

    from reliquary.validator.corpus_periods import price_params

    return replace(entry, params={**entry.params, "pricing": "auction", **price_params(1.0, 1.0, NOW), **extra})


def test_an_async_rl_task_is_declared_priced_by_auction(tmp_path):
    from reliquary.shared.task_registry import RegistryError, validate_entry
    from reliquary.validator.rl_async_service import build_rl_async, rl_async_settings

    entry, env = _rl_entry()
    declared = _auctioned(entry, rl_auction_lag_target=3)
    validate_entry(declared)
    settings = rl_async_settings(declared, 0.4, max_tokens=lambda e: 4000)
    assert settings.auction["pricing"] == "auction" and settings.auction["rl_auction_lag_target"] == 3
    assert rl_async_settings(entry, 0.4, max_tokens=lambda e: 4000).auction is None
    rl = build_rl_async(settings, path=tmp_path / "rl.sqlite3", grader=None)
    try:
        assert rl.book.quota is None and rl.book.auctions.lag == 3
        assert rl.book.config.max_pending_groups is None                 # the batch bounds the leases, not a cap
        assert set(rl.book.auctions.auctions) == {env}
    finally:
        rl.close()
    with pytest.raises(RegistryError):
        validate_entry(_auctioned(entry, rl_auction_lag_target=0))
    with pytest.raises(ValueError, match="no use on a task priced by auction"):
        rl_async_settings(_auctioned(entry, rl_env_pay_k=2.0), 0.4, max_tokens=lambda e: 4000)


def test_a_lag_target_is_an_rl_param():
    from dataclasses import replace

    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, RegistryError, validate_entry

    entry, _ = _rl_entry()
    corpus = replace(_auctioned(entry, rl_auction_lag_target=2), mechanism=MECHANISM_CORPUS_GENERATION)
    with pytest.raises(RegistryError):
        validate_entry(corpus)


# --- the miner's offers ------------------------------------------------------------------------------------------


class AskingTransport:
    def __init__(self, refuse=None):
        from reliquary.miner.rl_async import DryRunTransport

        self.inner, self.asks, self.refuse = DryRunTransport(quota=4), [], refuse

    async def ask(self, env, price, capacity):
        from reliquary.miner.rl_async import RequestRefused

        if self.refuse:
            raise RequestRefused(self.refuse, retryable=False)
        self.asks.append((env, price, capacity))
        return {"accepted": True, "auction": {"allowance": capacity}}

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _miner(transport, clock):
    from reliquary.miner.rl_async import AsyncMiner, AsyncMinerConfig, FakeGroupGenerator, PromptPlanner

    config = AsyncMinerConfig(environments=(ENV,), asks={ENV: (500_000, 8)})
    return AsyncMiner(config, transport=transport, generator=FakeGroupGenerator(),
                      planners={ENV: PromptPlanner(ENV, (0, 50), rng=random.Random(0), clock=clock)},
                      loaded_version=0, rng=random.Random(0), clock=clock)


def test_the_miner_posts_its_offers_before_leasing_and_every_ten_minutes(tmp_path):
    import asyncio

    clock = Clock()
    transport = AskingTransport()
    miner = _miner(transport, clock)
    asyncio.run(miner.refresh_quota())
    asyncio.run(miner.refresh_quota())
    assert transport.asks == [(ENV, 500_000, 8)]                       # not again within ten minutes
    clock.t += 601
    asyncio.run(miner.refresh_quota())
    miner._note_refusal(miner.planners[ENV], 3, NO_ASK)                # the validator lost it: at once
    asyncio.run(miner.post_asks())
    assert len(transport.asks) == 3


def test_a_validator_without_an_auction_is_not_asked_again(tmp_path):
    import asyncio

    transport = AskingTransport(refuse="not_auctioned")
    miner = _miner(transport, Clock())
    assert asyncio.run(miner.post_asks()) == 0 and miner._asks_served is False


def test_the_cli_reads_one_price_or_one_per_env():
    from reliquary.cli.main import _rl_asks

    assert _rl_asks(["0.5"], ("a", "b"), 8) == {"a": (500_000, 8), "b": (500_000, 8)}
    assert _rl_asks(["0.5", "b=2"], ("a", "b"), 4) == {"a": (500_000, 4), "b": (2_000_000, 4)}
    assert _rl_asks(["b=2"], ("a", "b"), 4) == {"b": (2_000_000, 4)}
    with pytest.raises(ValueError):
        _rl_asks(["c=1"], ("a", "b"), 4)
    with pytest.raises(ValueError):
        _rl_asks(["0"], ("a",), 4)


def test_the_yield_is_measured_on_the_envs_graded_groups(paid_book):
    from reliquary.validator import rl_auction

    book, auctions, clock = paid_book
    subs = service(book, probation_groups=0)
    auctions.post_ask(ENV, "A", 0.5, 64)
    assert auctions.trainable(ENV) == rl_auction.PRIOR_YIELD
    for prompt in range(rl_auction.YIELD_MIN_GROUPS):
        held = book.grant([LeaseRequest("A", ENV, (prompt,), 0)])[0].leases[0]
        subs.submit(held.lease_id, "A", group(8 if prompt % 4 == 0 else 16))      # a quarter in zone
    assert auctions.trainable(ENV) == pytest.approx(0.25)
    buffered = auctions.buffered(ENV)
    assert buffered == rl_auction.YIELD_MIN_GROUPS // 4
    assert auctions.budget(ENV) == env_budget(1.0, 64, 2, buffered, 0.25)


# --- a training order's limits ------------------------------------------------------------------------------------


def _limited(tmp_path, **limits):
    clock = Clock()
    auctions = auctions_for(clock=clock, targets={ENV: 1.0}, batch_groups=64)
    auctions.limits = limits
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}, cooldown_of=lambda env: 2),
                     quota=None, clock=clock, auctions=auctions)
    book.adopt_checkpoint(0)
    return book, auctions, clock


def test_an_order_past_its_deadline_leases_nothing(tmp_path):
    book, auctions, clock = _limited(tmp_path, deadline_at=NOW + 10)
    assert auctions.budget(ENV) > 0 and auctions.stopped() is None
    clock.t = NOW + 10
    assert auctions.budget(ENV) == 0 and auctions.stopped() == "deadline"
    auctions.post_ask(ENV, "A", 0.5, 8)
    assert not book.grant([LeaseRequest("A", ENV, (1,), 0)])[0].leases
    assert auctions.view()["stopped"] == "deadline"
    book.close()


def test_an_order_at_its_last_step_leases_nothing(tmp_path):
    book, auctions, _ = _limited(tmp_path, max_steps=3)
    book.adopt_checkpoint(2)
    assert auctions.budget(ENV) > 0
    book.adopt_checkpoint(3)
    assert auctions.budget(ENV) == 0 and auctions.stopped() == "max_steps"
    book.close()


def test_an_order_that_paid_its_tokens_leases_nothing(tmp_path):
    book, auctions, clock = _limited(tmp_path, max_tokens=2 * GROUP_TOKENS)
    RlPayLedger(book, service(book, probation_groups=0), None, LedgerConfig(task_id="rl-test", cap=0.5), clock=clock)
    row = "INSERT INTO rl_accruals VALUES(?, 'l', 'A', ?, ?, ?, 0, ?, ?, ?, ?, ?)"
    book.db.execute(row, (1, ENV, "trained", NOW, GROUP_TOKENS, 1e-6, GROUP_TOKENS * 1e-6, "accrued", GROUP_TOKENS))
    book.db.execute(row, (2, ENV, "first_scan", NOW, 0, 1e-6, 0.25 * GROUP_TOKENS * 1e-6, "accrued", .25 * GROUP_TOKENS))
    assert auctions.paid_tokens() == pytest.approx(1.25 * GROUP_TOKENS) and auctions.budget(ENV) > 0
    book.db.execute(row, (3, ENV, "trained", NOW, GROUP_TOKENS, 1e-6, GROUP_TOKENS * 1e-6, "forfeited", GROUP_TOKENS))
    book.db.commit()
    assert auctions.budget(ENV) == 0 and auctions.stopped() == "max_tokens"
    book.close()


def test_a_first_scan_counts_gross_usage_even_when_its_entire_payment_repays_debt(paid_book):
    book, auctions, clock = paid_book
    subs = service(book, probation_groups=0)
    ledger = RlPayLedger(book, subs, None, LedgerConfig(task_id="rl-test", cap=0.5), clock=clock)
    auctions.post_ask(ENV, "A", .5, 8)
    missed = book.grant([ask("A", [1])])[0].leases[0]
    clock.t += 601
    book.expire_due()
    held = book.grant([ask("A", [2])])[0].leases[0]
    assert held.lease_id != missed.lease_id
    subs.submit(held.lease_id, "A", group(16))
    ledger.ingest()
    amount, tokens = book.db.execute("SELECT amount,budget_tokens FROM rl_accruals").fetchone()
    assert amount == 0 and tokens == .25 * ledger.config.default_group_tokens
    assert auctions.paid_tokens() == tokens


def test_an_old_first_scan_without_gross_usage_cannot_reopen_a_token_budget(tmp_path):
    book, auctions, clock = _limited(tmp_path, max_tokens=1_000_000)
    RlPayLedger(book, service(book, probation_groups=0), None, LedgerConfig(task_id="rl-test", cap=.5), clock=clock)
    book.db.execute("INSERT INTO rl_accruals(seq,lease_id,hotkey,environment,kind,at,period,matched_tokens,"
                    "token_price,amount,status) VALUES(1,'l','A',?,'first_scan',?,0,0,1e-6,0,'accrued')", (ENV, NOW))
    book.db.commit()
    assert auctions.stopped() == "max_tokens" and auctions.budget(ENV) == 0
    book.close()


def test_ledger_upgrade_preserves_old_trained_work_without_guessing_first_scan_usage(tmp_path):
    book, auctions, clock = _limited(tmp_path, max_tokens=1_000_000)
    subs = service(book, probation_groups=0)
    config = LedgerConfig(task_id="rl-test", cap=.5)
    RlPayLedger(book, subs, None, config, clock=clock)
    with book.transaction():
        book.db.execute("ALTER TABLE rl_accruals DROP COLUMN budget_tokens")
        book.db.execute("INSERT INTO rl_accruals VALUES(1,'l1','A',?,'trained',?,0,123,1e-6,0,'forfeited')", (ENV, NOW))
        book.db.execute("INSERT INTO rl_accruals VALUES(2,'l2','A',?,'first_scan',?,0,0,1e-6,0,'accrued')", (ENV, NOW))
    RlPayLedger(book, subs, None, config, clock=clock)
    assert book.db.execute("SELECT budget_tokens FROM rl_accruals ORDER BY seq").fetchall() == [(123.0,), (None,)]
    assert auctions.stopped() == "max_tokens"
    book.close()


def test_a_reached_deadline_remains_stopped_after_clock_rollback_and_restart(tmp_path):
    book, auctions, clock = _limited(tmp_path, deadline_at=NOW + 10)
    clock.t = NOW + 10
    assert auctions.stopped() == "deadline"
    clock.t = NOW
    assert auctions.stopped() == "deadline"
    book.close()
    book, auctions, _ = _limited(tmp_path, deadline_at=NOW + 10)
    assert auctions.stopped() == "deadline"
    book.close()


def test_a_training_order_task_carries_its_limits_to_the_auction():
    from reliquary.validator.corpus_periods import price_params

    params = {"pricing": "auction", **price_params(1.0, 1.0, NOW),
              "training_order": {"training_id": "t", "max_steps": 5, "deadline_at": NOW + 60, "max_tokens": 9}}
    auctions = RlAuctions.from_params(params, {ENV: 1.0}, batch_groups=4, task_id="rl-lim", clock=lambda: NOW)
    assert auctions.limits == {"max_steps": 5, "deadline_at": NOW + 60, "max_tokens": 9}


def test_real_service_settings_preserve_and_enforce_the_training_order_limits(tmp_path):
    from reliquary.validator.rl_async_service import build_rl_async, rl_async_settings

    entry, env = _rl_entry()
    limits = {"max_steps": 5, "deadline_at": NOW + 60, "max_tokens": 9}
    entry = _auctioned(entry, training_order=limits)
    settings = rl_async_settings(entry, .4, max_tokens=lambda env: 4000)
    rl = build_rl_async(settings, path=tmp_path / "rl.sqlite3", grader=None, clock=lambda: NOW + 60)
    try:
        assert rl.book.auctions.limits == limits
        assert rl.book.auctions.budget(env) == 0 and rl.book.auctions.stopped() == "deadline"
    finally:
        rl.close()


def test_a_token_budget_waits_for_the_ledger_to_consume_pending_work(tmp_path):
    book, auctions, clock = _limited(tmp_path, max_tokens=GROUP_TOKENS * 2)
    subs = service(book, probation_groups=0)
    ledger = RlPayLedger(book, subs, None, LedgerConfig(task_id="rl-test", cap=.5), clock=clock)
    auctions.post_ask(ENV, "A", .5, 8)
    held = book.grant([ask("A", [1])])[0].leases[0]
    subs.submit(held.lease_id, "A", group(8))
    subs.report_training(held.lease_id, rollout_match=[1.0] * 16, accepted=True)
    assert auctions.stopped() == "ledger_pending"
    assert not book.grant([ask("A", [2])])[0].leases
    ledger.ingest()
    assert auctions.stopped() is None
    assert book.grant([ask("A", [2])])[0].leases
    book.close()


def test_a_token_budget_without_a_ledger_is_fenced(tmp_path):
    book, auctions, _ = _limited(tmp_path, max_tokens=1_000_000)
    service(book, probation_groups=0)
    auctions.post_ask(ENV, "A", .5, 8)
    assert auctions.stopped() == "ledger_pending"
    assert not book.grant([ask("A", [1])])[0].leases
    book.close()
