"""The async RL reference miner (``miner/rl_async.py``): the prompt planner and its cursor, the quota-bound batch
leasing, refusals, the submit-on-every-lease invariant, and checkpoint swaps between generation batches only. CPU,
no network: ``DryRunTransport`` stands for the validator and ``FakeGroupGenerator`` for vLLM."""
from __future__ import annotations

import asyncio
import json
import random
import threading

import pytest

from reliquary.miner.rl_async import (
    ROLLOUTS, AsyncMiner, AsyncMinerConfig, DryRunTransport, FakeGroupGenerator, FakeStreamGenerator, PromptPlanner,
    PromptRecord, RequestRefused, encode_submission, submission_digest,
)
from reliquary.protocol.rl_submit import decode_tokens

ENV = "math"
NOW = 1_800_000_000.0


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def page(*, cursor, latest=0, scans=(), prompts=(), more=False):
    return {"latest": latest, "cursor": cursor, "more": more, "scans": list(scans), "prompts": list(prompts)}


def scan(seq, prompt, verdict, k=0, rewards=(0.0,) * 16):
    return {"seq": seq, "environment": ENV, "prompt_idx": prompt, "checkpoint": k, "rewards": list(rewards),
            "verdict": verdict, "first_scan": True, "at": NOW}


def miner_for(*, quota=8, lease_batch=8, queue_groups=8, batch_groups=4, max_open=64, prompt_range=(0, 50),
              loaded_version=0, clock=None, transport=None, sync=None, state_dir=None, priority=None, generator=None):
    clock = clock or Clock()
    transport = transport or DryRunTransport(quota=quota, clock=clock)
    planner_kwargs = {} if priority is None else {"priority": priority}
    planner = PromptPlanner(ENV, prompt_range, rng=random.Random(0), clock=clock, **planner_kwargs)
    generator = generator or FakeGroupGenerator()
    if loaded_version is not None:
        generator.load_weights({}, loaded_version)
    config = AsyncMinerConfig(environments=(ENV,), lease_batch=lease_batch, queue_groups=queue_groups,
                              batch_groups=batch_groups, max_open=max_open, submit_backoff_s=(0.0,),
                              state_dir=state_dir)
    miner = AsyncMiner(config, transport=transport, generator=generator, planners={ENV: planner},
                       checkpoint_sync=sync, loaded_version=loaded_version, rng=random.Random(0), clock=clock)
    return miner, transport, generator, planner


# --- planner ---------------------------------------------------------------------------------------------
def test_a_prompt_taken_by_a_rival_waits_one_lease_lifetime_of_its_env():
    clock = Clock()
    planner = PromptPlanner(ENV, (0, 10), rng=random.Random(0), clock=clock)
    planner.wait_for_outcome(1)                                   # lifetime unknown: the fallback
    planner.note_lease_expiry(120.0)
    planner.wait_for_outcome(2)
    clock.now += 121
    assert planner.leasable(2) and not planner.leasable(1)        # the rival's lease has expired by now
    planner.wait_for_outcome(3, until=clock.now + 10)             # our own lease: until its own expiry
    clock.now += 11
    assert planner.leasable(3)


async def test_the_lease_answer_sets_the_envs_lease_lifetime_for_taken_prompts():
    clock = Clock()
    miner, transport, _, planner = miner_for(quota=50, prompt_range=(0, 4), clock=clock)
    transport.expiry_s = 90.0
    transport.refuse = {0: "taken"}
    await miner.refresh_quota()
    await miner.lease_more()
    assert planner.lease_expiry_s == 90.0 and not planner.leasable(0)
    clock.now += 91
    assert planner.leasable(0)


async def test_open_leases_lost_with_the_journal_are_recovered_from_the_validator(tmp_path):
    clock = Clock()
    miner, transport, _, _ = miner_for(quota=3, clock=clock)
    await miner.refresh_quota()
    await miner.lease_more()
    held = set(miner.open)
    again, _, generator, planner = miner_for(quota=3, clock=clock, transport=transport, state_dir=str(tmp_path))
    assert again.open == {}                                        # no journal
    assert await again.recover_leases() == 3 and set(again.open) == held
    assert await again.recover_leases() == 0                       # idempotent
    assert all(not planner.leasable(lease.prompt_idx) for lease in again.queue)
    while again.queue:
        await again.generate_once()
    await again.drain()
    assert {s["lease_id"] for s in transport.submissions} == held and again.stats["recovered"] == 3


def test_the_planner_skips_blacklisted_cooling_held_and_out_of_range_prompts():
    planner = PromptPlanner(ENV, (10, 20), rng=random.Random(0))
    planner.apply(page(cursor=3, latest=5, scans=[scan(1, 11, "too_easy"), scan(2, 12, "in_zone"),
                                                  scan(3, 13, "too_hard")],
                       prompts=[{"environment": ENV, "prompt_idx": 11, "state": "blacklisted", "until_k": None},
                                {"environment": ENV, "prompt_idx": 12, "state": "cooldown", "until_k": 8},
                                {"environment": ENV, "prompt_idx": 13, "state": "cooldown", "until_k": 5}]))
    planner.wait_for_outcome(14)
    assert not planner.leasable(11) and not planner.leasable(12) and not planner.leasable(14)
    assert planner.leasable(13)                                   # latest 5 reached until_k 5: the cooldown is over
    assert not planner.leasable(9) and not planner.leasable(20)
    picked = planner.pick(100)
    assert set(picked) <= set(range(10, 20)) - {11, 12, 14} and 13 in picked
    planner.apply(page(cursor=3, latest=8))
    assert planner.leasable(12)                                   # its cooldown ended at k 8
    planner.apply(page(cursor=4, scans=[scan(4, 14, "too_hard")]))
    assert planner.leasable(14)                                   # its scan arrived (no state: leasable again)


def test_the_default_order_prefers_never_scanned_then_in_zone_history():
    planner = PromptPlanner(ENV, (0, 4), rng=random.Random(0))
    planner.apply(page(cursor=3, scans=[scan(1, 0, "too_hard"), scan(2, 1, "in_zone"), scan(3, 1, "too_hard")]))
    planner.apply(page(cursor=4, scans=[scan(4, 2, "in_zone")]))
    # 3 never scanned, 2 always in zone, 1 half, 0 never in zone
    assert planner.pick(4) == [3, 2, 1, 0]
    assert planner.pick(2) == [3, 2]


def test_the_priority_is_overridable_and_none_skips():
    def only_scanned(record: PromptRecord | None):
        return None if record is None else -len(record.scans)

    planner = PromptPlanner(ENV, (0, 100), rng=random.Random(0), priority=only_scanned)
    planner.apply(page(cursor=3, scans=[scan(1, 7, "in_zone"), scan(2, 7, "too_hard"), scan(3, 9, "in_zone")]))
    assert planner.pick(10) == [7, 9]


async def test_the_state_reader_follows_the_cursor_through_pages():
    miner, transport, _, planner = miner_for()
    transport.pages = [page(cursor=2, scans=[scan(1, 3, "in_zone"), scan(2, 4, "too_easy")], more=True),
                       page(cursor=3, scans=[scan(3, 5, "too_hard")]),
                       page(cursor=3, scans=[scan(3, 5, "too_hard")])]     # a resent scan is not counted twice
    await miner.refresh_state()
    assert transport.state_calls == [0, 2]
    assert planner.cursor == 3 and sorted(planner.records) == [3, 4, 5]
    await miner.refresh_state()
    assert transport.state_calls == [0, 2, 3] and len(planner.records[5].scans) == 1


# --- leasing ---------------------------------------------------------------------------------------------
async def test_leases_never_exceed_the_quota():
    miner, transport, _, _ = miner_for(quota=3, max_open=10)
    await miner.lease_more()                                      # no quota read yet: nothing asked
    assert transport.lease_calls == []
    await miner.refresh_quota()
    assert await miner.lease_more() == 3
    assert len(transport.lease_calls[0][2]) == 3
    assert await miner.lease_more() == 0 and len(transport.lease_calls) == 1
    assert len(miner.open) == 3


async def test_leases_are_asked_in_batches_up_to_the_queue_at_the_held_checkpoint():
    miner, transport, _, _ = miner_for(quota=20, lease_batch=3, queue_groups=7, loaded_version=4)
    transport.latest = 4
    await miner.refresh_quota()
    while await miner.lease_more():
        pass
    assert [len(ids) for _, _, ids in transport.lease_calls] == [3, 3, 1]
    assert {k for _, k, _ in transport.lease_calls} == {4}
    asked = [p for _, _, ids in transport.lease_calls for p in ids]
    assert len(asked) == len(set(asked))                          # a held prompt is never asked again
    assert len(miner.queue) == 7


async def test_refusals_move_the_planner_and_the_scheduler():
    miner, transport, _, planner = miner_for(quota=50, prompt_range=(0, 6))
    transport.refuse = {0: "taken", 1: "blacklisted", 2: "cooling", 3: "quota"}
    await miner.refresh_quota()
    await miner.lease_more()
    assert miner.stats["refused"] == {"taken": 1, "blacklisted": 1, "cooling": 1, "quota": 1}
    assert not planner.leasable(0) and not planner.leasable(2)    # waited for until their scans arrive
    assert planner.records[1].state == "blacklisted"
    assert miner.quota == len(miner.open) == 2                    # the validator counted fewer: no more for now
    assert miner.lease_room() == 0
    transport.refuse = {4: "stale"}
    miner.quota = 50
    miner.open.clear()
    miner.queue.clear()
    planner._waiting.clear()
    planner.records.clear()
    await miner.lease_more()
    assert miner._stale and miner.lease_room() == 0               # sync first; the swap lifts it


async def test_an_episode_task_the_fleet_cannot_host_is_held_back_for_a_while():
    clock = Clock()
    miner, transport, _, planner = miner_for(quota=50, prompt_range=(0, 6), clock=clock)
    transport.refuse = {0: "fleet_unavailable", 1: "task_unservable"}
    await miner.refresh_quota()
    await miner.lease_more()
    assert miner.stats["refused"] == {"fleet_unavailable": 1, "task_unservable": 1}
    assert not planner.leasable(0) and not planner.leasable(1)
    clock.now += 61
    assert planner.leasable(0) and not planner.leasable(1)        # a full fleet is asked again soon


async def test_a_refused_request_backs_off_without_leasing():
    miner, transport, _, _ = miner_for()

    async def busy(*args):
        raise RequestRefused("lease_busy", retryable=True, retry_after=0.0)

    transport.lease = busy
    await miner.refresh_quota()
    assert await miner.lease_more() == 0 and miner.open == {}


# --- generation and submission ---------------------------------------------------------------------------
async def test_every_lease_ends_in_a_submission():
    clock = Clock()
    miner, transport, generator, _ = miner_for(quota=6, batch_groups=4, clock=clock)
    transport.submit_failures = [RequestRefused("internal_error", retryable=True)]
    transport.lost_answers = 1                     # received, answer lost: the resend gets the original verdict
    await miner.refresh_quota()
    await miner.lease_more()
    granted = set(miner.open)
    assert len(granted) == 6
    while miner.queue:
        await miner.generate_once()
    await miner.drain()
    assert [len(call) for call in generator.calls] == [4, 2]
    submitted = {s["lease_id"] for s in transport.submissions}
    # one transient failure retried, one lost answer resent and answered with the original verdict
    assert miner.stats["submitted"] == 6 and len(submitted) == 6 == len(transport.submissions) and submitted == granted
    assert miner.open == {} and miner.stats["submit_failed"] == 0 and miner.stats["verdicts"] == {"received": 6}
    first = transport.submissions[0]
    assert first["k_generated"] == 0 and [r["index"] for r in first["rollouts"]] == list(range(ROLLOUTS))
    assert len(decode_tokens(first["rollouts"][0]["tokens"], max_tokens=64)) == generator.length


async def test_leases_are_generated_earliest_expiry_first_and_dead_ones_dropped():
    clock = Clock()
    miner, transport, generator, _ = miner_for(quota=3, batch_groups=1, clock=clock)
    await miner.refresh_quota()
    transport.expiry_s = 100.0
    await miner.lease_more()
    ids = list(miner.open)
    leases = [miner.open[i] for i in ids]
    miner.queue = [leases[2], leases[0], leases[1]]
    object.__setattr__(leases[1], "expires_at", NOW - 1)          # already dead: never generated
    object.__setattr__(leases[2], "expires_at", NOW + 50)         # due before leases[0]
    await miner.generate_once()
    await miner.generate_once()
    await miner.drain()
    assert generator.calls == [(leases[2].lease_id,), (leases[0].lease_id,)]
    assert miner.stats["expired_unsent"] == 1 and leases[1].lease_id not in miner.open


async def test_a_submission_stops_retrying_at_the_lease_expiry():
    clock = Clock()
    miner, transport, _, _ = miner_for(quota=1, clock=clock)
    await miner.refresh_quota()
    await miner.lease_more()
    lease = next(iter(miner.open.values()))

    async def down(payload):
        clock.now = lease.expires_at + 1
        raise RequestRefused("ConnectError", retryable=True)

    transport.submit = down
    assert await miner.submit(lease, [[1]] * ROLLOUTS, 0) is False
    assert miner.open == {} and miner.stats["submit_failed"] == 1


async def test_a_failed_generation_is_retried_once_then_given_up():
    miner, transport, generator, _ = miner_for(quota=1)
    calls = []

    def broken(leases):
        calls.append(len(leases))
        raise RuntimeError("cuda")

    generator.generate = broken
    await miner.refresh_quota()
    await miner.lease_more()
    await miner.generate_once()
    assert len(miner.queue) == 1
    await miner.generate_once()
    assert calls == [1, 1] and miner.queue == [] and miner.open == {}
    await miner.drain()
    assert miner.stats["forfeited"] == 1 and transport.open == {}          # the validator frees its slot now


async def test_a_lease_that_failed_alone_is_given_up_and_its_batch_mates_are_submitted():
    """Episodes: a lease's sessions are single-use, so a failed lease comes back as its exception, is given up at
    once (no replay) and the other leases of the batch keep their groups."""
    miner, transport, generator, _ = miner_for(quota=2)
    plain = generator.generate

    async def agenerate(leases):
        groups = plain(leases)
        return [RuntimeError("rollout 3: unusable") if i == 0 else g for i, g in enumerate(groups)]

    generator.agenerate = agenerate
    await miner.refresh_quota()
    await miner.lease_more()
    assert await miner.generate_once() == 1 and miner.queue == []
    await miner.drain()
    assert miner.stats["forfeited"] == 1 and miner.stats["submitted"] == 1 and transport.open == {}


async def test_open_leases_survive_a_restart(tmp_path):
    clock = Clock()
    miner, transport, _, _ = miner_for(quota=2, clock=clock, state_dir=str(tmp_path))
    await miner.refresh_quota()
    await miner.lease_more()
    assert len(json.loads((tmp_path / "rl-open-leases.json").read_text())) == 2
    again, _, _, planner = miner_for(quota=2, clock=clock, transport=transport, state_dir=str(tmp_path))
    assert set(again.open) == set(miner.open) and len(again.queue) == 2
    assert all(not planner.leasable(lease.prompt_idx) for lease in again.queue)
    await again.generate_once()
    await again.drain()
    assert len(transport.submissions) == 2
    assert json.loads((tmp_path / "rl-open-leases.json").read_text()) == []


def test_the_submission_is_the_validators_wire_and_a_group_needs_all_rollouts():
    from reliquary.miner.rl_async import HeldLease
    from reliquary.protocol.rl_submit import rollouts_digest
    from reliquary.validator.rl_lease_routes import RlSubmitRequestBody, submitted_rollouts

    rollouts = [[i, i + 70_000, 5] for i in range(ROLLOUTS)]
    rollouts[3] = ([7, 8, 9], (1, 3))                           # a BFT force span where the generator has one
    lease = HeldLease("l", ENV, 1, 0, "ab", NOW + 1)
    payload = encode_submission(lease, rollouts, 2)
    body = RlSubmitRequestBody.model_validate({"miner_hotkey": "x", "at": 1, **payload, "signature": "s"})
    group = submitted_rollouts(body, max_tokens=8)
    assert [list(r.tokens) for r in group] == [list(r[0]) if isinstance(r, tuple) else r for r in rollouts]
    assert group[3].force_span == (1, 3) and body.k_generated == 2
    assert submission_digest(payload) == rollouts_digest(
        [(r.index, r.tokens, r.force_span) for r in body.rollouts])     # what the validator verifies
    assert "k_generated" not in encode_submission(lease, rollouts)
    with pytest.raises(ValueError):
        encode_submission(lease, rollouts[:-1], 0)


# --- checkpoints -----------------------------------------------------------------------------------------
class GatedSync:
    """``CheckpointSync`` stand-in: each sync to the index's latest waits for ``gate`` (a download in flight)."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.calls: list[tuple[int | None, int]] = []

    def sync(self, response, state, version):
        latest = response["index"]["latest"]["version"]
        self.calls.append((version, latest))
        assert self.gate.wait(5)
        plan = type("Plan", (), {"anchor": None, "deltas": [object()], "bytes": 10})()
        return {"v": latest}, latest, plan


def index(version):
    return {"index": {"latest": {"version": version, "sha256": "x"}}}


async def test_weights_are_swapped_between_generation_batches_only():
    sync = GatedSync()
    miner, transport, generator, _ = miner_for(quota=8, batch_groups=2, sync=sync)
    await miner.refresh_quota()
    await miner.lease_more()
    transport.checkpoints = [index(1)]
    follow = asyncio.create_task(miner.follow_checkpoint_once())    # downloading while generating on k=0
    await asyncio.sleep(0)
    assert await miner.generate_once() == 2
    assert generator.versions_used == [0] and generator.loads == [0]
    sync.gate.set()
    assert await follow == 1
    assert generator.loads == [0] and miner.loaded_version == 0     # nothing swapped outside a batch boundary
    transport.checkpoints = [index(2)]
    assert await miner.follow_checkpoint_once() is None             # k 1 not taken yet: the copy is not patched
    assert sync.calls == [(0, 1)]
    assert await miner.generate_once() == 2                         # swap first, then this batch on k 1
    assert generator.loads == [0, 1] and generator.versions_used == [0, 1]
    await miner.drain()
    by_k = sorted(s["k_generated"] for s in transport.submissions)
    assert by_k == [0, 0, 1, 1] and {transport.granted[s["lease_id"]].checkpoint for s in transport.submissions} == {0}
    await miner.lease_more()
    assert transport.lease_calls[-1][1] == 1                        # new leases are asked at the held k
    transport.checkpoints = [index(2)]
    assert await miner.follow_checkpoint_once() == 2 and sync.calls[-1] == (1, 2)


async def test_a_stale_refusal_waits_for_newer_weights():
    sync = GatedSync()
    sync.gate.set()
    miner, transport, generator, _ = miner_for(quota=8, sync=sync)
    transport.refuse = {p: "stale" for p in range(50)}
    await miner.refresh_quota()
    await miner.lease_more()
    assert miner._stale and miner._wake_checkpoint.is_set()
    transport.checkpoints = [index(3)]
    assert await miner.follow_checkpoint_once() == 3
    await miner.generate_once()
    assert miner.loaded_version == 3 and not miner._stale and generator.loads == [0, 3]


async def test_the_loops_run_and_stop_after_submitting_what_was_generated():
    miner, transport, _, _ = miner_for(quota=4, batch_groups=2)
    miner.config = AsyncMinerConfig(environments=(ENV,), lease_poll_s=0.01, state_poll_s=0.01, quota_poll_s=0.01,
                                    checkpoint_poll_s=0.01, batch_groups=2, submit_backoff_s=(0.0,))
    stop = asyncio.Event()
    run = asyncio.create_task(miner.run(stop))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(transport.submissions) >= 8:
            break
    stop.set()
    await asyncio.wait_for(run, 5)
    assert len(transport.submissions) >= 8
    assert all(s["lease_id"] not in miner.open for s in transport.submissions)


async def test_a_generator_that_cannot_load_weights_stops_the_miner_instead_of_leasing_for_it():
    miner, transport, generator, _ = miner_for(quota=4, batch_groups=2)
    miner.config = AsyncMinerConfig(environments=(ENV,), lease_poll_s=0.01, state_poll_s=0.01, quota_poll_s=0.01,
                                    checkpoint_poll_s=0.01, batch_groups=2, submit_backoff_s=(0.0,))

    def broken(state, version):
        raise RuntimeError("engine core initialization failed")
    generator.load_weights = broken
    miner._state, miner._state_version = {}, 1
    miner._ready.set()                               # a new checkpoint to swap in before the next batch
    with pytest.raises(RuntimeError, match="engine core"):
        await asyncio.wait_for(miner.run(asyncio.Event()), 5)


def test_a_batch_larger_than_the_queue_is_refused():
    with pytest.raises(ValueError, match="queue_groups"):
        AsyncMinerConfig(environments=(ENV,), batch_groups=32, queue_groups=8)


# --- wire: the real lease routes -------------------------------------------------------------------------
async def test_the_http_transport_leases_from_the_real_routes(tmp_path):
    bt = pytest.importorskip("bittensor")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    from reliquary.miner.rl_async import HttpRlTransport
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_lease_routes import build_rl_lease_router
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector

    miner_key = bt.Keypair.create_from_uri("//Alice")
    validator_key = bt.Keypair.create_from_uri("//Bob")
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}, prompt_range={ENV: (0, 50)}),
                     quota=QuotaController(QuotaConfig(initial=4)))
    book.adopt_checkpoint(0)
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=validator_key.ss58_address))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v") as client:
        transport = HttpRlTransport(client, base_url="http://v", hotkey=miner_key.ss58_address,
                                    validator_hotkey=validator_key.ss58_address,
                                    sign_binding=lambda binding: miner_key.sign(binding).hex())
        miner, _, _, _ = miner_for(transport=transport, lease_batch=8)
        await miner.refresh_quota()
        assert miner.quota == 4
        assert await miner.lease_more() == 4
        assert book.in_use(miner_key.ss58_address) == 4
        book.record_outcome(ENV, 7, 0, [1.0] * 16, "too_easy")
        book.record_outcome(ENV, 7, 0, [1.0] * 16, "too_easy")              # too easy twice: blacklisted
        await miner.refresh_state()
        assert miner.planners[ENV].records[7].state == "blacklisted"
        with pytest.raises(RequestRefused) as refused:
            await transport.lease(ENV, 0, [1_000])
        assert refused.value.reason == "lease_invalid" and not refused.value.retryable


async def test_the_http_transport_submits_resends_and_recovers_against_the_real_routes(tmp_path):
    """The miner's submission wire and signature are the validator's: a signed group is graded, a resend (lost
    answer) gets the original verdict, and a miner without its journal recovers its open leases (``/rl/lease/mine``)."""
    bt = pytest.importorskip("bittensor")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    from reliquary.miner.rl_async import HeldLease, HttpRlTransport
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_lease_routes import build_rl_lease_router
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions
    from tests.unit.test_rl_submit import ENV as MATH, MathEnv, Tokenizer, group

    miner_key = bt.Keypair.create_from_uri("//Alice")
    rival_key = bt.Keypair.create_from_uri("//Charlie")
    validator_key = bt.Keypair.create_from_uri("//Bob")
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={MATH: 600.0}, cooldown_of=lambda e: 2),
                     quota=QuotaController(QuotaConfig(initial=4)))
    book.adopt_checkpoint(0)
    subs = RlSubmissions(book, GroupGrader({MATH: MathEnv()}, Tokenizer(), eos_token_ids=(1,)), rng=random.Random(0))
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=validator_key.ss58_address, submissions=subs))

    def transport_for(client, key):
        return HttpRlTransport(client, base_url="http://v", hotkey=key.ss58_address,
                               validator_hotkey=validator_key.ss58_address,
                               sign_binding=lambda binding: key.sign(binding).hex())

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v") as client:
        transport = transport_for(client, miner_key)
        answer = await transport.lease(MATH, 0, [1, 2])
        assert answer["lease_expiry_s"] == 600.0 and len(answer["leases"]) == 2
        held = [HeldLease.from_wire(item) for item in answer["leases"]]
        mine = await transport.my_leases()
        assert {item["lease_id"] for item in mine["leases"]} == {lease.lease_id for lease in held}
        assert mine["lease_expiry_s"] == {MATH: 600.0}
        assert (await transport_for(client, rival_key).my_leases())["leases"] == []    # only the caller's own
        rollouts = [list(r.tokens) for r in group(8)]
        payload = encode_submission(held[0], rollouts, 0)
        first = await transport.submit(payload)
        assert first["verdict"] == "in_zone" and first["rewards"] == [1.0] * 8 + [0.0] * 8
        again = await transport.submit(payload)
        assert again == {**first, "resent": True} and subs.buffer.depth() == {MATH: 1}
        # the recovered miner (no journal) finds the second lease and submits it through the same wire
        planner = PromptPlanner(MATH, (0, 50), rng=random.Random(0))
        generator = FakeGroupGenerator()
        generator.load_weights({}, 0)
        generator.generate = lambda leases: [rollouts for _ in leases]
        miner = AsyncMiner(AsyncMinerConfig(environments=(MATH,), submit_backoff_s=(0.0,)), transport=transport,
                           generator=generator, planners={MATH: planner}, loaded_version=0)
        assert await miner.recover_leases() == 1 and set(miner.open) == {held[1].lease_id}
        assert planner.lease_expiry_s == 600.0
        await miner.generate_once()
        await miner.drain()
        assert miner.stats["verdicts"] == {"in_zone": 1} and book.lease(held[1].lease_id).state == "submitted"


def test_write_hf_dir_reloads_in_transformers_from_several_shards(tmp_path, monkeypatch):
    import functools

    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    from reliquary.miner.rl_async import write_hf_dir
    from reliquary.shared import ckpt_delta

    base = tmp_path / "base"
    LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=2).save_pretrained(base)
    state = LlamaForCausalLM(LlamaConfig.from_pretrained(base)).state_dict()
    monkeypatch.setattr(ckpt_delta, "write_anchor_shards",
                        functools.partial(ckpt_delta.write_anchor_shards, shard_bytes=4096))
    dest = write_hf_dir(state, base, tmp_path / "hf")
    assert len(list(dest.glob("*.safetensors"))) > 1
    loaded = AutoModelForCausalLM.from_pretrained(dest).state_dict()
    assert all(torch.equal(loaded[k], v) for k, v in state.items() if k in loaded)


# --- continuous batching ------------------------------------------------------------------------------------
def streaming(steps=None, **kwargs):
    return miner_for(generator=FakeStreamGenerator(steps=steps), **kwargs)


async def test_a_long_group_does_not_hold_back_the_others():
    miner, transport, generator, _ = streaming(quota=8, batch_groups=2, queue_groups=8)
    await miner.refresh_quota()
    await miner.lease_more()
    slow = sorted(miner.queue, key=lambda lease: lease.expires_at)[0]
    generator.steps = {slow.prompt_idx: 4}                          # its longest rollout takes 4 steps
    for _ in range(3):
        await miner.generate_once()
    await miner.drain()
    assert list(miner.running) == [slow.lease_id]                   # each fast group starts and ends in one call
    assert len(transport.submissions) == 3                          # the slot beside it served three leases
    assert slow.lease_id not in {s["lease_id"] for s in transport.submissions}
    await miner.generate_once()
    await miner.drain()
    assert slow.lease_id in {s["lease_id"] for s in transport.submissions}


class FakeEpisodeStream(FakeStreamGenerator):
    """``aadvance`` on the event loop, as the episode generator plays leases; ``fail`` names leases that end
    as their exception."""

    def __init__(self, steps=None, fail=()):
        super().__init__(steps=steps)
        self.fail = set(fail)

    async def aadvance(self, leases, budget_s):
        done = self.advance(leases, budget_s)
        return [(lease_id, RuntimeError("unusable") if lease_id in self.fail else group) for lease_id, group in done]

    def abort(self):
        self.running.clear()


async def test_episode_leases_are_played_continuously_and_a_failed_one_alone_is_given_up():
    miner, transport, generator, _ = miner_for(generator=FakeEpisodeStream(), quota=4, batch_groups=2,
                                               queue_groups=8)
    await miner.refresh_quota()
    await miner.lease_more()
    ordered = sorted(miner.queue, key=lambda lease: lease.expires_at)
    generator.steps = {ordered[0].prompt_idx: 5}                    # a long lease
    generator.fail = {ordered[1].lease_id}
    for _ in range(3):
        await miner.generate_once()
    await miner.drain()
    assert ordered[0].lease_id in miner.running                     # still playing, the slot beside it went on
    assert miner.stats["forfeited"] == 1 and len(transport.submissions) == 2


async def test_new_weights_wait_for_the_running_groups():
    miner, transport, generator, _ = streaming(quota=8, batch_groups=2)
    await miner.refresh_quota()
    await miner.lease_more()
    generator.steps = {lease.prompt_idx: 2 for lease in miner.queue}
    await miner.generate_once()                                     # two groups start on k 0
    miner._state, miner._state_version = {}, 1
    miner._ready.set()
    started = len(generator.started)
    await miner.generate_once()                                     # they finish; nothing new starts
    assert len(generator.started) == started and not miner.running and generator.loads == [0]
    await miner.generate_once()                                     # drained: swap, then start on k 1
    await miner.drain()
    assert generator.loads == [0, 1] and len(generator.started) == started + 2
    assert sorted(s["k_generated"] for s in transport.submissions) == [0, 0]


async def test_a_generator_that_swaps_running_takes_new_weights_at_once():
    """No drain: the running groups go on with the new weights, and new leases keep starting."""
    miner, transport, generator, _ = streaming(quota=8, batch_groups=2)
    generator.swaps_running = True
    await miner.refresh_quota()
    await miner.lease_more()
    generator.steps = {lease.prompt_idx: 3 for lease in miner.queue}
    await miner.generate_once()                                     # two groups start on k 0
    miner._state, miner._state_version = {}, 1
    miner._ready.set()
    running = set(miner.running)
    await miner.generate_once()                                     # swapped with both still running
    assert generator.loads == [0, 1] and set(miner.running) == running and not miner._ready.is_set()
    assert miner.stats["swap_wait_max_s"] < 1.0
    await miner.generate_once()
    await miner.drain()
    assert {s["lease_id"] for s in transport.submissions} == running   # the groups that spanned the swap
    assert sorted(s["k_generated"] for s in transport.submissions) == [1, 1]


async def test_a_failed_running_swap_falls_back_to_draining():
    miner, transport, generator, _ = streaming(quota=8, batch_groups=2)
    generator.swaps_running = True
    await miner.refresh_quota()
    await miner.lease_more()
    generator.steps = {lease.prompt_idx: 3 for lease in miner.queue}
    await miner.generate_once()
    loads = generator.load_weights

    def refuse(state, version):
        raise RuntimeError("push failed")

    generator.load_weights = refuse
    miner._state, miner._state_version = {}, 1
    miner._ready.set()
    await miner.generate_once()                                     # the swap fails: the miner drains instead
    assert miner._ready.is_set() and not miner._swaps_running()
    generator.load_weights = loads
    for _ in range(4):
        await miner.generate_once()
    await miner.drain()
    assert generator.loads[-1] == 1 and len(transport.submissions) >= 2


async def test_a_failed_engine_step_requeues_the_running_leases():
    miner, transport, generator, _ = streaming(quota=4, batch_groups=2)
    await miner.refresh_quota()
    await miner.lease_more()
    generator.fail_next = True
    assert await miner.generate_once() == 0
    assert not miner.running and not generator.running and len(miner.queue) == 4
    await miner.generate_once()
    await miner.drain()
    assert len(transport.submissions) == 2



# --- an episode env's declared rows --------------------------------------------------------------------------------
def test_the_planner_never_picks_an_excluded_row():
    from reliquary.protocol.rl_prompt_ranges import PromptRows

    rows = PromptRows(10, 40, ((12, 30), (35, 36)))
    planner = PromptPlanner(ENV, rows, rng=random.Random(0))
    assert planner.range == (10, 40)
    picked = set()
    for _ in range(50):
        picked.update(planner.pick(8))
    assert picked and all(p in rows for p in picked) and picked.isdisjoint({*range(12, 30), 35})
    assert not planner.leasable(20) and planner.leasable(11)


def test_the_miner_draws_an_episode_env_from_its_blocks_rows():
    from reliquary.miner.rl_async import planner_ranges
    from reliquary.protocol.rl_prompt_ranges import PromptRows

    blocks = {"swe": {"episode": {}, "rows": {"start": 5, "end": 50, "exclude": [[10, 20], 30]}},
              "tmax": {"episode": {}}}
    ranges = planner_ranges(("swe", "tmax", ENV), sources={"swe": [None] * 60, "tmax": [None] * 7}, blocks=blocks,
                            load_environment=lambda name: [None] * 9)
    assert ranges == {"swe": PromptRows(5, 50, ((10, 20), (30, 31))), "tmax": PromptRows(0, 7), ENV: (0, 9)}
    with pytest.raises(ValueError, match="past the installed taskset"):
        planner_ranges(("swe",), sources={"swe": [None] * 49}, blocks=blocks)


# --- sandbox failures: forfeited at once, with the reason -----------------------------------------------------
async def test_a_lease_our_sandbox_failed_is_forfeited_with_that_reason_and_others_without():
    from reliquary.miner.rl_async_episodes import EpisodeLeaseFailed

    miner, transport, generator, _ = miner_for(quota=3)
    plain = generator.generate

    async def agenerate(leases):
        groups = plain(leases)
        failures = [EpisodeLeaseFailed("rollout 3: the sandbox ended its episode box_failed",
                                       sandbox_failure="box_failed"), EpisodeLeaseFailed("rollout 1: unusable")]
        return [failures[i] if i < 2 else g for i, g in enumerate(groups)]

    generator.agenerate = agenerate
    await miner.refresh_quota()
    await miner.lease_more()
    ids = [lease.lease_id for lease in sorted(miner.queue, key=lambda lease: lease.expires_at)]
    assert await miner.generate_once() == 1
    await miner.drain()
    assert sorted(transport.forfeits) == sorted([(ids[0], "sandbox_failure"), (ids[1], None)])
    assert miner.stats["forfeited"] == 2 and miner.stats["released"] == 1 and miner.stats["submitted"] == 1


async def test_the_http_forfeit_names_the_sandbox_failure_and_falls_back_for_an_older_validator():
    httpx = pytest.importorskip("httpx")
    from reliquary.miner.rl_async import HttpRlTransport

    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reason" in body:                                        # an older validator: extra fields forbidden
            return httpx.Response(422, json={"reason": "malformed_request"})
        return httpx.Response(200, json={"lease_id": body["lease_id"], "state": "expired", "quota": 3})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://v") as client:
        transport = HttpRlTransport(client, base_url="http://v", hotkey="5Miner", validator_hotkey="5Val",
                                    sign_binding=lambda binding: "00", clock=lambda: NOW)
        answer = await transport.forfeit("L1", "sandbox_failure")
        assert answer["state"] == "expired"
        assert [b.get("reason") for b in bodies] == ["sandbox_failure", None]
        bodies.clear()
        await transport.forfeit("L2")
        assert [b.get("reason") for b in bodies] == [None]


async def test_the_transport_tags_submissions_with_its_instance_and_drops_the_tag_for_an_older_validator(tmp_path):
    """--share-setup: each submission names the process's /miner/setup instance; a validator that does not know the
    field refuses it as malformed, and the same group goes through once without it (the tag is then never sent)."""
    bt = pytest.importorskip("bittensor")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from reliquary.miner.rl_async import HeldLease, HttpRlTransport
    from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
    from reliquary.validator.rl_lease_routes import build_rl_lease_router
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector
    from reliquary.validator.rl_submit import GroupGrader, RlSubmissions
    from tests.unit.test_rl_submit import ENV as MATH, MathEnv, Tokenizer, group

    miner_key = bt.Keypair.create_from_uri("//Alice")
    validator_key = bt.Keypair.create_from_uri("//Bob")
    book = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={MATH: 600.0}, cooldown_of=lambda e: 2),
                     quota=QuotaController(QuotaConfig(initial=4)))
    book.adopt_checkpoint(0)
    subs = RlSubmissions(book, GroupGrader({MATH: MathEnv()}, Tokenizer(), eos_token_ids=(1,)), rng=random.Random(0))
    app = FastAPI()
    app.include_router(build_rl_lease_router(book=book, collector=LeaseCollector(book, interval_s=0.0),
                                             validator_hotkey=validator_key.ss58_address, submissions=subs))
    seen, older = [], False

    @app.middleware("http")
    async def older_validator(request, call_next):           # forbids the field, as before it existed
        body = json.loads(await request.body() or b"{}")
        seen.append("instance_id" in body)
        if request.url.path == "/rl/submit" and "instance_id" in body and older:
            return JSONResponse(
                status_code=422, content={"error": "malformed_request", "reason": "malformed_request",
                                          "errors": [{"loc": ["instance_id"], "type": "extra_forbidden"}]})
        return await call_next(request)

    rollouts = [list(r.tokens) for r in group(8)]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v") as client:
        transport = HttpRlTransport(client, base_url="http://v", hotkey=miner_key.ss58_address,
                                    validator_hotkey=validator_key.ss58_address,
                                    sign_binding=lambda binding: miner_key.sign(binding).hex())
        transport.instance_id = "c" * 32
        held = [HeldLease.from_wire(item) for item in (await transport.lease(MATH, 0, [1, 2, 3]))["leases"]]
        assert (await transport.submit(encode_submission(held[0], rollouts, 0)))["verdict"] == "in_zone"
        assert book.db.execute("SELECT instance_id FROM rl_group_instances").fetchall() == [("c" * 32,)]
        older, seen[:] = True, []
        assert (await transport.submit(encode_submission(held[1], rollouts, 0)))["verdict"] == "in_zone"
        assert seen == [True, False] and transport.instance_id is None
        seen[:] = []
        assert (await transport.submit(encode_submission(held[2], rollouts, 0)))["verdict"] == "in_zone"
        assert seen == [False]
