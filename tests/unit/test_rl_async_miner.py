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
    ROLLOUTS, AsyncMiner, AsyncMinerConfig, DryRunTransport, FakeGroupGenerator, PromptPlanner, PromptRecord,
    RequestRefused, decode_tokens, encode_submission, encode_tokens,
)

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
              loaded_version=0, clock=None, transport=None, sync=None, state_dir=None, priority=None):
    clock = clock or Clock()
    transport = transport or DryRunTransport(quota=quota, clock=clock)
    planner_kwargs = {} if priority is None else {"priority": priority}
    planner = PromptPlanner(ENV, prompt_range, rng=random.Random(0), clock=clock, **planner_kwargs)
    generator = FakeGroupGenerator()
    if loaded_version is not None:
        generator.load_weights({}, loaded_version)
    config = AsyncMinerConfig(environments=(ENV,), lease_batch=lease_batch, queue_groups=queue_groups,
                              batch_groups=batch_groups, max_open=max_open, submit_backoff_s=(0.0,),
                              state_dir=state_dir)
    miner = AsyncMiner(config, transport=transport, generator=generator, planners={ENV: planner},
                       checkpoint_sync=sync, loaded_version=loaded_version, rng=random.Random(0), clock=clock)
    return miner, transport, generator, planner


# --- planner ---------------------------------------------------------------------------------------------
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
    transport.submit_failures = [RequestRefused("internal_error", retryable=True),
                                 RequestRefused("already_submitted", retryable=False)]
    await miner.refresh_quota()
    await miner.lease_more()
    granted = set(miner.open)
    assert len(granted) == 6
    while miner.queue:
        await miner.generate_once()
    await miner.drain()
    assert [len(call) for call in generator.calls] == [4, 2]
    submitted = {s["lease_id"] for s in transport.submissions}
    # one transient failure retried, one "already received" (a lost answer's resend) counted as delivered
    assert miner.stats["submitted"] == 6 and len(submitted) == 5 and submitted <= granted
    assert miner.open == {} and miner.stats["submit_failed"] == 0
    first = transport.submissions[0]
    assert first["k_generated"] == 0 and len(decode_tokens(first["tokens"])) == ROLLOUTS


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
    miner, _, generator, _ = miner_for(quota=1)
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


def test_tokens_round_trip_and_a_group_needs_all_rollouts():
    rollouts = [[i, i + 70_000, 5] for i in range(ROLLOUTS)]
    assert decode_tokens(encode_tokens(rollouts)) == rollouts
    from reliquary.miner.rl_async import HeldLease

    lease = HeldLease("l", ENV, 1, 0, "ab", NOW + 1)
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
    assert by_k == [0, 0, 1, 1] and {s["checkpoint"] for s in transport.submissions} == {0}
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
        await miner.refresh_state()
        assert miner.planners[ENV].records[7].state == "blacklisted"
        with pytest.raises(RequestRefused) as refused:
            await transport.lease(ENV, 0, [1_000])
        assert refused.value.reason == "lease_invalid" and not refused.value.retryable
