"""Light leases on a draw job's slots: two miners do not decode the same sample, a job's tail never waits for a
vanished one, an older miner without leases still submits what no one holds."""
from __future__ import annotations

import random
from types import SimpleNamespace

from reliquary.corpus.draw_leases import (
    REASON_DRAW_SLOT_LEASED,
    DrawLeases,
    free_draw_slots,
    lease_seconds,
)
from reliquary.corpus.slots import SlotLedger
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _job(prompts=3, slots=2, max_new_tokens=1000, n=1, episode=None):
    return SimpleNamespace(prompt_start=10, prompt_count=prompts, prompt_end=10 + prompts, slots_per_prompt=slots,
                           sampling=SimpleNamespace(max_new_tokens=max_new_tokens, n=n), episode=episode)


def _setup(prompts=3, slots=2, **kw):
    job = _job(prompts, slots)
    ledger = SlotLedger(prompt_start=10, prompt_count=prompts, slots_per_prompt=slots)
    taken: set = set()
    clock = Clock()
    leases = DrawLeases(job, clock=clock, rng=random.Random(0), seconds=100.0, **kw)
    free = free_draw_slots(job, ledger, lambda p, s: (p, s) in taken)
    return job, ledger, taken, clock, leases, free


def test_each_lease_is_a_distinct_free_slot():
    _, _, _, _, leases, free = _setup()
    got = [leases.grant(f"h{i}", free) for i in range(6)]
    assert len({(lease.prompt_index, lease.slot) for lease in got}) == 6
    # Every slot is leased, none for long: nothing for a seventh miner.
    assert leases.grant("h7", free) is None


def test_another_hotkey_cannot_submit_a_leased_slot_its_holder_can():
    _, _, _, _, leases, free = _setup()
    lease = leases.grant("a", free)
    assert leases.refusal("b", lease.prompt_index, lease.slot) == REASON_DRAW_SLOT_LEASED
    assert leases.refusal("a", lease.prompt_index, lease.slot) is None
    # An older miner, leasing nothing, submits any slot no one holds.
    other = next((p, s) for p in range(10, 13) for s in range(2) if (p, s) != (lease.prompt_index, lease.slot))
    assert leases.refusal("old", *other) is None


def test_a_lease_ends_when_its_slot_is_taken_released_or_expired():
    _, _, taken, clock, leases, free = _setup(prompts=1, slots=1)
    lease = leases.grant("a", free)
    leases.settle(lease.prompt_index, lease.slot)
    assert leases.refusal("b", lease.prompt_index, lease.slot) is None
    lease = leases.grant("a", free)
    assert not leases.release("b", lease.lease_id) and leases.release("a", lease.lease_id)
    lease = leases.grant("a", free)
    clock.now += 101
    assert leases.refusal("b", lease.prompt_index, lease.slot) is None and len(leases) == 0
    taken.add((lease.prompt_index, lease.slot))
    assert leases.grant("a", free) is None                      # the only slot is taken


def test_the_tail_goes_to_a_second_lessee_after_half_the_lease():
    _, _, _, clock, leases, free = _setup(prompts=1, slots=1)
    first = leases.grant("a", free)
    assert leases.grant("b", free) is None
    clock.now += 50
    second = leases.grant("b", free)
    assert (second.prompt_index, second.slot) == (first.prompt_index, first.slot)
    # Both may submit; the first valid one takes it. Never a third, never the holder twice.
    assert leases.refusal("a", first.prompt_index, first.slot) is None
    assert leases.refusal("b", first.prompt_index, first.slot) is None
    assert leases.grant("c", free) is None and leases.grant("a", free) in (None, "too_many_leases")


def test_a_hotkey_holds_a_bounded_number_of_leases():
    _, _, _, _, leases, free = _setup(prompts=10, slots=4, max_per_hotkey=3)
    assert all(leases.grant("a", free) is not None for _ in range(3))
    assert leases.grant("a", free) == "too_many_leases"
    assert leases.grant("b", free) is not None


def test_a_hotkey_whose_leases_expire_unused_holds_fewer():
    _, _, _, clock, leases, free = _setup(prompts=10, slots=4)
    assert leases.cap("a") == 10                      # a quarter of the job's 40 slots
    for _ in range(2):
        leases.grant("a", free)
    clock.now += 101
    assert leases.cap("a") == 2 and leases.cap("b") == 10
    clock.now += 101                                   # a lease time later, forgotten
    assert leases.cap("a") == 10


def test_closed_prompts_and_exhausted_slots_are_never_leased():
    _, ledger, taken, _, leases, free = _setup(prompts=2, slots=1)
    taken.add((10, 0))
    ledger.consume(10)
    for attempt in range(3):
        ledger.record_failure(11, f"{attempt:012d}", slot=0, hotkey="x")
    assert leases.grant("a", free) is None


def test_a_lease_lasts_for_the_jobs_longest_sample():
    assert lease_seconds(_job(max_new_tokens=32768)) == 300 + 3276.8
    assert lease_seconds(_job(max_new_tokens=1000, n=4)) == 300 + 400
    assert lease_seconds(_job(episode=SimpleNamespace(max_total_tokens=65536))) == 300 + 6553.6
    assert lease_seconds(_job(max_new_tokens=10**6)) == 4 * 3600


# ------------------------------------------------------------------ through the job's route


def _route(fake_r2, *, prompts=2, slots=1):
    import asyncio

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.validator.corpus_service import build_corpus_router
    from tests.unit.test_corpus_service import (
        _CountingStore, _Environment, _manifest, _Renderer, _SeededJob, _Tokenizer,
    )

    raw = {**_manifest(), "prompt_count": prompts, "slots_per_prompt": slots, "seed": 5,
           "sampling": {**_manifest()["sampling"], "draw": "gumbel_max"}}
    asyncio.run(job_store.write_job(raw, None, **fake_r2))
    seeded = _SeededJob(_CountingStore(fake_r2), _Renderer(), _Environment(), raw)
    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=seeded.store, tokenizer=_Tokenizer(), renderer=seeded.renderer,
        verify_signature=lambda request: True, verify_lease_signature=lambda request: request.signature != "bad",
        prompt_job_for=seeded.prompt_job_for))
    return TestClient(app)


def _lease(client, hotkey, *, nonce=None, issued_at=None, signature="ok"):
    import secrets
    import time

    return client.post("/corpus/lease", json={
        "job_id": "swe-v1", "miner_hotkey": hotkey, "issued_at": int(time.time() if issued_at is None else issued_at),
        "nonce": nonce or secrets.token_hex(16), "signature": signature}).json()


def _submit_as(client, hotkey, prompt_index, cursor):
    from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
    from tests.unit.test_corpus_service import CHECKPOINT, EOS, _faithful_prompt, _text_for

    tokens = [7] * 16 + [EOS]
    request = CorpusSubmissionRequest(
        job_id="swe-v1", miner_hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT, rendered_prompt=_faithful_prompt(prompt_index),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok")
    return client.post("/corpus/submit", json=request.model_dump()).json()


def test_a_leased_slot_is_its_holders_to_submit(fake_r2):
    client = _route(fake_r2)
    lease = _lease(client, "5A")
    assert lease["leased"] is True, lease
    prompt, slot = lease["prompt_index"], lease["slot"]
    refused = _submit_as(client, "5B", prompt, slot)
    assert (refused["accepted"], refused["reason"]) == (False, "draw_slot_leased")
    assert _submit_as(client, "5A", prompt, slot)["accepted"] is True
    other = _lease(client, "5B")
    assert other["leased"] is True and (other["prompt_index"], other["slot"]) != (prompt, slot)
    # Every slot taken or leased: nothing left to lease.
    assert _lease(client, "5C")["reason"] == "no_free_slot"


def test_a_lease_request_is_signed_fresh_and_single_use(fake_r2):
    import time

    client = _route(fake_r2)
    assert _lease(client, "5A", signature="bad")["reason"] == "bad_signature"
    assert _lease(client, "5A", issued_at=time.time() - 3600)["reason"] == "stale_request"
    first = _lease(client, "5A", nonce="ab" * 16)
    assert first["leased"] is True
    assert _lease(client, "5A", nonce="ab" * 16)["reason"] == "replayed_request"


def test_a_release_frees_the_slot_for_another_miner(fake_r2):
    client = _route(fake_r2, prompts=1)
    lease = _lease(client, "5A")
    assert _lease(client, "5B")["reason"] == "no_free_slot"
    body = {"job_id": "swe-v1", "lease_id": lease["lease_id"], "signature": "ok"}
    assert client.post("/corpus/lease/release", json={**body, "miner_hotkey": "5B"}).json()["reason"] == "lease_unknown"
    assert client.post("/corpus/lease/release", json={**body, "miner_hotkey": "5A"}).json()["reason"] == "released"
    assert _lease(client, "5B")["leased"] is True


def test_an_older_miner_without_a_lease_submits_any_unleased_slot(fake_r2):
    client = _route(fake_r2)
    assert _submit_as(client, "5Old", 0, 0)["accepted"] is True


# ------------------------------------------------------------------ the single-turn miner


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, ids, **kw):
        return "".join(chr(97 + i % 26) for i in ids)


class _Gen:
    def __init__(self, fail=()):
        self.calls, self._fail = [], set(fail)

    def generate_drawn(self, prompt_ids, draws):
        from reliquary.miner.corpus_miner import Generation

        self.calls.append(draws[0]["rollout_index"])
        if len(self.calls) in self._fail:
            raise ValueError("preempted")
        return [Generation([5, 6, 151645], []) for _ in draws]


class _LeasingClient:
    def __init__(self, leases, *, has_route=True):
        self.leases, self.has_route = list(leases), has_route
        self.submitted, self.released, self.lease_bodies = [], [], []

    def lease(self, body):
        if not self.has_route:
            return None
        self.lease_bodies.append(body)
        return self.leases.pop(0) if self.leases else {"leased": False, "reason": "job_complete"}

    def release_lease(self, body):
        self.released.append(body["lease_id"])
        return {"leased": False, "reason": "released"}

    def submit(self, body):
        self.submitted.append((body["prompt_index"], body["cursor"]))
        return {"accepted": True, "reason": "accepted"}

    def open_prompts(self):
        return None


def _granted(prompt, slot, lease_id):
    return {"leased": True, "reason": "accepted", "prompt_index": prompt, "slot": slot, "lease_id": lease_id,
            "expires_at": 1e12}


def _mine(client, generator, **kw):
    from reliquary.miner.corpus_miner import mine_steps
    from tests.unit.test_eval_draw import _job

    return mine_steps(job=_job(), hotkey="5A", client=client, generator=generator, tokenizer=_Tok(),
                      render=lambda i: f"prompt {i}", sign=lambda body: "sig", sleep=lambda s: None, **kw)


def test_the_miner_decodes_the_slot_it_leased():
    client = _LeasingClient([_granted(3, 1, "a" * 32), _granted(7, 0, "b" * 32)])
    counts = _mine(client, _Gen(), sign_lease=lambda body: "lsig")
    assert client.submitted == [(3, 1), (7, 0)]
    assert counts["job_complete"] == 1
    body = client.lease_bodies[0]
    assert body["miner_hotkey"] == "5A" and body["signature"] == "lsig" and len(body["nonce"]) == 32


def test_a_failed_generation_releases_its_lease():
    client = _LeasingClient([_granted(3, 1, "a" * 32), _granted(4, 0, "b" * 32)])
    _mine(client, _Gen(fail=(1,)), sign_lease=lambda body: "lsig")
    assert client.released == ["a" * 32] and client.submitted == [(4, 0)]


def test_a_miner_waits_while_every_slot_is_leased():
    client = _LeasingClient([{"leased": False, "reason": "no_free_slot"}, _granted(2, 0, "c" * 32)])
    slept = []
    from reliquary.miner.corpus_miner import mine_steps
    from tests.unit.test_eval_draw import _job

    counts = mine_steps(job=_job(), hotkey="5A", client=client, generator=_Gen(), tokenizer=_Tok(),
                        render=lambda i: f"prompt {i}", sign=lambda body: "sig", sleep=slept.append,
                        sign_lease=lambda body: "lsig")
    assert counts["lease_no_free_slot"] == 1 and slept and client.submitted == [(2, 0)]


def test_a_validator_without_leases_is_mined_from_the_open_map():
    client = _LeasingClient([], has_route=False)
    counts = _mine(client, _Gen(), sign_lease=lambda body: "lsig", max_steps=1)
    assert client.lease_bodies == [] and counts.get("job_complete", 0) == 0 and client.submitted


def test_a_refused_submission_releases_its_lease():
    class Refusing(_LeasingClient):
        def submit(self, body):
            self.submitted.append((body["prompt_index"], body["cursor"]))
            return {"accepted": False, "reason": "bad_termination"}

    client = Refusing([_granted(3, 1, "a" * 32)])
    _mine(client, _Gen(), sign_lease=lambda body: "lsig")
    assert client.released == ["a" * 32]


def _episode(wall_s=7200, sandbox=True, max_total_tokens=60000):
    budgets = SimpleNamespace(wall_s=wall_s)
    return SimpleNamespace(max_total_tokens=max_total_tokens,
                           sandbox=SimpleNamespace(budgets=budgets) if sandbox else None)


def test_a_signed_episode_lease_outlasts_the_miners_episode_deadline():
    from reliquary.corpus.draw_leases import EPISODE_LEASE_MARGIN_SECONDS, episode_lease_floor
    from reliquary.miner.signed_episode import SignedSweEpisodeRunner

    job = _job(episode=_episode())
    # The miner's own bound on one episode, as its runner computes it.
    runner = SignedSweEpisodeRunner.__new__(SignedSweEpisodeRunner)
    runner._job = job
    deadline = runner.deadline(0)
    assert deadline == 900 + 7200 + 1800 + 120
    assert episode_lease_floor(job) == deadline + EPISODE_LEASE_MARGIN_SECONDS
    assert lease_seconds(job) >= deadline
    ledger = SlotLedger(prompt_start=10, prompt_count=3, slots_per_prompt=2)
    clock = Clock()
    # Even a job or an auction asking for less: the lease still covers the episode.
    leases = DrawLeases(job, clock=clock, seconds=100.0, limit_per_hotkey=False)
    free = free_draw_slots(job, ledger, lambda p, s: False)
    for asked in (None, 1.0, 520.0):
        lease = leases.grant("a", free, seconds=asked)
        assert lease.expires_at - lease.granted_at >= deadline


def test_single_turn_and_unsigned_episode_leases_are_unchanged():
    from reliquary.corpus.draw_leases import episode_lease_floor

    assert lease_seconds(_job(max_new_tokens=1000)) == 300 + 100
    unsigned = _job(episode=_episode(sandbox=False))
    assert episode_lease_floor(unsigned) == 0 and lease_seconds(unsigned) == 300 + 6000
    _, _, _, _, leases, free = _setup()
    assert leases.seconds == 100.0
    lease = leases.grant("a", free, seconds=40.0)
    assert lease.expires_at - lease.granted_at == 40.0


def test_each_lease_event_is_one_log_line(caplog):
    import logging

    _, _, _, clock, leases, free = _setup(prompts=1, slots=2)
    with caplog.at_level(logging.INFO, logger="reliquary.corpus.draw_leases"):
        a = leases.grant("5Aaaaaaaaaaaa", free)
        b = leases.grant("5Bbbbbbbbbbbb", free)
        leases.settle(a.prompt_index, a.slot)
        leases.release("5Bbbbbbbbbbbb", b.lease_id)
        c = leases.grant("5Cccccccccccc", free)
        clock.now += 101
        for _ in range(3):
            len(leases)                       # polls log nothing
    events = [r.getMessage().split()[2] for r in caplog.records]
    assert events == ["granted", "granted", "taken", "released", "granted", "expired"]
    assert f"prompt={c.prompt_index} slot={c.slot} hotkey=5Ccccccc secs=100" in caplog.records[4].getMessage()


def test_a_restart_keeps_the_leases_still_running(tmp_path):
    # 2026-11-14: a control restart dropped every lease in flight, and a job priced by auction refuses a submission
    # without its lease. Written beside the auction's state, they come back; one taken before the restart does not.
    path = tmp_path / "job.leases.json"
    job, _, _, clock, leases, free = _setup(prompts=2, slots=1, limit_per_hotkey=False, state_path=path)
    kept = leases.grant("a", free)
    taken = leases.grant("b", free)
    leases.settle(taken.prompt_index, taken.slot)
    back = DrawLeases(job, clock=clock, rng=random.Random(1), seconds=100.0, limit_per_hotkey=False, state_path=path)
    assert back.held("a", kept.prompt_index, kept.slot) == kept
    assert back.holders(taken.prompt_index, taken.slot) == []
    assert back.counts() == {"a": 1}


def test_a_lease_that_ran_out_while_down_is_dropped_uncharged(tmp_path):
    path = tmp_path / "job.leases.json"
    job, _, _, clock, leases, free = _setup(prompts=1, slots=1, state_path=path)
    leases.grant("a", free)
    clock.now += 101
    charged = []
    back = DrawLeases(job, clock=clock, rng=random.Random(1), seconds=100.0, state_path=path,
                      on_unused=lambda lease, held: charged.append(lease.hotkey))
    assert len(back) == 0 and back.counts() == {} and charged == []


def test_an_unreadable_lease_state_starts_empty(tmp_path):
    path = tmp_path / "job.leases.json"
    path.write_text("{not json")
    job = _job()
    assert len(DrawLeases(job, clock=Clock(), rng=random.Random(0), seconds=100.0, state_path=path)) == 0
