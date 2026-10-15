"""The RL lease book (design rl-async-design.md §4.2, P2): exclusive leases per (env, prompt, checkpoint k),
random resolution of the conflicts collected in one interval, refusals that say only taken / cooling / quota /
stale, a nonce H(S_k, env, prompt, k) committed when k is adopted and revealed when k retires, fixed expiry per
env, and restart safety (SQLite, like the service runtime)."""
from __future__ import annotations

import asyncio
import hashlib
import random

import pytest

from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import (
    COOLING, QUOTA, RESTART_GRACE_S, STALE, TAKEN, LeaseBook, LeaseBookConfig, LeaseCollector, LeaseRequest, checkpoint_cooldown,
    lease_expiry_s, lease_nonce, secret_commitment, verify_reveal,
)

ENV, OTHER_ENV = "math", "code"
NOW = 1_800_000_000.0


class Clock:
    def __init__(self, t: float = NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def book_at(tmp_path, *, clock=None, cooling=None, rng=None, quota=None, name="leases.sqlite3", **config):
    clock = clock or Clock()
    cfg = LeaseBookConfig(expiry_s={ENV: 600.0, OTHER_ENV: 1200.0}, **config)
    quota = quota or QuotaController(QuotaConfig(initial=4, ceiling=4096))
    return LeaseBook(tmp_path / name, cfg, quota=quota, clock=clock, cooling=cooling,
                     rng=rng or random.Random(7))


def ask(hotkey, prompts, k=0, env=ENV):
    return LeaseRequest(hotkey=hotkey, environment=env, prompt_ids=tuple(prompts), checkpoint=k)


# --- nonce: deterministic, domain separated, committed then revealed ------------------------------------------

def test_the_nonce_is_a_pure_function_of_the_secret_env_prompt_and_checkpoint():
    secret = bytes(range(32))
    a = lease_nonce(secret, ENV, 5, 3)
    assert a == lease_nonce(secret, ENV, 5, 3) and len(a) == 32
    assert len({a, lease_nonce(secret, ENV, 6, 3), lease_nonce(secret, ENV, 5, 4),
                lease_nonce(secret, OTHER_ENV, 5, 3), lease_nonce(bytes(32), ENV, 5, 3)}) == 5


def test_the_nonce_and_the_commitment_are_domain_separated():
    secret = bytes(range(32))
    assert secret_commitment(secret, 3) != hashlib.sha256(secret).digest()
    assert lease_nonce(secret, ENV, 5, 3) != secret_commitment(secret, 3)
    # length-prefixed fields: ("ab", 1) and ("a", ...) never collide by concatenation
    assert lease_nonce(secret, "ab", 1, 0) != lease_nonce(secret, "a", 1, 0)


def test_a_revealed_secret_verifies_only_against_its_own_commitment():
    secret = bytes(range(32))
    commitment = secret_commitment(secret, 9)
    assert verify_reveal(commitment, 9, secret)
    assert not verify_reveal(commitment, 10, secret)
    assert not verify_reveal(commitment, 9, bytes(32))


def test_every_lease_of_one_prompt_and_checkpoint_gets_the_same_nonce(tmp_path):
    clock = Clock()
    book = book_at(tmp_path, clock=clock)
    book.adopt_checkpoint(0)
    first = book.grant([ask("A", [1])])[0]
    lease = first.leases[0]
    clock.t += 601                                           # expires: the prompt is free again at k
    book.expire_due()
    again = book.grant([ask("B", [1])])[0].leases[0]
    assert again.lease_id != lease.lease_id and again.nonce == lease.nonce


def test_the_commitment_is_published_at_adoption_and_the_secret_when_k_retires(tmp_path):
    book = book_at(tmp_path, max_lag=2)
    book.adopt_checkpoint(0)
    lease = book.grant([ask("A", [3], k=0)])[0].leases[0]
    view = book.commitments()
    assert [c["k"] for c in view["active"]] == [0] and view["revealed"] == []
    assert "secret" not in view["active"][0]
    for k in (1, 2):
        book.adopt_checkpoint(k)
    assert book.commitments()["revealed"] == []              # 0 is still within lag 2 of 2
    book.adopt_checkpoint(3)
    view = book.commitments()
    assert [c["k"] for c in view["active"]] == [1, 2, 3]
    (revealed,) = view["revealed"]
    secret = bytes.fromhex(revealed["secret"])
    assert revealed["k"] == 0 and verify_reveal(bytes.fromhex(revealed["commitment"]), 0, secret)
    assert lease_nonce(secret, ENV, 3, 0).hex() == lease.nonce   # anyone can now audit the nonce


def test_checkpoints_are_adopted_in_order(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(5)
    book.adopt_checkpoint(5)                                  # idempotent
    with pytest.raises(ValueError):
        book.adopt_checkpoint(4)


# --- exclusivity ----------------------------------------------------------------------------------------------

def test_one_open_lease_per_prompt_and_checkpoint(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    (a,) = book.grant([ask("A", [1, 2])])
    assert sorted(lease.prompt_idx for lease in a.leases) == [1, 2] and a.refused == {}
    (b,) = book.grant([ask("B", [2, 3])])
    assert [lease.prompt_idx for lease in b.leases] == [3] and b.refused == {2: TAKEN}
    # another env, or another checkpoint, is another key
    book.adopt_checkpoint(1)
    later, elsewhere = book.grant([ask("B", [2], k=1), ask("B", [2], env=OTHER_ENV)])
    assert [lease.prompt_idx for lease in later.leases] == [2] == [x.prompt_idx for x in elsewhere.leases]


def test_a_resend_by_the_holder_answers_its_own_lease(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    lease = book.grant([ask("A", [1])])[0].leases[0]
    again = book.grant([ask("A", [1])])[0]
    assert again.leases == [lease] and book.in_use("A") == 1


def test_a_submitted_prompt_is_closed_for_its_checkpoint(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    lease = book.grant([ask("A", [1])])[0].leases[0]
    book.submit(lease.lease_id, "A", in_zone=False)
    assert book.lease(lease.lease_id).state == "submitted"
    assert book.grant([ask("B", [1]), ask("A", [1])])[0].refused == {1: TAKEN}
    assert book.grant([ask("A", [1])])[0].refused == {1: TAKEN}
    book.adopt_checkpoint(1)
    assert [x.prompt_idx for x in book.grant([ask("B", [1], k=1)])[0].leases] == [1]


def test_a_lease_is_submitted_once_by_its_holder_before_expiry(tmp_path):
    clock = Clock()
    book = book_at(tmp_path, clock=clock)
    book.adopt_checkpoint(0)
    first, second = book.grant([ask("A", [1, 2])])[0].leases
    with pytest.raises(PermissionError):
        book.submit(first.lease_id, "B", in_zone=True)
    book.submit(first.lease_id, "A", in_zone=True)
    with pytest.raises(LookupError):
        book.submit(first.lease_id, "A", in_zone=True)
    clock.t = second.expires_at + 1
    with pytest.raises(LookupError):
        book.submit(second.lease_id, "A", in_zone=True)      # late is expired, never delivered
    assert book.lease(second.lease_id).state == "expired"


# --- fairness inside one interval ------------------------------------------------------------------------------

def test_a_conflict_inside_one_batch_goes_to_a_uniformly_random_requester(tmp_path):
    wins = {"A": 0, "B": 0, "C": 0}
    for trial in range(900):
        book = book_at(tmp_path, rng=random.Random(trial), name=f"t{trial}.sqlite3")
        book.adopt_checkpoint(trial)                          # a fresh k each trial: no carry-over
        answers = book.grant([ask(h, [1], k=trial) for h in ("A", "B", "C")])
        (winner,) = [h for h, answer in zip("ABC", answers) if answer.leases]
        wins[winner] += 1
        assert sorted(a.refused.get(1) for a in answers if not a.leases) == [TAKEN, TAKEN]
        book.close()
    assert all(240 < n < 360 for n in wins.values()), wins   # first in the list has no advantage


def test_the_collector_resolves_requests_arriving_within_one_interval_together(tmp_path):
    wins = {"early": 0, "late": 0}

    async def trial(seed: int) -> None:
        book = book_at(tmp_path, rng=random.Random(seed), name=f"c{seed}.sqlite3")
        book.adopt_checkpoint(seed)
        collector = LeaseCollector(book, interval_s=0.05)
        early = asyncio.create_task(collector.request(ask("early", [1], k=seed)))
        await asyncio.sleep(0.02)                               # arrives later, inside the same interval
        late = asyncio.create_task(collector.request(ask("late", [1], k=seed)))
        for name, answer in (("early", await early), ("late", await late)):
            if answer.leases:
                wins[name] += 1
        book.close()

    async def main():
        for seed in range(120):
            await trial(seed)

    asyncio.run(main())
    assert wins["early"] + wins["late"] == 120
    assert 35 < wins["late"] < 85, wins


def test_a_request_after_the_interval_sees_the_earlier_lease(tmp_path):
    async def main():
        book = book_at(tmp_path)
        book.adopt_checkpoint(0)
        collector = LeaseCollector(book, interval_s=0.01)
        first = await collector.request(ask("A", [1]))
        second = await collector.request(ask("B", [1]))
        return first, second

    first, second = asyncio.run(main())
    assert first.leases and second.refused == {1: TAKEN}


# --- refusals: stale, cooling, quota; nothing else leaks ---------------------------------------------------------

def test_leases_on_checkpoints_older_than_the_max_lag_or_unknown_are_stale(tmp_path):
    book = book_at(tmp_path, max_lag=4)
    for k in range(7):
        book.adopt_checkpoint(k)
    assert book.grant([ask("A", [1], k=1)])[0].refused == {1: STALE}     # 6 - 1 > 4
    assert book.grant([ask("A", [1], k=2)])[0].leases                    # lag 4 is allowed
    assert book.grant([ask("A", [2], k=7)])[0].refused == {2: STALE}     # not adopted yet


def test_a_prompt_trained_recently_is_cooling(tmp_path):
    maps = {ENV: CooldownMap(3)}
    book = book_at(tmp_path, cooling=checkpoint_cooldown(maps))
    book.adopt_checkpoint(10)
    maps[ENV].record_batched(1, 9)                                       # trained at checkpoint 9
    (answer,) = book.grant([ask("A", [1, 2], k=10)])
    assert answer.refused == {1: COOLING} and [x.prompt_idx for x in answer.leases] == [2]
    book.adopt_checkpoint(12)
    assert book.grant([ask("A", [1], k=12)])[0].leases                   # 12 - 9 >= 3


def test_a_hotkey_never_holds_more_open_leases_than_its_quota(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    (answer,) = book.grant([ask("A", range(10))])
    assert len(answer.leases) == 4 and set(answer.refused.values()) == {QUOTA} and len(answer.refused) == 6
    assert book.in_use("A") == 4
    assert book.grant([ask("A", [20])])[0].refused == {20: QUOTA}


def test_a_contested_prompt_goes_to_a_requester_with_quota_left(tmp_path):
    for seed in range(40):
        book = book_at(tmp_path, rng=random.Random(seed), name=f"q{seed}.sqlite3")
        book.adopt_checkpoint(seed)
        book.grant([ask("full", [100, 101, 102, 103], k=seed)])
        answers = book.grant([ask("full", [1], k=seed), ask("free", [1], k=seed)])
        # whatever the shuffle, the full hotkey never keeps the free one from it
        assert answers[0].refused[1] in (QUOTA, TAKEN) and answers[1].leases
        book.close()


def test_refusals_never_say_who_holds_a_prompt_or_when_it_expires(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    book.grant([ask("A", [1])])
    (answer,) = book.grant([ask("B", [1])])
    assert answer.to_dict()["refused"] == [{"prompt_idx": 1, "reason": "taken"}]
    assert "A" not in repr(answer.to_dict())


def test_the_request_is_validated(tmp_path):
    book = book_at(tmp_path, prompt_range={ENV: (0, 100)})
    book.adopt_checkpoint(0)
    for bad in (ask("A", []), ask("A", [1, 1]), ask("A", range(1025)), ask("A", [-1]), ask("A", [100]),
                ask("A", [1], env="unknown")):
        with pytest.raises(ValueError):
            book.check_request(bad)


# --- expiry ---------------------------------------------------------------------------------------------------

def test_the_expiry_is_fixed_per_env_from_its_max_length_at_a_reference_speed():
    assert lease_expiry_s(max_tokens=8192, reference_tokens_per_s=20.0, margin=1.5) == pytest.approx(614.4)
    with pytest.raises(ValueError):
        lease_expiry_s(max_tokens=0, reference_tokens_per_s=20.0, margin=1.5)


def test_an_expired_lease_frees_its_prompt_and_halves_the_quota(tmp_path):
    clock = Clock()
    quota = QuotaController(QuotaConfig(initial=8, floor=1))
    book = book_at(tmp_path, clock=clock, quota=quota)
    book.adopt_checkpoint(0)
    (lease,) = book.grant([ask("A", [1], env=ENV)])[0].leases
    (slow,) = book.grant([ask("A", [2], env=OTHER_ENV)])[0].leases
    assert lease.expires_at == NOW + 600 and slow.expires_at == NOW + 1200
    clock.t = NOW + 601
    assert book.expire_due() == 1
    assert book.lease(lease.lease_id).state == "expired" and book.lease(slow.lease_id).state == "open"
    assert quota.state("A")["quota"] == 4 and quota.state("A")["counters"]["expired"] == 1
    assert book.grant([ask("B", [1])])[0].leases                         # C12: the next lessee gets it


# --- restart ---------------------------------------------------------------------------------------------------

def test_open_leases_secrets_and_quota_survive_a_restart(tmp_path):
    clock = Clock()
    book = book_at(tmp_path, clock=clock)
    book.adopt_checkpoint(0)
    book.adopt_checkpoint(1)
    lease = book.grant([ask("A", [1, 2], k=1)])[0].leases[0]
    other = [x for x in book.grant([ask("A", [1, 2], k=1)])[0].leases if x.lease_id != lease.lease_id][0]
    book.submit(other.lease_id, "A", in_zone=True)
    commitments = book.commitments()
    book.close()

    clock.t += 10
    reopened = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0, OTHER_ENV: 1200.0}),
                         quota=QuotaController(QuotaConfig(initial=4)), clock=clock)
    assert reopened.latest_checkpoint == 1
    assert reopened.lease(lease.lease_id) == lease
    assert reopened.commitments() == commitments
    assert reopened.grant([ask("B", [1, 2], k=1)])[0].refused == {1: TAKEN, 2: TAKEN}
    assert reopened.in_use("A") == 1
    assert reopened.quota.state("A")["counters"]["submitted"] == 1
    # the same secret: a new lease of a new prompt still derives from S_1
    assert reopened.grant([ask("B", [3], k=1)])[0].leases[0].nonce != lease.nonce


def test_a_lease_that_fell_due_while_the_validator_was_down_gets_a_grace(tmp_path):
    clock = Clock()
    book = book_at(tmp_path, clock=clock)
    book.adopt_checkpoint(0)
    lease = book.grant([ask("A", [1])])[0].leases[0]
    book.close()
    clock.t = lease.expires_at + 3600                                     # down for an hour past its expiry
    reopened = LeaseBook(tmp_path / "leases.sqlite3", LeaseBookConfig(expiry_s={ENV: 600.0}),
                         quota=QuotaController(QuotaConfig(initial=4)), clock=clock)
    assert reopened.expire_due() == 0
    assert reopened.lease(lease.lease_id).expires_at == clock.t + RESTART_GRACE_S
    reopened.submit(lease.lease_id, "A", in_zone=True)


def test_the_database_file_is_private(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    assert (tmp_path / "leases.sqlite3").stat().st_mode & 0o077 == 0
