"""The RL lease book (design rl-async-design.md §4.2, P2): exclusive leases per (env, prompt, checkpoint k),
random resolution of the conflicts collected in one interval, refusals that say only taken / cooling / quota /
stale, a nonce H(S_k, env, prompt, k) committed when k is adopted and revealed when k retires, fixed expiry per
env, restart safety (SQLite, like the service runtime), and the prompt lifecycle of §4.5 (verdict, blacklist,
cooldown, first scan, the miners' incremental view)."""
from __future__ import annotations

import asyncio
import hashlib
import random

import pytest

from reliquary.constants import prompt_cooldown_windows_for_environment
from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController
from reliquary.validator.rl_leases import (
    BLACKLISTED, COOLING, FULL, IN_ZONE, QUOTA, RESTART_GRACE_S, SECRET_REVEAL_LAG, STALE, TAKEN, TOO_EASY, TOO_HARD,
    TRUNCATED, LeaseBook, LeaseBookConfig, LeaseCollector, LeaseRequest, classify_group, lease_expiry_s, lease_nonce,
    secret_commitment, verify_reveal,
)

ENV, OTHER_ENV = "math", "code"
NOW = 1_800_000_000.0


class Clock:
    def __init__(self, t: float = NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def book_at(tmp_path, *, clock=None, rng=None, quota=None, name="leases.sqlite3", **config):
    clock = clock or Clock()
    cfg = LeaseBookConfig(expiry_s={ENV: 600.0, OTHER_ENV: 1200.0}, **config)
    quota = quota or QuotaController(QuotaConfig(initial=4, ceiling=4096))
    return LeaseBook(tmp_path / name, cfg, quota=quota, clock=clock, rng=rng or random.Random(7))


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
    # another env is another key; another checkpoint is not while a lease of the prompt is open
    book.adopt_checkpoint(1)
    later, elsewhere = book.grant([ask("B", [2], k=1), ask("B", [2], env=OTHER_ENV)])
    assert later.refused == {2: TAKEN} and [x.prompt_idx for x in elsewhere.leases] == [2]


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


def test_a_checkpoint_whose_secret_is_revealed_is_never_leased(tmp_path):
    book = book_at(tmp_path)
    assert book.config.max_lag == SECRET_REVEAL_LAG == 16
    for k in range(18):
        book.adopt_checkpoint(k)
    assert [c["k"] for c in book.commitments()["revealed"]] == [0]       # 0 < 17 - 16
    assert book.grant([ask("A", [1], k=0)])[0].refused == {1: STALE}
    assert book.grant([ask("A", [1], k=1)])[0].leases                    # lag 16, secret still unrevealed


def test_a_hotkey_never_holds_more_open_leases_than_its_quota(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    (answer,) = book.grant([ask("A", range(10))])
    assert len(answer.leases) == 4 and set(answer.refused.values()) == {QUOTA} and len(answer.refused) == 6
    assert book.in_use("A") == 4
    assert book.grant([ask("A", [20])])[0].refused == {20: QUOTA}


def test_the_global_cap_counts_buffered_groups_only_and_names_the_full_envs(tmp_path):
    book = book_at(tmp_path, max_pending_groups=8)
    book.adopt_checkpoint(0)
    assert len(book.grant([ask("A", [1, 2])])[0].leases) == 2         # open leases do not count against it
    with book.transaction():                                             # groups waiting for the trainer do
        book.db.execute("CREATE TABLE rl_buffer(lease_id TEXT PRIMARY KEY)")
        book.db.executemany("INSERT INTO rl_buffer VALUES(?)", [(f"g{i}",) for i in range(7)])
    assert len(book.grant([ask("B", [3])])[0].leases) == 1              # 7 buffered < 8
    with book.transaction():
        book.db.execute("INSERT INTO rl_buffer VALUES('g7')")
    answers = book.grant([ask("C", [4]), ask("D", [5], env=OTHER_ENV)])
    assert [a.refused for a in answers] == [{4: FULL}, {5: FULL}]
    assert book.drain_full() == {ENV, OTHER_ENV} and book.drain_full() == set()
    with pytest.raises(ValueError):
        LeaseBookConfig(expiry_s={ENV: 1.0}, max_pending_groups=0)


def test_one_hotkey_holds_at_most_its_share_of_the_global_cap(tmp_path):
    book = book_at(tmp_path, max_pending_groups=8)                      # share 1/4: 2 open leases
    book.adopt_checkpoint(0)
    answer = book.grant([ask("A", [1, 2, 3])])[0]
    assert len(answer.leases) == 2 and answer.refused == {3: QUOTA}
    assert len(book.grant([ask("B", [4, 5])])[0].leases) == 2


def test_a_request_is_cut_to_the_hotkeys_room_before_any_lookup(tmp_path):
    book = book_at(tmp_path)                                             # quota 4
    book.adopt_checkpoint(0)
    first = book.grant([ask("A", [1, 2, 3])])[0]
    statements = []
    book.db.set_trace_callback(statements.append)
    answer = book.grant([ask("A", list(range(10, 1010)))])[0]
    book.db.set_trace_callback(None)
    assert [lease.prompt_idx for lease in answer.leases] == [10]         # room 1: the first asked, in order
    assert set(answer.refused.values()) == {QUOTA} and len(answer.refused) == 999
    assert sum("rl_prompt_state" in sql for sql in statements) == 1     # one claim looked up, not 1000
    statements.clear()
    book.db.set_trace_callback(statements.append)
    again = book.grant([ask("A", [1, 2, 3, 20])])[0]                     # no room: resends answered, 20 refused
    book.db.set_trace_callback(None)
    assert [lease.lease_id for lease in again.leases] == [lease.lease_id for lease in first.leases]
    assert again.refused == {20: QUOTA} and not any("rl_prompt_state" in sql for sql in statements)


def test_a_prompt_is_leased_at_one_checkpoint_at_a_time(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    book.adopt_checkpoint(1)
    held = book.grant([ask("A", [7], k=0)])[0].leases[0]
    assert book.grant([ask("B", [7], k=1)])[0].refused == {7: TAKEN}
    assert book.grant([ask("A", [7], k=1)])[0].refused == {7: TAKEN}   # its own holder too: one draw at a time
    book.submit(held.lease_id, "A", in_zone=False)                       # unscanned in this test: lifecycle free
    assert len(book.grant([ask("B", [7], k=1)])[0].leases) == 1


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


# --- prompt lifecycle (§4.5) ------------------------------------------------------------------------------------

def binary(correct, n=16):
    return [1.0] * correct + [0.0] * (n - correct)


def test_the_verdict_follows_sigma_mean_and_truncation():
    clean = [False] * 16
    assert classify_group(binary(1), clean) == IN_ZONE                   # sigma 0.242
    assert classify_group(binary(15), clean) == IN_ZONE
    assert classify_group(binary(16), clean) == TOO_EASY
    assert classify_group(binary(0), clean) == TOO_HARD
    one, two = [True] + [False] * 15, [True, True] + [False] * 14
    assert classify_group(binary(8), one) == IN_ZONE                     # one truncated rollout is allowed
    assert classify_group(binary(8), two) == TRUNCATED
    assert classify_group(binary(16), two) == TRUNCATED                  # never blacklisted on a truncated group
    # continuous rewards with too little variance: mean >= 0.5 is too easy, below too hard
    assert classify_group([0.5, 0.52] * 8, clean) == TOO_EASY
    assert classify_group([0.3, 0.32] * 8, clean) == TOO_HARD
    assert classify_group([0.0, 0.6] * 8, clean) == IN_ZONE
    for bad in (([], []), (binary(8), [False] * 15), ([float("nan")] * 16, clean)):
        with pytest.raises(ValueError):
            classify_group(*bad)


def test_too_easy_blacklists_the_prompt_for_the_run(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, binary(16), TOO_EASY)
    for k in (1, 500):
        book.adopt_checkpoint(k)
        assert book.grant([ask("A", [1], k=k)])[0].refused == {1: BLACKLISTED}
    book.record_outcome(ENV, 1, 0, binary(8), IN_ZONE)                    # a late in-zone group lifts nothing
    assert book.grant([ask("A", [1], k=500)])[0].refused == {1: BLACKLISTED}
    assert book.grant([ask("A", [1], k=500, env=OTHER_ENV)])[0].leases   # the same index elsewhere is another prompt


@pytest.mark.parametrize("verdict", [IN_ZONE, TOO_HARD, TRUNCATED])
def test_in_zone_too_hard_and_truncated_cool_down_then_lease_again(tmp_path, verdict):
    book = book_at(tmp_path, cooldown_of={ENV: 3}.__getitem__)
    book.adopt_checkpoint(10)
    book.record_outcome(ENV, 1, 9, binary(8), verdict)                   # leased at 9: counts from the latest, 10
    (answer,) = book.grant([ask("A", [1, 2], k=10)])
    assert answer.refused == {1: COOLING} and [x.prompt_idx for x in answer.leases] == [2]
    book.adopt_checkpoint(12)
    assert book.grant([ask("A", [1], k=12)])[0].refused == {1: COOLING}
    book.adopt_checkpoint(13)
    assert book.grant([ask("A", [1], k=13)])[0].leases                   # 13 = 10 + 3
    assert book.prompt_state()["prompts"] == [{"environment": ENV, "prompt_idx": 1, "state": "cooldown",
                                               "until_k": 13}]


def test_a_later_outcome_never_shortens_a_cooldown(tmp_path):
    book = book_at(tmp_path, cooldown_of={ENV: 10}.__getitem__)
    book.adopt_checkpoint(20)
    book.record_outcome(ENV, 1, 20, binary(8), IN_ZONE)
    book.record_outcome(ENV, 1, 5, binary(0), TOO_HARD)
    assert book.prompt_state()["prompts"][0]["until_k"] == 30


def test_the_cooldown_is_the_profiles_per_env_value_read_at_each_outcome(tmp_path):
    assert LeaseBookConfig(expiry_s={ENV: 1.0}).cooldown_of is prompt_cooldown_windows_for_environment
    live = {ENV: 5}
    book = book_at(tmp_path, cooldown_of=live.__getitem__)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, binary(8), IN_ZONE)
    live[ENV] = 2                                                         # an operator changes it live
    book.record_outcome(ENV, 2, 0, binary(8), IN_ZONE)
    assert [p["until_k"] for p in book.prompt_state()["prompts"]] == [5, 2]   # applies to the next cooldown only
    live[ENV] = -1
    with pytest.raises(ValueError):
        book.record_outcome(ENV, 3, 0, binary(8), IN_ZONE)


def test_only_the_first_scan_of_a_prompt_in_the_run_is_flagged(tmp_path):
    book = book_at(tmp_path, cooldown_of=lambda env: 1)
    book.adopt_checkpoint(0)
    first = book.record_outcome(ENV, 1, 0, binary(0), TOO_HARD)
    again = book.record_outcome(ENV, 1, 2, binary(0), TOO_HARD)
    other = book.record_outcome(OTHER_ENV, 1, 0, binary(4), IN_ZONE)
    assert (first.first_scan, again.first_scan, other.first_scan) == (True, False, True)
    assert first.rewards == tuple(binary(0)) and first.verdict == TOO_HARD and first.seq < again.seq


def test_an_outcome_is_validated(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    for bad in ((ENV, 1, 0, binary(8), "maybe"), ("unknown", 1, 0, binary(8), IN_ZONE), (ENV, -1, 0, binary(8),
                IN_ZONE), (ENV, 1, 0, [], IN_ZONE), (ENV, 1, 0, [float("inf")], IN_ZONE)):
        with pytest.raises(ValueError):
            book.record_outcome(*bad)
    assert book.prompt_state()["scans"] == []


def test_the_prompt_view_is_incremental_and_names_no_hotkey(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    assert book.prompt_state() == {"latest": 0, "cursor": 0, "more": False, "scans": [], "prompts": []}
    lease = book.grant([ask("hotkey-of-the-miner", [1, 2])])[0].leases[0]
    book.submit(lease.lease_id, "hotkey-of-the-miner", in_zone=False)
    book.record_outcome(ENV, 1, 0, binary(16), TOO_EASY)
    book.record_outcome(ENV, 2, 0, binary(8), IN_ZONE)
    book.record_outcome(OTHER_ENV, 3, 0, binary(0), TOO_HARD)
    full = book.prompt_state()
    assert [s["prompt_idx"] for s in full["scans"]] == [1, 2, 3] and full["cursor"] == full["scans"][-1]["seq"]
    assert [(p["prompt_idx"], p["state"], p["until_k"]) for p in full["prompts"]] == [
        (1, "blacklisted", None), (2, "cooldown", prompt_cooldown_windows_for_environment(ENV)),
        (3, "cooldown", prompt_cooldown_windows_for_environment(OTHER_ENV))]
    assert full["scans"][0]["rewards"] == binary(16) and full["scans"][0]["first_scan"] is True
    assert "hotkey-of-the-miner" not in repr(full) and "nonce" not in repr(full)
    # paging: a cursor returns only what moved after it, and a prompt once per page with its current state
    page = book.prompt_state(limit=2)
    assert page["more"] and [s["prompt_idx"] for s in page["scans"]] == [1, 2]
    book.record_outcome(ENV, 2, 0, binary(16), TOO_EASY)
    rest = book.prompt_state(page["cursor"])
    assert not rest["more"] and [s["prompt_idx"] for s in rest["scans"]] == [3, 2]
    assert [(p["prompt_idx"], p["state"]) for p in rest["prompts"]] == [(3, "cooldown"), (2, "blacklisted")]
    assert book.prompt_state(rest["cursor"])["scans"] == []
    only = book.prompt_state(environment=OTHER_ENV)
    assert [s["prompt_idx"] for s in only["scans"]] == [3] and len(only["prompts"]) == 1


def test_the_prompt_lifecycle_survives_a_restart(tmp_path):
    book = book_at(tmp_path)
    book.adopt_checkpoint(0)
    book.record_outcome(ENV, 1, 0, binary(16), TOO_EASY)
    view = book.prompt_state()
    book.close()
    reopened = book_at(tmp_path)
    assert reopened.prompt_state() == view
    assert reopened.grant([ask("A", [1])])[0].refused == {1: BLACKLISTED}
    assert reopened.record_outcome(ENV, 1, 0, binary(16), TOO_EASY).first_scan is False
