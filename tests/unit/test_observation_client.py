"""Miner reference client for the published run observation log, and the default prompt policy."""
import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from reliquary.constants import M_ROLLOUTS
from reliquary.miner import engine as engine_module
from reliquary.miner.engine import MiningEngine
from reliquary.miner.observation_client import (
    DefaultPromptPolicy, ObservationClient, ObservationTable, ObservationVerificationError, load_policy,
)
from reliquary.services import publication
from reliquary.services.publication import ObservationPublisher, index_key, page_key
from tests.unit.service_v2_fixtures import MATH
from tests.unit.test_service_runtime_v2 import HALF, ONES, ZERO, explore, runtime, train

BASE = "https://observations.example/"
RUN = "r1"
POOL = 2 * M_ROLLOUTS


def keypair():
    return Keypair.create_from_seed("0x" + "07" * 32)


class FakeR2:
    """A bucket: the real publisher writes to it, the client reads from it through ``fetch``."""

    def __init__(self):
        self.objects, self.fetched = {}, []

    async def put(self, key, body, content_type, cache_control):
        self.objects[key] = body

    def fetch(self, url):
        assert url.startswith(BASE)
        self.fetched.append(url[len(BASE):])
        try:
            return self.objects[url[len(BASE):]]
        except KeyError:
            raise OSError("404") from None

    def count(self, key):
        return self.fetched.count(key)


def publish_all(rt, bucket, key=None, clock=lambda: 1000.7):
    pub = ObservationPublisher(rt, run_id=RUN, task_id="t", wallet=SimpleNamespace(hotkey=key or keypair()),
                               put=bucket.put, clock=clock)
    while asyncio.run(pub.flush()) is not None:
        pass
    return pub


def small_pages(monkeypatch, page=3):
    monkeypatch.setattr(publication, "PAGE_SEGMENTS", page)
    monkeypatch.setattr(publication, "SERVICE_OBSERVATION_SEGMENT_MAX_EVENTS", 1)  # one event per segment


def client(bucket, tmp_path, *, key=None, run=RUN, clock=None, **kw):
    return ObservationClient(BASE, run, (key or keypair()).ss58_address, fetch=bucket.fetch,
                             directory=tmp_path / "miner", clock=clock or (lambda: 0.0), **kw)


def log_events(rt):
    return [e for _, e in rt.events(limit=10_000)]


def last_settles(events):
    return {e["id"]: e for e in events if e["type"] == "settle"}


# ------------------------------------------------------------------ reconstruction

def test_the_client_reconstructs_exactly_the_events_across_a_page_boundary(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path / "v")
    results = [explore(rt, prompt=10 + i, hotkey=f"h{i}") for i in range(5)]
    train(rt, prompt=30, rewards=HALF)
    with rt._txn():
        rt.log.settle(results[0]["observation_id"], status="exploration_paid", proof="audited", at=150.0)
    bucket = FakeR2()
    publish_all(rt, bucket)
    assert page_key(RUN, 1, 3) in bucket.objects          # a closed page exists: the boundary was crossed
    c = client(bucket, tmp_path)
    segments = c.sync()
    events = log_events(rt)
    observations = [e for e in events if e["type"] == "observation"]
    assert segments == c.table.last_number == len(events) > 3
    settles = last_settles(events)
    for e in observations:
        (record,) = c.table.records(e["env"], e["prompt_idx"])
        assert record["id"] == e["id"] and record["rewards_bps"] == e["rewards_bps"]
        assert record["seeds"] == e["candidate"]["seeds"] and record["pool_sha256"] == e["candidate"]["pool_sha256"]
        assert record["lane"] == e["lane"] and record["verdict"] == e["verdict"] and record["window"] == e["window"]
        final = settles.get(e["id"], e)
        assert (record["status"], record["proof"]) == (final["status"], final["proof"])
    assert sorted(c.table.prompts(MATH)) == sorted(e["prompt_idx"] for e in observations)
    assert c.table.cached_page(1) is not None
    assert c.table.last_seq == len(events)


def test_pages_are_walked_once_and_only_new_segments_are_downloaded(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path / "v")
    for i in range(4):
        explore(rt, prompt=10 + i, hotkey=f"h{i}")
    bucket = FakeR2()
    pub = publish_all(rt, bucket)
    c = client(bucket, tmp_path)
    assert c.sync(force=True) == 4
    first_total = c.table.last_number
    assert bucket.count(page_key(RUN, 1, 3)) == 1
    for i in range(4, 7):
        explore(rt, prompt=10 + i, hotkey=f"h{i}")
    while asyncio.run(pub.flush()) is not None:
        pass
    new = c.sync(force=True)
    assert new == c.table.last_number - first_total > 0
    assert bucket.count(page_key(RUN, 1, 3)) == 1           # page read once
    assert bucket.count(publication.segment_key(RUN, 1)) == 1  # old segments never re-read


def test_last_settle_wins(tmp_path):
    rt = runtime(tmp_path / "v")
    r = explore(rt, prompt=11)
    with rt._txn():
        rt.log.settle(r["observation_id"], status="exploration_paid", proof="audited", at=150.0)
    with rt._txn():
        rt.log.settle(r["observation_id"], status="exploration_forfeited", proof="failed", at=160.0)
    bucket = FakeR2()
    publish_all(rt, bucket)
    c = client(bucket, tmp_path)
    c.sync()
    (record,) = c.table.records(MATH, 11)
    assert (record["status"], record["proof"]) == ("exploration_forfeited", "failed")
    assert c.table.summary(MATH, 11).scanned is False        # a forfeited first scan is released


# ------------------------------------------------------------------ refusals

def published(tmp_path, n=2, monkeypatch=None, key=None):
    rt = runtime(tmp_path / "v")
    for i in range(n):
        explore(rt, prompt=10 + i, hotkey=f"h{i}")
    bucket = FakeR2()
    publish_all(rt, bucket, key)
    return rt, bucket


def test_a_tampered_head_is_refused_and_nothing_is_applied(tmp_path):
    _, bucket = published(tmp_path)
    document = json.loads(bucket.objects[index_key(RUN)])
    document["segments"][0]["sha256"] = "0" * 64
    bucket.objects[index_key(RUN)] = json.dumps(document).encode()
    c = client(bucket, tmp_path)
    with pytest.raises(ObservationVerificationError, match="signature"):
        c.sync()
    assert c.table.last_number == 0 and c.table.prompts(MATH) == []


def test_a_head_signed_by_another_key_or_for_another_run_is_refused(tmp_path):
    _, bucket = published(tmp_path)
    other = Keypair.create_from_seed("0x" + "08" * 32)
    with pytest.raises(ObservationVerificationError, match="another validator"):
        client(bucket, tmp_path / "a", key=other).sync()
    c = ObservationClient(BASE, "r2", keypair().ss58_address, fetch=lambda u: bucket.objects[index_key(RUN)],
                          directory=tmp_path / "b")
    with pytest.raises(ObservationVerificationError, match="another run"):
        c.sync()


def test_a_tampered_segment_is_refused(tmp_path):
    _, bucket = published(tmp_path)
    key = publication.segment_key(RUN, 1)
    bucket.objects[key] = bucket.objects[key][:-1] + bytes([bucket.objects[key][-1] ^ 1])
    c = client(bucket, tmp_path)
    with pytest.raises(ObservationVerificationError, match="segment"):
        c.sync()
    assert c.table.last_number == 0


def test_a_tampered_page_is_refused(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    _, bucket = published(tmp_path, n=4)
    key = page_key(RUN, 1, 3)
    bucket.objects[key] = bucket.objects[key] + b" "
    c = client(bucket, tmp_path)
    with pytest.raises(ObservationVerificationError, match="page"):
        c.sync()
    assert c.table.last_number == 0 and c.table.cached_page(1) is None


def test_a_page_signed_by_someone_else_is_refused_even_with_a_matching_head_digest(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt, bucket = published(tmp_path, n=4)
    key = page_key(RUN, 1, 3)
    forged = json.loads(bucket.objects[key])
    forged["segments"][0]["sha256"] = "1" * 64
    bucket.objects[key] = json.dumps(forged).encode()
    head = json.loads(bucket.objects[index_key(RUN)])
    import hashlib
    head["pages"][0]["sha256"] = hashlib.sha256(bucket.objects[key]).hexdigest()
    head["pages"][0]["size"] = len(bucket.objects[key])
    bucket.objects[index_key(RUN)] = json.dumps(head).encode()   # the head signature no longer holds either
    with pytest.raises(ObservationVerificationError):
        client(bucket, tmp_path).sync()


def test_a_stale_head_is_refused_and_the_table_never_goes_backwards(tmp_path):
    rt = runtime(tmp_path / "v")
    explore(rt, prompt=11)
    bucket = FakeR2()
    pub = publish_all(rt, bucket)
    old_head = bucket.objects[index_key(RUN)]
    explore(rt, prompt=12, hotkey="b")
    while asyncio.run(pub.flush()) is not None:
        pass
    c = client(bucket, tmp_path)
    c.sync()
    reached = c.table.last_number
    assert reached > 1
    bucket.objects[index_key(RUN)] = old_head
    with pytest.raises(ObservationVerificationError, match="stale"):
        c.sync(force=True)
    assert c.table.last_number == reached and c.table.prompts(MATH) == [11, 12]


def test_a_head_of_another_order_than_the_one_pinned_is_refused(tmp_path):
    rt, bucket = published(tmp_path)
    c = client(bucket, tmp_path)
    assert c.sync() > 0 and c.table.get_meta("order_sha256") == rt.contract.sha256
    c.table.set_meta("order_sha256", "f" * 64)
    with pytest.raises(ObservationVerificationError, match="order"):
        c.sync(force=True)


def test_segments_that_do_not_continue_the_log_are_refused(tmp_path):
    _, bucket = published(tmp_path, n=2)
    c = client(bucket, tmp_path)
    c.table.apply_segment(0, 5, [])            # pretend the log was already read to another place
    with pytest.raises(ObservationVerificationError, match="sequence"):
        c.sync()


# ------------------------------------------------------------------ restart, polling

def test_restart_resumes_without_double_counting(tmp_path, monkeypatch):
    small_pages(monkeypatch)
    rt = runtime(tmp_path / "v")
    for i in range(3):
        explore(rt, prompt=10 + i, hotkey=f"h{i}")
    bucket = FakeR2()
    pub = publish_all(rt, bucket)
    c = client(bucket, tmp_path)
    c.sync()
    done = c.table.last_number
    c.table.close()
    for i in range(3, 6):
        explore(rt, prompt=10 + i, hotkey=f"h{i}")
    while asyncio.run(pub.flush()) is not None:
        pass
    bucket.fetched.clear()
    c2 = client(bucket, tmp_path)                       # same directory: a restarted miner
    assert c2.table.last_number == done
    c2.sync()
    assert all(not k.endswith(f"seg-{n:06d}.jsonl.gz") for k in bucket.fetched for n in range(1, done + 1))
    summaries = [c2.table.summary(MATH, 10 + i) for i in range(6)]
    assert [s.n_observations for s in summaries] == [1] * 6
    assert c2.table.last_number == len(log_events(rt))
    c2.sync(force=True)                                  # nothing new: nothing applied twice
    assert [c2.table.summary(MATH, 10 + i).n_observations for i in range(6)] == [1] * 6


def test_a_failed_segment_application_commits_nothing():
    table = ObservationTable()
    good = {"type": "observation", "id": "a", "env": MATH, "prompt_idx": 1, "window": 1, "lane": "training",
            "verdict": "in-zone", "status": "proven", "proof": "proven", "rewards_bps": [0] * 16}
    with pytest.raises(ObservationVerificationError):
        table.apply_segment(1, 2, [good, {"type": "observation", "id": "b"}])
    assert table.last_number == 0 and table.prompts(MATH) == [] and table.records(MATH, 1) == []
    table.apply_segment(1, 2, [good, good])               # an id never counts twice
    assert table.summary(MATH, 1).n_observations == 1 and table.last_number == 1


def test_a_table_of_another_run_or_validator_is_refused(tmp_path):
    bucket = FakeR2()
    client(bucket, tmp_path).table.close()
    client(bucket, tmp_path).table.close()                 # the same run and validator reopen fine
    with pytest.raises(ValueError, match="another validator"):
        ObservationClient(BASE, RUN, "5Other", fetch=bucket.fetch, directory=tmp_path / "miner")
    with pytest.raises(ValueError, match="plain identifier"):
        ObservationClient(BASE, "../x", "5Other", fetch=bucket.fetch, directory=tmp_path / "miner")


def test_the_default_directory_is_never_tmp(monkeypatch):
    from reliquary.miner.observation_client import default_directory
    monkeypatch.delenv("RELIQUARY_OBSERVATIONS_DIR", raising=False)
    assert not str(default_directory()).startswith("/tmp")


def test_the_head_is_polled_at_most_every_15_seconds_and_errors_back_off(tmp_path):
    rt = runtime(tmp_path / "v")
    explore(rt, prompt=11)
    bucket = FakeR2()
    publish_all(rt, bucket)
    now = [100.0]
    c = client(bucket, tmp_path, clock=lambda: now[0])
    assert c.sync() == 1
    assert bucket.count(index_key(RUN)) == 1
    now[0] += 14
    assert c.sync() == 0 and bucket.count(index_key(RUN)) == 1       # honours max-age 15
    now[0] += 2
    c.sync()
    assert bucket.count(index_key(RUN)) == 2
    saved = bucket.objects.pop(index_key(RUN))
    now[0] += 20
    with pytest.raises(OSError):
        c.sync()
    now[0] += 20                                                      # 1st failure: wait 30
    assert c.sync() == 0
    now[0] += 11
    with pytest.raises(OSError):
        c.sync()
    now[0] += 40                                                      # 2nd failure: wait 60
    assert c.sync() == 0
    now[0] += 30
    bucket.objects[index_key(RUN)] = saved
    assert c.sync() == 0 and c._failures == 0                         # recovered, back to the normal pace


# ------------------------------------------------------------------ table semantics

def obs(id_, prompt=1, *, lane="training", verdict="in-zone", status="proven", proof="proven", window=1,
        checkpoint_n=1, rewards=None, seeds=None, pool="p" * 64, uncertain=None, env="math"):
    e = {"type": "observation", "id": id_, "env": env, "prompt_idx": prompt, "window": window,
         "checkpoint_n": checkpoint_n, "lane": lane, "verdict": verdict, "status": status, "proof": proof,
         "rewards_bps": rewards or [0] * M_ROLLOUTS}
    if seeds is not None:
        e["candidate"] = {"pool_sha256": pool, "seeds": seeds}
    if uncertain:
        e["uncertain"] = uncertain
    return e


def settle(id_, status, proof, **kw):
    return {"type": "settle", "id": id_, "status": status, "proof": proof, **kw}


def test_scanned_means_proven_training_or_an_entitled_exploration():
    t = ObservationTable()
    t.apply(obs("a", 1, lane="training"))
    t.apply(obs("b", 2, lane="exploration", verdict="uniform-low", status="exploration_pending", proof="pending"))
    t.apply(obs("c", 3, lane="exploration", verdict="uniform-low", status="exploration_unpaid", proof="unproven"))
    t.apply(obs("d", 4, lane="exploration", verdict="uniform-low", status="exploration_pending", proof="pending"))
    t.apply(settle("d", "exploration_forfeited", "failed"))
    t.apply(obs("e", 5, lane="exploration", verdict="uniform-low", status="exploration_pending", proof="pending"))
    t.apply(settle("e", "exploration_paid", "audited"))
    assert [t.summary("math", p).scanned for p in range(1, 6)] == [True, True, False, False, True]
    assert t.scanned_prompts("math") == {1, 2, 5}
    assert t.summary("math", 5).first_scan_status == "exploration_paid"
    assert t.summary("math", 3).first_scan_status is None
    # a later training observation takes the scan back to the prompt a forfeit released
    t.apply(obs("f", 4, lane="training", window=2))
    assert t.summary("math", 4).scanned and t.summary("math", 4).first_scan_status == "proven"
    assert t.summary("math", 4).n_observations == 2 and t.summary("math", 4).last_window == 2


def test_best_in_zone_evidence_and_per_seed_rewards():
    t = ObservationTable()
    seeds = list(range(0, 2 * M_ROLLOUTS, 2))
    rewards = [10000, 0] * (M_ROLLOUTS // 2)
    t.apply(obs("a", 1, window=1, seeds=seeds, rewards=rewards, pool="a" * 64))
    t.apply(obs("b", 1, window=2, seeds=seeds, rewards=[0] * M_ROLLOUTS, pool="b" * 64, verdict="uniform-low"))
    best = t.summary("math", 1).best_in_zone
    assert best["pool_sha256"] == "a" * 64 and best["rewards_bps"] == rewards and best["seeds"] == seeds
    assert t.seed_rewards("math", 1, pool_sha256="a" * 64)[0] == [10000]
    assert t.seed_rewards("math", 1)[2] == [0, 0]
    t.apply(settle("a", "exploration_forfeited", "failed"))
    assert 0 not in t.seed_rewards("math", 1, pool_sha256="a" * 64)       # failed audit: not evidence
    t2 = ObservationTable()
    t2.apply(obs("u", 1, seeds=seeds, rewards=[10000] * M_ROLLOUTS, uncertain=[1, 3]))
    assert 2 not in t2.seed_rewards("math", 1) and 4 in t2.seed_rewards("math", 1)   # uncertain positions dropped


def test_compaction_folds_old_observations_into_the_prompt_row():
    t = ObservationTable()
    t.apply(obs("old", 1, window=1, lane="training"))
    t.apply(obs("old2", 2, window=1, verdict="uniform-low", lane="exploration", status="exploration_unpaid",
                proof="unproven"))
    t.apply(obs("new", 3, window=50))
    assert t.compact(10) == 2
    assert t.records("math", 1) == [] and t.summary("math", 1).scanned and t.summary("math", 1).first_scan_status == "proven"
    assert not t.summary("math", 2).scanned and t.summary("math", 1).n_observations == 1
    t.apply(settle("old", "trained", "proven"))            # settle of a compacted id: ignored
    assert t.scanned_prompts("math") == {1, 3}


# ------------------------------------------------------------------ policy

def test_default_policy_skips_confident_all_pass_and_retries_all_fail_later():
    t = ObservationTable()
    t.apply(obs("a", 1, verdict="uniform-high", checkpoint_n=1))
    t.apply(obs("b", 2, verdict="uniform-low", checkpoint_n=4, lane="exploration", status="exploration_paid",
                proof="audited"))
    t.apply(obs("c", 3, verdict="uniform-high", lane="exploration", status="exploration_unpaid", proof="unproven"))
    t.apply(obs("d", 4, verdict="uniform-high", checkpoint_n=1))
    t.apply(obs("e", 4, verdict="in-zone", checkpoint_n=2))
    t.apply(obs("f", 5, verdict="uniform-high", proof="failed"))
    p = DefaultPromptPolicy(retry_all_fail_after_checkpoints=3)
    assert p.skip(t, "math", 1, checkpoint_n=10)
    assert p.skip(t, "math", 2, checkpoint_n=6) and not p.skip(t, "math", 2, checkpoint_n=7)
    assert not p.skip(t, "math", 3, checkpoint_n=10)       # one unproven 16/16: not confident
    t.apply(obs("c2", 3, verdict="uniform-high", lane="exploration", status="exploration_unpaid", proof="unproven"))
    assert p.skip(t, "math", 3, checkpoint_n=10)           # seen twice
    assert not p.skip(t, "math", 4, checkpoint_n=10)       # it was in zone once
    t.apply(obs("g", 6, verdict="in-zone", checkpoint_n=1))
    t.apply(obs("h", 6, verdict="uniform-high", checkpoint_n=3))
    assert not p.skip(t, "math", 6, checkpoint_n=10)       # in zone once, then 16/16: not "all uniform"
    assert not p.skip(t, "math", 5, checkpoint_n=10)       # a failed audit is not evidence
    assert not p.skip(t, "math", 99, checkpoint_n=10)
    assert not DefaultPromptPolicy(skip_all_pass=False).skip(t, "math", 1, checkpoint_n=10)


def test_default_policy_prefers_unscanned_deterministically():
    t = ObservationTable()
    p = DefaultPromptPolicy(unscanned_share=0.85)
    draws = [p.prefers_unscanned(t, "math", seed=s) for s in range(400)]
    assert draws == [p.prefers_unscanned(t, "math", seed=s) for s in range(400)]
    assert 0.75 < sum(draws) / 400 < 0.95
    assert not any(DefaultPromptPolicy(unscanned_share=0.0).prefers_unscanned(t, "math", seed=s) for s in range(50))
    assert all(DefaultPromptPolicy(unscanned_share=1.0).prefers_unscanned(t, "math", seed=s) for s in range(50))


def test_default_policy_prefers_seed_subsets_in_zone_from_the_same_pool():
    t = ObservationTable()
    p = DefaultPromptPolicy()
    kwargs = dict(pool_sha256="a" * 64, group_size=M_ROLLOUTS, pool_seeds=POOL)
    assert p.preferred_seeds(t, "math", 1, **kwargs) is None            # no evidence
    seeds = list(range(0, POOL, 2))
    # in pool "a" only the first seed succeeded; another pool says the opposite and must be ignored
    t.apply(obs("a", 1, seeds=seeds, rewards=[10000] + [0] * (M_ROLLOUTS - 1), pool="a" * 64))
    t.apply(obs("o", 1, seeds=seeds, rewards=[0] + [10000] * (M_ROLLOUTS - 1), pool="b" * 64, window=2))
    chosen = p.preferred_seeds(t, "math", 1, **kwargs)
    assert chosen == p.preferred_seeds(t, "math", 1, **kwargs)           # deterministic
    assert len(chosen) == M_ROLLOUTS == len(set(chosen)) and list(chosen) == sorted(chosen)
    assert 0 in chosen and len([x for x in chosen if x in seeds[1:]]) == M_ROLLOUTS // 2   # the success, half failures, unseen fill
    # known successes and failures plus unseen seeds: half of each, then unseen
    t3 = ObservationTable()
    t3.apply(obs("s", 1, seeds=seeds, rewards=[10000] * M_ROLLOUTS, pool="a" * 64, verdict="uniform-high"))
    t3.apply(obs("f", 1, seeds=list(range(1, POOL, 2)), rewards=[0] * M_ROLLOUTS, pool="a" * 64, verdict="uniform-low"))
    full = p.preferred_seeds(t3, "math", 1, **kwargs)
    assert sum(1 for x in full if x % 2 == 0) == M_ROLLOUTS // 2 == sum(1 for x in full if x % 2 == 1)
    # only failures known: nothing to mix with, so no preference
    t2 = ObservationTable()
    t2.apply(obs("z", 1, seeds=seeds, rewards=[0] * M_ROLLOUTS, pool="a" * 64, verdict="uniform-low"))
    assert p.preferred_seeds(t2, "math", 1, **kwargs) is None


def test_load_policy_by_import_path():
    assert isinstance(load_policy(None), DefaultPromptPolicy)
    assert type(load_policy("reliquary.miner.observation_client:DefaultPromptPolicy")) is DefaultPromptPolicy
    with pytest.raises(ValueError):
        load_policy("nocolon")


def test_skipped_set_follows_the_table_and_the_policy(tmp_path):
    bucket = FakeR2()
    c = client(bucket, tmp_path)
    c.table.apply(obs("a", 1, verdict="uniform-high"))
    policy = DefaultPromptPolicy()
    assert c.skipped("math", policy=policy, checkpoint_n=5) == {1}
    c.table.apply(obs("b", 2, verdict="uniform-high"))
    assert c.skipped("math", policy=policy, checkpoint_n=5) == {1, 2}      # cache follows the table
    assert c.scanned("math") == {1, 2}


# ------------------------------------------------------------------ engine

class _Rng:
    def getrandbits(self, n):
        return 7


def bare_engine():
    engine = object.__new__(MiningEngine)
    engine._cooldown_per_env = {"math": {3}}
    return engine


def test_legacy_engine_is_unchanged_without_an_observation_source(monkeypatch):
    for name in ("URL", "RUN_ID", "VALIDATOR_HOTKEY", "DIR"):
        monkeypatch.delenv(f"RELIQUARY_OBSERVATIONS_{name}", raising=False)
    engine = bare_engine()
    engine._observations = engine._prompt_policy = None
    calls = []
    monkeypatch.setattr(engine_module.asyncio, "to_thread", lambda *a, **k: calls.append(a))
    engine._configure_observations()
    assert engine._observations is None
    options = asyncio.run(engine._cooldown_options(SimpleNamespace(service_policy=object(), checkpoint_n=3), _Rng()))
    assert len(options) == 1 and options[0] is engine._cooldown_per_env and calls == []
    # even a bare engine that never ran __init__ takes the legacy path
    assert asyncio.run(bare_engine()._cooldown_options(SimpleNamespace(service_policy=object()), _Rng()))[0] == {"math": {3}}
    pool = SimpleNamespace(group_size=M_ROLLOUTS)
    asked = []
    out = engine.choose_public_seed_group(pool, lambda seeds: asked.append(tuple(seeds)) or "g", problem=None, env=None)
    assert out == "g" and asked == [tuple(range(M_ROLLOUTS))]
    assert "self._configure_observations()" in __import__("inspect").getsource(MiningEngine.__init__)


def test_engine_with_a_source_merges_skips_and_prefers_unscanned_only_for_service_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_URL", BASE)
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_RUN_ID", RUN)
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_VALIDATOR_HOTKEY", keypair().ss58_address)
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_DIR", str(tmp_path / "t"))
    engine = bare_engine()
    engine._configure_observations()
    assert isinstance(engine._prompt_policy, DefaultPromptPolicy)
    engine._observations.fetch = lambda url: (_ for _ in ()).throw(OSError("offline"))   # sync fails: table still used
    engine._observations.table.apply(obs("a", 1, verdict="uniform-high"))            # skipped
    engine._observations.table.apply(obs("b", 2, verdict="in-zone"))                  # scanned, still allowed
    service = SimpleNamespace(service_policy=object(), checkpoint_n=5)
    engine._prompt_policy = DefaultPromptPolicy(unscanned_share=1.0)
    fresh, hard = asyncio.run(engine._cooldown_options(service, _Rng()))
    assert hard == {"math": {1, 3}} and fresh == {"math": {1, 2, 3}}
    assert engine._cooldown_per_env == {"math": {3}}                                  # the engine's own set is untouched
    engine._prompt_policy = DefaultPromptPolicy(unscanned_share=0.0)
    assert asyncio.run(engine._cooldown_options(service, _Rng())) == [{"math": {1, 3}}]
    # a non-service window ignores the source altogether
    assert asyncio.run(engine._cooldown_options(SimpleNamespace(service_policy=None), _Rng()))[0] is engine._cooldown_per_env


def test_engine_hook_uses_the_policy_seeds_only_with_a_source(tmp_path):
    engine = bare_engine()
    engine._observations = SimpleNamespace(table=ObservationTable())
    engine._prompt_policy = DefaultPromptPolicy()
    seeds = list(range(1, POOL, 2))
    engine._observations.table.apply(obs("a", 7, seeds=seeds, rewards=[10000] + [0] * (M_ROLLOUTS - 1), pool="a" * 64))
    pool = SimpleNamespace(group_size=M_ROLLOUTS, pool_seeds=POOL, environment="math", prompt_idx=7, sha256="a" * 64)
    asked = []
    engine.choose_public_seed_group(pool, lambda s: asked.append(tuple(s)) or [], problem=None, env=None)
    assert 1 in asked[0] and asked[0] != tuple(range(M_ROLLOUTS))
    pool.prompt_idx = 8                                                            # no evidence: default seeds
    engine.choose_public_seed_group(pool, lambda s: asked.append(tuple(s)) or [], problem=None, env=None)
    assert asked[1] == tuple(range(M_ROLLOUTS))
    engine._prompt_policy = SimpleNamespace(skip=lambda *a, **k: False)            # a policy without the extra method
    pool.prompt_idx = 7
    engine.choose_public_seed_group(pool, lambda s: asked.append(tuple(s)) or [], problem=None, env=None)
    assert asked[2] == tuple(range(M_ROLLOUTS))
