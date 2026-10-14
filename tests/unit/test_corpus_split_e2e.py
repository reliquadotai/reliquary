"""Real processes, real time, one shared on-disk bucket with R2-like latency:
the split validator against the single process, and the split validator
losing a judge, its front, or its GPU process.

The math job is judged in its own process, the code job in the front (the
math-only rollout step); the single process judges both."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import signal
import time
from pathlib import Path

import pytest

from reliquary.infrastructure import corpus_record_store as record_store
from tests.unit import corpus_split_harness as h

MATH, CODE = "math-v1", "code-v1"
CHEAT = h.HOTKEYS[0]
HOTKEYS = h.HOTKEYS[:6]
HOLD = 600.0
AUDITOR = {"accept_slack_seconds": 5.0, "rescan_every_seconds": 2.0}


def _served(**audit):
    params = {"audit_q": 0.5, "audit_hold_seconds": HOLD, **audit}
    return [(h.entry("corpus-math", MATH, 0.1, **params), 0.1),
            (h.entry("corpus-code", CODE, 0.05, **params), 0.05)]


def _seed(root: Path, *, now: float, old: int, fresh: int) -> dict[str, list[str]]:
    """Per job: ``old`` records past their hold (undrawn ones pass unaudited,
    drawn ones are audited) and ``fresh`` ones inside it (they keep every
    hotkey off the slow-hotkey rule; drawn ones are audited, the rest wait).
    Received times are fixed, so two runs decide alike. One hotkey forges."""
    rng = random.Random(7)
    ids: dict[str, list[str]] = {}
    for job in (MATH, CODE):
        ids[job] = []
        for k in range(old + fresh):
            hotkey = HOTKEYS[k % len(HOTKEYS)]
            received = (now - HOLD - 60 - rng.uniform(0, 600) if k < old
                        else now - rng.uniform(30, 200))
            sid = "%064x" % rng.getrandbits(256)
            h.put_raw(root, record_store._key(job, "submissions", sid), h.synth_stub(
                sid=sid, job_id=job, hotkey=hotkey, received_at=received,
                n=256 + 32 * rng.randrange(8), forged=hotkey == CHEAT and k % 3 == 0))
            ids[job].append(sid)
    return ids


def _key(v):
    return (v["hotkey"], v["passed"], v["audited"], v["reason"],
            json.dumps(v.get("draw"), sort_keys=True))


def _wait(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    log = tmp_path / "children.log"
    with h.environment(h.harness_env(root, CORPUS_HARNESS_LOG=log, CORPUS_HARNESS_BEACON="const",
                                     CORPUS_HARNESS_GPU_TPS=20000)):
        h.seed_bucket(root, [MATH, CODE], hotkeys=HOTKEYS)
        yield root, log


def _outcome(root):
    out = {}
    for job in (MATH, CODE):
        verdicts = {sid: _key(v) for sid, v in h.listed(root, job, "verdicts").items()}
        archives = {}
        for path in sorted((root / "archives").glob(f"corpus-{job.split('-')[0]}-*.json")):
            # Keyed by work period: the entry period depends on when it settled.
            archives[path.stem.rsplit("-", 1)[0]] = json.loads(path.read_text())["rewards_by_hotkey"]
        out[job] = (verdicts, archives, sorted(h.settlement(root, job).get("settled") or []))
    return out


def _wait_stable(read, quiet_seconds, timeout, what):
    """Until ``read()`` returns the same value for ``quiet_seconds``."""
    deadline = time.monotonic() + timeout
    last, since = read(), time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(1.0)
        now = read()
        if now != last:
            last, since = now, time.monotonic()
        elif time.monotonic() - since >= quiet_seconds:
            return
    raise AssertionError(f"timed out waiting for {what} to settle down")


def _settled_all(root, job):
    state = h.settlement(root, job)
    return state.get("last_entry") is not None and not state.get("pending")


def _run_single(root, served, *, settle_every):
    port = h.free_port()
    process = h.start_single(served, port=port, settle_every_seconds=settle_every,
                             auditor_kwargs=AUDITOR)
    return process, port


def test_the_split_writes_the_single_process_verdicts_and_archives(bucket, tmp_path):
    """Same records, same received times, one beacon: the single process and
    the split (math in a judge process, code in the front) write the same
    verdicts and pay the same periods."""
    root, _ = bucket
    now = time.time()
    ids = _seed(root, now=now, old=120, fresh=60)
    served = _served()
    settle_every = 40.0

    def judged_enough():
        # Every old record and every fresh drawn one: verdict counts stop moving.
        return all(len(h.listed(root, job, "verdicts")) >= 120 for job in (MATH, CODE))

    process, _ = _run_single(root, served, settle_every=settle_every)
    try:
        _wait(judged_enough, 120, "the single process's verdicts")
        _wait(lambda: all(_settled_all(root, j) for j in (MATH, CODE)), 120, "its settlement")
        # Periods close one after another as their records are decided: wait
        # until nothing more can settle (the fresh undrawn records wait their
        # hold, past this run), i.e. two settle passes change nothing.
        _wait_stable(lambda: _outcome(root), settle_every * 2 + 5, 240, "its settlement")
    finally:
        process.kill()
        process.join(10)
    single = _outcome(root)

    # The same bucket again, before any verdict or settlement.
    for job in (MATH, CODE):
        for kind in ("verdicts", "voided"):
            for path in (root / "objects" / record_store._prefix(job, kind)).glob("*"):
                path.unlink()
        (root / "objects" / record_store._settlement_key(job)).unlink(missing_ok=True)
    for path in (root / "archives").glob("*"):
        path.unlink()
    h.seed_bucket(root, [MATH, CODE], hotkeys=HOTKEYS)  # miners.json as it was

    spec = h.split_spec(root, served, [[MATH]], port=h.free_port(),
                        settle_every_seconds=settle_every, auditor_kwargs=AUDITOR)
    with h.SupervisorThread(spec):
        _wait(judged_enough, 120, "the split's verdicts")
        _wait(lambda: _outcome(root) == single, 240, "the split's settlement")
    split = _outcome(root)

    for job in (MATH, CODE):
        s_verdicts, s_archives, s_settled = single[job]
        p_verdicts, p_archives, p_settled = split[job]
        assert set(s_verdicts) == set(p_verdicts), job
        assert s_verdicts == p_verdicts, job
        assert s_archives == p_archives and s_archives, job
        assert s_settled == p_settled, job
        audited = collections.Counter(v[2] for v in p_verdicts.values())
        assert audited[True] and audited[False], (job, audited)   # both paths exercised
        assert any(not v[1] for v in p_verdicts.values()), job      # the forger was caught
    assert ids


def _passes_counted(root, job):
    miners = json.loads(h.get_raw(root, record_store._miners_key(job)))
    return {hk: m.get("audited_passed", 0) - 1000 for hk, m in miners.items()}


def _audited_passes(root, job):
    count = collections.Counter()
    for v in h.listed(root, job, "verdicts").values():
        if v["passed"] and v["audited"]:
            count[v["hotkey"]] += 1
    return count


def _counts_caught_up(root, jobs) -> bool:
    """miners.json counts every audited pass written so far. A pass is counted
    one store round trip after its verdict lands, so a reader that stops the
    processes the moment the verdicts are there can cut that write: the
    single process loses the same count when it is stopped there. Waited for
    while the judging processes still run (exact there); a count that never
    catches up is a real inconsistency and fails the wait. Stopping the
    supervisor afterwards is itself a kill: a pass still writing then may
    lose its count, so after the stop only "never above" is checked."""
    for job in jobs:
        counted = _passes_counted(root, job)
        audited = _audited_passes(root, job)
        if {hk: counted.get(hk, 0) for hk in audited} != dict(audited):
            return False
    return True


def _invariants(root, *, killed=()):
    """Nothing judged or paid twice, however often a process was killed.

    A pass is counted in miners.json after its verdict is written, only by the
    call that wrote it: a process killed in between loses that count (never
    doubles it), as the single process does. So the count is exact for a job
    whose judging process was not killed, and never above it otherwise."""
    for job in (MATH, CODE):
        verdict_docs = h.listed(root, job, "verdicts")
        counted = _passes_counted(root, job)
        audited = _audited_passes(root, job)
        counted = {hk: counted.get(hk, 0) for hk in audited}
        if job in killed:
            assert all(counted[hk] <= audited[hk] for hk in audited), (job, counted, audited)
        else:
            assert counted == dict(audited), job
        state = h.settlement(root, job)
        settled = state.get("settled") or []
        assert len(settled) == len(set(settled)) and set(settled) <= set(verdict_docs), job
    # A period archived again after a crash pays the same rewards.
    by_window = collections.defaultdict(set)
    for line in (root / "archives" / "writes.jsonl").read_text().splitlines():
        doc = json.loads(line)
        by_window[doc["task"], doc["window"]].add(json.dumps(doc["rewards"], sort_keys=True))
    assert by_window and all(len(v) == 1 for v in by_window.values()), by_window


def _crash_run(root, *, old, gpu_tps):
    _seed(root, now=time.time(), old=old, fresh=60)
    os.environ[h.GPU_TPS_ENV] = str(gpu_tps)
    port = h.free_port()
    spec = h.split_spec(root, _served(), [[MATH]], port=port, settle_every_seconds=5.0,
                        auditor_kwargs=AUDITOR)
    return spec, f"http://127.0.0.1:{port}"


def _verdicts(root, job=MATH):
    return len(h.listed(root, job, "verdicts"))


def _audited(root, job=MATH):
    return sum(v["audited"] for v in h.listed(root, job, "verdicts").values())


def test_a_gpu_process_that_dies_makes_audits_wait_never_fail(bucket):
    root, log = bucket
    spec, base = _crash_run(root, old=600, gpu_tps=3000)
    with h.SupervisorThread(spec) as sup:
        asyncio.run(h.wait_http(base, timeout=120))
        # Both jobs' first passes are being scored (tens of seconds each).
        time.sleep(5.0)
        os.kill(sup.pid("gpu"), signal.SIGKILL)
        _wait(lambda: sup.children["gpu"].starts == 2, 30, "the GPU process's restart")
        _wait(lambda: "corpus gpu process unavailable" in log.read_text(), 30,
              "an auditor waiting for the GPU")
        _wait(lambda: _verdicts(root) >= 600 and _verdicts(root, CODE) >= 600, 300,
              "every old record judged")
        _wait(lambda: _settled_all(root, MATH) and _settled_all(root, CODE), 60, "settlements")
        # Waiting is not failing: no auditor halted, no child but the GPU restarted.
        assert sup.children["judge-0"].starts == 1 and sup.children["front"].starts == 1
        _wait(lambda: _counts_caught_up(root, (MATH, CODE)), 60, "every pass counted")
    assert _audited(root) > 100 and _audited(root, CODE) > 100
    _invariants(root, killed=(MATH, CODE))   # the final stop kills both


def test_a_front_that_dies_leaves_the_judges_judging(bucket):
    root, log = bucket
    spec, base = _crash_run(root, old=1200, gpu_tps=4000)
    with h.SupervisorThread(spec) as sup:
        asyncio.run(h.wait_http(base, timeout=120))
        _wait(lambda: _verdicts(root) >= 200, 360, "the first math verdicts")
        os.kill(sup.pid("front"), signal.SIGKILL)
        audited = _audited(root)
        # Audits go on without a front (unaudited passes wait for its feed).
        _wait(lambda: _audited(root) > audited + 20, 120, "audits while the front is down")
        _wait(lambda: sup.children["front"].starts == 2, 30, "the front's restart")
        asyncio.run(h.wait_http(base, timeout=120))
        results = asyncio.run(h.submit_load(base, MATH, seconds=4, rate=3, size=(256, 512),
                                            hotkeys=HOTKEYS[1:], seed=5, first_prompt=10_000))
        assert sum(r["accepted"] for r in results) >= 5, h.quantiles(results)
        _wait(lambda: _verdicts(root) >= 1200 and _verdicts(root, CODE) >= 1200, 300,
              "every old record judged")
        _wait(lambda: _settled_all(root, MATH), 60, "a settlement")
        assert sup.children["judge-0"].starts == 1
        _wait(lambda: _counts_caught_up(root, (MATH,)), 60, "every pass counted")
    # The judge listed the store for its first front and for the new one.
    assert log.read_text().count("corpus feed: front epoch") >= 2
    _invariants(root, killed=(MATH, CODE))   # code's front was killed; the stop kills all


def test_a_judge_killed_mid_pass_resumes_from_the_bucket(bucket):
    root, log = bucket
    spec, base = _crash_run(root, old=1200, gpu_tps=6000)
    with h.SupervisorThread(spec) as sup:
        asyncio.run(h.wait_http(base, timeout=120))
        _wait(lambda: _verdicts(root) >= 150, 360, "the first math verdicts")
        killed_at = _verdicts(root)
        os.kill(sup.pid("judge-0"), signal.SIGKILL)
        _wait(lambda: sup.children["judge-0"].starts == 2, 30, "the judge's restart")
        _wait(lambda: _verdicts(root) >= 1200, 300, "every old math record judged")
        _wait(lambda: _settled_all(root, MATH), 60, "a settlement")
        assert sup.children["front"].starts == 1 and sup.children["gpu"].starts == 1
        _wait(lambda: _counts_caught_up(root, (CODE,)), 60, "every pass counted")
    assert killed_at < 1200
    _invariants(root, killed=(MATH, CODE))   # the math judge was killed; the stop kills all
