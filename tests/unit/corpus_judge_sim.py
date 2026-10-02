"""A virtual-time harness for the corpus judge: Poisson arrivals at a chosen
rate, a store whose every call takes 50-100 ms, a drand beacon, and a GPU that
audits at a fixed token rate. Hours of production traffic run in seconds, so a
test can show whether the judge keeps up. Not a test module itself."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import random
import types
from dataclasses import dataclass, field

from reliquary.corpus.audit_policy import AuditParams, MinerState

T0 = 1_790_000_000.0
DRAND_PERIOD = 3.0


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps to the next timer when nothing is
    ready. Executor calls run inline (they are CPU work in this harness), after
    the virtual delay a callable may declare in ``virtual_seconds(*args)``."""

    def __init__(self) -> None:
        super().__init__()
        self._now = 0.0
        real_select = self._selector.select

        def select(timeout=None):
            events = real_select(0)
            if events or timeout == 0:
                return events
            if timeout is None:
                return real_select(None)
            self._now += timeout
            return []

        self._selector.select = select

    def time(self) -> float:
        return self._now

    def run_in_executor(self, executor, func, *args):
        future = self.create_future()
        target, inner = func, args
        if isinstance(func, functools.partial) and func.args:
            target, inner = func.args[0], func.args[1:]
        delay = getattr(target, "virtual_seconds", None)
        seconds = float(delay(*inner)) if delay is not None else 0.0

        def finish():
            if future.cancelled():
                return
            try:
                future.set_result(func(*args))
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting task
                future.set_exception(exc)

        if seconds > 0:
            self.call_later(seconds, finish)
        else:
            finish()
        return future


def virtual_clock():
    loop = asyncio.get_running_loop()
    return T0 + loop.time()


class _Clock:
    """The auditor's wall clock, read off the running virtual loop."""

    def __call__(self) -> float:
        return virtual_clock()


def round_at(t: float) -> int:
    return int(t // DRAND_PERIOD) + 1


class Beacon:
    def __init__(self, latency: float = 0.2) -> None:
        self.latency = latency
        self.calls = 0

    def virtual_seconds(self, round_number):
        return self.latency

    def __call__(self, round_number: int) -> str:
        self.calls += 1
        return hashlib.sha256(f"drand-{round_number}".encode()).hexdigest()


class Store:
    """The record store and miners.json, in memory, each call 50-100 ms.

    ``in_flight_cap`` models the process's shared S3 connection pool."""

    def __init__(self, *, seed: int = 0, latency=(0.05, 0.10), list_seconds_per_key=0.0014,
                 in_flight_cap: int | None = None, write_latency=None) -> None:
        self.submissions: dict[str, dict] = {}
        self.verdicts: dict[str, dict] = {}
        self.miners: dict = {}
        self.miners_etag = 0
        self._rng = random.Random(seed)
        self._latency = latency
        self._write_latency = write_latency or latency
        self._list_per_key = list_seconds_per_key
        self._pool = asyncio.Semaphore(in_flight_cap) if in_flight_cap else None
        self.calls: dict[str, int] = {}
        self.peak_in_flight = 0
        self._in_flight = 0

    async def _io(self, kind: str, seconds: float | None = None) -> None:
        self.calls[kind] = self.calls.get(kind, 0) + 1
        bounds = self._write_latency if kind.startswith("write") else self._latency
        delay = self._rng.uniform(*bounds) if seconds is None else seconds
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self._pool is not None:
                async with self._pool:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)
        finally:
            self._in_flight -= 1

    async def read_submission(self, job_id, sid):
        await self._io("read_submission")
        return self.submissions.get(sid)

    async def list_submission_ids(self, job_id):
        await self._io("list", len(self.submissions) * self._list_per_key)
        return sorted(self.submissions)

    async def list_verdict_ids(self, job_id):
        await self._io("list", len(self.verdicts) * self._list_per_key)
        return sorted(self.verdicts)

    async def write_verdict(self, job_id, sid, verdict):
        await self._io("write_verdict")
        if sid in self.verdicts:
            return False
        # What lands is what a read gives back: a JSON round trip.
        self.verdicts[sid] = json.loads(json.dumps(verdict))
        return True

    async def read_verdict(self, job_id, sid):
        await self._io("read_verdict")
        return self.verdicts.get(sid)

    async def read_miners(self, job_id):
        await self._io("read_miners")
        return json.loads(json.dumps(self.miners)), str(self.miners_etag)

    async def write_miners(self, job_id, state, etag):
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        await self._io("write_miners")
        if etag != str(self.miners_etag):
            raise CorpusStoreConflict("miners.json changed")
        self.miners = json.loads(json.dumps(state))
        self.miners_etag += 1
        return str(self.miners_etag)


class Gpu:
    """``_judge_many`` stand-in: every record passes, unless its hotkey cheats."""

    def __init__(self, tokens_per_second: float = 4860.0, cheaters=()) -> None:
        self.tokens_per_second = tokens_per_second
        self.cheaters = set(cheaters)
        self.busy = 0.0
        self.audited = 0
        self.calls = 0

    def virtual_seconds(self, records):
        seconds = sum(int(r["token_count"]) for r in records) / self.tokens_per_second
        self.busy += seconds
        return seconds

    def __call__(self, records):
        self.calls += 1
        self.audited += len(records)
        return [{"passed": r["hotkey"] not in self.cheaters,
                 "reason": None if r["hotkey"] not in self.cheaters else "exp_mismatch",
                 "worst_exp": 0, "worst_mant_mean": 0.01, "worst_mant_median": 0.01}
                for r in records]


@dataclass
class Population:
    """Hotkeys and their share of the traffic; each record ~``tokens`` long."""

    hotkeys: list[str]
    weights: list[float]
    tokens: int = 1400
    probation: set = field(default_factory=set)

    @classmethod
    def prod_like(cls, n: int = 24, probation: int = 2, seed: int = 0) -> "Population":
        rng = random.Random(seed)
        hotkeys = [f"5Hk{i:03d}" for i in range(n)]
        weights = [1.0 / (i + 1) ** 0.8 * rng.uniform(0.8, 1.2) for i in range(n)]
        return cls(hotkeys, weights, probation=set(hotkeys[-probation:]) if probation else set())


@dataclass
class Result:
    arrivals: int
    verdicts: int
    pending_end: int
    oldest_pending_age_end: float | None
    max_oldest_age: float
    pending_series: list
    gpu_busy: float
    store_calls: dict
    peak_in_flight: int
    verdict_set: dict


def _sid(rng: random.Random) -> str:
    return "%064x" % rng.getrandbits(256)


async def _drive(module, *, rate_per_hour, hours, params, population, seed, store, gpu,
                 beacon, auditor_kwargs, sample_every, start_backlog, backlog_age, drain_hours):
    from reliquary.validator.corpus_miner_states import MinerStates

    rng = random.Random(seed)
    clock = _Clock()
    states = MinerStates(store, "math-v1", sleep=asyncio.sleep, clock=clock)
    for hotkey in population.hotkeys:
        if hotkey not in population.probation:
            store.miners[hotkey] = MinerState(audited_passed=10 * params.probation_submissions).to_dict()
    auditor = module.CorpusAuditor(
        job_id="math-v1", records=store, model=None, tokenizer=None, proof=None,
        params=params, miner_states=states, beacon=beacon, round_at=round_at, clock=clock,
        **auditor_kwargs)
    auditor._judge_many = gpu
    received: dict[str, float] = {}

    def submit(at: float) -> str:
        hotkey = rng.choices(population.hotkeys, population.weights)[0]
        sid = _sid(rng)
        tokens = max(64, int(rng.gauss(population.tokens, population.tokens / 3)))
        store.submissions[sid] = {"hotkey": hotkey, "received_at": at, "token_count": tokens,
                                  "rendered_prompt": "", "completions": [{"tokens": [1] * 4}]}
        received[sid] = at
        return sid

    # A backlog older than the run, already listed by a restarted process.
    for _ in range(start_backlog):
        submit(virtual_clock() - rng.uniform(0, backlog_age))

    runner = asyncio.ensure_future(auditor.run())
    series = []
    max_oldest = 0.0
    end = virtual_clock() + hours * 3600
    next_sample = virtual_clock()
    mean_gap = 3600.0 / rate_per_hour

    async def arrivals():
        while virtual_clock() < end:
            await asyncio.sleep(rng.expovariate(1.0 / mean_gap))
            sid = submit(virtual_clock())
            auditor.enqueue(sid)

    feeder = asyncio.ensure_future(arrivals())
    stop = end + drain_hours * 3600
    try:
        while virtual_clock() < stop:
            await asyncio.sleep(min(sample_every, max(1.0, next_sample - virtual_clock())))
            if runner.done():
                runner.result()
            if virtual_clock() >= next_sample:
                pending = [s for s in received if s not in store.verdicts]
                oldest = (virtual_clock() - min(received[s] for s in pending)) if pending else 0.0
                max_oldest = max(max_oldest, oldest)
                series.append((round((virtual_clock() - T0) / 3600, 2), len(pending),
                               round(oldest), len(store.verdicts)))
                next_sample += sample_every
    finally:
        feeder.cancel()
        runner.cancel()
        for task in (feeder, runner):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    pending = [s for s in received if s not in store.verdicts]
    return Result(
        arrivals=len(received), verdicts=len(store.verdicts), pending_end=len(pending),
        oldest_pending_age_end=(virtual_clock() - min(received[s] for s in pending)) if pending else None,
        max_oldest_age=max_oldest, pending_series=series, gpu_busy=gpu.busy,
        store_calls=dict(store.calls), peak_in_flight=store.peak_in_flight,
        verdict_set={sid: _verdict_key(v) for sid, v in store.verdicts.items()},
    )


def _verdict_key(v: dict) -> tuple:
    return (v["hotkey"], v["passed"], v.get("audited"), v.get("reason"),
            json.dumps(v.get("draw"), sort_keys=True))


def simulate(module, *, rate_per_hour: float, hours: float, params: AuditParams | None = None,
             population: Population | None = None, seed: int = 0, store: Store | None = None,
             gpu: Gpu | None = None, beacon: Beacon | None = None, auditor_kwargs=None,
             sample_every: float = 600.0, start_backlog: int = 0, backlog_age: float = 7 * 3600,
             drain_hours: float = 0.0, patch_time: bool = True) -> Result:
    """Run ``module.CorpusAuditor.run()`` under ``rate_per_hour`` arrivals for
    ``hours`` of virtual time (then ``drain_hours`` without arrivals)."""
    params = params or AuditParams(q=0.15, probation_submissions=100, hold_seconds=4320.0,
                                   ban_after_failures=1000)
    loop = VirtualTimeLoop()
    saved = module.time
    if patch_time:
        # The run loop's own timers (full rescan, idle log, phase timings).
        module.time = types.SimpleNamespace(monotonic=loop.time, time=lambda: T0 + loop.time())
    try:
        asyncio.set_event_loop(loop)
        store = store or Store(seed=seed)
        return loop.run_until_complete(_drive(
            module, rate_per_hour=rate_per_hour, hours=hours, params=params,
            population=population or Population.prod_like(seed=seed), seed=seed, store=store,
            gpu=gpu or Gpu(), beacon=beacon or Beacon(), auditor_kwargs=auditor_kwargs or {},
            sample_every=sample_every, start_backlog=start_backlog, backlog_age=backlog_age,
            drain_hours=drain_hours))
    finally:
        module.time = saved
        asyncio.set_event_loop(None)
        loop.close()
