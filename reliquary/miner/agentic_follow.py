"""Following agentic orders (`reliquary corpus mine-agentic` without ``--job-id``).

The validator lists every order it serves (``/corpus/jobs``, ``orders``). This loop reads each listed order's
manifest, keeps the episode orders this install can serve (its env package at the order's pin, the sandbox extra
for a signed-sandbox order, the model and context the operator allows), ranks them by what they pay, and mines
the best ones, each in its own process (`reliquary corpus mine-agentic --job-id`), up to ``max_jobs`` at once.
A slot is freed when its child ends: a clean exit (the order complete) is never restarted, a failed one is
retried after a backoff that doubles per failure. A child is stopped (SIGTERM: it starts nothing more, its running
episodes are cancelled, their leases released and sessions reported) only when its order leaves the list, or stays
paused for ``PAUSED_STOP_SECONDS``; a draining order's child is left to finish. While no order is open it fetches
the model of one that is starting.

The validator's hotkey (signed-sandbox session requests are signed for it) is ``--validator-hotkey`` when given
(a validator naming another ``sandbox_hotkey`` then gets no signed-sandbox order: its signatures would be refused),
else the listing's ``sandbox_hotkey``.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from reliquary.miner.order_follow import IDLE_SECONDS, child_env, model_of

logger = logging.getLogger(__name__)

# Reads of the list in a row on which a mined order is missing before its child is stopped: one read may race a
# change of the list.
STALE_READS = 2
# A mined order paused (or held back to starting) this long has its child stopped; a shorter pause is waited out.
PAUSED_STOP_SECONDS = 1800.0
# How long a stopped child gets to cancel its episodes, release their leases and report their sessions before it is
# killed (its own report bound is shorter).
STOP_GRACE_SECONDS = 120.0
# A child that failed (any non-zero exit the follower did not cause) is retried after BACKOFF_SECONDS, doubled per
# failure of that order in a row, at most BACKOFF_MAX_SECONDS.
BACKOFF_SECONDS = 60.0
BACKOFF_MAX_SECONDS = 3600.0


@dataclass(frozen=True)
class Candidate:
    """An open episode order this miner can serve."""

    job_id: str
    model: str | None
    pay: float
    """Its auction's price of the moment, else its ceiling (alpha per million tokens); 0 when unpriced."""
    leases_free: bool
    signed: bool


def order_pay(order: Mapping) -> float:
    auction = order.get("auction") or {}
    for key in ("price_alpha_per_mtoken", "ceiling_alpha_per_mtoken"):
        value = auction.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def leases_free(order: Mapping) -> bool:
    """Whether its auction has leases left to grant (an order not priced by auction: unknown, so yes)."""
    auction = order.get("auction") or {}
    budget, leased = auction.get("budget"), auction.get("leased")
    if isinstance(budget, int) and isinstance(leased, int):
        return leased < budget
    return True


def rank(candidates: Sequence[Candidate], *, loaded: frozenset[str] = frozenset()) -> list[Candidate]:
    """Best first: leases left, then pay, then a model already on disk, then the listed order."""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda item: (not item[1].leases_free, -item[1].pay,
                                   item[1].model not in loaded, item[0]))
    return [candidate for _, candidate in indexed]


def serve_refusal(job) -> str | None:
    """Why this install cannot mine the episode ``job``, or None: the checks `corpus mine-agentic` makes before
    it downloads the model, without building the job's source (its child does that)."""
    if job.episode is None:
        return "not an episode job (`corpus mine` mines it)"
    try:
        return _serve_refusal(job)
    except ImportError as exc:
        return f"this install cannot load the agentic miner ({exc}): see `reliquary miner-envs`"


def _serve_refusal(job) -> str | None:
    from reliquary.miner.suite_episode import is_suite_episode

    if is_suite_episode(job.episode):
        from reliquary.corpus.suite_episode import suite_spec, suite_spec_refusal
        from reliquary.environment.agentic.external import verify_external_artifact

        refusal = suite_spec_refusal(job.episode)
        if refusal:
            return refusal
        spec = suite_spec(job.episode)
        if spec.external_distribution is not None:
            try:
                verify_external_artifact(spec)
            except (ValueError, TypeError) as exc:
                return f"{spec.name} needs the {spec.external_distribution} wheel at {job.episode.env.version}: {exc}"
        return None
    from reliquary.corpus.job import is_signed_sandbox
    from reliquary.environment.agentic_swe import episode_support_refusal, sandbox_support_refusal

    return episode_support_refusal(job.episode, need_verifiers=True) or (
        sandbox_support_refusal(job.episode, need_bridge=True) if is_signed_sandbox(job) else None)


@dataclass(frozen=True)
class Limits:
    """What the operator lets this miner take."""

    models: frozenset[str] = frozenset()
    """``repo`` or ``repo@revision`` entries; empty: any model."""
    max_context: int | None = None
    """The longest episode (``episode.max_total_tokens``) this miner's card holds; None: any."""

    def refusal(self, job) -> str | None:
        if self.models:
            repo = str(job.checkpoint_repo)
            if repo not in self.models and f"{repo}@{job.checkpoint_revision}" not in self.models:
                return f"its model {repo}@{job.checkpoint_revision} is not one of --model"
        if self.max_context is not None and job.episode is not None:
            needs = int(job.episode.max_total_tokens)
            if needs > self.max_context:
                return f"its episodes take up to {needs} tokens, over --max-context {self.max_context}"
        return None


class Child(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


@dataclass
class _Running:
    job_id: str
    slot: int
    child: Child
    started: float
    model: str | None
    stale: int = 0
    # Since when its order has been paused (or starting), while it is.
    paused_since: float | None = None


@dataclass
class AgenticFollower:
    """The loop's state. ``list_jobs`` returns the validator's `/corpus/jobs` document (None: it lists no
    orders); ``read_manifest(job_id)`` the order's raw manifest; ``launch(job_id, slot, validator_hotkey)`` starts
    its child. ``serve_refusal`` and ``parse`` are the checks (fakes in tests)."""

    list_jobs: Callable[[], Mapping | None]
    read_manifest: Callable[[str], Mapping]
    launch: Callable[[str, int, str | None], Child]
    fetch: Callable[[str], None]
    max_jobs: int = 1
    limits: Limits = field(default_factory=Limits)
    validator_hotkey: str | None = None
    serve_refusal: Callable[[Any], str | None] = serve_refusal
    # ``fits(job, slot)``: why the slot's share of its card cannot hold the order's model, or None.
    fits: Callable[[Any, int], str | None] | None = None
    parse: Callable[[Mapping], Any] | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    running: dict[int, _Running] = field(default_factory=dict)
    # job_id -> why this miner does not take it (kept: an install does not change while it runs).
    refused: dict[str, str] = field(default_factory=dict)
    # Parsed manifests by job id.
    jobs: dict[str, Any] = field(default_factory=dict)
    # Failed orders: their failures in a row, and when they may be tried again.
    failures: dict[str, int] = field(default_factory=dict)
    retry_at: dict[str, float] = field(default_factory=dict)
    fetched: set[str] = field(default_factory=set)
    warned: set = field(default_factory=set)
    # Orders whose child ended cleanly (complete): never started again.
    done: set[str] = field(default_factory=set)
    loaded: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.max_jobs < 1:
            raise ValueError("max_jobs must be at least 1")
        if self.parse is None:
            from reliquary.corpus.job import parse_job

            self.parse = parse_job

    # -- one order -----------------------------------------------------------

    def _job(self, job_id: str):
        """The order's parsed manifest, or None when it is refused (and why, in ``refused``)."""
        if job_id in self.refused:
            return None
        if job_id in self.jobs:
            return self.jobs[job_id]
        try:
            job = self.parse(self.read_manifest(job_id))
        except Exception as exc:  # noqa: BLE001 - one unreadable order never stops the loop
            logger.warning("order %s: its manifest is unreadable (%r); retried next read", job_id, exc)
            return None
        if job.episode is None:
            self.refused[job_id] = "single-turn"
            return None
        why = self.limits.refusal(job) or self.serve_refusal(job)
        if why:
            self.refused[job_id] = why
            logger.info("order %s not taken: %s", job_id, why)
            return None
        self.jobs[job_id] = job
        return job

    def candidates(self, orders: Sequence[Mapping], hotkey: str | None) -> list[Candidate]:
        from reliquary.corpus.job import is_signed_sandbox

        busy = {r.job_id for r in self.running.values()}
        now = self.clock()
        found = []
        for order in orders:
            job_id = str(order.get("job_id") or "")
            if (not job_id or order.get("state") != "open" or job_id in busy or job_id in self.done
                    or self.retry_at.get(job_id, 0.0) > now):
                continue
            job = self._job(job_id)
            if job is None:
                continue
            signed = is_signed_sandbox(job)
            if signed and not hotkey:
                if job_id not in self.warned:
                    self.warned.add(job_id)
                    logger.warning("order %s is a signed-sandbox one and no validator hotkey is known (the "
                                   "validator names none, or not --validator-hotkey's): not mined", job_id)
                continue
            found.append(Candidate(job_id, model_of(order) or f"{job.checkpoint_repo}@{job.checkpoint_revision}",
                                   order_pay(order), leases_free(order), signed))
        return rank(found, loaded=frozenset(self.loaded))

    # -- the children --------------------------------------------------------

    def _reap(self, orders: Sequence[Mapping]) -> None:
        """Free the slots whose child ended; stop those whose order left the list, or stayed paused too long. A
        draining order's child is never stopped: its running episodes are still paid, and it exits on its own."""
        states = {str(o.get("job_id")): o.get("state") for o in orders}
        now = self.clock()
        for slot, run in list(self.running.items()):
            code = run.child.poll()
            stopped = None
            if code is None:
                state = states.get(run.job_id)
                if state is None:
                    run.stale += 1
                    if run.stale >= STALE_READS:
                        stopped = "no longer listed"
                elif state in ("paused", "starting"):
                    run.stale = 0
                    run.paused_since = now if run.paused_since is None else run.paused_since
                    if now - run.paused_since >= PAUSED_STOP_SECONDS:
                        stopped = f"{state} for {round(now - run.paused_since)} s"
                else:                      # open, draining (or a state this build does not know): keep mining
                    run.stale, run.paused_since = 0, None
                if stopped is None:
                    continue
                logger.info("order %s is %s: stopping its miner", run.job_id, stopped)
                code = self._stop(run.child)
            del self.running[slot]
            if run.model:
                self.loaded.add(run.model)
            if stopped is not None:
                logger.info("order %s stopped (exit %s)", run.job_id, code)
            elif code == 0:
                # Its job complete (or its --episodes done): not started again while the list still shows it.
                self.done.add(run.job_id)
                self.failures.pop(run.job_id, None)
                logger.info("order %s ended", run.job_id)
            else:
                failures = self.failures[run.job_id] = self.failures.get(run.job_id, 0) + 1
                wait = min(BACKOFF_MAX_SECONDS, BACKOFF_SECONDS * 2 ** (failures - 1))
                self.retry_at[run.job_id] = now + wait
                logger.warning("order %s failed (exit %s, %d in a row): retried in %d s", run.job_id, code,
                               failures, round(wait))

    @staticmethod
    def _stop(child: Child) -> int:
        child.terminate()
        try:
            return child.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()
            return child.wait()

    def stop_all(self) -> None:
        """Stop every child at once (SIGTERM to all, then one shared grace), as on the follower's own stop."""
        live = [run.child for run in self.running.values() if run.child.poll() is None]
        for child in live:
            child.terminate()
        deadline = self.clock() + STOP_GRACE_SECONDS
        for child in live:
            try:
                child.wait(timeout=max(0.0, deadline - self.clock()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        self.running.clear()

    # -- the loop ------------------------------------------------------------

    def step(self) -> bool:
        """One read of the list: reap, then fill free slots. False when the validator lists no orders."""
        try:
            document = self.list_jobs()
        except Exception:
            logger.warning("the order list is unreadable; retrying", exc_info=True)
            return True
        if document is None or document.get("orders") is None:
            logger.warning("the validator lists no orders (an older validator): pass --job-id")
            return False
        orders = list(document["orders"])
        hotkey = self._hotkey(document.get("sandbox_hotkey"))
        self._reap(orders)
        free = [slot for slot in range(self.max_jobs) if slot not in self.running]
        chosen = []
        ranked = self.candidates(orders, hotkey)
        for slot in free:
            pick = next((c for c in ranked if c not in chosen and self._fits(c, slot)), None)
            if pick is not None:
                chosen.append(pick)
        for slot, candidate in zip(free, chosen):
            logger.info("mining order %s in slot %d (%s, %.3f alpha/M tokens%s)", candidate.job_id, slot,
                        candidate.model, candidate.pay, "" if candidate.leases_free else ", no lease left now")
            child = self.launch(candidate.job_id, slot, hotkey if candidate.signed else None)
            self.running[slot] = _Running(candidate.job_id, slot, child, self.clock(), candidate.model)
            if candidate.model:
                self.loaded.add(candidate.model)
        if not self.running and not chosen:
            self._fetch_ahead(orders)
        return True

    def _hotkey(self, served: str | None) -> str | None:
        """``--validator-hotkey`` wins; one the validator contradicts signs nothing (its requests would be
        refused): None, so no signed-sandbox order is started."""
        if not self.validator_hotkey:
            return served or None
        if served and served != self.validator_hotkey:
            if ("hotkey", served) not in self.warned:
                self.warned.add(("hotkey", served))
                logger.error("--validator-hotkey %s is not the hotkey this validator names (%s): no signed-sandbox "
                             "order is mined; fix or drop --validator-hotkey", self.validator_hotkey, served)
            return None
        return self.validator_hotkey

    def _fits(self, candidate: Candidate, slot: int) -> bool:
        if self.fits is None:
            return True
        why = self.fits(self.jobs.get(candidate.job_id), slot)
        if why and (candidate.job_id, slot) not in self.warned:
            self.warned.add((candidate.job_id, slot))
            logger.warning("order %s not started in slot %d: %s", candidate.job_id, slot, why)
        return why is None

    def _fetch_ahead(self, orders: Sequence[Mapping]) -> None:
        for order in orders:
            model = model_of(order)
            if order.get("state") != "starting" or model is None or model in self.fetched:
                continue
            job_id = str(order.get("job_id"))
            if self._job(job_id) is None:
                continue
            logger.info("no agentic order open; fetching %s for %s, starting", model, job_id)
            try:
                self.fetch(model)
                self.fetched.add(model)
                self.loaded.add(model)
            except Exception:
                logger.warning("fetching %s failed", model, exc_info=True)
            return

    def run(self, rounds: int | None = None) -> None:
        done = 0
        try:
            while rounds is None or done < rounds:
                done += 1
                if not self.step():
                    return
                self.sleep(IDLE_SECONDS)
        finally:
            self.stop_all()


class FollowerTerminated(Exception):
    """SIGTERM reached the follower (pm2, systemd): its children are stopped before it exits."""


def run_until_terminated(follower: AgenticFollower, rounds: int | None = None) -> None:
    """``follower.run``, with SIGTERM turned into a clean stop: every child gets its own SIGTERM and grace
    (``stop_all``) before the follower exits."""
    import signal

    def on_terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)      # a second one never interrupts the children's stop
        raise FollowerTerminated()

    previous = signal.signal(signal.SIGTERM, on_terminate)
    try:
        follower.run(rounds)
    except FollowerTerminated:
        logger.warning("SIGTERM: the order follower stopped its children")
    finally:
        signal.signal(signal.SIGTERM, previous)


# -- the real children -------------------------------------------------------


def child_command(job_id: str, argv: Sequence[str], *, validator_hotkey: str | None, port: int,
                  concurrency: int, gpu_memory_utilization: float | None) -> list[str]:
    """This CLI mining ``job_id`` with the parent's options, on its slot's port and share."""
    command = [sys.executable, "-m", "reliquary.cli.main", "corpus", "mine-agentic", *argv,
               "--job-id", job_id, "--port", str(port), "--concurrency", str(concurrency)]
    if validator_hotkey:
        command += ["--validator-hotkey", validator_hotkey]
    if gpu_memory_utilization is not None:
        command += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    return command


def slot_settings(slot: int, *, max_jobs: int, port: int, concurrency: int, gpus: Sequence[str],
                  gpu_memory_utilization: float | None) -> tuple[int, int, float | None, str | None]:
    """A slot's ``(port, concurrency, gpu_memory_utilization, CUDA_VISIBLE_DEVICES)``. Each slot has its own
    loopback port and an even share of ``concurrency``. Slots go round-robin over ``gpus`` (or the one visible
    card); the slots sharing a card split its memory (``gpu_memory_utilization``, vLLM's 0.9 when not given)."""
    share = max(1, concurrency // max_jobs)
    cards = len(gpus) or 1
    sharing = len(range(slot % cards, max_jobs, cards))
    memory = gpu_memory_utilization
    if sharing > 1:
        memory = round((0.9 if gpu_memory_utilization is None else gpu_memory_utilization) / sharing, 3)
    return (port + slot if max_jobs > 1 else port), share, memory, (gpus[slot % cards] if gpus else None)


# What a slot's share of its card must hold beyond the weights: activations, CUDA graphs and some KV cache.
MEMORY_HEADROOM_BYTES = 4 << 30
MEMORY_WEIGHT_FACTOR = 1.1


def model_weight_bytes(repo: str, revision: str) -> int | None:
    """The size of a model's safetensors weights, from the Hub's file metadata; None when unknown."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        sizes = [f.size for f in (info.siblings or ()) if f.rfilename.endswith(".safetensors") and f.size]
        return int(sum(sizes)) or None
    except Exception:  # noqa: BLE001 - unknown: the child's own vLLM start decides
        logger.info("the weight size of %s@%s is unknown", repo, revision, exc_info=True)
        return None


def card_memory_bytes(device: str | None) -> int | None:
    """The total memory of a CUDA card (``device`` as CUDA_VISIBLE_DEVICES names it; None: the first visible one),
    from nvidia-smi; None when unknown."""
    if shutil.which("nvidia-smi") is None:
        return None
    command = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    if device is not None:
        command += ["-i", str(device)]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout
        return int(float(out.split()[0])) << 20
    except Exception:  # noqa: BLE001
        return None


def memory_fit(*, max_jobs: int, gpus: Sequence[str], gpu_memory_utilization: float | None,
               weight_bytes: Callable[[str, str], int | None] = model_weight_bytes,
               card_bytes: Callable[[str | None], int | None] = card_memory_bytes):
    """``fits(job, slot)`` for the follower: why a slot's share of its card cannot hold the order's model (weights
    plus ``MEMORY_HEADROOM_BYTES``), or None (it fits, or a size is unknown)."""
    weights: dict[tuple[str, str], int | None] = {}
    cards: dict[str | None, int | None] = {}

    def fits(job, slot: int) -> str | None:
        if job is None:
            return None
        _, _, memory, device = slot_settings(slot, max_jobs=max_jobs, port=0, concurrency=max_jobs, gpus=gpus,
                                             gpu_memory_utilization=gpu_memory_utilization)
        fraction = 0.9 if memory is None else memory
        model = (str(job.checkpoint_repo), str(job.checkpoint_revision))
        if model not in weights:
            weights[model] = weight_bytes(*model)
        if device not in cards:
            cards[device] = card_bytes(device)
        size, total = weights[model], cards[device]
        if size is None or total is None:
            return None
        need = size * MEMORY_WEIGHT_FACTOR + MEMORY_HEADROOM_BYTES
        if need <= fraction * total:
            return None
        return (f"{model[0]} needs ~{math.ceil(need / 2**30)} GiB, the slot has {fraction:.2f} of a "
                f"{total / 2**30:.0f} GiB card: lower --max-jobs or give --gpus")

    return fits


def popen_launcher(argv: Sequence[str], *, max_jobs: int, port: int, concurrency: int, gpus: Sequence[str],
                   gpu_memory_utilization: float | None) -> Callable[[str, int, str | None], Child]:
    def launch(job_id: str, slot: int, validator_hotkey: str | None) -> Child:
        slot_port, share, memory, device = slot_settings(
            slot, max_jobs=max_jobs, port=port, concurrency=concurrency, gpus=gpus,
            gpu_memory_utilization=gpu_memory_utilization)
        env = child_env()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        return subprocess.Popen(child_command(job_id, argv, validator_hotkey=validator_hotkey, port=slot_port,
                                              concurrency=share, gpu_memory_utilization=memory), env=env)

    return launch


def http_listing(validator_url: str):
    """``(list_jobs, read_manifest)`` over the validator's HTTP routes."""
    import httpx

    http = httpx.Client(base_url=validator_url, timeout=60.0)

    def list_jobs() -> Mapping | None:
        response = http.get("/corpus/jobs")
        response.raise_for_status()
        return response.json()

    def read_manifest(job_id: str) -> Mapping:
        from reliquary.miner.corpus_miner import HttpCorpusClient

        return HttpCorpusClient(http, job_id=job_id).job()

    return list_jobs, read_manifest


def served_sandbox_hotkey(validator_url: str) -> str | None:
    """The hotkey the validator names for its sandbox sessions (``/corpus/jobs``'s ``sandbox_hotkey``), or None."""
    try:
        document = http_listing(validator_url)[0]() or {}
    except Exception:  # noqa: BLE001 - the flag is then required
        logger.warning("the order list of %s is unreadable", validator_url, exc_info=True)
        return None
    hotkey = document.get("sandbox_hotkey")
    return str(hotkey) if hotkey else None
