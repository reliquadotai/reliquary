"""The async RL submission path (rl-async-design.md §4.4-4.5, §4.7, §4.9; implementation map §1.4-1.8, P3).

``RlSubmissions.submit`` takes a completed lease's rollouts (completion tokens only, every index):

1. ``LeaseBook.check_submission``: the lease exists, is the caller's, is open and not expired at arrival, and the
   hotkey is not banned. Nothing is graded for a lease that cannot be submitted.
2. ``GroupGrader.grade``: the validator renders the prompt, checks every rollout's tokens, length and termination
   (``admission._classify_termination``, the window path's predicate) and grades the completions with the
   environment's own scorer (``EnvironmentSpec.score_many``). Rewards never come from the miner. A grader outage
   (``GraderInfrastructureError``) raises and leaves the lease open: the miner resends. In the service the scorer
   runs in a per-env process pool under the window path's wall budget (``PooledScorer``): a group that overruns
   it, or a worker that dies, is that outage (the window path's ``worker_dropped``: resendable, never the
   miner's fault), and the pool is replaced.
3. One transaction on the lease book's file: the lease closes (whatever the outcome; there is no release), the
   group is stored in ``rl_groups`` and:
   * refused for integrity (bad termination, tampered force span, out-of-vocab or over-long tokens):
     ``rejected``, no observation (decision A), and the quota is halved as for an expiry (malformed or forged
     work is no delivery). A group the env's scorer raised on (``reward_error``) is rejected too, but it is the
     validator's failure, not the miner's: its slot is freed without the halving (``GRADER_FAULTS``);
   * otherwise ``classify_group`` gives the verdict, ``LeaseBook.record_outcome`` records the scan and moves the
     prompt, and the quota frees the lease's slot (it credits the delivery only once the group is paid);
   * in zone: the group (tokens, lease nonce, k, arrival, hotkey) goes to the ``TrainingBuffer``. It is verified
     in the trainer's forward (§4.7), which reports back through ``report_training``;
   * out of zone: the audit draw, made after submission with the validator's own randomness (``SystemRandom``):
     a hotkey in probation: every first scan (it is paid) and ``probation_repeat_audit_rate`` of its other out-of-zone
     groups (they earn nothing; 2026-10-27: auditing them all buried the first scans past ``audit_max_lag``), else
     ``audit_rate`` (15 %). A drawn group keeps its tokens in
     ``rl_audits`` until ``report_audit``. An out-of-zone FIRST scan is payable (25 %, §4.5) at once when not
     drawn, after a passed audit when drawn, and only with at least ``first_scan_min_tokens`` model tokens (a
     group of near-empty rollouts verifies nothing).

Audit regime (decisions doc §C as amended 10-09, the design's §4.4): probation = fewer than
``probation_groups`` (100) passed out-of-zone audits, or fewer than ``SERVICE_REPROBATION_PASSES`` (20) passed
audits since the hotkey's last failure of any class. A failure is never sanctioned (user, 2026-10-17): the group
is unpaid and the hotkey back in probation, nothing more. Honest drift fails groups too, so a verdict cannot tell
a forger from a lagging miner; a forger gains nothing but burned GPU time. The audit's lag is kept in
``rl_audits``. An inconclusive audit (``passed=None``: no model, or queued past ``audit_max_lag`` checkpoints) is
unpaid. A ``suspect`` inconclusive (the audit
abstained on too few drawn positions, or its forward raised ``AUDIT_MAX_ATTEMPTS`` times, counted in
``rl_audits``) is logged as an ``inconclusive`` verdict: never a pass, and it re-opens probation like a failure.
``blacklist`` is the operator's ban (no lease, no submission, unsettled pay forfeited).

Interfaces left to later phases: the proof forward of an audit (``ProofForward``, implemented by
``rl_audit.audit_forward``: per-rollout match against the latest checkpoint with the training floor), the trainer's report
(``report_training``, P4), the money (``rl_events``: pay eligibility and the operator's bans, read by the P5 ledger), and
publication (``rl_prompt_scans`` is the public observation log; TODO: the R2 segment writer of decision D reads
it by ``seq``; ``rl_groups`` is the private store with hotkeys).
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
import multiprocessing.util
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    RL_AUDIT_MAX_LAG,
    RL_FIRST_SCAN_MIN_TOKENS,
    SERVICE_REPROBATION_PASSES,
    max_new_tokens_for_environment,
)
from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.environment.registry import get_environment_spec
from reliquary.protocol.rl_submit import GROUP_ROLLOUTS
from reliquary.services.exploration import AUDIT_FAILURE_CLASSES
from reliquary.validator.rl_buffer import SubmittedGroup, TrainingBuffer
from reliquary.validator.rl_leases import IN_ZONE, LeaseBook, classify_group

logger = logging.getLogger(__name__)

REJECTED = "rejected"
# Rejections that are the validator's failure to grade, never the miner's: the lease closes (the group is not graded
# again) but the quota is not halved.
GRADER_FAULTS = frozenset({"reward_error"})
# Episode envs a lease takes: signed by the sandbox (``rl_episode``) or replayed by the validator (``rl_suite_episodes``).
EPISODE_MODES = frozenset({"signed_episode", "episode"})
AUDIT_MAX_ATTEMPTS = 3        # raising audit forwards of one group before it is concluded inconclusive
NOT_DRAWN, PENDING, PASSED, FAILED, INCONCLUSIVE = "not_drawn", "pending", "passed", "failed", "inconclusive"
# rl_events kinds: pay eligibility (read by the P5 ledger) and the operator's bans
FIRST_SCAN, FIRST_SCAN_FORFEITED, TRAINED, UNTRAINED, BAN = (
    "first_scan", "first_scan_forfeited", "trained", "untrained", "ban")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_groups(seq INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT UNIQUE NOT NULL,
    hotkey TEXT NOT NULL, environment TEXT NOT NULL, prompt_idx INTEGER NOT NULL, k INTEGER NOT NULL,
    nonce TEXT NOT NULL, issued_at REAL NOT NULL, arrived_at REAL NOT NULL, verdict TEXT NOT NULL, reason TEXT,
    rewards TEXT, truncated TEXT, completion_tokens INTEGER NOT NULL, first_scan INTEGER NOT NULL,
    scan_seq INTEGER, audit TEXT, training TEXT);
CREATE INDEX IF NOT EXISTS rl_groups_hotkey ON rl_groups(hotkey, seq);
CREATE TABLE IF NOT EXISTS rl_audits(lease_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, forced INTEGER NOT NULL,
    queued_at REAL NOT NULL, body BLOB, attempts INTEGER NOT NULL DEFAULT 0, lag INTEGER);
CREATE TABLE IF NOT EXISTS rl_verdicts(seq INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT NOT NULL,
    hotkey TEXT NOT NULL, source TEXT NOT NULL, verdict TEXT NOT NULL, failure_class TEXT, forced INTEGER NOT NULL,
    at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rl_verdicts_hotkey ON rl_verdicts(hotkey, seq);
CREATE TABLE IF NOT EXISTS rl_events(seq INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT NOT NULL,
    hotkey TEXT NOT NULL, kind TEXT NOT NULL, at REAL NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rl_group_instances(lease_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL,
    instance_id TEXT NOT NULL, at REAL NOT NULL);
"""


@dataclass(frozen=True)
class Rollout:
    """One submitted rollout: its completion token ids and its BFT force span (completion-relative) or None.
    An episode's rollout also carries ``episode`` (``{spans, stop, transcript}`` as sent, ``protocol/rl_submit``);
    ``spans`` (its model spans, completion-relative) is read from it strictly (ValueError)."""
    tokens: tuple[int, ...]
    force_span: tuple[int, int] | None = None
    episode: Mapping[str, Any] | None = None
    spans: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.episode is not None and self.spans is None:
            from reliquary.protocol.rl_submit import episode_spans

            object.__setattr__(self, "spans", episode_spans(self.episode, len(self.tokens)))


@dataclass(frozen=True)
class Graded:
    prompt_tokens: tuple[int, ...]
    rewards: tuple[float, ...] = ()
    truncated: tuple[bool, ...] = ()
    reject: str | None = None
    spans: tuple[tuple[tuple[int, int], ...], ...] | None = None       # episodes: each rollout's model spans


@dataclass(frozen=True)
class AuditResult:
    """The proof forward's verdict on one group: ``passed`` None = inconclusive (no sanction); ``suspect``: the group
    itself made it inconclusive (no pass, the hotkey stays in probation)."""
    passed: bool | None
    failure_class: str | None = None
    rollout_match: tuple[float, ...] = ()
    suspect: bool = False


ProofForward = Callable[[SubmittedGroup], AuditResult]


@dataclass(frozen=True)
class SubmitConfig:
    audit_rate: float = 0.15
    probation_repeat_audit_rate: float = 1.0   # share of a probation hotkey's repeat out-of-zone scans audited
    probation_groups: int = 100
    reprobation_passes: int = SERVICE_REPROBATION_PASSES
    first_scan_fraction: float = 0.25
    first_scan_min_tokens: int = RL_FIRST_SCAN_MIN_TOKENS
    rollouts: int = GROUP_ROLLOUTS
    audit_max_lag: int = RL_AUDIT_MAX_LAG

    def __post_init__(self) -> None:
        if not (0.0 <= self.audit_rate <= 1.0 and 0.0 <= self.first_scan_fraction <= 1.0
                and 0.0 <= self.probation_repeat_audit_rate <= 1.0):
            raise ValueError("audit rate and first-scan fraction are fractions")
        if min(self.probation_groups, self.reprobation_passes, self.first_scan_min_tokens, self.audit_max_lag) < 0:
            raise ValueError("audit counts must be non-negative")
        if self.rollouts < 2:
            raise ValueError("a group has at least 2 rollouts")


class GroupGrader:
    """Grades single-turn groups as admission does, from completion tokens only (§4.4). ``environments`` maps an
    env name to its loaded environment (``get_problem`` and its reward materializer); ``tokenizer`` renders,
    encodes and decodes (the admission worker's). ``episodes`` maps each episode env to its checker, graded by
    ``grade_episodes``: a signed-episode env's ``rl_episode.LeaseEpisodeChecker`` (§4.10), a replayed env's
    ``rl_suite_episodes.LeaseSuiteChecker``."""

    def __init__(self, environments: Mapping[str, Any], tokenizer: Any, *, eos_token_ids: Sequence[int],
                 think_close_ids: Sequence[int] = (), canonical_force_ids: Sequence[int] = (),
                 vocab_size: int | None = None, episodes: Mapping[str, Any] | None = None,
                 scorer: Callable[[str, Any, Sequence[str], Any], Sequence[float]] | None = None) -> None:
        from reliquary.validator.admission import AdmissionContext

        self.episodes = dict(episodes or {})
        for name in environments:
            if get_environment_spec(name).interaction_mode != "single_turn":
                raise ValueError(f"{name}: a single-turn grader only; episode envs go in ``episodes``")
        for name in self.episodes:
            if get_environment_spec(name).interaction_mode not in EPISODE_MODES:
                raise ValueError(f"{name}: not an episode environment")
        if not eos_token_ids:
            raise ValueError("the EOS token ids are required to judge termination")
        self.eos_token_ids = frozenset(int(t) for t in eos_token_ids)
        self.environments = dict(environments)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        # (env, problem, texts, materials) -> rewards; in process unless the service gives a ``PooledScorer``
        self.scorer = scorer or score_group
        self._contexts = {name: AdmissionContext(
            randomness="", environment=name, vocab_size=vocab_size, max_sequence_length=MAX_NEW_TOKENS_PROTOCOL_CAP,
            eos_token_ids=tuple(sorted(eos_token_ids)), canonical_force_ids=tuple(canonical_force_ids),
            think_close_ids=tuple(think_close_ids), bootstrap=False, enforce_envelope_signature=False,
            enforce_legacy_merkle=False) for name in environments}

    def add_episodes(self, checkers: Mapping[str, Any]) -> None:
        """The episode envs' checkers, once the sandbox side that verifies their transcripts is built."""
        for name in checkers:
            if name in self.environments or get_environment_spec(name).interaction_mode not in EPISODE_MODES:
                raise ValueError(f"{name}: not an episode environment")
        self.episodes.update(checkers)

    def grade_episodes(self, lease, rollouts: Sequence[Rollout], *, received: float) -> Graded:
        """A lease's episodes (model spans; signed transcripts or replayed worlds): its env checker's ``grade``.
        Tokens are checked against the vocabulary here, as for a single-turn group."""
        checker = self.episodes.get(lease.environment)
        if checker is None:
            return Graded((), reject="episode_schema")
        if self.vocab_size is not None and any(t >= self.vocab_size for r in rollouts for t in r.tokens):
            return Graded((), reject="token_out_of_vocab")
        return checker.grade(lease, rollouts, received=received)

    def grade(self, environment: str, prompt_idx: int, rollouts: Sequence[Rollout]) -> Graded:
        from reliquary.validator.admission import _classify_termination, _force_span_valid
        from reliquary.validator.batcher import admission_problem_materials

        if environment in self.episodes or any(r.episode is not None for r in rollouts):
            return Graded((), reject="episode_schema")          # an episode env's group is all episodes

        materials = admission_problem_materials(self.environments[environment], self.tokenizer, prompt_idx)
        encoded = self.tokenizer.encode(materials.rendered_prompt, add_special_tokens=False)
        prompt = tuple(int(t) for t in getattr(encoded, "ids", encoded))
        cap = max_new_tokens_for_environment(environment)
        truncated = []
        for rollout in rollouts:
            tokens = rollout.tokens
            if len(tokens) > cap:
                return Graded(prompt, reject="too_long")
            if self.vocab_size is not None and any(t >= self.vocab_size for t in tokens):
                return Graded(prompt, reject="token_out_of_vocab")
            span = rollout.force_span
            meta = {"prompt_length": len(prompt), "completion_length": len(tokens), "forced": span is not None,
                    "force_span": None if span is None else [len(prompt) + span[0], len(prompt) + span[1]]}
            context = self._contexts[environment]
            # a force span is a claim: it must be the BFT one, in an env that forces, whatever the termination
            if span is not None and (get_environment_spec(environment).termination_policy != "math_bft"
                                     or not _force_span_valid([*prompt, *tokens], meta, context)):
                return Graded(prompt, reject="tampered")
            kind = _classify_termination(SimpleNamespace(commit={"tokens": [*prompt, *tokens], "rollout": meta}),
                                         context)
            if kind not in ("ok", "truncated"):
                return Graded(prompt, reject=kind)
            truncated.append(kind == "truncated")
        # The answer is graded without its terminator: an extractor that closes a fenced block on a line of its
        # own, or an "ends with" instruction, must not see "<|im_end|>" glued to the last line.
        eos = {int(t) for t in self._contexts[environment].eos_token_ids}
        texts = [str(self.tokenizer.decode(list(r.tokens[:-1] if r.tokens and int(r.tokens[-1]) in eos
                                                else r.tokens), skip_special_tokens=False)) for r in rollouts]
        try:
            rewards = self.scorer(environment, materials.problem, texts, materials.effective_reward_materials)
        except GraderInfrastructureError:
            raise
        except Exception:
            logger.exception("reward_error: %s prompt %d: the scorer raised", environment, prompt_idx)
            return Graded(prompt, reject="reward_error")
        try:
            rewards = tuple(float(r) for r in rewards)
        except (TypeError, ValueError):
            rewards = ()
        if len(rewards) != len(rollouts) or not all(math.isfinite(r) for r in rewards):
            logger.error("reward_error: %s prompt %d: %d rewards for %d rollouts (%s)", environment, prompt_idx,
                         len(rewards), len(rollouts), list(rewards)[:16])
            return Graded(prompt, reject="reward_error")
        # A rollout cut at the cap gave no answer: it scores 0, whatever an answer-like string in its unfinished
        # reasoning would grade (the failure reward, never masked: the group trains on it).
        rewards = tuple(0.0 if cut else r for r, cut in zip(rewards, truncated))
        return Graded(prompt, rewards, tuple(truncated))


def score_group(environment: str, problem: Any, texts: Sequence[str], materials: Any) -> list[float]:
    """One group's rewards by its environment's own scorer (module level: it runs in a pool worker too)."""
    return [float(r) for r in get_environment_spec(environment).score_many(problem, list(texts), materials)]


def initialize_grading_worker() -> None:
    """A grading worker as the window path's admission child: CPU only, below the control plane's priority."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.nice(5)
    except OSError:
        pass


def _score_in_worker(function: Callable[..., Any], *args: Any) -> Any:
    """``function(*args)`` in a grading worker. A grader that imported bittensor started bittensor's logging
    listener in the worker, blocked on its queue: when the pool retires the worker (``max_tasks``), the queue closes
    under it and its thread dies on EOFError. Once it exists, the listener is stopped (its sentinel, its thread
    joined) at the worker's exit, before the queue closes. Importing bittensor up front would put seconds of
    start-up into a group's wall budget."""
    try:
        return function(*args)
    finally:
        if "bittensor" in sys.modules:
            _stop_bittensor_listener_at_exit()


def _stop_bittensor_listener_at_exit() -> None:
    global _LISTENER_FINALIZED
    if _LISTENER_FINALIZED:
        return
    try:
        listener = getattr(getattr(sys.modules["bittensor"], "logging", None), "_listener", None)
        if listener is None or getattr(listener, "_thread", None) is None:
            return

        def drain() -> None:
            # QueueListener.stop without clearing ``_thread``: bittensor's own atexit stop runs later and joins it
            # again (a finished thread joins at once; a cleared one raises).
            thread = listener._thread
            if thread is not None and thread.is_alive():
                listener.enqueue_sentinel()
                thread.join(timeout=5.0)

        # multiprocessing runs its finalizers by priority at a worker's exit, the queue's own (closing) after.
        multiprocessing.util.Finalize(listener, drain, exitpriority=100)
        _LISTENER_FINALIZED = True
    except Exception:                       # logging cleanup never fails a group
        logger.debug("bittensor's logging listener could not be finalized", exc_info=True)


_LISTENER_FINALIZED = False


class PooledScorer:
    """``GroupGrader``'s scorer in the service: each env scores in its own process pool (spawned, workers recycled
    after ``max_tasks`` groups), and a group gets the window path's wall budget for its resource class
    (``CODE_ADMISSION_WALL_SECONDS`` for sandboxed code, ``MATH_ADMISSION_WALL_SECONDS`` otherwise), plus
    ``run_seconds`` per program run for a stdio-program env (one run per hidden test and rollout: a 250-test
    problem is 4,000 runs). A group that overruns it, or a pool whose worker died, raises
    ``GraderInfrastructureError`` (the submission is resent, as the window path's ``worker_dropped``) and the env's
    pool is replaced, so one group never holds a worker.

    At most ``workers`` groups are in an env's pool at once (the others wait here, before their wall clock starts):
    a group never spends its budget queued behind another's, and replacing the pool never cancels a group that was
    only queued in it. Whatever the pool itself raises (a group cancelled or killed with a replaced pool, a submit
    to a pool shut down meanwhile) is the grader's outage, never a ``reward_error`` against the miner."""

    def __init__(self, environments: Sequence[str], *, workers: int = 2, max_tasks: int | None = None,
                 wall_seconds: Callable[[str], float] | None = None, function: Callable[..., Any] = score_group,
                 mp_context: str = "spawn", run_seconds: Callable[[str], float] | None = None) -> None:
        from reliquary.constants import ADMISSION_PROCESS_MAX_TASKS

        self.environments = tuple(environments)
        self.workers = int(workers)
        self.max_tasks = ADMISSION_PROCESS_MAX_TASKS if max_tasks is None else int(max_tasks)
        self.wall_seconds = wall_seconds or _wall_seconds
        self.run_seconds = run_seconds or _run_seconds
        self.function = function
        self._context = multiprocessing.get_context(mp_context)
        self._pools: dict[str, ProcessPoolExecutor] = {}
        self._lock = threading.Lock()
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self.timeouts: dict[str, int] = {}
        self.restarts: dict[str, int] = {}

    def _pool(self, environment: str) -> ProcessPoolExecutor:
        with self._lock:
            pool = self._pools.get(environment)
            if pool is None:
                pool = self._pools[environment] = ProcessPoolExecutor(
                    max_workers=self.workers, mp_context=self._context, max_tasks_per_child=self.max_tasks,
                    initializer=initialize_grading_worker)
            return pool

    def _replace(self, environment: str, pool: ProcessPoolExecutor) -> None:
        with self._lock:
            if self._pools.get(environment) is pool:
                del self._pools[environment]
                self.restarts[environment] = self.restarts.get(environment, 0) + 1
        for process in tuple((getattr(pool, "_processes", None) or {}).values()):
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)

    def _slot(self, environment: str) -> threading.BoundedSemaphore:
        with self._lock:
            slot = self._slots.get(environment)
            if slot is None:
                slot = self._slots[environment] = threading.BoundedSemaphore(max(1, self.workers))
            return slot

    def budget(self, environment: str, texts: Sequence[str], materials: Any) -> float:
        """A group's wall budget: the env's, plus ``run_seconds`` per program run of a stdio-program group."""
        wall = float(self.wall_seconds(environment))
        per_run = float(self.run_seconds(environment))
        if per_run > 0 and isinstance(materials, Sequence) and not isinstance(materials, (str, bytes)):
            from reliquary.constants import STDIO_ADMISSION_MAX_WALL_SECONDS

            wall = min(max(wall, STDIO_ADMISSION_MAX_WALL_SECONDS),
                       wall + per_run * len(materials) * len(texts))
        return max(1.0, wall)

    def __call__(self, environment: str, problem: Any, texts: Sequence[str], materials: Any) -> list[float]:
        wall = self.budget(environment, texts, materials)
        with self._slot(environment):
            pool = self._pool(environment)
            try:
                future = pool.submit(_score_in_worker, self.function, environment, problem, list(texts), materials)
            except (BrokenProcessPool, RuntimeError) as exc:       # broken, or shut down by a concurrent replace
                self._replace(environment, pool)
                raise GraderInfrastructureError(f"{environment} grading pool unavailable ({exc})") from exc
            try:
                return list(future.result(timeout=wall))
            except FutureTimeout as exc:
                self.timeouts[environment] = self.timeouts.get(environment, 0) + 1
                self._replace(environment, pool)
                raise GraderInfrastructureError(f"{environment} grading overran {wall:.0f} s") from exc
            except BrokenProcessPool as exc:
                self._replace(environment, pool)
                raise GraderInfrastructureError(f"{environment} grading worker died") from exc
            except CancelledError as exc:                          # its pool was replaced under it
                raise GraderInfrastructureError(f"{environment} grading cancelled with its pool") from exc

    def close(self) -> None:
        with self._lock:
            pools, self._pools = list(self._pools.items()), {}
        for _, pool in pools:
            pool.shutdown(wait=False, cancel_futures=True)


def _run_seconds(environment: str) -> float:
    """The wall allowance per program run of a stdio-program env (0 for any other env, or an unknown name)."""
    from reliquary.constants import STDIO_ADMISSION_SECONDS_PER_RUN
    from reliquary.environment.stdio_program import STDIO_PROGRAM_CONTRACT

    try:
        spec = get_environment_spec(environment)
    except (KeyError, ValueError):
        return 0.0
    return STDIO_ADMISSION_SECONDS_PER_RUN if spec.contract_version == STDIO_PROGRAM_CONTRACT else 0.0


def _wall_seconds(environment: str) -> float:
    from reliquary.constants import CODE_ADMISSION_WALL_SECONDS, MATH_ADMISSION_WALL_SECONDS

    sandboxed = getattr(get_environment_spec(environment), "admission_resource_class", "cpu") == "sandbox"
    return CODE_ADMISSION_WALL_SECONDS if sandboxed else MATH_ADMISSION_WALL_SECONDS


class RlSubmissions:
    def __init__(self, book: LeaseBook, grader: GroupGrader, *, config: SubmitConfig | None = None,
                 buffer: TrainingBuffer | None = None, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.book, self.grader = book, grader
        self.config = config or SubmitConfig()
        self.clock = clock
        self._rng = rng or random.SystemRandom()
        with book.lock:
            book.db.executescript(_SCHEMA)
            columns = {row[1] for row in book.db.execute("PRAGMA table_info(rl_audits)")}
            for column, kind in (("attempts", "INTEGER NOT NULL DEFAULT 0"), ("lag", "INTEGER")):   # an older file
                if column not in columns:
                    book.db.execute(f"ALTER TABLE rl_audits ADD COLUMN {column} {kind}")
            book.db.commit()
        self.buffer = buffer or TrainingBuffer(book)

    # --- submission ----------------------------------------------------------------------------------------
    def submit(self, lease_id: str, hotkey: str, rollouts: Sequence[Rollout], *,
               arrived_at: float | None = None) -> dict:
        """Grade and record one lease's group. ValueError: not the lease's ``rollouts`` rollouts (lease left
        open); LookupError / PermissionError: as ``LeaseBook.submit``; GraderInfrastructureError: retry later.
        Returns what the miner is told: verdict, rewards, refusal reason (never the audit draw).

        Idempotent per lease: a resend by the holder of a lease already graded (its answer was lost) gets the
        original answer again, flagged ``resent``, and nothing is graded or recorded twice. The first group is the
        one that counts; a resend cannot replace it."""
        if len(rollouts) != self.config.rollouts or any(not r.tokens for r in rollouts):
            raise ValueError(f"a group is exactly {self.config.rollouts} non-empty rollouts")
        earlier = self.graded_answer(lease_id, hotkey)
        if earlier is not None:
            return earlier
        arrived_at = float(self.clock()) if arrived_at is None else float(arrived_at)
        try:
            return self._submit(lease_id, hotkey, rollouts, arrived_at)
        except LookupError:
            earlier = self.graded_answer(lease_id, hotkey)      # a concurrent resend closed it first
            if earlier is not None:
                return earlier
            raise

    def hold(self, lease_id: str, hotkey: str, arrived_at: float) -> bool:
        """Called at arrival, before the group waits for grading: the lease must be the caller's and submittable
        (raises as ``submit``), and the sweep will not expire it until ``release`` (``LeaseBook.hold_submission``).
        False (nothing held) for a lease already graded: ``submit`` answers the resend."""
        if self.graded_answer(lease_id, hotkey) is not None:
            return False
        self.book.hold_submission(lease_id, hotkey, arrived_at)
        return True

    def release(self, lease_id: str) -> None:
        self.book.release_submission(lease_id)

    def graded_answer(self, lease_id: str, hotkey: str) -> dict | None:
        """The answer given for an already graded lease, or None. PermissionError: another hotkey's lease."""
        with self.book.lock:
            row = self.book.db.execute("SELECT hotkey, verdict, reason, rewards FROM rl_groups WHERE lease_id=?",
                                       (lease_id,)).fetchone()
        if row is None:
            return None
        if row[0] != hotkey:
            raise PermissionError("lease held by another hotkey")
        answer = {"lease_id": lease_id, "verdict": row[1], "rewards": [] if row[3] is None else json.loads(row[3]),
                  "resent": True}
        if row[2]:
            answer["reason"] = row[2]
        return answer

    def _submit(self, lease_id: str, hotkey: str, rollouts: Sequence[Rollout], arrived_at: float) -> dict:
        lease = self.book.check_submission(lease_id, hotkey, arrived_at)
        if lease.environment in getattr(self.grader, "episodes", {}):
            graded = self.grader.grade_episodes(lease, rollouts, received=arrived_at)
        else:
            graded = self.grader.grade(lease.environment, lease.prompt_idx, rollouts)
        if graded.reject in GRADER_FAULTS:
            logger.error("%s: lease %s of %s (%s prompt %d) closed, quota kept", graded.reject, lease_id[:12],
                         hotkey[:12], lease.environment, lease.prompt_idx)
        # Truncated rollouts count as failures inside the group (no truncation ceiling): the draw, not the miner,
        # decides where a rollout stops, and a group dropped for its truncations would teach nothing about length.
        verdict = REJECTED if graded.reject else classify_group(graded.rewards, graded.truncated,
                                                                 max_truncated=len(graded.rewards))
        group = SubmittedGroup(lease_id, hotkey, lease.environment, lease.prompt_idx, lease.checkpoint, lease.nonce,
                               arrived_at, graded.rewards, graded.truncated, graded.prompt_tokens,
                               tuple(r.tokens for r in rollouts), tuple(r.force_span for r in rollouts),
                               graded.spans)
        db = self.book.db
        tokens = sum(group.model_tokens(i) for i in range(len(rollouts)))
        with self.book.transaction():
            # a group the validator failed to grade is not the miner's fault: its slot is freed, unhalved
            self.book.submit(lease_id, hotkey, in_zone=verdict == IN_ZONE,
                             delivered=verdict != REJECTED or graded.reject in GRADER_FAULTS, now=arrived_at)
            scan = None if verdict == REJECTED else self.book.record_outcome(
                lease.environment, lease.prompt_idx, lease.checkpoint, graded.rewards, verdict, now=arrived_at)
            audit = training = None
            if verdict == IN_ZONE:
                training = "queued"
                self.buffer.push(group)
            elif scan is not None:
                forced = self._in_probation(hotkey)
                rate = ((1.0 if scan.first_scan else self.config.probation_repeat_audit_rate) if forced
                        else self.config.audit_rate)
                if rate >= 1.0 or self._rng.random() < rate:
                    audit = PENDING
                    db.execute("INSERT INTO rl_audits(lease_id, hotkey, forced, queued_at, body) VALUES(?,?,?,?,?)",
                               (lease_id, hotkey, int(forced), arrived_at, group.pack()))
                else:
                    audit = NOT_DRAWN
                    if scan.first_scan:
                        self._first_scan(lease_id, hotkey, arrived_at, NOT_DRAWN, tokens)
            db.execute("INSERT INTO rl_groups(lease_id, hotkey, environment, prompt_idx, k, nonce, issued_at, "
                       "arrived_at, verdict, reason, rewards, truncated, completion_tokens, first_scan, scan_seq, "
                       "audit, training) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (lease_id, hotkey, lease.environment, lease.prompt_idx, lease.checkpoint, lease.nonce,
                        lease.issued_at, arrived_at, verdict, graded.reject,
                        None if scan is None else json.dumps(graded.rewards),
                        None if scan is None else json.dumps(graded.truncated),
                        tokens, int(bool(scan and scan.first_scan)),
                        None if scan is None else scan.seq, audit, training))
        answer = {"lease_id": lease_id, "verdict": verdict, "rewards": list(graded.rewards)}
        if graded.reject:
            answer["reason"] = graded.reject
        return answer

    def group(self, lease_id: str) -> dict | None:
        with self.book.lock:
            cursor = self.book.db.execute("SELECT * FROM rl_groups WHERE lease_id=?", (lease_id,))
            row = cursor.fetchone()
            names = [c[0] for c in cursor.description]
        return None if row is None else dict(zip(names, row))

    def events(self, since: int = 0) -> list[dict]:
        """Pay eligibility and sanctions after cursor ``since`` (the P5 ledger's input)."""
        with self.book.lock:
            rows = self.book.db.execute("SELECT seq, lease_id, hotkey, kind, at, detail FROM rl_events WHERE seq>? "
                                        "ORDER BY seq", (int(since),)).fetchall()
        return [{"seq": s, "lease_id": l, "hotkey": h, "kind": k, "at": a, "detail": json.loads(d)}
                for s, l, h, k, a, d in rows]

    def _first_scan(self, lease_id: str, hotkey: str, at: float, audit: str, tokens: int) -> None:
        """An out-of-zone first scan: payable after ``audit`` (not drawn, or passed) with enough model tokens."""
        if audit in (NOT_DRAWN, PASSED) and tokens >= self.config.first_scan_min_tokens:
            self._event(lease_id, hotkey, FIRST_SCAN, at,
                        {"fraction": self.config.first_scan_fraction, "audited": audit == PASSED})
            self.book.credit(lease_id, now=at)
        else:
            self._event(lease_id, hotkey, FIRST_SCAN_FORFEITED, at, {"audit": audit, "tokens": tokens})

    def tag_instance(self, lease_id: str, hotkey: str, instance_id: str) -> None:
        """Keep the miner's optional, self-declared ``instance_id`` of a received group (the first one sent wins).
        Read by the dashboard's card metrics only: nothing scores or pays on it."""
        with self.book.lock:
            self.book.db.execute("INSERT OR IGNORE INTO rl_group_instances(lease_id, hotkey, instance_id, at) "
                                 "VALUES(?,?,?,?)", (lease_id, hotkey, instance_id, self.clock()))
            self.book.db.commit()

    def _event(self, lease_id: str, hotkey: str, kind: str, at: float, detail: dict) -> None:
        self.book.db.execute("INSERT INTO rl_events(lease_id, hotkey, kind, at, detail) VALUES(?,?,?,?,?)",
                             (lease_id, hotkey, kind, at, json.dumps(detail, sort_keys=True)))

    # --- audit regime --------------------------------------------------------------------------------------
    def _passes(self, hotkey: str, after: int = 0) -> int:
        return int(self.book.db.execute(
            "SELECT COUNT(*) FROM rl_verdicts WHERE hotkey=? AND source='audit' AND verdict=? AND seq>?",
            (hotkey, PASSED, after)).fetchone()[0])

    def _in_probation(self, hotkey: str) -> bool:
        if self._passes(hotkey) < self.config.probation_groups:
            return True
        last_failure = self.book.db.execute("SELECT MAX(seq) FROM rl_verdicts WHERE hotkey=? AND verdict IN (?, ?)",
                                            (hotkey, FAILED, INCONCLUSIVE)).fetchone()[0]
        return last_failure is not None and self._passes(hotkey, last_failure) < self.config.reprobation_passes

    def in_probation(self, hotkey: str) -> bool:
        with self.book.lock:
            return self._in_probation(hotkey)

    def _verdict(self, lease_id: str, hotkey: str, source: str, passed: bool, failure_class: str | None,
                 forced: bool, now: float) -> None:
        """Log one verdict (a failure with the class of the strongest failure, decided by the caller)."""
        if not passed and failure_class not in AUDIT_FAILURE_CLASSES:
            raise ValueError(f"a failure needs its class, one of {AUDIT_FAILURE_CLASSES}")
        db = self.book.db
        db.execute("INSERT INTO rl_verdicts(lease_id, hotkey, source, verdict, failure_class, forced, at) "
                   "VALUES(?,?,?,?,?,?,?)", (lease_id, hotkey, source, PASSED if passed else FAILED,
                                             None if passed else failure_class, int(forced), now))

    def blacklist(self, hotkey: str, until: float, reason: str, now: float | None = None) -> None:
        """The operator's ban: no lease and no submission until ``until``; the ledger forfeits the hotkey's
        unsettled pay and what accrues during the ban. Never decided by a verdict."""
        now = float(self.clock()) if now is None else now
        with self.book.transaction():
            self.book.ban(hotkey, float(until))
            self._event("", hotkey, BAN, now, {"reason": str(reason), "until": float(until)})

    def audit_jobs(self, limit: int) -> list[SubmittedGroup]:
        """Queued audits: hotkeys past probation before those in probation (decisions doc §C), then first scans (they
        are paid) before repeats, newest first.
        The forward runs against the latest checkpoint (§4.4): under a backlog the freshest groups are judged and the
        stale ones dropped (``drop_stale_audits``)."""
        with self.book.lock:
            rows = self.book.db.execute(
                "SELECT a.body FROM rl_audits a LEFT JOIN rl_groups g ON g.lease_id=a.lease_id WHERE a.body IS NOT "
                "NULL ORDER BY a.forced, COALESCE(g.first_scan, 0) DESC, a.queued_at DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [SubmittedGroup.unpack(body) for body, in rows]

    def report_audit(self, lease_id: str, result: AuditResult, now: float | None = None) -> None:
        """Conclude a drawn out-of-zone group (once). Passed: its first scan becomes payable. Failed: forfeited,
        the hotkey back in probation. Inconclusive: forfeited (a ``suspect`` one is logged and keeps the hotkey in
        probation). The lag is recorded. LookupError: not a pending audit."""
        now = float(self.clock()) if now is None else now
        with self.book.transaction():
            db = self.book.db
            row = db.execute("SELECT a.hotkey, a.forced, g.first_scan, g.k, g.completion_tokens FROM rl_audits a "
                             "JOIN rl_groups g ON g.lease_id=a.lease_id WHERE a.lease_id=? AND a.body IS NOT NULL",
                             (lease_id,)).fetchone()
            if row is None:
                raise LookupError("no pending audit for this lease")
            hotkey, forced, first_scan, k, tokens = row[0], bool(row[1]), bool(row[2]), int(row[3]), int(row[4])
            lag = (self.book.latest_checkpoint or 0) - k
            passed = result.passed
            state = INCONCLUSIVE if passed is None else PASSED if passed else FAILED
            if passed is not None:
                self._verdict(lease_id, hotkey, "audit", passed, result.failure_class, forced, now)
            elif result.suspect:
                db.execute("INSERT INTO rl_verdicts(lease_id, hotkey, source, verdict, failure_class, forced, at) "
                           "VALUES(?,?,?,?,?,?,?)", (lease_id, hotkey, "audit", INCONCLUSIVE, None, int(forced), now))
            db.execute("UPDATE rl_audits SET body=NULL, lag=? WHERE lease_id=?", (lag, lease_id))
            db.execute("UPDATE rl_groups SET audit=? WHERE lease_id=?", (state, lease_id))
            if first_scan:
                self._first_scan(lease_id, hotkey, now, state, tokens)

    def drop_stale_audits(self, now: float | None = None) -> int:
        """Conclude inconclusive (unpaid, no probation effect) every queued audit whose lease is more than
        ``audit_max_lag`` checkpoints behind the latest: judged on drifted weights, an honest group would fail."""
        latest = self.book.latest_checkpoint
        if latest is None:
            return 0
        with self.book.lock:
            stale = [lease for lease, in self.book.db.execute(
                "SELECT a.lease_id FROM rl_audits a JOIN rl_groups g ON g.lease_id=a.lease_id "
                "WHERE a.body IS NOT NULL AND g.k < ?", (latest - self.config.audit_max_lag,))]
        for lease in stale:
            self.report_audit(lease, AuditResult(None), now)
        if stale:
            logger.info("dropped %d audits queued past %d checkpoints", len(stale), self.config.audit_max_lag)
        return len(stale)

    def take_audits(self, limit: int) -> list[SubmittedGroup]:
        """The trainer's side of the audits (``trainer.audits``): stale ones dropped, then up to ``limit`` queued
        audits (``audit_jobs``' order); each comes back through ``report_audit``."""
        self.drop_stale_audits()
        return self.audit_jobs(limit)

    def run_audits(self, forward: ProofForward, limit: int, *, max_attempts: int = AUDIT_MAX_ATTEMPTS) -> int:
        """Drain up to ``limit`` queued audits through ``forward`` (the proof forward, the Gumbel verifier).
        A forward that raises leaves its audit queued and the others still run; after ``max_attempts`` raising
        forwards (counted in ``rl_audits``, across restarts) the audit is concluded a suspect inconclusive
        (unpaid, unsanctioned, no probation pass), so one poison group (e.g. an OOM) cannot stall the queue. Stale
        audits are dropped first (``drop_stale_audits``, not counted)."""
        self.drop_stale_audits()
        done = 0
        for group in self.audit_jobs(limit):
            try:
                result = forward(group)
            except Exception:
                with self.book.transaction():
                    self.book.db.execute("UPDATE rl_audits SET attempts=attempts+1 WHERE lease_id=?",
                                         (group.lease_id,))
                    failures = int(self.book.db.execute("SELECT attempts FROM rl_audits WHERE lease_id=?",
                                                        (group.lease_id,)).fetchone()[0])
                logger.exception("audit forward of %s failed (%d/%d)", group.lease_id[:12], failures, max_attempts)
                if failures < max_attempts:
                    continue
                result = AuditResult(None, suspect=True)
            self.report_audit(group.lease_id, result)
            done += 1
        return done

    # --- the trainer's report (P4) ------------------------------------------------------------------------
    def _rewards_hold(self, group, substitutions) -> bool:
        """The reward re-check: every rollout with suspect mismatches (``substitutions[i]``: (completion position,
        the trainer's pick), gap over ``RL_SUBSTITUTE_GAP``) is graded again with the picks there; any reward that
        moves means the tokens were chosen for the reward. Only the first position of a run of consecutive suspects
        is substituted: the picks after it were drawn under the miner's prefix. A pick that ends the rollout (an EOS
        the trainer lists whatever its gap) ends the re-graded rollout there. Episodes are not re-graded (their
        reward is the environment's record; the gap test covers them)."""
        stops = getattr(self.grader, "eos_token_ids", frozenset())
        changed = []
        for i, pairs in enumerate(substitutions or ()):
            if not pairs or group.spans is not None:
                continue
            tokens, previous = list(group.completions[i]), None
            for position, pick in sorted((int(p), int(t)) for p, t in pairs):
                if previous != position - 1 and 0 <= position < len(tokens):
                    tokens[position] = pick
                    if pick in stops:
                        tokens = tokens[:position + 1]
                        break
                previous = position
            changed.append((i, Rollout(tuple(tokens), group.force_spans[i])))
        if not changed:
            return True
        graded = self.grader.grade(group.environment, group.prompt_idx, [r for _, r in changed])
        return not graded.reject and all(abs(float(graded.rewards[k]) - float(group.rewards[i])) < 1e-9
                                         for k, (i, _) in enumerate(changed))

    def report_training(self, lease_id: str, *, rollout_match: Sequence[float], accepted: bool,
                        failure_class: str | None = None, matched_tokens: Sequence[int] | None = None,
                        substitutions: Sequence[Sequence[tuple[int, int]]] | None = None,
                        judged: bool | None = None, now: float | None = None) -> None:
        """The trainer's forward judged a buffered group: ``rollout_match`` per rollout (match rate against the
        published checkpoint, §4.6), ``accepted`` the group verdict. An accepted group is payable (P5 pays the
        matched tokens of rollouts above the floor); a rejected one is unpaid, and a ``failure_class`` logs it as a
        failed verdict (the hotkey back in probation).
        ``matched_tokens``: per rollout, the positions equal to the validator's pick (what P5 pays); without it the
        ledger uses ``floor(match x completion tokens)``. ``judged`` is telemetry only: False for a skipped report,
        True when a ratio was judged, None for an older trainer. ``substitutions``: per rollout, the suspect mismatches
        (``_rewards_hold``); an accepted group whose rewards move is reported untrained, unsanctioned
        (``reward_moved``), so it is not paid. LookupError: not a buffered group."""
        if judged is not None and not isinstance(judged, bool):
            raise ValueError("judged must be true, false or null")
        now = float(self.clock()) if now is None else now
        moved = False
        if accepted and substitutions:
            pending = self.buffer.get(lease_id)
            if pending is not None and len(substitutions) != len(pending.completions):
                raise ValueError("one substitution list per rollout")
            moved = pending is not None and not self._rewards_hold(pending, substitutions)
            accepted = accepted and not moved
        with self.book.transaction():
            group = self.buffer.get(lease_id)
            if group is None or not self.buffer._remove(lease_id):
                raise LookupError("no buffered group for this lease")
            match = [float(m) for m in rollout_match]
            if len(match) != len(group.completions) or not all(0.0 <= m <= 1.0 for m in match):
                raise ValueError("one match rate in [0, 1] per rollout")
            model = [group.model_tokens(i) for i in range(len(match))]
            if matched_tokens is not None and (len(matched_tokens) != len(match) or not all(
                    0 <= int(n) <= m for n, m in zip(matched_tokens, model))):
                raise ValueError("one matched token count in [0, model tokens] per rollout")
            self.book.db.execute("UPDATE rl_groups SET training=? WHERE lease_id=?",
                                 ("accepted" if accepted else REJECTED, lease_id))
            # an episode's tool observations are no work of the miner's model: pay counts model tokens only
            detail = {"rollout_match": match, "completion_tokens": model,
                      "checkpoint": group.checkpoint, "environment": group.environment}
            if judged is not None:
                detail["judged"] = judged
            if matched_tokens is not None:
                detail["matched_tokens"] = [int(n) for n in matched_tokens]
            if moved:
                detail["reward_moved"] = True
            if accepted:
                self._event(lease_id, group.hotkey, TRAINED, now, detail)
                self.book.credit(lease_id, now=now)
                self._verdict(lease_id, group.hotkey, "training", True, None, False, now)
            else:
                self._event(lease_id, group.hotkey, UNTRAINED, now, {**detail, "failure_class": failure_class})
                if failure_class is not None:
                    self._verdict(lease_id, group.hotkey, "training", False, failure_class, False, now)


__all__ = [
    "AUDIT_MAX_ATTEMPTS", "BAN", "FIRST_SCAN", "FIRST_SCAN_FORFEITED", "REJECTED", "TRAINED", "UNTRAINED", "AuditResult", "Graded",
    "GroupGrader", "PooledScorer", "ProofForward", "RlSubmissions", "Rollout", "SubmitConfig", "score_group",
]
