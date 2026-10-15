"""The async RL submission path (rl-async-design.md §4.4-4.5, §4.7, §4.9; implementation map §1.4-1.8, P3).

``RlSubmissions.submit`` takes a completed lease's rollouts (completion tokens only, every index):

1. ``LeaseBook.check_submission``: the lease exists, is the caller's, is open and not expired at arrival, and the
   hotkey is not banned. Nothing is graded for a lease that cannot be submitted.
2. ``GroupGrader.grade``: the validator renders the prompt, checks every rollout's tokens, length and termination
   (``admission._classify_termination``, the window path's predicate) and grades the completions with the
   environment's own scorer (``EnvironmentSpec.score_many``). Rewards never come from the miner. A grader outage
   (``GraderInfrastructureError``) raises and leaves the lease open: the miner resends.
3. One transaction on the lease book's file: the lease closes (whatever the outcome; there is no release), the
   group is stored in ``rl_groups`` and:
   * refused for integrity (bad termination, tampered force span, out-of-vocab or over-long tokens, a scorer that
     cannot grade it): ``rejected``, no observation (decision A), and the quota is halved as for an expiry
     (malformed or forged work is no delivery);
   * otherwise ``classify_group`` gives the verdict, ``LeaseBook.record_outcome`` records the scan and moves the
     prompt, and the quota counts the delivery (in or out of zone);
   * in zone: the group (tokens, lease nonce, k, arrival, hotkey) goes to the ``TrainingBuffer``. It is verified
     in the trainer's forward (§4.7), which reports back through ``report_training``;
   * out of zone: the audit draw, made after submission with the validator's own randomness (``SystemRandom``):
     every group of a hotkey in probation, else ``audit_rate`` (15 %). A drawn group keeps its tokens in
     ``rl_audits`` until ``report_audit``. An out-of-zone FIRST scan is payable (25 %, §4.5) at once when not
     drawn, after a passed audit when drawn.

Audit regime (decisions doc §C as amended 10-09, the design's §4.4): probation = fewer than
``probation_groups`` (100) passed out-of-zone audits, or fewer than ``SERVICE_REPROBATION_PASSES`` (20) passed
audits since the hotkey's last failure of any class. A ``deterministic`` failure (forgery an honest miner cannot
produce) bans for ``ban_s`` (24 h; ``LeaseBook.ban``: no lease, no submission); a ``statistical`` one re-opens
probation and bans only when more than ``SERVICE_STATISTICAL_FAIL_MAX`` (3) of the hotkey's last
``SERVICE_STATISTICAL_FAIL_WINDOW`` (20) non-forced verdicts failed statistically. A failed training verification
reported with a class sanctions the same way (``source='training'``: training verdicts count in the
recidivism window, never as probation passes; a rejection without a class, e.g. staleness, is no verdict). An
inconclusive audit (``passed=None``: the validator could not run it) is unpaid and unsanctioned.

Interfaces left to later phases: the proof forward of an audit (``ProofForward``, the Gumbel verifier on another
branch: per-rollout match against the latest checkpoint with the training floor), the trainer's report
(``report_training``, P4), the money (``rl_events``: pay eligibility and sanctions, read by the P5 ledger), and
publication (``rl_prompt_scans`` is the public observation log; TODO: the R2 segment writer of decision D reads
it by ``seq``; ``rl_groups`` is the private store with hotkeys).
"""
from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    SERVICE_REPROBATION_PASSES,
    SERVICE_STATISTICAL_FAIL_MAX,
    SERVICE_STATISTICAL_FAIL_WINDOW,
    max_new_tokens_for_environment,
)
from reliquary.environment.grader_client import GraderInfrastructureError
from reliquary.environment.registry import get_environment_spec
from reliquary.protocol.rl_submit import GROUP_ROLLOUTS
from reliquary.services.exploration import AUDIT_FAILURE_CLASSES, AUDIT_FAILURE_DETERMINISTIC, AUDIT_FAILURE_STATISTICAL
from reliquary.validator.rl_buffer import SubmittedGroup, TrainingBuffer
from reliquary.validator.rl_leases import IN_ZONE, LeaseBook, classify_group

REJECTED = "rejected"
NOT_DRAWN, PENDING, PASSED, FAILED, INCONCLUSIVE = "not_drawn", "pending", "passed", "failed", "inconclusive"
# rl_events kinds: pay eligibility (read by the P5 ledger) and sanctions
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
    queued_at REAL NOT NULL, body BLOB);
CREATE TABLE IF NOT EXISTS rl_verdicts(seq INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT NOT NULL,
    hotkey TEXT NOT NULL, source TEXT NOT NULL, verdict TEXT NOT NULL, failure_class TEXT, forced INTEGER NOT NULL,
    at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rl_verdicts_hotkey ON rl_verdicts(hotkey, seq);
CREATE TABLE IF NOT EXISTS rl_events(seq INTEGER PRIMARY KEY AUTOINCREMENT, lease_id TEXT NOT NULL,
    hotkey TEXT NOT NULL, kind TEXT NOT NULL, at REAL NOT NULL, detail TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Rollout:
    """One submitted rollout: its completion token ids and its BFT force span (completion-relative) or None."""
    tokens: tuple[int, ...]
    force_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Graded:
    prompt_tokens: tuple[int, ...]
    rewards: tuple[float, ...] = ()
    truncated: tuple[bool, ...] = ()
    reject: str | None = None


@dataclass(frozen=True)
class AuditResult:
    """The proof forward's verdict on one group: ``passed`` None = inconclusive (no verdict, no sanction)."""
    passed: bool | None
    failure_class: str | None = None
    rollout_match: tuple[float, ...] = ()


ProofForward = Callable[[SubmittedGroup], AuditResult]


@dataclass(frozen=True)
class SubmitConfig:
    audit_rate: float = 0.15
    probation_groups: int = 100
    reprobation_passes: int = SERVICE_REPROBATION_PASSES
    statistical_window: int = SERVICE_STATISTICAL_FAIL_WINDOW
    statistical_max: int = SERVICE_STATISTICAL_FAIL_MAX
    ban_s: float = 86400.0
    first_scan_fraction: float = 0.25
    rollouts: int = GROUP_ROLLOUTS

    def __post_init__(self) -> None:
        if not (0.0 <= self.audit_rate <= 1.0 and 0.0 <= self.first_scan_fraction <= 1.0):
            raise ValueError("audit rate and first-scan fraction are fractions")
        if min(self.probation_groups, self.reprobation_passes, self.statistical_window, self.statistical_max) < 0:
            raise ValueError("audit counts must be non-negative")
        if self.rollouts < 2 or not self.ban_s >= 0:
            raise ValueError("a group has at least 2 rollouts; ban_s >= 0")


class GroupGrader:
    """Grades single-turn groups as admission does, from completion tokens only (§4.4). ``environments`` maps an
    env name to its loaded environment (``get_problem`` and its reward materializer); ``tokenizer`` renders,
    encodes and decodes (the admission worker's)."""

    def __init__(self, environments: Mapping[str, Any], tokenizer: Any, *, eos_token_ids: Sequence[int],
                 think_close_ids: Sequence[int] = (), canonical_force_ids: Sequence[int] = (),
                 vocab_size: int | None = None) -> None:
        from reliquary.validator.admission import AdmissionContext

        for name in environments:
            if get_environment_spec(name).interaction_mode != "single_turn":
                raise ValueError(f"{name}: only single-turn environments are submitted by lease (P3)")
        if not eos_token_ids:
            raise ValueError("the EOS token ids are required to judge termination")
        self.environments = dict(environments)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self._contexts = {name: AdmissionContext(
            randomness="", environment=name, vocab_size=vocab_size, max_sequence_length=MAX_NEW_TOKENS_PROTOCOL_CAP,
            eos_token_ids=tuple(sorted(eos_token_ids)), canonical_force_ids=tuple(canonical_force_ids),
            think_close_ids=tuple(think_close_ids), bootstrap=False, enforce_envelope_signature=False,
            enforce_legacy_merkle=False) for name in environments}

    def grade(self, environment: str, prompt_idx: int, rollouts: Sequence[Rollout]) -> Graded:
        from reliquary.validator.admission import _classify_termination, _force_span_valid
        from reliquary.validator.batcher import admission_problem_materials

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
        texts = [str(self.tokenizer.decode(list(r.tokens), skip_special_tokens=False)) for r in rollouts]
        try:
            rewards = get_environment_spec(environment).score_many(materials.problem, texts,
                                                                  materials.effective_reward_materials)
        except GraderInfrastructureError:
            raise
        except Exception:
            return Graded(prompt, reject="reward_error")
        rewards = tuple(float(r) for r in rewards)
        if len(rewards) != len(rollouts) or not all(math.isfinite(r) for r in rewards):
            return Graded(prompt, reject="reward_error")
        return Graded(prompt, rewards, tuple(truncated))


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
            book.db.commit()
        self.buffer = buffer or TrainingBuffer(book)

    # --- submission ----------------------------------------------------------------------------------------
    def submit(self, lease_id: str, hotkey: str, rollouts: Sequence[Rollout], *,
               arrived_at: float | None = None) -> dict:
        """Grade and record one lease's group. ValueError: not the lease's ``rollouts`` rollouts (lease left
        open); LookupError / PermissionError: as ``LeaseBook.submit``; GraderInfrastructureError: retry later.
        Returns what the miner is told: verdict, rewards, refusal reason (never the audit draw)."""
        if len(rollouts) != self.config.rollouts or any(not r.tokens for r in rollouts):
            raise ValueError(f"a group is exactly {self.config.rollouts} non-empty rollouts")
        arrived_at = float(self.clock()) if arrived_at is None else float(arrived_at)
        lease = self.book.check_submission(lease_id, hotkey, arrived_at)
        graded = self.grader.grade(lease.environment, lease.prompt_idx, rollouts)
        verdict = REJECTED if graded.reject else classify_group(graded.rewards, graded.truncated)
        group = SubmittedGroup(lease_id, hotkey, lease.environment, lease.prompt_idx, lease.checkpoint, lease.nonce,
                               arrived_at, graded.rewards, graded.truncated, graded.prompt_tokens,
                               tuple(r.tokens for r in rollouts), tuple(r.force_span for r in rollouts))
        db = self.book.db
        with self.book.transaction():
            self.book.submit(lease_id, hotkey, in_zone=verdict == IN_ZONE, delivered=verdict != REJECTED,
                             now=arrived_at)
            scan = None if verdict == REJECTED else self.book.record_outcome(
                lease.environment, lease.prompt_idx, lease.checkpoint, graded.rewards, verdict, now=arrived_at)
            audit = training = None
            if verdict == IN_ZONE:
                training = "queued"
                self.buffer.push(group)
            elif scan is not None:
                forced = self._in_probation(hotkey)
                if forced or self._rng.random() < self.config.audit_rate:
                    audit = PENDING
                    db.execute("INSERT INTO rl_audits(lease_id, hotkey, forced, queued_at, body) VALUES(?,?,?,?,?)",
                               (lease_id, hotkey, int(forced), arrived_at, group.pack()))
                else:
                    audit = NOT_DRAWN
                    if scan.first_scan:
                        self._event(lease_id, hotkey, FIRST_SCAN, arrived_at,
                                    {"fraction": self.config.first_scan_fraction, "audited": False})
            db.execute("INSERT INTO rl_groups(lease_id, hotkey, environment, prompt_idx, k, nonce, issued_at, "
                       "arrived_at, verdict, reason, rewards, truncated, completion_tokens, first_scan, scan_seq, "
                       "audit, training) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (lease_id, hotkey, lease.environment, lease.prompt_idx, lease.checkpoint, lease.nonce,
                        lease.issued_at, arrived_at, verdict, graded.reject,
                        None if scan is None else json.dumps(graded.rewards),
                        None if scan is None else json.dumps(graded.truncated),
                        sum(len(r.tokens) for r in rollouts), int(bool(scan and scan.first_scan)),
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
        last_failure = self.book.db.execute("SELECT MAX(seq) FROM rl_verdicts WHERE hotkey=? AND verdict=?",
                                            (hotkey, FAILED)).fetchone()[0]
        return last_failure is not None and self._passes(hotkey, last_failure) < self.config.reprobation_passes

    def in_probation(self, hotkey: str) -> bool:
        with self.book.lock:
            return self._in_probation(hotkey)

    def _verdict(self, lease_id: str, hotkey: str, source: str, passed: bool, failure_class: str | None,
                 forced: bool, now: float) -> None:
        """Log one verdict; a failure sanctions (the class of the strongest failure, decided by the caller)."""
        if not passed and failure_class not in AUDIT_FAILURE_CLASSES:
            raise ValueError(f"a failure needs its class, one of {AUDIT_FAILURE_CLASSES}")
        db = self.book.db
        db.execute("INSERT INTO rl_verdicts(lease_id, hotkey, source, verdict, failure_class, forced, at) "
                   "VALUES(?,?,?,?,?,?,?)", (lease_id, hotkey, source, PASSED if passed else FAILED,
                                             None if passed else failure_class, int(forced), now))
        if passed:
            return
        ban = failure_class == AUDIT_FAILURE_DETERMINISTIC
        if not ban:
            recent = db.execute("SELECT COUNT(*) FROM (SELECT verdict, failure_class FROM rl_verdicts WHERE "
                                "hotkey=? AND forced=0 ORDER BY seq DESC LIMIT ?) WHERE verdict=? AND "
                                "failure_class=?", (hotkey, self.config.statistical_window, FAILED,
                                                    AUDIT_FAILURE_STATISTICAL)).fetchone()[0]
            ban = recent > self.config.statistical_max
        if ban:
            self.book.ban(hotkey, now + self.config.ban_s)
            self._event(lease_id, hotkey, BAN, now, {"failure_class": failure_class, "source": source,
                                                     "until": now + self.config.ban_s})

    def audit_jobs(self, limit: int) -> list[SubmittedGroup]:
        """Queued audits, oldest first, hotkeys past probation before those in probation (decisions doc §C).
        Audit promptly: the forward runs against the latest checkpoint (§4.4), so a late audit costs match."""
        with self.book.lock:
            rows = self.book.db.execute("SELECT body FROM rl_audits WHERE body IS NOT NULL ORDER BY forced, "
                                        "queued_at LIMIT ?", (int(limit),)).fetchall()
        return [SubmittedGroup.unpack(body) for body, in rows]

    def report_audit(self, lease_id: str, result: AuditResult, now: float | None = None) -> None:
        """Conclude a drawn out-of-zone group (once). Passed: its first scan becomes payable. Failed: forfeited
        and sanctioned by class. Inconclusive: forfeited, no sanction. LookupError: not a pending audit."""
        now = float(self.clock()) if now is None else now
        with self.book.transaction():
            db = self.book.db
            row = db.execute("SELECT a.hotkey, a.forced, g.first_scan FROM rl_audits a JOIN rl_groups g "
                             "ON g.lease_id=a.lease_id WHERE a.lease_id=? AND a.body IS NOT NULL",
                             (lease_id,)).fetchone()
            if row is None:
                raise LookupError("no pending audit for this lease")
            hotkey, forced, first_scan = row[0], bool(row[1]), bool(row[2])
            state = INCONCLUSIVE if result.passed is None else PASSED if result.passed else FAILED
            if result.passed is not None:
                self._verdict(lease_id, hotkey, "audit", result.passed, result.failure_class, forced, now)
            db.execute("UPDATE rl_audits SET body=NULL WHERE lease_id=?", (lease_id,))
            db.execute("UPDATE rl_groups SET audit=? WHERE lease_id=?", (state, lease_id))
            if first_scan:
                if state == PASSED:
                    self._event(lease_id, hotkey, FIRST_SCAN, now,
                                {"fraction": self.config.first_scan_fraction, "audited": True})
                else:
                    self._event(lease_id, hotkey, FIRST_SCAN_FORFEITED, now, {"audit": state})

    def run_audits(self, forward: ProofForward, limit: int) -> int:
        """Drain up to ``limit`` queued audits through ``forward`` (the proof forward, the Gumbel verifier).
        A forward that raises leaves its audit queued."""
        done = 0
        for group in self.audit_jobs(limit):
            self.report_audit(group.lease_id, forward(group))
            done += 1
        return done

    # --- the trainer's report (P4) ------------------------------------------------------------------------
    def report_training(self, lease_id: str, *, rollout_match: Sequence[float], accepted: bool,
                        failure_class: str | None = None, now: float | None = None) -> None:
        """The trainer's forward judged a buffered group: ``rollout_match`` per rollout (match rate against the
        published checkpoint, §4.6), ``accepted`` the group verdict. An accepted group is payable (P5 pays the
        matched tokens of rollouts above the floor); a rejected one is unpaid and, when ``failure_class`` says the
        rejection is evidence (not staleness from waiting in the buffer), sanctioned like a failed audit.
        LookupError: not a buffered group."""
        now = float(self.clock()) if now is None else now
        with self.book.transaction():
            group = self.buffer.get(lease_id)
            if group is None or not self.buffer._remove(lease_id):
                raise LookupError("no buffered group for this lease")
            match = [float(m) for m in rollout_match]
            if len(match) != len(group.completions) or not all(0.0 <= m <= 1.0 for m in match):
                raise ValueError("one match rate in [0, 1] per rollout")
            self.book.db.execute("UPDATE rl_groups SET training=? WHERE lease_id=?",
                                 ("accepted" if accepted else REJECTED, lease_id))
            detail = {"rollout_match": match, "completion_tokens": [len(c) for c in group.completions],
                      "checkpoint": group.checkpoint, "environment": group.environment}
            if accepted:
                self._event(lease_id, group.hotkey, TRAINED, now, detail)
                self._verdict(lease_id, group.hotkey, "training", True, None, False, now)
            else:
                self._event(lease_id, group.hotkey, UNTRAINED, now, {**detail, "failure_class": failure_class})
                if failure_class is not None:
                    self._verdict(lease_id, group.hotkey, "training", False, failure_class, False, now)


__all__ = [
    "BAN", "FIRST_SCAN", "FIRST_SCAN_FORFEITED", "REJECTED", "TRAINED", "UNTRAINED", "AuditResult", "Graded",
    "GroupGrader", "ProofForward", "RlSubmissions", "Rollout", "SubmitConfig",
]
