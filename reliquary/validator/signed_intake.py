"""The intake of a signed-sandbox trajectory (spec §5; plan 3), the `signed_sandbox`
mode next to the replay intake. `check` is synchronous and CPU-bound (signatures, a
parse through the renderer): the route runs it in a thread.

In order, cheapest first:
0. the machine directory is fresh (`directory_ready` on THIS validator's clock), else
   a retryable refusal: a stale directory must never read as `unknown_key`;
1. one trajectory carrying its transcript, for a prompt index the job owns;
2. §5.A/B: `verify_transcript` with EVERY `Expected` field (hotkey, engagement
   `corpus:<job>:<index>`, env, split, index, checkpoint) and the paid-session snapshot,
   `graded` required (the only payable final);
3. the deadline: received no later than `expires_at + GRADING_GRACE_S` by THIS
   validator's clock (the verifier uses none);
4. record 0: a tools version this build renders, the job's tools, the job's
   env_package (its image is the token's, checked by `verify_transcript`);
5. prompt fidelity and vocabulary (the replay intake's own steps);
6. the span structure, then §5.C (`parse_signed_trajectory`, record 0's tools offered,
   the final record);
7. §5.D: the final diff is the state the sandbox graded;
8. the replay intake's turn checks (short turns, budget, termination, proof shape).

Then the route asks `claim` BEFORE its ledger write: the issuer holds the session
(only `live` or `closed_graded`, see `sessions.session_submittable`) so no close,
drain or lapse moves it while the write runs. The facts carry the session's seen key;
`corpus_service` adds it to the ledger turn, so the slot and the session are recorded
in one compare-and-swap, and `admit` refuses a key already seen. After the turn,
`accepted` (the write accepted it) or `release` (anything else) ends the claim.

The reward is the final record's, never the miner's. The TOPLOC audit samples the
spans exactly as for a replay trajectory.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from reliquary_sandbox.attest import GRADING_GRACE_S, Expected, verify_transcript
from reliquary_sandbox.observation import TOOLS_VERSIONS

from reliquary.corpus.checks import check_turn_spans
from reliquary.corpus.job import sandbox_split
from reliquary.corpus.signed_parse import parse_signed_trajectory
from reliquary.corpus.signed_reasons import (
    REASON_SANDBOX_EXPIRED, REASON_SANDBOX_SESSION_REUSED, REASON_SANDBOX_STATE_MISMATCH,
    REASON_SANDBOX_TRANSCRIPT, corpus_engagement, session_seen_key, state_matches,
)
from reliquary.corpus.trajectory_parse import TrajectoryRefused
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
from reliquary.validator.agentic_intake import EpisodeIntake, IntakeFacts, IntakeRefusal

REASON_DIRECTORY_UNAVAILABLE = "sandbox_directory_unavailable"
REASON_SESSION_BUSY = "sandbox_session_busy"
DEFAULT_RETRY_AFTER_S = 10


@dataclass(frozen=True)
class SignedIntakeFacts(IntakeFacts):
    session_id: str = ""
    session_key: str = ""
    machine_id: str = ""
    hotkey: str = ""
    reward: float = 0.0


class SignedEpisodeIntake(EpisodeIntake):
    """`sessions` is the session issuer (`claim`, `release_claim`, `submitted`)."""

    def __init__(self, *, job, source, renderer, tokenizer, vocab_size: int | None,
                 chunk_tokens: int, directory: Callable[[], Any],
                 directory_ready: Callable[[float], bool], token_verifier, sessions,
                 seen: Callable[[], Collection[str]] = frozenset,
                 clock: Callable[[], float] = time.time,
                 retry_after_s: int = DEFAULT_RETRY_AFTER_S,
                 min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        super().__init__(job=job, source=source, renderer=renderer, tokenizer=tokenizer,
                         vocab_size=vocab_size, chunk_tokens=chunk_tokens,
                         min_chunk_tokens=min_chunk_tokens)
        self._directory = directory
        self._directory_ready = directory_ready
        self._tokens = token_verifier
        self._sessions = sessions
        self._seen = seen
        self._clock = clock
        self._retry_after = retry_after_s

    async def claim(self, facts: SignedIntakeFacts) -> IntakeRefusal | None:
        """Hold the session for this submission, before the ledger write."""
        refusal = await self._sessions.claim(facts.session_id, hotkey=facts.hotkey)
        if refusal is None:
            return None
        if refusal.reason == "session_claimed":
            return IntakeRefusal(REASON_SESSION_BUSY, {"session_id": facts.session_id},
                                 retry_after=refusal.retry_after or self._retry_after)
        if refusal.reason == "session_submitted":
            return IntakeRefusal(REASON_SANDBOX_SESSION_REUSED, {"session_id": facts.session_id})
        state = refusal.detail.get("state") if refusal.reason == "session_not_submittable" else None
        return IntakeRefusal(REASON_SANDBOX_TRANSCRIPT, {
            "session_id": facts.session_id, "session_state": state or "unknown"})

    async def release(self, facts: SignedIntakeFacts) -> None:
        """The ledger did not accept the claimed submission."""
        await self._sessions.release_claim(facts.session_id)

    async def accepted(self, facts: SignedIntakeFacts) -> None:
        """After the ledger write accepted it: the claim and the reservation end."""
        await self._sessions.submitted(facts.session_id)

    def check(self, request) -> SignedIntakeFacts | IntakeRefusal:
        received = self._clock()
        if not self._directory_ready(received):
            return IntakeRefusal(REASON_DIRECTORY_UNAVAILABLE,
                                 {"why": "the machine directory is stale"},
                                 retry_after=self._retry_after)
        trajectory = request.trajectory
        if trajectory is None or request.completions:
            return IntakeRefusal("malformed_submission", {
                "why": "an episode job takes one trajectory, not completions",
                "trajectory": trajectory is not None, "completions": len(request.completions)})
        if trajectory.transcript is None:
            return IntakeRefusal("malformed_submission", {
                "transcript": "a signed-sandbox job takes the episode's signed transcript"})
        if not self._job.owns(request.prompt_index):
            return IntakeRefusal("prompt_mismatch", {"got": request.prompt_index})
        episode = self._job.episode
        spec = episode.sandbox
        expected = Expected(hotkey=request.miner_hotkey,
                            engagement=corpus_engagement(self._job.job_id, request.prompt_index),
                            env=spec.env, split=sandbox_split(episode), index=request.prompt_index,
                            checkpoint=self._job.checkpoint_sha256,
                            seen_session_ids=self._seen(), require_graded=True)
        result = verify_transcript(trajectory.transcript, self._directory(), self._tokens, expected)
        if not result.ok:
            return IntakeRefusal(REASON_SANDBOX_TRANSCRIPT,
                                 {"reasons": [reason.value for reason in result.reasons]})
        claims, opened, final = result.claims, result.open, result.final
        deadline = claims.expires_at + GRADING_GRACE_S
        if received > deadline:
            return IntakeRefusal(REASON_SANDBOX_EXPIRED,
                                 {"received_at": int(received), "deadline": deadline})
        if opened.tools_version not in TOOLS_VERSIONS:
            return IntakeRefusal(REASON_SANDBOX_TRANSCRIPT,
                                 {"record0": "tools_version", "got": opened.tools_version})
        if tuple(opened.tools) != tuple(spec.tools):
            return IntakeRefusal(REASON_SANDBOX_TRANSCRIPT,
                                 {"record0": "tools", "got": list(opened.tools)})
        if opened.env_package != spec.env_package:
            return IntakeRefusal(REASON_SANDBOX_TRANSCRIPT,
                                 {"record0": "env_package", "got": opened.env_package})
        prompt_ids, refusal = self._prompt_refusal(request)
        if refusal is not None:
            return refusal
        tokens = trajectory.tokens
        spans = [(turn.start, turn.end) for turn in trajectory.turns]
        spans_ok = check_turn_spans(spans, len(tokens), episode.max_turns)
        if not spans_ok.ok:
            return IntakeRefusal(spans_ok.reason or "", dict(spans_ok.detail))
        try:
            parse_signed_trajectory(self._renderer, prompt_ids=prompt_ids, tokens=tokens,
                                    spans=spans, stop=trajectory.stop, max_turns=episode.max_turns,
                                    calls=result.calls, offered=opened.tools, final=final)
        except TrajectoryRefused as refused:
            return IntakeRefusal(refused.reason, refused.detail)
        if not state_matches(final.state_sha256, trajectory.final_diff):
            return IntakeRefusal(REASON_SANDBOX_STATE_MISMATCH, {"state_sha256": final.state_sha256})
        refusal = self._shape_refusal(trajectory, spans, prompt_ids)
        if refusal is not None:
            return refusal
        facts = self._facts(request, spans, prompt_ids)
        return SignedIntakeFacts(prompt_ids=facts.prompt_ids, token_count=facts.token_count,
                                 digest=facts.digest, last_token_id=facts.last_token_id,
                                 session_id=claims.session_id,
                                 session_key=session_seen_key(claims.session_id),
                                 machine_id=claims.machine_id, hotkey=claims.hotkey,
                                 reward=float(final.reward))


def build_signed_episode_intake(job, *, checkpoint_dir: str, tokenizer, vocab_size: int | None,
                                chunk_tokens: int, directory, directory_ready, token_verifier,
                                sessions, seen: Callable[[], Collection[str]] = frozenset,
                                retry_after_s: int = DEFAULT_RETRY_AFTER_S) -> SignedEpisodeIntake:
    """The intake a validator serves a signed-sandbox job with; refuses to start on a
    pin this process does not have (ValueError: permanent until a restart)."""
    from reliquary.environment import agentic_swe

    refusal = (agentic_swe.episode_support_refusal(job.episode, need_verifiers=False)
               or agentic_swe.sandbox_support_refusal(job.episode))
    if refusal:
        raise ValueError(f"job {job.job_id!r}: {refusal}")
    return SignedEpisodeIntake(
        job=job, source=agentic_swe.SignedSweSource(sandbox_split(job.episode)),
        renderer=agentic_swe.load_turn_renderer(checkpoint_dir, tools=job.episode.sandbox.tools),
        tokenizer=tokenizer, vocab_size=vocab_size, chunk_tokens=chunk_tokens,
        directory=directory, directory_ready=directory_ready, token_verifier=token_verifier,
        sessions=sessions, seen=seen, retry_after_s=retry_after_s)


__all__ = ["REASON_DIRECTORY_UNAVAILABLE", "REASON_SESSION_BUSY", "SignedEpisodeIntake",
           "SignedIntakeFacts", "build_signed_episode_intake"]
