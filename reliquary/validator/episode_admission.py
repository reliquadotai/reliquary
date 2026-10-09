"""Admission of a signed-episode group in the v2 service RL path (phase 2, plan 2C; spec §4.1.6). No GPU.

Per episode, cheapest first:
1. the precommit: recorded, this hotkey's, env's, task's, window's, checkpoint's and pool's; the episode
   names it and its own chosen seed;
2. the contract's per-env maximum episode length (set at qualification from the trainer's memory);
3. prompt fidelity: the prompt tokens are the validator's own render of the task prompt;
4. ``verify_transcript`` with every RL binding (hotkey, engagement ``rl:{window}:{precommit}:{seed}``,
   env, split, task, checkpoint), the paid-session snapshot and a graded final;
5. the grading deadline, on this validator's clock;
6. record 0: a tools version this build renders, the contract's tools and env package;
7. the span structure and §5.C (``parse_signed_trajectory``): the model's calls are the signed records,
   every observation is their rendering, token for token;
8. the turn shape: short turns, the contract's turn and episode budgets, termination (a stop token or
   exactly the cap), one TOPLOC proof per span chunk.

The reward is the final record's. An episode cut by a turn or token limit is a normal episode: its box
was graded on its final state, so nothing here is "uncertain". A refusal reuses an existing wire reason
(an older miner parses the enum) and names its check in the stage."""
from __future__ import annotations

import functools
from collections.abc import Collection
from dataclasses import dataclass, field

from reliquary_sandbox.attest import GRADING_GRACE_S, Expected, Reason, verify_transcript
from reliquary_sandbox.observation import TOOLS_VERSIONS

from reliquary.corpus.checks import (
    check_short_turns, check_turn_budget, check_turn_spans, check_turn_termination,
)
from reliquary.corpus.signed_parse import parse_signed_trajectory
from reliquary.corpus.trajectory_parse import TrajectoryRefused
from reliquary.protocol.service_episode import rl_engagement
from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA, RejectReason
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, span_chunk_count


@dataclass(frozen=True)
class EpisodeRefusal:
    reason: RejectReason
    stage: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeGroupFacts:
    """What admission derived; nothing the miner declared."""

    rewards: tuple[float, ...]
    session_ids: tuple[str, ...]
    spans: tuple[tuple[tuple[int, int], ...], ...]     # per episode, absolute positions in its tokens
    model_tokens: int


_TRANSCRIPT_REASONS = {
    Reason.CHECKPOINT_MISMATCH: (RejectReason.WRONG_CHECKPOINT, "episode_checkpoint"),
    Reason.SESSION_REUSED: (RejectReason.HASH_DUPLICATE, "episode_session_reused"),
}


class EpisodeGroupChecker:
    """One per episode env: its contract policy, the turn renderer of the policy's tokenizer, its task
    source (plan 2A) and the TOPLOC chunk size. ``check`` is synchronous and CPU-bound (signatures, a
    parse through the renderer): the intake runs it in a thread."""

    def __init__(self, *, policy, renderer, source, chunk_tokens: int,
                 min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        self.policy = policy
        self._renderer = renderer
        self._source = source
        self._chunk = int(chunk_tokens)
        self._min_chunk = int(min_chunk_tokens)
        self._prompt_ids = functools.lru_cache(maxsize=4096)(self._render_prompt)

    def _render_prompt(self, task_index: int) -> tuple[int, ...]:
        return tuple(self._renderer.initial_ids(self._source.prompt(task_index)))

    def check(self, request, *, precommit, directory, token_verifier, seen: Collection[str],
              received: float) -> EpisodeGroupFacts | EpisodeRefusal:
        from reliquary.protocol.seed_pool import PoolSelection, SeedPoolError
        from reliquary.protocol.service_submission import ServiceBinding

        policy = self.policy
        if precommit is None:
            return EpisodeRefusal(RejectReason.PRECOMMIT_INVALID, "episode_precommit", {"why": "unknown"})
        try:
            order = ServiceBinding.from_dict(request.service_binding).contract_sha256
        except (TypeError, ValueError):
            order = None
        if (precommit.hotkey != request.miner_hotkey or precommit.environment != policy.environment
                or precommit.task_index != request.prompt_idx or precommit.window != request.window_start
                or precommit.order != order
                or any(rollout.env_name != policy.environment for rollout in request.rollouts)):
            return EpisodeRefusal(RejectReason.PRECOMMIT_INVALID, "episode_precommit",
                                  {"why": "the precommit is not this group's"})
        if precommit.checkpoint != request.checkpoint_hash:
            return EpisodeRefusal(RejectReason.WRONG_CHECKPOINT, "episode_checkpoint", {})
        try:
            selection = PoolSelection.from_dict(request.pool_selection)
        except (SeedPoolError, TypeError):
            return EpisodeRefusal(RejectReason.BAD_SCHEMA, "episode_selection", {})
        if selection.pool_sha256 != precommit.pool_sha256 or len(selection.seeds) != len(request.rollouts):
            return EpisodeRefusal(RejectReason.PRECOMMIT_INVALID, "episode_precommit",
                                  {"why": "the selection is not of the precommitted pool"})
        try:
            prompt_ids = self._prompt_ids(int(request.prompt_idx))
        except Exception:
            return EpisodeRefusal(RejectReason.WORKER_DROPPED, "episode_prompt_source", {})
        rewards: list[float] = []
        sessions: list[str] = []
        spans_out: list[tuple[tuple[int, int], ...]] = []
        model_tokens = 0
        for index, (rollout, seed) in enumerate(zip(request.rollouts, selection.seeds)):
            outcome = self._episode(index, rollout, seed, precommit, prompt_ids, directory, token_verifier,
                                    seen, received)
            if isinstance(outcome, EpisodeRefusal):
                return outcome
            reward, session_id, spans = outcome
            rewards.append(reward)
            sessions.append(session_id)
            spans_out.append(spans)
            model_tokens += sum(end - start for start, end in spans)
        if len(set(sessions)) != len(sessions):
            return EpisodeRefusal(RejectReason.HASH_DUPLICATE, "episode_session_reused",
                                  {"why": "one session twice in the group"})
        return EpisodeGroupFacts(tuple(rewards), tuple(sessions), tuple(spans_out), model_tokens)

    def _episode(self, index, rollout, seed, precommit, prompt_ids, directory, token_verifier, seen, received):
        policy = self.policy
        commit = rollout.commit or {}
        meta = commit.get("rollout") or {}
        episode = meta.get("episode")
        where = {"rollout": index}
        if not isinstance(episode, dict) or episode.get("schema_version") != SIGNED_EPISODE_SCHEMA:
            return EpisodeRefusal(RejectReason.BAD_SCHEMA, "episode_schema", where)
        if episode.get("precommit_sha256") != precommit.sha256 or episode.get("seed_index") != seed:
            return EpisodeRefusal(RejectReason.PRECOMMIT_INVALID, "episode_precommit", where)
        tokens = list(commit.get("tokens") or [])
        if len(tokens) > policy.max_episode_tokens:
            return EpisodeRefusal(RejectReason.BAD_TOKENS, "episode_length",
                                  {**where, "tokens": len(tokens), "max": policy.max_episode_tokens})
        try:
            spans = [(int(start), int(end)) for start, end in episode.get("assistant_spans") or ()]
        except (TypeError, ValueError):
            return EpisodeRefusal(RejectReason.BAD_SCHEMA, "episode_schema", where)
        offset = len(prompt_ids)
        if not spans or spans[0][0] != offset or tuple(tokens[:offset]) != tuple(prompt_ids):
            return EpisodeRefusal(RejectReason.PROMPT_MISMATCH, "episode_prompt", where)
        # The declared lengths every later stage slices the tokens with (proof, pi_old, payload).
        if meta.get("prompt_length") != offset or meta.get("completion_length") != len(tokens) - offset:
            return EpisodeRefusal(RejectReason.BAD_TOKENS, "episode_length",
                                  {**where, "why": "declared lengths are not the episode's"})
        expected = Expected(hotkey=precommit.hotkey,
                            engagement=rl_engagement(precommit.window, precommit.sha256, seed),
                            env=policy.sandbox_env, split=policy.split, index=precommit.task_index,
                            checkpoint=precommit.checkpoint, seen_session_ids=seen, require_graded=True)
        result = verify_transcript(episode.get("transcript"), directory, token_verifier, expected)
        if not result.ok:
            reasons = [reason.value for reason in result.reasons]
            # A milder reason only when it is the whole story: any other failure is a forged transcript.
            mapped = {_TRANSCRIPT_REASONS.get(reason) for reason in result.reasons}
            if len(mapped) == 1 and None not in mapped:
                wire, stage = mapped.pop()
                return EpisodeRefusal(wire, stage, {**where, "reasons": reasons})
            return EpisodeRefusal(RejectReason.REWARD_MISMATCH, "episode_transcript", {**where, "reasons": reasons})
        claims, opened, final = result.claims, result.open, result.final
        deadline = claims.expires_at + GRADING_GRACE_S
        if received > deadline:
            return EpisodeRefusal(RejectReason.PRECOMMIT_EXPIRED, "episode_deadline", {**where, "deadline": deadline})
        if (opened.tools_version not in TOOLS_VERSIONS or tuple(opened.tools) != tuple(policy.tools)
                or opened.env_package != policy.env_package):
            return EpisodeRefusal(RejectReason.REWARD_MISMATCH, "episode_record0", where)
        relative = [(start - offset, end - offset) for start, end in spans]
        completion = tokens[offset:]
        structure = check_turn_spans(relative, len(completion), policy.max_turns)
        if not structure.ok:
            return EpisodeRefusal(RejectReason.BAD_TOKENS, "episode_turns",
                                  {**where, "check": structure.reason, "detail": dict(structure.detail)})
        try:
            parse_signed_trajectory(self._renderer, prompt_ids=prompt_ids, tokens=completion, spans=relative,
                                    stop=episode.get("stop"), max_turns=policy.max_turns, calls=result.calls,
                                    offered=opened.tools, final=final)
        except TrajectoryRefused as refused:
            return EpisodeRefusal(RejectReason.BAD_TOKENS, "episode_parse", {**where, "check": refused.reason})
        except (ValueError, TypeError, KeyError, IndexError) as error:
            # The renderer on miner-chosen tokens: a refusal, never an exception out of admission.
            return EpisodeRefusal(RejectReason.BAD_TOKENS, "episode_parse",
                                  {**where, "check": type(error).__name__})
        renderer = self._renderer
        checks = (
            (RejectReason.BAD_TOKENS, "episode_turns",
             lambda: check_short_turns(relative, self._min_chunk)),
            (RejectReason.BAD_TOKENS, "episode_length",
             lambda: check_turn_budget(relative, prompt_len=offset, length=len(completion),
                                       max_tokens_per_turn=policy.max_tokens_per_turn,
                                       max_total_tokens=policy.max_episode_tokens)),
            (RejectReason.BAD_TERMINATION, "episode_termination",
             lambda: check_turn_termination(completion, relative, prompt_len=offset,
                                            terminator_id=renderer.terminator_id, stop_ids=renderer.stop_ids,
                                            max_tokens_per_turn=policy.max_tokens_per_turn,
                                            max_total_tokens=policy.max_episode_tokens)),
        )
        for wire, stage, run in checks:
            outcome = run()
            if not outcome.ok:
                return EpisodeRefusal(wire, stage, {**where, "check": outcome.reason, "detail": dict(outcome.detail)})
        proofs = commit.get("toploc_proofs")
        expected_proofs = sum(span_chunk_count(end - start, self._chunk, self._min_chunk) for start, end in relative)
        if not isinstance(proofs, list) or len(proofs) != expected_proofs:
            return EpisodeRefusal(RejectReason.BAD_SCHEMA, "episode_proof_shape", {**where, "expected": expected_proofs})
        return float(final.reward), claims.session_id, tuple(spans)


def finish_prepared(prepared, facts: EpisodeGroupFacts, contract) -> None:
    """Hand the batcher a normal prepared group, in place: the final records' rewards (validator
    authoritative), the validated spans the Episode v1 masking path trains on (policy positions, pi_old,
    payload), no uncertain rollout (spec §4.1.6), and the phase 1 lane's verdict on the vector: None is
    no observation (OUT_OF_ZONE), anything else goes on (training / exploration / unproven)."""
    from reliquary.services.admission_policy import service_lane

    request = prepared.request
    for rollout, reward, spans in zip(request.rollouts, facts.rewards, facts.spans, strict=True):
        rollout.reward = float(reward)
        meta = rollout.commit.get("rollout")
        if isinstance(meta, dict):
            meta["success"] = reward > 0.5
            meta["total_reward"] = float(reward)
            meta["truncated"] = False
        rollout._validated_assistant_spans = tuple(spans)
    prepared.rewards = [float(reward) for reward in facts.rewards]
    prepared.completion_texts = [""] * len(facts.rewards)
    prepared.truncated_indices = ()
    prepared.uncertain_indices = ()
    prepared.attainable_rewards = ()
    prepared.episode_pending = False
    if service_lane(request, contract, prepared.rewards) is None:
        prepared.reject_reason = RejectReason.OUT_OF_ZONE
        prepared.reject_stage = "zone"


__all__ = ["EpisodeGroupChecker", "EpisodeGroupFacts", "EpisodeRefusal", "finish_prepared"]
