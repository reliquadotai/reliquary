"""The intake of one agentic trajectory (spec §5 N3): every check a
submission passes before it may consume a slot. Synchronous and CPU-bound
(a parse of up to 60k tokens through the renderer): the route runs it in a
thread, never on the event loop."""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field

from reliquary.corpus.checks import (
    check_short_turns,
    check_turn_budget,
    check_turn_proof_shape,
    check_turn_spans,
    check_turn_termination,
    completion_digest,
)
from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
from reliquary.validator.corpus_text import REASON_PROMPT_MISMATCH, REASON_TOKEN_OUT_OF_VOCAB


@dataclass(frozen=True)
class IntakeFacts:
    """What the route derives from an accepted trajectory, never declared by the miner."""

    prompt_ids: tuple[int, ...]
    token_count: int
    digest: str
    last_token_id: int


@dataclass(frozen=True)
class IntakeRefusal:
    reason: str
    detail: dict = field(default_factory=dict)


class EpisodeIntake:
    def __init__(self, *, job, source, renderer, tokenizer, vocab_size: int | None,
                 chunk_tokens: int, min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> None:
        self._job = job
        self._source = source
        self._renderer = renderer
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._chunk_tokens = chunk_tokens
        self._min_chunk = min_chunk_tokens
        # Intake runs on several threads; a HF tokenizer is not thread-safe
        # (ruling P6). The renderer guards itself (QwenTurnRenderer's RLock).
        self._tokenizer_lock = threading.Lock()
        self.initial_ids = functools.lru_cache(maxsize=4096)(self._initial_ids)

    def _initial_ids(self, prompt_index: int) -> tuple[int, ...]:
        return tuple(self._renderer.initial_ids(self._source.prompt(prompt_index)))

    def check(self, request) -> IntakeFacts | IntakeRefusal:
        trajectory = request.trajectory
        if trajectory is None or request.completions:
            return IntakeRefusal("malformed_submission", {
                "why": "an episode job takes one trajectory, not completions",
                "trajectory": trajectory is not None, "completions": len(request.completions)})
        if not self._job.owns(request.prompt_index):
            return IntakeRefusal("prompt_mismatch", {"got": request.prompt_index})
        prompt_ids = self.initial_ids(request.prompt_index)
        with self._tokenizer_lock:
            expected = self._tokenizer.decode(list(prompt_ids), skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
        if expected != request.rendered_prompt:
            return IntakeRefusal(REASON_PROMPT_MISMATCH, {
                "prompt_index": request.prompt_index, "expected_chars": len(expected),
                "rendered_chars": len(request.rendered_prompt)})
        tokens = trajectory.tokens
        if self._vocab_size is not None and max(tokens) >= self._vocab_size:
            return IntakeRefusal(REASON_TOKEN_OUT_OF_VOCAB, {"vocab_size": self._vocab_size})
        spans = [(turn.start, turn.end) for turn in trajectory.turns]
        episode = self._job.episode

        refusal = check_turn_spans(spans, len(tokens), episode.max_turns)
        if not refusal.ok:
            return IntakeRefusal(refusal.reason or "", dict(refusal.detail))
        try:
            parse_trajectory(self._renderer, prompt_ids=prompt_ids, tokens=tokens, spans=spans,
                             stop=trajectory.stop, max_turns=episode.max_turns)
        except TrajectoryRefused as refused:
            return IntakeRefusal(refused.reason, refused.detail)
        # Only after the spans are known sound: termination indexes them.
        checks = (
            lambda: check_short_turns(spans, self._min_chunk),
            lambda: check_turn_budget(spans, prompt_len=len(prompt_ids), length=len(tokens),
                                      max_tokens_per_turn=episode.max_tokens_per_turn,
                                      max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_termination(tokens, spans, prompt_len=len(prompt_ids),
                                           terminator_id=self._renderer.terminator_id,
                                           stop_ids=self._renderer.stop_ids,
                                           max_tokens_per_turn=episode.max_tokens_per_turn,
                                           max_total_tokens=episode.max_total_tokens),
            lambda: check_turn_proof_shape(spans, [len(turn.proofs) for turn in trajectory.turns],
                                           self._chunk_tokens),
        )
        for check in checks:
            result = check()
            if not result.ok:
                return IntakeRefusal(result.reason or "", dict(result.detail))
        return IntakeFacts(prompt_ids=prompt_ids, token_count=sum(e - s for s, e in spans),
                           digest=completion_digest(request.prompt_index, tokens),
                           last_token_id=int(tokens[-1]))


def build_episode_intake(job, *, checkpoint_dir: str, tokenizer, vocab_size: int | None,
                         chunk_tokens: int) -> EpisodeIntake:
    """The intake a validator serves an episode job with; refuses to start on
    a pin this binary does not have installed."""
    from reliquary.environment import agentic_swe

    refusal = agentic_swe.episode_support_refusal(job.episode, need_verifiers=False)
    if refusal:
        raise RuntimeError(f"job {job.job_id!r}: {refusal}")
    return EpisodeIntake(job=job, source=agentic_swe.load_swe_source(job.episode.env.num_images),
                         renderer=agentic_swe.load_turn_renderer(checkpoint_dir),
                         tokenizer=tokenizer, vocab_size=vocab_size, chunk_tokens=chunk_tokens)


__all__ = ["EpisodeIntake", "IntakeFacts", "IntakeRefusal", "build_episode_intake"]
