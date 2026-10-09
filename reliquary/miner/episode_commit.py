"""A signed episode as the miner submits it: the token sequence and model spans of one
generate session, the stop the validator admits, and its rollout commit. One forward over the whole
episode gives the GRAIL commitments (every token), the token logprobs at model positions and the TOPLOC
proofs per model span; the signature binds the service binding, the pool binding and the episode (its
transcript by digest)."""
from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

UNOBSERVED_STOPS = ("max_turns", "context_length")
ADMITTED_STOPS = ("agent_completed", *UNOBSERVED_STOPS)


class AdmittedStop(str):
    """A stop ``episode_stop`` computed: the only stop ``episode_metadata`` signs (never a raw label)."""


def episode_sequence(turns: Sequence[Any]) -> tuple[list[int], list[tuple[int, int]]]:
    """``(tokens, spans)`` of a generate session's turns (``GeneratedTurn``): the tokens start with the
    first prompt; each span is a turn's completion, in absolute positions. Each turn's prompt must extend
    the previous turn's prompt and completion by an observation (ValueError otherwise)."""
    if not turns:
        raise ValueError("the session has no turn")
    sequence = list(turns[0].prompt_ids)
    if not sequence:
        raise ValueError("the session's first prompt is empty")
    spans: list[tuple[int, int]] = []
    for k, turn in enumerate(turns):
        prompt = list(turn.prompt_ids)
        if not turn.completion_ids:
            raise ValueError(f"turn {k} has no completion")
        if prompt[:len(sequence)] != sequence or (k and len(prompt) == len(sequence)):
            raise ValueError(f"turn {k}'s prompt does not extend the episode by an observation")
        start = len(prompt)
        spans.append((start, start + len(turn.completion_ids)))
        sequence = prompt + list(turn.completion_ids)
    return sequence, spans


def episode_stop(raw: Any, *, tokens: Sequence[int], spans: Sequence[tuple[int, int]], max_turns: int,
                 max_tokens_per_turn: int, max_episode_tokens: int, stop_ids: Collection[int]) -> str | None:
    """The stop the validator's admission takes for this episode, from the harness's raw stop condition,
    or None when none can be admitted (the episode is then withdrawn, never submitted).

    * The last turn ends on a stop token or exactly at the cap the endpoint applied,
      ``min(max_tokens_per_turn, max_episode_tokens - its start)``; anything else is never admitted.
    * ``max_turns`` needs exactly ``max_turns`` model turns.
    * A last turn cut at the EPISODE's token cap (its span ends at ``max_episode_tokens``, no stop token)
      is ``context_length``: verifiers labels such a cut ``agent_completed`` (the length-finished reply
      carries no call, so the harness ends) and the validator admits it only as ``context_length``. A
      last turn cut at the per-turn cap elsewhere is admitted only as the ``max_turns``-th turn.
    * Otherwise (a closed last turn) ``agent_completed`` and ``context_length`` (verifiers' overlong
      NEXT prompt) are kept; the validator checks the rest (calls, closed reasoning, the overflow)."""
    stop = _mapped_stop(raw, tokens=tokens, spans=spans, max_turns=max_turns,
                        max_tokens_per_turn=max_tokens_per_turn, max_episode_tokens=max_episode_tokens,
                        stop_ids=stop_ids)
    return None if stop is None else AdmittedStop(stop)


def _mapped_stop(raw, *, tokens, spans, max_turns, max_tokens_per_turn, max_episode_tokens, stop_ids):
    if not spans or raw not in ADMITTED_STOPS:
        return None
    start, end = spans[-1]
    if end != len(tokens) or end > max_episode_tokens:
        return None
    closed = tokens[end - 1] in stop_ids
    if not closed and end - start != min(max_tokens_per_turn, max_episode_tokens - start):
        return None
    if raw == "max_turns":
        return raw if len(spans) == max_turns else None
    if not closed:
        if end == max_episode_tokens:
            return "context_length"
        return "max_turns" if len(spans) == max_turns else None
    return raw


def episode_metadata(*, precommit_sha256: str, seed_index: int, spans, stop: AdmittedStop,
                     transcript: dict) -> dict:
    """``rollout.episode`` of a signed-episode commit; ``stop`` is ``episode_stop``'s (ValueError for a raw
    label or anything else)."""
    from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA

    if not isinstance(stop, AdmittedStop) or str(stop) not in ADMITTED_STOPS:
        raise ValueError(f"the stop {stop!r} is not one episode_stop admitted")
    return {"schema_version": SIGNED_EPISODE_SCHEMA, "precommit_sha256": precommit_sha256,
            "seed_index": int(seed_index), "assistant_spans": [[int(s), int(e)] for s, e in spans],
            "stop": str(stop), "transcript": transcript}


def episode_refusal(*, policy, renderer, prompt: str, tokens: Sequence[int], spans: Sequence[tuple[int, int]],
                    stop: Any, transcript: Any, min_chunk_tokens: int | None = None) -> str | None:
    """Why the validator's admission would refuse this episode at its token level, or None: the episode
    length, prompt fidelity (``renderer.initial_ids(prompt)``, the validator's own render) and the
    admission's own ``episode_shape_refusal`` (spans, the signed parse, short turns, budgets, termination, a
    limit stop really reached). ``renderer`` is the miner's turn renderer of the policy's checkpoint over
    the CONTRACT's tools (``agentic_swe.load_turn_renderer(dir, tools=tuple(policy.tools))``, as the
    validator builds it); ``stop`` is ``episode_stop``'s (None is refused). The transcript's signatures
    and bindings are ``rl_transcript_refusal``'s, checked when the episode ended."""
    from reliquary.corpus.signed_parse import signed_records
    from reliquary.protocol.toploc import MIN_CHUNK_TOKENS
    from reliquary.validator.episode_admission import episode_shape_refusal

    if stop is None:
        return "episode_stop: no stop the validator admits"
    tokens = [int(token) for token in tokens]
    spans = [(int(start), int(end)) for start, end in spans]
    if len(tokens) > policy.max_episode_tokens:
        return f"episode_length: {len(tokens)} tokens > {policy.max_episode_tokens}"
    prompt_ids = list(renderer.initial_ids(prompt))
    if not spans or spans[0][0] != len(prompt_ids) or tokens[:len(prompt_ids)] != prompt_ids:
        return "episode_prompt: the prompt tokens are not the validator's render"
    try:
        found = signed_records(transcript)
    except (KeyError, TypeError, ValueError) as error:
        return f"episode_transcript: {type(error).__name__}"
    refused = episode_shape_refusal(
        policy, renderer, prompt_ids=prompt_ids, tokens=tokens, spans=spans, stop=str(stop), calls=found.calls,
        opened=found, final=found.final,
        min_chunk_tokens=MIN_CHUNK_TOKENS if min_chunk_tokens is None else int(min_chunk_tokens))
    return None if refused is None else f"{refused.stage}: {refused.detail}"


@dataclass(frozen=True)
class AdmissibleEpisode:
    """What the miner signs for one episode it keeps: its sequence and the admitted stop."""

    tokens: list[int]
    spans: list[tuple[int, int]]
    stop: AdmittedStop


async def withdraw_inadmissible(outcomes, *, policy, renderer, prompt: str,
                                min_chunk_tokens: int | None = None) -> list[tuple[Any, AdmissibleEpisode]]:
    """Before ``choose_episodes``: each outcome (``seed_index``, ``result``: ``EpisodeResult``, ``session``:
    the generate session's log) the validator would admit, with its sequence and stop; every other one is
    withdrawn at once (``result.release()``: its slot and the hotkey's caps are freed), so no group is ever
    built of an episode the validator refuses."""
    kept: list[tuple[Any, AdmissibleEpisode]] = []
    for outcome in outcomes:
        result, session = outcome.result, outcome.session
        why = None
        try:
            if not getattr(result, "ok", False) or result.transcript is None or session is None:
                raise ValueError("no graded episode")
            tokens, spans = episode_sequence(session.turns)
            stop = episode_stop(result.stop, tokens=tokens, spans=spans, max_turns=policy.max_turns,
                                max_tokens_per_turn=policy.max_tokens_per_turn,
                                max_episode_tokens=policy.max_episode_tokens, stop_ids=renderer.stop_ids)
            why = episode_refusal(policy=policy, renderer=renderer, prompt=prompt, tokens=tokens, spans=spans,
                                  stop=stop, transcript=result.transcript, min_chunk_tokens=min_chunk_tokens)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            why = f"{type(error).__name__}: {error}"
        if why is None:
            kept.append((outcome, AdmissibleEpisode(tokens, spans, stop)))
            continue
        logger.info("seed %s withdrawn: the validator would refuse it (%s)", outcome.seed_index, why)
        release = getattr(result, "release", None)
        if release is not None:
            try:
                await release()
            except Exception:
                logger.exception("withdrawing the episode of seed %s failed", outcome.seed_index)
    return kept


def build_signed_episode_commit(*, model, verifier, tokens: Sequence[int], spans: Sequence[tuple[int, int]],
                                episode: dict, service_binding: dict, seed_pool: dict, randomness: str, wallet,
                                toploc) -> dict:
    """The commit of one episode (``public-group-proof/v1``): ``toploc`` is the contract's TOPLOC profile
    (``chunk_tokens``, ``topk``)."""
    import torch

    from reliquary.constants import LAYER_INDEX
    from reliquary.protocol.seed_pool import PROOF_VERSION
    from reliquary.protocol.signatures import sign_service_episode_commit_binding
    from reliquary.protocol.toploc_proof import span_proofs_b64
    from reliquary.shared.forward import forward_single_layer

    tokens = [int(token) for token in tokens]
    spans = [(int(start), int(end)) for start, end in spans]
    if not spans or [list(span) for span in spans] != [list(span) for span in episode["assistant_spans"]]:
        raise ValueError("the commit's spans are the episode's")
    if not 0 < spans[0][0] or spans[-1][1] != len(tokens):
        raise ValueError("the spans run from after the prompt to the last token")
    device = next(model.parameters()).device
    with torch.no_grad():
        hidden, logits = forward_single_layer(model, torch.tensor([tokens], device=device), None, LAYER_INDEX)
    hidden = hidden[0]
    commitments = verifier.create_commitments_batch(hidden, verifier.generate_r_vec(randomness))
    # fp32 log_softmax, as the validator and the single-turn miner.
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    positions = [position for start, end in spans for position in range(start, end)]
    token_logprobs = [log_probs[position - 1, tokens[position]].item() for position in positions]
    model_name = getattr(model, "name_or_path", "unknown")
    prompt_length = spans[0][0]
    rollout = {"prompt_length": prompt_length, "completion_length": len(tokens) - prompt_length,
               "success": False, "total_reward": 0.0, "advantage": 0.0, "token_logprobs": token_logprobs,
               "forced": False, "force_span": None, "episode": dict(episode), "seed_pool": dict(seed_pool),
               "service_binding": dict(service_binding)}
    signature = sign_service_episode_commit_binding(tokens, randomness, model_name, LAYER_INDEX, commitments,
                                                    rollout["service_binding"], rollout["seed_pool"],
                                                    rollout["episode"], wallet)
    return {"tokens": tokens, "commitments": commitments, "proof_version": PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX}, "signature": signature.hex(),
            "beacon": {"randomness": randomness}, "rollout": rollout,
            "toploc_proofs": span_proofs_b64(hidden, spans, chunk_tokens=toploc.chunk_tokens, topk=toploc.topk)}


__all__ = ["ADMITTED_STOPS", "AdmissibleEpisode", "AdmittedStop", "build_signed_episode_commit", "episode_metadata",
           "episode_refusal", "episode_sequence", "episode_stop", "withdraw_inadmissible"]
