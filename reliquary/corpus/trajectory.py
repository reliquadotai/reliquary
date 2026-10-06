"""A `CorpusTrajectory` from the generate endpoint's session log (spec §5 N2).

Pure. Each turn's prompt must extend the previous prompt and completion:
verifiers' train client bridges history token for token, and a turn it had to
re-render from messages instead (a rewritten history) is not one sequence the
audit could prefill, so it is refused here, before anything is signed.
Stop condition and final-span consistency are checked at intake (parse_trajectory),
not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reliquary.corpus.job import EPISODE_STOPS
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
from reliquary.protocol.toploc import span_chunk_count


@dataclass(frozen=True)
class GeneratedTurn:
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    proofs: tuple[str, ...]


@dataclass(frozen=True)
class BuiltTrajectory:
    prompt_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    proofs: tuple[tuple[str, ...], ...]
    final_diff: str
    stop: str
    transcript: dict | None = None

    def wire(self) -> dict:
        out = {"tokens": list(self.tokens),
               "turns": [{"start": start, "end": end, "proofs": list(proofs)}
                         for (start, end), proofs in zip(self.spans, self.proofs)],
               "final_diff": self.final_diff, "stop": self.stop}
        if self.transcript is not None:
            out["transcript"] = self.transcript
        return out


class TrajectoryUnbuildable(ValueError):
    """The session cannot be one trajectory: nothing is submitted for it."""


def build_trajectory(turns: Sequence[GeneratedTurn], *, final_diff: str, stop: str,
                     chunk_tokens: int = TOPLOC_DEPLOYED_DEFAULTS.chunk_tokens) -> BuiltTrajectory:
    if stop not in EPISODE_STOPS:
        raise TrajectoryUnbuildable(f"stop {stop!r} is not one of {EPISODE_STOPS}")
    if not turns:
        raise TrajectoryUnbuildable("the session has no turn")
    first = list(turns[0].prompt_ids)
    sequence = list(first)
    spans: list[tuple[int, int]] = []
    proofs: list[tuple[str, ...]] = []
    for k, turn in enumerate(turns):
        prompt = list(turn.prompt_ids)
        if not turn.completion_ids:
            raise TrajectoryUnbuildable(f"turn {k} has no completion")
        if prompt[:len(sequence)] != sequence:
            raise TrajectoryUnbuildable(f"turn {k}'s prompt does not extend the previous turn")
        if k and len(prompt) == len(sequence):
            raise TrajectoryUnbuildable(f"turn {k} follows turn {k - 1} with no observation between")
        start = len(prompt) - len(first)
        end = start + len(turn.completion_ids)
        span_length = end - start
        expected_proof_count = span_chunk_count(span_length, chunk_tokens)
        if len(turn.proofs) != expected_proof_count:
            raise TrajectoryUnbuildable(
                f"turn {k} has {len(turn.proofs)} proofs but {span_length} tokens require {expected_proof_count}")
        spans.append((start, end))
        proofs.append(tuple(turn.proofs))
        sequence = prompt + list(turn.completion_ids)
    return BuiltTrajectory(prompt_ids=tuple(first), tokens=tuple(sequence[len(first):]),
                           spans=tuple(spans), proofs=tuple(proofs), final_diff=final_diff,
                           stop=stop)
