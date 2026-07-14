"""Cheap, validator-derived rollout diagnostics for private calibration.

These signals are observational only. Repetition is not proof of a bad rollout,
and none of the thresholds in this module participate in acceptance or reward.
"""

from __future__ import annotations

from typing import Iterable


def _repeated_ngram_metrics(tokens: list[int], n: int) -> tuple[float, int | None]:
    if n <= 0 or len(tokens) < n:
        return 0.0, None
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    first_repeated_offset: int | None = None
    total = len(tokens) - n + 1
    for start in range(total):
        gram = tuple(tokens[start:start + n])
        if gram in seen:
            repeated += 1
            if first_repeated_offset is None:
                first_repeated_offset = start
        else:
            seen.add(gram)
    return repeated / total, first_repeated_offset


def _same_token_run_metrics(
    tokens: list[int], *, onset_length: int,
) -> tuple[int, int | None]:
    if not tokens:
        return 0, None
    best = 1
    run = 1
    first_onset: int | None = None
    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            run += 1
            best = max(best, run)
            if run == onset_length and first_onset is None:
                first_onset = index - onset_length + 1
        else:
            run = 1
    return best, first_onset


def token_degeneracy_metrics(
    completion_tokens: Iterable[int],
    *,
    ngram_size: int = 4,
    tail_size: int = 256,
    same_token_onset_length: int = 8,
) -> dict[str, int | float | None]:
    """Return low-cost token repetition diagnostics.

    The first repeated n-gram offset marks the start of the second occurrence.
    It can be compared with the first CDF mismatch offset to test directionality
    without retaining token text in private telemetry.
    """
    tokens = [int(token) for token in completion_tokens]
    repeated_fraction, first_repeated = _repeated_ngram_metrics(
        tokens, ngram_size,
    )
    tail = tokens[-max(0, int(tail_size)):] if tail_size else []
    tail_repeated_fraction, _ = _repeated_ngram_metrics(tail, ngram_size)
    max_run, first_run_onset = _same_token_run_metrics(
        tokens,
        onset_length=max(2, int(same_token_onset_length)),
    )
    return {
        "token_count": len(tokens),
        "unique_token_ratio": (
            len(set(tokens)) / len(tokens) if tokens else 0.0
        ),
        "repeated_ngram_fraction": repeated_fraction,
        "tail_repeated_ngram_fraction": tail_repeated_fraction,
        "max_same_token_run": max_run,
        "first_repeated_ngram_offset": first_repeated,
        "first_same_token_run_offset": first_run_onset,
    }


def classify_bft_termination(
    tokens: list[int],
    *,
    prompt_length: int,
    completion_length: int,
    eos_ids: set[int],
    think_close_ids: set[int],
    validated_force_span: tuple[int, int] | None,
    thinking_budget: int,
    answer_budget: int,
) -> str:
    """Classify the validator-observed BFT generation path.

    The result is derived from signed tokens plus a canonically validated force
    span. It never trusts the miner's ``forced`` flag on its own.
    """
    start = max(0, int(prompt_length))
    end = min(len(tokens), start + max(0, int(completion_length)))
    completion = [int(token) for token in tokens[start:end]]
    if not completion:
        return "empty"

    ended_eos = completion[-1] in {int(token) for token in eos_ids}
    if validated_force_span is not None:
        force_end = int(validated_force_span[1])
        phase2_length = max(0, end - force_end)
        if ended_eos:
            return "forced_phase2_eos"
        if phase2_length >= int(answer_budget):
            return "forced_phase2_cap"
        return "forced_phase2_other"

    if ended_eos:
        if len(completion) <= int(thinking_budget):
            return "phase1_eos"
        return "natural_phase2_eos"

    close_offsets = [
        index
        for index, token in enumerate(completion)
        if token in {int(value) for value in think_close_ids}
    ]
    if close_offsets:
        phase2_length = len(completion) - close_offsets[0] - 1
        if phase2_length >= int(answer_budget):
            return "natural_phase2_cap"
        return "natural_phase2_other"
    if len(completion) >= int(thinking_budget):
        return "phase1_cap"
    return "unterminated_short"
