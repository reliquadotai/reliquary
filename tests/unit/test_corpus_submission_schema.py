"""The miner->validator wire for corpus generation. Named `corpus` rather than
`batch` because `BatchSubmissionRequest` is the GRPO training batch."""

import pytest
from pydantic import ValidationError

from reliquary.protocol.corpus_submission import (
    MAX_COMPLETIONS_PER_SUBMISSION,
    MAX_COMPLETION_TEXT_CHARS,
    MAX_COMPLETION_TOKENS,
    CorpusCompletion,
    CorpusRejectReason,
    CorpusSubmissionRequest,
    CorpusSubmissionResponse,
)


def _completion(**overrides):
    payload = {"tokens": [1, 2, 3], "text": "hello", "termination": "eos"}
    payload.update(overrides)
    return payload


def _request(**overrides):
    payload = {
        "job_id": "math-v1",
        "miner_hotkey": "5Gx",
        "cursor": 0,
        "prompt_index": 42,
        "checkpoint_sha256": "a" * 64,
        "completions": [_completion()],
        "signature": "de" * 32,
    }
    payload.update(overrides)
    return payload


def test_a_well_formed_submission_parses():
    request = CorpusSubmissionRequest(**_request())
    assert request.completions[0].termination == "eos"
    assert request.cursor == 0


def test_the_checks_agree_with_the_pure_module_on_reason_names():
    # reliquary.corpus.checks emits these as plain strings; they must match.
    assert CorpusRejectReason.BAD_COMPLETION_COUNT.value == "bad_completion_count"
    assert CorpusRejectReason.TOKEN_BUDGET_EXCEEDED.value == "token_budget_exceeded"
    assert CorpusRejectReason.TOKEN_BUDGET_UNDERRUN.value == "token_budget_underrun"
    assert CorpusRejectReason.BAD_TERMINATION.value == "bad_termination"
    assert CorpusRejectReason.HASH_DUPLICATE.value == "hash_duplicate"
    assert CorpusRejectReason.MALFORMED_SUBMISSION.value == "malformed_submission"
    # The service maps a refusal to the wire by its own reason string, so a
    # divergence here would raise on the response instead of naming the fault.
    from reliquary.validator.corpus_text import REASON_TEXT_MISMATCH

    assert CorpusRejectReason.TEXT_MISMATCH.value == REASON_TEXT_MISMATCH


def test_an_unknown_field_is_refused():
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**_request(surprise=1))
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(surprise=1))


@pytest.mark.parametrize(
    "overrides",
    [
        {"cursor": -1},
        {"prompt_index": -1},
        {"completions": []},
        {"checkpoint_sha256": "a" * 63},
        {"job_id": ""},
        {"miner_hotkey": ""},
    ],
)
def test_an_impossible_submission_is_refused(overrides):
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**_request(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [{"tokens": []}, {"termination": "truncated"}, {"tokens": [-1]}],
)
def test_an_impossible_completion_is_refused(overrides):
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(**overrides))


def test_the_wire_is_bounded():
    # Without a ceiling the whole payload is parsed before a check can refuse it.
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(tokens=[1] * (MAX_COMPLETION_TOKENS + 1)))
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**_request(
            completions=[_completion()] * (MAX_COMPLETIONS_PER_SUBMISSION + 1)
        ))
    # `text` carries the most bytes of any field, so leaving it unbounded
    # leaves the whole ceiling decorative.
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(text="x" * (MAX_COMPLETION_TEXT_CHARS + 1)))


def test_the_response_carries_the_verdict_and_what_is_left():
    response = CorpusSubmissionResponse(
        reason=CorpusRejectReason.ACCEPTED,
        accepted=True,
        slots_remaining=3,
        detail={},
    )
    assert response.accepted is True
    assert response.slots_remaining == 3

    refused = CorpusSubmissionResponse(
        reason=CorpusRejectReason.PROMPT_FULL,
        accepted=False,
        slots_remaining=0,
        detail={"prompt_index": 42},
    )
    assert refused.accepted is False
