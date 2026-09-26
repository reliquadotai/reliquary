"""The miner->validator wire for corpus generation. Named `corpus` rather than
`batch` because `BatchSubmissionRequest` is the GRPO training batch."""

import pytest
from pydantic import ValidationError

from reliquary.protocol.corpus_submission import (
    MAX_COMPLETIONS_PER_SUBMISSION,
    MAX_RENDERED_PROMPT_CHARS,
    MAX_COMPLETION_TEXT_CHARS,
    MAX_COMPLETION_TOKENS,
    CorpusCompletion,
    CorpusRejectReason,
    CorpusSubmissionRequest,
    CorpusSubmissionResponse,
)


def _completion(**overrides):
    payload = {"tokens": [1, 2, 3], "text": "hello"}
    payload.update(overrides)
    return payload


def _request(**overrides):
    payload = {
        "job_id": "math-v1",
        "miner_hotkey": "5Gx",
        "cursor": 0,
        "prompt_index": 42,
        "checkpoint_sha256": "a" * 64,
        "rendered_prompt": "<prompt row-42>",
        "completions": [_completion()],
        "signature": "de" * 32,
    }
    payload.update(overrides)
    return payload


def test_a_well_formed_submission_parses():
    request = CorpusSubmissionRequest(**_request())
    assert request.completions[0].tokens == [1, 2, 3]
    assert request.cursor == 0


def test_a_completion_may_not_declare_how_it_terminated():
    """`check_termination` derives that label from the tokens and `admit` has
    no parameter it could travel through, so the field read nothing and could
    only 422 an honest miner whose own word is "stop" or "length"."""
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(termination="eos"))


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

    from reliquary.validator.corpus_text import REASON_PROMPT_MISMATCH

    assert CorpusRejectReason.PROMPT_NOT_FAITHFUL.value == REASON_PROMPT_MISMATCH

    # Two different operational facts, so two different wire names: a miner
    # that gets `job_not_served` has the right job and the wrong validator.
    assert CorpusRejectReason.JOB_NOT_SERVED.value == "job_not_served"
    assert CorpusRejectReason.JOB_UNKNOWN.value == "job_unknown"

    # "your signature is wrong" and "I cannot check signatures" are likewise
    # two facts, and only one of them is the miner's to act on.
    assert CorpusRejectReason.SIGNATURE_UNVERIFIABLE.value == "signature_unverifiable"
    assert CorpusRejectReason.BAD_SIGNATURE.value == "bad_signature"

    from reliquary.corpus.checks import REASON_MINER_BANNED

    assert CorpusRejectReason.MINER_BANNED.value == REASON_MINER_BANNED == "miner_banned"


def test_the_manifest_parser_carries_the_same_ceilings_as_the_wire():
    """`reliquary.corpus.job` stays free of pydantic like its siblings, so it
    duplicates these two numbers rather than importing them. A job declared
    above either one refuses every submission as a bare 422, so a drift here
    is a job that can only refuse."""
    from reliquary.corpus import job as corpus_job

    assert corpus_job.MAX_COMPLETIONS_PER_SUBMISSION == MAX_COMPLETIONS_PER_SUBMISSION
    assert corpus_job.MAX_COMPLETION_TOKENS == MAX_COMPLETION_TOKENS


def test_a_submission_without_the_prompt_it_conditioned_on_is_refused():
    # Optional would mean bypassable: a miner omitting the field would switch
    # prompt fidelity off for itself.
    payload = _request()
    payload.pop("rendered_prompt")
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**payload)


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
        # Prompt fidelity is a free-tier check, so the prompt it checks is not
        # a field a miner may leave out or leave empty.
        {"rendered_prompt": ""},
        {"rendered_prompt": "x" * (MAX_RENDERED_PROMPT_CHARS + 1)},
    ],
)
def test_an_impossible_submission_is_refused(overrides):
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**_request(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [{"tokens": []}, {"tokens": [-1]}],
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


def test_a_completion_carries_base64_proofs():
    completion = CorpusCompletion(**_completion(proofs=["/9kAAQ=="]))
    assert completion.proofs == ["/9kAAQ=="]
    assert CorpusCompletion(**_completion()).proofs == []


@pytest.mark.parametrize("bad", ["not base64!", "A" * 5000])
def test_a_proof_must_be_bounded_base64(bad):
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(proofs=[bad]))


def test_the_proof_reasons_have_their_names():
    assert CorpusRejectReason.BAD_PROOF_SHAPE.value == "bad_proof_shape"
    assert CorpusRejectReason.PROOF_FAIL.value == "proof_fail"


def test_a_completion_cannot_carry_more_proofs_than_tokens():
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(tokens=[1, 2], proofs=["/9kAAQ=="] * 3))


def test_proof_bytes_are_bounded_by_the_completions_own_length():
    # 344 base64 characters is one honest 128-point proof. Three tokens cannot
    # need two of them, whatever the chunking.
    proof = "A" * 344
    with pytest.raises(ValidationError):
        CorpusCompletion(**_completion(tokens=[1, 2, 3], proofs=[proof, proof]))


@pytest.mark.parametrize("tokens,proofs", [(1, 1), (32, 1), (70, 3)])
def test_honest_proof_volumes_fit(tokens, proofs):
    CorpusCompletion(**_completion(tokens=list(range(1, tokens + 1)), proofs=["A" * 344] * proofs))


def test_the_out_of_vocab_reason_is_one_name_everywhere():
    from reliquary.validator.corpus_text import REASON_TOKEN_OUT_OF_VOCAB

    assert CorpusRejectReason.TOKEN_OUT_OF_VOCAB.value == REASON_TOKEN_OUT_OF_VOCAB == "token_out_of_vocab"


def test_the_not_registered_reason_is_the_same_string_on_both_sides():
    from reliquary.corpus.checks import REASON_HOTKEY_NOT_REGISTERED
    from reliquary.protocol.corpus_submission import CorpusRejectReason

    assert (CorpusRejectReason.HOTKEY_NOT_REGISTERED.value
            == REASON_HOTKEY_NOT_REGISTERED == "hotkey_not_registered")
