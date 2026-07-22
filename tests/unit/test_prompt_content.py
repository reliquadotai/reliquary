import hashlib

from reliquary.validator.prompt_content import (
    prompt_content_sha256,
    target_content_sha256,
)


def test_prompt_content_digest_uses_exact_domain_separated_bytes():
    expected = hashlib.sha256(
        b"reliquary/prompt-content/v1\0openmathinstruct\0Question\n"
    ).hexdigest()

    assert prompt_content_sha256("openmathinstruct", "Question\n") == expected
    assert prompt_content_sha256("openmathinstruct", "Question") != expected
    assert prompt_content_sha256("opencodeinstruct", "Question\n") != expected


def test_code_target_digest_is_independent_of_mapping_order():
    first = [{"inputs": [1], "expected": 2, "meta": {"b": 2, "a": 1}}]
    second = [{"meta": {"a": 1, "b": 2}, "expected": 2, "inputs": [1]}]

    assert target_content_sha256(
        "opencodeinstruct", {}, code_cases=first
    ) == target_content_sha256(
        "opencodeinstruct", {}, code_cases=second
    )


def test_math_target_digest_preserves_source_answer_bytes():
    compact = target_content_sha256(
        "openmathinstruct", {"ground_truth": "\\boxed{2}"}
    )
    spaced = target_content_sha256(
        "openmathinstruct", {"ground_truth": " \\boxed{2} "}
    )

    assert compact != spaced
