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


class _ChatTokenizerStub:
    """Declares a chat template, like every recent "-Base" tokenizer config."""

    chat_template = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kw):
        return "WRAPPED:" + messages[0]["content"]

    def encode(self, text, *, add_special_tokens):
        return list(text.encode("utf-8"))


def test_render_canonical_prompt_is_verbatim_in_raw_completion_mode(monkeypatch):
    import reliquary.constants as C
    from reliquary.validator.prompt_content import render_canonical_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)

    assert render_canonical_prompt(_ChatTokenizerStub(), "Solve: 1+1") == "Solve: 1+1"


def test_raw_completion_keeps_canonical_render_and_encoding_in_agreement(monkeypatch):
    """The two sites must switch together or content hashing forks.

    render_canonical_prompt feeds prompt_content_sha256 (the content cooldown)
    while encode_prompt feeds the tokens the miner generates from. If only one
    bypasses the template they describe different prompts, and a content-hash
    mismatch is dropped *silently* at seal -- never published, never rejected.
    """
    import reliquary.constants as C
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.validator.prompt_content import render_canonical_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)

    tok = _ChatTokenizerStub()
    rendered = render_canonical_prompt(tok, "Solve: 1+1")
    assert encode_prompt(tok, "Solve: 1+1") == list(rendered.encode("utf-8"))
