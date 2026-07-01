from reliquary.protocol.tokens import encode_prompt, int_to_bytes, hash_tokens


class TestIntToBytes:
    def test_zero(self):
        assert int_to_bytes(0) == b"\x00\x00\x00\x00"

    def test_one(self):
        assert int_to_bytes(1) == b"\x00\x00\x00\x01"

    def test_big_endian(self):
        assert int_to_bytes(256) == b"\x00\x00\x01\x00"

    def test_large_value(self):
        result = int_to_bytes(0xFFFFFFFF)
        assert result == b"\xff\xff\xff\xff"

    def test_always_4_bytes(self):
        for val in [0, 1, 255, 65535, 2**32 - 1]:
            assert len(int_to_bytes(val)) == 4


class TestHashTokens:
    def test_deterministic(self):
        tokens = [1, 2, 3, 4, 5]
        assert hash_tokens(tokens) == hash_tokens(tokens)

    def test_32_bytes(self):
        assert len(hash_tokens([1, 2, 3])) == 32

    def test_different_tokens_differ(self):
        assert hash_tokens([1, 2, 3]) != hash_tokens([3, 2, 1])

    def test_order_matters(self):
        assert hash_tokens([1, 2]) != hash_tokens([2, 1])

    def test_empty(self):
        result = hash_tokens([])
        assert len(result) == 32


class _ChatTokenizer:
    """Minimal stand-in for an instruct-model tokenizer.

    Mirrors transformers >=5: ``apply_chat_template(tokenize=True)`` returns
    a BatchEncoding (dict) *unless* ``return_dict=False`` is passed. encode_prompt
    must request a flat list explicitly, or it ends up iterating dict keys.
    """

    chat_template = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_dict=True):
        assert add_generation_prompt is True
        assert tokenize is True
        prompt = messages[0]["content"]
        ids = [1] + list(prompt.encode("utf-8"))
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class _ChatTokenizerIgnoresReturnDict:
    """Pathological tokenizer that returns a BatchEncoding even when
    ``return_dict=False`` is requested. encode_prompt must still extract the
    flat token ids."""

    chat_template = "x"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_dict=False):
        ids = [2] + list(messages[0]["content"].encode("utf-8"))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text, *, add_special_tokens):
        raise AssertionError("chat-template path expected, not encode()")


class _ThinkingTokenizer:
    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self):
        self.enable_thinking_seen = None

    def apply_chat_template(
        self, messages, *, add_generation_prompt, tokenize, return_dict=False,
        enable_thinking=None,
    ):
        self.enable_thinking_seen = enable_thinking
        return [3] + list(messages[0]["content"].encode("utf-8"))

    def encode(self, text, *, add_special_tokens):
        raise AssertionError("chat-template path expected, not encode()")


class _BareTokenizer:
    chat_template = None

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class TestEncodePrompt:
    def test_uses_chat_template_when_present(self):
        tok = _ChatTokenizer()
        out = encode_prompt(tok, "hi")
        assert out == [1] + list(b"hi")
        assert all(isinstance(t, int) for t in out), "must be int token ids, not dict keys"

    def test_extracts_input_ids_when_return_dict_ignored(self):
        """A tokenizer that ignores return_dict and returns a BatchEncoding
        must still yield flat int token ids."""
        tok = _ChatTokenizerIgnoresReturnDict()
        out = encode_prompt(tok, "hi")
        assert out == [2] + list(b"hi")
        assert all(isinstance(t, int) for t in out)

    def test_enables_thinking_when_template_supports_it(self):
        tok = _ThinkingTokenizer()
        out = encode_prompt(tok, "hi")
        assert out == [3] + list(b"hi")
        assert tok.enable_thinking_seen is True

    def test_falls_back_when_no_chat_template(self):
        tok = _BareTokenizer()
        assert encode_prompt(tok, "hi") == list(b"hi")

    def test_miner_and_validator_agree(self):
        """The single shared helper guarantees byte-equal prompt tokens on
        both sides, so the validator's PROMPT_MISMATCH gate cannot trip on
        an honest miner that went through the same path."""
        tok = _ChatTokenizer()
        prompt = "Solve 2 + 2"
        miner_tokens = encode_prompt(tok, prompt)
        validator_tokens = encode_prompt(tok, prompt)
        assert miner_tokens == validator_tokens

    def test_magicmock_tokenizer_falls_back(self):
        """MagicMock returns truthy values for any attribute access; the
        helper must still fall through to plain encode() so existing test
        stubs continue to work."""
        from unittest.mock import MagicMock

        tok = MagicMock()
        tok.encode.return_value = [7, 8, 9]
        out = encode_prompt(tok, "anything")
        assert out == [7, 8, 9]
        tok.apply_chat_template.assert_not_called()


# Frozen reference encoding for the production model + pinned transformers.
# Regenerate ONLY on a deliberate, coordinated encoding change (and bump the
# transformers pin in pyproject.toml in lockstep). Any drift here means
# miners and the validator would disagree -> mass PROMPT_MISMATCH.
_GOLDEN_PROMPT = "What is the capital of France? Answer in one word."
# thinking-on: ends with an open <think>\n block (248068, 198) for the model to
# fill, vs the old thinking-off empty <think></think> block.
_GOLDEN_TOKENS_QWEN35_2B = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30,
    21134, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198,
    248068, 198,
]


def test_golden_encoding_matches_production_tokenizer():
    """Pin the exact canonical encoding for Qwen3.5-2B (thinking on).

    Network-gated: skips if the tokenizer can't be fetched. Where it runs
    (validators, CI with the tokenizer cached), it locks the encoding so a
    transformers bump that changes apply_chat_template output is caught."""
    import pytest

    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from reliquary.constants import DEFAULT_BASE_MODEL

    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    except Exception as e:  # offline / model gated / network down
        pytest.skip(f"production tokenizer unavailable: {e}")

    assert encode_prompt(tok, _GOLDEN_PROMPT) == _GOLDEN_TOKENS_QWEN35_2B
