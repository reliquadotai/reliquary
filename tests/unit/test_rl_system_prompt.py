"""A single-turn problem's system prompt reaches the RL prompt the same way on the miner and the validator."""

import pytest

from reliquary.protocol.tokens import encode_prompt
from reliquary.validator.prompt_content import render_canonical_prompt


class _TemplateTokenizer:
    """Renders turns as text, then encodes it byte by byte: rendering and encoding stay one function apart."""

    chat_template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>{% if enable_thinking %}"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_dict=False,
                            enable_thinking=None):
        assert add_generation_prompt is True
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        text += "<think>" if enable_thinking else ""
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


@pytest.mark.parametrize("system", [None, "Answer with a number."])
@pytest.mark.parametrize("thinking", [True, False])
def test_the_miner_tokens_are_the_validator_render_encoded(system, thinking):
    tok = _TemplateTokenizer()
    miner = encode_prompt(tok, "2+2=", thinking=thinking, system=system)
    validator = tok.encode(render_canonical_prompt(tok, "2+2=", thinking=thinking, system=system),
                           add_special_tokens=False)
    assert miner == validator


def test_a_system_prompt_is_a_turn_before_the_user_and_none_renders_as_before():
    tok = _TemplateTokenizer()
    assert render_canonical_prompt(tok, "2+2=", thinking=False, system="Be brief.") == \
        "<system>Be brief.<user>2+2=<assistant>"
    assert render_canonical_prompt(tok, "2+2=", thinking=False) == "<user>2+2=<assistant>"
    assert encode_prompt(tok, "2+2=", thinking=False, system="") == encode_prompt(tok, "2+2=", thinking=False)


def test_without_a_chat_template_the_system_prompt_leads_the_text(monkeypatch):
    class Bare:
        chat_template = None

        def encode(self, text, *, add_special_tokens):
            return list(text.encode("utf-8"))

    assert encode_prompt(Bare(), "2+2=", system="Be brief.") == list(b"Be brief.\n\n2+2=")
    assert render_canonical_prompt(Bare(), "2+2=", system="Be brief.") == "Be brief.\n\n2+2="
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "RAW_COMPLETION_PROMPTS", True)
    assert encode_prompt(_TemplateTokenizer(), "2+2=", system="Be brief.") == list(b"Be brief.\n\n2+2=")
    assert render_canonical_prompt(_TemplateTokenizer(), "2+2=", system="Be brief.") == "Be brief.\n\n2+2="
