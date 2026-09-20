"""A turn that overruns the action byte budget must score zero, not raise."""

from types import SimpleNamespace

from reliquary.environment.agentic.types import MAX_ACTION_BYTES, GeneratedAction
from reliquary.miner.episode_policy import _within_action_bytes


class _Tokenizer:
    """One byte per token, so the budget lands on a token boundary."""

    def decode(self, tokens, skip_special_tokens=False):
        return "".join("x" for _ in tokens)


def test_a_turn_inside_the_budget_is_untouched():
    tokens = list(range(10))
    kept, text = _within_action_bytes(_Tokenizer(), tokens)
    assert kept == tokens
    assert text == "x" * 10


def test_an_overlong_turn_is_cut_to_what_fits():
    tokens = list(range(MAX_ACTION_BYTES + 500))

    kept, text = _within_action_bytes(_Tokenizer(), tokens)

    assert len(kept) == MAX_ACTION_BYTES
    assert len(text.encode("utf-8")) == MAX_ACTION_BYTES
    # The point of the cut: this used to raise inside the policy.
    GeneratedAction(text=text, tokens=tuple(kept))


def test_the_cut_falls_on_a_token_boundary():
    class _Wide:
        def decode(self, tokens, skip_special_tokens=False):
            return "".join("ààà" for _ in tokens)  # 6 bytes per token

    tokens = list(range(4000))
    kept, text = _within_action_bytes(_Wide(), tokens)

    assert len(text.encode("utf-8")) <= MAX_ACTION_BYTES
    assert _Wide().decode(kept) == text
