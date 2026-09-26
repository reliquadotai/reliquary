"""Payment counts tokens; the corpus is made of text. Nothing checked that
they describe the same completion, so a submission could be paid for tokens
it never turned into anything."""

from reliquary.environment.agentic.types import EpisodeTask
from reliquary.validator.corpus_text import (
    REASON_PROMPT_MISMATCH,
    REASON_TEXT_MISMATCH,
    check_prompt_fidelity,
    check_text_matches_tokens,
)


class _Tokenizer:
    """Decodes an id to its own digits, so a test can reason about the text."""

    def decode(self, ids, **kwargs):
        return "".join(str(i) for i in ids)


# Not the miner's tokens' own digits, so it reads unambiguously as the
# terminator in a decoded string built by concatenating digit strings.
EOS = 151645


def test_text_that_is_what_the_tokens_decode_to_passes():
    result = check_text_matches_tokens(
        [1, 2, 3], "123", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert result.ok


def test_empty_text_with_a_full_token_array_is_refused():
    # The money leak, exactly: paid for 3 tokens, contributes nothing.
    result = check_text_matches_tokens(
        [1, 2, 3], "", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok
    assert result.reason == REASON_TEXT_MISMATCH


def test_text_from_a_different_completion_is_refused():
    result = check_text_matches_tokens(
        [1, 2, 3], "999", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok


def test_the_detail_names_both_lengths_without_quoting_the_text():
    # A rejection has to be diagnosable without copying a 16k completion into
    # a log line.
    result = check_text_matches_tokens(
        [1, 2, 3], "", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert "123" not in str(result.detail)
    assert result.detail["decoded_chars"] == 3
    assert result.detail["submitted_chars"] == 0


def test_a_cap_terminated_completion_has_no_terminator_to_strip():
    # No trailing eos id in the tokens, so behaviour is exactly the
    # pre-ruling comparison: cap completions are unaffected by the new rule.
    result = check_text_matches_tokens(
        [1, 2, 3], "123", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert result.ok


def test_an_honest_completion_may_omit_the_stripped_terminator():
    # Ordinary generation (skip_special_tokens=True) never returns the eos
    # text at all; rejecting this would refuse every honest miner using the
    # default, and the corpus should not carry <|endoftext|> in its text.
    result = check_text_matches_tokens(
        [1, 2, 3, EOS], "123", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert result.ok


def test_the_terminator_spelled_out_is_refused():
    # Only the stripped spelling is legal: accepting both forms would give
    # one completion two valid digests for the same work.
    result = check_text_matches_tokens(
        [1, 2, 3, EOS], f"123{EOS}", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok
    assert result.reason == REASON_TEXT_MISMATCH


def test_only_the_last_of_two_trailing_terminators_is_stripped():
    # The one BEFORE the last is not the one that "ends" the completion, so
    # it must still show up in the text like any other token would.
    result = check_text_matches_tokens(
        [1, 2, EOS, EOS], "12", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok


def test_only_the_last_of_two_trailing_terminators_is_stripped_positive():
    result = check_text_matches_tokens(
        [1, 2, EOS, EOS], f"12{EOS}", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert result.ok


def test_a_mid_sequence_terminator_id_must_still_appear_in_the_text():
    # Only the LAST token may ever be dropped; the same id earlier in the
    # array is ordinary content the corpus is paying for.
    result = check_text_matches_tokens(
        [EOS, 1, 2], "12", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok


def test_a_trailing_token_other_than_the_manifest_eos_must_still_appear():
    # Stripping is keyed to the manifest's own eos_token_id, not "trailing
    # and looks special": a different trailing id leaves no trace to strip.
    other_special = 5
    result = check_text_matches_tokens(
        [1, 2, other_special], "12", tokenizer=_Tokenizer(), eos_token_id=EOS
    )
    assert not result.ok


class _Renderer:
    """Renders a task down to its prompt verbatim, so a test can build the
    exact expected string without depending on a real renderer's dialect."""

    @staticmethod
    def initial_text(task):
        return task.prompt


class _Job:
    """Two source rows, indexed the way a corpus job's prompts are."""

    def __init__(self):
        self._tasks = {
            0: EpisodeTask(id="row-0", prompt="what is 2+2?", tools=()),
            1: EpisodeTask(id="row-1", prompt="what is 3+3?", tools=()),
        }

    def task_for(self, prompt_index):
        return self._tasks[prompt_index]


def test_a_rendered_prompt_equal_to_the_renderers_own_output_passes():
    result = check_prompt_fidelity(
        "what is 2+2?", job=_Job(), prompt_index=0, renderer=_Renderer()
    )
    assert result.ok


def test_a_prompt_rendered_for_a_different_index_is_refused():
    # The text is faithful to prompt 1, not the prompt 0 slot it is submitted
    # against: a miner answering an easier question than the one it claimed.
    result = check_prompt_fidelity(
        "what is 3+3?", job=_Job(), prompt_index=0, renderer=_Renderer()
    )
    assert not result.ok
    assert result.reason == REASON_PROMPT_MISMATCH


def test_a_prompt_with_an_appended_hint_is_refused():
    result = check_prompt_fidelity(
        "what is 2+2? (hint: 4)", job=_Job(), prompt_index=0, renderer=_Renderer()
    )
    assert not result.ok
    assert result.reason == REASON_PROMPT_MISMATCH


def test_the_prompt_mismatch_detail_names_both_lengths_without_quoting_the_text():
    result = check_prompt_fidelity(
        "what is 2+2? (hint: 4)", job=_Job(), prompt_index=0, renderer=_Renderer()
    )
    assert "what is 2+2?" not in str(result.detail)
    assert result.detail["expected_chars"] == len("what is 2+2?")
    assert result.detail["rendered_chars"] == len("what is 2+2? (hint: 4)")
