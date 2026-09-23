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


def test_text_that_is_what_the_tokens_decode_to_passes():
    result = check_text_matches_tokens([1, 2, 3], "123", tokenizer=_Tokenizer())
    assert result.ok


def test_empty_text_with_a_full_token_array_is_refused():
    # The money leak, exactly: paid for 3 tokens, contributes nothing.
    result = check_text_matches_tokens([1, 2, 3], "", tokenizer=_Tokenizer())
    assert not result.ok
    assert result.reason == REASON_TEXT_MISMATCH


def test_text_from_a_different_completion_is_refused():
    result = check_text_matches_tokens([1, 2, 3], "999", tokenizer=_Tokenizer())
    assert not result.ok


def test_the_detail_names_both_lengths_without_quoting_the_text():
    # A rejection has to be diagnosable without copying a 16k completion into
    # a log line.
    result = check_text_matches_tokens([1, 2, 3], "", tokenizer=_Tokenizer())
    assert "123" not in str(result.detail)
    assert result.detail["decoded_chars"] == 3
    assert result.detail["submitted_chars"] == 0


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
