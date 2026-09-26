"""A single-turn job may render its rows through the model's own chat template,
with or without thinking. The template ships with the checkpoint's tokenizer,
which the job's revision pins, so one rule serves every model."""

import pytest

from reliquary.environment.agentic.types import EpisodeTask
from reliquary.validator.corpus_service import (
    CorpusPromptSourceError,
    renderer_for_job,
    resolve_prompt_source,
)
from reliquary.validator.corpus_text import check_prompt_fidelity
from tests.unit.test_corpus_single_turn_prompts import (
    SOURCE,
    _SingleTurnSpec,
    _job,
    _profile,
    _prompts,
)


class _ChatTokenizer:
    """Wraps a message the way a ChatML template would, and records the call."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        think = "<think>\n" if kwargs.get("enable_thinking") else "<think>\n\n</think>\n\n"
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n" + think


def _renderer(renderer_id, tokenizer):
    return renderer_for_job(
        _job(renderer_id=renderer_id),
        lambda text: [],
        environments={SOURCE: _SingleTurnSpec()},
        profile=_profile(),
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize("renderer_id,thinking", [
    ("chat-template-thinking-v1", True),
    ("chat-template-v1", False),
])
def test_a_chat_template_job_wraps_the_row_in_the_models_template(renderer_id, thinking):
    tokenizer = _ChatTokenizer()
    rendered = _renderer(renderer_id, tokenizer).initial_text(
        EpisodeTask(id="row", prompt="question 3", tools=())
    )
    messages, kwargs = tokenizer.calls[-1]
    assert messages == [{"role": "user", "content": "question 3"}]
    assert kwargs["tokenize"] is False and kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is thinking
    assert rendered.startswith("<|im_start|>user\nquestion 3")


def test_a_chat_template_job_needs_the_tokenizer():
    with pytest.raises(CorpusPromptSourceError, match="tokenizer"):
        _renderer("chat-template-thinking-v1", None)


def test_the_tokenizer_may_arrive_after_the_renderer_is_built():
    box = {}
    renderer = _renderer("chat-template-thinking-v1", lambda: box["tokenizer"])
    box["tokenizer"] = _ChatTokenizer()
    assert "question 1" in renderer.initial_text(EpisodeTask(id="r", prompt="question 1", tools=()))


def test_a_chat_template_job_still_needs_the_contract_to_render_its_rows():
    """The template wraps the row the contract renders; a source the contract
    does not render has no row to wrap."""
    with pytest.raises(CorpusPromptSourceError):
        resolve_prompt_source(
            SOURCE, environments={SOURCE: _SingleTurnSpec()},
            renderer_id="chat-template-thinking-v1", profile=_profile(None),
        )
    resolve_prompt_source(
        SOURCE, environments={SOURCE: _SingleTurnSpec()},
        renderer_id="chat-template-thinking-v1", profile=_profile(),
    )


def test_fidelity_accepts_the_templated_prompt_and_refuses_the_raw_row():
    job = _job(renderer_id="chat-template-thinking-v1")
    renderer = _renderer("chat-template-thinking-v1", _ChatTokenizer())
    prompts = _prompts(job)
    templated = renderer.initial_text(prompts.task_for(2))
    assert check_prompt_fidelity(templated, job=prompts, prompt_index=2, renderer=renderer).ok
    raw = prompts.task_for(2).prompt
    assert not check_prompt_fidelity(raw, job=prompts, prompt_index=2, renderer=renderer).ok


def test_an_unknown_renderer_is_still_refused():
    with pytest.raises(CorpusPromptSourceError):
        _renderer("chat-template-v9", _ChatTokenizer())
