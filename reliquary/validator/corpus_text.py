"""The two checks that need a tokenizer and a renderer, so they cannot live
beside the pure checks in ``corpus/checks.py``.

Both close the same class of gap: a submission's *tokens* are what gets
audited and paid, but the *text* is what the corpus is made of, and nothing
before this module ever required the two to agree. A miner could submit real
tokens with empty (or unrelated) ``text`` and be paid in full for a
completion that contributes nothing to the corpus; the same gap lets a miner
answer an easier prompt than the one its slot claims, if the prompt it
rendered is never checked against the one the job actually assigned.

Kept pure and injectable like ``corpus/checks.py``: the tokenizer and the
renderer are passed in by the caller, never looked up here, so this module
needs no GPU, no bucket, and no network to test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from reliquary.corpus.checks import CheckResult
from reliquary.environment.agentic.types import EpisodeTask

REASON_TEXT_MISMATCH = "text_does_not_match_tokens"
REASON_PROMPT_MISMATCH = "prompt_not_faithful"


def _ok() -> CheckResult:
    """A fresh result every time, matching ``corpus/checks.py``'s own helper:
    ``CheckResult`` is frozen but its default ``detail`` dict is not."""
    return CheckResult(ok=True)


class Tokenizer(Protocol):
    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...


class Renderer(Protocol):
    def initial_text(self, task: EpisodeTask) -> str: ...


class PromptJob(Protocol):
    """What a job must supply to check the prompt it assigned to a slot: the
    source row for that slot, as an ``EpisodeTask`` ready to render. Resolving
    ``prompt_source`` to this is the caller's job, not this module's — the
    same division the renderer parameter already draws."""

    def task_for(self, prompt_index: int) -> EpisodeTask: ...


def check_text_matches_tokens(
    tokens: Sequence[int], text: str, *, tokenizer: Tokenizer
) -> CheckResult:
    """Refuse a completion whose ``text`` is not what its ``tokens`` decode to.

    Compared EXACTLY: no stripping, no whitespace normalisation, no
    casefolding. A miner able to add characters the tokens do not contain
    could inject arbitrary content into the corpus, which is the product
    being sold, so relaxing this comparison would reopen the gap it exists
    to close. ``skip_special_tokens`` and ``clean_up_tokenization_spaces``
    are pinned to keep the tokenizer's own decode as literal as possible;
    without them some tokenizers rewrite whitespace by default, which would
    fail an honest submission for a reason that has nothing to do with the
    text it sent.
    """

    decoded = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != text:
        return CheckResult(
            ok=False,
            reason=REASON_TEXT_MISMATCH,
            # Lengths, not the strings: a rejection must be diagnosable
            # without copying a 16k completion into a log line.
            detail={"decoded_chars": len(decoded), "submitted_chars": len(text)},
        )
    return _ok()


def check_prompt_fidelity(
    rendered: str, *, job: PromptJob, prompt_index: int, renderer: Renderer
) -> CheckResult:
    """Refuse a rendered prompt that is not the job's own source row, in the
    job's own declared renderer, for the slot the submission claims.

    Without this, a miner could answer an easier question than the one its
    prompt index names, or splice extra instructions into the prompt it
    conditions on, and nothing downstream would notice: the tokens would
    still decode to matching text, and the completion would still get paid.
    """

    task = job.task_for(prompt_index)
    expected = renderer.initial_text(task)
    if rendered != expected:
        return CheckResult(
            ok=False,
            reason=REASON_PROMPT_MISMATCH,
            detail={
                "prompt_index": prompt_index,
                "expected_chars": len(expected),
                "rendered_chars": len(rendered),
            },
        )
    return _ok()
