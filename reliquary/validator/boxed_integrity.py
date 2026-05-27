r"""Structural validity of a rollout's final boxed answer.

The OMI reward scores the LAST ``\boxed{...}`` in the completion. A reward=0
rollout whose final box is malformed (empty, special-token, or unclosed) did not
produce a parseable answer — it is a "fake negative" used to manufacture a group
reward vector (k=4 / sigma=0.5) that passes the zone filter. Examples: appending
``\boxed{<|im_end|>`` after a correct ``\boxed{121}``, an empty ``\boxed{}``, or
spamming boxes to the token cap so the final one is cut off.

This check is purely structural and aligned with what the env scores (the last
box): it does NOT compare to the ground truth and does NOT judge intent. A
well-formed final box that is simply wrong is a legitimate negative and is not
flagged here (forced wrong answers are covered by the boxed-answer probability
check). Pure, side-effect-free; called by the batcher before GRAIL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Stop/special tokens that must never appear inside a final answer box.
SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

# A rollout within this many tokens of the protocol cap is treated as
# budget-exhausted (governed by the termination guard), not a deliberate
# malformed final answer.
CAP_MARGIN = 256

_MARKER = r"\boxed{"


@dataclass(frozen=True)
class BoxedSpan:
    content: str
    well_formed: bool


def extract_boxed_spans(text: str) -> list[BoxedSpan]:
    r"""All ``\boxed{...}`` occurrences with a well-formed flag.

    Malformed when unclosed, empty/whitespace, or containing a special token.
    """
    spans: list[BoxedSpan] = []
    i = 0
    while True:
        j = text.find(_MARKER, i)
        if j == -1:
            break
        k = j + len(_MARKER)
        depth = 1
        buf: list[str] = []
        closed = False
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    k += 1
                    break
            buf.append(c)
            k += 1
        content = "".join(buf)
        well_formed = (
            closed
            and content.strip() != ""
            and not any(tok in content for tok in SPECIAL_TOKENS)
        )
        spans.append(BoxedSpan(content=content, well_formed=well_formed))
        i = k if k > j + len(_MARKER) else j + len(_MARKER)
    return spans


def has_malformed_final_answer(
    reward: float,
    text: str,
    completion_length: Optional[int] = None,
    cap: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    r"""True when a reward=0 rollout's final ``\boxed{}`` is malformed.

    Aligned with the env (which scores the last box). Only evaluated for
    reward < 0.5. Returns ``(False, None)`` when there is no box at all (a clean
    give-up), when the last box is well-formed (a legitimate wrong answer), or
    when the rollout is within ``CAP_MARGIN`` of the protocol cap (budget
    exhaustion — governed by the termination guard, not flagged here).
    """
    if reward is not None and reward >= 0.5:
        return False, None
    if completion_length is not None and cap is not None and completion_length >= cap - CAP_MARGIN:
        return False, None
    spans = extract_boxed_spans(text)
    if not spans:
        return False, None
    if not spans[-1].well_formed:
        return True, "malformed_final_boxed"
    return False, None
